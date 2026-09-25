# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DSP's own `sin` and `cos`, measured on the parts that run them.

Neither has an HVX walk: `htp_ops_unary_compute_fp16_chunk` has a vector branch
for exp, log, expm1 and the PWL activations and none for these two, so both fall
through to the scalar `htp_ops_unary_apply_fp16`, which reduces the angle in fp32
and evaluates a truncated series (unary_ops.cc:106-129). What that costs was not
known before this file.

The measurement is not a comparison of the simulator with the device: both run
the same approximation, so agreeing with each other says nothing about accuracy.
The reference here is the mathematically correct value of the function at the
fp16 input, computed in fp64, and the numbers asserted below are the worst cases
over a grid that includes the arguments the series is worst at -- the ends of
[-pi/2, pi/2], where the truncation error peaks -- and the ones where the fp32
angle reduction, not the series, is what fails.

The bound is stated against the band the backend already lives in rather than
against an arbitrary number: `tanh` is wired and its relative error is 2.1e-2,
`log` is wired and its relative error at x = 1 is 1.7e1, and both are measured in
`probe/measure_unary_band.py`. A few fp16 ulp on a value bounded by one is well
inside that.

`expm1` is deliberately not wired, and `test_expm1_stays_on_the_portable_kernel`
is where that decision is held: its vector path subtracts 1 in fp16, which for
small arguments leaves none of the value the function exists to compute, and its
relative error there is measured at 1.0 -- no correct digits at all.
"""


from types import SimpleNamespace

import numpy as np
import pytest
import torch


import hexagon_sim
from blob_interpreter import Arena, read_blob
from executorch.backends.hexagon.hexagon_backend import (
    HexagonBackend,
    SUPPORTED_TARGETS,
)
from executorch.backends.hexagon.hexagon_ops import UNARY_OP_TYPES
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import (
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)

from executorch.exir.dialects._ops import ops as exir_ops
from test_blob_on_sim import (
    _fixture_header,
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
)
from torch.export import export

EXPM1 = exir_ops.edge.aten.expm1.default


# One node carrying this target, for the support predicate to read.
def _node_with_op(target):
    graph = torch.fx.Graph()
    placeholder = graph.placeholder("x")
    placeholder.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
    node = graph.call_function(target, args=(placeholder,))
    node.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
    return node


_DSP_OP_UNARY = 4

#: The worst absolute error each of these may have against the correctly rounded
#: value, per argument range. The two ranges are separate claims: the first is
#: the series' own truncation error, and the second is the fp32 angle reduction
#: losing phase as k*two_pi grows, which is a different failure and only shows up
#: past a thousand radians.
_BOUND_WITHIN_4PI = {"sin": 4.0e-4, "cos": 9.0e-4}
_BOUND_BEYOND_4PI = {"sin": 2.0e-3, "cos": 2.5e-3}
_SMALLEST_INTERESTING = 1.0e-9


class _Unary(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, x):
        return getattr(torch, self.name)(x)


def _probe_input() -> torch.Tensor:
    """The arguments the measurement is taken over.

    Dense across two periods, the ends of [-pi/2, pi/2] and of [-pi, pi] exactly,
    the small values a series is exact at, and a decade either side of a thousand
    radians where the reduction rather than the series is what moves. Two thousand
    and sixty-five of them is past the kernel's 2048-element threshold
    (unary_ops.cc:33), so the multi-task walk is the one taken, and a length that
    is not a multiple of the 64-element vector leaves the last seventeen elements
    to the scalar tail -- both halves of the walk are measured.
    """
    halves = [
        -2 * np.pi,
        -np.pi,
        -np.pi / 2,
        -1e-3,
        -1e-5,
        0.0,
        1e-5,
        1e-3,
        np.pi / 2,
        np.pi,
        2 * np.pi,
    ]
    far = [-5e4, -1e4, -1e3, 1e3, 1e4, 5e4]
    values = np.concatenate(
        [
            np.linspace(-2 * np.pi, 2 * np.pi, 2048),
            np.asarray(halves + far, dtype=np.float64),
        ]
    )
    return torch.from_numpy(values.astype(np.float16))


def _blob_for(name: str) -> bytes:
    """The delegate blob one `torch.sin(x)` subgraph lowers to."""
    example = _probe_input()
    program = to_edge(
        export(_Unary(name), (example,)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    return bytes(HexagonBackend.preprocess(program, []).processed_bytes)


@pytest.fixture(scope="module")
def measured():
    """(reference, dsp answer per op) over the probe input, from the simulator."""
    example = _probe_input()
    fixtures = []
    for name in ("sin", "cos"):
        blob = _blob_for(name)
        header, commands = read_blob(blob)
        assert len(commands) == 1, f"{name}: {len(commands)} commands"
        assert (
            commands[0].type == _DSP_OP_UNARY
        ), f"{name}: the command is type {commands[0].type}, not the unary one"
        assert (
            commands[0].params[1] == UNARY_OP_TYPES[name]
        ), f"{name}: the command's subtype is {commands[0].params[1]}"
        assert commands[0].params[0] == example.numel(), (
            f"{name}: the command walks {commands[0].params[0]} elements of "
            f"{example.numel()}"
        )
        fixtures.append(
            SimpleNamespace(
                tag=name.upper(),
                blob=blob,
                inputs=example.numpy().tobytes(),
                length=0,
                arena_bytes=len(Arena(header, blob, "fixture").bytes),
            )
        )

    try:
        answers = hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={
                "blob_fixture.h": _fixture_header(fixtures),
                "htp_ops.h": _HT_P_OP_SHIM,
            },
            includes_more=[str(_SCHEMA)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))

    got = {}
    for name in ("sin", "cos"):
        key = f"{name.upper()}0"
        assert (
            key in answers
        ), f"the simulator answered nothing for {name}: {sorted(answers)}"
        got[name] = np.asarray(answers[key], dtype=np.uint16).view(np.float16)
    return example, got


def _reference(name: str, x: torch.Tensor) -> np.ndarray:
    return getattr(np, name)(x.numpy().astype(np.float64))


@pytest.mark.parametrize("name", ["sin", "cos"])
def test_the_approximation_is_within_the_band_the_backend_already_accepts(
    measured, name
):
    """The measured error, split by argument range, against a stated bound.

    Both halves are asserted with the same grid, because the two failures have
    different causes: inside four pi the series' own truncation is the whole of
    it, and past that the fp32 reduction's phase error is. A single bound over
    the whole grid would hide which one is being met.
    """
    x, got = measured
    want = _reference(name, x)
    error = np.abs(got[name].astype(np.float64) - want)
    near = np.abs(x.numpy().astype(np.float64)) <= 4 * np.pi
    for label, mask, bound in (
        ("inside 4*pi", near, _BOUND_WITHIN_4PI[name]),
        ("past 4*pi", ~near, _BOUND_BEYOND_4PI[name]),
    ):
        worst = float(error[mask].max())
        at = float(x.numpy().astype(np.float64)[mask][int(np.argmax(error[mask]))])
        assert worst <= bound, (
            f"the DSP's {name} is off by {worst:.3e} at x = {at:.6g}, past the "
            f"{bound:.3e} this backend accepts for {label}"
        )
        # And the bound is not vacuous: a kernel that stopped approximating
        # altogether would pass a bound nothing comes near.
        assert worst > bound / 64, (
            f"the DSP's {name} is exact to {worst:.3e} for {label}, which is not "
            "what this kernel computes -- the descriptor or the run is wrong"
        )


@pytest.mark.parametrize("name", ["sin", "cos"])
def test_the_worst_case_is_at_the_end_of_the_half_period(measured, name):
    """Where the error is largest, so the bound above is not met by luck.

    `sin`'s series is truncated after x**7 and `cos`'s after x**6, and both
    reduce the angle into [-pi/2, pi/2] first, so their largest truncation error
    is at the ends of that interval. An emitter or a kernel that swapped the two
    subtypes would put the wrong function's error in that shape.
    """
    x, got = measured
    want = _reference(name, x)
    error = np.abs(got[name].astype(np.float64) - want)
    near = np.abs(x.numpy().astype(np.float64)) <= 4 * np.pi
    worst_at = x.numpy().astype(np.float64)[near][int(np.argmax(error[near]))]
    assert abs(abs(worst_at) - np.pi / 2) < 0.05, (
        f"the largest error of {name} is at x = {worst_at:.6g}, not at the end of "
        "the interval the angle is reduced into"
    )


def test_expm1_stays_on_the_portable_kernel():
    """The measurement that decided it, held as the decision.

    `expm1` runs an HVX walk below 64 elements and not above: the walk computes
    `exp2(x * log2(e))` and subtracts 1 in fp16 (unary_ops.cc:438-453), while a
    shorter array falls through to the scalar `expf(x) - 1` in fp32 (:201). The
    subtract is between two fp16 numbers whose spacing at 1.0 is 9.8e-4, so for
    |x| below about 1e-3 the walk answers the spacing rather than the value, and
    the measured relative error against the correct answer is 1.0 -- no correct
    digits at all -- on an array long enough to take it. On a 21-element array
    the same inputs come back exact, so which answer a tensor gets depends on its
    length. A function whose whole purpose is accuracy near zero cannot be
    delegated to an implementation whose error is worst there, so the emitter is
    absent and the node stays portable: a decision, not a gap.
    """
    assert EXPM1 not in SUPPORTED_TARGETS, (
        "torch.expm1 is in the emitter table; if the kernel has been changed so "
        "that the vector path is accurate near zero, this test is the one to "
        "replace with a measurement of the new kernel"
    )

    class _Expm1(torch.nn.Module):
        def forward(self, x):
            return torch.expm1(x)

    x = torch.linspace(-3.0, 3.0, 128, dtype=torch.float16)
    program = to_edge_transform_and_lower(
        export(_Expm1(), (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    delegates = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert not delegates, (
        "torch.expm1 reached the delegate, so it is running the kernel measured "
        "at 100% relative error near zero"
    )
    assert not HexagonOperatorSupport().is_node_supported(
        {}, _node_with_op(EXPM1)
    ), "the support predicate takes an expm1, which the measurement did not"
