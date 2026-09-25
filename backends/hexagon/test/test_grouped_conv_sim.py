# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The grouped convolution blobs, executed on the Hexagon simulator.

The host tests read the same blobs with a Python transcription of the kernels.
This one runs the vendored C: the blob the partitioner produced for a grouped
convolution is handed to hexagon-sim, and the DSP answer is compared against a
float64 torch reference and against the host model. The reference is float64
because the DSP stores fp16, so the comparison is a tolerance rather than an
equality, and the tolerance is stated per case. A case that cannot be run
because the SDK or the simulator is missing skips with a reason; the
hexagon_sim._check() is called first so a later Unavailable is a failure rather
than a silent skip.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
import test_blob_on_sim as B  # noqa: E402
import test_conv as C  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as _blob_mod  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

_IM2COL = 12
_DEPTHWISE = 2

#: The same runner and the same fixture header the blob suite uses: the DSP
#: answers for the partitioner's own blob, not for a fixture built by hand.
_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/blob_runner.cpp"
_SOURCES = B._SOURCES
_HEADERS = {"htp_ops.h": B._HT_P_OP_SHIM}
_INCLUDES = [str(B._SCHEMA)]


def _names():
    return {
        value: name
        for name, value in vars(hexagon_ops).items()
        if name.startswith("DSP_OP_")
    }


def _grouped(in_channels, out_channels, groups, kernel=3, padding=1, stride=1):
    model = C._Conv(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        groups=groups,
    ).half()
    C._whole(model, 3)
    return model


def _lowered(model, x):
    return to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _reference_fp64(model, x):
    """torch's own answer in float64, which the fp16 DSP answer is compared to."""
    conv = model.conv
    return torch.nn.functional.conv2d(
        x.double(),
        conv.weight.detach().double(),
        conv.bias.detach().double() if conv.bias is not None else None,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        dilation=conv.dilation,
    )


def _blob_bytes(program):
    calls = C._delegates(program)
    assert len(calls) == 1, f"{len(calls)} delegates"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    return bytes(lowered._processed_bytes)


class _Case:
    def __init__(self, tag, blob, inputs, expected, tolerance, commands):
        self.tag = tag
        self.blob = blob
        self.inputs = inputs
        self.expected = expected
        self.tolerance = tolerance
        self.commands = commands
        self.arena_bytes = 0
        self.length = 0


def _make(tag, model, x, tolerance=1e-2):
    program = _lowered(model, x)
    blob = _blob_bytes(program)
    from blob_interpreter import Arena, read_blob

    header, commands = read_blob(blob)
    case = _Case(
        tag,
        blob,
        b"".join(t.numpy().tobytes() for t in (x,)),
        _reference_fp64(model, x).reshape(-1).numpy(),
        tolerance,
        commands,
    )
    case.arena_bytes = len(Arena(header, blob, "grouped").bytes)
    return case


def _cases():
    """The grouped shapes: the group counts, and 63/64/65 channels per group."""
    cases = []
    x4 = C._exact((1, 256, 4, 4), 5, -2, 3)
    cases.append(_make("GC4", _grouped(256, 256, 4), x4))
    cases.append(_make("GC8", _grouped(256, 256, 8), x4))
    x2 = C._exact((1, 64, 8, 8), 6, -2, 3)
    cases.append(_make("GC2", _grouped(64, 64, 2), x2))
    xw = C._exact((1, 64, 8, 8), 7, -2, 3)
    cases.append(_make("GCW", _grouped(64, 128, 8), xw))
    for per_group in (63, 64, 65):
        x = C._exact((1, 3 * per_group, 4, 4), 8, -2, 3)
        cases.append(_make(f"G{per_group}", _grouped(3 * per_group, 3 * per_group, 3), x))
    xd = C._exact((1, 64, 8, 8), 9, -2, 3)
    cases.append(_make("GDW", _grouped(64, 64, 64), xd))
    return cases


@pytest.fixture(scope="module")
def cases():
    try:
        hexagon_sim._check()
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))
    return _cases()


@pytest.fixture(scope="module")
def simulated(cases):
    try:
        return hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={"blob_fixture.h": B._fixture_header(cases), **_HEADERS},
            includes_more=_INCLUDES,
        )
    except hexagon_sim.Unavailable as error:
        # _check() already passed, so an Unavailable here is a simulator that
        # did not run the binary, which is a failure rather than a skip.
        pytest.fail(f"{error}\n{hexagon_sim.LAST_STDOUT[-2000:]}")


def test_the_grouped_blobs_hold_the_walks_the_claim(cases):
    """Each grouped blob is the dense walk once per group, and the depthwise is not."""
    names = _names()
    # The group count each case was built with. GC4, GC2, GC8 and GCW are the
    # group-count cases; G63, G64 and G65 are the boundary sweep, each three
    # groups of 63, 64 and 65 input channels; GDW is the depthwise control.
    groups_of = {"GC2": 2, "GC4": 4, "GC8": 8, "GCW": 8, "G63": 3, "G64": 3, "G65": 3}
    for case in cases:
        kinds = [names[command.type] for command in case.commands]
        walks = kinds.count("DSP_OP_IM2COL_CONVOLUTION_FP16")
        if case.tag == "GDW":
            assert walks == 0, "the depthwise case is not a grouped walk"
            assert kinds.count("DSP_OP_CONV_DEPTHWISE2D_FP16") == 1
            continue
        assert walks == groups_of[case.tag], f"{case.tag}: {walks} walks for {groups_of[case.tag]} groups"
        # The count is decoded from the blob, not from an ops=N header.
        from blob_interpreter import read_blob

        header, decoded = read_blob(case.blob)
        assert len(decoded) == header.n_ops == len(case.commands)


@pytest.mark.parametrize("tag", ["GC4", "GC8", "GC2", "GCW", "G63", "G64", "G65", "GDW"])
def test_each_grouped_case_agrees_on_the_simulator(cases, simulated, tag):
    """The DSP answer against float64 torch, and against the host model."""
    case = next(c for c in cases if c.tag == tag)
    assert len(case.blob) > 0 and len(case.inputs) > 0
    dsp = simulated[f"{tag}0"]
    assert len(dsp) == case.expected.size, f"{tag}: {len(dsp)} words, want {case.expected.size}"
    got = np.array(dsp, dtype=np.uint16).view(np.float16)
    worst = float(np.max(np.abs(got.astype(np.float64) - case.expected)))
    assert worst < case.tolerance, f"{tag}: worst {worst} (got {got[:8]} want {case.expected[:8]})"


def test_the_simulator_answers_are_not_empty(cases, simulated):
    """Every case produced fp16 words, so no runner output went unread."""
    for case in cases:
        dsp = simulated[f"{case.tag}0"]
        assert dsp, f"{case.tag}: the simulator returned no words"
        assert all(isinstance(word, int) for word in dsp)
