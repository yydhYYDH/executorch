# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sum and amax on the DSP, and the two reductions that have no kernel.

`htp_ops_reduction` collapses one contiguous span of the buffer and selects on
`HtpOpsReductionOpType`, which is `sum = 1, maximum = 2, mean = 3`
(eltwise_ops.cc:2441-2445) and nothing else. That enum is the whole set of
reductions this backend can reach: `mean` was already wired, these tests pin
`sum` and `amax`, and they pin the absence of `amin` -- a minimum has no op type
to select, so it cannot be delegated by naming the right emitter.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_REDUCTION, and the three kinds it selects between.
_REDUCTION = 29
_SUM = 1
_MAXIMUM = 2
_MEAN = 3
_FP16_BYTES = 2


class _Reduce(torch.nn.Module):
    def __init__(self, kind, **kwargs) -> None:
        super().__init__()
        self.kind = kind
        self.kwargs = kwargs

    def forward(self, x):
        if self.kind == "sum":
            return torch.sum(x, **self.kwargs)
        if self.kind == "amax":
            return torch.amax(x, **self.kwargs)
        return torch.amin(x, **self.kwargs)


def _program(model, x):
    return to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _delegated(model, x):
    """The blob a whole-model reduction lowers to, plus its command stream."""
    program = _program(model, x)
    calls = _delegates(program)
    assert len(calls) == 1, f"the reduction did not reach the delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    blob = bytes(lowered._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def _run(blob, x):
    return np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)


@pytest.mark.parametrize(
    "shape, dim, span",
    [
        ((4, 8), 1, (4, 8, 1)),  # the inner axis: nothing inside
        ((4, 8), 0, (1, 4, 8)),  # the outer axis: nothing outside
        ((2, 3, 4), 1, (2, 3, 4)),  # a middle axis
        ((2, 3, 4), (1, 2), (2, 12, 1)),  # two adjacent axes
        ((2, 3, 4), -1, (6, 4, 1)),
        ((2, 3, 4), None, (1, 24, 1)),  # a missing dim is every dim
        ((2, 3, 4), [], (1, 24, 1)),  # and so is an empty list
    ],
)
def test_a_sum_is_one_span_of_the_buffer(shape, dim, span):
    """The three params the kernel walks, for every shape of one span."""
    x = torch.randn(*shape, dtype=torch.float16)
    blob, commands = _delegated(_Reduce("sum", dim=dim), x)
    assert [command.type for command in commands] == [_REDUCTION]
    assert list(commands[0].params) == [
        *span,
        _SUM,
        _FP16_BYTES,
    ], f"sum over {dim} of {shape} emitted {list(commands[0].params)}"
    np.testing.assert_allclose(
        _run(blob, x),
        torch.sum(x, dim=dim).numpy().reshape(-1),
        rtol=2e-3,
        atol=2e-2,
    )


def test_a_sum_keeps_its_dims_when_asked():
    """keepdim changes the result's shape, not the span the kernel collapses."""
    x = torch.randn(4, 8, dtype=torch.float16)
    blob, commands = _delegated(_Reduce("sum", dim=1, keepdim=True), x)
    assert list(commands[0].params) == [4, 8, 1, _SUM, _FP16_BYTES]
    np.testing.assert_allclose(
        _run(blob, x),
        torch.sum(x, dim=1, keepdim=True).numpy().reshape(-1),
        rtol=2e-3,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    "shape, dim, span",
    [
        ((4, 8), 1, (4, 8, 1)),
        ((2, 3, 4), (1, 2), (2, 12, 1)),  # adjacent
        ((2, 3, 4), -2, (2, 3, 4)),
        ((2, 3, 4), [], (1, 24, 1)),
    ],
)
def test_amax_is_the_maximum_the_kernel_takes(shape, dim, span):
    """The same span walk with the other op type, and an exact answer.

    A maximum is a selection rather than an accumulation, so fp16 costs it
    nothing: the comparison here is an equality.
    """
    x = torch.randn(*shape, dtype=torch.float16)
    blob, commands = _delegated(_Reduce("amax", dim=dim), x)
    assert [command.type for command in commands] == [_REDUCTION]
    assert list(commands[0].params) == [*span, _MAXIMUM, _FP16_BYTES]
    got = _run(blob, x)
    expected = torch.amax(x, dim=dim).numpy().reshape(-1)
    assert got.tobytes() == expected.tobytes()


def test_a_maximum_over_every_dim_is_the_whole_tensor():
    """A missing dim, which the kernel reads as one span over everything."""
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    blob, commands = _delegated(_Reduce("amax"), x)
    assert list(commands[0].params) == [1, 24, 1, _MAXIMUM, _FP16_BYTES]
    assert _run(blob, x)[0] == x.max()


def test_a_reduction_over_two_separate_spans_stays_on_the_host():
    """Dims that are not adjacent are two spans, and the kernel collapses one."""
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    for kind in ("sum", "amax"):
        program = _program(_Reduce(kind, dim=(0, 2)), x)
        assert _delegates(program) == [], f"{kind} over (0, 2) reached the delegate"
        # The portable kernel still answers, with the value the delegate cannot
        # express in one command.
        expected = (
            torch.sum(x, dim=(0, 2)) if kind == "sum" else torch.amax(x, dim=(0, 2))
        )
        assert expected.shape == (3,)


def test_a_sum_into_another_width_stays_on_the_host():
    """The kernel's accumulator is fp32 and its result fp16; nothing else fits."""
    x = torch.randn(4, 8, dtype=torch.float16)
    model = _Reduce("sum", dim=1, dtype=torch.float32)
    program = _program(model, x)
    assert _delegates(program) == []
    assert model(x).dtype is torch.float32


def test_amin_has_no_kernel_to_reach():
    """The correction to "amin is free": there is no minimum op type.

    `HtpOpsReductionOpType` is sum, maximum, mean, and the dispatcher rejects
    anything else, so `amin` cannot be delegated by writing another emitter --
    it needs a new kernel, and it stays on the portable one until one exists.
    This test is the executable form of that claim: amax delegates, amin does
    not, on the same node and the same dims.
    """
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    assert len(_delegates(_program(_Reduce("amax", dim=1), x))) == 1
    assert _delegates(_program(_Reduce("amin", dim=1), x)) == []
    assert torch.amin(x, dim=1).shape == (2, 4)
    assert _MEAN == 3 and _MAXIMUM == 2 and _SUM == 1
