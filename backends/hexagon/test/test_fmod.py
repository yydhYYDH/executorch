# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`fmod`, which is the one remainder the DSP's element-wise kernel computes.

`HTP_OPS_BINARY_MOD` is `a - trunc(a/b)*b` with a zero divisor answering zero
(eltwise_ops.cc:148-166), which is torch's `fmod` and not its `remainder`: the
two disagree in sign whenever the operands do, so the second is refused rather
than approximated. The kernel's zero-divisor guard is a real difference from
torch as well -- torch gives NaN -- and it is pinned here rather than left to be
discovered on device.
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

#: DSP_OP_BINARY_ELEMENTWISE and HTP_OPS_BINARY_MOD.
_BINARY = 19
_MOD = 12
_FP16_BYTES = 2


class _Fmod(torch.nn.Module):
    def __init__(self, kind="fmod") -> None:
        super().__init__()
        self.kind = kind

    def forward(self, x, y):
        if self.kind == "remainder":
            return torch.remainder(x, y)
        return torch.fmod(x, y)


def _program(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _delegated(model, inputs):
    program = _program(model, inputs)
    calls = _delegates(program)
    assert len(calls) == 1, f"fmod did not reach the delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    blob = bytes(lowered._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def _run(blob, inputs):
    return np.frombuffer(execute(blob, [x.numpy() for x in inputs])[0], dtype=np.float16)


def test_fmod_is_the_truncated_remainder_the_kernel_computes():
    """The subtype, the params, and the signs torch gives.

    Every value here is negative on one side, because that is the whole
    difference between the truncated and the floored remainder.
    """
    x = torch.tensor([-7.5, 7.5, -0.5, 0.5, 3.0, -3.0, 1.0, -1.0], dtype=torch.float16)
    y = torch.tensor([2.0, -2.0, 3.0, -3.0, 4.0, 4.0, 0.5, -0.5], dtype=torch.float16)
    x, y = x.reshape(1, 8), y.reshape(1, 8)
    blob, commands = _delegated(_Fmod(), (x, y))
    assert [command.type for command in commands] == [_BINARY]
    assert list(commands[0].params[:8]) == [
        8,  # result elements
        8,  # left operand elements
        8,  # right operand elements
        _MOD,
        _FP16_BYTES,  # element width
        _FP16_BYTES,  # operand width
        0,  # operands are not 4-byte floats
        0,  # result is not a 4-byte float
    ]
    # The broadcast path's 25 params: the rank, the output's shape, then each
    # operand's row-major strides right-aligned to eight axes, zero on a
    # broadcast dim, each padded out to eight.
    assert list(commands[0].params[8:]) == [
        2,
        1,
        8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    got = _run(blob, (x, y))
    expected = torch.fmod(x, y).numpy().reshape(-1)
    # Where the remainder is exactly zero the kernel's `a - trunc(a/b)*b` lands
    # on a positive zero, and torch's fmod on a negative one for a negative
    # dividend: equal as numbers, not the same bytes. Every other element has to
    # match bit for bit.
    zeros = expected == 0
    assert zeros.any() and np.all(got[zeros] == 0)
    assert np.array_equal(got[~zeros], expected[~zeros])
    assert expected[0].item() == -1.5 and expected[2].item() == -0.5


def test_fmod_broadcasts_the_divisor_the_way_torch_does():
    """A one-element or one-column operand rides in the broadcast tail."""
    x = torch.randn(4, 8, dtype=torch.float16)
    for shape in ((1,), (4, 1), (1, 8)):
        y = (torch.randn(*shape, dtype=torch.float16) + 3.0).abs() + 0.5
        blob, commands = _delegated(_Fmod(), (x, y))
        assert [command.type for command in commands] == [_BINARY]
        assert commands[0].params[3] == _MOD
        got = _run(blob, (x, y))
        expected = torch.fmod(x, y).numpy().reshape(-1)
        assert got.tobytes() == expected.tobytes()


def test_remainder_is_not_fmod_and_stays_on_the_host():
    """The floored remainder is a different function, so it has no command.

    Asserting the values differ is the point: were the two the same, refusing
    one of them would be pointless ceremony.
    """
    x = torch.tensor([-7.5, 7.5, -1.0, 1.0], dtype=torch.float16)
    y = torch.tensor([2.0, -2.0, 3.0, -3.0], dtype=torch.float16)
    assert not torch.equal(torch.fmod(x, y), torch.remainder(x, y))

    program = _program(_Fmod("remainder"), (x, y))
    assert _delegates(program) == [], "the floored remainder reached the delegate"


def test_a_zero_divisor_answers_zero_where_torch_answers_nan():
    """The one place the kernel's answer differs from torch's, pinned.

    The kernel guards the divisor so an inf quotient cannot reach its int32
    cast; torch lets the division happen and returns NaN. This is a divergence
    the graph inherits, not a bug in the emitter, and the test states it so a
    later reader does not have to find it on device.
    """
    x = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float16)
    zero = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float16)
    blob, _ = _delegated(_Fmod(), (x, zero))
    got = _run(blob, (x, zero))
    assert not np.isnan(got).any()
    assert got.tobytes() == np.zeros(3, dtype=np.float16).tobytes()
    assert torch.isnan(torch.fmod(x, zero)).all()


@pytest.mark.parametrize("kind", ["fmod", "remainder"])
def test_the_host_interpreter_computes_the_kernel_s_remainder(kind):
    """The transcription of the scalar path, against torch's own two functions.

    `HTP_OPS_BINARY_MOD` is absent from the vector-tail list
    (`htp_ops_binary_supports_fp16_vector_tail`, eltwise_ops.cc:517-522), so the
    kernel takes the same per-element float path this models; a value here is
    exact for every pair the guards do not catch.
    """
    x = torch.randn(64, dtype=torch.float16) * 8
    y = (torch.randn(64, dtype=torch.float16) + 0.5).abs() * 4 + 0.25
    blob, _ = _delegated(_Fmod(), (x, y))
    got = _run(blob, (x, y))
    expected = (
        torch.fmod(x, y) if kind == "fmod" else torch.remainder(x, y)
    ).numpy()
    if kind == "fmod":
        assert got.tobytes() == expected.tobytes()
    else:
        assert not np.array_equal(got, expected)
