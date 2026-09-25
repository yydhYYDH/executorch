# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""softmax: the softmax command for the rows it is right for, five for wider ones.

There is a `DSP_OP_SOFTMAX` command and a kernel behind it, and for rows up to
one HVX vector long it agrees with torch. Above that width it does not: the
kernel's vector loop computes its exponential up to 1.76x the correctly rounded
value, as a function of the argument's fractional part, while the tail it copies
in and masks answers the same argument within an fp16 ulp. What that costs on a
device is measured in `../../README.md`'s companion notes and reproduced here in
the numbers the gate is stated against: a (3, 197, 197) softmax over uniform
logits came back wrong on 116284 of 116427 elements, mean relative error 9.6%,
row sums still one, and the last five columns -- the tail -- clean; at 63
columns, where the whole row is the tail, the same kernel is within rounding, and
at 64, where the whole row is the vector loop, it is not.

So this emitter keeps the command for rows shorter than a vector and writes every
wider row as the log-sum-exp the log_softmax emitter next door already uses, with
a division where that one has a log:

    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Five commands, no new kernel, and the same three kernels the shifted form already
spends: the reduction for the maximum and the sum, the unary table for the
exponential, and the element-wise op for the shift and the division. The tests
below are the two halves of that claim -- which commands a width gets, and what
the numbers are at the width the defect was found at.
"""

import os
import pathlib
import struct
import sys

import numpy as np
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    SOFTMAX_VECTOR_WIDTH,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import Dim, export  # noqa: E402

F16 = torch.float16

#: The DSP command types the two forms are made of.
_REDUCTION = 29
_BINARY = 19
_UNARY = 4
_SOFTMAX = 28
_MAXIMUM = 2
_SUM = 1
_SUB = 2
_DIV = 4
_EXP = 5

#: The width the defect was measured at: the ViT's key length, one row of
#: attention scores, which is one full vector plus a tail of five.
_VIT_WIDTH = 197
_VIT_ROWS = 591


class _Softmax(torch.nn.Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)


def _lowered(model, args):
    """The single delegate a whole model lowers to, and its commands."""
    program = to_edge_transform_and_lower(
        export(model, tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the model did not lower to one delegate: {calls}"
    inner = program.graph_module.get_submodule(calls[0].args[0].target)
    assert inner.backend_id == "HexagonBackend"
    blob = bytes(inner._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def test_a_wide_softmax_is_the_shifted_sum_of_exponentials():
    """Five commands, and the params that say what each one is.

    One row of the answer is `exp(x - max(x)) / sum(exp(x - max(x)))` and each
    command is one of those steps: the maximum over the row, the shift, the
    exponentials of at most one, their sum, and the division by it. The two
    reduced commands write a one-wide row and the two element-wise commands
    broadcast it back over the row.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, _VIT_WIDTH).half()
    blob, commands = _lowered(_Softmax(), (x,))
    assert [command.type for command in commands] == [
        _REDUCTION,
        _BINARY,
        _UNARY,
        _REDUCTION,
        _BINARY,
    ]
    assert _SOFTMAX not in [command.type for command in commands], (
        "the standalone softmax command reached a width the kernel is wrong at"
    )
    maximum, shifted, exponentials, total, out = commands
    assert list(maximum.params) == [6, _VIT_WIDTH, 1, _MAXIMUM, 2], list(maximum.params)
    assert list(total.params) == [6, _VIT_WIDTH, 1, _SUM, 2], list(total.params)
    # 6 rows of 197; the maximum is read as one value per row.
    assert list(shifted.params[:4]) == [6 * _VIT_WIDTH, 6 * _VIT_WIDTH, 6, _SUB], list(
        shifted.params[:4]
    )
    assert list(shifted.params[25:28]) == [3, 1, 0], (
        "the row maximum is not broadcast along the reduced axis"
    )
    assert list(exponentials.params) == [6 * _VIT_WIDTH, _EXP, 2], list(
        exponentials.params
    )
    assert list(out.params[:4]) == [6 * _VIT_WIDTH, 6 * _VIT_WIDTH, 6, _DIV], list(
        out.params[:4]
    )


def test_a_row_that_fits_one_vector_keeps_the_softmax_command():
    """The gate, and the width on either side of it.

    A row shorter than one HVX vector is the whole of the kernel's tail path, so
    the command is kept there: one command instead of five, and byte-for-byte the
    same command this emitter has always emitted. At the width of the vector
    itself the kernel is on its vector loop and the composition is what is left.
    """
    for width in (8, 63):
        x = torch.randn(2, width).half()
        _, commands = _lowered(_Softmax(), (x,))
        assert [command.type for command in commands] == [_SOFTMAX]
        assert list(commands[0].params) == [2, width, 1, 2], list(commands[0].params)
    for width in (SOFTMAX_VECTOR_WIDTH, SOFTMAX_VECTOR_WIDTH + 1, 1000):
        x = torch.randn(2, width).half()
        _, commands = _lowered(_Softmax(), (x,))
        assert _SOFTMAX not in [command.type for command in commands]
        assert len(commands) == 5


def test_the_softmax_command_is_used_only_below_the_vector_width():
    """The two widths the gate turns on, named once.

    The kernel's own chunking is what the 64 is: `vec_end = channel & -64` leaves
    a row of exactly 64 with no tail at all, and a row of 63 with nothing but
    tail. This asserts the emitter's constant against that arithmetic rather than
    against a copied literal.
    """
    assert SOFTMAX_VECTOR_WIDTH == 64
    assert 63 & -SOFTMAX_VECTOR_WIDTH == 0
    assert SOFTMAX_VECTOR_WIDTH & -SOFTMAX_VECTOR_WIDTH == SOFTMAX_VECTOR_WIDTH


def test_a_wide_softmax_answers_torch():
    """The composition's numbers, over the row shape the defect is measured at.

    The host model of the command stream runs the five commands in Python, so a
    green answer here says the shapes, the broadcast and the arithmetic the
    emitter describes are the softmax -- not that the kernels compute it, which is
    what `test_blob_on_sim.py` and the device notes beside this directory are for.
    """
    torch.manual_seed(0)
    for rows, width in ((4, 64), (4, 197), (591, 197)):
        x = (torch.rand(rows, width) * 2 - 1).half()
        blob, commands = _lowered(_Softmax(), (x,))
        assert len(commands) == 5
        got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)
        want = torch.softmax(x.float(), dim=-1).numpy().reshape(-1)
        np.testing.assert_allclose(got.astype(np.float32), want, rtol=2e-3, atol=2e-4)
        rows_got = got.astype(np.float32).reshape(rows, width).sum(1)
        assert np.abs(rows_got - 1.0).max() < 3e-3, (
            "the shifted exponentials are not normalised:"
            f" row sums {rows_got.min()}..{rows_got.max()}"
        )


#: The trailer's magic, and the offset of its patch records inside it.
_TRAILER_MAGIC = 0x44594E48


def _lowered_with_bound(width_bound):
    """A softmax whose last axis is dynamic, and the single delegate's commands."""
    x = torch.randn(1, 3, 6).half()
    program = to_edge_transform_and_lower(
        export(
            _Softmax(),
            (x,),
            dynamic_shapes={"x": {2: Dim("tokens", min=1, max=width_bound)}},
        ),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the model did not lower to one delegate: {calls}"
    blob = bytes(
        program.graph_module.get_submodule(calls[0].args[0].target)._processed_bytes
    )
    _, commands = read_blob(blob)
    return blob, commands


def test_a_dynamic_row_takes_the_gate_at_the_export_bound():
    """The gate reads the bound, and the composition is patched for every product.

    A width the export bounds below one vector keeps the single command and its
    one patch; at the vector width and above, the five commands carry the bound
    and the trailer's records recompute it at run time. The gate is conservative
    on purpose -- a bound of 64 takes the composition even if the row that arrives
    is shorter, which is the cost of not being able to see the values.
    """
    short_blob, short = _lowered_with_bound(16)
    assert [command.type for command in short] == [_SOFTMAX]
    assert list(short[0].params) == [3, 16, 1, 2], list(short[0].params)

    for bound in (SOFTMAX_VECTOR_WIDTH, 256):
        blob, commands = _lowered_with_bound(bound)
        assert [command.type for command in commands] == [
            _REDUCTION,
            _BINARY,
            _UNARY,
            _REDUCTION,
            _BINARY,
        ]
        assert list(commands[0].params[:5]) == [3, bound, 1, _MAXIMUM, 2], list(
            commands[0].params[:5]
        )
        assert list(commands[3].params[:5]) == [3, bound, 1, _SUM, 2], list(
            commands[3].params[:5]
        )
        offset = blob.find(struct.pack("<I", _TRAILER_MAGIC))
        assert offset >= 0, "a dynamic graph emits a trailer"
        header = struct.unpack_from("<7I", blob, offset)
        assert header[4] == bound, f"the trailer's longest length is {header[4]}"
        patched = set()
        for record in range(header[5]):
            index = struct.unpack_from("<4i", blob, offset + 28 + record * 16)[0]
            patched.add(index)
        assert patched == {0, 1, 2, 3, 4}, (
            "the composition's products are not all patched for the run-time width:"
            f" {sorted(patched)}"
        )


def test_a_softmax_over_a_row_narrower_than_a_vector_answers_torch():
    """The same numbers from the path the gate keeps, so the two agree.

    The host model runs the standalone command's own implementation here, which is
    torch's softmax to fp16: what this test pins is that the gate's two sides
    describe the same function, not which one is faster.
    """
    torch.manual_seed(0)
    for width in (2, 8, 63):
        x = (torch.rand(4, width) * 2 - 1).half()
        blob, commands = _lowered(_Softmax(), (x,))
        assert [command.type for command in commands] == [_SOFTMAX]
        got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)
        want = torch.softmax(x.float(), dim=-1).numpy().reshape(-1)
        np.testing.assert_allclose(got.astype(np.float32), want, rtol=2e-3, atol=2e-4)
