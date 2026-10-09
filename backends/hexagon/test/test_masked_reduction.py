# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`(a * m.unsqueeze(-1)).sum(-2)` as one DSP command.

`htp_ops_masked_reduction` sums an `[outside][reduce][inside]` activation over
the `reduce` axis, taking one fp16 mask entry per `(outside, reduce)` pair and
broadcasting it over `inside` (eltwise_ops.cc:2961-2969, :2901-2935). The shape
that reaches it, `(a * m.unsqueeze(-1)).sum(-2)`, already lowers to a multiply and
a reduction that answer correctly, so this is a fusion and not a new capability:
the one command reads the activation and the mask and writes the result, where
the two commands also wrote and read the whole product in between.

Two things the tests here are for. The merge has to leave the answer alone, so
the fused blob and the two-command blob are compared **bit for bit** on the same
inputs -- inside the admitted row width, where the two
paths are the same fp32 accumulation. And the kernel tests `mask != 0` rather
than `mask == 1`, so an operand that is not exactly 0.0/1.0 is summed at full
weight instead of being scaled: the gate can only prove that about a constant,
which is why a run-time mask and a constant holding 0.5 are both refused, and
each of those is a test rather than a comment.
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

from blob_interpreter import Arena, execute, read_blob  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_MASKED_REDUCTION and DSP_OP_REDUCTION, and the four params the masked
#: kernel reads after the span: the op type it accepts (SUM, 1) and the element
#: size it accepts (2).
_MASKED_REDUCTION = 31
_REDUCTION = 29
_BINARY = 19
_SUM = 1
_FP16_BYTES = 2

#: The span the unfused lowering writes for this shape, which is what the fused
#: command has to reproduce: the mask's multiply emitted the same extents as its
#: own broadcast tail, so the two blobs differ by a command and not by a number.
_UNFUSED_MUL_PARAMS = [
    24,
    24,
    6,
    3,
    2,
    2,
    0,
    0,
    3,
    2,
    3,
    4,
    0,
    0,
    0,
    0,
    0,
    12,
    4,
    1,
    0,
    0,
    0,
    0,
    0,
    3,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]


class _MaskedReduce(torch.nn.Module):
    """The one shape whose mask operand the kernel can be handed."""

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, a):
        return (a * self.mask.unsqueeze(-1)).sum(-2)


class _RuntimeMask(_MaskedReduce):
    """The same arithmetic with the mask as a method input."""

    def __init__(self) -> None:
        super().__init__(None)

    def forward(self, a, m):
        return (a * m.unsqueeze(-1)).sum(-2)


class _MaskedReduceWithProduct(_MaskedReduce):
    """The same sum, with the product read somewhere else as well."""

    def forward(self, a):
        product = a * self.mask.unsqueeze(-1)
        return product.sum(-2), product


def _program(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegate(inputs, model):
    program = _program(model, inputs)
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"this graph is {len(calls)} delegates, not one"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    blob = bytes(lowered._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def _bits(values):
    """A fp16 array as the 16-bit words the kernel writes, for an exact compare."""
    return np.ascontiguousarray(values).view(np.uint16)


def _mask_of(shape, seed=11, keep=0.5):
    generator = np.random.default_rng(seed)
    return torch.tensor(
        (generator.random(shape) < keep).astype(np.float16), dtype=torch.float16
    )


def _activation(shape, seed=5):
    generator = np.random.default_rng(seed)
    return torch.tensor(generator.standard_normal(shape) * 0.5, dtype=torch.float16)


def _reference(a, mask):
    """`(a * m.unsqueeze(-1)).sum(-2)` in fp32, which is what torch computes."""
    product = a.numpy().astype(np.float32) * mask.numpy()[:, :, None].astype(np.float32)
    return product.sum(axis=1).astype(np.float16)


# The shapes whose row is narrower than the kernel's vector path, which is where
# the fusion is exact. The last one is a single row of mask on a four-axis
# activation, whose leading axes fold into `outside`.
@pytest.mark.parametrize(
    "shape, mask_shape, span",
    [
        ((2, 3, 4), (2, 3), (2, 3, 4)),
        ((4, 2, 8), (4, 2), (4, 2, 8)),
        ((3, 5, 31), (3, 5), (3, 5, 31)),
        ((2, 3, 4, 8), (2, 3, 4), (6, 4, 8)),
    ],
)
def test_the_masked_sum_is_one_command_over_the_same_span(shape, mask_shape, span):
    """One command, the kernel's three extents, and the mask as its operand."""
    a = _activation(shape)
    mask = _mask_of(mask_shape)
    blob, commands = _delegate((a,), _MaskedReduce(mask))
    assert [command.type for command in commands] == [_MASKED_REDUCTION]
    assert list(commands[0].params) == [
        *span,
        _SUM,
        _FP16_BYTES,
    ], f"masked sum of {shape} emitted {list(commands[0].params)}"
    # The second operand is the mask itself: the kernel reads it as its own
    # operand rather than reading a product that was never written.
    header, _commands = read_blob(blob)
    arena = Arena(header, blob, "masked")
    execute(blob, [a.numpy()], arena=arena)
    assert len(commands[0].inputs) == 2
    operand = bytes(arena.view(commands[0].inputs[1]))
    assert operand == mask.numpy().tobytes(), "the second operand is not the mask"


@pytest.mark.parametrize("shape", [(2, 3, 4), (4, 8, 16), (3, 5, 31)])
def test_the_one_command_is_the_two_commands_answer_bit_for_bit(shape):
    """The primary claim: the merge moves the command count and not the result.

    Both blobs are run on the same activation, from the same mask. They are
    built from two models -- the constant mask fuses, a method input cannot --
    so this is the fused lowering against the unfused one rather than one blob
    against itself.
    """
    a = _activation(shape)
    mask = _mask_of(shape[:-1])
    fused_blob, fused = _delegate((a,), _MaskedReduce(mask))
    plain_blob, plain = _delegate((a, mask), _RuntimeMask())
    assert [command.type for command in fused] == [_MASKED_REDUCTION]
    assert [command.type for command in plain] == [_BINARY, _REDUCTION]
    got = np.frombuffer(execute(fused_blob, [a.numpy()])[0], dtype=np.float16)
    want = np.frombuffer(
        execute(plain_blob, [a.numpy(), mask.numpy()])[0], dtype=np.float16
    )
    assert np.array_equal(_bits(got), _bits(want)), (
        f"the fusion moved {int((_bits(got) != _bits(want)).sum())} of "
        f"{got.size} outputs by up to "
        f"{np.abs(got.astype(np.float32) - want.astype(np.float32)).max():.3e}"
    )


@pytest.mark.parametrize("shape", [(2, 3, 4), (4, 8, 16), (3, 5, 31)])
def test_the_fused_answer_is_torch_within_one_fp16_step(shape):
    """A reduction accumulates, so the reference is a tolerance and not equality.

    The tolerance is stated in the output's own fp16 step rather than as an
    absolute number: the fused path accumulates in fp32 over the rows the mask
    keeps, which is what torch's own sum does, so the two agree to the last bit
    of the accumulator's rounding.
    """
    a = _activation(shape)
    mask = _mask_of(shape[:-1])
    blob, _commands = _delegate((a,), _MaskedReduce(mask))
    got = np.frombuffer(execute(blob, [a.numpy()])[0], dtype=np.float16)
    want = _reference(a, mask).reshape(-1)
    worst = np.abs(got.astype(np.float32) - want.astype(np.float32)).max()
    step = np.spacing(np.abs(want).max().astype(np.float16))
    assert worst <= step, f"worst {worst:.3e} is past one fp16 step ({step:.3e})"


def test_a_run_time_mask_is_refused():
    """The precondition's other half: a mask the export cannot read stays two
    commands, whatever its values are, because nothing can prove them.

    The refusal is asserted as the command stream and its params, so "refused"
    cannot mean "emitted nothing": the multiply and the reduction are both
    there, and they are the ones this shape lowered to before the fusion.
    """
    shape = (2, 3, 4)
    a = _activation(shape)
    mask = _mask_of(shape[:-1])
    blob, commands = _delegate((a, mask), _RuntimeMask())
    assert [command.type for command in commands] == [_BINARY, _REDUCTION]
    assert list(commands[0].params) == _UNFUSED_MUL_PARAMS
    assert list(commands[1].params) == [2, 3, 4, _SUM, _FP16_BYTES]
    # And the two commands are the right answer, so the refusal is a slower
    # path rather than a broken one.
    got = np.frombuffer(execute(blob, [a.numpy(), mask.numpy()])[0], dtype=np.float16)
    assert np.array_equal(_bits(got), _bits(_reference(a, mask).reshape(-1)))


@pytest.mark.parametrize(
    "value", [0.5, 2.0, 0.25, -1.0, 1.0009765625, 65504.0]
)
def test_a_constant_mask_that_is_not_zero_or_one_is_refused(value):
    """The precondition itself: the kernel tests `mask != 0`, so 0.5 is summed
    at full weight rather than at half, and every one of these would be a wrong
    answer if it reached the kernel.

    1.0009765625 is the fp16 above 1.0, which is the smallest value that is
    neither the skip nor the keep the kernel can express.
    """
    shape = (2, 3, 4)
    a = _activation(shape)
    mask = torch.full(shape[:-1], value, dtype=torch.float16)
    blob, commands = _delegate((a,), _MaskedReduce(mask))
    assert [command.type for command in commands] == [_BINARY, _REDUCTION], (
        f"a mask of {value} reached the kernel: {[c.type for c in commands]}"
    )
    assert list(commands[1].params) == [2, 3, 4, _SUM, _FP16_BYTES]


def test_one_bad_entry_refuses_the_whole_mask():
    """The check is over every entry, not the first one.

    A mask that is 0.0/1.0 everywhere but one entry would be summed at full
    weight for that whole row, so the refusal has to be over the operand.
    """
    shape = (2, 3, 4)
    a = _activation(shape)
    mask = _mask_of(shape[:-1])
    mask[1, 2] = torch.tensor(0.5, dtype=torch.float16)
    _blob, commands = _delegate((a,), _MaskedReduce(mask))
    assert [command.type for command in commands] == [_BINARY, _REDUCTION]
    # The control on the same shape, one entry away: the unmodified mask fuses.
    _blob, commands = _delegate((a,), _MaskedReduce(_mask_of(shape[:-1])))
    assert [command.type for command in commands] == [_MASKED_REDUCTION]


@pytest.mark.parametrize(
    "inside, fused",
    [(1, True), (16, True), (31, True), (32, False), (40, False), (128, False)],
)
def test_the_admitted_row_width_is_the_plain_paths_scalar_limit(inside, fused):
    """The second condition, as a boundary rather than a verdict.

    Below 32 the reduction this command replaces is its scalar walk, which
    accumulates in fp32 and rounds once, and the masked kernel's own scalar tail
    is the same arithmetic over the same lanes. At 32 the plain kernel switches
    to a small-vector walk that is not fp32-exact, and above 64 the masked
    kernel starts rounding to fp16 after every unmasked row. Both of those move
    the answer, so both are refused -- the first of them measured on
    hexagon-sim in `test_masked_reduction_sim.py`, where the plain kernel at
    inside 40 is off an fp32 sum by up to 1.4 ULP and equal to it at inside 31.
    """
    shape = (2, 3, inside)
    a = _activation(shape)
    mask = _mask_of(shape[:-1])
    _blob, commands = _delegate((a,), _MaskedReduce(mask))
    wanted = [_MASKED_REDUCTION] if fused else [_BINARY, _REDUCTION]
    assert [command.type for command in commands] == wanted


def test_a_second_reader_of_the_product_keeps_both_commands():
    """The fused command replaces both, so the multiply may not have another
    reader: it would be left with no producer at all.

    This is the boundary in the other direction from the mask checks -- the
    pattern matches and the mask is provable, and the fusion still has to stand
    down.
    """
    shape = (2, 3, 4)
    a = _activation(shape)
    mask = _mask_of(shape[:-1])
    _blob, commands = _delegate((a,), _MaskedReduceWithProduct(mask))
    assert [command.type for command in commands] == [_BINARY, _REDUCTION]
    assert list(commands[0].params) == _UNFUSED_MUL_PARAMS


def test_a_fp32_mask_is_refused():
    """The kernel reads fp16, so a mask of another width cannot be checked.

    The values it would be narrowed to are not the values the graph carries, and
    the operand the kernel reads is two bytes per element whatever the graph
    declares.
    """
    shape = (2, 3, 4)
    a = _activation(shape)
    mask = _mask_of(shape[:-1]).to(torch.float32)
    _blob, commands = _delegate((a,), _MaskedReduce(mask))
    assert [command.type for command in commands] == [_BINARY, _REDUCTION]


def test_a_reduction_that_is_not_the_axis_before_the_last_is_refused():
    """The kernel's mask is indexed by `(outside, reduce)`, so the axis it
    reduces has to be the one the mask's own axes stand for.

    `sum(-1)` is a different shape -- the mask would have to be indexed by
    `(outside, inside)` -- and it stays two commands. So does a sum over two
    axes, whose flattened mask would be a different operand than the graph
    carries.
    """
    shape = (2, 3, 4)
    a = _activation(shape)
    mask = _mask_of(shape[:-1])
    _blob, commands = _delegate((a,), _MaskedReduce(mask))
    assert [command.type for command in commands] == [_MASKED_REDUCTION]

    class _LastAxis(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.register_buffer("mask", m)

        def forward(self, x):
            return (x * self.mask).sum(-1)

    _blob, commands = _delegate((a,), _LastAxis(_mask_of(shape)))
    assert [command.type for command in commands] == [_BINARY, _REDUCTION]
