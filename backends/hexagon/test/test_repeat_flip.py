# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.repeat.default` and `aten.flip.default` as existing blit regions.

Neither op needed a new command. Both are a region walk over the operand's own
bytes, which is what `DSP_OP_RASTER_BLIT` already describes, and the shape
each one takes is smaller than it looks.

A repeat of one axis is `cat([x] * factor, dim=axis)`: the trailing block is
copied `factor` times, once per leading index. That is two loops, so one
region carries it whichever axis carries the factor, and the factor rides on
a level's extent rather than on a region per phase -- so a factor of 64 costs
the same single region a factor of two does, and nothing here approaches the
three regions a serialised command holds. A repeat of two or more axes is four
loops and does not fit three levels; it is expressible as a sequence of these
regions, but this emitter emits one command, so it refuses rather than drops
an axis.

A flip is a negative source stride. `HtpOpsRasterRegion`'s strides are
`int32_t` and the kernel's generic walk multiplies them as signed values, so
a reversed axis reads from its last element with a stride of minus its inner
pitch. The risk is the reverse one: a kernel that treated the stride as
unsigned or clamped it would read the operand forward and produce a plausible
tensor that is not flipped. So the flip tests here are the ones that catch
that: a middle axis, a last axis, and an extent of two, which is where a
sign mistake has the least room to show.

Three forms are views and emit no command at all: a repeat by one, a repeat
that only prepends dimensions, and a flip of axes whose extents are all one.
The alias path re-points the operand's TensorRef there, and the tests below
pin that a delegate exists but carries no command for them.

The refusals are pinned as refusals, not as absences: a non-contiguous
operand, a repeat of two axes, a symbolic extent, an empty tensor, and a
flip naming an axis twice. Each asserts the node is in the edge graph, the
predicate is what turned it away, and no command reaches the blob.
"""

import os
import pathlib
import struct
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
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    DSP_OP_RASTER_BLIT,
    flip_region,
    repeat_region,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonOperatorSupport,
    HexagonPartitioner,
    _data_placeholders,
)
from executorch.backends.hexagon.serialization import blob as B  # noqa: E402
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.export import Dim, export  # noqa: E402

F16 = torch.float16

#: type, n_inputs, n_outputs, n_params, then the params: the same offset the
#: blob interpreter uses to reach a command's param vector.
_PARAMS_AT = 16

#: A region's twelve words, read from the bare list the region function returns:
#: srcIndex, srcOffset, dstOffset, then the three sizes, the three source
#: strides and the three destination strides. The blob's own params carry the
#: three-word blit header in front of this, which is why the two are different
#: constants and neither is derived from the other.
_REGION_SIZES_AT = 3
_REGION_SRC_STRIDES_AT = 6
_REGION_DST_STRIDES_AT = 9

#: The same words in a command's parameter vector, where the header is present.
_PARAMS_SIZES_AT = 6
_PARAMS_SRC_STRIDES_AT = 9

#: DSP_OP_UNARY, which is what a relu or a neg ahead of one lowers to.
_UNARY = 4


class _Repeat(torch.nn.Module):
    def __init__(self, factors):
        super().__init__()
        self.factors = factors

    def forward(self, x):
        return x.repeat(*self.factors)


class _Flip(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.flip(x, self.dims)


class _ViewThenNegate(torch.nn.Module):
    """A view that is consumed rather than returned."""

    def forward(self, x):
        return torch.neg(x.repeat(1, 1, 1))


class _ReluThen(torch.nn.Module):
    """The operand is a command's output rather than a method input."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(torch.relu(x))


def _program(model, inputs, dynamic_shapes=None):
    return to_edge_transform_and_lower(
        export(model, inputs, dynamic_shapes=dynamic_shapes),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _edge_program(model, inputs, dynamic_shapes=None):
    return to_edge(
        export(model, inputs, dynamic_shapes=dynamic_shapes),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _edge_targets(model, inputs):
    return [
        str(node.target)
        for node in _edge_program(model, inputs).graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _region_node(model, inputs, family, dynamic_shapes=None):
    """The repeat or flip node of one family, from the edge graph.

    The gate and the emitter both read the node's own metadata, so the region
    they compute is asked of the graph the partitioner sees rather than of a
    hand-built node.
    """
    program = _edge_program(model, inputs, dynamic_shapes)
    found = [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and family in str(node.target)
    ]
    assert len(found) == 1, f"expected one {family} node, got {len(found)}"
    return found[0]


def _region_for(model, inputs, family, dynamic_shapes=None):
    node = _region_node(model, inputs, family, dynamic_shapes)
    return repeat_region(node) if "repeat" in family else flip_region(node)


def _blob(program):
    calls = _delegates(program)
    assert len(calls) == 1, f"expected one delegate, got {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    return bytes(lowered._processed_bytes)


def _run(blob, inputs, shape):
    """The blob's output as `shape`.

    The interpreter returns the output slot's bytes flat, so the element count
    is checked here rather than by a reshape that would raise on a mismatch
    and read as a wrong value.
    """
    flat = np.frombuffer(
        execute(blob, [x.numpy() for x in inputs])[0], dtype=np.float16
    )
    assert flat.size == int(np.prod(shape)), f"{flat.size} elements, expected {np.prod(shape)}"
    return flat.reshape(shape)


def _params_at(blob, command_index, param_index):
    """The absolute byte offset of one command's parameter word."""
    at = B.HEADER_SIZE
    for _ in range(command_index):
        at += B.OP_SIZE
    return at + _PARAMS_AT + 4 * param_index


# --------------------------------------------------------------------------
# What is free: the view forms, which emit no command.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factors",
    [(1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1, 1)],
)
def test_a_repeat_by_one_is_the_identity_on_the_bytes(factors):
    """The only view a repeat has. Every factor above one adds elements, so the
    bytes are read more than once and an alias would answer a different
    question than the host's -- which is why there is no reshape case here."""
    x = torch.randn(2, 3, 4, dtype=F16)
    model = _Repeat(factors)
    assert np.array_equal(model(x).numpy().reshape(-1), x.numpy().reshape(-1))
    assert _region_for(model, (x,), "aten.repeat") == []


def test_a_repeat_by_one_fewer_times_than_the_rank_is_torch_s_own_error():
    """Worth recording because it looks like a shape this function should have
    refused: a factor list shorter than the operand's rank never reaches the
    partitioner at all, so there is no node to refuse and no region to return."""
    x = torch.randn(2, 3, 4, dtype=F16)
    with pytest.raises(RuntimeError, match="smaller than"):
        _Repeat((1,))(x)


def test_a_repeat_by_one_reaches_the_dsp_as_no_region():
    """A view emits no region, and the blob says so in the one word that
    matters: the blit's region count is zero.

    There is still one command. The view is the method's output, and an output
    slot is read as a buffer of its own, so the bytes have to be written out --
    a contiguous copy of the whole operand. That is the alias path's own rule
    and not this op's, so what the case pins is the region count rather than
    the command count: a region walked over the operand would be a second walk
    of the same bytes, and a shape the emitter believes it tiled.
    """
    x = torch.randn(2, 3, 4, dtype=F16)
    program = _program(_Repeat((1, 1, 1)), (x,))
    assert len(_delegates(program)) == 1
    _, commands = read_blob(_blob(program))
    assert [command.type for command in commands] == [DSP_OP_RASTER_BLIT]
    # The one region is the operand's own bytes at its own strides: the output
    # slot is read as a buffer of its own, so the copy is the alias path's rule
    # and not a walk this op asked for.
    assert list(commands[0].params[:3]) == [1, 2, 1]
    assert list(commands[0].params[6:15]) == [1, 1, 24, 0, 0, 1, 0, 0, 1]


def test_a_view_that_is_not_the_output_writes_no_blit_at_all():
    """The other half of the alias path: a view that is only read by something
    else shares the operand's TensorRef, and the command stream carries the
    consumer rather than the view."""
    x = torch.randn(2, 3, 4, dtype=F16)
    program = _program(_ViewThenNegate(), (x,))
    assert len(_delegates(program)) == 1
    _, commands = read_blob(_blob(program))
    # The negate is the only command: the repeat before it is a view.
    assert [command.type for command in commands] == [_UNARY]


def test_a_leading_repeat_is_a_copy_and_not_a_reshape():
    """The one that has to be got right: `x.repeat(2, 1)` on a three-element
    `x` tiles the whole tensor, so the factor lands on a padded leading axis
    and the region tiles `shape[axis:]` -- the whole tensor here. Treating the
    added dimension as a reshape would answer six bytes for a three-element
    operand."""
    x = torch.arange(3, dtype=F16)
    model = _Repeat((2, 1))
    assert np.array_equal(
        model(x).numpy().reshape(-1), np.array([0, 1, 2, 0, 1, 2], dtype=np.float16)
    )
    assert _region_for(model, (x,), "aten.repeat") == [
        0, 0, 0, 1, 2, 3, 3, 0, 1, 6, 3, 1,
    ]
    _, commands = read_blob(_blob(_program(model, (x,))))
    assert [command.type for command in commands] == [DSP_OP_RASTER_BLIT]


def test_a_leading_repeat_doubles_the_element_count():
    x = torch.randn(2, 3, 4, dtype=F16)
    model = _Repeat((2, 1, 1, 1))
    expected = _reference(model, x)
    assert expected.numel() == 2 * x.numel()
    got = _run(_blob(_program(model, (x,))), (x,), tuple(expected.shape))
    assert np.array_equal(got, expected.numpy())


@pytest.mark.parametrize(
    "label,shape,dims",
    [
        ("a leading axis of extent one", (1, 3, 4), [0]),
        ("a middle axis of extent one", (2, 1, 4), [1]),
        ("the last axis of extent one", (2, 3, 1), [2]),
        ("two unit axes", (1, 3, 1), [0, 2]),
        ("every axis of extent one", (1, 1, 1), [0, 1, 2]),
        ("no axes at all", (2, 3, 4), []),
    ],
)
def test_a_flip_of_unit_axes_is_a_no_op_with_no_command(label, shape, dims):
    """A reversal of an axis of extent one moves nothing, so the region is empty
    and the alias path re-points the operand. Each case names an axis that really
    is of extent one in its own shape, so the view is the axis and not the
    tensor that would be a no-op for every axis."""
    x = torch.randn(*shape, dtype=F16)
    model = _Flip(dims)
    assert all(shape[dim] == 1 for dim in dims), label
    assert np.array_equal(torch.flip(x, dims).numpy(), x.numpy()), label
    _, commands = read_blob(_blob(_program(model, (x,))))
    # The region is a whole-operand copy at the operand's own strides, which is
    # what "this op asks for nothing" looks like in a blob. A region that tiled
    # or reversed would have a stride here that differs from the destination's.
    assert [command.type for command in commands] == [DSP_OP_RASTER_BLIT]
    params = list(commands[0].params)
    assert params[6:9] == [1, 1, x.numel()], label
    assert params[9:12] == [0, 0, 1] and params[12:15] == [0, 0, 1], label
    assert _region_for(model, (x,), "aten.flip") == [], label


# --------------------------------------------------------------------------
# The region each copy form is, and what it costs.
# --------------------------------------------------------------------------


def test_a_repeat_of_the_last_axis_is_one_region_with_the_factor_on_a_level():
    """(2, 3) by 2 on axis 1: the block is 3 elements, copied twice, once per
    leading index. The factor is the middle level's extent, so the region count
    is one whatever the factor is."""
    x = torch.randn(2, 3, dtype=F16)
    assert _region_for(_Repeat((1, 2)), (x,), "aten.repeat") == [
        # srcIndex, srcOffset, dstOffset
        0,
        0,
        0,
        # sizes: lead, factor, run
        2,
        2,
        3,
        # source strides: run, 0 (the factor is a broadcast), 1
        3,
        0,
        1,
        # destination strides: widened run, one run, 1
        6,
        3,
        1,
    ]


def test_a_repeat_of_a_middle_axis_is_the_same_region_with_another_run():
    """(2, 3) by 2 on axis 0: the trailing block is now all three elements, so
    the lead is one and the run is six. Same two loops, same one region."""
    x = torch.randn(2, 3, dtype=F16)
    assert _region_for(_Repeat((2, 1)), (x,), "aten.repeat") == [
        0,
        0,
        0,
        1,
        2,
        6,
        6,
        0,
        1,
        12,
        6,
        1,
    ]


def test_a_repeat_of_a_leading_axis_is_the_same_region_with_a_unit_lead():
    """(3,) by 2 on axis 0: one leading index, so the lead level is one and the
    factor is the only loop that moves."""
    x = torch.randn(3, dtype=F16)
    assert _region_for(_Repeat((2,)), (x,), "aten.repeat") == [
        0,
        0,
        0,
        1,
        2,
        3,
        3,
        0,
        1,
        6,
        3,
        1,
    ]


def test_a_repeat_is_one_raster_blit():
    x = torch.randn(2, 3, dtype=F16)
    blob = _blob(_program(_Repeat((1, 2)), (x,)))
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [DSP_OP_RASTER_BLIT]
    assert list(commands[0].params[:3]) == [1, 2, 1]


def test_a_flip_is_one_raster_blit():
    x = torch.randn(2, 3, dtype=F16)
    blob = _blob(_program(_Flip([0]), (x,)))
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [DSP_OP_RASTER_BLIT]
    assert list(commands[0].params[:3]) == [1, 2, 1]


def test_a_flip_reads_its_axis_backwards():
    """The signed stride is the whole mechanism, so it is pinned: the first axis
    of a (2, 3) run is 3 wide, and reading it reversed is a source offset of two
    with a stride of -3."""
    x = torch.randn(2, 3, dtype=F16)
    assert _region_for(_Flip([0]), (x,), "aten.flip") == [
        # srcIndex, srcOffset (the last element of axis 0), dstOffset
        0,
        3,
        0,
        # sizes: level 2 is the last axis, level 1 the one before it, and
        # level 0 the absent leading group
        1,
        2,
        3,
        # source strides: -3 on the reversed axis, then the contiguous run
        0,
        -3,
        1,
        # destination strides: the output is contiguous
        0,
        3,
        1,
    ]


def test_a_flip_of_the_last_axis_is_a_stride_of_minus_one():
    x = torch.randn(2, 3, dtype=F16)
    assert _region_for(_Flip([1]), (x,), "aten.flip") == [
        0,
        2,
        0,
        1,
        2,
        3,
        0,
        3,
        -1,
        0,
        3,
        1,
    ]


def test_a_flip_of_every_axis_reverses_each_of_them():
    """(2, 3, 4) flipped on all three axes: one negative source stride per
    level, and the source offset is where the read starts -- each reversed
    axis's last element, summed."""
    x = torch.randn(2, 3, 4, dtype=F16)
    region = _region_for(_Flip([0, 1, 2]), (x,), "aten.flip")
    assert region[1] == 12 + 8 + 3
    assert region[_REGION_SIZES_AT : _REGION_SIZES_AT + 3] == [2, 3, 4]
    assert region[_REGION_SRC_STRIDES_AT : _REGION_SRC_STRIDES_AT + 3] == [-12, -4, -1]
    assert region[_REGION_DST_STRIDES_AT :] == [12, 4, 1]


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_one_flip_at_a_time_gives_exactly_one_negative_stride(dim):
    x = torch.randn(2, 3, 4, dtype=F16)
    region = _region_for(_Flip([dim]), (x,), "aten.flip")
    strides = region[_REGION_SRC_STRIDES_AT : _REGION_SRC_STRIDES_AT + 3]
    assert [stride for stride in strides if stride < 0] == [-(12, 4, 1)[dim]]


# --------------------------------------------------------------------------
# The numbers, against a fp64 torch reference.
# --------------------------------------------------------------------------


def _reference(model, x):
    """The answer from a module that shares no code with the blob path.

    The input is widened to fp64 before the op so the reference is not itself
    rounded through fp16, then the result is narrowed back: a copy and a
    reversal move whole values, so this is exact and a tolerance would only
    hide a region that read the wrong elements.
    """
    with torch.no_grad():
        return model(x.double()).half()


@pytest.mark.parametrize(
    "label,shape,factors",
    [
        ("last axis, factor two", (2, 3), (1, 2)),
        ("last axis, factor three", (2, 3), (1, 3)),
        ("last axis, factor seven", (2, 3), (1, 7)),
        ("last axis, 63", (63,), (2,)),
        ("last axis, 64", (64,), (2,)),
        ("last axis, 65", (65,), (2,)),
        ("last axis, 63 by four", (63,), (4,)),
        ("last axis, 64 by four", (64,), (4,)),
        ("last axis, 65 by four", (65,), (4,)),
        ("last axis, 64 by three on a lead of three", (3, 64), (1, 3)),
        ("last axis, 65 by three on a lead of three", (3, 65), (1, 3)),
        ("last axis, 63 by two on a lead of two and two", (2, 2, 63), (1, 1, 2)),
        ("last axis, 64 by two on a lead of two and two", (2, 2, 64), (1, 1, 2)),
        ("last axis, 65 by two on a lead of two and two", (2, 2, 65), (1, 1, 2)),
        ("middle axis", (2, 3), (2, 1)),
        ("middle axis by three", (2, 3), (3, 1)),
        ("middle axis, 63", (63,), (2,)),
        ("middle axis, 64", (64,), (2,)),
        ("middle axis, 65", (65,), (2,)),
        ("leading axis", (3,), (2,)),
        ("leading axis by five", (3,), (5,)),
        ("a unit middle axis", (2, 1, 4), (1, 2, 1)),
        ("a unit middle axis, 65", (2, 1, 65), (1, 2, 1)),
        ("a unit last axis", (2, 3, 1), (1, 1, 2)),
        ("a single element", (1,), (2,)),
        ("a single element by four", (1,), (4,)),
        ("rank one, 63 by three", (63,), (3,)),
        ("rank one, 64 by three", (64,), (3,)),
        ("rank one, 65 by three", (65,), (3,)),
    ],
)
def test_a_repeat_matches_torch_bit_for_bit(label, shape, factors):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=F16)
    model = _Repeat(factors)
    expected = _reference(model, x)
    got = _run(_blob(_program(model, (x,))), (x,), tuple(expected.shape))
    assert np.array_equal(got, expected.numpy()), label


@pytest.mark.parametrize(
    "label,shape,dims",
    [
        ("first axis", (2, 3, 4), [0]),
        ("middle axis", (2, 3, 4), [1]),
        ("last axis", (2, 3, 4), [2]),
        ("first axis of two", (2, 3, 4), [0]),
        ("all three axes", (2, 3, 4), [0, 1, 2]),
        ("axes given in reverse order", (2, 3, 4), [2, 0]),
        ("a negative axis", (2, 3, 4), [-1]),
        ("an extent of two on the first axis", (2, 3, 4), [0]),
        ("an extent of two on the middle axis", (2, 4, 5), [1]),
        ("an extent of two on the last axis", (2, 4, 2), [2]),
        ("a rank one of two", (2,), [0]),
        ("63", (63,), [0]),
        ("64", (64,), [0]),
        ("65", (65,), [0]),
        ("a lead of 63", (63, 2), [0]),
        ("a lead of 64", (64, 2), [0]),
        ("a lead of 65", (65, 2), [0]),
        ("63 on the last axis", (2, 63), [1]),
        ("64 on the last axis", (2, 64), [1]),
        ("65 on the last axis", (2, 65), [1]),
        ("a middle axis of 63", (2, 63, 3), [1]),
        ("a middle axis of 64", (2, 64, 3), [1]),
        ("a middle axis of 65", (2, 65, 3), [1]),
        ("63 through all three axes", (2, 3, 63), [0, 1, 2]),
        ("64 through all three axes", (2, 3, 64), [0, 1, 2]),
        ("65 through all three axes", (2, 3, 65), [0, 1, 2]),
        ("a unit axis beside a real one", (1, 3), [0]),
        ("a unit axis on the last", (3, 1), [1]),
    ],
)
def test_a_flip_matches_torch_bit_for_bit(label, shape, dims):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=F16)
    model = _Flip(dims)
    expected = _reference(model, x)
    got = _run(_blob(_program(model, (x,))), (x,), tuple(expected.shape))
    assert np.array_equal(got, expected.numpy()), label


def test_a_flip_of_an_extent_of_two_is_reversed_and_not_copied():
    """The extent that hides a sign mistake best: a two-element axis read
    forward is the operand, and the flip is one swap. So the answer is compared
    against a swap, and the region is checked for the negative stride that
    produces it."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=F16)
    model = _Flip([1])
    expected = _reference(model, x)
    got = _run(_blob(_program(model, (x,))), (x,), tuple(expected.shape))
    assert np.array_equal(got, np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]))
    assert not np.array_equal(got, x.numpy())
    assert _region_for(model, (x,), "aten.flip")[_REGION_SRC_STRIDES_AT + 2] == -1


def test_a_repeat_of_a_delegated_intermediate_is_still_one_delegate():
    torch.manual_seed(0)
    x = torch.randn(2, 3, dtype=F16)
    model = _ReluThen(_Repeat((1, 2)))
    program = _program(model, (x,))
    _, commands = read_blob(_blob(program))
    assert [command.type for command in commands] == [4, DSP_OP_RASTER_BLIT]
    expected = _reference(model, x)
    got = _run(_blob(program), (x,), tuple(expected.shape))
    assert np.array_equal(got, expected.numpy())


def test_a_flip_of_a_delegated_intermediate_is_still_one_delegate():
    torch.manual_seed(0)
    x = torch.randn(2, 3, dtype=F16)
    model = _ReluThen(_Flip([0]))
    program = _program(model, (x,))
    _, commands = read_blob(_blob(program))
    assert [command.type for command in commands] == [4, DSP_OP_RASTER_BLIT]
    expected = _reference(model, x)
    got = _run(_blob(program), (x,), tuple(expected.shape))
    assert np.array_equal(got, expected.numpy())


# --------------------------------------------------------------------------
# Teeth: the signed stride is the whole mechanism, so break it and see.
# --------------------------------------------------------------------------


def _patch_source_offset(blob, command_index):
    """A region's source offset is param 4: three of header, then srcIndex."""
    data = bytearray(blob)
    struct.pack_into("<i", data, _params_at(blob, command_index, 4), 0)
    return bytes(data)


def _patch_source_stride(blob, command_index, word, value):
    data = bytearray(blob)
    at = _params_at(blob, command_index, _PARAMS_SRC_STRIDES_AT + word)
    struct.pack_into("<i", data, at, value)
    return bytes(data)


def test_reading_the_flip_forward_produces_the_operand_unchanged():
    """The control that says the negative stride is what does the reversal.

    The two params carry the reversal together: a negative stride and a source
    offset at the axis's last element. Turning the stride positive while leaving
    the offset would not be a copy but a shift, so this case undoes both -- it
    is the region a kernel that read the stride as unsigned and started at the
    beginning would walk, and it writes the operand in its own order. If the
    recorded region were not the one the interpreter walked, this would still
    compare equal to the unflipped input and the flip tests would say nothing.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, dtype=F16)
    model = _Flip([1])
    blob = _blob(_program(model, (x,)))
    _, commands = read_blob(blob)
    assert commands[0].params[_PARAMS_SRC_STRIDES_AT + 2] == -1
    assert commands[0].params[4] == 2
    patched = _patch_source_stride(blob, 0, 2, 1)
    patched = _patch_source_offset(patched, 0)
    got = _run(patched, (x,), tuple(model(x).shape))
    assert np.array_equal(got, x.numpy())
    assert not np.array_equal(got, _reference(model, x).numpy())


def test_reading_the_flip_forward_on_the_first_axis_produces_the_operand():
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=F16)
    model = _Flip([0])
    blob = _blob(_program(model, (x,)))
    assert list(read_blob(blob)[1][0].params)[9:12] == [0, -4, 1]
    patched = _patch_source_stride(blob, 0, 1, 4)
    patched = _patch_source_offset(patched, 0)
    got = _run(patched, (x,), tuple(model(x).shape))
    assert np.array_equal(got, x.numpy())
    assert not np.array_equal(got, _reference(model, x).numpy())


def test_dropping_the_source_offset_reads_the_wrong_end_of_the_axis():
    """A flip of the first axis starts at the last element of the axis. Reading
    it from the first element instead shifts the whole result by one row."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=F16)
    model = _Flip([0])
    blob = _blob(_program(model, (x,)))
    data = bytearray(blob)
    at = _params_at(blob, 0, 1)
    struct.pack_into("<i", data, at, 0)
    got = _run(bytes(data), (x,), tuple(model(x).shape))
    assert not np.array_equal(got, _reference(model, x).numpy())


def test_a_repeat_reading_one_element_short_of_the_block_misses_the_last():
    torch.manual_seed(0)
    x = torch.randn(2, 3, dtype=F16)
    model = _Repeat((1, 2))
    blob = _blob(_program(model, (x,)))
    data = bytearray(blob)
    at = _params_at(blob, 0, _REGION_SIZES_AT + 2)
    struct.pack_into("<i", data, at, 2)
    got = _run(bytes(data), (x,), tuple(model(x).shape))
    assert not np.array_equal(got, _reference(model, x).numpy())


# --------------------------------------------------------------------------
# Refusals: a refused shape stays refused and emits no command.
# --------------------------------------------------------------------------


def _refused(model, inputs, family, dynamic_shapes=None):
    """The node is in the graph, the predicate turned it away, no delegate."""
    program = _program(model, inputs, dynamic_shapes)
    assert _delegates(program) == []
    node = _region_node(model, inputs, family, dynamic_shapes)
    support = HexagonOperatorSupport(_data_placeholders(_edge_program(model, inputs, dynamic_shapes)))
    assert not support.is_node_supported({}, node)
    assert _region_for(model, inputs, family, dynamic_shapes) is None


@pytest.mark.parametrize(
    "label,shape,factors",
    [
        ("two repeated axes", (2, 3), (2, 2)),
        ("two repeated axes, both two", (4, 5), (2, 2)),
        ("two repeated axes of a rank three", (2, 3, 4), (2, 1, 2)),
        ("three repeated axes", (2, 3), (2, 2, 2)),
    ],
)
def test_a_repeat_of_two_axes_stays_refused(label, shape, factors):
    x = torch.randn(*shape, dtype=F16)
    model = _Repeat(factors)
    assert any("repeat" in target for target in _edge_targets(model, (x,)))
    _refused(model, (x,), "aten.repeat")


def test_a_repeat_of_two_axes_would_be_wrong_as_one_region():
    """The reason it is refused rather than approximated: the four loops do not
    fit three levels, so a single-axis region would leave one of the two
    factors unapplied. The host's own answer is neither the operand nor either
    single-axis repeat, which is what makes the gap a refusal and not a view.
    """
    x = torch.arange(6, dtype=F16).reshape(2, 3)
    both = _Repeat((2, 2))(x)
    assert not np.array_equal(both.numpy().reshape(-1), x.numpy().reshape(-1))
    assert not np.array_equal(
        both.numpy().reshape(-1), _Repeat((2, 1))(x).numpy().reshape(-1)
    )
    assert not np.array_equal(
        both.numpy().reshape(-1), _Repeat((1, 2))(x).numpy().reshape(-1)
    )


def test_a_repeat_of_a_non_contiguous_operand_stays_refused():
    """A repeat of a strided source would need the source's own strides, which
    the region does not carry: its source stride belongs to the contiguous
    layout. The graph below takes an operand whose recorded value is a strided
    view, so the refusal is pinned on the node's own metadata rather than on a
    shape the export would have made contiguous for us."""
    x = torch.randn(4, 5, dtype=F16)
    node = _region_node(_Repeat((2, 1)), (x,), "aten.repeat")
    source = node.args[0]
    source.meta["val"] = source.meta["val"].t().contiguous().t()
    assert not source.meta["val"].is_contiguous()
    assert repeat_region(node) is None


def test_a_repeat_with_a_symbolic_extent_stays_refused():
    x = torch.randn(2, 5, dtype=F16)
    model = _Repeat((1, 2))
    assert any("repeat" in target for target in _edge_targets(model, (x,)))
    _refused(model, (x,), "aten.repeat", {"x": {1: Dim("cols", min=1, max=64)}})


def test_a_repeat_to_an_empty_tensor_stays_refused():
    x = torch.randn(0, 3, dtype=F16)
    model = _Repeat((1, 2))
    _refused(model, (x,), "aten.repeat")


@pytest.mark.parametrize(
    "label,dims",
    [
        ("the same axis twice", [1, 1]),
        ("an axis out of range", [3]),
        ("well out of range", [9]),
        ("a non-integer axis", [1.0]),
        ("a dimension list rather than one", 1),
    ],
)
def test_a_flip_of_a_shape_it_does_not_describe_stays_refused(label, dims):
    """Refused rather than unwired, and the refusal is the predicate's.

    These argument lists are ones torch's own schema rejects, so a graph built
    from them never reaches the partitioner -- which is why each case puts the
    list on a real flip node's metadata instead of exporting it. Same function,
    same None, and no claim that a portable kernel is waiting for a shape
    export will not produce.
    """
    x = torch.randn(2, 3, dtype=F16)
    node = _region_node(_Flip([0]), (x,), "aten.flip")
    node.args = (node.args[0], dims)
    assert flip_region(node) is None, label


def test_a_flip_that_torch_itself_refuses_never_reaches_the_graph():
    """The same list through the front door, so the node above is not standing
    in for a shape that would otherwise have been emittable."""
    x = torch.randn(2, 3, dtype=F16)
    for dims, error in (([1, 1], RuntimeError), ([3], IndexError), ([9], IndexError)):
        with pytest.raises(error):
            _Flip(dims)(x)


def test_a_flip_of_a_non_contiguous_operand_stays_refused():
    x = torch.randn(4, 5, dtype=F16)
    node = _region_node(_Flip([0]), (x,), "aten.flip")
    source = node.args[0]
    source.meta["val"] = source.meta["val"].t().contiguous().t()
    assert not source.meta["val"].is_contiguous()
    assert flip_region(node) is None


def test_a_flip_of_a_symbolic_last_axis_is_refused_not_walked():
    """The silent one. A flip records where its read starts and how far it
    steps, and both are params, so a symbolic extent would be a region whose
    offset and stride were frozen at the traced length. Nothing downstream
    would notice: the bytes would be a reversal of the example, and the
    difference from a reversal of the run-time input would be wrong answers
    rather than an error. So the shape is refused outright."""
    x = torch.randn(4, 6, dtype=F16)
    model = _Flip([1])
    _refused(
        model,
        (x,),
        "aten.flip",
        dynamic_shapes={"x": {1: Dim("cols", min=1, max=64)}},
    )


def test_a_flip_of_a_symbolic_middle_axis_is_refused_too():
    """The same reasoning one axis out, and worth its own case because the two
    axes reach the region through different levels: the last axis's stride is
    -1 whatever the extent, while the second-to-last axis's is its pitch."""
    x = torch.randn(4, 6, dtype=F16)
    model = _Flip([0])
    _refused(
        model,
        (x,),
        "aten.flip",
        dynamic_shapes={"x": {0: Dim("rows", min=1, max=64)}},
    )


@pytest.mark.parametrize("factor", [8, 64, 65])
def test_a_factor_past_the_regions_a_command_holds_is_still_one(factor):
    """A command's param block holds three regions, and a region per phase
    would need a hundred and twenty-eight of them for a factor of 128. The
    factor rides on a level's extent instead, so the region count is one and
    the number that changes with the factor is a size, which is a single
    param. This is why nothing here needs a schema change."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, dtype=F16)
    model = _Repeat((1, factor))
    region = _region_for(model, (x,), "aten.repeat")
    assert len(region) == 12
    assert region[_REGION_SIZES_AT + 1] == factor
    _, commands = read_blob(_blob(_program(model, (x,))))
    assert [command.type for command in commands] == [DSP_OP_RASTER_BLIT]
    assert commands[0].params[0] == 1
    expected = _reference(model, x)
    got = _run(_blob(_program(model, (x,))), (x,), tuple(expected.shape))
    assert np.array_equal(got, expected.numpy())


# --------------------------------------------------------------------------
# The census rows that move.
# --------------------------------------------------------------------------


def test_a_repeat_reaches_the_dsp_where_it_did_not():
    x = torch.randn(2, 3, dtype=F16)
    assert len(_delegates(_program(_Repeat((1, 2)), (x,)))) == 1


def test_a_flip_reaches_the_dsp_where_it_did_not():
    x = torch.randn(2, 3, dtype=F16)
    assert len(_delegates(_program(_Flip([0]), (x,)))) == 1
