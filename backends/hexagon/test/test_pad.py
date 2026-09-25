# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The zero-filling constant pad, as a memset and one region.

A pad is one of the few ATen ops that maps onto two commands the backend was
already emitting for other ops: `htp_ops_zero` clears the result, which is what
writes the border, and one `DSP_OP_RASTER_BLIT` region copies the operand into
the interior. Nothing new is asked of the library, which is why this was a table
entry rather than a kernel.

What the region can describe is bounded by its shape: three nested loops with a
stride per side. A pad on the last two axes is exactly that -- the axes before
them have the same extent on both sides, so the whole leading product is one
level advancing by a fixed output pitch. A pad on a third axis from the end
needs a fourth level, and what it would read is elsewhere rather than an error,
so the gate refuses it. The value is the other bound: the memset writes zero and
nothing else in the library fills a buffer without reading one.

The tests below pin the commands, the numbers bit for bit against torch, the
region itself, and the shapes the gate turns away. The last one pins something
that is not this op at all: `mode='reflect'` and `mode='replicate'` never reach
the partitioner as a pad node, because torch lowers them to a small indexing
program before export sees them.
"""

import struct

import numpy as np
import pytest
import torch


from blob_interpreter import execute, read_blob
from executorch.backends.hexagon.hexagon_ops import constant_pad_region
from executorch.backends.hexagon.reflect_pad import PreserveReflectPad
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as B
from executorch.exir import (
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.export import export

#: DSP_OP_ZERO and DSP_OP_RASTER_BLIT, the two commands a pad is; and
#: DSP_OP_UNARY, which is what a relu ahead of one lowers to.
_ZERO = 24
_RASTER_BLIT = 3
_UNARY = 4

#: type, n_inputs, n_outputs, n_params, then the params. The same offset the
#: blob interpreter uses to reach a command's param vector.
_PARAMS_AT = 16

#: The blit header is three ints (region count, element bytes, source count)
#: and then one region, so the destination offset is the third word of that
#: region.
_DST_OFFSET_PARAM = 3 + 2

F16 = torch.float16


class _Pad(torch.nn.Module):
    def __init__(self, pads, value=None, mode="constant"):
        super().__init__()
        self.pads = pads
        self.value = value
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.pad(x, self.pads, self.mode, self.value)


class _ReluThenPad(torch.nn.Module):
    """The operand is a command's output rather than a method input.

    This is the case the memset could break: the two commands write one buffer,
    so if the allocator ever gave the operand and the result the same block, the
    clear would run before the copy read it. They share no block because an op's
    inputs and outputs have overlapping live ranges by construction, and this
    test is what says so.
    """

    def __init__(self, pads):
        super().__init__()
        self.pads = pads

    def forward(self, x):
        return torch.nn.functional.pad(torch.relu(x), self.pads)


def _program(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _edge_targets(model, inputs):
    program = to_edge(
        PreserveReflectPad()(export(model, inputs)).exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    return [
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _blob(program):
    calls = _delegates(program)
    assert len(calls) == 1, f"expected one delegate, got {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    return bytes(lowered._processed_bytes)


def _run(blob, inputs, shape):
    """The blob's output as `shape`.

    The interpreter returns the output slot's bytes flat, so the element count is
    checked here rather than by a reshape that would raise on a mismatch and read
    as a wrong value.
    """
    flat = np.frombuffer(
        execute(blob, [x.numpy() for x in inputs])[0], dtype=np.float16
    )
    assert flat.size == int(
        np.prod(shape)
    ), f"{flat.size} elements, expected {np.prod(shape)}"
    return flat.reshape(shape)


# --------------------------------------------------------------------------
# The shape gate, at the function the predicate and the emitter share.
# --------------------------------------------------------------------------


def _region_for(shape, pads, value=None):
    """The region `constant_pad_region` gives for one pad, or None."""
    model = _Pad(pads, value)
    program = to_edge(
        export(model, (torch.zeros(*shape, dtype=F16),)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    node = next(
        node
        for node in program.graph_module.graph.nodes
        if "constant_pad_nd" in str(node.target)
    )
    return constant_pad_region(node)


def test_a_pad_on_the_last_axis_is_one_region():
    """(4, 6) padded by one each side: four rows of six, pitch ten."""
    assert _region_for((4, 6), (1, 1)) == [
        0,
        0,
        1,
        1,
        4,
        6,
        24,
        6,
        1,
        32,
        8,
        1,
    ]


def test_a_pad_on_the_last_two_axes_folds_the_leading_axes_into_one_level():
    """(1, 2, 4, 5) padded by one on the last two: two leading steps, four rows.

    The leading 1x2 is a single level because neither axis is padded, so its
    output pitch is the same for both -- which is the whole reason the gate can
    take a rank-4 pad with a three-level region.
    """
    assert _region_for((1, 2, 4, 5), (1, 1, 1, 1)) == [
        0,
        0,
        8,
        2,
        4,
        5,
        20,
        5,
        1,
        42,
        7,
        1,
    ]


@pytest.mark.parametrize(
    "label,shape,pads,value",
    [
        ("a nonzero value", (4, 6), (1, 1), 1.0),
        ("a negative pad", (4, 6), (-1, -1), None),
        ("a pad of nothing", (4, 6), (0, 0), None),
        ("a pad on a third axis from the end", (1, 2, 3, 4), (0, 0, 0, 0, 1, 1), None),
        ("a pad on all three of three axes", (2, 3, 4), (1, 1, 1, 1, 1, 1), None),
    ],
)
def test_the_shapes_the_gate_refuses_have_no_region(label, shape, pads, value):
    assert _region_for(shape, pads, value) is None, label


# --------------------------------------------------------------------------
# The commands, and what they compute.
# --------------------------------------------------------------------------


def test_a_pad_is_a_memset_then_one_blit():
    x = torch.randn(4, 6, dtype=F16)
    blob = _blob(_program(_Pad((1, 1)), (x,)))
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [_ZERO, _RASTER_BLIT]
    # The memset covers the whole result, which is what writes the border.
    assert list(commands[0].params[:1]) == [4 * 8 * 2]
    assert list(commands[1].params) == [
        1,
        2,
        1,
        0,
        0,
        1,
        1,
        4,
        6,
        24,
        6,
        1,
        32,
        8,
        1,
    ]


@pytest.mark.parametrize(
    "label,shape,pads,value",
    [
        ("last axis, symmetric", (4, 6), (1, 1), None),
        ("last axis, lopsided", (4, 6), (3, 0), None),
        ("last axis, zero on one side", (4, 6), (0, 4), None),
        ("both of the last two axes", (4, 6), (1, 2, 3, 1), None),
        ("rank one", (6,), (2, 1), None),
        ("rank three, last two axes", (2, 3, 4), (1, 1, 1, 0), None),
        ("rank four, last two axes", (1, 2, 4, 5), (1, 1, 1, 1), None),
        ("an explicit zero value", (4, 6), (1, 1), 0.0),
    ],
)
def test_the_padded_operand_matches_torch_bit_for_bit(label, shape, pads, value):
    """A copy and a memset are exact, so the comparison is equality, not a band.

    Both commands move bytes without arithmetic, so there is no rounding to
    allow for and a tolerance would hide a region that reads the wrong elements.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=F16)
    model = _Pad(pads, value)
    expected = model(x)
    got = _run(_blob(_program(model, (x,))), (x,), tuple(expected.shape))
    assert np.array_equal(got, expected.numpy()), label


def test_a_pad_of_a_delegated_intermediate_is_still_one_delegate():
    """relu runs on the DSP, so the pad's operand is a command's own output."""
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=F16)
    model = _ReluThenPad((2, 3))
    blob = _blob(_program(model, (x,)))
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [_UNARY, _ZERO, _RASTER_BLIT]
    assert np.array_equal(_run(blob, (x,), tuple(model(x).shape)), model(x).numpy())


# --------------------------------------------------------------------------
# Teeth: the region is what decides the answer, so move it.
# --------------------------------------------------------------------------


def test_moving_the_destination_offset_moves_the_answer():
    """The control that says the region test above is not vacuous.

    The same blob with its destination offset set to zero puts the operand's
    first element at the result's first element instead of one past it, which
    overwrites the left border and leaves the last element of every row clear.
    If the recorded region were not the one the interpreter walked, this test
    would come back equal and the equality above would be saying nothing.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=F16)
    model = _Pad((1, 1))
    blob = _blob(_program(model, (x,)))
    _, commands = read_blob(blob)
    assert commands[1].type == _RASTER_BLIT
    assert commands[1].params[_DST_OFFSET_PARAM] == 1

    data = bytearray(blob)
    at = B.HEADER_SIZE + B.OP_SIZE + _PARAMS_AT + 4 * _DST_OFFSET_PARAM
    struct.pack_into("<i", data, at, 0)
    assert not np.array_equal(
        _run(bytes(data), (x,), tuple(model(x).shape)), model(x).numpy()
    )


def test_the_closed_pad_is_a_delegate_where_it_was_none():
    """The positive half, through a real lowering: no delegate before, one now."""
    x = torch.randn(4, 6, dtype=F16)
    assert len(_delegates(_program(_Pad((1, 1)), (x,)))) == 1


@pytest.mark.parametrize(
    "label,shape,pads,value",
    [
        ("a nonzero value", (4, 6), (1, 1), 1.0),
        ("a negative pad", (4, 6), (-1, -1), None),
        ("a pad of nothing", (4, 6), (0, 0), None),
        ("a pad on a third axis from the end", (1, 2, 3, 4), (0, 0, 0, 0, 1, 1), None),
        ("a pad on all three of three axes", (2, 3, 4), (1, 1, 1, 1, 1, 1), None),
    ],
)
def test_the_gate_keeps_the_refused_shapes_off_the_dsp(label, shape, pads, value):
    """Refused, not unwired: the node is in the graph and the predicate is what
    turned it away, so the symptom is a portable kernel rather than a cell that
    looks supported and never runs."""
    x = torch.randn(*shape, dtype=F16)
    model = _Pad(pads, value)
    assert any("constant_pad_nd" in target for target in _edge_targets(model, (x,)))
    assert _delegates(_program(model, (x,))) == []


# --------------------------------------------------------------------------
# What does not reach this op at all.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["reflect", "replicate"])
def test_the_other_modes_are_a_program_rather_than_a_pad_node(mode):
    """`mode` is not an argument of `aten.constant_pad_nd`, and the other modes
    do not arrive as the `*_padNd` op a reader would look for: torch lowers them
    to a gather over an index built from arange. So there is no emitter to write
    for them here -- the pieces are separately on the portable kernels -- and the
    reason belongs on the record rather than in a TODO."""
    x = torch.randn(4, 6, dtype=F16)
    targets = _edge_targets(_Pad((1, 1), mode=mode), (x,))
    if mode == "replicate":
        assert any("aten.index.Tensor" in target for target in targets)
    else:
        assert any("et_hexagon.reflect_pad" in target for target in targets)
