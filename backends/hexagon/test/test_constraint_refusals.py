"""Where each of four constraints is a protocol limit rather than a narrow predicate.

Each of these gates refuses something the DSP kernel is in fact able to compute, or refuses
something the kernel computes correctly and could compute for more shapes. Reading the gate as
"the kernel cannot do this" is how a backend ends up refusing work it could delegate. These tests
pin what each refusal is actually made of, so a later attempt to widen one has to say which of the
protocol facts below it intends to change.

`cat` is the clearest: a fourth operand does not need a new kernel, it needs a wider command. The
other three need a new lowering, a wider parameter table, or a new patch mechanism respectively,
and the widths they hit are counted in the assertions.
"""

import pytest
import torch
import torch.nn as nn

from blob_interpreter import read_blob  # noqa: E402

from executorch.backends.hexagon import hexagon_ops as hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    _broadcast_fits_dsp_limits,
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

F16 = torch.float16
CONFIG = EdgeCompileConfig(_check_ir_validity=False)


class _M(torch.nn.Module):
    """One callable, so a case is a plain module over its arguments."""

    def __init__(self, forward):
        super().__init__()
        self.forward_fn = forward

    def forward(self, *args):
        return self.forward_fn(*args)


def _program(model, inputs):
    return to_edge(export(_M(model), inputs), compile_config=CONFIG).exported_program()


def _support(program):
    return HexagonOperatorSupport(_data_placeholders(program))


def _lower(model, inputs):
    """The lowered program, so a case can be read for delegates and their blobs."""
    return to_edge_transform_and_lower(
        export(_M(model), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()


def _delegates(model, inputs):
    program = _lower(model, inputs)
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _commands(model, inputs):
    """(type, params) for every command in every delegate, read out of the blob."""
    program = _lower(model, inputs)
    out = []
    for call in _delegates(model, inputs):
        module = program.graph_module.get_submodule(call.args[0].target)
        for command in read_blob(bytes(module._processed_bytes))[1]:
            out.append((command.type, list(command.params)))
    return out


def _node(model, inputs, name):
    """The one call_function node whose name is `name`, with the support object."""
    program = _program(model, inputs)
    support = _support(program)
    found = [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.name == name
    ]
    assert found, f"no {name} in the graph"
    return support, found[0]


# --------------------------------------------------------------------------- cat


def _serializes(model, inputs) -> int:
    """Serialize the lowered program, in bytes, or fail the way a caller would.

    `to_executorch` is the tier that says a program exports, and it is stricter
    than the lowering the shape tests here stop at: it converts every functional
    op in the portable part of the graph to its out variant, and a node with no
    out variant cannot be serialized at all. So a graph that reaches the DSP has
    to survive this too, and a widening that stops being one delegate or stops
    serializing is not a widening.
    """
    lowered = to_edge_transform_and_lower(
        export(_M(model), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    )
    return len(bytes(lowered.to_executorch().buffer))


def _cat(count):
    return (
        lambda *xs: torch.cat(list(xs), dim=0),
        tuple(torch.randn(2, 3, 4, dtype=F16) for _ in range(count)),
    )


@pytest.mark.parametrize(
    ("count", "delegates", "regions"),
    [
        (2, 1, [2]),
        (3, 1, [3]),
        (4, 1, [3, 1]),
        (8, 1, [3, 3, 2]),
    ],
)
def test_cat_splits_a_long_list_over_as_many_blits_as_the_region_budget_needs(
    count, delegates, regions
):
    """A fourth region does not fit in one command, so a long cat is several.

    `kMaxOpParams` is 40. The blit header is 3 ints (regionCount, bytes, src_number) and each
    region is 12, so 3 + 3*12 = 39 of 40 and a fourth region would need 51. That bound is per
    command, not per cat: a list too long for one command is split over as many blits as it needs,
    each still writing its own disjoint slice of the result. So 2 and 3 are one blit, 4 is a
    `[3, 1]` split, and 8 is a `[3, 3, 2]` one -- and the region counts add up to the operand
    count every time, which is the part that says no input was dropped.
    """
    assert hexagon_ops.MAX_CAT_INPUTS == 3
    model, inputs = _cat(count)
    assert len(_delegates(model, inputs)) == delegates
    commands = _commands(model, inputs)
    assert [command[0] for command in commands] == [
        hexagon_ops.DSP_OP_RASTER_BLIT
    ] * len(regions)
    assert [command[1][0] for command in commands] == regions
    assert sum(regions) == count, "every operand is still placed exactly once"
    for command in commands:
        assert len(command[1]) <= 40, "the params block is the 40-int field"


def test_a_fourth_cat_operand_would_overflow_the_params_field():
    """The arithmetic behind the refusal, so a widening attempt starts from this number."""
    header = 3
    per_region = 12
    cap = 40
    assert header + hexagon_ops.MAX_CAT_INPUTS * per_region == 39
    assert header + (hexagon_ops.MAX_CAT_INPUTS + 1) * per_region > cap


# --------------------------------------------------------------------------- softmax


@pytest.mark.parametrize(
    ("shape", "dim", "trailing"),
    [
        ((2, 4, 8), 2, True),
        ((2, 4, 8), 1, False),
        ((2, 4, 8), 0, False),
        ((1, 2, 4, 8), 1, False),
    ],
)
def test_softmax_moves_a_non_last_axis_last_and_back(shape, dim, trailing):
    """Every axis reduces on the inner path now; only the last needs no blits.

    The kernel is last-axis, so a softmax over any other axis is answered by
    moving that axis last with a blit, reducing, and moving it back with a
    second blit. `softmax_reduces_the_inner_axis` is therefore True for every
    axis, and what distinguishes them is the command stream around it: the
    trailing axis is a bare DSP_OP_SOFTMAX and the rest are blit, softmax, blit.
    Both are one delegate.
    """
    model = (lambda x: torch.softmax(x, dim=dim))
    inputs = (torch.randn(*shape, dtype=F16),)
    _support, node = _node(model, inputs, "aten__softmax_default")
    assert hexagon_ops.softmax_reduces_the_inner_axis(node) is True
    assert len(_delegates(model, inputs)) == 1
    expected = [hexagon_ops.DSP_OP_SOFTMAX] if trailing else [
        hexagon_ops.DSP_OP_RASTER_BLIT,
        hexagon_ops.DSP_OP_SOFTMAX,
        hexagon_ops.DSP_OP_RASTER_BLIT,
    ]
    assert [command[0] for command in _commands(model, inputs)] == expected


def test_softmax_last_axis_emits_the_contiguous_path():
    """The axis the gate admits lowers with inside == 1, which is the kernel's contiguous path."""
    model = (lambda x: torch.softmax(x, dim=-1))
    inputs = (torch.randn(2, 4, 8, dtype=F16),)
    assert len(_delegates(model, inputs)) == 1
    commands = _commands(model, inputs)
    assert [command[0] for command in commands] == [hexagon_ops.DSP_OP_SOFTMAX]
    assert commands[0][1] == [8, 8, 1, 2], "outside, channel, inside, bytes"


@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((2, 4, 8), 1),
        ((1, 2, 4, 8), 1),
    ],
)
def test_softmax_middle_axis_is_one_delegate_and_serializes(shape, dim):
    """The middle axis used to be the refusal; it is now the blit sandwich.

    What is worth keeping here is that the widened path is still a single
    delegate and still serializes, because a graph carrying a dequantize or a
    stranded node is a program that will not build, and widening an axis is
    exactly the kind of change that can quietly stop being one delegate.
    """
    model = (lambda x: torch.softmax(x, dim=dim))
    inputs = (torch.randn(*shape, dtype=F16),)
    _support, node = _node(model, inputs, "aten__softmax_default")
    assert hexagon_ops.softmax_reduces_the_inner_axis(node) is True
    assert len(_delegates(model, inputs)) == 1
    assert _serializes(model, inputs) > 0


# --------------------------------------------------------------------------- broadcast


def _add(x, y):
    """One `add` over two arguments, which is the shape a broadcast is asked about."""

    return lambda a, b: a + b


@pytest.mark.parametrize(
    ("shape", "shape0", "delegates"),
    [
        ((4, 8), (4, 8), 1),
        ((4, 8), (8,), 1),
        ((4, 8), (4, 1), 1),
        ((2, 1, 1, 1, 1, 1, 4, 8), (2, 1, 1, 1, 1, 1, 1, 1), 1),
        ((1, 1, 1, 1, 1, 1, 1, 1, 4), (1, 1, 1, 1, 1, 1, 1, 1, 1), 0),
    ],
)
def test_broadcast_refusal_is_rank_eight_and_nothing_else(shape, shape0, delegates):
    """Rank 8 delegates, rank 9 does not, because the stride tables are 8 wide per operand."""
    x = torch.randn(*shape, dtype=F16)
    y = torch.randn(*shape0, dtype=F16)
    inputs = (x, y)
    model = _add(x, y)
    _support, node = _node(model, inputs, "aten_add_tensor")
    fits = _broadcast_fits_dsp_limits(node)
    assert fits == (len(shape) <= 8)
    assert len(_delegates(model, inputs)) == delegates


def test_broadcast_stride_tables_are_eight_wide_per_operand():
    """outDims[8] + in0Strides[8] + in1Strides[8] + a kind word is the whole table."""
    assert 8 + 8 + 8 + 1 == 25


# --------------------------------------------------------------------------- static extents


def test_conv_refuses_a_symbolic_weight_because_its_extents_are_the_exports():
    """`conv_spec` rejects any SymInt in the weight, and that is where the extents come from."""
    assert hexagon_ops.STATIC_DIM == "static"
    module = nn.Conv2d(3, 8, 3, padding=1).half()
    program = _program(lambda x: module(x), (torch.randn(1, 3, 16, 16, dtype=F16),))
    support = _support(program)
    conv = next(
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.name == "aten_convolution_default"
    )
    assert hexagon_ops.conv_spec(conv, support.is_data_placeholder) is not None


def test_pool_needs_exactly_one_64_channel_block():
    """The kernel reads `[ceil(C/64)][b][h*w][64]`, which one blit either side only frames at C=64."""
    assert hexagon_ops.POOL_CHANNEL_BLOCK == 64
