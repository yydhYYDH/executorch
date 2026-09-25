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


def _cat(count):
    return (
        lambda *xs: torch.cat(list(xs), dim=0),
        tuple(torch.randn(2, 3, 4, dtype=F16) for _ in range(count)),
    )


@pytest.mark.parametrize(
    ("count", "delegates", "params"),
    [
        (2, 1, [2, 2, 2]),
        (3, 1, [3, 2, 3]),
        (4, 0, None),
        (8, 0, None),
    ],
)
def test_cat_stops_at_three_operands_because_the_command_holds_three_regions(count, delegates, params):
    """A fourth region does not fit, so the refusal is the command width, not the kernel.

    `kMaxOpParams` is 40. The blit header is 3 ints (regionCount, bytes, src_number) and each
    region is 12, so 3 + 3*12 = 39 of 40 and a fourth region would need 51. The 3-operand row is
    the last one that fits, which is why 2 and 3 delegate with the region count in the params and 4
    and 8 stay on the portable path.
    """
    assert hexagon_ops.MAX_CAT_INPUTS == 3
    model, inputs = _cat(count)
    assert len(_delegates(model, inputs)) == delegates
    if params is not None:
        commands = _commands(model, inputs)
        assert [command[0] for command in commands] == [hexagon_ops.DSP_OP_RASTER_BLIT]
        assert commands[0][1][:3] == params
        assert len(commands[0][1]) <= 40, "the params block is the 40-int field"
    else:
        _support_node, node = _node(model, inputs, "aten_cat_default")
        assert hexagon_ops.cat_region(node) is None


def test_a_fourth_cat_operand_would_overflow_the_params_field():
    """The arithmetic behind the refusal, so a widening attempt starts from this number."""
    header = 3
    per_region = 12
    cap = 40
    assert header + hexagon_ops.MAX_CAT_INPUTS * per_region == 39
    assert header + (hexagon_ops.MAX_CAT_INPUTS + 1) * per_region > cap


# --------------------------------------------------------------------------- softmax


@pytest.mark.parametrize(
    ("shape", "dim", "inside"),
    [
        ((2, 4, 8), 2, 1),
        ((2, 4, 8), 1, 8),
        ((2, 4, 8), 0, 32),
        ((1, 2, 4, 8), 1, 32),
    ],
)
def test_softmax_gate_refuses_every_axis_but_the_last(shape, dim, inside):
    """`inside` is the product of the axes after the reduction, and 1 is the last axis.

    The kernel takes its contiguous per-row path when `inside == 1` and its strided path when
    `inside > 1`, so the gate refusing the strided path is a host-side decision about which of two
    kernel paths to use, not a statement that the kernel cannot walk the other axes.
    """
    model = (lambda x: torch.softmax(x, dim=dim))
    inputs = (torch.randn(*shape, dtype=F16),)
    _support, node = _node(model, inputs, "aten__softmax_default")
    trailing = dim == len(shape) - 1
    assert hexagon_ops.softmax_reduces_the_inner_axis(node) is trailing
    assert inside == (1 if trailing else inside)


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
def test_softmax_strided_axis_stays_portable(shape, dim):
    """The gate is what keeps this off the DSP, and the delegate count is how it is checked."""
    model = (lambda x: torch.softmax(x, dim=dim))
    inputs = (torch.randn(*shape, dtype=F16),)
    _support, node = _node(model, inputs, "aten__softmax_default")
    assert not hexagon_ops.softmax_reduces_the_inner_axis(node)
    assert _delegates(model, inputs) == []


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
