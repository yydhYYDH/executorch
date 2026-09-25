# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convolution over one axis, read from the 4-D spelling the DSP carries.

A graph that convolves a (B, C, T) run over its time axis with a (O, I, k)
weight asks for a three-dimensional tensor, and the two convolution kernels here
take a four-dimensional one. This file pins that the one-axis form is read as
(1, C, T, 1) with a (k, 1) window rather than refused, that the command it emits
is the command the 4-D spelling emits -- the same bytes for the same inputs --
and that every shape the reading does not cover stays on a portable kernel with
no command at all.

The identity the reading rests on is torch's own: conv1d(x, w, ...) over
(B, C, T) equals conv2d(x[:, :, :, None], w[:, :, :, None], ...).squeeze(2)
whenever the window is a single column, and the weight is (O, I, k, 1) in the
same order. Each test below therefore compares against a torch 4-D computation
of the same graph, not against a second spelling of the same claim.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    CONV_VTCM_BYTES,
    conv_spec,
    conv_vtcm_bytes,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_CONV_DEPTHWISE2D_FP16, DSP_OP_RASTER_BLIT, DSP_OP_IM2COL_CONVOLUTION_FP16
#: and DSP_OP_ZERO: the command numbers the blobs carry.
_DEPTHWISE = 2
_BLIT = 3
_IM2COL = 12
_ZERO = 24

#: One HVX vector of fp16, the channel count a block of the blocked layout
#: carries.
_PACK = 64


class _Conv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class _Conv1dTransposed(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel, padding=padding
        )

    def forward(self, x):
        return self.conv(x)


class _AsConv2d(torch.nn.Module):
    """The same convolution spelled in four dimensions.

    The weight is a view of the 3-D one, so the bytes the 4-D graph carries are
    the bytes the 3-D graph carries and the two blobs can be compared for
    equality rather than for agreement.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        y = torch.nn.functional.conv2d(
            x.unsqueeze(-1),
            self.conv.weight.unsqueeze(-1),
            self.conv.bias,
            stride=(self.conv.stride[0], 1),
            padding=(self.conv.padding[0], 0),
            dilation=(self.conv.dilation[0], 1),
            groups=self.conv.groups,
        )
        return y.squeeze(-1)


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _lower(module, inputs):
    return to_edge_transform_and_lower(
        export(module, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _blob(program):
    """The one delegate's blob bytes and its command stream."""
    calls = _delegates(program)
    assert len(calls) == 1, f"the model did not reach the delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    data = bytes(lowered._processed_bytes)
    _, commands = read_blob(data)
    return data, commands


def _run(data, inputs):
    return np.frombuffer(execute(data, inputs)[0], dtype=np.float16)


def _exact(shape, seed, low, high):
    """Integer-valued fp16, so every sum in these tests is exact.

    A convolution of integers that stay inside fp16's exact range adds up the
    same however the taps are ordered, which makes a bit-exact comparison
    against torch a statement about the layout rather than about rounding.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low, high, shape, generator=generator).half()


def _whole(model, seed):
    """Integer weights and bias, so every product in these tests is exact."""
    generator = torch.Generator().manual_seed(seed)
    conv = getattr(model, "conv", model)
    weight = conv.weight
    weight.data = torch.randint(-1, 2, weight.shape, generator=generator).to(
        weight.dtype
    )
    if conv.bias is not None:
        bias = conv.bias
        bias.data = torch.randint(-1, 2, bias.shape, generator=generator).to(bias.dtype)
    return model


def _types(commands):
    return [command.type for command in commands]


#: CAMPPlus's TDNN windows, each one a geometry torch.export spells as a 3-D
#: convolution. The identity and the channel counts are the model's; the time
#: length is cut down so the command is small enough to execute on the host
#: interpreter. (in, out, kernel, stride, padding, groups).
_TDNN = [
    (128, 128, 5, 2, 2, 1),
    (128, 128, 1, 1, 0, 1),
    (128, 32, 3, 1, 1, 1),
    (128, 128, 3, 1, 1, 1),
    (64, 64, 3, 1, 1, 1),
    (512, 128, 3, 1, 1, 1),
    (256, 256, 3, 1, 1, 1),
    (288, 288, 3, 1, 1, 1),
    (320, 320, 3, 1, 1, 1),
]


def test_a_one_axis_convolution_reaches_the_kernel_through_its_4_d_reading():
    """Three-dimensional in, three-dimensional weight, and the im2col command.

    The exporter leaves nn.Conv1d as a 3-D convolution, and the weight is
    3-D with it. Both are read here as the 4-D form the kernel takes -- a unit
    batch, the time axis as the height, a unit width, a weight with a unit
    column -- so the command is the general path's with a k x 1 window.
    """
    in_channels, out_channels, kernel, stride, padding = 128, 128, 5, 2, 2
    length = 24
    model = _Conv1d(
        in_channels, out_channels, kernel, stride=stride, padding=padding
    ).half()
    _whole(model, 40)
    x = _exact((1, in_channels, length), 0, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert _types(commands) == [_BLIT, _IM2COL, _BLIT]
    walk = commands[1]
    # The command's first eight params are pad_x, pad_y, dil_x, dil_y,
    # stride_x, stride_y, kernel_x, kernel_y. kernel_x is the 1 the reading
    # gives the width, and stride_x and dil_x are the 1s beside it; the
    # following six are in_w, in_h, out_w, out_h after the two reduction
    # numbers, so the extents are torch's own for the 3-D graph with a unit
    # width on both sides.
    out_length = (length + 2 * padding - kernel) // stride + 1
    assert list(walk.params[:8]) == [0, padding, 1, 1, 1, stride, 1, kernel]
    assert list(walk.params[10:14]) == [1, length, 1, out_length]
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


@pytest.mark.parametrize("in_channels,out_channels,kernel,stride,padding,groups", _TDNN)
def test_every_campplus_window_matches_its_4_d_spelling_bit_for_bit(
    in_channels, out_channels, kernel, stride, padding, groups
):
    """Every command the 3-D graph emits is the 4-D graph's own command.

    The control is the same weights and the same bytes through an unsqueeze and
    a squeeze. The 4-D graph writes the command's 4-D result and then blits the
    3-D view out of it, so it carries exactly one more command than the 3-D
    graph and the rest are identical command for command, parameter for
    parameter; if the reading changed the command in any way this would
    differ, and if the weight were laid out the wrong way round the
    interpreter's output would differ from torch's on both sides.
    """
    length = 40
    three = _Conv1d(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        groups=groups,
    ).half()
    _whole(three, 41)
    four = _AsConv2d(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        groups=groups,
    ).half()
    four.conv.weight.data = three.conv.weight.data.clone()
    if three.conv.bias is not None:
        four.conv.bias.data = three.conv.bias.data.clone()
    x = _exact((1, in_channels, length), 1, -2, 3)

    data3, commands3 = _blob(_lower(three, (x,)))
    data4, commands4 = _blob(_lower(four, (x,)))
    assert len(commands4) == len(commands3) + 1, "the 4-D view costs one blit"
    for mine, theirs in zip(commands3, commands4):
        assert mine.type == theirs.type
        assert list(mine.params) == list(theirs.params), (
            f"command {mine.type} differs between the 3-D and 4-D spellings"
        )
    assert commands4[-1].type == _BLIT, "the extra command is the view blit"
    expected = three(x).detach().numpy().reshape(-1)
    got3 = _run(data3, [x.numpy()])
    got4 = _run(data4, [x.numpy()])
    assert got3.tobytes() == expected.tobytes()
    assert got4.tobytes() == expected.tobytes()


def test_a_one_axis_convolution_without_a_bias_delegates():
    """No bias is the same command with a zero vector, not a refusal."""
    model = _Conv1d(64, 64, 3, padding=1, bias=False).half()
    _whole(model, 42)
    x = _exact((1, 64, 20), 2, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert _types(commands) == [_BLIT, _IM2COL, _BLIT]
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_one_axis_depthwise_convolution_delegates():
    """groups == in_channels == out_channels is the per-channel walk, in 3-D too."""
    model = _Conv1d(64, 64, 3, padding=1, groups=64).half()
    _whole(model, 43)
    x = _exact((1, 64, 20), 3, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert _types(commands) == [_BLIT, _DEPTHWISE, _BLIT]
    walk = commands[1]
    # batch, in_h, in_w, out_h, out_w, blocks, then kernel_y, kernel_x, the
    # stride, the padding, the dilation and the two activations: the reading's
    # width is 1 on both sides of the window and 1 along it.
    assert list(walk.params) == [1, 20, 1, 20, 1, 1, 3, 1, 1, 1, 1, 0, 1, 1, 0, 0]
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_dilated_one_axis_convolution_delegates():
    """Dilation applies along the height in the reading, and torch's number agrees."""
    model = _Conv1d(64, 64, 3, padding=2, dilation=2).half()
    _whole(model, 44)
    x = _exact((1, 64, 40), 4, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert _types(commands) == [_BLIT, _IM2COL, _BLIT]
    # pad_x, pad_y, dil_x, dil_y, stride_x, stride_y, kernel_x, kernel_y:
    # the dilation and the padding belong to the height, the window is 3 x 1
    # and the width carries torch's units.
    assert list(commands[1].params[:8]) == [0, 2, 1, 2, 1, 1, 1, 3]
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def _conv_node(args, source_shape, weight_shape, result_shape, target):
    """A convolution node with the values its predicate reads, built by hand."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(source_shape, dtype=torch.float16)
    weight = graph.get_attr("weight")
    weight.meta["val"] = torch.empty(weight_shape, dtype=torch.float16)
    node = graph.call_function(target, args=(source, weight, *args))
    node.meta["val"] = torch.empty(result_shape, dtype=torch.float16)
    return node


_CONVOLUTION = exir_ops.edge.aten.convolution.default


def test_conv_spec_reads_a_3_d_node_as_the_4_d_spelling_it_is():
    """The spec a 3-D node gets is the one the 4-D node with a unit column gets.

    This is the claim the emitter's packing rests on, checked on the node
    rather than through a lowering: kernel_x is 1, in_w and out_w are 1, and
    the flag the weight packer reads is set. The 4-D spelling of the same
    geometry is admitted with the flag off, so the two differ only in where
    they were read from.
    """
    # The convolution overload takes bias, stride, padding, dilation,
    # transposed, output_padding and groups after the weight, so a one-axis
    # geometry spells each of the pairs with a single entry.
    three = _conv_node(
        (None, [2], [2], [1], False, [0], 1),
        (1, 64, 20),
        (32, 64, 5),
        (1, 32, 10),
        _CONVOLUTION,
    )
    four = _conv_node(
        (None, [2, 1], [2, 0], [1, 1], False, [0, 0], 1),
        (1, 64, 20, 1),
        (32, 64, 5, 1),
        (1, 32, 10, 1),
        _CONVOLUTION,
    )
    read = conv_spec(three, lambda _n: True)
    assert read.conv1d is True
    assert (read.kernel_y, read.kernel_x) == (5, 1)
    assert (read.in_h, read.in_w, read.out_h, read.out_w) == (20, 1, 10, 1)
    assert (read.stride_y, read.stride_x) == (2, 1)
    assert (read.pad_y, read.pad_x) == (2, 0)
    assert (read.dilate_y, read.dilate_x) == (1, 1)
    direct = conv_spec(four, lambda _n: True)
    assert direct.conv1d is False
    assert direct._replace(conv1d=True) == read


def test_conv_spec_refuses_a_3_d_input_with_a_4_d_weight():
    """A mismatched rank is not the reading, it is a node the exporter cannot make.

    The reading is defined on three three-dimensional operands. A node whose
    input is 3-D and whose weight is 4-D is not a graph torch writes, and
    reading it as the degenerate form would be a guess, so it stays refused.
    """
    node = _conv_node(
        (None, [1], [1], [1], False, [0], 1),
        (1, 64, 20),
        (32, 64, 5, 3),
        (1, 32, 20),
        _CONVOLUTION,
    )
    assert conv_spec(node, lambda _n: True) is None


def test_conv_spec_refuses_a_transposed_one_axis_convolution():
    """The transposed form is the one 3-D case the reading does not take.

    A transposed convolution's weight is indexed the other way round and its
    output extent is a function of the stride and the output padding, and no
    identity between that and a single-column forward window has been measured
    here, so the node is refused rather than read.
    """
    node = _conv_node(
        (None, [1], [1], [1], True, [0], 1),
        (1, 64, 20),
        (64, 64, 3),
        (1, 64, 20),
        _CONVOLUTION,
    )
    assert conv_spec(node, lambda _n: True) is None


def test_a_transposed_one_axis_convolution_stays_on_a_portable_kernel():
    """The refusal above at the level the graph sees it, with no command."""
    model = _Conv1dTransposed(64, 64, 3, padding=1).half()
    _whole(model, 45)
    x = _exact((1, 64, 20), 46, -2, 3)
    assert _delegates(_lower(model, (x,))) == []


def test_a_one_axis_convolution_with_a_middle_group_count_stays_on_the_host():
    """Neither kernel walks a group count between 1 and the channel count.

    16 groups over 64 input channels is a grouped convolution; the reading does
    not change that, and a 4-D graph of the same groups is refused by the same
    clause, which is the control.
    """
    model = _Conv1d(64, 32, 3, padding=1, groups=16).half()
    _whole(model, 47)
    x = _exact((1, 64, 20), 48, -2, 3)
    assert _delegates(_lower(model, (x,))) == []
    four = torch.nn.Conv2d(64, 32, 3, padding=1, groups=16).half()
    _whole(four, 49)
    assert _delegates(_lower(four, (x.unsqueeze(-1),))) == []


def test_the_vtcm_gate_moves_with_the_window_and_not_with_the_spelling():
    """Over 4864 channels the gate is at 8 taps for a one-column window.

    The kernel's staging request is three tiles of kp = kernel_y * kernel_x *
    ceil(C / 32) elements, so a one-column window costs a fifth of a square
    one over the same channels: a 5-tap one-axis layer over 4864 channels fits
    where a 5x5 layer does not, and the widest one-axis window that still fits
    is 8 taps. Both sides of that boundary are exercised here through the
    partitioner, so the refusal below is the gate's arithmetic and not the
    reading.
    """
    assert conv_vtcm_bytes(8, 1, 4864) <= CONV_VTCM_BYTES
    assert conv_vtcm_bytes(9, 1, 4864) > CONV_VTCM_BYTES
    assert conv_vtcm_bytes(5, 5, 4864) > CONV_VTCM_BYTES

    inside = _Conv1d(4864, 32, 8, padding=3).half()
    _whole(inside, 51)
    x = _exact((1, 4864, 20), 50, -2, 3)
    assert len(_delegates(_lower(inside, (x,)))) == 1

    outside = _Conv1d(4864, 32, 9, padding=4).half()
    _whole(outside, 53)
    assert _delegates(_lower(outside, (x,))) == []

    square = torch.nn.Conv2d(4864, 32, 5, padding=2).half()
    _whole(square, 52)
    assert _delegates(_lower(square, (x.unsqueeze(-1),))) == []


def test_a_one_axis_convolution_over_a_run_time_weight_stays_on_the_host():
    """A 3-D weight the export cannot see has no bytes to rearrange either."""

    class _Runtime(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.nn.functional.conv1d(x, weight, bias, padding=1)

    weight = _exact((64, 64, 3), 53, -2, 3)
    bias = _exact((64,), 54, -2, 3)
    x = _exact((1, 64, 20), 55, -3, 4)
    assert _delegates(_lower(_Runtime().half(), (x, weight, bias))) == []


def test_the_3_d_reading_is_a_blit_and_not_a_copy():
    """The command reads the graph's own bytes; the unit dims cost nothing.

    The pack blit's input is the 3-D tensor the graph hands over and its output
    is one 64-channel block of the interleaved plane, so a leading unit dim and
    a trailing unit dim are not a reshaped copy of anything: the same bytes
    reach the kernel either way, and the command says so.
    """
    model = _Conv1d(64, 64, 3, padding=1).half()
    _whole(model, 56)
    length = 20
    x = _exact((1, 64, length), 57, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    pack = commands[0]
    # The blit's source is the graph's own (1, C, T) tensor: the reading adds
    # a unit width to the command, not a copy of the input to it. Its
    # destination is the same plane in the 64-lane blocked layout.
    assert pack.inputs[0].size == 64 * length * 2
    assert pack.outputs[0].size == 64 * length * 2
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_2_d_graph_whose_window_is_a_single_column_is_not_the_3_d_reading():
    """A (k, 1) window on a 4-D graph is a square-kernel graph, and stays 4-D.

    This is the reverse direction: a 4-D tensor with a unit width is not a
    3-D one, so the reading must not fire on it. The command's kernel_x is
    the 1 the graph asked for and the flag is off, which is what the spec test
    above pins; here it is pinned through the blob.
    """
    model = torch.nn.Conv2d(64, 32, (3, 1), padding=(1, 0)).half()
    _whole(model, 58)
    x = _exact((1, 64, 8, 1), 59, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert _types(commands) == [_BLIT, _IM2COL, _BLIT]
    assert list(commands[1].params[6:8]) == [1, 3]
    assert list(commands[1].params[10:14]) == [1, 8, 1, 8]
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_the_equal_above_is_a_comparison_that_can_fail():
    """Reverse control: the 4-D spelling's numbers move when the data moves.

    The blob-equality test compares two blobs the emitter built from the same
    weights and the same input, so it would pass on a pair of emitters that
    both read the input wrongly. Handing the 3-D blob the 4-D graph's own
    values and a tensor with one element changed has to move the answer, which
    is what says the comparison above is about the bytes that reached the
    kernel.
    """
    model = _Conv1d(64, 64, 3, padding=1).half()
    _whole(model, 60)
    x = _exact((1, 64, 20), 61, -2, 3)
    data, _ = _blob(_lower(model, (x,)))
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()
    moved = x.clone()
    moved[0, 0, 0] = x[0, 0, 0] + 1
    assert _run(data, [moved.numpy()]).tobytes() != expected.tobytes()


@pytest.mark.parametrize("channels", [63, 64, 65])
def test_the_vector_boundary_is_structural_for_a_one_axis_convolution(channels):
    """63 lanes and one, 64 lanes, and 64 lanes and one, all bit-exact.

    The unit's activation is a vector of 64 channels, so a channel count that
    is not a multiple of it leaves lanes the graph has no channels for, and
    the emitter clears them with DSP_OP_ZERO before the fill reads them. The
    boundary is therefore visible in the command stream: 63 and 65 carry a
    zero command and 64 does not, and all three agree with torch.
    """
    model = _Conv1d(channels, 32, 3, padding=1).half()
    _whole(model, 70 + channels)
    x = _exact((1, channels, 20), 200 + channels, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert _types(commands).count(_IM2COL) == 1
    assert (_ZERO in _types(commands)) == (channels % _PACK != 0)
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()
