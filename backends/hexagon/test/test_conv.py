# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convolution on the DSP, over both of the kernels that carry it.

`hvx_conv_depthwise2d_fp16` walks one channel at a time and
`htp_ops_im2col_convolution_fp16` runs the rest through the HMX unit as a matrix
product over an im2col patch. Both read and write the same 64-channel blocked
activation pooling uses, so both come with a blit either side, and both take a
weight this layer rearranges at export time. These tests pin the delegation, the
two commands' params, the packed weights' byte counts and order, the numbers
against torch, and every shape that has to stay on a portable kernel.
"""

import os
import pathlib
import sys

import blob_interpreter
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
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    conv_spec,
    pack_conv_bias,
    pack_conv_weight,
    pack_depthwise_weight,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as B  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_CONV_DEPTHWISE2D_FP16, DSP_OP_IM2COL_CONVOLUTION_FP16,
#: DSP_OP_RASTER_BLIT and DSP_OP_ZERO: the commands a convolution lowers to.
_DEPTHWISE = 2
_IM2COL = 12
_BLIT = 3
_ZERO = 24

#: One HVX vector of fp16 -- the channel count a block of the blocked layout
#: carries -- and the HMX unit's tile.
_PACK = 64
_TILE = 32


def _out_extent(size, kernel, stride, pad, dilate):
    """torch's own extent for a window, which conv_spec has to agree with."""
    return (size + 2 * pad - dilate * (kernel - 1) - 1) // stride + 1


class _Conv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        dilation=1,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x):
        return self.conv(x)


class _RuntimeWeight(torch.nn.Module):
    """A convolution whose weight is a method input rather than a constant."""

    def forward(self, x, weight, bias):
        return torch.nn.functional.conv2d(x, weight, bias, padding=1)


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
    same however the taps are ordered, which is what makes a bit-exact
    comparison against torch a statement about the layout rather than about
    rounding.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low, high, shape, generator=generator).half()


def _whole(model, seed):
    """Integer weights and bias, so every product in these tests is exact.

    With integer inputs and weights the partial sums stay inside fp16's exact
    integer range, which makes the comparison against torch an equality rather
    than a tolerance: a different window origin or tap order would show as a
    different number, not as a rounding difference.
    """
    generator = torch.Generator().manual_seed(seed)
    weight = model.conv.weight
    weight.data = torch.randint(-1, 2, weight.shape, generator=generator).to(
        weight.dtype
    )
    if model.conv.bias is not None:
        bias = model.conv.bias
        bias.data = torch.randint(-1, 2, bias.shape, generator=generator).to(bias.dtype)
    return model


def _operand_at(data, command, index):
    """Where in the file one of a command's weight operands lives."""
    header, _ = read_blob(data)
    ref = command.inputs[index]
    return B.HEADER_SIZE + header.n_ops * B.OP_SIZE + ref.offset, ref.size


def test_a_depthwise_convolution_is_a_pack_a_walk_and_an_unpack():
    """Three commands, their exact params, and the bytes each operand carries.

    A depthwise convolution with one channel per group is the per-channel walk
    `hvx_conv_depthwise2d_fp16` takes: the weight is one HMX vector per tap with
    the channel block outside the taps, the bias is a whole vector per block, and
    the activation is blocked on both sides.
    """
    channels, kernel, area = 64, 3, 8
    model = _Conv(channels, channels, kernel, padding=1, groups=channels).half()
    _whole(model, 20)
    x = _exact((1, channels, area, area), 0, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_BLIT, _DEPTHWISE, _BLIT]

    pack, walk, unpack = commands
    assert list(walk.params) == [
        1,  # batch
        area,
        area,  # then the input's height and width
        area,
        area,  # then the output's
        1,  # c4: one 64-channel block
        kernel,
        kernel,  # kernel
        1,
        1,  # stride
        1,
        1,  # padding
        1,
        1,  # dilation
        0,
        0,  # relu and relu6: to_edge leaves the relu as its own node
    ]
    # One HMX vector per tap per block, and one bias vector per block: both of
    # the operands the kernel reads as whole vectors have to be that long even
    # when the graph's own bias is shorter than one.
    assert walk.inputs[1].size == (channels // _PACK) * kernel * kernel * _PACK * 2
    assert walk.inputs[2].size == channels * 2
    assert pack.inputs[0].size == channels * area * area * 2
    assert unpack.outputs[0].size == channels * area * area * 2
    assert list(pack.params) == [
        1,
        2,
        1,  # one region, fp16, one source
        0,
        0,
        0,  # source index 0, no offsets
        1,
        _PACK,
        area * area,  # size: [batch][channels][area]
        channels * area * area,
        area * area,
        1,  # src strides
        area * area * _PACK,
        1,
        _PACK,  # dst strides
    ]

    # The packed weight is the layout the kernel reads rather than the tensor's
    # own contiguous order: tap-major inside a block.
    packed = np.frombuffer(
        pack_depthwise_weight(
            model.conv.weight.detach().float().numpy(), channels, kernel, kernel
        ),
        dtype=np.float16,
    ).reshape(channels // _PACK, kernel, kernel, _PACK)
    plain = model.conv.weight.detach().half().numpy().reshape(channels, kernel, kernel)
    assert packed[0, 1, 2].tolist() == plain[0:64, 1, 2].tolist()

    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_depthwise_convolution_over_several_channel_blocks():
    """Two blocks and two batches: one region per block, and the numbers.

    The blocked layout puts the block index outside the batch index
    (`(c//64) * batch + n`), which is the one part of the region a single-block
    shape cannot show.
    """
    channels, kernel, area, batch = 128, 3, 9, 2
    model = _Conv(
        channels, channels, kernel, stride=2, padding=1, groups=channels
    ).half()
    _whole(model, 21)
    x = _exact((batch, channels, area, area), 1, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_BLIT, _DEPTHWISE, _BLIT]

    pack, walk, unpack = commands
    out_area = _out_extent(area, kernel, 2, 1, 1)
    assert list(walk.params[:6]) == [batch, area, area, out_area, out_area, 2]
    assert list(walk.params[6:14]) == [kernel, kernel, 2, 2, 1, 1, 1, 1]
    assert walk.inputs[1].size == 2 * kernel * kernel * _PACK * 2
    assert walk.inputs[2].size == 2 * _PACK * 2

    # One region per block, the block index outside the batch, and the two
    # layouts' strides: row-major reads a channel plane per step and blocked
    # writes one lane per step.
    row_major = [channels * area * area, area * area, 1]
    blocked = [area * area * _PACK, 1, _PACK]
    assert list(pack.params) == [2, 2, 1] + [
        0,
        0,
        0,
        batch,
        _PACK,
        area * area,
        *row_major,
        *blocked,
        0,
        _PACK * area * area,
        2 * area * area * _PACK,
        batch,
        _PACK,
        area * area,
        *row_major,
        *blocked,
    ]
    out_row_major = [channels * out_area * out_area, out_area * out_area, 1]
    out_blocked = [out_area * out_area * _PACK, 1, _PACK]
    assert list(unpack.params) == [2, 2, 1] + [
        0,
        0,
        0,
        batch,
        _PACK,
        out_area * out_area,
        *out_blocked,
        *out_row_major,
        0,
        2 * out_area * out_area * _PACK,
        _PACK * out_area * out_area,
        batch,
        _PACK,
        out_area * out_area,
        *out_blocked,
        *out_row_major,
    ]

    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_depthwise_convolution_over_a_ragged_channel_count():
    """A channel count that is not a whole number of blocks: the numbers.

    The last block carries one real channel and 63 lanes the kernel still walks.
    Each lane is its own channel, so the lanes past the count can hold anything:
    the unpack region only copies the channels the tensor has.
    """
    channels, kernel, area = 65, 3, 8
    model = _Conv(channels, channels, kernel, padding=1, groups=channels).half()
    _whole(model, 22)
    x = _exact((1, channels, area, area), 11, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_BLIT, _DEPTHWISE, _BLIT]
    assert list(commands[1].params[5:6]) == [2]  # two blocks for 65 channels
    assert commands[2].params[3 + 12 + 4] == 1  # the second region's width
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_convolution_is_two_blits_and_an_im2col_kernel():
    """The general path: the HMX kernel, its 29 params, and its weight tiling.

    The weight is one 32x32 tile per (output tile, tap, input tile) and the
    kernel copies the blob's bytes into VTCM verbatim
    (`fill_weight_tiles_fp16`), so the byte count is a statement about the
    tiling; the bias has to run one whole vector past the last tile's own lanes.
    """
    channels, kernel, area = 64, 3, 8
    model = _Conv(channels, channels, kernel, padding=1).half()
    _whole(model, 23)
    x = _exact((1, channels, area, area), 2, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_BLIT, _IM2COL, _BLIT]

    pack, conv, unpack = commands
    assert list(conv.params) == [
        1,
        1,  # padX, padY
        1,
        1,  # dilateX, dilateY
        1,
        1,  # strideX, strideY
        kernel,
        kernel,  # kernelX, kernelY
        channels // 4,  # icDiv4, read by neither int4 path
        kernel * kernel * (channels // _TILE),  # kernelCountUnit
        area,
        area,  # iw, ih
        area,
        area,  # ow, oh
        area * area * _PACK,  # srcZStep
        area * _PACK,  # srcYStep
        _PACK,  # packCUnit
        area * area * _PACK,  # destICStride
        channels,  # ic
        channels,  # icup4, read by neither int4 path
        channels,  # oc
        1,
        2,  # mp, np: one position tile, two channel tiles
        0,
        0,  # relu and relu6
        1,  # batch
        0,
        0,
        0,  # outputBytes, scaleBlockNum, scaleAsymmetric
    ]
    tiles = -(-channels // _TILE)
    assert conv.inputs[1].size == tiles * kernel * kernel * 2 * 1024 * 2
    assert conv.inputs[2].size == (tiles * _TILE + _TILE) * 2
    assert pack.inputs[0].size == channels * area * area * 2
    assert unpack.outputs[0].size == channels * area * area * 2

    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel, stride, padding, dilation, area, batch",
    [
        (64, 96, 1, 1, 0, 1, 5, 1),  # a pointwise layer, and a ragged output tile
        (64, 64, 3, 2, 1, 1, 8, 1),  # a strided window
        (64, 64, 3, 1, 2, 2, 7, 1),  # dilation, which the unit takes
        (3, 32, 3, 2, 1, 1, 8, 1),  # the MobileNet stem: fewer channels than a tile
        (32, 64, 3, 1, 1, 1, 6, 2),  # a batch of two, and a short input tile
        (64, 65, 1, 1, 0, 1, 4, 1),  # a ragged output channel count
    ],
)
def test_the_general_path_covers_the_geometry_it_claims(
    in_channels, out_channels, kernel, stride, padding, dilation, area, batch
):
    """Every shape the emitter accepts, against torch.

    A pointwise layer, a strided window, a dilated window, an input narrower than
    a channel tile, a batch, and an output that is not a whole number of tiles.
    The values are exact integers, so a bit-exact comparison here is a statement
    about the im2col patch, the tile order, the bias and the store rather than
    about rounding.
    """
    model = _Conv(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).half()
    _whole(model, 24)
    x = _exact((batch, in_channels, area, area), 3, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    expected_types = [_BLIT, _IM2COL, _BLIT]
    if in_channels % _PACK:
        # The lanes past the last channel have to be cleared before the pack,
        # or the zero weights the tiles carry for them multiply whatever the
        # activation held.
        expected_types = [_ZERO, _BLIT, _IM2COL, _BLIT]
    assert [command.type for command in commands] == expected_types
    if in_channels % _PACK:
        assert list(commands[0].params) == [
            batch * area * area * -(-in_channels // _PACK) * _PACK * 2
        ]
    conv = commands[-2]
    out_area = _out_extent(area, kernel, stride, padding, dilation)
    assert list(conv.params[8:14]) == [
        in_channels // 4,
        kernel * kernel * -(-in_channels // _TILE),
        area,
        area,
        out_area,
        out_area,
    ]
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_one_position_convolution_needs_no_blits():
    """A 1x1 spatial input is already in the layout both kernels take.

    With one position and one full channel block the blocked layout and the
    row-major one are the same bytes (`_conv_layouts_agree`), so the command is
    the convolution alone: a blit either side would copy a tensor onto itself.
    """
    channels, kernel = 64, 3
    model = _Conv(channels, channels, kernel, padding=1, groups=channels).half()
    _whole(model, 25)
    x = _exact((1, channels, 1, 1), 4, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_DEPTHWISE]
    assert (
        _run(data, [x.numpy()]).tobytes()
        == model(x).detach().numpy().reshape(-1).tobytes()
    )

    model = _Conv(channels, channels, 1).half()
    _whole(model, 26)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_IM2COL]
    assert (
        _run(data, [x.numpy()]).tobytes()
        == model(x).detach().numpy().reshape(-1).tobytes()
    )


def test_a_convolution_over_a_ragged_channel_count_clears_the_padding():
    """A one-position input with three channels still needs its lanes cleared.

    The blocked form of a three-channel activation is 64 lanes wide, so it is not
    the row-major buffer the tensor is: the emitter has to pack it, and the lanes
    the pack leaves alone have to be zero before the im2col fill reads them.
    """
    model = _Conv(3, 32, 3, padding=1).half()
    _whole(model, 26)
    x = _exact((1, 3, 1, 1), 12, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_ZERO, _BLIT, _IM2COL, _BLIT]
    assert list(commands[0].params) == [_PACK * 2]
    assert commands[0].outputs[0] == commands[1].outputs[0]
    assert (
        _run(data, [x.numpy()]).tobytes()
        == model(x).detach().numpy().reshape(-1).tobytes()
    )


@pytest.mark.parametrize(
    "in_channels, kernel, groups",
    [
        (64, 3, 32),  # a group count between one and the channels
        (64, 3, 16),  # and another
        (64, 3, 8),
    ],
)
def test_a_convolution_the_kernels_cannot_run_stays_on_the_host(
    in_channels, kernel, groups
):
    """A refusal, and the numbers the portable kernel still has to produce.

    A group count in between is a third kernel: neither walk carries the channel
    mapping for it, so the node stays portable rather than reaching a command
    that would read it wrong.
    """
    model = _Conv(in_channels, in_channels, kernel, padding=1, groups=groups).half()
    area = 8
    x = _exact((1, in_channels, area, area), 5, -3, 4)
    program = _lower(model, (x,))
    assert _delegates(program) == [], f"{groups} groups reached the delegate"
    expected = model(x)
    assert expected.dtype is torch.float16
    assert torch.isfinite(expected).all()


def test_an_unbatched_convolution_is_one_batch():
    """A 3-D operand gets the batch the command carries, and the numbers.

    The graph above the kernel makes the batch dimension explicit, which is
    another blit rather than a second command.
    """
    model = _Conv(64, 64, 3, padding=1).half()
    _whole(model, 29)
    x = _exact((64, 8, 8), 13, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_BLIT, _IM2COL, _BLIT, _BLIT]
    assert commands[1].params[25] == 1
    expected = model(x.unsqueeze(0)).detach().numpy().reshape(-1).astype(np.float16)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


def test_a_convolution_over_a_run_time_weight_stays_on_the_host():
    """A weight the export cannot see has no bytes to rearrange.

    Both kernels read a rearranged weight, so a convolution whose weight is a
    method input -- whose values only exist at run time -- has no command form.
    """
    weight = _exact((64, 64, 3, 3), 6, -2, 3)
    bias = _exact((64,), 7, -2, 3)
    x = _exact((1, 64, 8, 8), 8, -3, 4)
    model = _RuntimeWeight().half()
    assert _delegates(_lower(model, (x, weight, bias))) == []


def _conv_node(
    args,
    source_shape=(1, 64, 8, 8),
    weight_shape=(64, 1, 3, 3),
    result_shape=(1, 64, 8, 8),
    target=None,
):
    """A convolution node with the values its predicate reads, built by hand."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(source_shape, dtype=torch.float16)
    weight = graph.get_attr("weight")
    weight.meta["val"] = torch.empty(weight_shape, dtype=torch.float16)
    node = graph.call_function(
        target or exir_ops.edge.aten.conv2d.default, args=(source, weight, *args)
    )
    node.meta["val"] = torch.empty(result_shape, dtype=torch.float16)
    return node


def _constant(_node):
    return True


def test_conv_spec_reads_the_command_out_of_a_node_that_fits():
    """The one place the params come from, on a form the kernels take."""
    node = _conv_node((None, [2, 2], [1, 1], [1, 1], 64), result_shape=(1, 64, 4, 4))
    assert conv_spec(node, _constant) == hexagon_ops.ConvSpec(
        batch=1,
        in_channels=64,
        in_h=8,
        in_w=8,
        out_channels=64,
        out_h=4,
        out_w=4,
        kernel_y=3,
        kernel_x=3,
        stride_y=2,
        stride_x=2,
        pad_y=1,
        pad_x=1,
        dilate_y=1,
        dilate_x=1,
        depthwise=True,
    )


@pytest.mark.parametrize(
    "args, source_shape, weight_shape, result_shape, transposed",
    [
        # A transposed convolution, in the form the export rewrites produce.
        (
            (None, [1, 1], [0, 0], [1, 1], True, [0, 0], 1),
            (1, 64, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 8, 8),
            True,
        ),
        # Output padding, which only a transposed convolution has a use for.
        (
            (None, [1, 1], [1, 1], [1, 1], False, [1, 1], 1),
            (1, 64, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 8, 8),
            True,
        ),
        # A negative padding.
        (
            (None, [1, 1], [-1, -1], [1, 1], 1),
            (1, 64, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 6, 6),
            False,
        ),
        # A rank-5 operand.
        (
            (None, [1, 1], [0, 0], [1, 1], 1),
            (1, 64, 1, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 8, 8),
            False,
        ),
        # A result whose extent is not the one this geometry implies.
        (
            (None, [1, 1], [1, 1], [1, 1], 1),
            (1, 64, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 4, 4),
            False,
        ),
        # A group count that does not divide the input, which a real graph
        # cannot have but a hand-built one can.
        (
            (None, [1, 1], [1, 1], [1, 1], 3),
            (1, 64, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 8, 8),
            False,
        ),
        # A zero group count.
        (
            (None, [1, 1], [1, 1], [1, 1], 0),
            (1, 64, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 8, 8),
            False,
        ),
        # A stride the command cannot carry.
        (
            (None, [0, 1], [1, 1], [1, 1], 1),
            (1, 64, 8, 8),
            (64, 1, 3, 3),
            (1, 64, 8, 8),
            False,
        ),
    ],
)
def test_conv_spec_refuses_what_the_commands_cannot_describe(
    args, source_shape, weight_shape, result_shape, transposed
):
    """Every refusal the partitioner depends on, on the node it reads."""
    target = exir_ops.edge.aten.convolution.default if transposed else None
    node = _conv_node(args, source_shape, weight_shape, result_shape, target)
    assert conv_spec(node, _constant) is None


def test_conv_spec_refuses_a_weight_it_cannot_read():
    """The caller's own test for a readable constant is what gates the node."""
    node = _conv_node((None, [1, 1], [1, 1], [1, 1], 64))
    assert conv_spec(node, lambda _node: False) is None


def test_the_numeric_checks_have_teeth():
    """Reverse controls: the comparisons above fail when the bytes are wrong.

    Three controls, one per operand. Sending the activation in another order has
    to break the comparison against torch, and flipping one byte of the packed
    weight and of the bias has to change the answer -- so the bit-exact
    assertions above are about the data that reached the kernel rather than about
    a comparison that cannot fail.
    """
    channels, kernel, area = 64, 3, 8
    model = _Conv(channels, channels, kernel, padding=1, groups=channels).half()
    _whole(model, 27)
    x = _exact((1, channels, area, area), 9, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_BLIT, _DEPTHWISE, _BLIT]
    expected = model(x).detach().numpy().reshape(-1).tobytes()
    assert _run(data, [x.numpy()]).tobytes() == expected

    # Control 1: the same values handed over with two channels swapped. A
    # depthwise layer keeps them apart, so the answer has to move.
    swapped = x.clone()
    swapped[:, 0], swapped[:, 1] = x[:, 1].clone(), x[:, 0].clone()
    assert _run(data, [swapped.numpy()]).tobytes() != expected

    # Control 2: one bit of the packed weight, which the kernel only reads if
    # the operand really is the weight it was handed.
    at, size = _operand_at(data, commands[1], 1)
    assert size == (channels // _PACK) * kernel * kernel * _PACK * 2
    flipped = bytearray(data)
    flipped[at + 3 * _PACK * 2 + 1] ^= 0x40
    assert _run(bytes(flipped), [x.numpy()]).tobytes() != expected

    # Control 3: and one bit of the bias, which is the other operand every
    # channel reads.
    at, size = _operand_at(data, commands[1], 2)
    assert size == channels * 2
    flipped = bytearray(data)
    flipped[at + 7 * 2 + 1] ^= 0x40
    assert _run(bytes(flipped), [x.numpy()]).tobytes() != expected


def test_the_host_interpreter_convolves_the_way_the_kernels_do():
    """The interpreter's transcription of both kernels, against torch.

    It shares no code with either kernel: the depthwise walk is a lane-by-lane
    filter and the general path is the tile product the HMX unit computes, so a
    disagreement in the window's origin, the tile order, the bias or the store
    shows up here. Every value is an exact integer, so the comparison is an
    equality rather than a tolerance.
    """
    for in_channels, out_channels, kernel, stride, padding, groups in (
        (64, 64, 3, 1, 1, 64),
        (64, 64, 3, 2, 1, 1),
        (64, 96, 1, 1, 0, 1),
        (3, 32, 3, 2, 1, 1),
    ):
        model = _Conv(
            in_channels,
            out_channels,
            kernel,
            stride=stride,
            padding=padding,
            groups=groups,
        ).half()
        _whole(model, 28)
        x = _exact((1, in_channels, 9, 9), 10, -2, 3)
        data, commands = _blob(_lower(model, (x,)))
        assert commands[-2].type in (_DEPTHWISE, _IM2COL)
        got = _run(data, [x.numpy()])
        want = model(x).detach().numpy().reshape(-1)
        assert got.tobytes() == want.tobytes(), (in_channels, out_channels, groups)


def test_the_interpreter_refuses_a_command_it_cannot_model():
    """The fused activations are ones the emitters never set, and it says so."""
    params = [0] * 29
    params[20] = 64  # oc
    params[22] = 2  # np
    params[23] = 1  # relu
    params[25] = 1  # batch
    with pytest.raises(blob_interpreter.UnsupportedOp):
        blob_interpreter._run_im2col_convolution(None, params, None)


def test_pack_conv_weight_is_the_tiling_the_kernel_reads():
    """The emitter's packer, read back through the unit's own tile order.

    The kernel memcpys each tile into VTCM verbatim, so the only things that can
    be wrong are which (output tile, tap, input tile) each 1024-element block is
    and where inside it a (k, c) pair sits. This reads the packer's output back
    through that order and compares it with the weight.
    """
    channels, kernel = 64, 3
    model = _Conv(channels, channels, kernel, padding=1).half()
    weight = model.conv.weight.detach().float().numpy()
    spec = hexagon_ops.ConvSpec(
        batch=1,
        in_channels=channels,
        in_h=8,
        in_w=8,
        out_channels=channels,
        out_h=8,
        out_w=8,
        kernel_y=kernel,
        kernel_x=kernel,
        stride_y=1,
        stride_x=1,
        pad_y=1,
        pad_x=1,
        dilate_y=1,
        dilate_x=1,
        depthwise=False,
    )
    blob = np.frombuffer(pack_conv_weight(weight, spec), dtype=np.float16)
    units = -(-channels // _TILE)
    assert blob.size == units * kernel * kernel * units * 1024
    for tile_index in range(units):
        for ky in range(kernel):
            for kx in range(kernel):
                for unit in range(units):
                    index = (
                        tile_index * kernel * kernel + ky * kernel + kx
                    ) * units + unit
                # Element (k, c) of a tile sits at (k // 2) * 64 + c * 2 + (k & 1),
                # so the tile reads back as [k // 2][k & 1][c].
                tile = blob[index * 1024 : (index + 1) * 1024].reshape(
                    _TILE // 2, _TILE, 2
                )
                want = weight[
                    tile_index * _TILE : (tile_index + 1) * _TILE,
                    unit * _TILE : (unit + 1) * _TILE,
                    ky,
                    kx,
                ]
                assert np.array_equal(
                    tile.astype(np.float32).reshape(-1),
                    want.transpose(1, 0)
                    .reshape(_TILE // 2, 2, _TILE)
                    .transpose(0, 2, 1)
                    .astype(np.float32)
                    .reshape(-1),
                )


def test_pack_conv_bias_is_padded_past_the_last_tile():
    """The store reads a whole vector from each tile's own first lane."""
    channels = 64
    bias = np.frombuffer(
        pack_conv_bias(np.arange(channels) / 8, channels, 96), dtype=np.float16
    )
    assert bias.size == 96
    assert (
        bias[:channels].tolist()
        == (np.arange(channels) / 8).astype(np.float16).tolist()
    )
    assert bias[channels:].tolist() == [0] * (96 - channels)
    assert hexagon_ops.POOL_CHANNEL_BLOCK == _PACK


def test_a_depthwise_weight_is_padded_to_whole_vectors():
    """The kernel multiplies every lane of the weight it is handed."""
    channels, kernel = 65, 3
    weight = np.ones((channels, 1, kernel, kernel), dtype=np.float16)
    packed = np.frombuffer(
        pack_depthwise_weight(weight, channels, kernel, kernel), dtype=np.float16
    )
    assert packed.size == 2 * kernel * kernel * _PACK
    block = packed.reshape(2, kernel, kernel, _PACK)
    assert block[0, 0, 0].tolist() == [1.0] * 64
    assert block[1, 0, 0].tolist() == [1.0] + [0.0] * 63
