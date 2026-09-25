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



import blob_interpreter
import numpy as np
import pytest
import torch


from blob_interpreter import execute, read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_ops import (
    conv_1x1_direct_applies,
    conv_spec,
    CONV_VTCM_BYTES,
    conv_vtcm_bytes,
    CONV_VTCM_FIXED_BYTES,
    CONV_VTCM_STAGING_TILES,
    pack_conv_bias,
    pack_conv_weight,
    pack_depthwise_weight,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as B
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

#: DSP_OP_CONV_DEPTHWISE2D_FP16, DSP_OP_IM2COL_CONVOLUTION_FP16,
#: DSP_OP_RASTER_BLIT and DSP_OP_ZERO: the commands a convolution lowers to.
_DEPTHWISE = 2
_IM2COL = 12
#: DSP_OP_CONV1X1_DIRECT_FP16. It resolves to the same C function as _IM2COL
#: (htp_ops_conv1x1_direct_fp16 forwards to hmx_im2col_convolution_fp16), so
#: what the command type records is which of the function's own activation
#: fills is eligible: the 1x1 direct copy or gather, rather than the general
#: per-position window walk.
_CONV1X1 = 17
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
    # A 1x1 layer with no padding, unit dilation, batch 1 and a whole number of
    # reduction blocks is the geometry the function's own 1x1 fill takes, so the
    # command names it (17); every other window in this list walks the general
    # path (12).
    expected_types = [_BLIT, _IM2COL, _BLIT]
    if (
        kernel == 1
        and batch == 1
        and padding == 0
        and dilation == 1
        and in_channels % 32 == 0
    ):
        expected_types = [_BLIT, _CONV1X1, _BLIT]
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
    assert [command.type for command in commands] == [_CONV1X1]
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
def test_a_grouped_convolution_reaches_the_dsp_as_one_walk_per_group(
    in_channels, kernel, groups
):
    """A group count in between is no longer a refusal, and no longer one walk.

    Neither kernel carries a channel mapping, so the host supplies it: the group
    count is emitted as one dense im2col command per group. These shapes moved
    from the host to the DSP with the per-group partition, which is what the
    older version of this test asserted as a refusal.
    """
    model = _Conv(in_channels, in_channels, kernel, padding=1, groups=groups).half()
    _whole(model, 5)
    area = 8
    x = _exact((1, in_channels, area, area), 5, -3, 4)
    data, commands = _blob(_lower(model, (x,)))
    walks = [command for command in commands if command.type == _IM2COL]
    assert len(walks) == groups, f"{groups} groups, {len(walks)} walks"
    assert len(walks) > 1, "a group count in between is not one walk"
    expected = model(x).detach().numpy().reshape(-1)
    assert _run(data, [x.numpy()]).tobytes() == expected.tobytes()


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
        groups=64,
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
        # A valid intermediate group count was one of these and no longer is: the
        # host supplies the channel mapping by emitting one dense walk per group,
        # so test_a_grouped_convolution_reaches_the_dsp_as_one_walk_per_group below
        # asserts it delegates, and the one grouped case still refused -- a run-time
        # height, which has no per-group patch -- is pinned in test_grouped_conv.py.
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
        # A reduction so wide the kernel's staging buffers outgrow VTCM: 4864
        # channels over a 3x3 window is kp = 1368, and the kernel asks for three
        # kp-sized staging tiles before it runs anything.
        (
            (None, [1, 1], [1, 1], [1, 1], 1),
            (1, 4864, 4, 4),
            (32, 4864, 3, 3),
            (1, 32, 4, 4),
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


def test_the_vtcm_gate_refuses_one_step_over_what_the_kernel_asks_for():
    """The gate is arithmetic on the kernel's own allocation, not a hunch.

    The buffer sizes are in im2col_convolution_fp16.cc:1783-1786 and the tiles
    are the mp = 1, np = 2 the emitter pins, so the widest reduction that fits is
    the one this computes -- 4832 channels over a 3x3 window, and one 32-channel
    k-slice more does not.
    """
    assert CONV_VTCM_STAGING_TILES == 3, "mp = 1 and np = 2 are what is emitted"
    assert (CONV_VTCM_BYTES - CONV_VTCM_FIXED_BYTES) // 6144 == 1364, "the bound moved"
    fitting = _conv_node(
        (None, [1, 1], [1, 1], [1, 1], 1),
        (1, 4832, 4, 4),
        (32, 4832, 3, 3),
        (1, 32, 4, 4),
    )
    assert conv_spec(fitting, _constant) is not None, "refused one step inside the gate"
    assert (
        conv_vtcm_bytes(3, 3, 4832) <= CONV_VTCM_BYTES
    ), "the accepted side does not fit"
    assert conv_vtcm_bytes(3, 3, 4864) > CONV_VTCM_BYTES, "the refused side does fit"


def test_a_convolution_over_the_vtcm_gate_is_left_on_the_host():
    """The refusal at the level the graph sees it, on the smallest such window.

    4864 channels over a 3x3 window is one 32-channel k-slice past the widest
    reduction that fits, and the exporter keeps it rather than emitting a command
    whose staging buffers do not exist. 4832 is the same graph one slice inside
    the gate, and that one is delegated.
    """
    fits = _Conv(4832, 32, 3, padding=1).half()
    assert len(_delegates(_lower(fits, (_exact((1, 4832, 4, 4), 3, -1, 1),)))) == 1
    over = _Conv(4864, 32, 3, padding=1).half()
    assert _delegates(_lower(over, (_exact((1, 4864, 4, 4), 3, -1, 1),))) == []


def test_a_wide_tensor_moves_its_channel_blocks_in_several_commands():
    """A command carries at most three 64-channel blocks, and no more than forty ints.

    The parameter block is a fixed 40 ints and a region takes twelve of them, so
    448 channels -- seven blocks -- take three commands and the last of them
    carries one. Each command's count is its own chunk's, and the chunks add up to
    the blocks the tensor has, which is what the kernel reads.
    """
    model = _Conv(448, 32, 3, padding=1).half()
    _whole(model, 25)
    data, commands = _blob(_lower(model, (_exact((1, 448, 4, 4), 3, -1, 1),)))
    assert [command.type for command in commands] == [
        _BLIT,
        _BLIT,
        _BLIT,
        _IM2COL,
        _BLIT,
    ]
    assert [command.params[0] for command in commands if command.type == _BLIT] == [
        3,
        3,
        1,
        1,
    ]
    assert all(len(command.params) <= 40 for command in commands)


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
        assert commands[-2].type in (_DEPTHWISE, _IM2COL, _CONV1X1)
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


# DSP_OP_CONV1X1_DIRECT_FP16 names the 1x1 fill. htp_ops_conv1x1_direct_fp16
# forwards to hmx_im2col_convolution_fp16 (im2col_convolution_fp16.cc:1840), so
# the op type does not pick a kernel; it records that the function's own
# activation fill has a 1x1 path this geometry takes, so the stream says which
# fill runs rather than naming 12 for a pointwise layer that never walks a
# window.


def _conv1x1_op(model, x):
    data, commands = _blob(_lower(model, (x,)))
    types = [command.type for command in commands]
    assert types.count(_CONV1X1) + types.count(_IM2COL) == 1, types
    return data, types


@pytest.mark.parametrize(
    "in_channels, out_channels, stride, area",
    [
        (64, 64, 1, 5),  # the common pointwise layer, unit stride
        (64, 96, 1, 5),  # a ragged output tile, which the fill handles
        (128, 64, 1, 4),  # more than one 32-channel reduction block
        (32, 32, 1, 1),  # a single reduction block and a single pixel
        (64, 64, 2, 8),  # a strided pointwise layer: the gather, not the copy
        (96, 64, 3, 7),  # strided on both axes, batch 1
    ],
)
def test_a_1x1_convolution_the_fill_can_copy_is_emitted_as_17(
    in_channels, out_channels, stride, area
):
    """A 1x1 layer takes the direct fill, so the command is 17 and not 12.

    The kernel's 1x1 fill needs a whole number of 32-channel reduction blocks
    (kp == ceil(ic / 32), :1593), unit dilation, no padding, batch 1, and
    either unit stride with the output plane equal to the input plane or a
    stride above one. The interpreter's product is the same for 12 and 17, so
    the numbers are compared against torch to show the command is not merely a
    different number for the same bytes.
    """
    model = _Conv(in_channels, out_channels, 1, stride=stride).half()
    _whole(model, 31)
    x = _exact((1, in_channels, area, area), 12, -2, 3)
    data, types = _conv1x1_op(model, x)
    assert types[-2] == _CONV1X1, types
    assert _IM2COL not in types, types
    got = _run(data, [x.numpy()])
    want = model(x).detach().numpy().reshape(-1)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize(
    "label, in_channels, out_channels, kernel, stride, padding, dilation, area, batch",
    [
        ("a batch of two", 64, 64, 1, 1, 0, 1, 4, 2),
        ("padding", 64, 64, 1, 1, 1, 1, 5, 1),
        ("dilation", 64, 64, 1, 1, 0, 2, 6, 1),
        ("a ragged reduction", 48, 64, 1, 1, 0, 1, 4, 1),
        ("a 3x3 window", 64, 64, 3, 1, 1, 1, 6, 1),
        ("depthwise", 64, 64, 1, 1, 0, 1, 4, 1),
    ],
)
def test_a_1x1_geometry_the_direct_fill_cannot_take_stays_im2col(
    label, in_channels, out_channels, kernel, stride, padding, dilation, area, batch
):
    """The negative test: outside the restrictions the command stays 12.

    Each row is a case where the C's 1x1 fill selection does not fire, so
    fill_im2col_activation_1x1_pack64_tiles is never reached and the general
    per-position window walk is what runs. Emitting 17 there would put a fast
    path in the stream that the DSP does not take.
    """
    model = _Conv(
        in_channels,
        out_channels,
        kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=in_channels if label == "depthwise" else 1,
    ).half()
    _whole(model, 33)
    x = _exact((batch, in_channels, area, area), 14, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    types = [command.type for command in commands]
    assert _CONV1X1 not in types, (label, types)
    if label == "depthwise":
        assert _DEPTHWISE in types, (label, types)
    else:
        assert types.count(_IM2COL) == 1, (label, types)
    got = _run(data, [x.numpy()])
    want = model(x).detach().numpy().reshape(-1)
    assert got.tobytes() == want.tobytes(), label


def test_the_predicate_is_the_kernel_selection_and_nothing_else():
    """The predicate reads the same fields the C does, on the spec it builds.

    This is the transcription of fill_im2col_activation_tiles (:1593) and
    fill_im2col_activation_1x1_pack64_tiles (:723): a 1x1 window, unit
    dilation, no padding, a whole number of 32-channel reduction blocks, batch
    1, and either unit stride with the output plane equal to the input plane
    or a stride above one. Each row is one field moved off its value.
    """
    base = hexagon_ops.ConvSpec(
        batch=1,
        in_channels=64,
        in_h=4,
        in_w=4,
        out_channels=64,
        out_h=4,
        out_w=4,
        kernel_y=1,
        kernel_x=1,
        stride_y=1,
        stride_x=1,
        pad_y=0,
        pad_x=0,
        dilate_y=1,
        dilate_x=1,
        depthwise=False,
    )
    assert conv_1x1_direct_applies(base)
    assert not conv_1x1_direct_applies(base._replace(batch=2))
    assert not conv_1x1_direct_applies(base._replace(in_channels=48))
    assert not conv_1x1_direct_applies(base._replace(kernel_x=3))
    assert not conv_1x1_direct_applies(base._replace(kernel_y=3))
    assert not conv_1x1_direct_applies(base._replace(pad_x=1))
    assert not conv_1x1_direct_applies(base._replace(dilate_x=2))
    assert not conv_1x1_direct_applies(base._replace(depthwise=True))
    assert not conv_1x1_direct_applies(base._replace(transposed=True))
    assert not conv_1x1_direct_applies(base._replace(out_h=3))
    strided = base._replace(stride_x=2, stride_y=2, out_h=2, out_w=2)
    assert conv_1x1_direct_applies(strided)
    assert not conv_1x1_direct_applies(strided._replace(batch=2))
    assert not conv_1x1_direct_applies(strided._replace(pad_x=1))


def test_the_1x1_command_is_absent_from_a_non_1x1_graph():
    """A 3x3 convolution's stream has no 17 in it at all.

    The direct command must be absent from a stream that does not take the
    direct fill, not merely accompanied by 12: if a graph on im2col carried a
    17 as well, the claim that 1x1 avoids the im2col materialisation would be
    uncheckable from the blob.
    """
    model = _Conv(64, 64, 3, padding=1).half()
    _whole(model, 35)
    x = _exact((1, 64, 6, 6), 16, -2, 3)
    data, commands = _blob(_lower(model, (x,)))
    assert [command.type for command in commands] == [_BLIT, _IM2COL, _BLIT]
    got = _run(data, [x.numpy()])
    want = model(x).detach().numpy().reshape(-1)
    assert got.tobytes() == want.tobytes()


def test_the_1x1_stream_counts_17_and_no_12_from_the_blob():
    """The command types, counted the way the runtime reads them.

    The header's op count says how many commands there are, not which; the
    types come out of read_blob's command list, which is the same bytes the
    DSP's dispatcher switches on.
    """
    model = _Conv(64, 96, 1).half()
    _whole(model, 37)
    x = _exact((1, 64, 5, 5), 18, -2, 3)
    data, types = _conv1x1_op(model, x)
    header, commands = read_blob(data)
    assert header.n_ops == len(commands) == len(types)
    assert [command.type for command in commands] == types
    assert types.count(_CONV1X1) == 1
    assert types.count(_IM2COL) == 0
    assert types.count(_BLIT) == 2
