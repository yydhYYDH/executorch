# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The convolution kernels themselves, executed on the Hexagon simulator.

Every other convolution test in this directory compares torch against a
transcription of the C. This one runs the C: `hvx_conv_depthwise2d_fp16` and
`hmx_im2col_convolution_fp16`, compiled for v79 and executed under hexagon-sim,
with the fp16 bit patterns compared against torch. It needs HEXAGON_SDK_ROOT and
a libncurses5 for the simulator, and skips with a reason when either is missing.

The runner builds its operands the way the emitters do, and each case prints a
digest of the packed weight it used, which the host side reproduces through the
emitter's own packers: the numbers below are therefore about the bytes this
backend would hand the kernel rather than about a layout written down twice.
"""

import pathlib

import numpy as np
import pytest
import torch


import hexagon_sim
from executorch.backends.hexagon.hexagon_ops import (
    ConvSpec,
    pack_conv_weight,
    pack_depthwise_weight,
)

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/conv_runner.cpp"

#: The vendored sources these two kernels need; nothing here is reimplemented.
_SOURCES = [
    "conv_depthwise_ops.cc",
    "im2col_convolution_fp16.cc",
    "blit_ops.cc",
    "worker_pool.cc",
    "vtcm_mgr.cc",
    "hmx_mgr.cc",
    "hmx_queue.cc",
    "power.cc",
    "ops/depthwise_conv_fp16.c",
]

#: htp_ops.h is generated from the IDL, so the runner gets a shim with the one
#: AEEStdDef declaration the vendored headers expect.
_HEADERS = {"htp_ops.h": '#pragma once\n#include "AEEStdDef.h"\n'}

_PACK = 64

#: The runner's own tables: tag, and the geometry each case was built with.
_DEPTHWISE = [
    ("DW_1X1", 64, 3, 1, 1, 1, 8, 0, 0, 1),
    ("DW_2X2", 64, 3, 2, 1, 1, 9, 0, 0, 1),
    ("DW_DIL", 64, 3, 1, 0, 2, 7, 0, 0, 1),
    ("DW_RELU", 64, 3, 1, 1, 1, 8, 1, 0, 1),
    ("DW_RELU6", 64, 3, 1, 1, 1, 8, 0, 1, 1),
    ("DW_BLOCKS", 128, 3, 2, 1, 1, 9, 0, 0, 2),
]

_CONV = [
    ("CV_3X3", 64, 64, 3, 1, 1, 1, 5, 1),
    ("CV_STRIDE", 64, 64, 3, 2, 1, 1, 8, 1),
    ("CV_NOPAD", 64, 64, 3, 1, 0, 1, 5, 1),
    ("CV_1X1", 64, 96, 1, 1, 0, 1, 5, 1),
    ("CV_IC96", 96, 64, 3, 1, 1, 1, 5, 1),
    ("CV_DIL", 64, 64, 3, 1, 2, 2, 7, 1),
    ("CV_BATCH", 64, 64, 3, 1, 1, 1, 5, 2),
    ("CV_IC3", 3, 32, 3, 2, 1, 1, 8, 1),
]

#: A geometry whose position count is not a whole number of 32-position tiles,
#: run with one channel tile per pass: the shape the emitter's pair avoids.
_SINGLE = ("CV_ODD", 64, 64, 3, 1, 1, 1, 5, 1)

#: The same chunking over a single channel tile, where the store has no second
#: tile to hand the wrong bias to.
_SINGLE_EXACT = ("CV_ONE", 3, 32, 3, 2, 1, 1, 8, 1)

#: The two shapes with the padding lanes filled in, and the value the finite one
#: moves the answer by.
_POISONED = [("CV_INF", 1), ("CV_READ", 2)]
_PADDING_VALUE = 3.0
_WEIGHT_PADDING_VALUE = 1.0


@pytest.fixture(scope="module")
def simulated():
    try:
        return hexagon_sim.run(_RUNNER, _SOURCES, headers=_HEADERS)
    except hexagon_sim.BuildFailed as error:
        # A runner that does not compile is a failure, because every assertion
        # below is about what the simulator computed and a skip would read as
        # agreement.
        pytest.fail(str(error), pytrace=False)
    except hexagon_sim.Unavailable as error:
        # A machine without the SDK has nothing to say here and skips.
        pytest.skip(str(error))


def _bits(values):
    return np.asarray(values, dtype=np.float16).view(np.uint16).reshape(-1).tolist()


def _uint16(result, tag):
    return np.array(result[tag], dtype=np.uint16)


def _source(batch, channels, width):
    """The activation both sides build, in the blocked layout the DSP reads."""
    values = np.zeros((batch, channels, width, width), dtype=np.float16)
    for b in range(batch):
        for c in range(channels):
            for y in range(width):
                for x in range(width):
                    values[b, c, y, x] = np.float16(
                        ((y * width + x) * 7 + c * 3) % 11 - 5
                    )
    return values


def _depthwise_weight(channels, kernel):
    weight = np.zeros((channels, 1, kernel, kernel), dtype=np.float16)
    for c in range(channels):
        for ky in range(kernel):
            for kx in range(kernel):
                weight[c, 0, ky, kx] = np.float16(((c * 3 + ky * 7 + kx * 11) % 13) - 6)
    return weight


def _conv_weight(oc, ic, kernel):
    weight = np.zeros((oc, ic, kernel, kernel), dtype=np.float16)
    for o in range(oc):
        for c in range(ic):
            for ky in range(kernel):
                for kx in range(kernel):
                    weight[o, c, ky, kx] = np.float16(
                        ((o * 3 + c * 5 + ky * 7 + kx * 11) % 13) - 6
                    )
    return weight


def _bias(channels):
    return np.array(
        [
            np.float16((i % 5) - 2 if i < channels else 0)
            for i in range(channels + _PACK)
        ],
        dtype=np.float16,
    )


def _digest(packed, count):
    """The runner's own hash of a packed weight, over fp16 words."""
    words = np.frombuffer(packed, dtype="<u2")[:count]
    hash_value = 0
    for word in words:
        hash_value = (hash_value * 31 + int(word)) & 0xFFFFFFFF
    return hash_value, words[:16].tolist()


def _spec(batch, ic, hw, oc, kernel, stride, pad, dilate):
    extent = (hw + 2 * pad - dilate * (kernel - 1) - 1) // stride + 1
    return ConvSpec(
        batch=batch,
        in_channels=ic,
        in_h=hw,
        in_w=hw,
        out_channels=oc,
        out_h=extent,
        out_w=extent,
        kernel_y=kernel,
        kernel_x=kernel,
        stride_y=stride,
        stride_x=stride,
        pad_y=pad,
        pad_x=pad,
        dilate_y=dilate,
        dilate_x=dilate,
        depthwise=False,
    )


def _convolution_reference(case):
    """What torch computes for one of the runner's cases."""
    _, ic, oc, kernel, stride, pad, dilate, hw, batch = case
    return torch.nn.functional.conv2d(
        torch.from_numpy(_source(batch, ic, hw)),
        torch.from_numpy(_conv_weight(oc, ic, kernel)),
        torch.from_numpy(_bias(oc)[:oc]),
        stride=stride,
        padding=pad,
        dilation=dilate,
    ).numpy()


def _unblock(flat, channels, batch, area):
    """[ceil(C/64)][batch][area][64] back to [batch][channels][area]."""
    blocks = -(-channels // _PACK)
    values = (
        np.asarray(flat, dtype=np.uint16)
        .view(np.float16)
        .reshape(blocks, batch, area, _PACK)
    )
    out = np.zeros((batch, channels, area), dtype=np.float16)
    for cb in range(blocks):
        first = cb * _PACK
        width = min(_PACK, channels - first)
        out[:, first : first + width] = values[cb, :, :, :width].transpose(0, 2, 1)
    return out


def test_every_operand_the_kernels_read_is_vector_aligned(simulated):
    """The addresses the kernels were handed, low bits and all.

    An unaligned HVX access is wrong data rather than a crash, and these buffers
    live in a shared object whose layout a change to the case list can move, so
    the runner prints the low bits of every operand it passes and the cases below
    are only worth reading if they are zero.
    """
    assert simulated["ALIGN"] == [0] * 8, "an operand is not 128-byte aligned"


@pytest.mark.parametrize("case", _DEPTHWISE, ids=[case[0] for case in _DEPTHWISE])
def test_the_runner_receives_the_weight_the_emitter_packs(simulated, case):
    """The digest has to be over the emitter's own bytes.

    The kernels read a rearranged weight, so a case whose operand this layer
    would not have produced says nothing about the emitter. Both sides build the
    same tensor -- the runner from the same formula -- and compare their hashes
    of the packed result.
    """
    tag, channels, kernel = case[0], case[1], case[2]
    packed = pack_depthwise_weight(
        _depthwise_weight(channels, kernel), channels, kernel, kernel
    )
    count = -(-channels // _PACK) * kernel * kernel * _PACK
    digest, head = _digest(packed, count)
    assert (
        int(simulated[f"{tag}_WSUM"][0]) == digest
    ), f"{tag}: not the emitter's packing"
    assert _uint16(simulated, f"{tag}_WHEAD")[:16].tolist() == head


@pytest.mark.parametrize("case", _CONV, ids=[case[0] for case in _CONV])
def test_the_runner_receives_the_weight_the_emitter_packs_for_the_unit(simulated, case):
    """The im2col path's operand is the HMX unit's 32x32 tiling."""
    tag, ic, oc, kernel, stride, pad, dilate, hw, batch = case
    weight = _conv_weight(oc, ic, kernel)
    packed = pack_conv_weight(
        weight, _spec(batch, ic, hw, oc, kernel, stride, pad, dilate)
    )
    tiles, groups = -(-oc // 32), -(-ic // 32)
    count = tiles * kernel * kernel * groups * 1024
    digest, head = _digest(packed, count)
    assert (
        int(simulated[f"{tag}_WSUM"][0]) == digest
    ), f"{tag}: not the emitter's packing"
    assert _uint16(simulated, f"{tag}_WHEAD")[:16].tolist() == head


@pytest.mark.parametrize("case", _DEPTHWISE, ids=[case[0] for case in _DEPTHWISE])
def test_the_depthwise_walk_on_the_dsp_matches_torch(simulated, case):
    """The kernel's own numbers, including its window origin and its rounding.

    The values are small integers, so every partial sum is exact in fp16 and the
    comparison is an equality: a different origin, a transposed tap or a bias
    applied in the wrong place would be a different number rather than a
    rounding difference.
    """
    tag, channels, kernel, stride, pad, dilate, hw, relu, relu6, batch = case
    x = _source(batch, channels, hw)
    weight = _depthwise_weight(channels, kernel)
    bias = _bias(channels)[:channels]
    expected = torch.nn.functional.conv2d(
        torch.from_numpy(x),
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        stride=stride,
        padding=pad,
        dilation=dilate,
        groups=channels,
    ).numpy()
    if relu or relu6:
        expected = np.maximum(expected, 0)
        if relu6:
            expected = np.minimum(expected, np.float16(6))
    area = expected.shape[2] * expected.shape[3]
    got = _unblock(simulated[f"{tag}_OUT"], channels, batch, area).reshape(
        batch, channels, area
    )
    assert _bits(got.reshape(-1)) == _bits(
        expected.reshape(-1)
    ), f"{tag}: not bit-exact"


@pytest.mark.parametrize("case", _CONV, ids=[case[0] for case in _CONV])
def test_the_im2col_convolution_on_the_dsp_matches_torch(simulated, case):
    """The unit's product, its activation fill, its bias and its store.

    This is the whole command as one comparison: the im2col patch, the tile order
    the weight is packed in, the store's output layout and the bias the store
    adds. The values are small integers, so it is an equality and not a
    tolerance.
    """
    tag, ic, oc, kernel, stride, pad, dilate, hw, batch = case
    expected = _convolution_reference(case)
    area = expected.shape[2] * expected.shape[3]
    got = _unblock(simulated[f"{tag}_OUT"], oc, batch, area).reshape(batch, oc, area)
    assert _bits(got.reshape(-1)) == _bits(
        expected.reshape(-1)
    ), f"{tag}: not bit-exact"


def test_the_padding_lanes_reach_the_product(simulated):
    """The lanes past the last channel are read, and the zeros in them matter.

    With the weight's own padding lanes made nonzero and the activation's set to
    3, the answer moves by exactly one contribution per in-range tap: the 29
    lanes of the 32-channel group past the third channel, times the value in
    each. That is what makes the emitter's clearing of those lanes (DSP_OP_ZERO)
    load-bearing rather than housekeeping.
    """
    _, _, _, kernel, stride, pad, dilate, hw, batch = next(
        case for case in _CONV if case[0] == "CV_IC3"
    )
    assert _uint16(simulated, "CV_READ_A")[:4].view(np.float16).tolist() == [3.0] * 4
    assert _uint16(simulated, "CV_READ_W")[:4].view(np.float16).tolist() == [1.0] * 4

    extent = (hw + 2 * pad - dilate * (kernel - 1) - 1) // stride + 1
    plain = _unblock(simulated["CV_IC3_OUT"], 32, batch, extent * extent)
    poisoned = _unblock(simulated["CV_READ_OUT"], 32, batch, extent * extent)
    moved = (poisoned.astype(np.float32) - plain.astype(np.float32)).reshape(
        32, extent * extent
    )
    taps = np.array(
        [
            sum(
                0 <= oy * stride - pad + ky < hw and 0 <= ox * stride - pad + kx < hw
                for ky in range(kernel)
                for kx in range(kernel)
            )
            for oy in range(extent)
            for ox in range(extent)
        ],
        dtype=np.float32,
    )
    lanes = 32 - 3  # the group's lanes past the third channel
    expected = lanes * _PADDING_VALUE * _WEIGHT_PADDING_VALUE * taps
    # Every output channel moves by the same amount, which is the contribution
    # of the lanes the graph has no channels for.
    assert np.array_equal(
        moved, np.tile(expected, (32, 1))
    ), "the padding lanes did not reach the product"


def test_the_simulated_unit_does_not_turn_a_zero_weight_into_a_nan(simulated):
    """An infinity in a padding lane is absorbed, and why the clear is still there.

    `CV_INF` is the same convolution with those lanes holding an infinity: the
    unit's answer is unchanged, so this model flushes 0 x inf rather than
    propagating a NaN. That is a property of the unit this runs on and not
    something the emitter can assume of a real device, which is why the
    activation's padding lanes are cleared before the fill reads them.
    """
    assert _uint16(simulated, "CV_INF_A")[:4].view(np.float16).tolist() == [np.inf] * 4
    assert np.array_equal(
        _uint16(simulated, "CV_IC3_OUT"), _uint16(simulated, "CV_INF_OUT")
    )


def test_the_single_tile_chunking_is_what_the_emitter_avoids(simulated):
    """The pair is load-bearing, and this is the shape that shows it.

    With one position tile and one channel tile per pass the store's ragged-tile
    path rotates the accumulator *after* adding the bias: the last position of a
    position tile that is not 32-position aligned comes back with the bias of the
    channel tile 32 lanes up instead of its own. The same geometry through the
    pair store is exact, and the emitter pins `mp = 1, np = 2`.
    """
    _, ic, oc, kernel, stride, pad, dilate, hw, batch = _SINGLE
    expected = _convolution_reference(_SINGLE)
    area = expected.shape[2] * expected.shape[3]
    got = _unblock(simulated["CV_ODD_OUT"], oc, batch, area)
    want = expected.reshape(batch, oc, area)

    moved = np.nonzero(got.reshape(oc, area) != want.reshape(oc, area))
    bias = _bias(oc)[:oc].astype(np.float32)
    assert set(moved[0].tolist()) == set(range(32, 64)), "the wrong channels moved"
    assert set(moved[1].tolist()) == {area - 1}, "the wrong positions moved"
    assert got[0, 32:, -1].tolist() == [
        float(want[0, 32 + lane, -1]) + float(bias[lane]) - float(bias[32 + lane])
        for lane in range(32)
    ], "the change is not the neighbouring tile's bias"


def test_the_single_tile_chunking_is_exact_with_one_channel_tile(simulated):
    """The control for the case above: `np = 1` is not broken by itself.

    One channel tile has no neighbour's bias to take, so the same chunking is
    exact -- which is what makes the failure above a statement about the store's
    ragged path rather than about the chunking parameters.
    """
    expected = _convolution_reference(_SINGLE_EXACT)
    _, ic, oc, kernel, stride, pad, dilate, hw, batch = _SINGLE_EXACT
    area = expected.shape[2] * expected.shape[3]
    got = _unblock(simulated["CV_ONE_OUT"], oc, batch, area)
    assert _bits(got.reshape(-1)) == _bits(expected.reshape(-1)), "not bit-exact"


def test_the_zero_command_clears_exactly_its_size(simulated):
    """htp_ops_zero is the memset the ragged channel count needs."""
    result = simulated["ZERO"]
    assert int(result[0]) == 0  # the call's own status
    assert [int(word) for word in result[1:4]] == [0, 0, 0]
    # The word after the cleared span still holds the 0xAB the buffer was filled
    # with, and the cleared span ends at 130 rather than 128.
    assert int(result[4]) == 0xAB


def test_the_comparisons_can_fail():
    """Reverse controls: the equalities above compare values a change would move.

    Without these, a comparison that always passed would look like agreement.
    """
    values = np.arange(8, dtype=np.uint16)
    assert _bits(values.view(np.float16)) != _bits(values[::-1].copy().view(np.float16))
    # The unblocking is a permutation, so two channels cannot come out equal.
    flat = np.arange(2 * _PACK, dtype=np.uint16)
    unblocked = _unblock(flat, 64, 1, 2)
    assert _bits(unblocked[0, 0]) != _bits(unblocked[0, 1])
