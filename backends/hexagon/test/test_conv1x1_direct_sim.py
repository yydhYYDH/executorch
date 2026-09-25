# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The 1x1 direct convolution, executed on the Hexagon simulator.

The command 17 entry point `htp_ops_conv1x1_direct_fp16` forwards to
`hmx_im2col_convolution_fp16`, and what the emitter chooses it for is the
function's own 1x1 activation fill: the direct copy of the plane, and the strided
direct gather. This compiles the vendored sources for v79 and runs the entry
point under hexagon-sim, so what is compared with torch is the DSP's own
answer, not a transcription of it. The reference is torch in fp64, so the
comparison says how far the fp16 unit is from the arithmetic rather than what a
second fp16 implementation happens to do.

The output is the whole blocked buffer printed as hex words, and it has to be
non-empty: a runner that printed nothing would leave the key missing and the
parse would skip the line.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import hexagon_sim  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    ConvSpec,
    conv_1x1_direct_applies,
    pack_conv_weight,
)

_PACK = 64
_TILE = 32

_RUNNER = (
    pathlib.Path(__file__).resolve().parent / "sim/conv1x1_direct_runner.cpp"
)

#: The vendored sources this kernel needs, the same list the im2col runner uses.
_SOURCES = [
    "im2col_convolution_fp16.cc",
    "blit_ops.cc",
    "worker_pool.cc",
    "vtcm_mgr.cc",
    "hmx_mgr.cc",
    "hmx_queue.cc",
    "power.cc",
]

_HEADERS = {"htp_ops.h": '#pragma once\n#include "AEEStdDef.h"\n'}

#: tag, in-channels, out-channels, stride, width: the runner's own table.
_DIRECT = [
    ("D1_COPY", 64, 64, 1, 5),
    ("D1_STRIDE", 64, 64, 2, 8),
    ("D1_SPLIT", 64, 64, 3, 7),
    ("D1_OC", 64, 96, 1, 5),
]


def _check():
    """The SDK has to be there before a run, or Unavailable is ambiguous.

    hexagon_sim.run raises Unavailable both for a machine without the SDK and
    for a runner that did not execute, so the availability check is made first
    and a later Unavailable is a failure rather than a skip.
    """
    return hexagon_sim._check()


@pytest.fixture(scope="module")
def simulated():
    try:
        _check()
        return hexagon_sim.run(_RUNNER, _SOURCES, headers=_HEADERS)
    except hexagon_sim.BuildFailed as error:
        # A runner that does not compile is a failure, not a skip: every
        # assertion below is about what the simulator computed.
        pytest.fail(str(error), pytrace=False)
    except hexagon_sim.Unavailable as error:
        # The SDK was present a line ago, so the simulator could not run the
        # binary it just built. That is a failure, not a missing SDK.
        pytest.fail(f"the SDK is present but the runner did not run: {error}")


def _bits(values):
    return np.asarray(values, dtype=np.float16).view(np.uint16).reshape(-1).tolist()


def _uint16(result, tag):
    return np.array(result[tag], dtype=np.uint16)


def _source(channels, width):
    """The activation the runner builds, in the blocked layout the DSP reads."""
    values = np.zeros((channels, width, width), dtype=np.float16)
    for c in range(channels):
        for y in range(width):
            for x in range(width):
                values[c, y, x] = np.float16(((y * width + x) * 7 + c * 3) % 11 - 5)
    return values


def _weight(oc, ic):
    weight = np.zeros((oc, ic, 1, 1), dtype=np.float16)
    for o in range(oc):
        for c in range(ic):
            weight[o, c, 0, 0] = np.float16(((o * 3 + c * 5) % 13) - 6)
    return weight


def _bias(channels):
    return np.array(
        [np.float16((i % 5) - 2 if i < channels else 0) for i in range(channels + _PACK)],
        dtype=np.float16,
    )


def _spec(ic, oc, stride, width):
    extent = (width - 1) // stride + 1
    return ConvSpec(
        batch=1,
        in_channels=ic,
        in_h=width,
        in_w=width,
        out_channels=oc,
        out_h=extent,
        out_w=extent,
        kernel_y=1,
        kernel_x=1,
        stride_y=stride,
        stride_x=stride,
        pad_y=0,
        pad_x=0,
        dilate_y=1,
        dilate_x=1,
        depthwise=False,
    )


def _unblock(flat, channels, area):
    """[ceil(C/64)][area][64] back to [channels][area], lane last to lane first."""
    blocks = -(-channels // _PACK)
    values = (
        np.asarray(flat, dtype=np.uint16)
        .view(np.float16)
        .reshape(blocks, area, _PACK)
    )
    out = np.zeros((channels, area), dtype=np.float16)
    for cb in range(blocks):
        first = cb * _PACK
        width = min(_PACK, channels - first)
        out[first : first + width] = values[cb, :, :width].transpose(1, 0)
    return out


def _fp64_reference(ic, oc, stride, width):
    """torch's own convolution, in fp64, from the operands the runner built."""
    return (
        torch.nn.functional.conv2d(
            torch.from_numpy(_source(ic, width)).double().unsqueeze(0),
            torch.from_numpy(_weight(oc, ic)).double(),
            torch.from_numpy(_bias(oc)[:oc]).double(),
            stride=stride,
        )
        .squeeze(0)
        .numpy()
    )


@pytest.mark.parametrize("case", _DIRECT, ids=[case[0] for case in _DIRECT])
def test_the_direct_entry_point_answers_on_the_simulator(simulated, case):
    """The command 17 entry point returns 0 and writes a non-empty output."""
    tag, ic, oc, stride, width = case
    assert int(simulated[f"{tag}_RET"][0]) == 0, f"{tag}: nonzero return"
    extent = (width - 1) // stride + 1
    area = extent * extent
    words = simulated[f"{tag}_OUT"]
    expected_words = -(-oc // _PACK) * area * _PACK
    assert len(words) == expected_words, f"{tag}: the output is the wrong size"
    assert len(words) > 0, f"{tag}: the output is empty"
    values = _uint16(simulated, f"{tag}_OUT")
    assert np.any(values != 0), f"{tag}: every output word is zero"


@pytest.mark.parametrize("case", _DIRECT, ids=[case[0] for case in _DIRECT])
def test_the_direct_1x1_convolution_on_the_dsp_matches_fp64_torch(simulated, case):
    """The DSP's 1x1 direct answer against torch computed in fp64.

    The operands are small integers, so an exact product in fp16 is within
    half an ulp of the fp64 value and the two differ only where fp16 cannot
    hold the sum exactly. The tolerance is one ulp of the largest magnitude in
    the case, which is what the fp16 output format can express; a different
    window origin, a transposed tile or a bias in the wrong place would be
    orders of magnitude away from it rather than inside it.
    """
    tag, ic, oc, stride, width = case
    extent = (width - 1) // stride + 1
    area = extent * extent
    got = _unblock(simulated[f"{tag}_OUT"], oc, area)
    expected = _fp64_reference(ic, oc, stride, width).reshape(oc, area)
    scale = max(1.0, float(np.abs(expected).max()))
    tolerance = np.spacing(np.float16(scale)) * 2
    assert np.all(np.abs(got.astype(np.float64) - expected) <= tolerance), (
        f"{tag}: max |dsp - fp64| = "
        f"{np.abs(got.astype(np.float64) - expected).max()} > {tolerance}"
    )
    # The exact fp16 expectation, which the integer operands should give
    # bit for bit, is checked as well so a drift in one element is not hidden
    # by a tolerance on the rest.
    expected16 = expected.astype(np.float16)
    assert _bits(got.reshape(-1)) == _bits(expected16.reshape(-1)), f"{tag}: not exact"


@pytest.mark.parametrize("case", _DIRECT, ids=[case[0] for case in _DIRECT])
def test_the_runner_receives_the_weight_the_emitter_packs_for_the_direct(
    simulated, case
):
    """The weight bytes are the ones pack_conv_weight produces for this spec."""
    tag, ic, oc, stride, width = case
    packed = pack_conv_weight(_weight(oc, ic), _spec(ic, oc, stride, width))
    count = -(-oc // _TILE) * -(-ic // _TILE) * 1024
    values = np.frombuffer(packed, dtype=np.uint16)[:count]
    digest = 0
    for value in values.tolist():
        digest = (digest * 31 + value) & 0xFFFFFFFF
    # The parser has already read the digest as a hex int; a second base-16
    # conversion would be a TypeError rather than a wrong number.
    assert int(simulated[f"{tag}_WSUM"][0]) == digest, f"{tag}: not the packing"
    assert _uint16(simulated, f"{tag}_WHEAD").tolist() == values[:16].tolist()


@pytest.mark.parametrize("case", _DIRECT, ids=[case[0] for case in _DIRECT])
def test_every_simulated_geometry_is_one_the_emitter_would_call_direct(case):
    """The runner's cases and the emitter's predicate agree.

    A case the emitter would refuse would still run here -- the DSP's fill
    selection is the C's -- but it would say nothing about the command 17 the
    emitter emits, because the emitter would not have emitted 17 for it.
    """
    _, ic, oc, stride, width = case
    assert conv_1x1_direct_applies(_spec(ic, oc, stride, width))
