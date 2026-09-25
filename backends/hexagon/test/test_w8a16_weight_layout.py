# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The int8 weight layout the w8a16 kernels read, against the two sources that state it.

`pack_w8a16_gemv_weight` is the only int8 weight order this backend packs, and it
serves two entries rather than one. The M == 1 GEMV command reads the whole blob;
the M > 1 prefill command reads the same tiles and takes its scales from where
they end. So the tile order is a property of the layout, not of either kernel,
and a test of it is a test of both.

Two independent statements of that order are readable from here, and the two are
not the same kind of evidence. The bytes are produced on a device by
`reorderInt8SymWeightForHmx` (`MNN/source/backend/hexagon/execution/HexagonConvolution.cpp:464`),
a host function that is not in the vendored tree, so `_host_packer` below is a
transcription of its weight loop for a 1x1 window. The same order is also written
out in prose in the vendored kernel's own header
(`third-party/mnn-htp-ops/src/dsp/ops/matmul_w8a16_gemv_i8.c:11-22`), and
`_vendored_contract` is that prose as an index expression. The two are compared
against each other and against this backend's packer, so a misreading of either
one is a failure rather than a shared assumption.

The last test is the control that gives the comparison teeth: the vendored fp16
tile order read two ways, and the host packer's own index with its four-k
permutation dropped, must each hold the same weights in a different arrangement.
A fixture whose bytes could not tell those apart could not see a packing bug
either.

Tier: host. Two statements of one order agreeing, and an implementation agreeing
with both, is evidence about the order and nothing about a DSP: whether the
kernel consumes it is what `test_gemv_on_sim.py` and `test_prefill_on_sim.py`
measure on the simulator.
"""

import os
import pathlib
import sys
import zlib

import numpy as np
import pytest

# The checkout directory is itself named executorch, so putting its parent on the
# path makes `import executorch` resolve to this tree; the editable install in
# this environment points at a different checkout.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

from executorch.backends.hexagon.hexagon_ops import pack_w8a16_gemv_weight  # noqa: E402

#: The tile an int8 weight blob is made of, as the kernel's own scale pointer
#: spells it -- `scales = (const __fp16*)(weight + np*kp*1024)`,
#: `conv1x1_w8a16_sym_per_channel.cc:521` -- and as the GEMV contract's blob
#: arithmetic does (`matmul_w8a16_gemv_i8.c:19-22`).
_TILE_BYTES = 1024

#: K a multiple of 64 and N of 32, every one of them a whole number of packs in
#: both axes: the host packer has branches that skip padded output channels and
#: padded k (`:532-539`), and neither transcription models those, so a ragged N
#: would be testing something this file does not claim.
_SHAPES = ((64, 32), (64, 64), (128, 96), (192, 32), (256, 64))

#: Byte p of a group holds the weight for k = kx*32 + 4g + perm[p], which is what
#: the GEMV kernel compensates for by splatting the activation in the same order
#: (matmul_w8a16_gemv_i8.c:19-20, and `splat_group_permuted` at `:155`).
_PERM = (0, 2, 1, 3)


def _operands(k, n):
    """A weight and scales that make a permuted tile a different tile.

    Seeded from the shape rather than from `hash`, which is randomized per
    process: a corpus whose inputs move between runs cannot tell a layout change
    from a different input.
    """
    rng = np.random.default_rng(zlib.crc32(f"w8a16-{k}x{n}".encode()))
    weight = rng.integers(-128, 128, size=(k, n))
    scale = np.array([2.0 ** (-3 + (i % 5)) for i in range(n)], dtype=np.float32)
    return weight, scale


def _host_packer(weight, scale, k, n):
    """`reorderInt8SymWeightForHmx` for a 1x1 window, byte for byte.

    HexagonConvolution.cpp:464-551 with `ic = k` and `oc = n`. The raw weight
    arrives in `[oc][ic]` order and its source index is `o * ic + i` (`:540`); the
    loop nests oz, kk, oy, ix as the original does; and one element's destination
    is `blockBase + (ixPair/2)*128 + lane*2 + (ixPair&1)` (`:544`, with
    `lane = oy*2 + ixRem` at `:543`). With one scale block and no asymmetry the
    tail is the whole per-channel array narrowed in one call (`:489-491`), which
    for these shapes is exactly `oc` halves.
    """
    ic, oc = k, n
    ic_pack = oc_pack = 32
    ic_p, oc_p = ic // ic_pack, (oc + oc_pack - 1) // oc_pack
    kp = ic_p  # kernelY * kernelX * icP, with a 1x1 window
    packs = ic_pack * oc_pack
    weight_bytes = oc_p * kp * packs
    out = np.zeros(weight_bytes + oc_p * 32 * 2, dtype=np.uint8)
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    out[weight_bytes : weight_bytes + 2 * oc] = np.asarray(scales, dtype="<f2").view(
        np.uint8
    )
    raw = np.asarray(weight, dtype=np.int32).T.astype(np.uint8)  # [oc][ic]
    for oz in range(oc_p):
        for kk in range(kp):
            block_base = (oz * kp + kk) * packs
            for oy in range(oc_pack):
                o = oz * oc_pack + oy
                for ix in range(ic_pack):
                    i = kk * ic_pack + ix
                    ix_pair, ix_rem = ix // 2, ix & 1
                    lane = oy * 2 + ix_rem
                    out[
                        block_base + (ix_pair // 2) * 128 + lane * 2 + (ix_pair & 1)
                    ] = raw[o, i]
    return out.tobytes()


def _vendored_contract(weight, k, n):
    """The same order as the vendored kernel states it, tiles only.

    `matmul_w8a16_gemv_i8.c:11-22`: the blob is `kp*np` tiles of 1024 bytes with
    tile (oy, kx) at `(oy*kp + kx)*1024`, and byte `g*128 + ocIn*4 + p` of a tile
    holds the weight for `k = kx*32 + 4g + perm[p]`.
    """
    w = np.asarray(weight, dtype=np.int32)
    kp, np_ = k // 32, n // 32
    blob = np.zeros(np_ * kp * _TILE_BYTES, dtype=np.uint8)
    for oy_tile in range(np_):
        for kx in range(kp):
            base = (oy_tile * kp + kx) * _TILE_BYTES
            for g in range(8):
                for oc_in in range(32):
                    for p in range(4):
                        k_index = kx * 32 + 4 * g + _PERM[p]
                        blob[base + g * 128 + oc_in * 4 + p] = (
                            w[k_index, oy_tile * 32 + oc_in] & 0xFF
                        )
    return blob.tobytes()


def _fp16_layout_widened(weight, k, n):
    """The vendored fp16 tile order, read with one-byte elements.

    `htp_ops_weight_reorder` (`blit_ops.cc:1842`) writes an `int16_t` at
    `blockBase + ix_pair*64 + oy*2 + ixRem` inside a 2048-byte tile (`:1878`).
    Read as byte offsets with the element width dropped, that is
    `(ix//2)*64 + oy*2 + (ix&1)` inside a 1024-byte tile: the same blob shape and
    the same tile count, and no permutation of the four k of a group.
    """
    return _tile_order(weight, k, n, lambda ix, oy: (ix // 2) * 64 + oy * 2 + (ix & 1))


def _unpermuted_groups(weight, k, n):
    """The host packer's own index with the four-k permutation dropped.

    `(ix//4)*128 + oy*4 + ix%4` is the expression `HexagonConvolution.cpp:544`
    reduces to when `perm` is the identity, i.e. the same layout with each group of
    four k in natural order. This is the candidate that differs from the real one
    in exactly the place the kernel's own activation splat has to compensate for.
    """
    return _tile_order(weight, k, n, lambda ix, oy: (ix // 4) * 128 + oy * 4 + ix % 4)


def _tile_order(weight, k, n, at):
    """A blob of the same shape that places k index ix of channel oy at `at`."""
    w = np.asarray(weight, dtype=np.int32)
    kp, np_ = k // 32, n // 32
    blob = np.zeros(np_ * kp * _TILE_BYTES, dtype=np.uint8)
    for oz in range(np_):
        for kk in range(kp):
            base = (oz * kp + kk) * _TILE_BYTES
            for ix in range(32):
                for oy in range(32):
                    blob[base + at(ix, oy)] = w[kk * 32 + ix, oz * 32 + oy] & 0xFF
    return blob.tobytes()


def _tile_bytes(k, n):
    return (k // 32) * (n // 32) * _TILE_BYTES


@pytest.mark.parametrize("k,n", _SHAPES)
def test_the_int8_packer_writes_the_host_packers_bytes(k, n):
    """The packer's tiles, and the tail the prefill entry reads after them."""
    weight, scale = _operands(k, n)
    expected = _host_packer(weight, scale, k, n)
    tiles = pack_w8a16_gemv_weight(weight, k, n)
    assert len(tiles) == _tile_bytes(k, n)
    assert tiles == expected[: len(tiles)]
    # An int8 prefill blob is these tiles plus one fp16 scale per output channel,
    # which is what upstream's single `int8Weight` buffer holds for both entries
    # (HexagonConvolution.cpp:1152-1170, :892-908).
    tail = np.frombuffer(expected[len(tiles) :], dtype="<f2")
    assert np.array_equal(tail, scale.astype(np.float16))


@pytest.mark.parametrize("k,n", _SHAPES)
def test_the_vendored_contract_and_the_host_packer_state_one_order(k, n):
    """The two readable statements of the layout agree with each other."""
    weight, _ = _operands(k, n)
    contract = _vendored_contract(weight, k, n)
    assert (
        contract
        == _host_packer(weight, np.ones(n, dtype=np.float32), k, n)[: len(contract)]
    )
    assert contract == pack_w8a16_gemv_weight(weight, k, n)


@pytest.mark.parametrize("k,n", _SHAPES)
def test_the_fp16_tile_order_is_not_this_order(k, n):
    """Each candidate must differ, from both authorities and from the packer."""
    weight, _ = _operands(k, n)
    upstream = _host_packer(weight, np.ones(n, dtype=np.float32), k, n)[
        : _tile_bytes(k, n)
    ]
    packer = pack_w8a16_gemv_weight(weight, k, n)
    for candidate, name in (
        (_fp16_layout_widened(weight, k, n), "fp16 order widened"),
        (_unpermuted_groups(weight, k, n), "groups unpermuted"),
    ):
        # Teeth: a candidate that differed by holding wrong bytes would be a weak
        # control, so require the same weights in a different arrangement.
        got, want = np.frombuffer(candidate, np.uint8), np.frombuffer(
            upstream, np.uint8
        )
        assert np.array_equal(np.sort(got), np.sort(want)), name
        assert candidate != upstream, name
        # And the comparison the first two tests make has to be able to reject a
        # packer that writes either candidate: a check that cannot tell this
        # layout from the fp16 one cannot see the permutation either.
        assert candidate != packer, name
    # And the two candidates are not the same control twice.
    assert _fp16_layout_widened(weight, k, n) != _unpermuted_groups(weight, k, n)
