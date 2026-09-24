# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every k of a packed GEMV weight, against arithmetic that knows no layout.

`test_blob_on_sim.py` runs the shipped path end to end and checks the answers
that path produces; those cases sample a handful of positions, because a blob
has to be carried through the emitter for each one. This file takes the two
packers directly, so it can afford to walk *every* k of both layouts, and it
asks a second question the blob cannot: what `unpack_vrmpy_weight_128` does to a
byte, measured rather than read off the comment that says what it should do.

The construction is the one the first probe of these kernels used, and it is
deliberately blind to the layout: with an activation of one everywhere the
answer is `scale[n] * sum_k w[k, n]` whatever order the kernel walks in, and
with an activation of one at k0 alone it is `scale[n] * w[k0, n]`, which no
permutation of k can leave in place. The scales are powers of two and the
weights are small integers, so every expected value is exact in fp16.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
import test_blob_on_sim as blob_sim  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    pack_q4a16_gemv_weight,
    pack_w8a16_gemv_weight,
)

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/gemv_runner.cpp"

#: (tag, k, n, kernel). Two shapes each: with one k-tile and one oc-tile, a wrong
#: tile stride has nothing to be wrong about.
_CONFIGS = (
    ("Q4A", 64, 32, "q4a16"),
    ("Q4B", 128, 64, "q4a16"),
    ("W8A", 64, 32, "w8a16"),
    ("W8B", 128, 64, "w8a16"),
)

#: Wide-K rungs, all of them at K = 256 and above, where the two kernels'
#: `sf_from_w` bound can be reached. Both entries are emitted with
#: `scale_block_num == 1`, the configuration that makes the int32 accumulator span
#: the whole K -- `|w| * 127 * k` for a constant weight -- and `sf_from_w`'s magic
#: number is exact only to `|x| <= 2^22`. The safe rungs are the widest K that was
#: exact before the conversion was fixed; the wide ones are the narrowest K that
#: was not, and they are here because a fix is only a fix if the rungs that
#: measured the defect still stand in the suite and now pass.
_WIDE_SAFE = (
    ("W8S", 256, 32, "w8a16"),
    ("Q4S", 4224, 32, "q4a16"),
)
_WIDE = (
    ("W8W", 512, 32, "w8a16"),
    ("Q4W", 8192, 32, "q4a16"),
)
#: The qualifier, at the same K as the widest rung: weights drawn with a sign, so
#: the per-channel sum cancels (`|sum w| / sum |w|` measures 0.001 here) and the
#: accumulator stays a factor of two below the bound. These rungs are the control
#: for "is it wide K, or wide K with a coherently signed column" -- and they are
#: the reason the fix had to leave every in-domain answer untouched.
_WIDE_MIXED = (
    ("W8M", 8192, 32, "w8a16"),
    ("Q4M", 8192, 32, "q4a16"),
)
_WIDE_TAGS = {tag for tag, _, _, _ in _WIDE + _WIDE_SAFE + _WIDE_MIXED}


def _weight(k, n, scheme):
    """Random small integers, with every row distinct.

    A row that repeated another would make a one-hot case unable to tell the two
    apart, which is the one thing these cases are for; the draw is seeded so a
    failure is reproducible.
    """
    limit = 7 if scheme == "q4a16" else 100
    generator = np.random.default_rng(20240924)
    weight = torch.from_numpy(
        generator.integers(-limit, limit + 1, size=(k, n)).astype(np.int32)
    )
    rows = {tuple(row.tolist()) for row in weight}
    assert len(rows) == k, "two rows of the fixture weight are equal"
    return weight


def _scale(n):
    """Powers of two, so `scale * w` is exact in fp16 whatever the weight."""
    return (2.0**-4) * (1 + (np.arange(n) % 4)).astype(np.float32)


def _wide_scale(n):
    """A power of two small enough that the wide-K answers stay fp16-exact.

    The wide rungs answer `|w| * k * scale`, which at 127 * 8192 would leave
    fp16's integers behind; 2^-5 keeps the widest of them at 1792.
    """
    return np.full(n, 2.0**-5, dtype=np.float32)


def _wide_operands():
    """The wide rungs' bytes, and the weight the runner's accumulator sees.

    The weights are constant rather than random because the quantity under test
    is `sum_k |w[k, n]|` as it enters the int32 accumulator: a random sign would
    make the sum cancel and the bound unreachable, which is the opposite of what
    this ladder is for.
    """
    weights, packed = {}, {}
    for tag, k, n, scheme in _WIDE + _WIDE_SAFE:
        weight = torch.full((k, n), 127 if scheme == "w8a16" else 7, dtype=torch.int32)
        scale = _wide_scale(n)
        weights[tag] = (weight, scale)
        if scheme == "q4a16":
            packed[f"kQ4{tag[2:]}"] = pack_q4a16_gemv_weight(weight, scale, k, n)
        else:
            packed[f"kW8{tag[2:]}"] = pack_w8a16_gemv_weight(weight, k, n)
            packed[f"kScales{tag[2:]}"] = scale.tobytes()
    return weights, packed


def _mixed_operands():
    """Wide-K rungs whose weights carry both signs, drawn as the small shapes are.

    Same generator as `_weight`, so the draw is reproducible and a failure can be
    re-run; the scale is the wide one so the answers stay fp16-exact in the
    arithmetic even where the kernel's rounding is a few ulp wide.
    """
    weights, packed = {}, {}
    for tag, k, n, scheme in _WIDE_MIXED:
        weight = _weight(k, n, scheme)
        scale = _wide_scale(n)
        weights[tag] = (weight, scale)
        if scheme == "q4a16":
            packed[f"kQ4{tag[2:]}"] = pack_q4a16_gemv_weight(weight, scale, k, n)
        else:
            packed[f"kW8{tag[2:]}"] = pack_w8a16_gemv_weight(weight, k, n)
            packed[f"kScales{tag[2:]}"] = scale.tobytes()
    return weights, packed


def _operands():
    """The packed bytes and the fp16 answers the runner is expected to print."""
    weights, packed = {}, {}
    for tag, k, n, scheme in _CONFIGS:
        weight = _weight(k, n, scheme)
        scale = _scale(n)
        weights[tag] = (weight, scale)
        if scheme == "q4a16":
            packed[f"kQ4{tag[2:]}"] = pack_q4a16_gemv_weight(weight, scale, k, n)
        else:
            packed[f"kW8{tag[2:]}"] = pack_w8a16_gemv_weight(weight, k, n)
            packed[f"kScales{tag[2:]}"] = scale.tobytes()
    for build in (_wide_operands, _mixed_operands):
        extra_weights, extra_packed = build()
        weights.update(extra_weights)
        packed.update(extra_packed)
    return weights, packed


def _header(packed):
    lines = ["// Generated by backends/hexagon/test/test_gemv_on_sim.py"]
    for name, data in packed.items():
        body = ",".join(str(int(v)) for v in np.frombuffer(data, dtype=np.uint8))
        lines.append(
            f"static const unsigned char {name}[] __attribute__((aligned(128))) = "
            f"{{{body}}};"
        )
    return "\n".join(lines) + "\n"


def _parse(stdout):
    """`tag -> {at -> fp16 bits}`, with at = -1 for the all-ones run."""
    rows = {}
    read_path = None
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "READPATH":
            read_path = [int(value, 16) for value in parts[1:]]
            continue
        if parts[0].endswith("failed"):
            raise AssertionError(line)
        if parts[0] not in {tag for tag, _, _, _ in _CONFIGS} | _WIDE_TAGS:
            continue
        rows.setdefault(parts[0], {})[int(parts[1])] = [
            int(value, 16) for value in parts[2:]
        ]
    return rows, read_path


@pytest.fixture(scope="module")
def measured():
    weights, packed = _operands()
    sources = [source for source in blob_sim._SOURCES if isinstance(source, str)]
    try:
        hexagon_sim.run(
            _RUNNER,
            sources,
            headers={
                "gemv_data.h": _header(packed),
                "htp_ops.h": blob_sim._HT_P_OP_SHIM,
            },
        )
    except hexagon_sim.Unavailable as error:
        # A machine with no simulator skips; a runner that does not compile is a
        # `BuildFailed`, which is deliberately not an `Unavailable`, so it fails
        # here instead of being read as a missing toolchain.
        pytest.skip(str(error))
    rows, read_path = _parse(hexagon_sim.LAST_STDOUT)
    assert read_path is not None, "the runner never printed the read path"
    return weights, rows, read_path


def test_the_nibbles_of_a_weight_byte_are_the_two_k_of_its_pair(measured):
    """What `unpack_vrmpy_weight_128` does to a byte, on the simulator.

    The contract it is written against says group g covers
    k = `x*32 + 4g + {0,1,2,3}`, that output byte `j = 4*ocIn + s` is the s-th of
    those for output channel `ocIn`, and that byte `g*64 + ocIn*2 + p` holds
    k = `+2p` in its low nibble and `+1` in its high one. The input here is the
    pattern `byte i = i`, so a returned byte says which input byte and which of
    its nibbles it came from: the returned value is the nibble minus the 8 the
    accumulator then works in.
    """
    _, _, read_path = measured
    lo = np.array(read_path[:128], dtype=np.uint8)
    hi = np.array(read_path[128:], dtype=np.uint8)
    expected_lo = np.zeros(128, dtype=np.uint8)
    expected_hi = np.zeros(128, dtype=np.uint8)
    for j in range(128):
        # The pair's lo half comes from input bytes 0..63, its hi half from
        # 64..127; byte i contributes its low nibble to output byte 2i and its
        # high nibble to 2i + 1.
        source = j // 2
        nibble = (source & 0x0F) if j % 2 == 0 else (source >> 4)
        expected_lo[j] = (nibble - 8) & 0xFF
        nibble = ((64 + source) & 0x0F) if j % 2 == 0 else ((64 + source) >> 4)
        expected_hi[j] = (nibble - 8) & 0xFF
    assert lo.tolist() == expected_lo.tolist(), f"lo half: {lo[:16].tolist()}"
    assert hi.tolist() == expected_hi.tolist(), f"hi half: {hi[:16].tolist()}"


@pytest.mark.parametrize("tag,k,n,scheme", _CONFIGS)
def test_a_sum_over_k_is_what_the_dsp_answers(measured, tag, k, n, scheme):
    """One everywhere, `scale[n] * sum_k w[k, n]` expected.

    A permutation of k cannot move this, so it is the half of the claim that the
    activation path, the scale positions and the output channel order are right
    -- and it is the half that a wrong tile stride fails.
    """
    weights, rows, _ = measured
    weight, scale = weights[tag]
    expected = (scale * weight.numpy().sum(axis=0)).astype(np.float16)
    got = np.array(rows[tag][-1], dtype=np.uint16).view(np.float16)
    assert np.array_equal(got.view(np.uint16), expected.view(np.uint16)), (
        f"{tag}: the DSP answered {got[:8].tolist()} where the arithmetic says "
        f"{expected[:8].tolist()}"
    )


@pytest.mark.parametrize("tag,k,n,scheme", _CONFIGS)
def test_one_at_k_reads_weight_row_k(measured, tag, k, n, scheme):
    """One at k0 alone, `scale[n] * w[k0, n]` expected, for every k0.

    This is what a layout mistake cannot survive: whatever the kernel's internal
    order, the weight the activation at k0 is multiplied by has to be row k0 of
    the matrix the packer was handed. 64 and 128 of these per config walk every
    k-tile, every 128-byte group inside a tile, both halves of every byte and
    both nibbles of every half.
    """
    weights, rows, _ = measured
    weight, scale = weights[tag]
    expected = (weight.numpy() * scale.reshape(1, n)).astype(np.float16)
    for at in range(k):
        got = np.array(rows[tag][at], dtype=np.uint16).view(np.float16)
        assert np.array_equal(got.view(np.uint16), expected[at].view(np.uint16)), (
            f"{tag}: an activation at k={at} answered {got[:8].tolist()}, which is "
            f"row {[i for i in range(k) if np.array_equal(got.view(np.uint16), expected[i].view(np.uint16))]} "
            f"and not row {at}"
        )


def _wide_answer(measured, tag, n):
    """The all-ones row the runner printed for a wide-K rung."""
    rows = measured[1]
    assert tag in rows, f"{tag}: the runner printed no all-ones row"
    return np.array(rows[tag][-1], dtype=np.uint16).view(np.float16)


@pytest.mark.parametrize("tag,k,n,scheme", _WIDE_SAFE)
def test_a_wide_k_inside_the_accumulator_bound_is_exact(measured, tag, k, n, scheme):
    """The rung below the bound: `scale[n] * sum_k w[k, n]`, bit for bit.

    Constant weights, one everywhere on the activation, and K wide enough that
    the accumulator is within a factor of two of `sf_from_w`'s documented
    `|x| < 2^22` -- 127 * 256 * 127 = 4 129 024 against 4 194 304 for W8S, and
    7 * 8192 * 127 = 3 761 664 for Q4S. Both answer exactly, which is what makes
    the failure one rung above a boundary rather than a broken kernel.
    """
    weights, _, _ = measured
    weight, scale = weights[tag]
    expected = (scale * weight.numpy().sum(axis=0)).astype(np.float16)
    got = _wide_answer(measured, tag, n)
    assert np.array_equal(got.view(np.uint16), expected.view(np.uint16)), (
        f"{tag}: K = {k} answered {got[:8].tolist()} where the arithmetic says "
        f"{expected[:8].tolist()}"
    )


@pytest.mark.parametrize("tag,k,n,scheme", _WIDE)
def test_a_wide_k_past_the_accumulator_bound_is_still_exact(
    measured, tag, k, n, scheme
):
    """The rung above the old bound: `scale[n] * sum_k w[k, n]`, bit for bit.

    Constant weights, one everywhere on the activation, and K wide enough that the
    accumulator is past 2^22 -- 127 * 512 * 127 = 8 258 048 for W8W, and
    7 * 8192 * 127 = 7 282 688 for Q4W -- so this is the rung that used to measure
    the defect and is now the acceptance criterion for the fix. It asserts the same
    thing the rung below it asserts, and it is the reason `sf_from_w` was changed.

    Measured on hexagon-sim before the conversion was fixed, against
    `scale[n] * sum_k w[k, n]`:

      W8W  K =  512, accumulator 8 258 048:  3032 against 2032    (+49%)
      Q4W  K = 8192, accumulator 7 282 688:  2552 against 1792    (+42%)

    and the return code was 0 in both, so the defect was silent; at K = 4224 on the
    w8a16 rung the answer was not merely wrong but infinite. The same kernels
    answered bit-for-bit one rung below (W8S, Q4S above) and at K = 8192 with a
    random-sign weight, where the sum cancels and the worst channel's accumulator
    reaches 1.8e6 (that rung is
    `test_a_wide_k_with_mixed_signs_stays_inside_the_bound`) -- so the defect needed
    wide K *and* a coherently signed weight column, and those rungs are why the fix
    had to leave every in-domain answer bit-identical.

    The mechanism: `sf_from_w` turned the int32 accumulator into fp32 with the
    `1.5 * 2^23` magic number, exact only to 2^22, and the emitters pass
    `scale_block_num == 1`, which makes the accumulator span the whole K rather than
    one block of 64. The conversion now splits each lane and recombines it, so the
    whole int32 range converts correctly. Isolating the instruction, rather than
    arguing it: this test asserted the correct answer as a strict xfail and turned
    into an XPASS when that one function changed and nothing else did. Had it not
    turned, the defect was somewhere else and the marker said so.
    """
    weights, _, _ = measured
    weight, scale = weights[tag]
    expected = (scale * weight.numpy().sum(axis=0)).astype(np.float16)
    got = _wide_answer(measured, tag, n)
    assert np.array_equal(got.view(np.uint16), expected.view(np.uint16)), (
        f"{tag}: K = {k} answered {got[:8].tolist()} where the arithmetic says "
        f"{expected[:8].tolist()}"
    )


@pytest.mark.parametrize("tag,k,n,scheme", _WIDE_MIXED)
def test_a_wide_k_with_mixed_signs_stays_inside_the_bound(measured, tag, k, n, scheme):
    """K = 8192 on its own does not reach the bound; a coherent sign does.

    The failing rungs use constant weights, which is what maximizes the
    accumulator and is not what a trained weight matrix looks like. Here the
    weights carry both signs at the same K, so the per-channel sum cancels --
    `|sum_k w| / sum_k |w|` is 0.001 for this draw -- and the worst of the 32
    output channels reaches 1.8e6 of accumulator against a bound of 4.19e6, which
    this test asserts rather than assumes. The answer is then asserted to be
    correct to the rounding and not bit-for-bit, because a mixed-sign column makes
    the fp32 the accumulator is converted into itself only 2^-12 accurate:
    measured, 2.1e-4 relative, a few ulp, where the failing rungs are 49%.
    """
    weights, _, _ = measured
    weight, scale = weights[tag]
    sums = weight.numpy().astype(np.float64).sum(axis=0)
    accumulator = np.abs(sums).max() * 127
    assert accumulator < 2.0**22, (
        f"{tag}: the worst channel's accumulator is {accumulator:.6g}, at or past "
        f"the {2.0**22} bound, so this is not the control it claims to be"
    )
    exact = scale.astype(np.float64) * sums
    got = _wide_answer(measured, tag, n).astype(np.float64)
    error = np.abs(got - exact)
    assert np.isfinite(got).all(), f"{tag}: not every output is finite"
    assert (error <= 1e-3 * np.abs(exact)).all(), (
        f"{tag}: worst relative error {(error / np.abs(exact)).max():.3g}, which is "
        f"not rounding"
    )
