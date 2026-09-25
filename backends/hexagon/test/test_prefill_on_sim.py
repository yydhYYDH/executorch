# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The quantized prefill weight layout and kernel, measured on hexagon-sim.

This is the kernel layer: no blob, no delegate, no arena layout -- a runner
that hands the vendored kernels operands on their own and prints what comes
back. Two things are decided here that nothing on the host can decide:

* the int4 tile order. `htp_ops_weight_reorder_int4` is the vendored tree's own
  reorder into the layout the prefill kernels read, and the bytes it produces
  are compared with the bytes `pack_q4a16_prefill_weight` produces. The two HVX
  operations that reorder is built on are measured the same way, on an input
  whose bytes say where they came from, so the host's model of them is a
  measurement rather than a reading of a manual.
* the kernel's arithmetic, against an expectation that mentions no tile, no
  nibble and no scale position: `sum_k a[m, k] * w[k, n] * scale[n]`.

What is *not* here: the emitter's two blits and the command's parameters. Those
are decided by `test_blob_on_sim.py`, which runs whole blobs the emitter built
through both this runner's kernels and the host interpreter.

Every fixture is exact by construction -- the weights and activations are small
integers and the scales are powers of two -- so the comparison is bit equality
and not a tolerance that could hide a wrong layout inside a rounding error.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

import pytest  # noqa: E402

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    pack_q4a16_gemv_weight,
    pack_q4a16_prefill_weight,
)

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/prefill_runner.cpp"

#: The translation units the two measured entries live in, and nothing else:
#: the reorder and the matmul in `matmul_q4fp16.c`, the M <= 32 matmul in
#: `matmul_q4fp16_mle32.c`, the wrapper that dispatches between them, and the
#: four managers the kernels bring up.
_SOURCES = [
    "matmul_ops.cc",
    "ops/matmul_q4fp16.c",
    "ops/matmul_q4fp16_mle32.c",
    "worker_pool.cc",
    "vtcm_mgr.cc",
    "hmx_mgr.cc",
    "hmx_queue.cc",
    "power.cc",
]


# ---------------------------------------------------------------- the layouts


def _raw_int4(weight):
    """The `[oc][ic/2]` plane, even ic in the high nibble, the reorder reads."""
    k, n = weight.shape
    quantized = (np.clip(weight, -8, 7) + 8).astype(np.uint8)
    return (quantized[0::2, :].T * 16 + quantized[1::2, :].T).astype(np.uint8).tobytes()


def _pack_activation(activation):
    """`[m][k]` fp16 -> the `[k/64][m][64]` blocked layout the kernels DMA.

    The emitter does this with a blit, whose regions `test_blob_on_sim.py`
    checks; this builds the same bytes directly, so a disagreement about the
    activation layout separates the kernel from the blit that feeds it.
    """
    m, k = activation.shape
    return (
        np.ascontiguousarray(activation.reshape(m, k // 64, 64).transpose(1, 0, 2))
        .astype(np.float16)
        .tobytes()
    )


def _unpack_output(raw, m, n):
    """The kernel's blocked output buffer -> `[m][n]` fp16.

    The kernel writes one 64-channel pack per M rows; the emitter's repack blit
    does the same move on the DSP, and is checked in the blob suite.
    """
    npacks = (n + 63) // 64
    values = np.frombuffer(raw, dtype=np.float16).reshape(npacks, m, 64)
    return values.transpose(1, 0, 2).reshape(m, npacks * 64)[:, :n].copy()


# ------------------------------------------------------------------ fixtures


def _weight(k, n, seed):
    """A (k, n) int4 weight: small integers, so every product is exact."""
    return np.random.default_rng(seed).integers(-7, 8, size=(k, n)).astype(np.int32)


def _scale(n):
    """Per-channel scales that are powers of two, so the scaling is exact."""
    return np.array([2.0 ** (-4 + (i % 4)) for i in range(n)], dtype=np.float32)


def _pattern(m, k):
    """Small values, distinct in every (m, k), exact in fp16."""
    rows = np.arange(m, dtype=np.float32).reshape(m, 1)
    columns = ((np.arange(k) % 5) - 2).astype(np.float32).reshape(1, k)
    return (rows * 0.25 + columns).astype(np.float16)


def _one_hot(m, k, at):
    activation = np.zeros((m, k), dtype=np.float16)
    activation[at] = np.float16(1.0)
    return activation


def _case(tag, activation, weight, scale, np_chunk=2, bias=None, scale_blocks=1):
    m, k = activation.shape
    n = weight.shape[1]
    return {
        "tag": tag,
        "m": m,
        "k": k,
        "n": n,
        "np": np_chunk,
        "kp": k // 32,
        "scale_blocks": scale_blocks,
        "act": _pack_activation(activation),
        "weight": pack_q4a16_prefill_weight(weight, scale, k, n, scale_blocks),
        "bias": None if bias is None else bias.astype(np.float16).tobytes(),
        "a": activation,
        "w": weight,
        "scale": scale,
    }


def _matmul_cases():
    """One case per branch of the two kernels' walk.

    P0 and P1 are the same shape with different activations: a pattern where
    every (m, k) differs, and a one-hot, which is the activation that reads one
    k of the weight and so can see a permutation a sum over k cannot. P2 is
    three output tiles with the last one unpaired. P3 is M == 32, the boundary
    the M <= 32 dispatcher turns on. P4 is M == 40, which is the other kernel.
    P5 has a bias, the one thing the kernel adds itself. P6 is a single output
    tile -- N = 32 -- at K = 64, which is the shape `np_chunk == 1` gets wrong
    below. P7 is a wide shape whose output tail is ragged and whose M spans two
    activation bands.
    """
    w0, s0 = _weight(64, 64, 1), _scale(64)
    w2, s2 = _weight(128, 96, 2), _scale(96)
    w6, s6 = _weight(64, 32, 6), _scale(32)
    bias = np.array([(i % 7) - 3 for i in range(64)], dtype=np.float32)
    return [
        _case("P0", _pattern(3, 64), w0, s0),
        _case("P1", _one_hot(3, 64, (1, 5)), w0, s0),
        _case("P2", _pattern(5, 128), w2, s2),
        _case("P3", _pattern(32, 64), w0, s0),
        _case("P4", _pattern(40, 64), w0, s0),
        _case("P5", _pattern(3, 64), w0, s0, bias=bias),
        _case("P6", _pattern(3, 64), w6, s6),
        _case("P7", _pattern(64, 128), w2, s2),
        _case("BLOCK2", _one_hot(3, 128, (1, 65)), _weight(128, 64, 21), np.tile(_scale(64)[:, None], (1, 2)), scale_blocks=2),
    ]


def _one_chunk_case():
    """The same shape as P6 asking for one output chunk.

    The emitter never asks for this -- `_emit_quantized_prefill` passes two for
    every shape -- and this case exists to measure why. See
    `test_one_output_chunk_is_not_a_chunk_the_kernel_handles`.
    """
    return _case("CHUNK1", _pattern(3, 64), _weight(64, 32, 6), _scale(32), np_chunk=1)


def _reorder_cases():
    """The vendored reorder's own cases: its input, and the packer's answer."""
    out = []
    for tag, k, n, seed in (("R0", 64, 32, 11), ("R1", 128, 96, 12)):
        weight, scale = _weight(k, n, seed), _scale(n)
        out.append(
            {
                "tag": tag,
                "ic": k,
                "oc": n,
                "alpha_bytes": 4 * n,
                # The vendored entry reads its fp32 alphas from the byte after
                # the raw plane, so the two are one buffer.
                "raw": _raw_int4(weight) + scale.tobytes(),
                "expected": pack_q4a16_prefill_weight(weight, scale, k, n),
            }
        )
    return out


def _fixture_header(cases, reorders):
    lines = ["// Generated by test_prefill_on_sim.py from the emitter's own packer."]

    def array(name, data):
        body = ",".join(str(byte) for byte in data)
        lines.append(f"static const unsigned char {name}[] = {{{body}}};")

    for index, case in enumerate(reorders):
        array(f"kReorderSrc{index}", np.frombuffer(case["raw"], dtype=np.uint8))
    for index, case in enumerate(cases):
        array(f"kAct{index}", np.frombuffer(case["act"], dtype=np.uint8))
        array(f"kWeight{index}", np.frombuffer(case["weight"], dtype=np.uint8))
        if case["bias"] is not None:
            array(f"kBias{index}", np.frombuffer(case["bias"], dtype=np.uint8))

    for name, value in (
        ("kMaxOutputBytes", max((c["n"] + 63) // 64 * c["m"] * 128 for c in cases)),
        ("kMaxActivationBytes", max(len(c["act"]) for c in cases)),
        ("kMaxWeightBytes", max(len(c["weight"]) for c in cases)),
        ("kMaxReorderBytes", max([len(c["expected"]) for c in reorders] + [0])),
        (
            "kMaxBiasBytes",
            max([len(c["bias"] or b"") for c in cases] + [2 * 64]),
        ),
    ):
        lines.append(f"enum {{ {name} = {value} }};")

    lines.append("struct ReorderCase {")
    lines.append(
        "  const char *tag; int ic, oc, alpha_bytes, reorder_bytes;"
        " const unsigned char *raw; };"
    )
    lines.append("static const ReorderCase kReorderCases[] = {")
    for index, case in enumerate(reorders):
        lines.append(
            f'  {{"{case["tag"]}", {case["ic"]}, {case["oc"]}, '
            f'{case["alpha_bytes"]}, {len(case["expected"])}, kReorderSrc{index}}},'
        )
    lines.append("};")

    lines.append("struct PrefillCase {")
    lines.append(
        "  const char *tag; int m, k, n, mp, np, kp, scale_blocks, act_bytes, weight_bytes,"
        " bias_bytes, out_bytes; const unsigned char *act, *weight, *bias; };"
    )
    lines.append("static const PrefillCase kCases[] = {")
    for index, case in enumerate(cases):
        bias = f"kBias{index}" if case["bias"] is not None else "0"
        lines.append(
            f'  {{"{case["tag"]}", {case["m"]}, {case["k"]}, {case["n"]}, 1, '
            f'{case["np"]}, {case["kp"]}, {case["scale_blocks"]}, {len(case["act"])}, '
            f'{len(case["weight"])}, {len(case["bias"] or b"")}, '
            f'{(case["n"] + 63) // 64 * case["m"] * 128}, kAct{index}, '
            f"kWeight{index}, {bias}}},"
        )
    lines.append("};")
    return "\n".join(lines) + "\n"


def _returned(measured, tag):
    """The value the runner printed for `<tag>_RET`: a named result per line,
    so the parser hands back the one-word list the printf produced."""
    return measured[f"{tag}_RET"][0]


def _halfwords(values):
    """The runner prints little-endian halfwords; rebuild the byte stream."""
    return b"".join(bytes((value & 0xFF, value >> 8)) for value in values)


@pytest.fixture(scope="module")
def measured():
    cases, reorders = _matmul_cases(), _reorder_cases()
    try:
        return hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={
                "prefill_data.h": _fixture_header(cases + [_one_chunk_case()], reorders)
            },
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def cases():
    return _matmul_cases()


# ------------------------------------------------------------------- the tests


def test_the_two_hvx_operations_do_what_the_layout_model_says(measured):
    """The reorder's two byte operations, measured on an input that names itself.

    `Q6_Vb_vshuff_Vb` and `Q6_Vh_vasl_VhR` are what the host's port of the
    vendored reorder is built on, and neither the shuffle's pairing nor the
    shift's halfword width can be read off a result that would also come out
    right if they were the other way round. The runner feeds the shuffle the
    byte values 0..127 and the shift the odd byte values, so each result byte
    says which input byte it came from.
    """
    shuffled = measured["SHUFF"]
    assert len(shuffled) == 128
    # Out[2i] = in[i], out[2i + 1] = in[64 + i]: the two halves interleaved.
    assert shuffled[0::2] == list(range(64))
    assert shuffled[1::2] == list(range(64, 128))

    shifted = measured["SHIFT"]
    assert len(shifted) == 128
    # In is `(2i + 1) & 0xff`, so halfword j holds (4j + 1, 4j + 3) and a
    # halfword shift of four leaves `(4j + 1) << 4` in the byte that was low.
    for j in range(64):
        low, high = 4 * j + 1, 4 * j + 3
        assert shifted[2 * j] == (low << 4) & 0xFF
        assert shifted[2 * j + 1] == ((low >> 4) | ((high << 4) & 0xF0)) & 0xFF


def test_the_packers_bytes_are_the_vendored_reorders_bytes(measured):
    """The emitter's packer against the tree's own reorder into this layout.

    `htp_ops_weight_reorder_int4` is the vendored implementation of the int4
    tile order the prefill kernels read, and this compares its output with
    `pack_q4a16_prefill_weight` byte for byte -- tiles and the fp16 scales that
    follow them. A packer that disagreed with it anywhere would have been
    checked against nothing: the kernel would read the wrong weight, and the
    only symptom on a device is an answer that is merely wrong.
    """
    for case in _reorder_cases():
        measured_bytes = _halfwords(measured[f'{case["tag"]}_BYTES'])
        assert (
            _returned(measured, case["tag"]) == 0
        ), f'{case["tag"]}: the vendored reorder refused'
        assert len(measured_bytes) == len(case["expected"])
        if measured_bytes != case["expected"]:
            differing = [
                at
                for at in range(len(case["expected"]))
                if measured_bytes[at] != case["expected"][at]
            ]
            raise AssertionError(
                f'{case["tag"]}: the packer and the vendored reorder differ at '
                f"{len(differing)} of {len(case['expected'])} bytes, first at "
                f"{differing[:12]}"
            )


def test_the_prefill_layout_is_not_the_gemv_one():
    """The control for the comparison above: the two layouts are not the same.

    Two packers that produced the same bytes would make the reorder comparison
    pass whatever either of them did. The GEMV entry reads a different layout
    for the same weight -- the vrmpy one -- so its bytes have to differ, and on
    a shape where both are defined.
    """
    weight, scale = _weight(128, 64, 3), _scale(64)
    prefill = pack_q4a16_prefill_weight(weight, scale, 128, 64)
    gemv = pack_q4a16_gemv_weight(weight, scale, 128, 64)
    tiles = 4 * 2 * 512
    # The difference is in the tiles and not only in the scales that follow them:
    # the GEMV weight carries fp32 scales and this one fp16, so the tails differ
    # by construction and a comparison of them alone would say nothing.
    assert prefill[:tiles] != gemv[:tiles], "the two tile orders agree"


def test_the_kernel_computes_the_layout_blind_arithmetic(measured, cases):
    """Every fixture against `a @ (w * scale)`, bit for bit.

    The expectation is built from the weight and the activation the fixture
    chose, not from the bytes either of them were packed into, so it says
    nothing about a tile, a nibble, a scale position or the order the rows come
    back in. It is exact because the weights and activations are small integers
    and the scales are powers of two: every product and partial sum is an
    integer below 2048, which fp16 holds exactly, and a bias of small integers
    keeps it there.
    """
    for case in cases:
        tag = case["tag"]
        k = case["k"]
        assert _returned(measured, tag) == 0, f"{tag}: the kernel refused"
        got = _unpack_output(_halfwords(measured[f"{tag}_OUT"]), case["m"], case["n"])
        scales = (
            case["scale"]
            if case["scale_blocks"] == 1
            else np.take(case["scale"].reshape(-1, case["scale_blocks"]), ((np.arange(k) // 32) * case["scale_blocks"]) // (k // 32), axis=1).T
        )
        want = (
            case["a"].astype(np.float32)
            @ (case["w"].astype(np.float32) * scales)
        ).astype(np.float16)
        if case["bias"] is not None:
            want = (
                want.astype(np.float32)
                + np.frombuffer(case["bias"], dtype=np.float16).astype(np.float32)
            ).astype(np.float16)
        if not np.array_equal(got, want):
            wrong = np.argwhere(got != want)
            raise AssertionError(
                f"{tag}: {len(wrong)} of {got.size} elements differ, first at "
                f"{wrong[:6].tolist()}\n  got  {got.reshape(-1)[:8]}"
                f"\n  want {want.reshape(-1)[:8]}"
            )


def test_one_output_chunk_is_not_a_chunk_the_kernel_handles(measured):
    """`np_chunk == 1` at K = 64 answers NaNs, and the emitter never asks for it.

    Measured, not deduced: with the same weight, activation and walk, this entry
    answers the right numbers at every width for `np_chunk == 2` and NaNs at
    K == 64 for `np_chunk == 1` -- at N = 32 and at N = 64 alike, so it is the
    chunk count and the width, not the ragged tile. The M > 32 kernel and the
    general one were checked the same way. That is why the emitter passes two
    chunks for every shape (`_emit_quantized_prefill`), including the ones with
    one output tile of 32 channels, where the walk is one chunk of work either
    way. This case is that measurement: it is the shape and chunk count the
    first draft of the emitter would have passed, and the exact bytes below are
    what that draft would have shipped.
    """
    broken = _one_chunk_case()
    got = _unpack_output(_halfwords(measured["CHUNK1_OUT"]), 3, 32)
    want = (
        broken["a"].astype(np.float32)
        @ broken["w"].astype(np.float32)
        * broken["scale"].reshape(1, -1)
    ).astype(np.float16)
    assert not np.array_equal(got, want), (
        "one output chunk answered the right numbers, so this is no longer the "
        "reason the emitter asks for two"
    )
    # And it is a NaN answer rather than a wrong one, which is what makes it
    # worth an emitter rule: a wrong finite answer is harder to notice.
    assert np.isnan(got).any(), f"the one-chunk answer was {got.reshape(-1)[:8]}"


def test_the_fixtures_can_tell_a_k_permutation_apart(measured, cases):
    """The control for the comparison above, on the shape that can see one.

    A weight whose k values are permuted answers `sum` the same way when the
    activation is flat, which is why the one-hot case exists: it reads a single
    k per output channel, and P1's expectation is therefore sensitive to where
    each k sits. If the permuted expectation matched the measured output, the
    two would not be testing the layout at all.
    """
    case = next(case for case in cases if case["tag"] == "P1")
    m, k, n = case["m"], case["k"], case["n"]
    measured_out = _unpack_output(_halfwords(measured["P1_OUT"]), m, n)
    permuted = case["w"][np.arange(k) ^ 1]
    want = (
        case["a"].astype(np.float32)
        @ permuted.astype(np.float32)
        * case["scale"].reshape(1, -1)
    ).astype(np.float16)
    assert not np.array_equal(measured_out, want), (
        "the permuted weight answers what the kernel did, so no k permutation "
        "can be seen and the fixtures do not test the layout"
    )
