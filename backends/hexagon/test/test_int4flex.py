# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
from executorch.backends.hexagon.hexagon_ops import (
    pack_q4a16_gemv_weight,
    pack_q4a16_prefill_weight,
    pack_w8a16_gemv_weight,
    pack_w8a16_prefill_weight,
)


def test_block_scale_tail_is_dsp_tile_records():
    w = np.arange(64 * 32, dtype=np.int32).reshape(64, 32) % 16 - 8
    scales = np.stack([np.full(32, 0.25), np.full(32, 0.5)], axis=1)
    packed = pack_q4a16_prefill_weight(w, scales, 64, 32, 2)
    tail = np.frombuffer(packed[2 * 512 :], dtype=np.float16)
    assert tail.size == 2 * 64
    assert np.array_equal(tail.reshape(1, 2, 64)[:, :, 0::2], np.array([[[0.25] * 32, [0.5] * 32]], dtype=np.float16))
    assert np.array_equal(tail.reshape(1, 2, 64)[:, :, 1::2], np.array([[[0.25] * 32, [0.5] * 32]], dtype=np.float16))


def test_int4_overflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = 8
    with pytest.raises(RuntimeError, match="outside"):
        pack_q4a16_prefill_weight(w, np.ones(32), 64, 32)


def test_int4_underflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = -9
    with pytest.raises(RuntimeError, match="outside"):
        pack_q4a16_prefill_weight(w, np.ones(32), 64, 32)


def test_int4_gemv_overflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = 8
    with pytest.raises(RuntimeError, match="q4a16 weight contains values outside"):
        pack_q4a16_gemv_weight(w, np.ones(32), 64, 32)


def test_int4_gemv_underflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = -9
    with pytest.raises(RuntimeError, match="q4a16 weight contains values outside"):
        pack_q4a16_gemv_weight(w, np.ones(32), 64, 32)


def test_int8_gemv_overflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = 128
    with pytest.raises(RuntimeError, match="w8a16 weight contains values outside"):
        pack_w8a16_gemv_weight(w, 64, 32)


def test_int8_gemv_underflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = -129
    with pytest.raises(RuntimeError, match="w8a16 weight contains values outside"):
        pack_w8a16_gemv_weight(w, 64, 32)


def test_int8_prefill_overflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = 128
    with pytest.raises(RuntimeError, match="w8a16 weight contains values outside"):
        pack_w8a16_prefill_weight(w, np.ones(32), 64, 32)


def test_int8_prefill_underflow_is_not_silent():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = -129
    with pytest.raises(RuntimeError, match="w8a16 weight contains values outside"):
        pack_w8a16_prefill_weight(w, np.ones(32), 64, 32)


def test_range_guard_runs_before_the_clip(monkeypatch):
    real_clip = np.clip
    calls = []

    def clip(*args, **kwargs):
        calls.append(args[1:3])
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(np, "clip", clip)
    k, n = 64, 32
    in_range = np.zeros((k, n), dtype=np.int32)
    int4 = in_range.copy()
    int4[0, 0] = 8
    int8 = in_range.copy()
    int8[0, 0] = 128
    for pack, weight, scale in (
        (pack_q4a16_gemv_weight, int4, np.ones(n)),
        (pack_q4a16_prefill_weight, int4, np.ones(n)),
        (pack_w8a16_gemv_weight, int8, None),
        (pack_w8a16_prefill_weight, int8, np.ones(n)),
    ):
        before = len(calls)
        args = (weight, k, n) if scale is None else (weight, scale, k, n)
        with pytest.raises(RuntimeError, match="weight contains values outside"):
            pack(*args)
        assert len(calls) == before, "the clip ran before the range guard rejected"
    # The clip stays active for in-range weights, so the counts above are not zero
    # only because the replacement is dead code.
    pack_q4a16_gemv_weight(in_range, np.ones(n), k, n)
    pack_w8a16_gemv_weight(in_range, k, n)
    assert len(calls) == 2
    assert calls[0] == (-8, 7)
    assert calls[1] == (-128, 127)


def test_range_guard_runs_before_the_scale_check():
    w = np.zeros((64, 32), dtype=np.int32)
    w[0, 0] = 8
    with pytest.raises(RuntimeError, match="q4a16 weight contains values outside"):
        pack_q4a16_gemv_weight(w, np.ones(31), 64, 32)
    with pytest.raises(RuntimeError, match="q4a16 weight contains values outside"):
        pack_q4a16_prefill_weight(w, np.ones(31), 64, 32)
    w[0, 0] = 128
    with pytest.raises(RuntimeError, match="w8a16 weight contains values outside"):
        pack_w8a16_prefill_weight(w, np.ones(31), 64, 32)
