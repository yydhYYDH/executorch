import struct

import numpy as np


def make_cases():
    cases = []
    rng = np.random.default_rng(20261012)
    for i in range(40):
        za = [0, 1, 16384, 32768, 33401, 65534, 65535][i % 7]
        zq = [0, 1, 29520, 32768, 65534, 65535][i % 6]
        zw = [0, 1, -3, 7][i % 4]
        sa = np.float32(2 ** rng.uniform(-14, -5))
        sq = np.float32(2 ** rng.uniform(-10, -3))
        gains = np.exp2(rng.uniform(-15, -3, 64)).astype(np.float32)
        sw = (gains * sq / sa).astype(np.float32)
        r, k = np.indices((32, 32))
        a = np.clip(za + (r * 71 + k * 113 + r * k) % 4097 - 2048, 0, 65535).astype(
            np.uint16
        )
        w = rng.integers(-2, 2, (32, 64), dtype=np.int8)
        w[:, 0] = 0
        w[:, 1] = 1
        w[:, 2] = -2
        w[:, 3] = 0
        w[0, 3] = 1
        cases.append((a, (w + zw).astype(np.int8), sw, sa, sq, za, zw, zq))
    return cases


def prepare(case):
    a, raw, sw, sa, sq, za, zw, zq = case
    w = raw.astype(np.int64) - zw
    gs = np.max(sw)
    base = np.float32(np.float32(gs * np.float32(sa)) / np.float32(sq))
    gains = np.float32(np.float32(sw / gs) * base)
    bits = gains.view(np.uint32) + np.uint32(2)
    mantissa = ((bits >> 2) & 0x1FFFFF) | 0x200000
    shift = ((bits >> 23) & 255).astype(np.int32) - 127 + 24
    assert np.all((shift >= 8) & (shift <= 22))
    em = mantissa ^ 0x200000
    bias = 16 * zq + 8
    first = ((em & 0x7FF) >> 1) | ((em & 1) << 31)
    first |= (shift.astype(np.uint32) << 10) | ((bias & 0xFFF) << 19)
    second = ((em >> 11) & 0x3FF) | ((bias & 0xFF000) << 11)
    num = za * w.sum(axis=0)
    corr = -np.sign(num) * ((np.abs(num) + 128) // 256)
    table = np.zeros(256, dtype=np.uint32)
    for n in range(64):
        idx = ((n // 32) * 2 + n % 2) * 64 + (n % 32 // 2) * 2
        table[idx : idx + 2] = [first[n], second[n]]
        table[idx + 32] = int(corr[n]) & 0xFFFFFFFF
    wt = np.empty(512, dtype=np.uint8)
    for k in range(0, 32, 16):
        for n in range(64):
            for i in range(4):
                idx = (n // 32) * 256 + (k // 16) * 128 + (n % 32) * 4 + i
                wt[idx] = sum(
                    (int(w[k + i + 4 * s, n]) & 3) << (2 * s) for s in range(4)
                )
    return wt, table, mantissa.astype(np.int64), shift, corr


def reference(case):
    a, raw, sw, sa, sq, za, zw, zq = case
    _, _, mantissa, shift, corr = prepare(case)
    t = (a.astype(np.int64) @ (raw.astype(np.int64) - zw)) // 256 + corr
    q = (
        (t * np.left_shift(np.int64(1), shift - 7) + (t != 0)) * mantissa
        + (16 * zq + 8) * (1 << 26)
    ) // (1 << 30)
    q = np.clip(q, 0, 65535).astype(np.uint16)
    y = (q.astype(np.int32) - zq).astype(np.float32) * np.float32(sq)
    return q, y


def load_qnn_fixture(path):
    import json

    meta = json.loads((path / "logical.json").read_text())
    a = np.load(path / "Aq.npy")
    w = np.load(path / "Wq.npy")
    sw = np.load(path / "sw.npy")
    return (
        a,
        w,
        sw,
        np.float32(meta["sa"]),
        np.float32(meta["sq"]),
        meta["za"],
        0,
        meta["zq"],
    )
