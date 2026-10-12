import json

import numpy as np


def make_cases():
    cases = []
    for test in range(12):
        state = 1234567 + test
        av, wv = [], []
        for i in range(32 * 128):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            width = 9 if test == 1 else 257 if test == 2 else 1201
            value = 32768 + (state >> 16) % width - width // 2
            if test == 0:
                value = 32768
            elif test == 4:
                value = state >> 16
            elif test in (6, 7, 11):
                value = 65535
            elif test == 8:
                value = 32768 + (1024 if i % 128 == 111 else 0)
            av.append(value)
        for i in range(128 * 64):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            value = (state >> 16) % 16 - 8
            if test == 5:
                value = 0
            elif test in (6, 7):
                value = -8 if test == 6 else 7
            elif test == 8:
                value = -8 if i // 64 == 111 and i % 64 == 49 else 0
            elif test == 11:
                value = 7 if i // 64 < 32 else -7 if i // 64 < 64 else 0
            wv.append(value)
        scales = np.empty((64, 4), dtype="<f4")
        for n in range(64):
            for block in range(4):
                maximum = (16 + n % 16) / 1024
                ratio = 16 if block == 3 else 1 + (n * 3 + block * 5) % 16
                if test == 10:
                    ratio = 16 if block == 3 else 0.5 + (n + block) % 15
                scales[n, block] = maximum * ratio / 16
                if test == 9 and n % 3 == 0 and block < 3:
                    scales[n, block] = 0
                if test == 11:
                    scales[n, block] = maximum
        cases.append(
            (
                np.array(av, dtype="<u2").reshape(32, 128),
                np.array(wv, dtype="i1").reshape(128, 64),
                scales,
                np.float32(1 / 4096),
                np.float32(1 / 8192),
                0 if test == 11 else 32768,
                32768 if test == 4 else 65500 if test == 9 else 29520,
                np.float32(0),
            )
        )
    return cases


def reference(case):
    a, w, scales, sa, sq, za, zq, global_weight_scale = case
    maximum = scales.astype(np.float64).max(axis=1)
    ratio = np.divide(
        scales,
        (maximum / 16).astype(np.float32)[:, None],
        out=np.zeros_like(scales),
        where=maximum[:, None] != 0,
    )
    multiples = np.where(
        maximum[:, None] != 0, np.clip(np.rint(ratio), 1, 16), 0
    ).astype(np.int64)
    weights = w.astype(np.int64) * np.repeat(multiples, 32, axis=1).T
    channel_scale = (maximum / 16).astype(np.float32)
    if global_weight_scale == 0:
        global_weight_scale = channel_scale.max()
    normalized = channel_scale / global_weight_scale
    global_gain = np.float32(np.float32(global_weight_scale * sa) / sq)
    gain = normalized * global_gain
    bits = gain.view(np.uint32) + 2
    exponent = ((bits >> 23) & 255).astype(np.int64) - 127
    shift = np.where(gain != 0, exponent + 24, 15)
    mult = np.where(gain != 0, ((bits >> 2) & 0x1FFFFF) | 0x200000, 0).astype(np.int64)
    correction = (-za * weights.sum(axis=0) + 127) // 256
    acc = (a.astype(np.int64) @ weights) // 256 + correction
    value = (
        (acc * np.exp2(shift - 7).astype(np.int64) + (acc != 0)) * mult
        + (zq * 16 + 8) * (1 << 26)
    ) // (1 << 30)
    q = np.clip(value, 0, 65535).astype("<u2")
    y = ((q.astype(np.int32) - zq).astype(np.float32) * sq).astype("<f4")
    encoded = mult ^ 0x200000
    bias = zq * 16 + 8
    words0 = (
        ((encoded & 0x7FF) >> 1)
        | ((encoded & 1) << 31)
        | (shift << 10)
        | ((bias & 0xFFF) << 19)
    )
    words1 = (
        ((encoded >> 11) & 0x3FF)
        | (((encoded >> 21) & 1) << 16)
        | ((bias & 0xFF000) << 11)
    )
    table = np.zeros(256, dtype="<u4")
    for n in range(64):
        index = ((n // 32) * 2 + n % 2) * 64 + (n % 32 // 2) * 2
        table[index : index + 2] = [words0[n], words1[n]]
        table[index + 32] = int(correction[n]) & 0xFFFFFFFF
    return q, y, table


def load_qnn_fixture(directory):
    # Logical export tensors; no HMX buffers or conversion words are read here.
    metadata = (
        json.loads((directory / "logical.json").read_text())
        if (directory / "logical.json").exists()
        else {
            "shape_mkn": [128, 128, 128],
            "sa": 1.4731697774550412e-05,
            "sq": 2.2494559743790887e-05,
            "za": 33401,
            "zq": 29520,
        }
    )
    rows, depth, cols = metadata["shape_mkn"]
    assert rows % 32 == cols % 64 == 0 and depth == 128
    w = np.load(directory / "Wq.npy").astype("i1").T.copy()
    scales = np.load(directory / "sc.npy").astype("<f4")
    sa, sq = np.float32(metadata["sa"]), np.float32(metadata["sq"])
    za, zq = metadata["za"], metadata["zq"]
    x = np.fromfile(directory / "input.f32", dtype="<f4").reshape(rows, depth)
    a = np.clip(np.rint(x.astype(np.float64) / float(sa)) + za, 0, 65535).astype("<u2")
    y = np.fromfile(directory / "qnn_before.raw", dtype="<f4").reshape(rows, cols)
    q = np.rint(y.astype(np.float64) / float(sq) + zq).astype("<u2")
    assert w.shape == (depth, cols) and scales.shape == (cols, 4)
    cases, outputs = [], []
    for r in range(0, rows, 32):
        for n in range(0, cols, 64):
            cases.append(
                (
                    a[r : r + 32].copy(),
                    w[:, n : n + 64].copy(),
                    scales[n : n + 64].copy(),
                    sa,
                    sq,
                    za,
                    zq,
                    np.float32(scales.max() / 16),
                )
            )
            outputs.append(
                (q[r : r + 32, n : n + 64].copy(), y[r : r + 32, n : n + 64].copy())
            )
    return cases, outputs
