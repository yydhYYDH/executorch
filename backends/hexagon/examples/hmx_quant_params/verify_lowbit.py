import struct

import numpy as np

from verify import coefficient as u16_coefficient


def coefficient(case):
    sa, sw, sq, za, zw, zq, kind = case
    if kind == 2:
        return u16_coefficient((sa, sw, sq, za, zq, 8))
    gain = np.float32(np.float32(np.float32(sa) * np.float32(sw)) / np.float32(sq))
    bits = struct.unpack("<I", struct.pack("<f", gain))[0] + 0x800
    shift = ((bits >> 23) & 255) - 127 + 24 - (4 if kind == 1 else 0)
    mantissa = ((bits >> 12) & 2047) | 2048
    word = ((mantissa >> 1) & 1023) | ((mantissa & 1) << 31)
    word |= (shift << 10) | (zq << 23) | (1 << 22)
    return np.array([word, 0], np.uint32), mantissa, shift


def inputs(case):
    _, _, _, za, zw, _, kind = case
    m, n = (32, 64) if kind == 2 else (64, 128 if kind == 1 else 64)
    r, k = np.indices((m, 32))
    span = 4097 if kind == 2 else 257
    a = np.clip(
        za + (r * 71 + k * 113 + r * k) % span - span // 2,
        0,
        65535 if kind == 2 else 255,
    ).astype(np.uint16)
    k, nindex = np.indices((32, n))
    # The first columns isolate zero/tie cases; the rest exercise every weight.
    width = 256 if kind == 0 and zw == 0 else 127 if kind == 0 else 16
    w = (k * 11 + nindex * 19 + k * nindex) % width
    w -= width // 2
    w[:, 0] = 0
    w[:, 1] = 0
    w[0, 1] = 1
    w[:, 2] = 0
    w[0, 2] = -1
    return a, (w + zw).astype(np.int8)


def reference(case):
    _, _, sq, za, zw, zq, kind = case
    a, raw = inputs(case)
    w = raw.astype(np.int64) - zw
    _, mantissa, shift = coefficient(case)
    if kind == 2:
        num = za * w.sum(axis=0)
        corr = -np.sign(num) * ((np.abs(num) + 128) // 256)
        t = (a.astype(np.int64) @ w) // 256 + corr
        value = (
            (t * (1 << (shift - 7)) + (t != 0)) * mantissa + (16 * zq + 8) * (1 << 26)
        ) // (1 << 30)
    else:
        t = (a.astype(np.int64) - za) @ w
        t *= 16 if kind == 1 else 1
        shifted = t * (1 << shift)
        intermediate = shifted // (1 << 19)
        intermediate |= (shifted // (1 << 20) != 0).astype(np.int64)
        value = (intermediate * mantissa + (16 * zq + 8) * (1 << 12)) // (1 << 16)
    q = np.clip(value, 0, 65535 if kind == 2 else 255).astype(np.uint16)
    y = (q.astype(np.int32) - zq).astype(np.float32) * np.float32(sq)
    return q, y


def make_cases():
    cases = []
    for i in range(24):
        for kind in range(3):
            if kind == 2:
                za = [0, 1, 16384, 32768, 33401, 65534, 65535][i % 7]
                zq = [0, 1, 29520, 32768, 65534, 65535][i % 6]
                gain = np.float32([1 / 256, 0.01371, 1 / 65536, 0.25][i % 4])
            else:
                za = [0, 1, 64, 128, 130, 254, 255][i % 7]
                zq = [0, 1, 64, 128, 115, 254, 255][(i * 3) % 7]
                gain = np.float32([1.0, 0.0029766483, 1 / 65536, 0.17319][i % 4])
            sa = np.float32([0.125, 0.0037860463, 0.00713][i % 3])
            sw = np.float32([0.25, 0.0045451913, 0.01973][i % 3])
            sq = np.float32(np.float32(sa * sw) / gain)
            zw = [0, 1, -3, 7][i % 4]
            cases.append((float(sa), float(sw), float(sq), za, zw, zq, kind))
    rng = np.random.default_rng(20261012)
    for i in range(64):
        for kind in range(3):
            gain = np.float32(2 ** rng.uniform(-16, -2 if kind == 2 else 2))
            sa = np.float32(2 ** rng.uniform(-14, -3))
            sw = np.float32(2 ** rng.uniform(-10, -1))
            sq = np.float32(np.float32(sa * sw) / gain)
            maximum = 65535 if kind == 2 else 255
            cases.append(
                (
                    float(sa),
                    float(sw),
                    float(sq),
                    int(rng.integers(0, maximum + 1)),
                    int(rng.integers(-8, 9)),
                    int(rng.integers(0, maximum + 1)),
                    kind,
                )
            )
    for kind in range(3):
        for shift in [8, 9, 21, 22] if kind == 2 else [0, 1, 30, 31]:
            gain = 2.0 ** (shift - (24 if kind == 2 else 20 if kind == 1 else 24))
            cases.append(
                (
                    0.125,
                    0.25,
                    0.03125 / gain,
                    32768 if kind == 2 else 128,
                    0,
                    32768 if kind == 2 else 128,
                    kind,
                )
            )
    return cases
