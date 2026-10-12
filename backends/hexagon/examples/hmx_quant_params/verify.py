import json
from pathlib import Path
import re

import numpy as np


def coefficient(case):
    sa, sw, sq, za, zq, d = case
    product = np.float32(np.float32(sa) * np.float32(sw))
    gain = np.float32(product / np.float32(sq))
    bits = int(gain.view(np.uint32)) + 2
    shift = ((bits >> 23) & 255) - 127 + 16 + d
    assert np.isfinite(gain) and gain > 0 and 8 <= shift <= 22
    mult = ((bits >> 2) & 0x1FFFFF) | 0x200000
    encoded, bias = mult ^ 0x200000, 16 * zq + 8
    words = [
        ((encoded & 0x7FF) >> 1)
        | ((encoded & 1) << 31)
        | (shift << 10)
        | ((bias & 0xFFF) << 19),
        ((encoded >> 11) & 0x3FF)
        | (((encoded >> 21) & 1) << 16)
        | ((bias & 0xFF000) << 11),
    ]
    return np.array(words, dtype="<u4"), mult, shift


def make_cases():
    cases = []
    for d in (8, 16):
        for i in range(16):
            sa = np.float32(0.00317 * (1 + i % 3))
            sq = np.float32(0.02191 * (1 + i % 5))
            gain = 2.0 ** ((8 + i % 15) - 16 - d)
            sw = np.float32(gain * float(sq) / float(sa))
            za = (0, 1, 16384, 33401, 65534, 65535)[i % 6]
            zq = (0, 1, 29520, 32768, 65534, 65535)[i % 6]
            cases.append((sa, sw, sq, za, zq, d))
    return cases


def reference(case):
    _, _, _, za, zq, d = case
    r, k = np.indices((32, 32), dtype=np.int64)
    wk, n = np.indices((32, 64), dtype=np.int64)
    a = np.clip(za + (r * 71 + k * 113 + r * k) % 4097 - 2048, 0, 65535)
    low = (wk * 11 + n * 19 + wk * n) % 256 - 128
    high = (wk * 17 + n * 7 + wk * n) % 63 - 31
    w = low if d == 8 else high * 256 + low
    numerator = za * w.sum(axis=0)
    correction = -np.sign(numerator) * ((abs(numerator) + (1 << (d - 1))) // (1 << d))
    t = (a @ w) // (1 << d) + correction
    _, mult, shift = coefficient(case)
    q = ((t * (1 << (shift - 7)) + (t != 0)) * mult + (16 * zq + 8) * (1 << 26)) // (
        1 << 30
    )
    return np.clip(q, 0, 65535).astype("<u2")


def captured_cases(directory: Path):
    cases, pairs, tags = [], [], []
    seen = set()
    for file in sorted(directory.glob("capture_*capture.json")):
        if "hhh" not in file.name and "hbh" not in file.name:
            continue
        for tag, entry in json.loads(file.read_text()).items():
            if entry["quant_dtype"] not in ("use_16a8w", "use_16a16w") or tag in seen:
                continue
            qdq = directory / (tag + ".qdq.txt")
            if not qdq.exists():
                continue
            matches = re.findall(
                r"dequantize_per_tensor.default\]\(args = \([^,]+, ([^,]+), ([^,]+),",
                qdq.read_text(),
            )
            assert len(matches) == 3, tag
            (sw, _), (sa, za), (sq, zq) = [
                (np.float32(float(scale)), int(zero)) for scale, zero in matches
            ]
            d = 8 if entry["quant_dtype"] == "use_16a8w" else 16
            kind = next(iter(entry["kinds"].values()))
            # The saved Q4B window starts at the first effective bank read.
            pair = np.frombuffer(bytes.fromhex(kind["Q4B"]), "<u4")[:2].copy()
            cases.append((sa, sw, sq, za, zq, d))
            pairs.append(pair)
            tags.append(tag)
            seen.add(tag)
    assert cases, directory
    return cases, pairs, tags
