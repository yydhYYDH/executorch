import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def make_cases():
    rng = np.random.default_rng(20261013)
    cases = []
    for i in range(20):
        count = [0, 1, 2, 3, 31, 32, 33, 257][i % 8]
        s8, s4 = [(0.125, 0.5), (0.03, 0.17), (1, 1), (0.005, 0.009), (10, 100)][i % 5]
        z8 = [0, -128, 127, -7, 35][i % 5]
        z4 = [0, -8, 7, -3, 2][i % 5]
        x = rng.normal(0, 17, count).astype(np.float32)
        edges = np.array(
            [
                0,
                -0.0,
                np.inf,
                -np.inf,
                65504,
                65519,
                65520,
                -65520,
                2**-24,
                -(2**-24),
                2**-25,
                -(2**-25),
                -2.5 * s8,
                -1.5 * s8,
                -0.5 * s8,
                0.5 * s8,
                1.5 * s8,
                2.5 * s8,
                (-128 - z8) * s8,
                (127 - z8) * s8,
                (-8 - z4) * s4,
                (7 - z4) * s4,
            ],
            np.float32,
        )
        x[: min(count, len(edges))] = edges[:count]
        cases.append((x, np.float32(s8), np.float32(s4), z8, z4))
    return cases


def write_fixture(path, cases):
    data = bytearray(struct.pack("<I", len(cases)))
    for x, s8, s4, z8, z4 in cases:
        data.extend(struct.pack("<Iffii", len(x), s8, s4, z8, z4))
        data.extend(x.astype("<f4").tobytes())
    path.write_bytes(data)


def quantize(x, bits, scale, zero):
    return np.clip(
        np.rint(np.float32(np.float32(x / scale) + np.float32(zero))),
        -(1 << (bits - 1)),
        (1 << (bits - 1)) - 1,
    ).astype(np.int8)


def check_output(path, cases):
    data = path.read_bytes()
    offset = 0
    checks = []

    def take(dtype, count):
        nonlocal offset
        out = np.frombuffer(data, dtype, count, offset)
        offset += out.nbytes
        return out

    for i, (x, s8, s4, z8, z4) in enumerate(cases):
        q8 = quantize(x, 8, s8, z8)
        q4 = quantize(x, 4, s4, z4)
        packed = np.zeros((len(x) + 1) // 2, np.uint8)
        packed[:] = q4[::2].astype(np.uint8) & 15
        packed[: len(x) // 2] |= (q4[1::2].astype(np.uint8) & 15) << 4
        f8 = (q8.astype(np.float32) - z8) * s8
        f4 = (q4.astype(np.float32) - z4) * s4
        with np.errstate(over="ignore"):
            h = x.astype(np.float16)
        expected = [
            q8,
            q4,
            packed,
            q4,
            f8,
            f4,
            quantize(f8, 4, s4, z4),
            quantize(f4, 8, s8, z8),
            h.view(np.uint16),
            h.astype(np.float32),
            f8.astype(np.float16).view(np.uint16),
            f4.astype(np.float16).view(np.uint16),
            quantize(h.astype(np.float32), 8, s8, z8),
            quantize(h.astype(np.float32), 4, s4, z4),
        ]
        mismatches = {}
        for name, want in zip(
            [
                "q8",
                "q4",
                "packed",
                "unpacked",
                "f8",
                "f4",
                "int8_to_int4",
                "int4_to_int8",
                "half",
                "widened",
                "int8_to_half",
                "int4_to_half",
                "half_to_int8",
                "half_to_int4",
            ],
            expected,
        ):
            got = take(want.dtype, len(want))
            # Bitwise checks include signed zero and infinities.
            mismatches[name] = int(
                np.count_nonzero(got.view(np.uint8) != want.view(np.uint8))
            )
        assert not any(mismatches.values()), (i, mismatches)
        checks.append({"case": i, "count": len(x), "byte_mismatches": mismatches})
    records = np.frombuffer(
        data,
        dtype=np.dtype([("f", "<f4", (4096,)), ("h", "<u2", (4096,))]),
        count=16,
        offset=offset,
    )
    offset += records.nbytes
    assert offset == len(data), (offset, len(data))
    got_f = records["f"].reshape(-1)
    got_h = records["h"].reshape(-1)
    original = np.arange(65536, dtype=np.uint16)
    expected_f = original.view(np.float16).astype(np.float32)
    finite_or_inf = ~np.isnan(expected_f)
    assert np.array_equal(
        got_f[finite_or_inf].view(np.uint32), expected_f[finite_or_inf].view(np.uint32)
    )
    assert np.array_equal(got_h[finite_or_inf], original[finite_or_inf])
    assert np.isnan(got_f[~finite_or_inf]).all()
    assert np.isnan(got_h.view(np.float16)[~finite_or_inf]).all()
    return {
        "cases": checks,
        "fp16_patterns": 65536,
        "nan_payload_policy": "NaN class checked; payload preservation not required",
        "output_sha256": hashlib.sha256(data).hexdigest(),
    }


def check_rejections(library):
    lib = ctypes.CDLL(str(library.resolve()))
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int8)
    bp = ctypes.POINTER(ctypes.c_uint8)
    lib.quantize_sint.argtypes = [
        fp,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_float,
        ctypes.c_int,
        ip,
    ]
    lib.pack_int4.argtypes = [ip, ctypes.c_size_t, bp]
    lib.dequantize_sint.argtypes = [
        ip,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_float,
        ctypes.c_int,
        fp,
    ]
    x = np.array([1, 2, 3], np.float32)
    dst = np.full(3, 42, np.int8)
    params = [
        (4, 0, 0),
        (4, float("nan"), 0),
        (4, float("inf"), 0),
        (2, 1, 0),
        (4, 1, 8),
    ]
    for bits, scale, zero in params:
        assert (
            lib.quantize_sint(
                x.ctypes.data_as(fp), len(x), bits, scale, zero, dst.ctypes.data_as(ip)
            )
            == 1
        )
        assert np.all(dst == 42)
    x[-1] = np.nan
    assert (
        lib.quantize_sint(x.ctypes.data_as(fp), len(x), 8, 1, 0, dst.ctypes.data_as(ip))
        == 1
    )
    assert np.all(dst == 42)
    q = np.array([0, -8, 8], np.int8)
    packed = np.full(2, 0xA5, np.uint8)
    assert lib.pack_int4(q.ctypes.data_as(ip), len(q), packed.ctypes.data_as(bp)) == 1
    assert np.all(packed == 0xA5)
    f = np.full(3, 42, np.float32)
    assert (
        lib.dequantize_sint(q.ctypes.data_as(ip), len(q), 4, 1, 0, f.ctypes.data_as(fp))
        == 1
    )
    assert np.all(f == 42)
    return 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / ".tmp/scratch/mnv2/integer_float_conversion",
    )
    args = parser.parse_args()
    root = args.build_dir.resolve()
    cases = make_cases()
    write_fixture(root / "fixture.bin", cases)
    subprocess.run(
        [
            str(root / "conversion_host"),
            str(root / "fixture.bin"),
            str(root / "host_output.bin"),
        ],
        check=True,
    )
    evidence = check_output(root / "host_output.bin", cases)
    evidence["tier"] = "host"
    evidence["invalid_cases"] = check_rejections(root / "libconversion_host.so")
    (root / "host_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("Host: 20 cases, all 65,536 FP16 patterns and 8 invalid cases passed")


if __name__ == "__main__":
    main()
