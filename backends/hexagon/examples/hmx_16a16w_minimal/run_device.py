import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np


def main():
    """Run the integer HMX example on the phone and verify its complete outputs."""
    repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="oneplus13-reverse")
    parser.add_argument(
        "--sdk", default="/home/yydh/hexagon-tutorial/tools/hexagon-sdk"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=repo / ".tmp/scratch/mnv2/hmx_16a16w_minimal/example",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_16a16w_minimal/example",
    )
    args = parser.parse_args()
    build = args.build_dir.resolve()
    sdk = Path(args.sdk)
    remote = shlex.quote(args.remote_dir)
    ssh_options = ["-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1"]

    def ssh(command, timeout=20):
        return subprocess.run(
            ["ssh", *ssh_options, args.host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )

    def idle():
        while True:
            output = ssh("/system/bin/ps -A -o COMM").stdout.splitlines()
            assert "COMM" in [line.strip() for line in output]
            busy = [
                line.strip()
                for line in output
                if line.strip() in ("llama-server", "llama-bench")
            ]
            if not busy:
                return
            print("Waiting for phone workloads:", busy, flush=True)
            time.sleep(10)

    files = [
        build / "libhmx_16a16w_example.so",
        sdk / "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon",
        sdk
        / "libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/librun_main_on_hexagon_skel.so",
    ]
    idle()
    ssh(f"mkdir -p {remote}")
    subprocess.run(
        ["scp", *ssh_options, *map(str, files), f"{args.host}:{args.remote_dir}/"],
        check=True,
        timeout=40,
    )
    device_files = [shlex.quote(args.remote_dir + "/" + path.name) for path in files]
    expected_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    hashes = ssh("sha256sum " + " ".join(device_files)).stdout
    assert [line.split()[0] for line in hashes.splitlines()] == expected_hashes
    (build / "device_hashes.txt").write_text(hashes)
    ssh(
        f"chmod +x {remote}/run_main_on_hexagon; "
        f"printf '0x1f\n' > {remote}/run_main_on_hexagon.farf; "
        f"rm -f {remote}/output.u8 {remote}/output.f32 {remote}/output.u16 {remote}/a16.f32"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_16a16w_example.so {remote}"
    )
    capture = (
        "su -c 'timeout 6 /system/bin/logcat -v threadtime -T 1 -s adsprpc I4CONTROL'"
    )
    command = (
        f"{capture} > {remote}/example.farf.log 2>&1 & i4_log=$!; "
        "sleep 0.2; /system/bin/log -t I4CONTROL BEGIN; "
        f"{command}; i4_rc=$?; sleep 0.2; /system/bin/log -t I4CONTROL END; "
        "wait $i4_log || true; exit $i4_rc"
    )
    # A failed/ambiguous DSP invocation is never retried automatically.
    result = ssh(command, timeout=45)
    (build / "device.log").write_text(result.stdout + result.stderr)
    for name in ("example.farf.log",):
        subprocess.run(
            [
                "scp",
                *ssh_options,
                f"{args.host}:{args.remote_dir}/{name}",
                str(build / name),
            ],
            check=True,
            timeout=30,
        )

    assert (
        "Successfully called main() on Hexagon DSP and received return value of 0."
        in result.stdout
    )
    for name in ("output.u8", "output.f32", "output.u16", "a16.f32"):
        subprocess.run(
            [
                "scp",
                *ssh_options,
                f"{args.host}:{args.remote_dir}/{name}",
                str(build / name),
            ],
            check=True,
            timeout=30,
        )

    q = np.fromfile(build / "output.u8", dtype=np.uint8).reshape(5, 64, 128)
    y = np.fromfile(build / "output.f32", dtype="<f4").reshape(5, 64, 128)
    r, k = np.indices((64, 32), dtype=np.int32)
    wk, n = np.indices((32, 128), dtype=np.int32)
    checks = []
    for test in range(5):
        if test == 0:
            a = 128 + (r * 3 + k * 7 + r * k) % 5 - 2
        elif test == 1:
            a = (r * 17 + k * 29 + r * k * 3) % 256
        else:
            a = np.full((64, 32), 128, dtype=np.int32)
        b = (
            (wk * 11 + n * 19 + wk * n) % 16 - 8
            if test == 1
            else (wk * 3 + n * 7 + wk * n) % 5 - 2
        )
        if test == 3:
            state = 123456789
            values = []
            for _ in range(64 * 32 + 32 * 128):
                state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                values.append((state >> 24) % 3 - 1)
            a = 128 + np.array(values[: 64 * 32], dtype=np.int32).reshape(64, 32)
            b = np.array(values[64 * 32 :], dtype=np.int32).reshape(32, 128)
        elif test == 4:
            b.fill(0)
            a[17, 13], a[63, 31], a[0, 0] = 129, 130, 127
            b[13, 97], b[31, 0], b[0, 127] = 1, -8, 7
        accumulator = (a - 128) @ b
        expected_q = np.clip(accumulator + 128, 0, 255).astype(np.uint8)
        expected_y = (expected_q.astype(np.float32) - 128) / 32
        assert np.array_equal(q[test], expected_q), test
        assert np.array_equal(y[test], expected_y), test
        checks.append(
            {
                "case": test,
                "elements": 8192,
                "quantized_mismatches": 0,
                "fp32_mismatches": 0,
                "saturated_elements": int(
                    ((accumulator < -128) | (accumulator > 127)).sum()
                ),
                "first_q": q[test, 0, :8].tolist(),
                "first_y": y[test, 0, :8].tolist(),
            }
        )
    q16 = np.fromfile(build / "output.u16", dtype="<u2").reshape(8, 32, 64)
    y16 = np.fromfile(build / "a16.f32", dtype="<f4").reshape(8, 32, 64)
    r16, k16 = np.indices((32, 32), dtype=np.int64)
    bk16, n16 = np.indices((32, 64), dtype=np.int64)
    a16_checks = []
    for test in range(8):
        a = np.full((32, 32), 32768, dtype=np.int64)
        b = (bk16 * 11 + n16 * 19 + bk16 * n16) % 16 - 8
        if test == 1:
            a += 256 * ((r16 * 3 + k16 * 7 + r16 * k16) % 5 - 2)
        elif test == 2:
            a = (r16 * 2017 + k16 * 2893 + r16 * k16 * 113) % 65536
        elif test == 3:
            b.fill(0)
            a[17, 13], a[31, 31] = 33024, 33280
            b[13, 49], b[31, 0] = 1, -8
        elif test in (4, 5):
            a.fill(0 if test == 4 else 65535)
            b.fill(-8)
        elif test in (6, 7):
            state = 987654321
            av, bv = [], []
            for _ in range(32 * 32):
                state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                av.append(state >> 16 if test == 6 else 32766 + (state >> 24) % 5)
            for _ in range(32 * 64):
                state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                bv.append(
                    (state >> 24) % 16 - 8 if test == 6 else (state >> 24) % 5 - 2
                )
            a = np.array(av, dtype=np.int64).reshape(32, 32)
            b = np.array(bv, dtype=np.int64).reshape(32, 64)
        accumulator = (a - 32768) @ b
        unbounded = np.floor_divide(accumulator, 256) + 32768
        expected_q = np.clip(unbounded, 0, 65535).astype(np.uint16)
        expected_y = (expected_q.astype(np.float32) - 32768) / 4
        assert np.array_equal(q16[test], expected_q), ("A16 quantized", test)
        assert np.array_equal(y16[test], expected_y), ("A16 FP32", test)
        a16_checks.append(
            {
                "case": test,
                "elements": 2048,
                "quantized_mismatches": 0,
                "fp32_mismatches": 0,
                "saturated_elements": int(
                    ((unbounded < 0) | (unbounded > 65535)).sum()
                ),
                "first_q": q16[test, 0, :8].tolist(),
            }
        )
    evidence = {
        "tier": "device",
        "shape_mkn": [64, 32, 128],
        "a16_shape_mkn": [32, 32, 64],
        "a16_checks": a16_checks,
        "a16_q_sha256": hashlib.sha256((build / "output.u16").read_bytes()).hexdigest(),
        "a16_y_sha256": hashlib.sha256((build / "a16.f32").read_bytes()).hexdigest(),
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "q_sha256": hashlib.sha256((build / "output.u8").read_bytes()).hexdigest(),
        "y_sha256": hashlib.sha256((build / "output.f32").read_bytes()).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
