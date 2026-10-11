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
        default=repo / ".tmp/scratch/mnv2/hmx_int8_minimal/example",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_int8_minimal/example",
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
                if line.strip() in ("llama-server", "llama-cli", "llama-bench")
            ]
            if not busy:
                return
            print("Waiting for phone workloads:", busy, flush=True)
            time.sleep(10)

    files = [
        build / "libhmx_int8_example.so",
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
        f"rm -f {remote}/output.u8 {remote}/output.f32"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_int8_example.so {remote}"
    )
    capture = (
        "su -c 'timeout 6 /system/bin/logcat -v threadtime -T 1 -s adsprpc I8CONTROL'"
    )
    command = (
        f"{capture} > {remote}/example.farf.log 2>&1 & i8_log=$!; "
        "sleep 0.2; /system/bin/log -t I8CONTROL BEGIN; "
        f"{command}; i8_rc=$?; sleep 0.2; /system/bin/log -t I8CONTROL END; "
        "wait $i8_log || true; exit $i8_rc"
    )
    # A failed/ambiguous DSP invocation is never retried automatically.
    result = ssh(command, timeout=45)
    (build / "device.log").write_text(result.stdout + result.stderr)
    assert (
        "Successfully called main() on Hexagon DSP and received return value of 0."
        in result.stdout
    )
    for name in ("example.farf.log", "output.u8", "output.f32"):
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

    q = np.fromfile(build / "output.u8", dtype=np.uint8).reshape(3, 64, 64)
    y = np.fromfile(build / "output.f32", dtype="<f4").reshape(3, 64, 64)
    r, k = np.indices((64, 32), dtype=np.int32)
    wk, n = np.indices((32, 64), dtype=np.int32)
    checks = []
    for test in range(3):
        if test == 0:
            a = 128 + (r * 3 + k * 7 + r * k) % 5 - 2
        elif test == 1:
            a = (r * 17 + k * 29 + r * k * 3) % 256
        else:
            a = np.full((64, 32), 128, dtype=np.int32)
        b = (
            (wk * 11 + n * 19 + wk * n) % 17 - 8
            if test == 1
            else (wk * 3 + n * 7 + wk * n) % 5 - 2
        )
        accumulator = (a - 128) @ b
        expected_q = np.clip(accumulator + 128, 0, 255).astype(np.uint8)
        expected_y = (expected_q.astype(np.float32) - 128) / 32
        assert np.array_equal(q[test], expected_q), test
        assert np.array_equal(y[test], expected_y), test
        checks.append(
            {
                "case": test,
                "elements": 4096,
                "quantized_mismatches": 0,
                "fp32_mismatches": 0,
                "saturated_elements": int(
                    ((accumulator < -128) | (accumulator > 127)).sum()
                ),
                "first_q": q[test, 0, :8].tolist(),
                "first_y": y[test, 0, :8].tolist(),
            }
        )
    evidence = {
        "tier": "device",
        "shape_mkn": [64, 32, 64],
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "q_sha256": hashlib.sha256((build / "output.u8").read_bytes()).hexdigest(),
        "y_sha256": hashlib.sha256((build / "output.f32").read_bytes()).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
