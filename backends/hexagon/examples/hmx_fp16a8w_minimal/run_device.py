import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np


def main():
    """Run the FP16 weight-only HMX example and verify its complete outputs."""
    repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="oneplus13-reverse")
    parser.add_argument(
        "--sdk", default="/home/yydh/hexagon-tutorial/tools/hexagon-sdk"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=repo / ".tmp/scratch/mnv2/hmx_fp16a8w_minimal/example",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_fp16a8w_minimal/example",
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
        build / "libhmx_fp16a8w_example.so",
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
        f"rm -f {remote}/output.f16 {remote}/output.f32"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_fp16a8w_example.so {remote}"
    )
    capture = "su -c 'timeout 6 /system/bin/logcat -v threadtime -T 1 -s adsprpc FP16A8WCONTROL'"
    command = (
        f"{capture} > {remote}/example.farf.log 2>&1 & fp16a8w_log=$!; "
        "sleep 0.2; /system/bin/log -t FP16A8WCONTROL BEGIN; "
        f"{command}; fp16a8w_rc=$?; sleep 0.2; /system/bin/log -t FP16A8WCONTROL END; "
        "wait $fp16a8w_log || true; exit $fp16a8w_rc"
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
    for name in ("output.f16", "output.f32"):
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
    q = np.fromfile(build / "output.f16", dtype="<f2").reshape(10, 32, 64)
    y = np.fromfile(build / "output.f32", dtype="<f4").reshape(10, 32, 64)
    r, k = np.indices((32, 32), dtype=np.int64)
    wk, n = np.indices((32, 64), dtype=np.int64)
    checks = []
    for test in range(10):
        a = ((r * 3 + k * 7 + r * k) % 33 - 16) / 16
        if test == 2:
            a.fill(0)
        elif test == 3:
            a = np.eye(32, dtype=np.float64)
        elif test == 4:
            a.fill(0)
            a[17, 13] = -0.5
        elif test in (5, 6):
            a.fill(1 if test == 5 else -1)
        elif test == 9:
            a = ((r * 71 + k * 37) % 2049 - 1024) / 1024
        b = (wk * 11 + n * 19 + wk * n) % 255 - 127
        if test == 1:
            b.fill(0)
        elif test == 4:
            b.fill(0)
            b[13, 49] = -127
        elif test in (5, 6):
            b.fill(-128 if test == 5 else 127)
        columns = np.arange(64)
        scale = ((1 + columns % 7) / 256).astype(np.float32)
        if test == 7:
            scale = (
                (columns % 13 + 1).astype(np.float32) * np.float32(0.000731)
            ).astype(np.float32)
        elif test == 8:
            scale = np.where(
                columns % 2, np.float32(0.0078135006), np.float32(0.00452736672)
            ).astype(np.float32)
        elif test == 9:
            scale = ((1 + columns % 7) / 64).astype(np.float32)
        native_w = (
            b.astype(np.float32) * scale.astype(np.float16).astype(np.float32)
        ).astype(np.float16)
        expected = (
            a.astype(np.float16).astype(np.float64) @ native_w.astype(np.float64)
        ).astype(np.float16)
        mismatches = int(
            np.count_nonzero(q[test].view(np.uint16) != expected.view(np.uint16))
        )
        widened_mismatches = int(
            np.count_nonzero(
                y[test].view(np.uint32) != expected.astype(np.float32).view(np.uint32)
            )
        )
        assert mismatches == 0, (test, "fp16", mismatches)
        assert widened_mismatches == 0, (test, "fp32", widened_mismatches)
        without_scale_rounding = (b.astype(np.float32) * scale).astype(np.float16)
        without_scale_rounding_q = (
            a.astype(np.float16).astype(np.float64)
            @ without_scale_rounding.astype(np.float64)
        ).astype(np.float16)
        rounding_output_changes = int(
            np.count_nonzero(
                expected.view(np.uint16) != without_scale_rounding_q.view(np.uint16)
            )
        )
        if test in (7, 8):
            assert rounding_output_changes > 0, (test, "scale rounding control")
        checks.append(
            {
                "case": test,
                "elements": 2048,
                "fp16_mismatches": mismatches,
                "fp32_mismatches": widened_mismatches,
                "weights_changed_by_scale_rounding": int(
                    np.count_nonzero(
                        native_w.view(np.uint16)
                        != without_scale_rounding.view(np.uint16)
                    )
                ),
                "outputs_changed_by_scale_rounding": rounding_output_changes,
                "first_y": y[test, 0, :8].tolist(),
            }
        )
    evidence = {
        "tier": "device",
        "shape_mkn": [32, 32, 64],
        "native_kernel": "activation.hf / weight.hf",
        "scale_rounding": "fp16(weight_int8 * fp16(scale_fp32))",
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "fp16_sha256": hashlib.sha256((build / "output.f16").read_bytes()).hexdigest(),
        "fp32_sha256": hashlib.sha256((build / "output.f32").read_bytes()).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
