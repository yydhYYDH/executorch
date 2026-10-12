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
        default=repo / ".tmp/scratch/mnv2/hmx_16a8w_minimal/example",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_16a8w_minimal/example",
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
        build / "libhmx_16a8w_example.so",
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
        f"rm -f {remote}/output.u16"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_16a8w_example.so {remote}"
    )
    capture = "su -c 'timeout 6 /system/bin/logcat -v threadtime -T 1 -s adsprpc A16W8CONTROL'"
    command = (
        f"{capture} > {remote}/example.farf.log 2>&1 & a16w8_log=$!; "
        "sleep 0.2; /system/bin/log -t A16W8CONTROL BEGIN; "
        f"{command}; a16w8_rc=$?; sleep 0.2; /system/bin/log -t A16W8CONTROL END; "
        "wait $a16w8_log || true; exit $a16w8_rc"
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
    subprocess.run(
        [
            "scp",
            *ssh_options,
            f"{args.host}:{args.remote_dir}/output.u16",
            str(build / "output.u16"),
        ],
        check=True,
        timeout=30,
    )
    q = np.fromfile(build / "output.u16", dtype="<u2").reshape(12, 32, 64)
    checks = []
    for test in range(12):
        state = 1234567 + test
        av, bv = [], []
        for i in range(32 * 32):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            delta = (
                0
                if test == 0
                else (state >> 20) % (65 if test == 1 else 2049)
                - (32 if test == 1 else 1024)
            )
            value = 33401 + delta
            if test == 3:
                value = state >> 16
            elif test in (4, 6):
                value = 0
            elif test in (5, 7):
                value = 65535
            elif test == 8:
                value = 33401
            elif test == 9:
                value = i * 63
            elif test in (10, 11):
                value = 33401 + (-4872 if test == 10 else 4872)
            av.append(value)
        for i in range(32 * 64):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            value = (state >> 16) % 256 - 128
            if test in (4, 5, 6, 7):
                value = -128 if test < 6 else 127
            elif test in (8, 9):
                value = 0
            elif test in (10, 11):
                value = 64 if i // 64 == 0 else 0
            bv.append(value)
        a = np.array(av, dtype=np.int64).reshape(32, 32)
        b = np.array(bv, dtype=np.int64).reshape(32, 64)
        numerator = 33401 * b.sum(axis=0)
        correction = np.sign(numerator) * ((np.abs(numerator) + 128) // 256)
        shifted = (a @ b) // 256 - correction
        # Exact conversion for the captured a8403e00 / 39800212 coefficient pair.
        unbounded = (
            (shifted * 256 + (shifted != 0)) * 3183617 + 472328 * (1 << 26)
        ) // (1 << 30)
        expected = np.clip(unbounded, 0, 65535).astype(np.uint16)
        mismatches = int(np.count_nonzero(q[test] != expected))
        assert mismatches == 0, (test, mismatches)
        checks.append(
            {
                "case": test,
                "elements": 2048,
                "quantized_mismatches": mismatches,
                "saturated_elements": int(
                    ((unbounded < 0) | (unbounded > 65535)).sum()
                ),
                "first_q": q[test, 0, :8].tolist(),
            }
        )
    evidence = {
        "tier": "device",
        "shape_mkn": [32, 32, 64],
        "coefficient_words": ["a8403e00", "39800212"],
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "q_sha256": hashlib.sha256((build / "output.u16").read_bytes()).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
