import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import struct

from prepare_host import prepare_cases
from verify import make_cases, reference, load_qnn_fixture

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
        default=repo / ".tmp/scratch/mnv2/hmx_lpbq_minimal/example",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_lpbq_minimal/example",
    )
    parser.add_argument("--qnn-fixture", type=Path)
    parser.add_argument("--host-prepare", action="store_true")
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
        build / "libhmx_lpbq_example.so",
        sdk / "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon",
        sdk
        / "libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/librun_main_on_hexagon_skel.so",
    ]
    cases = make_cases()
    qnn_outputs = []
    if args.qnn_fixture:
        fixture_cases, qnn_outputs = load_qnn_fixture(args.qnn_fixture)
        data = bytearray(struct.pack("<I", len(fixture_cases)))
        for a, w, scales, sa, sq, za, zq, global_weight_scale in fixture_cases:
            data.extend(a.tobytes() + w.tobytes() + scales.tobytes())
            data.extend(struct.pack("<fffHH", global_weight_scale, sa, sq, za, zq))
        (build / "fixture.bin").write_bytes(data)
        files.append(build / "fixture.bin")
        cases.extend(fixture_cases)
    if args.host_prepare:
        if not args.qnn_fixture:
            (build / "fixture.bin").write_bytes(struct.pack("<I", 0))
            files.append(build / "fixture.bin")
        (build / "prepared.bin").write_bytes(
            prepare_cases(build / "libhmx_lpbq_prepare.so", cases)
        )
        files.append(build / "prepared.bin")
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
        f"rm -f {remote}/output.q {remote}/output.y"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_lpbq_example.so {remote}"
    )
    if args.qnn_fixture or args.host_prepare:
        command += f" {remote}/fixture.bin"
    if args.host_prepare:
        command += f" {remote}/prepared.bin"
    capture = (
        "su -c 'timeout 6 /system/bin/logcat -v threadtime -T 1 -s adsprpc LPBQCONTROL'"
    )
    command = (
        f"{capture} > {remote}/example.farf.log 2>&1 & lpbq_log=$!; "
        "sleep 0.2; /system/bin/log -t LPBQCONTROL BEGIN; "
        f"{command}; lpbq_rc=$?; sleep 0.2; /system/bin/log -t LPBQCONTROL END; "
        "wait $lpbq_log || true; exit $lpbq_rc"
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

    if args.host_prepare:
        assert (
            "LPBQEXAMPLE result 0 host_prepared 1"
            in (build / "example.farf.log").read_text()
        )
    assert (
        "Successfully called main() on Hexagon DSP and received return value of 0."
        in result.stdout
    )
    for name in ("output.q", "output.y"):
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
    q = np.fromfile(build / "output.q", dtype="<u2").reshape(len(cases), 32, 64)
    y = np.fromfile(build / "output.y", dtype="<f4").reshape(len(cases), 32, 64)
    checks = []
    for test, case in enumerate(cases):
        expected_q, expected_y, _ = reference(case)
        qe = int(np.count_nonzero(q[test] != expected_q))
        ye = int(
            np.count_nonzero(y[test].view(np.uint32) != expected_y.view(np.uint32))
        )
        check = {"case": test, "elements": 2048, "q_mismatches": qe, "y_mismatches": ye}
        if test >= 12:
            qref, yref = qnn_outputs[test - 12]
            check["qnn_q_mismatches"] = int(np.count_nonzero(q[test] != qref))
            check["qnn_y_mismatches"] = int(
                np.count_nonzero(y[test].view(np.uint32) != yref.view(np.uint32))
            )
        checks.append(check)
        print(check, flush=True)
    evidence = {
        "tier": "device",
        "shape_mkn": [32, 128, 64],
        "block_size": 32,
        "runtime_qnn_dependency": False,
        "preparation_location": "host" if args.host_prepare else "DSP",
        "conversion_parameters": "generated from logical FP32 scales and zero points",
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "q_sha256": hashlib.sha256((build / "output.q").read_bytes()).hexdigest(),
        "y_sha256": hashlib.sha256((build / "output.y").read_bytes()).hexdigest(),
    }
    if args.host_prepare:
        evidence["host_prepare_library_sha256"] = hashlib.sha256(
            (build / "libhmx_lpbq_prepare.so").read_bytes()
        ).hexdigest()
        evidence["prepared_sha256"] = hashlib.sha256(
            (build / "prepared.bin").read_bytes()
        ).hexdigest()
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert all(c["q_mismatches"] == c["y_mismatches"] == 0 for c in checks), checks
    if qnn_outputs:
        assert all(
            c.get("qnn_q_mismatches", 0) == c.get("qnn_y_mismatches", 0) == 0
            for c in checks
        ), checks
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
