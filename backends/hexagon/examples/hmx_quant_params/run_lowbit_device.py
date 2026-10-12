import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import struct

from verify_lowbit import make_cases, reference, coefficient, inputs
import ctypes

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
        default=repo / ".tmp/scratch/mnv2/hmx_quant_params/example/lowbit",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_quant_params/lowbit",
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
        build / "libhmx_lowbit_quant_params.so",
        sdk / "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon",
        sdk
        / "libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/librun_main_on_hexagon_skel.so",
    ]
    cases = make_cases()
    data = bytearray(struct.pack("<I", len(cases)))
    for case in cases:
        data.extend(struct.pack("<fffiiii", *case))
        a, w = inputs(case)
        data.extend(a.astype("<u2").tobytes())
        data.extend(w.tobytes())
    (build / "fixture.bin").write_bytes(data)
    files.append(build / "fixture.bin")
    native = ctypes.CDLL(str(build / "libconversion_host.so"))
    for bits, ds in [(8, [0, 4]), (16, [8])]:
        encoder = getattr(native, f"hmx_u{bits}_conversion_params")
        encoder.argtypes = [ctypes.c_float] * 3 + [
            ctypes.c_uint8 if bits == 8 else ctypes.c_uint16,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        encoder.restype = ctypes.c_int
        for case in cases:
            sa, sw, sq, _, _, zq, kind = case
            if (kind == 2) != (bits == 16):
                continue
            pair = (ctypes.c_uint32 * 2)()
            assert encoder(sa, sw, sq, zq, 8 if kind == 2 else kind * 4, pair) == 0
            assert np.array_equal(pair, coefficient(case)[0]), case
        for bad in [
            (0, 1, 1, 0, ds[0]),
            (1, 1, 0, 0, ds[0]),
            (float("nan"), 1, 1, 0, ds[0]),
            (float("inf"), 1, 1, 0, ds[0]),
            (1e-30, 1e-30, 1, 0, ds[0]),
            (1, 1, 1, 0, 3),
        ]:
            pair = (ctypes.c_uint32 * 2)(0x12345678, 0x9ABCDEF0)
            assert encoder(*bad, pair) == 1
            assert list(pair) == [0x12345678, 0x9ABCDEF0]
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
        f"rm -f {remote}/output.u16 {remote}/output.f32 {remote}/coefficients.u32"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_lowbit_quant_params.so {remote} {remote}/fixture.bin"
    )
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

    assert (
        "Successfully called main() on Hexagon DSP and received return value of 0."
        in result.stdout
    )
    for name in ("output.u16", "output.f32", "coefficients.u32"):
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
    q = np.fromfile(build / "output.u16", "<u2")
    y = np.fromfile(build / "output.f32", "<f4")
    words = np.fromfile(build / "coefficients.u32", "<u4").reshape(len(cases), 2)
    checks, offset = [], 0
    for index, case in enumerate(cases):
        expected_q, expected_y = reference(case)
        size = expected_q.size
        actual_q = q[offset : offset + size].reshape(expected_q.shape)
        actual_y = y[offset : offset + size].reshape(expected_y.shape)
        check = {
            "case": index,
            "kind": ["A8W8", "A8W4", "A16W4"][case[-1]],
            "parameters": list(case),
            "coefficient_mismatches": int(
                np.count_nonzero(words[index] != coefficient(case)[0])
            ),
            "q_mismatches": int(np.count_nonzero(actual_q != expected_q)),
            "y_mismatches": int(
                np.count_nonzero(actual_y.view(np.uint32) != expected_y.view(np.uint32))
            ),
            "q_count": size,
            "pair": [hex(int(v)) for v in words[index]],
        }
        checks.append(check)
        offset += size
        print(check, flush=True)
    assert offset == q.size == y.size
    evidence = {
        "tier": "device",
        "shape_mkn": [[64, 32, 64], [64, 32, 128], [32, 32, 64]],
        "runtime_qnn_dependency": False,
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "output_sha256": hashlib.sha256(
            (build / "output.u16").read_bytes()
        ).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert all(
        c["coefficient_mismatches"] == c["q_mismatches"] == 0 for c in checks
    ), checks
    assert all(c["y_mismatches"] == 0 for c in checks), checks
    print("All", len(cases), "cases passed")


if __name__ == "__main__":
    main()
