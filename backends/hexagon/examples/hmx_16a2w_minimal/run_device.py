import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import struct

from verify import make_cases, reference, prepare, load_qnn_fixture
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
        default=repo / ".tmp/scratch/mnv2/hmx_16a2w_minimal/example",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_16a2w_minimal/example",
    )
    parser.add_argument("--qnn-fixture", type=Path, action="append", default=[])
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
        build / "libhmx_16a2w_example.so",
        sdk / "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon",
        sdk
        / "libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/librun_main_on_hexagon_skel.so",
    ]
    cases = make_cases()
    cases.extend(load_qnn_fixture(path) for path in args.qnn_fixture)
    data = bytearray(struct.pack("<I", len(cases)))
    for a, w, sw, sa, sq, za, zw, zq in cases:
        data.extend(struct.pack("<ffiii", sa, sq, za, zw, zq))
        data.extend(sw.astype("<f4").tobytes())
        data.extend(a.astype("<u2").tobytes())
        data.extend(w.tobytes())
    (build / "fixture.bin").write_bytes(data)
    files.append(build / "fixture.bin")
    native = ctypes.CDLL(str(build / "libhmx_16a2w_prepare_host.so"))
    encoder = native.hmx_a16w2_prepare
    ptr8 = ctypes.POINTER(ctypes.c_int8)
    ptrf = ctypes.POINTER(ctypes.c_float)
    encoder.argtypes = [
        ptr8,
        ptrf,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint16,
        ctypes.c_int,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    encoder.restype = ctypes.c_int
    for case in cases:
        a, w, sw, sa, sq, za, zw, zq = case
        wt = np.zeros(512, np.uint8)
        table = np.zeros(256, np.uint32)
        assert (
            encoder(
                w.ctypes.data_as(ptr8),
                sw.ctypes.data_as(ptrf),
                sa,
                sq,
                za,
                zw,
                zq,
                wt.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                table.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
            == 0
        )
        expected_w, expected_table, _, _, _ = prepare(case)
        assert np.array_equal(wt, expected_w)
        assert np.array_equal(table, expected_table)
    invalid_cases = []
    for bad_sa, bad_sq in ((0, 1), (float("nan"), 1), (float("inf"), 1), (1, 0)):
        invalid_cases.append((cases[0][1], cases[0][2], bad_sa, bad_sq, 0))
    for bad_scale in (0, float("nan"), float("inf")):
        sw = cases[0][2].copy()
        sw[-1] = bad_scale
        invalid_cases.append((cases[0][1], sw, cases[0][3], cases[0][4], 0))
    invalid_cases.append((cases[0][1], cases[0][2], cases[0][3], cases[0][4], 128))
    w = cases[0][1].copy()
    w[-1, -1] = 2
    invalid_cases.append((w, cases[0][2], cases[0][3], cases[0][4], 0))
    invalid_cases.append((cases[0][1], cases[0][2], 1e-30, 1, 0))
    for w, sw, sa, sq, zw in invalid_cases:
        wt = np.full(512, 0xA5, np.uint8)
        table = np.full(256, 0xA5A5A5A5, np.uint32)
        assert (
            encoder(
                w.ctypes.data_as(ptr8),
                sw.ctypes.data_as(ptrf),
                sa,
                sq,
                0,
                zw,
                0,
                wt.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                table.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
            == 1
        )
        assert np.all(wt == 0xA5) and np.all(table == 0xA5A5A5A5)
    print(
        "Host preparation rejected",
        len(invalid_cases),
        "invalid cases without writes",
        flush=True,
    )
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
        f"rm -f {remote}/output.u16 {remote}/output.f32 {remote}/table.u32 {remote}/weights.u8"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_16a2w_example.so {remote} {remote}/fixture.bin"
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
    for name in ("output.u16", "output.f32", "table.u32", "weights.u8"):
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
    q = np.fromfile(build / "output.u16", "<u2").reshape(len(cases), 32, 64)
    y = np.fromfile(build / "output.f32", "<f4").reshape(len(cases), 32, 64)
    tables = np.fromfile(build / "table.u32", "<u4").reshape(len(cases), 256)
    weights = np.fromfile(build / "weights.u8", "u1").reshape(len(cases), 512)
    checks = []
    for index, case in enumerate(cases):
        expected_q, expected_y = reference(case)
        expected_w, expected_table, _, _, _ = prepare(case)
        check = {
            "case": index,
            "parameters": [float(v) for v in case[3:]],
            "table_mismatches": int(np.count_nonzero(tables[index] != expected_table)),
            "weight_mismatches": int(np.count_nonzero(weights[index] != expected_w)),
            "q_mismatches": int(np.count_nonzero(q[index] != expected_q)),
            "y_mismatches": int(
                np.count_nonzero(y[index].view(np.uint32) != expected_y.view(np.uint32))
            ),
        }
        if index >= 40:
            path = args.qnn_fixture[index - 40]
            qnn_y = np.fromfile(path / "qnn_before.raw", "<f4").reshape(32, 64)
            qnn_q = np.rint(qnn_y.astype(np.float64) / case[4] + case[7]).astype(
                np.uint16
            )
            check["qnn_q_mismatches"] = int(np.count_nonzero(q[index] != qnn_q))
            check["qnn_y_mismatches"] = int(
                np.count_nonzero(y[index].view(np.uint32) != qnn_y.view(np.uint32))
            )
        checks.append(check)
        print(check, flush=True)
    evidence = {
        "tier": "device",
        "shape_mkn": [32, 32, 64],
        "runtime_qnn_dependency": False,
        "host_invalid_cases": len(invalid_cases),
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "output_sha256": hashlib.sha256(
            (build / "output.u16").read_bytes()
        ).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert all(
        c["table_mismatches"] == c["weight_mismatches"] == c["q_mismatches"] == 0
        for c in checks
    ), checks
    assert all(c["y_mismatches"] == 0 for c in checks), checks
    assert all(
        c.get("qnn_q_mismatches", 0) == c.get("qnn_y_mismatches", 0) == 0
        for c in checks
    )
    print("All", len(cases), "cases passed")


if __name__ == "__main__":
    main()
