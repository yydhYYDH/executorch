import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

M, K, N, BLOCK, NBLOCK = 32, 64, 64, 32, 2


def main():
    """Run the block-wise HMX example on the phone and verify its complete outputs."""
    repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="oneplus13-reverse")
    parser.add_argument(
        "--sdk", default="/home/yydh/hexagon-tutorial/tools/hexagon-sdk"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=repo / ".tmp/scratch/mnv2/hmx_blockwise_minimal/example",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_blockwise_minimal/example",
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
        build / "libhmx_blockwise_example.so",
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
        f"printf '0x1f\\n' > {remote}/run_main_on_hexagon.farf; "
        f"rm -f {remote}/output.q {remote}/output.y"
    )
    idle()
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} "
        f"timeout 30 ./run_main_on_hexagon 3 libhmx_blockwise_example.so {remote}"
    )
    capture = (
        "su -c 'timeout 6 /system/bin/logcat -v threadtime -T 1 -s adsprpc BWCONTROL'"
    )
    command = (
        f"{capture} > {remote}/example.farf.log 2>&1 & bw_log=$!; "
        "sleep 0.2; /system/bin/log -t BWCONTROL BEGIN; "
        f"{command}; bw_rc=$?; sleep 0.2; /system/bin/log -t BWCONTROL END; "
        "wait $bw_log || true; exit $bw_rc"
    )
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
    ), (result.stdout + result.stderr)
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

    q = np.fromfile(build / "output.q", dtype="<u2").reshape(4, M, N)
    y = np.fromfile(build / "output.y", dtype="<f4").reshape(4, M, N)
    rr, cc = np.indices((M, K), dtype=np.int64)
    kk, nv = np.indices((K, N), dtype=np.int64)
    scale = np.empty((N, NBLOCK), dtype=np.int64)
    scale[:, 0] = (np.arange(N) % 3) + 2
    scale[:, 1] = (np.arange(N) % 5) + 1
    w = (kk * 11 + nv * 19 + kk * nv) % 16 - 8
    checks = []
    for test in range(4):
        a = (
            32768 + (rr * 3 + cc * 7) % 9 - 4
            if test == 0
            else (
                32768 + (rr * 17 + cc * 29) % 257 - 128
                if test == 1
                else (
                    32768 + 300 * ((rr * 5 + cc * 11) % 5 - 2)
                    if test == 3
                    else np.full((M, K), 32768)
                )
            )
        )
        acc = np.zeros((M, N), dtype=np.int64)
        for blk in range(NBLOCK):
            partial = np.zeros((M, N), dtype=np.int64)
            for i in range(BLOCK):
                kx = blk * BLOCK + i
                partial += (a[:, kx][:, None] - 32768) * w[kx]
            t = np.clip(np.floor_divide(partial, 256) + 32768, 0, 65535)
            acc += (t - 32768) * scale[:, blk][None, :]
        v = np.clip(acc + 32768, 0, 65535)
        expected_q = v.astype(np.uint16)
        expected_y = ((v - 32768) / 4).astype(np.float32)
        assert np.array_equal(q[test], expected_q), ("Q", test)
        assert np.array_equal(y[test], expected_y), ("Y", test)
        checks.append(
            {
                "case": test,
                "elements": M * N,
                "quantized_mismatches": 0,
                "fp32_mismatches": 0,
                "first_q": q[test, 0, :8].tolist(),
            }
        )
    evidence = {
        "tier": "device",
        "shape_mkn": [M, K, N],
        "block_size": BLOCK,
        "n_blocks": NBLOCK,
        "checks": checks,
        "so_sha256": expected_hashes[0],
        "q_sha256": hashlib.sha256((build / "output.q").read_bytes()).hexdigest(),
        "y_sha256": hashlib.sha256((build / "output.y").read_bytes()).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
