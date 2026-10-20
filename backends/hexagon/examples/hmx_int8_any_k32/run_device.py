#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
SDK = Path("/home/yydh/hexagon-tutorial/tools/hexagon-sdk")
HOST = "oneplus13-reverse"
REMOTE = "/data/data/com.termux/files/home/csm/hmx_int8_any_k32"
SSH_OPTIONS = ["-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1"]
SHAPES = [(1, 1, 1), (1, 1, 31), (1, 1, 33), (1, 63, 64),
          (63, 1, 95), (63, 63, 96), (64, 64, 127), (65, 65, 129),
          (127, 129, 256), (129, 127, 320), (200, 200, 512),
          (200, 200, 31), (1, 1, 512), (65, 65, 33)]


def ssh(command, timeout=30):
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, HOST, command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def clean(text):
    return "\n".join(
        line for line in text.splitlines() if line.strip() != "Welcome to VyOS"
    )


def native_index(r, n):
    return (n // 32) * 2048 + (r // 4) * 128 + (n % 32) * 4 + r % 4


def idle():
    while True:
        names = clean(ssh("/system/bin/ps -A -o COMM").stdout).splitlines()
        busy = [n for n in names if n.strip() in
                ("llama-server", "llama-cli", "llama-bench")]
        if not busy:
            return
        print("waiting for phone workloads:", busy, flush=True)
        time.sleep(10)


def main():
    build = REPO / ".tmp/scratch/mnv2/hmx_int8_any_k32"
    files = [
        build / "libhmx_any_k32.so",
        SDK / "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon",
        SDK / "libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/"
              "librun_main_on_hexagon_skel.so",
    ]
    idle()
    ssh(f"mkdir -p {shlex.quote(REMOTE)}")
    subprocess.run(["scp", *SSH_OPTIONS, *map(str, files),
                    f"{HOST}:{REMOTE}/"], check=True, timeout=40)
    expected = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    names = [shlex.quote(REMOTE + "/" + p.name) for p in files]
    got = clean(ssh("sha256sum " + " ".join(names)).stdout).splitlines()
    assert [line.split()[0] for line in got] == expected
    ssh(f"chmod +x {shlex.quote(REMOTE + '/run_main_on_hexagon')}; "
        f"printf '0x1f\\n' > {shlex.quote(REMOTE + '/run_main_on_hexagon.farf')}; "
        f"rm -f {shlex.quote(REMOTE + '/tiles.u8')}")
    libraries = "/system/lib64:/system_ext/lib64:/vendor/lib64:" + REMOTE
    command = (
        f"cd {shlex.quote(REMOTE)} && LD_LIBRARY_PATH={shlex.quote(libraries)} "
        f"ADSP_LIBRARY_PATH={shlex.quote(REMOTE)} timeout 60 "
        f"./run_main_on_hexagon 3 libhmx_any_k32.so {shlex.quote(REMOTE)}"
    )
    capture = ("su -c 'timeout 10 /system/bin/logcat -v threadtime -T 1 "
               "-s adsprpc ANYK'")
    command = (
        f"{capture} > {shlex.quote(REMOTE + '/run.farf.log')} 2>&1 & lp=$!; "
        "sleep 0.2; /system/bin/log -t ANYK BEGIN; "
        f"{command}; rc=$?; sleep 0.2; /system/bin/log -t ANYK END; "
        "wait $lp || true; exit $rc"
    )
    result = ssh(command, timeout=80)
    stdout = clean(result.stdout) + clean(result.stderr)
    (build / "device.log").write_text(stdout)
    assert ("Successfully called main() on Hexagon DSP and received return "
            "value of 0." in stdout), stdout
    for name in ("run.farf.log", "tiles.u8"):
        subprocess.run(["scp", *SSH_OPTIONS, f"{HOST}:{REMOTE}/{name}",
                        str(build / name)], check=True, timeout=30)

    raw = np.fromfile(build / "tiles.u8", dtype=np.uint8)
    checks = []
    offset = 0
    for case, (m, n, k) in enumerate(SHAPES):
        mt, nt = (m + 63) // 64, (n + 63) // 64
        count = mt * nt * 4096
        tiles = raw[offset:offset + count].reshape(mt, nt, 4096)
        offset += count
        test = case % 3
        r, kk = np.indices((m, k), dtype=np.int32)
        a = ((r * 17 + kk * 29 + r * kk * 3 + test * 11) % 256).astype(np.uint8)
        if test == 0:
            a = np.full((m, k), 128, dtype=np.uint8)
        wk, c = np.indices((k, n), dtype=np.int32)
        b = (((wk * 11 + c * 19 + wk * c * 5 + test * 7) % 17) - 8).astype(np.int8)
        acc = (a.astype(np.int64) - 128) @ b.astype(np.int64)
        expected_q = np.clip(acc + 128, 0, 255).astype(np.uint8)

        got = np.empty((m, n), dtype=np.uint8)
        rr, cc = np.indices((64, 64), dtype=np.int32)
        indices = native_index(rr, cc)
        for mi in range(mt):
            for ni in range(nt):
                for rr in range(64):
                    for cc in range(64):
                        gr, gc = mi * 64 + rr, ni * 64 + cc
                        if gr < m and gc < n:
                            got[gr, gc] = tiles[
                                mi, ni, indices[rr, cc]]
        errors = int(np.count_nonzero(got != expected_q))
        checks.append({
            "case": case, "m": m, "n": n, "k": k, "tiles_mn": [mt, nt],
            "elements": m * n, "mismatches": errors,
            "saturated_elements": int(((acc < -128) | (acc > 127)).sum()),
        })
        if errors:
            raise AssertionError(
                f"case {case} shape {m}x{n}x{k}: {errors} mismatches")
    assert offset == raw.size
    evidence = {
        "tier": "device", "k": "any positive", "k_pad": 32,
        "output": "native U8 tiles",
        "shapes": SHAPES, "checks": checks,
        "so_sha256": expected[0],
        "tiles_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
    }
    (build / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
