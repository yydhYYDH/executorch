import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
from verify import load_qnn_fixture, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--host", default="oneplus13-reverse")
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/hmx_lpbq_minimal/qnn_reference",
    )
    parser.add_argument(
        "--runner-dir", default="/data/data/com.termux/files/home/csm/qnnb"
    )
    parser.add_argument(
        "--skel-dir",
        default="/data/data/com.termux/files/home/csm/matmul_rebenchmark/dispatch_probe/baseline",
    )
    parser.add_argument(
        "--sdk",
        type=Path,
        default=Path(
            os.environ.get("QNN_SDK_ROOT", "/home/yydh/env/qairt/2.41.0.251128")
        ),
    )
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    options = ["-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1"]

    def ssh(command, timeout=30):
        return subprocess.run(
            ["ssh", *options, args.host, command],
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    while True:
        processes = ssh("/system/bin/ps -A -o COMM").stdout.splitlines()
        assert "COMM" in [name.strip() for name in processes]
        if not any(
            name.strip() in ("llama-server", "llama-bench") for name in processes
        ):
            break
        print("Waiting for phone workloads", flush=True)
        time.sleep(10)
    remote = shlex.quote(args.remote_dir)
    runner = args.runner_dir + "/qnn_executor_runner"
    skel = args.skel_dir + "/libQnnHtpV79Skel.so"
    hashes = ssh("sha256sum " + shlex.quote(runner) + " " + shlex.quote(skel)).stdout
    local_skel = args.sdk / "lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"
    rows = [line.split() for line in hashes.splitlines()]
    assert rows[1][0] == hashlib.sha256(local_skel.read_bytes()).hexdigest(), hashes
    (fixture / "qnn_device_hashes.txt").write_text(hashes)
    metadata = json.loads((fixture / "logical.json").read_text())
    for name, field in (("model.pte", "pte_sha256"), ("input.f32", "input_sha256")):
        assert (
            hashlib.sha256((fixture / name).read_bytes()).hexdigest() == metadata[field]
        )
    ssh(f"mkdir -p {remote}")
    (fixture / "input.txt").write_text(args.remote_dir + "/input.f32\n")
    files = [fixture / name for name in ("model.pte", "input.f32", "input.txt")]
    subprocess.run(
        ["scp", *options, *map(str, files), f"{args.host}:{args.remote_dir}/"],
        check=True,
        timeout=40,
    )
    deployed = ssh(
        "sha256sum "
        + " ".join(shlex.quote(args.remote_dir + "/" + path.name) for path in files)
    ).stdout
    assert [line.split()[0] for line in deployed.splitlines()] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    ]
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.runner_dir
    )
    command = (
        f"cd {remote} && mkdir -p output && rm -f output/output_0_0.raw && "
        f"LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={shlex.quote(args.skel_dir)} "
        f"timeout 30 taskset -c 7 {shlex.quote(runner)} --model_path {remote}/model.pte "
        f"--input_list_path {remote}/input.txt --output_folder_path {remote}/output "
        "--warm_up 0 --iteration 1 --profile_level 0"
    )
    result = ssh(command, 45)
    (fixture / "qnn_reference.log").write_text(result.stdout + result.stderr)
    subprocess.run(
        [
            "scp",
            *options,
            f"{args.host}:{args.remote_dir}/output/output_0_0.raw",
            str(fixture / "qnn_before.raw"),
        ],
        check=True,
        timeout=30,
    )
    cases, qnn_outputs = load_qnn_fixture(fixture)
    checks = []
    for case, (q, y) in zip(cases, qnn_outputs):
        expected_q, expected_y, _ = reference(case)
        checks.append(
            {
                "q_mismatches": int(np.count_nonzero(q != expected_q)),
                "y_mismatches": int(
                    np.count_nonzero(y.view(np.uint32) != expected_y.view(np.uint32))
                ),
            }
        )
    evidence = {
        "checks": checks,
        "original_skel_sha256": rows[1][0],
        "runtime_capture_dependency": False,
        "output_sha256": hashlib.sha256(
            (fixture / "qnn_before.raw").read_bytes()
        ).hexdigest(),
    }
    (fixture / "qnn_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    assert all(item["q_mismatches"] == item["y_mismatches"] == 0 for item in checks)


if __name__ == "__main__":
    main()
