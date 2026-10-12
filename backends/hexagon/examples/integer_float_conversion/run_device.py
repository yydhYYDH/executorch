import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from verify import check_output, make_cases, write_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="oneplus13-reverse")
    parser.add_argument(
        "--sdk",
        type=Path,
        default=Path("/home/yydh/hexagon-tutorial/tools/hexagon-sdk"),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / ".tmp/scratch/mnv2/integer_float_conversion",
    )
    parser.add_argument(
        "--remote-dir",
        default="/data/data/com.termux/files/home/csm/integer_float_conversion",
    )
    args = parser.parse_args()
    build = args.build_dir.resolve()
    options = ["-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1"]

    def ssh(command, timeout=30):
        return subprocess.run(
            ["ssh", *options, args.host, command],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    while True:
        processes = ssh("/system/bin/ps -A -o COMM").stdout.splitlines()
        assert "COMM" in [p.strip() for p in processes]
        if not any(p.strip() in ("llama-server", "llama-bench") for p in processes):
            break
        print("Waiting for phone workloads", flush=True)
        time.sleep(10)
    cases = make_cases()
    write_fixture(build / "fixture.bin", cases)
    files = [
        build / "libinteger_float_conversion.so",
        build / "fixture.bin",
        args.sdk / "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon",
        args.sdk
        / "libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/librun_main_on_hexagon_skel.so",
    ]
    remote = shlex.quote(args.remote_dir)
    ssh(f"mkdir -p {remote}")
    subprocess.run(
        ["scp", *options, *map(str, files), f"{args.host}:{args.remote_dir}/"],
        check=True,
        timeout=40,
    )
    hashes = ssh(
        "sha256sum "
        + " ".join(shlex.quote(args.remote_dir + "/" + p.name) for p in files)
    ).stdout
    expected = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    assert [line.split()[0] for line in hashes.splitlines()] == expected
    (build / "device_hashes.txt").write_text(hashes)
    libraries = shlex.quote(
        "/system/lib64:/system_ext/lib64:/vendor/lib64:" + args.remote_dir
    )
    command = (
        f"cd {remote} && chmod +x run_main_on_hexagon && rm -f device_output.bin && "
        f"LD_LIBRARY_PATH={libraries} ADSP_LIBRARY_PATH={remote} timeout 30 "
        "./run_main_on_hexagon 3 libinteger_float_conversion.so "
        f"{remote}/fixture.bin {remote}/device_output.bin"
    )
    result = ssh(command, timeout=45)
    (build / "device.log").write_text(result.stdout + result.stderr)
    assert (
        "Successfully called main() on Hexagon DSP and received return value of 0."
        in result.stdout
    )
    subprocess.run(
        [
            "scp",
            *options,
            f"{args.host}:{args.remote_dir}/device_output.bin",
            str(build / "device_output.bin"),
        ],
        check=True,
        timeout=30,
    )
    evidence = check_output(build / "device_output.bin", cases)
    evidence.update(
        {
            "tier": "device",
            "qnn_dependency": False,
            "hmx_dependency": False,
            "so_sha256": expected[0],
        }
    )
    (build / "device_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("DSP: 20 cases and all 65,536 FP16 bit patterns passed", flush=True)


if __name__ == "__main__":
    main()
