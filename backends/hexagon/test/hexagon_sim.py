"""Build a DSP runner against the vendored htp-ops kernels and run it in hexagon-sim.

The runner is a plain Hexagon shared object with a main(). The SDK's
run_main_on_hexagon_sim loads it into a simulated QuRT, so the kernels execute
for real -- HVX and all -- with no device and no FastRPC. Undefined qurt_*
symbols in the shared object are deliberate: the simulated RTOS resolves them
at load time, which is why the runner is a library and not an executable.

Nothing here links the ExecuTorch runtime. The point is to run the same
vendored sources the skel is built from.
"""

from __future__ import annotations

import glob
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

#: Where the vendored MNN op library lives, relative to this file.
MNN_OPS = pathlib.Path(__file__).resolve().parents[1] / "third-party/mnn-htp-ops"


def _sdk_root() -> Optional[pathlib.Path]:
    configured = os.environ.get("HEXAGON_SDK_ROOT")
    if configured and pathlib.Path(configured, "tools").is_dir():
        return pathlib.Path(configured)
    return None


def _tools(sdk: pathlib.Path) -> Optional[pathlib.Path]:
    found = sorted(glob.glob(str(sdk / "tools/HEXAGON_Tools/*/Tools")))
    return pathlib.Path(found[-1]) if found else None


def _sim_libs() -> str:
    """Directories holding the simulator's own runtime libraries.

    hexagon-sim is an old binary and wants libncurses.so.5, which recent
    distributions no longer ship. HEXAGON_SIM_LD_LIBRARY_PATH names a directory
    that provides it; the glob is the same thing preconfigured locally.
    """
    candidates = [os.environ.get("HEXAGON_SIM_LD_LIBRARY_PATH", "")]
    candidates += glob.glob(os.path.expanduser("~/*/.artifacts/*/sim-compat"))
    found = []
    for path in candidates:
        if path and pathlib.Path(path, "libncurses.so.5").exists():
            found.append(path)
    return os.pathsep.join(found)


#: The last simulator stdout, for diagnosing a runner that prints nothing.
LAST_STDOUT = ""


class Unavailable(RuntimeError):
    """The toolchain or simulator is not usable here."""


def _check() -> tuple:
    sdk = _sdk_root()
    if sdk is None:
        raise Unavailable("HEXAGON_SDK_ROOT is not set to a Hexagon SDK")
    tools = _tools(sdk)
    if tools is None:
        raise Unavailable(f"no HEXAGON_Tools under {sdk}")
    if not (sdk / "rtos/qurt/computev79/sdksim_bin/runelf.pbn").exists():
        raise Unavailable(f"no v79 QuRT runtime under {sdk}")
    return sdk, tools


_FLAGS = "-mv79 -mhvx -mhvx-length=128b -mhmx -O2 -fPIC -std=c++17 -w".split()
_DEFINES = [
    "-DHTP_OPS_SKEL_ARCH=0x79",
    "-DHTP_OPS_PWL_COMPANDED16=1",
    "-DHTP_OPS_PWL_LEARNED8=1",
]


def build(runner: pathlib.Path, sources: List[str], work: pathlib.Path,
          includes_more: List[str] = ()) -> pathlib.Path:
    """Compile one DSP runner plus the vendored sources into a shared object.

    includes_more holds extra -I directories, and the work directory is always
    one of them so a runner can include a header the caller generated there.
    """
    sdk, tools = _check()
    arch = "v79"
    lib = tools / f"target/hexagon/lib/{arch}/G0/pic"
    includes = [f"-I{work}", *[f"-I{path}" for path in includes_more]] + [
        f"-I{MNN_OPS}",
        f"-I{MNN_OPS}/include",
        f"-I{MNN_OPS}/include/dsp",
        f"-I{MNN_OPS}/src/dsp",
        f"-I{sdk}/incs",
        f"-I{sdk}/incs/stddef",
        f"-I{sdk}/rtos/qurt/compute{arch}/include",
        f"-I{sdk}/rtos/qurt/compute{arch}/include/qurt",
        f"-I{sdk}/rtos/qurt/compute{arch}/include/posix",
    ]
    def compile_one(source: str, obj: pathlib.Path) -> None:
        # The vendored library is mixed C and C++, and the ++ driver treats a
        # .c input as C++ regardless of its name, so pick the driver to match.
        cxx = not source.endswith(".c")
        cc = str(tools / ("bin/hexagon-clang++" if cxx else "bin/hexagon-clang"))
        # gnu dialects, which is what CMake's C_STANDARD selects: the kernels
        # use the asm keyword, which strict C11 does not have.
        std = "-std=gnu++17" if cxx else "-std=gnu11"
        flags = [flag for flag in _FLAGS if not flag.startswith("-std=")] + [std]
        done = subprocess.run(
            [cc, *flags, *_DEFINES, *includes, "-c", source, "-o", str(obj)],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            raise Unavailable(f"compiling {source}:\n{done.stderr[-3000:]}")

    objects = []
    for index, source in enumerate(sources):
        obj = work / f"unit{index}.o"
        compile_one(str(MNN_OPS / "src/dsp" / source), obj)
        objects.append(str(obj))
    main = work / "runner_main.o"
    compile_one(str(runner), main)
    out = work / "runner.so"
    # -nostdlib++ keeps the DSP C++ runtime static: a NEEDED entry for
    # libc++.so.1 makes the loader fail with AEE_EFAILED, the same trap the
    # skel's own build documents.
    subprocess.run(
        [str(tools / "bin/hexagon-clang++"), f"-m{arch}", "-shared", "-O2",
         "-nostdlib++", str(main), *objects,
         "-o", str(out), f"-L{lib}", str(lib / "libc++.a"), str(lib / "libc++abi.a")],
        check=True, capture_output=True,
    )
    return out


def _configs(sdk: pathlib.Path, tools: pathlib.Path, work: pathlib.Path) -> None:
    iss = tools / "lib/iss"
    (work / "q6ss.cfg").write_text(
        f"{iss}/qtimer.so --csr_base=0xFC900000 --irq_p=3 --freq=19200000 --cnttid=1\n"
        f"{iss}/l2vic.so 32 0xFC910000\n"
    )
    (work / "osam.cfg").write_text(
        f"{sdk}/rtos/qurt/computev79/debugger/lnx64/qurt_model.so\n"
    )


def run(runner: pathlib.Path, sources: List[str], headers: Dict[str, str] = None,
        includes_more: List[str] = ()) -> Dict[str, List[int]]:
    """Run a DSP runner and return its tagged fp16 bit patterns.

    headers are written into the build directory before compiling, which is how
    a blob produced on the host reaches the DSP without file IO.
    """
    sdk, tools = _check()
    work = pathlib.Path(tempfile.mkdtemp(prefix="hexagon-sim-"))
    try:
        for name, content in (headers or {}).items():
            (work / name).write_text(content)
        shared = build(runner, sources, work, includes_more)
        _configs(sdk, tools, work)
        env = dict(os.environ)
        libs = _sim_libs()
        if libs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                filter(None, [libs, env.get("LD_LIBRARY_PATH", "")])
            )
        command = [
            str(tools / "bin/hexagon-sim"), "-mv79na_1", "--simulated_returnval",
            "--usefs", str(work), "--mhmx=3",
            "--cosim_file", str(work / "q6ss.cfg"),
            "--l2tcm_base", "0xd800", "--subsystem_base", "0xFC90",
            "--rtos", str(work / "osam.cfg"),
            str(sdk / "rtos/qurt/computev79/sdksim_bin/runelf.pbn"), "--",
            str(sdk / "libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/run_main_on_hexagon_sim"),
            "--", str(shared),
        ]
        done = subprocess.run(command, capture_output=True, text=True, timeout=900, env=env)
        global LAST_STDOUT
        LAST_STDOUT = done.stdout
        if "Main() returned 0" not in done.stdout:
            interesting = [
                line for line in done.stdout.splitlines()
                if any(word in line for word in ("Error", "error", "symbol", "dlopen", "returned"))
            ]
            raise Unavailable(
                "hexagon-sim did not run the runner:\n" + "\n".join(interesting)[-3000:]
            )
        # A runner reports a named result per line: an uppercase tag followed by
        # the result as raw fp16 bit patterns.
        results: Dict[str, List[int]] = {}
        for line in done.stdout.splitlines():
            parts = line.split()
            if len(parts) > 1 and parts[0].isupper():
                try:
                    results[parts[0]] = [int(value, 16) for value in parts[1:]]
                except ValueError:
                    continue
        return results
    finally:
        shutil.rmtree(work, ignore_errors=True)
