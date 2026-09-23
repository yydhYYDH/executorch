# Hexagon backend setup

Everything beyond a normal ExecuTorch build. See `README.md` for what the
backend does; this file is only about getting a build and a test run.

## What the build needs

| dependency | used for | where to get it |
|---|---|---|
| Qualcomm Hexagon SDK | `qaic` (FastRPC stubs), QuRT headers, `libcdsprpc.so`, `hexagon-sim` | Qualcomm developer network |
| Hexagon Tools | `hexagon-clang++`, the per-arch DSP libc++ | shipped under `HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/` |
| Android NDK | the host driver and runner for an Android device | Android NDK r26+ |
| `flatc` | regenerating the DSP command schema | built by the ExecuTorch build (target `flatc`) |

The Hexagon toolchain is not redistributable and is not fetched by ExecuTorch.
Point the build at a local install through the variables below.

## Build variables

| variable | meaning |
|---|---|
| `EXECUTORCH_BUILD_HEXAGON` | turns the backend on (default `OFF`) |
| `HEXAGON_SDK_ROOT` | required when the backend is on |
| `HEXAGON_TOOLS_ROOT` | Hexagon Tools root; defaults to `${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/<version>` in `build.sh` |
| `HEXAGON_ARCHS` | `;`-separated DSP arches to build, default `v79` |
| `HEXAGON_DSPRPC_LIB` | `libcdsprpc.so` used at link time, default under `HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64` |

The host build runs `qaic` from `${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu` and
`flatc` from this tree. Both `hl` and `bin` for the DSP side are cross-compiled by
an `ExternalProject`, so the Hexagon toolchain never has to share a CMake
configure with the host toolchain.

## Build the host driver

```sh
cmake -B build \
  -DEXECUTORCH_BUILD_HEXAGON=ON \
  -DHEXAGON_SDK_ROOT=/path/to/hexagon-sdk \
  -DHEXAGON_TOOLS_ROOT=/path/to/hexagon-sdk/tools/HEXAGON_Tools/19.0.04 \
  -DHEXAGON_ARCHS="v79"
cmake --build build --target hexagon_backend hexagon_skel_v79
```

For an Android target, add the NDK toolchain file and `-DANDROID_ABI=arm64-v8a`,
as `build.sh` does. `build.sh skel`, `build.sh runner` and `build.sh all`
wrap the two configurations; every path it uses is overridable from the
environment.

## Run the tests

```sh
pytest backends/hexagon/test -q
```

Only the Python/AOT tests and the blob round-trip run without hardware. The
kernel tests build a real Hexagon shared object and run it under `hexagon-sim`,
so they need `HEXAGON_SDK_ROOT`; without it they `pytest.skip`. `hexagon-sim`
additionally wants `libncurses.so.5`, which recent distributions dropped: point
`HEXAGON_SIM_LD_LIBRARY_PATH` at a directory that provides it. The C++ runner
sources under `test/` are compiled directly by `test/hexagon_sim.py`, not by the
CMake build.

## Deploy to a device

The skel is pushed separately from the runner; the host driver loads it by name
(`kSkelNameFormat` in `runtime/hexagon_driver.cpp`), so the two have to be from
the same build.

```sh
DEVICE=<adb-or-ssh-host> MODEL=<model>.pte ./build.sh deploy
```

Set `ADSP_LIBRARY_PATH` to the directory holding
`libhex-htp-skel-<arch>.so` and `LD_LIBRARY_PATH` to the runner's libraries
before starting `executor_runner`.
