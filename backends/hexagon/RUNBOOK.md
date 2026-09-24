# Hexagon backend: build, deploy, run

Everything needed to go from a clean tree to inference on the DSP, in the order
it has to happen. Paths are the ones on the bring-up machine; adjust them.

Three artifacts have to be built and shipped: the **skel** (runs on the DSP), the
**runner** (runs on the phone's CPU), and the **.pte** (produced on the host).

## 1. Build the skel

The skel is the shared object the DSP loads. It is cross-compiled by the Hexagon
toolchain, not the NDK, and it is independent of the ExecuTorch host build.

```bash
HEXAGON_SDK=/home/yydh/hexagon-tutorial/tools/hexagon-sdk
HEXAGON_TOOLS=$HEXAGON_SDK/tools/HEXAGON_Tools/19.0.04
ET=/home/yydh/executorch
CMAKE=/home/yydh/miniconda3/envs/et/bin/cmake

mkdir -p /tmp/skelbuild && cd /tmp/skelbuild
$CMAKE $ET/backends/hexagon/skel \
  -DCMAKE_TOOLCHAIN_FILE=$HEXAGON_SDK/build/cmake/hexagon_toolchain.cmake \
  -DHEXAGON_SDK_ROOT=$HEXAGON_SDK \
  -DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS \
  -DMNN_OPS_ROOT=$ET/backends/hexagon/third-party/mnn-htp-ops \
  -DHEXAGON_FLATBUFFERS_ROOT=$ET/third-party/flatbuffers \
  -DSKEL_ARCH=v79 \
  -DIDL_DIR=/tmp/mport/gen2
$CMAKE --build . -j4
```

**Check the optimization flag landed.** `CMAKE_BUILD_TYPE` is empty here, so
nothing adds optimization on its own and the explicit `-O2` in
`skel/CMakeLists.txt` is the only thing that does:

```bash
grep -m1 CXX_FLAGS CMakeFiles/hex-htp-v79.dir/flags.make
# want: ... -mhvx -mhvx-length=128b -mhmx -O2
```

Without it the DSP dies with `0x8000040d` before reading a byte. See the README
for why.

Output: `ship/libhex-htp-skel-v79.so`.

## 2. Build the runner

Android arm64, NDK cross-compiled. `EXECUTORCH_BUILD_HEXAGON=ON` is what links
the backend in, and it needs `--whole-archive` or the linker drops the static
self-registration and every model fails with "Backend HexagonBackend is not
registered".

```bash
cd $ET
$CMAKE -B cmake-out-android-arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE=/home/yydh/android-ndk-r28c/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_HEXAGON=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON
$CMAKE --build cmake-out-android-arm64-v8a --target executor_runner -j4
```

## 3. Export the model

On the host, in the Python env (`conda activate et`). The partitioner is passed
directly rather than through the llama export path:

```python
from executorch.backends.hexagon.partition.hexagon_partitioner import HexagonPartitioner
from executorch.exir import to_edge_transform_and_lower

edge = to_edge_transform_and_lower(exported, partitioner=[HexagonPartitioner()])
```

Count what actually got delegated before shipping it, because a delegate census
counts *delegates*, not operators, and any op the partitioner declines stays
silently on the CPU:

```python
for n in edge._edge_programs["forward"].graph.nodes:
    if "executorch_call_delegate" in str(n.target):
        print("hexagon", n)
```

## 4. Deploy to the phone

```bash
R=/data/data/com.termux/files/home/csm/qwen3run
scp /tmp/skelbuild/ship/libhex-htp-skel-v79.so oneplus13-reverse:$R/skel/
scp $ET/cmake-out-android-arm64-v8a/executor_runner   oneplus13-reverse:$R/
scp model.pte                                          oneplus13-reverse:$R/
```

## 5. Run on the phone

In Termux. Both environment variables matter: the first finds the shared
libraries, the second is how the DSP's loader locates the skel.

```bash
cd /data/data/com.termux/files/home/csm/qwen3run
export LD_LIBRARY_PATH=$PWD/libs:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH=$PWD/skel
./executor_runner --model_path model.pte --num_executions 1
```

Useful flags: `--print_output none|summary|full`, `--output_file <base>` (writes
`<base>-0.bin`, raw tensor bytes), `--inputs <a,b,c>`.

**Do not verify numbers from the printed output.** `--print_output` wraps the
listing and parsing it with a regex silently mismatches; read the raw tensor
instead:

```bash
./executor_runner --model_path model.pte --output_file devout --print_output none
# devout-0.bin is numel * sizeof(dtype)
```

## 6. When something breaks

```bash
# DSP-side messages, including the skel's own error codes
adb logcat -s ExecuTorch:* adsprpc:*

# or, on the phone, watch the runner's stderr directly
./executor_runner --model_path model.pte 2>&1 | tail
```

| symptom | likely cause |
|---|---|
| `Backend HexagonBackend is not registered` | runner linked without `--whole-archive` |
| `0x8000040d` from `execute_command_group` | skel built without `-O2`; check `flags.make` |
| `0x8000040d` persisting after a rebuild | stale `.so` on the phone, or `ADSP_LIBRARY_PATH` wrong |
| numbers wrong but no error | op params are positional and unchecked; see the contract table in the README |
| `.pte` will not load | blob format changed; re-export, old blobs are not compatible |
| run prints nothing after `enter d0` and hangs for minutes | FastRPC's default is no RPC timeout, so a DSP-side hang never returns an error; the driver arms a 10 s timeout now — see `RPC_TIMEOUT_EXPERIMENT.md` |
| `0x0000000c` (`0x8000040c` from the DSP) | the RPC call exceeded the armed timeout (`AEE_EEXPIRED`); on a known-good model this means the timeout is too small |
| `rpc timeout NOT armed: ... failed: 0x80000414` | kernel does not implement the v2 notification struct (the SDK gates it to "after Kaanapali"); the run is otherwise unaffected |

## 7. Hangs, timeouts, and the 5 s SSR fallback

Three different clocks decide whether a stuck run ends, and only one of them used to
apply to us:

| what is stuck | what happens | where |
|---|---|---|
| a **framework** operation (mmap, unmap, pd kill) | SSR after **5 s**, automatically | `docs/examples/calculator/index.html`, SDK calculator README |
| an **RPC invoke** | nothing, ever — this is why a hang is silent and only a reboot clears it | device log prints `RPC timeout:0` at PD creation |
| an invoke, **after** `Open()` armed the timeout | returns `AEE_EEXPIRED` after **10 s** plus a `USERPD_TIMEOUT` notification, then a 2-step liveness probe reports whether the session survived | `hexagon_driver.cpp` (`ArmRpcTimeout`, `ProbeAfterTimeout`), `RPC_TIMEOUT_EXPERIMENT.md` |

The timeout is armed with `remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS, &remote_rpc_notif_register_v2, sizeof)`; `timeout` is in ms and `0` means
"not enabled". It is per-PD, so it covers every invoke this process makes on domain
3. The registration result is logged at startup — that line is the only way to tell
whether the kernel on the phone supports it.
