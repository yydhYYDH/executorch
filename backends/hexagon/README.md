# Hexagon backend

Runs delegated subgraphs on the Qualcomm Hexagon DSP through FastRPC, with no
QNN and no Qualcomm AI Engine Direct anywhere in the path.

## Shape of the thing

Two halves, built by different toolchains, talking over a fixed interface:

| | built by | runs on | talks to |
|---|---|---|---|
| host | the normal ExecuTorch build | CPU | DSP, over FastRPC |
| skel | Hexagon toolchain, one `.so` per arch | DSP | nothing else |

The interface between them is the one MNN already uses: the `htp_ops.idl`
FastRPC interface plus the `command_generated` FlatBuffers schema. Both sides
are generated from those files by `qaic`, so handle numbers and struct layouts
cannot drift.

The DSP side is the MNN op library vendored under `third-party/mnn-htp-ops`
(provenance and the two local build fixes are in its `VENDORING.md`). It is
consumed unmodified: the only contract is

    execute_command_group(groupFd, groupOffset, count, syncFd, syncOffset, syncSize)

so one RPC carries a whole batch of ops. There is no per-op round trip and no
need for DspQueue.

## How a model gets to the DSP

**Partitioning (AOT, on the host, no device needed).** Ops the DSP can run are
tagged with `HexagonBackend` and handed to `BackendDetails.preprocess`, which
serializes a blob:

    [HexagonBlobHeader][HexagonOp × n_ops][weights][activations]

The blob is a plain packed layout (`serialization/hexagon_schema.h`), not
FlatBuffers, so the Python side needs no code generation. Weights are packed in
the layout the DSP kernels expect at this point; anything that needs a runtime
reorder is delegated to `htp_ops_weight_reorder` at init instead.

**Init.** `HexagonBackend::init` opens the FastRPC session, picks the skel for
the device's arch, and turns the blob into the wire format the DSP wants:

1. allocate one shared arena, copy the weights in,
2. build one `DSPCOMMAND::Command` FlatBuffer per op and one `SyncGroup`,
3. fill the command group array with `(arena_fd, command_offset, size)`.

Everything after this point is addressed by fd and offset only.

**Execute.** Copy the method's inputs into their arena slots, flush the host
cache, issue the single `execute_command_group` RPC, invalidate, copy the
outputs back out.

## Memory and cache protocol

The DSP reaches buffers through `HAP_mmap_get(fd)`, which resolves the fd to the
base of its FastRPC mapping. Consequences the host code has to respect:

- Every arena buffer is `rpcmem_alloc` + `fastrpc_mmap(..., FASTRPC_MAP_FD)`.
  Without the mmap the DSP has no address for the fd.
- `FASTRPC_MAP_FD` puts cache maintenance on the caller, so a host write is
  flushed before the DSP reads it and a DSP write is invalidated before the host
  reads it.
- `Alloc()` may return a pointer inside its mapping to satisfy an alignment, so
  every offset handed to the DSP is biased by `HexagonDriver::MappingOffset()`.

`rpcmem_cache_flush`/`rpcmem_cache_invalidate` exist on the device but are
absent from the SDK's link-time `libcdsprpc.so`, so the driver resolves them
with `dlsym`. When they are missing, the arena is allocated uncached
(`RPCMEM_FLAG_UNCACHED`) and the flush/invalidate calls become no-ops: slower
host access, but no chance of a stale cache line reaching the DSP.

The sync group carries the same information to the DSP side: it invalidates its
own view of the tensors on the way in and flushes them on the way out.

## Building

```sh
cmake -B build \
  -DEXECUTORCH_BUILD_HEXAGON=ON \
  -DHEXAGON_SDK_ROOT=/path/to/hexagon-sdk \
  -DHEXAGON_TOOLS_ROOT=/path/to/HEXAGON_Tools/19.0.04 \
  -DHEXAGON_ARCHS="v79"
```

`HEXAGON_ARCHS` is a list; each arch is cross-compiled into its own
`libhex-htp-skel-<arch>.so`, which is deployed to the device separately from the
ExecuTorch runner. The host driver loads it by name, so the two have to agree:
see `kSkelNameFormat` in `hexagon_driver.cpp` and `OUTPUT_NAME` in
`skel/CMakeLists.txt`.

The skel can also be built on its own, which is how it was brought up here:

```sh
cmake -S backends/hexagon/skel -B build-skel \
  -DCMAKE_TOOLCHAIN_FILE=${HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake \
  -DHEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT} -DHEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT} \
  -DMNN_OPS_ROOT=$(pwd)/backends/hexagon/third-party/mnn-htp-ops \
  -DSKEL_ARCH=v79 -DIDL_DIR=${IDL_DIR}
```

## Op contracts, and the traps in them

Parameters are positional and unchecked: a wrong order or a wrong tensor count
produces wrong numbers, not an error. These are the facts the emitters in
`hexagon_ops.py` are built on, each read from the dispatch in
`third-party/mnn-htp-ops/src/dsp/execute_command.cc` and the op wrapper.

| op | params | tensors |
|---|---|---|
| UNARY (4) | size in **elements**, op kind, element bytes | 1 in, 1 out |
| BINARY_ELEMENTWISE (19) | outSize, in0Size, in1Size, kind, bytes, inputBytes, inputIsFloat, outputIsFloat | 2 in, 1 out |
| SOFTMAX (28) | outside, channel, inside, bytes (must be 2) | 1 in, 1 out |
| LAYER_NORM (8) | outer, inner, **epsilon as float bits**, rmsNorm | src, gamma, beta; 1 out |

Things that bite:

- **Absent operands need `fd = -1`**, not size 0. The dispatcher maps a negative
  fd to a null pointer, which is the only way to say "no bias", "no gamma". A
  zero-size tensor still maps to a live address and is read as data. That is
  what `HexagonTensorSpace::kAbsent` and `blob.ABSENT` exist for.
- **Layer norm reads epsilon through a `float*` cast over the int params**, so
  the emitter bit-casts the f32 rather than rounding it.
- **Outputs are hard-coded by input count.** Layer norm writes
  `mapped_ptrs[3]` and binary elementwise `mapped_ptrs[2]`, so an extra input
  silently retargets the output.
- **Binary broadcasting is unreachable.** The DSP's broadcast path wants 25 more
  params than a command carries, so only same-shape and scalar operands work;
  the emitter raises instead of emitting a command that would leave the output
  stale.
- **q4a16 activations and outputs are pack64-blocked**, `[ceil(K/64)][M][64]` and
  `[ceil(N/64)][M][64]`, not row-major. This coincides with row-major only when
  `M == 1` or the dimension is at most 64. Prefill (`M > 1`) therefore needs a
  host-side repack, and that is the main reason the quantized matmul is not
  wired up yet.
- **`htp_ops_matmul_q4a16_fp16` returns success even when the kernel fails.** The
  block variant propagates the error; the plain one logs and returns 0.
- **The q4a16 weight tensor also carries its scales**: tiles of `icP*ocP*512`
  bytes followed by `ocP*32` fp16 scales, with `icP=(k+31)/32`,
  `ocP=(n+31)/32`. `DSP_OP_WEIGHT_REORDER_INT4` produces exactly that layout, so
  weight packing can be delegated to the DSP at init rather than reimplemented
  on the host.

## Status

Working and verified without a device:

- the vendored DSP library cross-compiles for v79 through the CMake build,
  exporting all 53 `htp_ops_*` entry points including `execute_command_group`,
  `matmul_q4a16_fp16`, `flash_attn`, `layer_norm` and `rope`;
- the host driver compiles clean under `-Wall -Wextra`;
- the AOT blob layout is byte-identical between the C++ reader and the Python
  writer: `sizeof(HexagonBlobHeader)` is 36, `sizeof(HexagonOp)` 476 and
  `sizeof(HexagonTensorRef)` 24, pinned by `static_assert` in the header and
  compared against `blob.OP_SIZE` by `test/test_blob_roundtrip.py`;
- the FlatBuffers commands the runtime builds decode back to exactly the type,
  operands and params that went in;
- the AOT pipeline runs: a graph is partitioned, `preprocess` emits a blob, and
  the blob decodes to the intended commands. For
  `softmax(sigmoid(x) * 2.0)` it produces three commands — unary sigmoid over
  64 elements, a flat binary multiply whose scalar operand is materialized as a
  one-element buffer, and a softmax reduced over `[1][64][1]` — with the last
  command writing straight into the output slot;
- on Qwen3-0.6B the partitioner takes 646 nodes and leaves 3383 on the portable
  kernels, including all 28 attention nodes. `STATUS.md` has the per-op table.

Not done yet:

- an end-to-end run on a device, and numeric parity against the CPU reference.
  Every kernel checked so far was verified by transcribing it and comparing
  against torch, which is not the same thing;
- of the 20 registered emitters only `mm`, `mul`, `add`, `neg` and
  `custom_sdpa` have produced a command on a real graph;
- attention delegates on fp32 operands that the runtime narrows to fp16 on the
  way into the arena, so the DSP runs fp16 attention. That trade is deliberate
  but unmeasured, and it means a working delegation is not yet a correct one;
- the fp32 normalisation segment — `mean`, `rsqrt`, `sigmoid` and the `_to_copy`
  nodes around them — stays on the CPU, because the vendored kernels take fp16
  in and out. `STATUS.md` records the measured kernel limits and why the fix is
  a fused RMSNorm kernel rather than more emitters;
- the quantized matmul path, which needs the pack64 repack described above;
- layer norm is not reachable yet: `to_edge` turns it into
  `native_layer_norm`, which returns three tensors, so it needs the aliasing
  machinery for multi-output ops before it can be delegated.

## Open design points

- **Zero-copy inputs.** `execute()` gets runtime-owned tensors, so today they are
  copied into the arena. Registering externally allocated buffers (as the QNN
  backend does with `QnnMem_register`) would remove that copy, but only if the
  producer can be told to allocate from rpcmem.
- **Persistent buffers.** KV caches have to survive across `execute()` calls;
  they belong in the arena but outside the per-call input/output slots.
- **Arch at init.** The skel is chosen from the device's arch at `init()`, so a
  single `.pte` runs on any arch, but weight layout is fixed then too. The vrmpy
  layout only pays off on v81+, so the AOT step may need to know the target.
