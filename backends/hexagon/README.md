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

1. allocate the delegate's resident block and copy the weights in,
2. build one `DSPCOMMAND::Command` FlatBuffer per op and one `SyncGroup`,
3. fill the command group array with `(resident_fd, command_offset, size)`.

Everything after this point is addressed by fd and offset only.

**Execute.** Copy the method's inputs into their scratch slots, flush the host
cache, issue the single `execute_command_group` RPC, invalidate, copy the
outputs back out.

## Memory and cache protocol

A delegate's memory is two blocks, because a model can hold hundreds of
delegates and the DSP's rpcmem heap holds a few gigabytes:

- the **resident** block is written once at init and read on every execute:
  command descriptors, sync group, command group and weights. It stays private
  to the delegate.
- the **scratch** block holds the method inputs, the activations and the method
  outputs. Nothing in it outlives an execute -- inputs are copied in and outputs
  are copied out, both inside `execute()` -- and a graph runs one delegate at a
  time, so `SharedArenaPool` keeps one block per size and hands the same block
  to every delegate that needs that size. Subgraphs repeat shapes, so a model
  with hundreds of delegates over a handful of shapes holds a handful of blocks.

The DSP reaches buffers through `HAP_mmap_get(fd)`, which resolves the fd to the
base of its FastRPC mapping. Consequences the host code has to respect:

- Every block is `rpcmem_alloc` + `fastrpc_mmap(..., FASTRPC_MAP_FD)`. Without
  the mmap the DSP has no address for the fd.
- `FASTRPC_MAP_FD` puts cache maintenance on the caller, so a host write is
  flushed before the DSP reads it and a DSP write is invalidated before the host
  reads it.
- `Alloc()` may return a pointer inside its mapping to satisfy an alignment, so
  every offset handed to the DSP is biased by the offset the `Arena` it returns
  carries. Tensors and command entries each name the fd of the block their
  offset is measured in.

`rpcmem_cache_flush`/`rpcmem_cache_invalidate` exist on the device but are
absent from the SDK's link-time `libcdsprpc.so`, so the driver resolves them
with `dlsym`. When they are missing, the blocks are allocated uncached
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

## The skel must be built with optimization

`skel/CMakeLists.txt` sets `-O2`, and it is not a performance preference. The
DSP kernels are written as one function per operator kind carrying every branch
of that kind inline, and unoptimized every temporary they would keep in a
register is spilled to the stack instead: a single fp16 branch chain asks for a
14848-byte frame (`allocframe(#0x3a00)`), which does not fit the RPC thread's
stack. The fault is in the prologue, so it happens before the kernel reads a
byte, reports as `0x8000040d` from `remote_handle64_invoke`, and takes the
whole user PD down with it. Nothing about it points at the build.

Two things made that hard to see, and both are worth remembering:

- `-fstack-usage` on the file compiled standalone reports 512 bytes, not 14848.
  `hexagon-llvm-objdump -d` on the built `.so` tells the truth. When the frame
  size is the question, read the disassembly.
- more stack is genuinely available than the failing frame needs at first glance,
  so measuring headroom is not enough on its own: 4608 bytes of recursion
  succeed one call below a frame that wants 14848.

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
  writer: `sizeof(HexagonBlobHeader)` is 36, `sizeof(HexagonOp)` 480 and
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
- on Qwen3-0.6B the partitioner takes 1967 nodes into 29 subgraphs -- one per
  layer, with all 28 attention nodes among them -- and leaves 825 on the
  portable kernels, 711 of which are shape guards. With the fusion passes
  switched off the export splits into 169 subgraphs and leaves 2121 nodes
  behind.

Not done yet:

- an end-to-end run on a device. A kernel is checked by building a real blob,
  running it on hexagon-sim and comparing the result three ways against torch and
  a host model of the same command stream. The simulator is a functional model,
  so it says what the kernels compute and nothing about what the DSP costs or
  whether the FastRPC path and the skeleton deployment work;
- of the 20 registered emitters, the ones that have produced a command on a real
  graph are `mm`, the binary and unary families, `custom_sdpa`, `rms_norm`,
  `mul_silu`, `update_cache` and the narrowing blits. The view and cast
  emitters have run as well but emit nothing by design;
- attention delegates on fp32 operands that the runtime narrows to fp16 on the
  way into the arena, so the DSP runs fp16 attention. That trade is deliberate
  but unmeasured, and it means a working delegation is not yet a correct one;
- the fp32 normalisation segment is gone from the delegated graphs: the
  `mean`/`rsqrt`/`sigmoid` chain is fused into an `rms_norm` op and the casts
  around it are absorbed, since the arena holds fp16 and the vendored kernels
  take fp16 in and out;
- the quantized matmul path, which needs the pack64 repack described above;
- softmax is delegated on its last axis only. The kernel's strided path, for a
  reduction over any other axis, disagrees with torch on hardware: `[1,2,4,8]`
  reduced over dim 1 came back with 8 of 64 elements past 1e-2, the worst by
  1.1e-1, where the last-axis form is exact to 4.9e-4. `softmax_reduces_the_inner_axis`
  keeps that form off the delegate until the kernel is checked.

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
