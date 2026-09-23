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

    [HexagonBlobHeader][HexagonOp × n_ops][weights][external weights][activations]

The blob is a plain packed layout (`serialization/hexagon_schema.h`), not
FlatBuffers, so the Python side needs no code generation. Weights are packed in
the layout the DSP kernels expect at this point; anything that needs a runtime
reorder is delegated to `htp_ops_weight_reorder` at init instead. What goes into
the blob and what is left outside it is decided by `HexagonCompileOptions` --
see [Tunables](#tunables) below.

**Init.** `HexagonBackend::init` opens the FastRPC session, picks the skel for
the device's arch, and turns the blob into the wire format the DSP wants:

1. check the compile specs against the blob, then allocate the delegate's
   resident block and copy the weights in, fetching any weight the blob left
   out from the named data map,
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

## Tunables

A hexagon knob has to be on one of two sides, and the side is decided by whether
it changes a byte of the blob:

- a **compile spec** (`HexagonCompileOptions`, `hexagon_backend.py`) is part of
  what the DSP will read, so it is fixed when the blob is built and travels in
  the `.pte`. A runner that flipped it at load time would be running a layout the
  program was never compiled for.
- a **backend option** (`HexagonBackendOptions`, `runtime/`) only changes what
  the delegate does with those bytes -- what it traces, what it caps, when it
  gives up -- so it can be set per load.

| spec | width | meaning |
|---|---|---|
| `hexagon_hmx_prepack` | 1 byte | weights are stored in the HMX tile order (default 1) |
| `hexagon_attn_paged` | 1 byte | attention reads a paged cache (default 0) |
| `hexagon_external_weights_max_bytes` | 8 bytes, little endian | per-weight inline threshold; a weight larger than this is left out of the blob |

The specs are a claim about the blob, and `init()` does not take the claim on
faith. `hexagon_compat.h` re-derives the same three facts from the bytes --
whether a matmul's flags carry the plan magic with the prepacked bit set,
whether an attention command is the paged entry point, whether the blob has an
external weights trailer -- and compares them. A program whose weights are
packed and whose spec says they are not fails to load with
`DelegateInvalidCompatibility` rather than running with the DSP reading the
wrong order. An unknown key or a payload of the wrong width is `InvalidArgument`
instead, because that is a broken caller rather than a broken program.
`hexagon_compat.h` includes no ExecuTorch, FlatBuffers or rpcmem header, so it
can be compiled and driven on the host, which is what
`test/test_compile_specs.py` does.

Options are resolved once per delegate at `init()`. Each one keeps the
environment variable it has always had as its fallback, so a process that sets
neither an option nor a variable behaves exactly as it did before options
existed:

| option | fallback | effect |
|---|---|---|
| `trace` | `HEXAGON_TRACE` | per-command trace |
| `delegate` | `HEXAGON_DELEGATE` | which delegate to trace, by execute order (`-1` = all) |
| `cmd_start` | `HEXAGON_CMD_START` | first command of that delegate |
| `cmd_limit` | `HEXAGON_CMD_LIMIT` | how many commands from there |
| `stop_after` | `HEXAGON_STOP_AFTER` | exit once delegate n has run |
| `fake_cache` | `HEXAGON_FAKE_CACHE` | fill an empty attention cache from K/V |
| `phase` | `HEXAGON_PHASE` | init/execute phase timeline |
| `kv_state` | `HEXAGON_KV_STATE` | keep the attention state in the arena |
| `watchdog_seconds` | `HEXAGON_WATCHDOG_SECONDS` | how long an invoke may block |
| `tile_budget` | `HEXAGON_HMX_TILE_BUDGET` | VTCM tile cap for matmul (`-1` = the blob's) |
| `acct` | `HEXAGON_ACCT` | per-stage accounting |

`tile_budget` is the one option that reaches into a command: it overwrites the
tile cap in a matmul's params at load time, because a cap is not a layout. An
unset option leaves the blob's own value alone. The two knobs that *do* change
the layout -- `HEXAGON_HMX_PREPACK` and `HEXAGON_ATTN_PAGED` -- are not options
any more but the specs above, and the runtime no longer reads them.

### Weights outside the `.pte`

With `hexagon_external_weights_max_bytes` set, every weight larger than the
threshold is left out of the blob, which gains a trailer naming them:

    [HexagonOp × n_ops][inline weights][{n_ext, {offset, size, key[32]} × n_ext}][activations]

The key is `hx` plus 29 hex digits of the weight's SHA-256, so two equal weights
share one entry in the store -- worth more than the `.pte` size on a model whose
layers tie tensors. `preprocess` returns them as a `NamedDataStore`
(`data_store_output`), the same path Vulkan's external constants take, and the
runner writes it as a `.ptd` beside the `.pte`. A blob with nothing externalized
is still version 2 and byte-identical to what the writer produced before this
existed; only a blob that left something out is version 3.

At `init()` each key is read with `NamedDataMap::get_data` and copied into the
resident block, at the offset the trailer names. What this saves is the size of
the `.pte` and the store-level dedup. What it does not save is the rpcmem copy:
the weight still has to be in the block the DSP reads from, so a model that
externalizes everything moves the same bytes at load time that it did before.
Removing that copy needs a runner that can hand the delegate a file-backed
mapping the DSP mappings itself, which is a new contract with the runner rather
than a change here.

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
| MATMUL_Q4A16_GEMV_I8 (41) | (unused), K, N, 0…0, **scale block count**, asymmetric | activation, weight+scales, bias; 1 out |
| MATMUL_W8A16_GEMV_I8 (45) | (unused), K, N, 0…0, **scale block count** | activation, weight, scales, bias; 1 out |

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
- **The fp16 quantized matmul's activations and outputs are pack64-blocked**,
  `[ceil(K/64)][M][64]` and `[ceil(N/64)][M][64]`, not row-major. This coincides
  with row-major only when `M == 1` or the dimension is at most 64, which is why
  prefill needs a host-side repack. The integer GEMV entries (41, 45) that the
  M == 1 path uses read and write linearly instead, so they need no repack --
  see "Quantized matmuls" below.
- **`htp_ops_matmul_q4a16_fp16` returns success even when the kernel fails.** The
  block variant propagates the error; the plain one logs and returns 0. The two
  GEMV entries propagate and are checked by `execute_command.cc`.
- **The fp16 q4a16 weight tensor also carries its scales**: tiles of
  `icP*ocP*512` bytes followed by `ocP*32` **fp16** scales, with
  `icP=(k+31)/32`, `ocP=(n+31)/32`. `DSP_OP_WEIGHT_REORDER_INT4` produces exactly
  that layout, so weight packing can be delegated to the DSP at init rather than
  reimplemented on the host. The **integer** GEMV entries use **fp32** scales,
  which is what `hexagon_ops.pack_q4a16_gemv_weight` appends; the two layouts are
  not interchangeable.

## Quantized matmuls

Weight-only quantization lands on the two integer GEMV entries, for `M == 1`
(decode). The AOT side is `quantizer.py`, the pattern and packers are in
`hexagon_ops.py`:

```
quantizer = get_hexagon_quantizer("q4a16")          # or "w8a16"
prepared  = prepare_pt2e(exported.module(), quantizer)
prepared(*calibration_inputs)
converted = convert_pt2e(prepared)                  # -> dequantize_per_channel
blob      = HexagonBackend.preprocess(to_edge(exported).exported_program(), [])
```

What the graph says and what the DSP does are different, and this is the part
worth reading twice:

- **The annotation is weight-only.** No observer is inserted on an activation,
  so activations keep the width the graph was exported at (the runtime narrows a
  fp32 operand to fp16 at the arena boundary) and the weight is symmetric,
  per-output-channel, int4 or int8.
- **The arithmetic is int8 x int4/int8.** Both kernels quantize the fp16
  activation *inside the kernel*, per token: absmax over the row, scale
  `absmax/127`, round, clamp to `[-127, 127]`. There is no calibration for it
  and no static activation scale anywhere. The scheme names (`q4a16`, `w8a16`)
  describe the operand widths at the command boundary, not the multiply.
- **True static-scale w8a8 is not reachable from here.** No vendored kernel
  takes an int8 activation tensor, and the runtime reads every non-`Float` arena
  entry as two bytes per element. A w8a8 graph would need both of those to
  change.
- The accuracy cost is therefore the weight error *plus* the per-token int8
  activation error, and `test/test_hexagon_quantizer.py` measures it against the
  dequantized reference instead of assuming it away.

The granularity is one fp32 scale per output channel, which is what the kernel
sees as `scale_block_num == 1`: it derives `blocksize = K/nblk`, so that one
block spans all of K. The kernel's own producer contract (`nblk = K/blocksize`)
describes 64-element group quantization instead, which would be a different
scale operand; nothing here builds that.

Delegated only when all of these hold, since the emitter is past the partition
boundary and cannot fall back: `M == 1`; `K % 64 == 0` and `N % 32 == 0` (the
kernels' own guards, which they answer with an error code); a symmetric
per-channel dequantize whose every reader is a runnable quantized matmul; and,
for `addmm`, `alpha == 1`, `beta` in `{0, 1}` and a bias of exactly `n` values
(the kernel adds `n` contiguous halfs and would otherwise read past the
operand). Prefill (`M > 1`) needs the pack64 activation and output repack, so it
stays on the portable kernels.

Verified offline: the pattern is matched, the weight is packed at export, the
command decodes, and `test/blob_interpreter.py` runs the bytes with the same
arithmetic the kernels use. Not verified: no DSP has executed either entry, and
the packers' tile orders are transcriptions -- the int4 one is written down
twice in the vendored tree and cross-checked against the kernel's read path, the
int8 one only against the kernel's own permuted activation splat. The speedup,
the real numbers and the FastRPC/skel deployment all need a device.

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
  behind;
- the compile-spec check is exercised on the host over real blobs, from both
  sides: the writer's side in `test/test_compile_specs.py` and the runtime's in
  the ET-free `hexagon_compat.h`, which the same test compiles and drives. That
  is where the plan-magic mask in the checker was caught reading a packed weight
  as unpacked;
- the runtime options are resolved against a real `BackendInitContext` on the
  host (`test/test_runtime_options.py`), including the fallback to the variable
  each one has always had and the refusal of a value of the wrong type;
- the external-weights path round-trips without a device: the `.ptd` a
  `preprocess` returns deserializes to the same bytes the blob left out, a blob
  that externalizes nothing is byte-identical to one written before the option
  existed, and a blob whose weight is missing or short is refused
  (`test/test_external_weights.py`);
- a weight-only quantized matmul goes all the way through: the PT2E-annotated
  graph partitions into one delegate, `preprocess` emits one
  `MATMUL_Q4A16_GEMV_I8` (41) or `MATMUL_W8A16_GEMV_I8` (45) command, and the
  host interpreter reads the blob back and reproduces the kernels' arithmetic
  within a few percent of the dequantized reference. See "Quantized matmuls" for
  what that arithmetic is and what is still unverified.

Not done yet:

- an end-to-end run on a device. A kernel is checked by building a real blob,
  running it on hexagon-sim and comparing the result three ways against torch and
  a host model of the same command stream. The simulator is a functional model,
  so it says what the kernels compute and nothing about what the DSP costs or
  whether the FastRPC path and the skeleton deployment work;
- the external-weight copy at `init()` and the tile-budget override run only
  inside the delegate, so they are code-only here. The `.pte`/`.ptd` pair is
  checked on disk instead: `test_external_weights.py` writes a real `.pte`, reads
  the `.ptd` back and pins the exact account `len(external .pte) ==
  len(inline .pte) - weight + trailer` (9860 - 8192 + 72 = 1740). That test
  needs the generated `exir/_serialize/{program,scalar_type}.fbs`, which are
  gitignored and have to be copied into a fresh worktree; without them it skips
  rather than passing silently;
- of the registered emitters, the ones that have produced a command on a real
  graph are `mm` (including the quantized weight-only form), `bmm`, the binary
  and unary families, `custom_sdpa`, `rms_norm`, `mul_silu`, `update_cache` and
  the narrowing blits. The view, cast, getitem and dequantize emitters have run
  as well but emit nothing by design;
- attention delegates on fp32 operands that the runtime narrows to fp16 on the
  way into the arena, so the DSP runs fp16 attention. That trade is deliberate
  but unmeasured, and it means a working delegation is not yet a correct one;
- the fp32 normalisation segment is gone from the delegated graphs: the
  `mean`/`rsqrt`/`sigmoid` chain is fused into an `rms_norm` op and the casts
  around it are absorbed, since the arena holds fp16 and the vendored kernels
  take fp16 in and out;
- the quantized matmul path is wired up for `M == 1` only. Prefill needs the
  pack64 activation and the output repack, and neither is built: the GEMV
  entries wired up here read a single row linearly, which is why they needed no
  repack at all;
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
