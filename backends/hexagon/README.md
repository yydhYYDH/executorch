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
(provenance, the two local build fixes and the profiling instrumentation are in
its `VENDORING.md`). It is consumed unmodified: the only contract is

    execute_command_group(groupFd, groupOffset, count, syncFd, syncOffset, syncSize)

so one RPC carries a whole batch of ops. There is no per-op round trip and no
need for DspQueue.

## How a model gets to the DSP

**Partitioning (AOT, on the host, no device needed).** Ops the DSP can run are
tagged with `HexagonBackend` and handed to `BackendDetails.preprocess`, which
serializes a blob:

    [HexagonBlobHeader][HexagonOp × n_ops][weights][external weights][activations]

The blob is a plain packed layout (`serialization/hexagon_schema.h`), not
FlatBuffers, so the Python side needs no code generation. Weight packing is done
on the host as the blob is built: in particular, `pack_q4a16_prefill_weight`
produces the int4 prefill tile order consumed by its kernel. No DSP weight-reorder
command is emitted at init. What goes into the blob and what is left outside it is
decided by `HexagonCompileOptions` -- see [Tunables](#tunables) below.

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

## Profiling

A whole subgraph is one delegate call, so an ETDump of a hexagon model would
otherwise say only that `HexagonBackend` took X ms. The backend closes that gap
from both ends:

- **AOT.** `preprocess` returns `PreprocessResult.debug_handle_map`, keyed by the
  index of each DSP command in the blob, which is what names the graph node a
  command came from. The blob format is untouched: the mapping travels in the
  ETRecord, not in the `.pte`.
- **Runtime.** `execute()` logs a span per phase -- `HEXAGON_EXECUTE`,
  `HEXAGON_COPY_IN`, `HEXAGON_RESIZE`, `HEXAGON_DSP_CALL`, `HEXAGON_COPY_OUT` --
  and then one event per DSP command, with that command's index as its delegate
  debug identifier. The times are the DSP's own `HAP_perf_get_time_us` deltas,
  measured around each kernel.

The events land in whatever ETDump the runner already produces, so a profiling
run is an ordinary run:

```sh
python3 -m devtools.inspector.inspector_cli \
    --etdump_path model.etdump --etrecord_path model.etrecord
```

With the ETRecord the per-op rows resolve to node names; without one they are
numbered by command index. `delegate_debug_metadata` carries the fields the
event has no room for, little-endian:

| event | bytes | fields |
|---|---|---|
| one per command | 12 | `op_type` (DSPOpType), `microseconds`, `ret` |
| `HEXAGON_DSP_CALL` | 8 x n | `{op_type, microseconds}` per op type the call ran |

The second layout is the per-op-type accumulation `execute_command.cc` already
keeps, which needs nothing of the skel, so a group run on an older skel still
reports where its time went. That skel also cannot report individual commands:
it says so by leaving the probe header's version int at zero, and the per-op
rows are then simply absent rather than zero.

Worth knowing:

- the DSP's clock is its own, so per-op events are laid out along the host's span
  for the call, in the order the commands ran. Durations are the DSP's; the
  absolute times place them inside `HEXAGON_DSP_CALL`.
- the probe buffer that carries the per-command data costs 21 KB of rpcmem per
  delegate and a cache clean per command on the DSP, so it is armed only when a
  tracer is attached or `HEXAGON_TRACE` asks for it. Nothing else pays.
- the records cover the first 508 commands of a delegate; past that the phase
  events and the op-type totals are all there is.
- the Inspector scales a delegated row by the same factor as any other, taking
  the delegate's timestamps to be on the source time scale. These are on it
  (`pal_current_ticks`, nanoseconds), so a proxy run needs nothing extra; a
  delegate that logged in some other unit would need a
  `delegate_time_scale_converter`.
- `HEXAGON_TRACE=1` keeps its stderr trace, now with each command's time when
  the skel reports one.

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

## Tests

```sh
pytest backends/hexagon/test                                          # this backend's own tests
PYTHONPATH=src python backends/hexagon/scripts/gen_op_support.py      # regenerate OP_SUPPORT.md
```

The directory is in `pytest.ini`'s `testpaths`, so the generic CI unittest job
runs it. The cases that need the SDK's toolchain or its simulator skip when it is
absent, so the run is meaningful on a machine without the SDK.
`test/test_op_support.py` is the guard that keeps `OP_SUPPORT.md` equal to what
`gen_op_support.py` renders from `hexagon_ops.EMITTERS`; add an emitter and that
test fails until the table is regenerated.

`test/tester/` is this backend's half of the shared operator suite: the suite
collects a backend's cases through a flow object, and the tester is what that
flow drives. It lowers through `HexagonPartitioner` and skips the run stage,
because a serialized program needs the DSP. The flow that instantiates it is
registered in the suite's own directory and is not part of this change, so no
`-m flow_hexagon` selection collects anything from here yet. Running a case
instead of skipping it needs a runner built with `EXECUTORCH_BUILD_HEXAGON=ON`,
the matching skel on the device, and a way to hand the `.pte` over; see the
execution policy note in `test/tester/tester.py`.

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
| UNARY (4) | size in **elements**, op kind, element bytes, and two more when the kind carries operands: clamp's fp16 bounds and row_guard's row length and masked value (mul_scalar's fp32 scale is one) | 1 in, 1 out |
| BINARY_ELEMENTWISE (19) | outSize, in0Size, in1Size, kind, bytes, inputBytes, inputIsFloat, outputIsFloat | 2 in, 1 out |
| SOFTMAX (28) | outside, channel, inside, bytes (must be 2) | 1 in, 1 out |
| REDUCTION (29) | outside, reduce, inside, kind (sum 1, maximum 2, mean 3), bytes | 1 in, 1 out |
| POOL2D_FP16 (1) | batch, ih, iw, oh, ow, **c4**, kY, kX, sY, sX, pY, pX, padType, countType, poolType | 1 in, 1 out |
| LAYER_NORM (8) | outer, inner, **epsilon as float bits**, rmsNorm | src, gamma, beta; 1 out |
| SHARED_GATHER (23) | selectSize (rows out), ic (row width), oc (table rows), bytes (2), isInt4 (0) | in0 = **indices**, in1 = table; 1 out |
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
  silently retargets the output. SHARED_GATHER takes the output at
  `mapped_ptrs[inputs->size()]`, so its two operands have to stay in the order
  above and no more may be added.
- **The pool kernel reads its activation pack64-blocked, and `c4` counts channel
  blocks rather than channels.** `hvx_pool2d_fp16` addresses element `(n, c, y,
  x)` at `((c//64)*batch + n)*ih*iw*64 + (y*iw + x)*64 + c%64`
  (`ops/pool_fp16.c:22`), and the dispatch's sixth param is `ceil(C/64)` -- the
  pool command carries **fifteen** ints, not twelve (`execute_command.cc:324`).
  The arena holds row-major NCHW, so the emitter brackets the command with two
  `RASTER_BLIT`s, one into that layout and one back out; both are the geometry
  the DSP's own pack-area fast paths take (`blit_ops.cc:843-857`). They are free
  only when the spatial extent is one, where the two layouts agree element for
  element. A channel count that is not exactly one block is refused rather than
  emitted with padded lanes.
- **A target with no emitter is not a refusal, and looks like one.**
  `SUPPORTED_TARGETS` is `EMITTERS`, and the partitioner only ever sees a node
  whose target is in it. An op that is missing from the table is therefore never
  rejected -- it is never considered, no message is printed, and the only
  symptom is a delegate that does not appear. The visible cost so far has been
  three ops that a sibling in the same family already covered:
  `aten.mean.default` (where `aten.mean.dim` was wired), the reduce-all
  `aten.max.default` (where `aten.amax.default` was wired), and
  `aten.relu.default`, `aten.hardtanh.default` and `aten.pow.Tensor_Scalar` with
  exponent 2 (where `aten.clamp.default` and the unary table's clamp and square
  entry points were wired). `test/test_overload_census.py` is the census for the
  rest of them, one row per overload, and it fails in both directions -- when an
  unwired target gains an emitter and when a wired one starts refusing --
  `test/test_overload_census2.py` extends the same shape to the families that
  census listed as unexamined (the quantized path, `llama.*`, `et_hexagon.*`, the
  recurrent ops, the remaining getitem producers, `_to_copy` against
  `_to_dim_order_copy`), and `test/test_unwired_targets.py` checks the reporting
  below.
- **The partitioner now counts that case itself.** `is_node_supported` records
  every target that is absent from the table while *its family* is in it --
  `unwired_overload_census()` for a test or a script to read, and one `DEBUG`
  line on `executorch.backends.hexagon.partition.hexagon_partitioner` for a
  reader. Both run only on the path that was already returning `False`, so a
  delegated node pays neither, and `test/test_unwired_targets.py` pins that the
  verdicts are identical with the counting replaced by a no-op. The family is the
  schema name without the overload, which is exactly the class of gap above, and
  which is blind by construction to a name the exporter rewrites from one to
  another: a per-tensor `dequantize` is not the per-channel one the table speaks
  for, and `aten.amin` is not the `aten.amax` it speaks for, so neither is
  counted and both are pinned row by row in the second census instead.
- **The partitioner counts the refusals of the ops it does have, too.** The case
  above is a target the table is missing; the other half of the same complaint is
  a target the table *has* and that a gate turned away anyway -- a geometry, a
  dtype, an operand width or a group count the kernels here cannot run. Those
  look supported to a reader of the table and to whoever reads a delegate count,
  and the gates that can refuse one are spread over a dozen predicates and a dozen
  argument checks, so nothing prints. `refused_overload_census()`, keyed by
  target, and one `DEBUG` line report them, from the same place and on the same
  terms as the unwired counter: only on the path that was already returning
  `False`, and with a no-op control pinning that no verdict moves. The boundaries
  themselves -- which channel count, which group count, which index dtype, and
  which reader rule puts a two-output node back on the host -- are §4 of
  `OP_GAPS.md`: each gate is stated there with the line that decides it and the
  measurement behind it, and not here.
  `test/test_refused_targets.py` pins each boundary in both directions, and it
  builds its `HexagonOperatorSupport` the way `HexagonPartitioner.partition`
  does, with `_data_placeholders` of the program: an object built without them
  has an empty `data_names`, so a parameter reads as an ordinary placeholder and
  every gate that wants a constant weight refuses more than the partitioner does.
  That is stricter, never looser, so it can only invent a refusal -- but a census
  row built on one would be exactly that invention.
- **Three edge ops are the same clamp entry point, and it carries its bounds in
  params.** `HTP_OPS_UNARY_CLAMP` (`unary_ops.cc:29`) is not a function of the
  op kind alone: params[3] and params[4] are the fp16 bit patterns of its two
  bounds, and the dispatcher routes it to `htp_ops_clamp_fp16_chunk` instead of
  the rest of the table. torch computes `hardtanh` as `clamp` and gives
  `min_val`/`max_val` the slots `min`/`max` occupy, so one emitter covers
  `clamp.default` and `hardtanh.default`; `relu` is that clamp between zero and
  infinity, which is the `max(x, 0)` torch computes; and `relu6` is
  `hardtanh(x, 0, 6)`. The kernel's compares are unordered, so a NaN would come
  back as the upper bound, and it restores the input where `|x| > 0x7c00`
  instead (`unary_ops.cc:505-509`, and the scalar tail below it since this
  branch) -- that bit test is the whole reason relu may
  be spelled this way, since `add_relu(x, 0)` is the form whose NaN behaviour is
  not documented anywhere.
- **`torch.max(x)` and `torch.max(x, dim)` are different ops.** The reduce-all
  form is a value and nothing else, and its target is `aten.max.default`; it
  reaches the same REDUCTION command `amax` already reached. The `dim` form is
  `aten.max.dim`, a two-output node whose second output is indices: its values
  are that same command over the same span -- the overload's `dim` is a single
  int, so the one-span rule always holds -- and it is placed under the rule the
  pool is under, every reader taking `getitem 0` (`max_dim_is_emittable`), with
  that getitem as the sink the command fills. A graph that reads the indices
  keeps the whole node portable, values reader included, because the kernel
  computes values and not positions. `torch.max(x)` and `torch.amax(x)` agree in
  every value and differ only in which zero a signed-zero input returns and in
  the payload of a NaN, which is the byte-level caveat `fmod` already carries and
  which the `dim` form carries against `torch.amax(x, dim)`.
- **The reduction enum has no minimum.** `HtpOpsReductionOpType` is `sum = 1,
  maximum = 2, mean = 3` (`eltwise_ops.cc:2441-2445`) and the dispatcher rejects
  every other value, so `aten.amin` is not reachable by writing an emitter: it
  needs a kernel that does not exist, and stays on the portable kernels.
  `argmax`/`argmin` are in the same position, since the kernel returns values
  only. Sum and maximum are one command each, over a single contiguous span of
  the buffer, and `torch.mean(x)` is the widest span of the same kind:
  `[1][numel][1]`, which is the kernel's `inside == 1` path.
- **`fmod` is the truncated remainder, `remainder` is not.** `HTP_OPS_BINARY_MOD`
  computes `a - trunc(a/b)*b` with a zero divisor answering zero
  (`eltwise_ops.cc:148-166`), which is torch's `fmod`; the floored remainder
  agrees with it only when the dividend is non-negative, so `aten.remainder` is
  refused. Where the remainder is exactly zero the kernel lands on `+0` and torch
  on `-0` for a negative dividend: equal, not bit-identical.
- **Binary broadcasting is limited to eight dimensions.** The broadcast command's
  25-entry tail plus its 9-int head uses 33 or 34 of the 40-int budget, so
  ranks through 8 fit. Rank 9 has no representation and stays on the portable
  kernel.
- **The fp16 quantized matmul's activations and outputs are pack64-blocked**,
  `[ceil(K/64)][M][64]` and `[ceil(N/64)][M][64]`, not row-major. This coincides
  with row-major only when `M == 1` or the dimension is at most 64, which is why
  prefill needs a host-side repack. The integer GEMV entries (41, 45) that the
  M == 1 path uses read and write linearly instead, so they need no repack --
  see "Quantized matmuls" below.
- **The gather's table is not row-major, and the tiling is not optional.** The
  fp16 path reads a grid of 32x32 tiles in which the column pairs inside a tile
  come first: element `(row, col)` of the tile at `(row // 32, col // 32)` sits at
  `((col % 32) // 2) * 64 + (row % 32) * 2 + ((col % 32) & 1)`
  (`shared_gather_ops.cc:296-311`). There is no row-major mode: `isInt4 = 0` is
  the only value that reaches an unquantized table, and it reaches this layout.
  So the table is rearranged at export by `pack_shared_gather_table`, one pass,
  and a tile costs 1024 elements whether the table fills it or not.
- **A tiled table is padded to whole tiles, so it can be larger than the weight.**
  It costs `ceil(oc/32) * ceil(ic/32) * 2048` bytes: equal to the row-major size
  when both sides are multiples of 32, and 32x when the row width is one element.
  A 1024-wide token embedding -- the case this exists for -- is exactly its
  row-major size.
- **The gather's indices are read as `int32`, so an `int64` index tensor stays on
  a portable kernel.** The kernel takes `const int32_t*`, and a method input keeps
  the width its own dtype declares, so five int64 tokens would read as five low
  words interleaved with five zero high words. `gather_table` refuses the node
  instead, which leaves `nn.Embedding` on the CPU for a graph whose tokens are
  int64 -- the dtype `torch.export` gives you by default. Adding a
  `tokens.to(torch.int32)` in the model is enough: the cast is not delegated, so
  it stays on a portable kernel and only the gather after it reaches the DSP.
- **An index outside the table clears the row rather than raising**
  (`shared_gather_ops.cc:292-295`). Torch's `embedding` raises there. The count
  and the table come from the graph and the index comes from the caller, so a
  token outside the vocabulary is a zero row on the DSP and an error on the CPU.
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

- **The two convolution kernels take the same blocked activation and different
  weights.** `hvx_conv_depthwise2d_fp16` reads `[c4][ky][kx][64]` and walks the
  window from `oy * stride - pad` with `+ky * dilate`, accumulating each product
  in fp16 from a bias vector it always dereferences; `hmx_im2col_convolution_fp16`
  reads `[ceil(oc/32)][ky*kx*ceil(ic/32)][1024]`, the 32x32 tiles the HMX unit
  takes, and adds the bias after the product. Both were read out of the sources
  and then run: `test/sim/conv_runner.cpp` builds their operands the way the
  emitters do, prints a digest of the packed weight, and the host side reproduces
  that digest through `pack_depthwise_weight` and `pack_conv_weight` before
  comparing the output against torch bit for bit.
- **The im2col kernel has to be given `mp = 1, np = 2`.** With the default
  single-tile chunking a ragged position tile reaches `store_output_tile_fp16`,
  whose leftover path rotates the accumulator *after* adding the bias, so the
  tile's upper half is handed the neighbouring channel tile's bias: for a 3x3 s1
  p1 convolution over 5x5 with 64 channels the simulator returns channels 32..63
  of the last position off by `bias[l] - bias[32 + l]`, and nothing else moves.
  The pair store (`store_output_tile_pair_fp16`) has no such path, and every
  shape in `test/sim/conv_runner.cpp` is exact with it; `CV_ODD` is that failure
  case and `test_the_single_tile_chunking_is_what_the_emitter_avoids` pins it.
  This is a defect in the vendored kernel, worked around from the emitter rather
  than fixed in place.
- **The lanes past the input's last channel are part of the product.** The fill
  reads whole 32-lane groups, so with three input channels the k lanes 3..31 of
  the first group are multiplied by whatever the weight holds there. Setting the
  weight's padding lanes to 1 and the activation's to 3 moves the simulator's
  answer by exactly 29 lanes times 3 per in-range tap. The weight's padding lanes
  are zero by construction, and the activation's are cleared by a `DSP_OP_ZERO`
  command ahead of the pack when the channel count is not a multiple of 64; that
  clear is defensive, since the simulated unit flushes `0 * inf` to zero and this
  is not a property to rely on from a device.
- **The im2col weight and activation fills reach VTCM by DMA, which copies a
  misaligned operand wrong rather than failing.** Every tensor in a blob is laid
  out on a 128-byte boundary, and `test/sim/blob_runner.cpp` had to place its
  arena on one too: without it the whole-blob pointwise convolution came back with
  5938 of 6144 elements wrong while the host model of the same commands was
  exact. This is why the alignment is a property of the arena rather than of a
  tensor in it.
- **Bias is never a null operand.** The depthwise walk reads a 64-lane bias vector
  unconditionally and the im2col store reads one starting at every 32-channel
  group's first lane, so a convolution with no bias still gets a zero-filled
  buffer, long enough for the last group to run a whole vector past the channel
  count.

## Quantized matmuls

Weight-only quantization lands on the two integer GEMV entries for `M == 1`
(decode) and, for the int4 weight, on `MATMUL_Q4A16_FP16` (22) for `M > 1`
(prefill). The AOT side is `quantizer.py`, the pattern and packers are in
`hexagon_ops.py`:

```
quantizer = get_hexagon_quantizer("q4a16")          # or "w8a16"
prepared  = prepare_pt2e(exported.module(), quantizer)
prepared(*calibration_inputs)
converted = convert_pt2e(prepared)                  # -> dequantize_per_channel
blob      = HexagonBackend.preprocess(to_edge(exported).exported_program(), [])
```

Which matmuls `annotate` reaches is the set the emitters can run rather than the
set of matmuls in the graph, because an annotation the emitters refuse is not a
fallback but a failed export: it leaves a `dequantize_per_channel` in the
portable part of the graph, and `quantized_decomposed` has no out variant for
`to_executorch` to convert it to (`RuntimeError: Missing out variants`). The
guards below are therefore asked of the operands before the observer goes in,
through the same helpers the emitters use (`weight_only_matmul_fits`), so an
admitted annotation is one the partitioner delegates.

- `mm`, `addmm`, and the two-dimensional `matmul` they are the lowering of. A
  `matmul` with a batch axis in front of its operands is not one of these: it
  only becomes a row-major `mm` at `to_edge`, where the batch is folded into the
  `M` the entries above have an opinion about. A `matmul` whose operands are two
  run-time tensors has no weight at all: annotating one would leave the run-time
  quantize beside the dequantize, and neither has an out variant.
- `nn.Linear` and `F.linear`, which is what a model's projections are written
  as. Its weight is stored `[out, in]`, so `to_edge` lowers it to an `addmm` over
  a `permute_copy` of the weight -- the dequantize would sit behind that permute,
  which is a shape no emitter matches. `transform_for_annotation` rewrites an
  admissible one into the `addmm` spelling over a `[k, n]` constant before the
  observers go in, so the entry a caller reaches by writing `nn.Linear` is the
  same one `mm` reaches, byte for byte (`test_blob_on_sim.py` compares the two
  blobs).
- Anything else: a weight the export does not own, an activation that is strided
  or has a dynamic axis, and every shape the guards below turn away, including a
  classifier head whose `N` is not a multiple of 32. What such a layer gets is
  the fp16 path it would have had without a quantizer.

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
- **The accumulator's int32-to-fp32 conversion was wrong past `2^22`, and this
  tree carries the fix.** Both `M == 1` kernels converted the accumulator with
  the `1.5 * 2^23` magic number, which is exact only while `|x| <= 2^22`. The
  comment that justified it assumed the accumulator was bounded by one 64-wide
  block, but both emitters pass `scale_block_num == 1`, so the scale covers the
  whole `K` -- past `2^22` from `K = 512` at int8 weights and from about
  `K = 4.7k` at int4 ones. Past it the conversion returned a wrong number, an
  infinity at larger `K`, while the kernel still returned 0: measured on the
  simulator with constant weights, `K = 512` at int8 answered 3032 against 2032,
  and `K = 8192` at int4 answered 2552 against 1792. A random-sign weight
  cancels, so the rungs in `test_gemv_on_sim.py` need a coherently signed
  column. The fix splits each lane into its high and low 16 bits and recombines
  them, which converts every int32 correctly and leaves every answer inside
  `2^22` bit for bit what it was -- measured rather than argued: over all `2^32`
  int32 values this conversion is the correctly rounded fp32 everywhere, and the
  old sequence's first failure is at `2^22 + 1`. This is the `M == 1` GEMV path
  only -- it says
  nothing about the prefill entry -- and the defect is pre-existing in the
  vendored kernels rather than something this work introduced; the upstream
  paths are in the commit that fixed it.
- **Both GEMV entries stage their operands in VTCM and refuse with `-1` when
  there is none.** `matmul_q4block_gemv_i8.c:323-325` takes the size-zero path
  out of the kernel before it has written a byte, so the output is left as it
  was: an empty answer that reads as a wrong one, with only the return code to
  say so. On the device the delegate's setup acquires VTCM and the dispatcher
  re-acquires it when it is missing (`execute_command.cc:995`); a caller that
  brings its own arena has to do the same, which is what the simulator's runners
  had to learn (`test/sim/blob_runner.cpp`).
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
boundary and cannot fall back: `K % 64 == 0` and `N % 32 == 0` (the kernels' own
guards, which they answer with an error code); a symmetric per-channel
dequantize whose every reader is a runnable quantized matmul; and, for `addmm`,
`alpha == 1`, `beta` in `{0, 1}` and a bias of exactly `n` values (the kernel
adds `n` contiguous halfs and would otherwise read past the operand). Which
entry a matmul reaches is the `M` it carries: `M == 1` is the GEMV pair, and
`M > 1` is the prefill entry for both the int4 and int8 weight layouts. The
int8 prefill command is 42, with its fp16 per-channel scale tail appended to the
packed weight. The quantizer asks these same conditions, through
`weight_only_matmul_fits` and the bias check, before it annotates anything, so
a width with no prefill emitter is refused at the annotation rather than left to
fail downstream, where a dequantize nobody can lower is a program that will not
serialize.

### Prefill

`M > 1` is one command where the GEMV path is one command, and three where the
layouts differ:

```
K == 64     [ 22 ] [ 3 ]            # the blocked activation layout is the row-major tensor
K > 64      [ 3 ] [ 22 ] [ 3 ]      # pack the activation, matmul, repack the output
```

The kernel reads fp16 activations directly and dequantizes the weight itself, so
neither the per-token int8 activation nor the `scale_block_num == 1` contract of
the GEMV entries applies. The int4 entry uses 512-byte tiles of 32 output
channels by 32 k values, followed by fp16 per-channel scales; this is what
`pack_q4a16_prefill_weight` builds and what the vendored reorder writes, and
the two are compared byte for byte on the simulator
(`test/test_prefill_on_sim.py`). The int8 entry uses 1024-byte HMX tiles,
followed by the same fp16 per-channel scale tail; `pack_w8a16_prefill_weight`
builds that operand for command 42. The activation goes into
`[k/64][m][64]` and the output comes back out of `[n/64][m][64]`, which is what
the two blits are for.

Two things about this entry are worth knowing before touching it:

- **The output-channel chunk count is two, never one.** That is not the shape
  the allocator would have picked for a single 32-channel tile: on the simulator
  one chunk answers NaNs at `K == 64` (any N), and two answer exactly. The
  vendored entry corrects an odd chunk count above one and leaves one alone
  (`matmul_q4fp16_mle32.c:723`).
- **The dispatch wrapper swallows the kernel's return code.** It logs a FARF and
  returns 0 (`matmul_ops.cc:27-40`), so a failing prefill looks like a successful
  command with the output left as it was; the interpreter and the simulator's
  runner keep the code for that reason.

Verified offline: the pattern is matched, the weight is packed at export, the
command decodes, and `test/blob_interpreter.py` runs the bytes with the same
arithmetic the kernels use. The prefill entry goes further than that, because
its weight layout had an authority to check a packer against: the packer's bytes
equal the vendored reorder's on the simulator, both kernels answer
layout-blind expectations bit for bit across both dispatch branches, and a whole
blob built by the emitter agrees three ways (torch, the host interpreter and the
simulator) on six shapes. Not verified at the time of writing: no device had
executed any of these entries, so VTCM as a device sizes it, alignment and the
multi-worker DMA staging the simulator leaves out are all unmeasured; and of the
three packers here, two are transcriptions -- the int4 GEMV one is written down
twice in the vendored tree and cross-checked against the kernel's read path, the
int8 one only against the kernel's own permuted activation splat -- while the
prefill one is the byte-for-byte comparison above. That gap has since narrowed
for the entry itself: a device run at `M > 1` is in "Status", which measures the
numbers and the K at which the `m <= 32` path stops. The alignment and the
multi-worker DMA staging the simulator's missing worker pool leaves out are still
unmeasured, and the speedup is still unmeasured.

## Delegating a row gather

`embedding`, `index_select(dim=0)` and `index.Tensor` over a single axis are the
same read: k rows out of a table that lives in the weights section, one
SHARED_GATHER command. The target case is the token table of an LLM, which is the
largest single weight in the model. Three things have to hold, and
`gather_table` is the one place that decides them:

- the table is a parameter, buffer or lifted constant, because its bytes are
  tiled at export and a table that only exists at run time has none to tile;
- the indices are an `int32` tensor, for the reason above;
- the read is over axis 0 of a contiguous 2-D table, and the indices are at most
  one moving axis (the exported sequence symbol).

Anything else is refused in the support check, so it stays on a portable kernel:
a refusal that happened while emitting would fail the whole export instead.

A table only this gather reads is stored tiled and nothing else: `preprocess`
writes the row-major copy of every weight first, and `BlobBuilder.build` drops
every weight no command reads, so the file carries one copy rather than two. A
table that is *also* a matmul's weight -- tied embeddings, where `lm_head` shares
`embed_tokens.weight` -- does keep both, since the matmul reads rows and the
gather reads tiles: expect the file to grow by one table's size there.

Where the numbers here come from: `pack_shared_gather_table` and the interpreter's
`untile_shared_gather` are two independent readings of `shared_gather_ops.cc`, and
`test/test_shared_gather.py` checks them against each other, against offsets
derived by hand from that source, and end to end against `torch.embedding` through
`blob_interpreter`. What none of that covers is the device: see below.


## Delegating a vision tower's attention

A vision transformer's attention is the decomposed form: split the projections
into heads, `q @ k.T * scale`, `softmax`, `@ v`, merge the heads back. Lowered as
it stands, a ViT block is three partitions -- the three projections, the
attention core (two `BATCH_MATMUL` and the softmax), and the output projection
-- with all four head transposes left on the portable kernels, because a permute
at a partition boundary is not something the backend can keep. One
`VISION_ATTENTION_FP16` replaces the middle of that, so the shape is worth
having; what it costs is that the kernel reads its operands in the layout the
*cache* path uses, and the two are not the same.

`htp_ops_vision_attention_fp16` (`third-party/mnn-htp-ops/src/dsp/attention_entry.cc:18`)
is unmasked, non-causal, and walks a token-major source: a token's rows are
`heads * headDim` apart (`:36`), and the query, key, value and
output rows are all `((b * tokens + t) * heads * headDim) + h * headDim + d`
(`:41`, `:43`, `:57`, `:61`). That is `[batch, tokens, heads, headDim]`, not the
`[batch, heads, tokens, headDim]` a `bmm` in torch already holds. An emitter
cannot paper over that difference -- the head transpose is not a detail of the
command, it is part of which tensor the command is handed -- so the operands have
to arrive already token-major, which means the fusion has to consume the
transposes rather than sit after them.

That is what `vision_attention.py` does. `et_hexagon.vision_attention` is the
whole pattern stated once, with the operands *under* the head splits, and
`FuseVisionAttention` (opt-in, via `transform_passes`) replaces the transpose
that follows the second matmul with that one node. Its operands are then exactly
the `[batch, tokens, heads, headDim]` views, its own output is that layout, and
the caller's dim-order copy and view back to `[batch, tokens, embed]` are
aliases that emit nothing. In a ViT block the whole thing becomes one delegate
with ten commands -- the three projections, their bias adds and their head-split
blits, the attention, and the output projection -- where before the same block
was three partitions and four portable permutes.

Two things are refused where they are seen rather than in the emitter, because a
refusal during emit fails the whole export:

- **a mask, and a causal bias.** The command binds three operands, so the
  dispatcher hands the kernel a null mask pointer, so `maskStride > 0`
  (`attention_entry.cc:49`) never passes and no mask is read. The emitter could
  write any stride it liked and it would never be consulted -- which is exactly
  how masked attention came to be silently ignored in the FLASH_ATTN emitter,
  where `mask_stride = -1` looks like a mask is being passed and means the
  opposite. Here there is no operand for a mask to arrive in, and the fusion
  pattern refuses the shapes a mask would have (`_Attention(masked=True)` and
  `is_causal=True` both stay portable).
- **anything but a square attention.** Query and key runs share one `tokens`
  param, so a cross-attention with different lengths is another function.

`scaled_dot_product_attention` decomposes to the same matmuls plus `row_guard`,
which is where the masked-row guard lives. The kernel returns NaN for a row of
all `-inf` and the guard returns 0, so the guard is not the identity on the
inputs attention can actually see, and the pattern refuses it: a graph written
with `F.scaled_dot_product_attention` does not fuse even when it is not causal.
The explicit `(q @ k.T) * scale -> softmax -> @ v` form, which is what
`CLIPAttention` and `SiglipAttention` write, does.

`batch`, `heads` and `headDim` reach the command as params, so they have to be
static; `tokens` is patched from the run-time sequence length the way the matmul
patches its rows, and the workspace is sized for the longest export. The
workspace is not optional: the kernel aligns the pointer up by 127 bytes and
refuses the command without `tokens * 4 + 128` (`:26-31`), which is why the
command has two outputs and the second one is scratch.

Where the numbers here come from: the interpreter's
`vision_attention_kernel_offset` is a transcription of the kernel's own index
expression and `vision_attention_row_major_offset` is the same index derived from
the shape, sharing no term with it, and `test/test_vision_attention.py` checks
them against each other over a shape with four different extents, checks that the
head-major reading would disagree, and then runs the real command stream end to
end against torch's own answer. What none of that covers is the device: see
below.


## Folding a batch norm into the convolution before it

`FoldBatchNormIntoConv` (opt-in, via `transform_passes`) applies a batch norm that
follows a convolution to that convolution's weights and drops the node. There is
no `DSP_OP_*` for a normalization, so a batch norm not only falls back to a
portable kernel but splits the delegate chain around it: a stack of
conv/BN/relu blocks lowers to one delegate per convolution plus one for the
trailing rectifier, and every split in between is a round trip through the CPU.
Over four such blocks the fold takes 5 delegates to 1 -- the phone's runner then
prints a single `enter d0: ops=20` where it printed `ops=4`, `ops=5`, `ops=5`,
`ops=5` and `ops=1` -- with the same twenty commands in the same one blob, and
the `.pte` 90,756 -> 86,308 bytes. The delegate bytes are identical to those of
the same weights fused by hand and written as a model with no batch norm at all,
on the host and on the phone.

The rewrite is
`backends/transforms/fuse_batch_norm_with_conv.FuseBatchNormWithConvPass`, imported
rather than reimplemented. Two things it leaves to its caller are done here, and
both are measurable: the unfused convolution weight and bias are pruned
(without that the `.pte` grows to 117,028 bytes on the fixture above, carrying a
dead fp32 copy of every convolution weight), and the output specs are re-pointed
when the folded batch norm is itself the graph's output (without that the export
fails with `User output aten_convolution_default is not in the correct order`).
The precondition upstream does not check is the training flag: it matches
`aten.native_batch_norm.default`, whose fifth argument says whether the node
means the batch statistics, and folding one that does would be a wrong answer
with no error anywhere. A graph holding such a node, a convolution a residual
branch also reads, and a batch norm with `affine=False` all keep their node.

Every number above is measured against batch norms that are *not* the identity.
A freshly constructed `BatchNorm2d` scales by `1/sqrt(1+eps)`, which is 1-5e-6:
folding that is a no-op that a broken fold would also pass, and
`test_fold_batch_norm.py` has a guard that keeps its own fixtures away from it.

Where the numbers come from: the count of delegates, of portable batch norms and
of commands is read from the lowered program and from the blob, not from whether
the whole graph's output matched torch -- a graph can agree with torch while an
op never reached the DSP. The device numbers are the phone's own
`[hexagon] enter d0: ops=N` lines around the outputs it wrote, and the checks are
run separately on host lowering, in `hexagon-sim`, and on the phone.


## Folding a weight's preparation before the split
`FoldConstantTransposes` turns a weight written as a chain of permutations
into the tensor that chain computes, once at export, into the blob's weight
section. It is registered on `transform_for_pre_decomposition`, which EXIR
calls ahead of the split, so no caller has to opt in with `transform_passes`
-- and `partition` cannot do it, because EXIR asserts that call leaves the
graph module unchanged. The hook runs on the ATen program before
decomposition, where a weight's preparation is spelled `aten::permute`,
`aten::reshape` and `aten::flip` and only the convolution's second argument
matters, so the pass matches schema names across both dialects; the walk
replays the graph's own targets rather than a table of equivalents, leaves
every chain that does not end at a constant alone, and the run-time-weight
case is tested from the lowering rather than from the pass's return value. It
is what carries a FIR upsampler's transposed convolution: without the fold the
weight is a computation `conv_spec` cannot read and the node stays portable.
The geometry that reaches the DSP includes both a `groups == 1`
transposed convolution and a grouped transposed convolution. A grouped
`FirUpsample2D` still folds its weight, then the host slices each group and
emits one dense im2col walk per group. A plain forward convolution remains
restricted to `groups == 1` or the genuine depthwise shape.

**Unresolved: the folded blob has never been executed on the simulator or a
device.** The simulator fixture that runs blobs builds its fx graph by hand
and never goes through the partitioner, and the convolution simulator's
weights come from a formula in its own case table, so a folded weight cannot
reach either. What the simulator does cover is the kernel that consumes it --
the 35 cases of `test_conv_sim.py`. Closing this needs a simulator runner that
takes weight bytes, or a device run of the whole `.pte`.

## Status

Working and verified on a device. The whole of this list is one OnePlus 13
(SM8750, Android 15, CDSP, v79), `executor_runner` built from this branch at
`10ab195` and the matching `libhex-htp-skel-v79.so`, in fp16, with every
reference recomputed from the same module instance that produced the `.pte` it is
compared against -- a separate script that re-creates the module draws different
weights after `torch.manual_seed(0)`, which cost this work two rounds of a
"wrong" device answer that was a wrong reference:

- a three-delegate convolution graph (`conv3x3`, a depthwise `conv3x3`, `relu`,
  `maxpool`, `conv1x1`) runs all three rounds and lands within one fp16 ULP of
  torch -- 4.88e-4 against a reference peaking at 0.57. Only part of that graph
  reached the DSP: the partitioner left the depthwise `conv3x3` and the
  `maxpool` outside the delegates, so what the three command streams contain is
  two im2col convolutions and both rectifiers -- four of the graph's six nodes.
  The whole-graph agreement
  therefore rests on those ops running on silicon with the other two on the
  portable kernels, which is a different claim from "this graph ran on the DSP";
- `embedding` gather, `add`, `amax(dim=1)` and `sum(dim=1)` in one delegate
  (`ops=3`): the `amax` is bit-for-bit torch's answer, the `sum` is one ULP out at
  1.56e-2 against a reference of 16.9. An `embedding` on its own is not delegated
  when its indices are int64 and lands on the portable kernels instead -- not
  because a gather cannot be delegated but because the kernel reads
  `const int32_t[]` and the partitioner refuses the dtype. The same table with
  int32 indices reaches one delegate, which is the form the simulator comparison
  below runs;
- single-op `add`, `amax` and `sum` graphs each reach one delegate; `add` and
  `amax` answer bit-for-bit, and `sum` is 5.86e-3 out, one and a half ULP of its
  largest element;
- the FP16 convolution at `in_channels` 64, 512, 2048 and 4832, and the VTCM gate
  measured from both sides: the widest window the emitter delegates answers within
  half an ULP and the first one it refuses fails with `0x8000040d` in under
  0.2 s;
- `relu`, `relu6` and `hardtanh(-1, 1)` on sign-clear and sign-set NaNs, and the
  fused rectifier on the same -- see the clamp and `add_relu` paragraphs, where
  silicon keeps the NaN the simulator loses on one path and loses the one it loses
  on the other;
- the reductions' NaN contract, which turns out to depend on the width of the
  fold -- see the reduction paragraph;
- `DSP_OP_ZERO` (24), in the command stream of the widest convolution, executed
  without error;
- **whether the simulator stands for silicon, which is a separate measurement from
  whether the device matches torch.** One blob per kernel family, extracted from
  the `.pte` the phone ran and handed to `hexagon_sim.run` unchanged, run both
  ways and compared as raw fp16 bits. Nine cases over seven paths are identical,
  including the 3x3 convolution -- the one that goes through the HMX unit, where
  the simulator's `--mhmx=3` is a second implementation of that hardware and not a
  recompilation of the first. The details are with the evidence tiers below;
- the transport and the deployment: `[hexagon] rpc timeout NOT armed ...
  0x80000414` is the first line of every run and changes nothing, every answer is
  byte-identical across repeats, and 430 runs of the convolution graph in three
  batches came back `exit d0: ok` in all but one. That one, and two others
  outside those batches, are the CDSP contention the RUNBOOK describes, not a
  wrong number: the round fails with `0x00000012` after 10.1 s
  when another process on the phone holds the VTCM, which is a shared device's
  problem rather than this backend's answer.

A later set of device runs, on the same phone and the **same unchanged
deployment** (`libhex-htp-skel-v79.so` md5 `db74410ba0f9a44ad8e6ed2d2d4421ec`,
the runner from 24 Sep, blobs lowered at `abd68f1` -- four commits after the
`10ab195` above, which is why these are listed separately rather than folded into
that list):

- **the element-wise select** (`WHERE`, `DSP_OP_SELECT` 26) answers torch **bit
  for bit** from one command, both with the condition as a graph input and with
  it baked in as a constant. A control that changes only the one-byte condition
  width -- same command, same blob length, `params[5]` moved from `1` to `2` --
  goes wrong on 12 of 24 elements, so the branch really does depend on the width
  the emitter declares rather than on the command merely having run;
- **the zero-filling `constant_pad_nd`** (`ZERO` 24 plus one `RASTER_BLIT`)
  answers torch bit for bit on all 27 outputs. Its control is the sharper of the
  two: moving the region's `dstOffset` by one leaves the blob the same length,
  still reports `enter d0: ops=2` and still reports **`exit d0: ok`**, and writes
  13 of 27 outputs in the wrong place. `exit d0: ok` therefore certifies that a
  command ran, never that the numbers are right;
- **`split`/`chunk`** writes its pieces with `RASTER_BLIT` and answers bit for
  bit, including the case where the region the third piece reads is offset by the
  extent of a middle piece nothing reads;
- **the int4 weight prefill entry (22) has now run on a device at `M > 1`**,
  which the offline sections below still say it has not. `[3][22][3]` at
  `M=8 K=128 N=96` answers within a relative 4.5e-4 of the converted graph's own
  fp32 `mm` over the *dequantized* weights -- the comparison that keeps 4-bit
  weight rounding out of the kernel's error budget. A K ladder at `M=4`, `N=64`
  holds that accuracy up to **`K = 12672`** (relative 3.9e-4 to 6.4e-4 at every
  rung) and then fails at **`K = 12736` and `K = 12800`** with
  `execute_command_group failed: 0x8000040d`, `exit d0: failed` and no output at
  all, in under a second and reproducibly across three interleaved repeats. That
  this is structural rather than the contention above is the clock and not the
  code: contention lands at 10.1 s, and this lands in under one;
- **the boundary belongs to the `m <= 32` branch, and the arena is not what
  runs out.** Holding `K = 12800` and changing only `M` puts the edge exactly on
  the dispatcher's own split (`matmul_ops.cc:28`): `M = 2`, `4` and `32` all fail
  the same way while `M = 33` and `M = 40` answer within 6.7e-4. The arenas run
  the other way from the outcome -- `M = 2` fails with 51 KB and `M = 40`
  succeeds at `K = 25216` with 2.0 MB, forty times the failing request -- so the
  width of the request, the arena and the VTCM budget are all excluded, and what
  is left is the one branch. Within it, the only stack object on the whole call
  path that grows with K is `dma_desc_2d_t act_descs[safe_kp]`
  (`matmul_q4fp16_mle32.c:528`), which is exactly K bytes because `safe_kp` is
  `K/32` 32-byte descriptors; the boundary is in K (12672 passes, 12736 does
  not), so whatever overflows is K-sized. `0x8000040d` is the signature
  `skel/CMakeLists.txt:44-47` already records for a frame that overflows the
  **DSP RPC thread's stack**, and it is measured there at 14 KB -- against the
  12.7 KB this boundary sits at.
- **the failure is a fault and not a returned error, which is a stronger
  statement than it looks.** `htp_ops_matmul_q4a16_fp16` keeps the kernel's
  return code in a local, logs it and then `return 0`s (`matmul_ops.cc:27-42`),
  and the kernel's own out-of-VTCM path is a returned `AEE_ENOMEMORY`
  (`matmul_q4fp16_mle32.c:739-741`). So a VTCM overrun here would be **silent**:
  `exit d0: ok` over an output left as it was. What this does instead is abort
  the process (`rc=134`, `exit d0: failed`, no output at all, in under a second),
  which is the shape of a fault rather than of a refusal.
- **the host gate was wrong in its M, not in its K, and now refuses rather than
  emitting it.** `_quantized_prefill_fits` used to admit `K = 25216` and ask
  nothing about `M`, yet `q4a16_m40_k25216` **passes** at that K (relative 7.1e-4):
  the ceiling itself is sound for the shape the gate was reasoned about, and what
  was missing is the other dimension, since the same K that works at `M = 40`
  destroys `M <= 32`. The gate now carries that branch's bound
  (`PREFILL_M32_MAX_M = 32`, `PREFILL_M32_MAX_K = 12672` in `hexagon_ops.py`),
  stated against the largest K seen to work rather than the first that failed.
  Moving the descriptor array off the stack, as the `m > 32` kernel already does,
  is the other fix and is still not in this tree.
- **a caveat, because a control was attempted here and it did not work.** The
  obvious way to separate the VLA length from K is to rewrite `params[7]`, which
  is the `kp` the emitter sends -- but the kernel recomputes `int kp = K / 32`
  inside the worker (`matmul_q4fp16_mle32.c:447`) and `kp_max` reaches only a
  validation at entry (`:687`), so `params[7]` never sizes the array. Both
  directions were run against the phone and neither moved the result (raising it
  to 420 at `K = 512` still passed with byte-identical output; lowering it to 100
  at `K = 12800` still aborted). That experiment is therefore **null and not a
  refutation**: it shows the parameter does not reach the mechanism, so the VLA
  attribution above rests on the branch comparison and the eliminated silence
  path rather than on a direct control. Settling it needs a skel change -- the
  array moved to the heap, as the `m > 32` kernel does -- which this section
  leaves undone.
- **the transposed convolution** runs its six commands -- `ZERO`, a weight blit,
  `ZERO` again, the zero-insert interleave, one `IM2COL_CONVOLUTION_FP16` and the
  output repack -- without error, and answers torch within a relative 6.6e-4 on a
  `(1, 8, 4, 4)` fp16 input at `stride 2, padding 1`. The convolution that
  consumes the interpolated plane had not been run even on the simulator, so this
  is the first time that path met either. It is also the first entry here whose
  blob was checked to be tree-independent: the same `.pte` and the same device
  bytes come out of the branch this work landed from and out of this tip, which
  is what makes a device result on the branch a result on the tip.

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
- an `embedding` reaches the DSP: a 64x8 table with five int32 tokens lowers to
  one delegate whose blob carries one SHARED_GATHER command, the table spanning
  exactly `ceil(oc/32) * ceil(ic/32) * 2048` bytes of the 4096 in the weight
  section, and the host interpreter reproduces `torch.embedding` on it bit for
  bit. `index_select(dim=0)` and `index.Tensor` take the same path, and the
  count of tokens gathered is patched from the runtime sequence length the way
  the matmul patches its rows;
- on Qwen3-0.6B the partitioner takes 1967 nodes into 29 subgraphs -- one per
  layer, with all 28 attention nodes among them -- and leaves 825 on the
  portable kernels, 711 of which are shape guards. This count predates the mask
  handling in `_sdpa_fits_dsp_limits`: an attention node now carries its mask
  into the delegate where the kernel applies it, and stays portable wherever the
  kernel would drop it (a query extent of one, a run-time query length or
  stride, more than 64 rows), so an export with masks delegates a count that
  follows those geometries.  The one-row clause is deliberately a step wider
  than the case that motivates it. Inside that case's window -- `qo_len == 1`,
  `seq_current == 0`, `seq_add == 1` -- the kernel attends a single key, so a
  finite mask changes no number there; what the shortcut would drop is a mask
  that hides that one key entirely. A decode step past the first has
  `seq_current != 0`, leaves the shortcut and applies the mask normally. The
  gate cannot narrow to the first step, because the position reaches the command
  at run time in `params[1]` (`runtime/hexagon_backend.cpp:1790-1825`), so every
  single-row query that carries a mask stays portable.  With the fusion passes
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
  what that arithmetic is and what is still unverified. **Both GEMV entries have
  now run on hexagon-sim** and answer the closed form bit-for-bit: an activation
  of one everywhere on weights whose every column reaches full scale, so the
  answer is `sum_k q[k, n] * scale[n]` with no tile order, nibble order or scale
  position anywhere in the expectation; and an activation of one at a single k,
  which answers row k of the stored weight and so moves if the packer and the
  kernel disagree about which weight belongs to which contraction index -- the
  closed form is blind to a permutation of k, that one is not. Both shapes a
  stride could be wrong about (64x32 and 128x64), both entries, and the biased
  form; two controls assert the opposite, a pair-exchanged nibble order and a
  column-major tile table each have to move the answer, and they do. The
  exhaustive half is `test/test_gemv_on_sim.py`, which walks every k of both
  layouts and measures what `unpack_vrmpy_weight_128` does to a byte instead of
  reading the comment that says what it should do. Two things about these entries
  are **not verified**, which is not the same as verified absent: the chunked
  weight path (`matmul_q4block_gemv_i8.c:320+`, the split it takes when the
  weights do not fit the VTCM budget) has **never been reached** by any case
  here -- every shape used fits one chunk -- so it is unexercised rather than
  correct; and no case measures the timing, on the simulator or anywhere else;
- the int4 weight prefill entry (22) does too, at every layer below the device
  and, since then, on one as well (see "Status"):
  `test/test_prefill_on_sim.py` compares its packer with the vendored reorder on
  the simulator byte for byte, measures the two HVX operations that reorder is
  built on, and runs both dispatch branches against expectations that mention no
  tile, nibble, scale position or row order, at eight shapes plus five more in
  the sweep that found the one-chunk behaviour below; `test/test_blob_on_sim.py`
  runs whole blobs the emitter built through torch, the host interpreter and the
  simulator and requires all three to agree. One case in
  `test_prefill_on_sim.py` is a **defect pin rather than a feature test**: one
  output-channel chunk answers NaNs at `K == 64`, which is why the emitter always
  asks for two. Not verified: the multi-worker DMA staging the
  simulator's missing worker pool leaves unexercised (`qurt_cb_fwk_worker_init
  returned: -4`) has never run, and the device run in "Status" says what the
  numbers are and where the `m <= 32` path stops rather than anything about that
  staging, which no device case here reaches;
- a vision attention block goes all the way through once `FuseVisionAttention` is
  in the caller's `transform_passes`: a three-projection ViT attention over a
  dynamic patch count partitions into one delegate whose blob carries one
  `VISION_ATTENTION_FP16` (43) command with the geometry the graph had, three
  token-major operands and the fp32 workspace the kernel requires. The host
  interpreter runs that command and reproduces the module to 3.1e-4 in fp16. See
  "Delegating a vision tower's attention";
- pooling, `sum`, `amax` and `fmod` reach the DSP, blob and numbers included. A
  `max_pool2d` or `avg_pool2d` over 64 channels lowers to one delegate whose blob
  carries a blit into the kernel's blocked layout, one POOL2D command and a blit
  back out; the host interpreter reproduces torch exactly on maxima and to fp16
  precision on both average divisor modes, for kernel, stride and padding
  combinations that hang off every edge. `sum` and `amax` each lower to one
  REDUCTION command, with the three span params pinned for an inner, outer,
  middle, multi-axis, negative, missing and empty dim, and `amax` returns torch's
  values exactly. `fmod` lowers to one BINARY command with subtype 12 and
  reproduces the truncated remainder, and the floored `remainder` is pinned as
  not delegated. `FuseAddReluPass` rewrites `relu(x + y)` into the one
  `BINARY_ELEMENTWISE` command that does `max(a + b, 0)`, the only command here
  that adds and rectifies in one step; without the pass the pair delegates as an
  add and a unary clamp, so the pass buys a command rather than a round trip.
  `FuseCumsumPass` does the same for a prefix scan, which the reduction table does
  not have: `cumsum(x, -1)` along the last axis becomes one `BATCH_MATMUL` against
  a host constant `(L, L)` **upper**-triangular ones mask (the lower triangle is
  the same command stream and the reverse scan), and `cumsum(x, -1) + carry` adds
  the carry with the ordinary binary emitter, so a streaming caller threads the
  total as a method argument and no state is kept on the DSP. The scan length is
  the matmul's `K` and the staged row width, so it must be a multiple of 64; 32 is
  a whole number of tiles and is still refused, because the scan's staging
  descriptor counts rows in 64-wide units and a shorter `K` would be read from
  the wrong row if the skel copied rows straight through. That copy is the
  defect a later phone run did not reproduce; see "Status" for what that leaves
  open. The accumulator is the matmul's fp32 narrowed once, so the comparison is
  against an fp64 prefix
  and not against `torch.cumsum`, which rounds at every step. See
  `test/test_pool.py`, `test/test_sum_amax.py`, `test/test_fmod.py`,
  `test/test_add_relu.py` and `test/test_cumsum.py`.
- the overloads the ops above were missing reach the DSP too. `torch.mean(x)`
  lowers to one REDUCTION over `[1][numel][1]`, the same command
  `torch.mean(x, dim=None)` produces; `torch.max(x)` lowers to one REDUCTION of
  kind 2, the command `torch.amax(x)` already produced; and `relu`, `hardtanh`,
  `relu6` and `x ** 2` each lower to one UNARY command, with the fp16 bit
  patterns of their bounds pinned as the params. The host interpreter reproduces
  torch on all of them, including NaN, both infinities and a signed zero, and
  `test/test_overload_census.py` pins every other overload of every family in
  `OP_SUPPORT.md` as wired, refused or unwired. `torch.mean(x, dim=())` and
  `torch.sum(x, dim=())` and `torch.amax(x, dim=())` all read an empty dim set
  as every dim, which is what torch computes for them, and all three now
  delegate. `torch.max(x, dim).values` reaches that same REDUCTION of kind 2
  through the getitem that hands the values on, and it was the first gap found by
  the partitioner's own census rather than by hand: before it,
  `torch.max(x, dim=1).values` was a graph with no delegate at all.
  `torch.mean(x, dim=1, dtype=torch.float32)` and `torch.mean(x,
  dtype=torch.float32)` are now refused the way `sum` already was: the kernel
  stores fp16, so a mean that requests a width is a different op, and a fp32
  tensor rounded to fp16 was the same defect the sum gate was added for. See
  `test/test_overload_reductions.py`, `test/test_overload_clamp.py`,
  `test/test_overload_census.py`, `test/test_overload_census2.py` and
  `test/test_unwired_targets.py`.
- convolutions reach the DSP, and all three of the usual CNN shapes are among
  the cases. A depthwise `groups == C_in == C_out` convolution lowers to a blit,
  one `CONV_DEPTHWISE2D_FP16` (2) command and a blit back, and every other
  supported convolution to the same three commands with
  `IM2COL_CONVOLUTION_FP16` (12) between them, plus a `ZERO` (24) ahead of the
  pack when the channel count is not a multiple of 64. Both kernels were run
  under hexagon-sim against torch bit for bit: six depthwise geometries (origin,
  stride, dilation, both activations, a second channel block over a batch) in
  `test/sim/conv_runner.cpp`, and ten im2col geometries covering 3x3 with stride
  and padding, 1x1, no padding, a dilated window, a batch of two, an input
  narrower than a 32-channel group and an output that is not a whole number of
  32-position tiles, including MobileNetV2's `ic=3, oc=32, k=3, s=2, p=1` stem.
  Two whole blobs also run the full three-way comparison -- emitter, host
  interpreter, simulator, torch -- in `test/test_blob_on_sim.py`. See
  "Op contracts, and the traps in them" for the kernel defect the same simulator
  found in the im2col store.
- the commands this branch added have run on hexagon-sim and answer torch
  bit-for-bit: the row gather on an `ic=33`/`oc=35` shape whose 32x32 tiles are
  neither square nor whole, with two indices past the vocabulary that the kernel
  clears and torch refuses; a reduction whose span arrives as a run-time patch,
  run at a length shorter than its export with the arena past that length filled
  with a value no correct answer can contain; max pooling; the clamp family on a
  NaN, on both sides of the kernel's vector boundary and over five lengths; the
  two other kernels whose scalar tail answered a NaN differently from its vector
  loop, `add_relu` and a reduction's maximum, each with a NaN in both halves; the
  values of `torch.max(x, dim)`, which are the same reduction command reached
  through a getitem; the
  vision attention on `[1,4,2,64]` and `[2,3,4,64]`; and both quantized GEMV
  entries, whose expectation mentions neither a tile, a group, a nibble nor a
  scale position. Average
  pooling comes back within 1.95e-3, which is its fp16 `1/count` divisor, and the
  same file carries three controls that assert the opposite -- a row-major tile
  table, a blanked dynamic trailer and an exchanged pool layout each have to move
  the answer, and they do. See `test/test_blob_on_sim.py`.
- **one command cannot be covered by the simulator at all, and its refusals are
  therefore host-only evidence.** `DSP_OP_FLASH_ATTN` (18), the entry point behind
  `llama.custom_sdpa`, submits work to a worker pool, and the simulated QuRT
  refuses to start one -- `qurt_cb_fwk_worker_init` returns -4 and the runtime
  aborts, taking the whole run with it, so the fixture cannot even be built. The
  geometries whose mask the kernel would drop (a query extent of one, a run-time
  query length or stride, more than 64 rows) and the rest of the defensive
  rejections that protect that emitter (a causal bias, the non-square case, the
  all-`-inf` row) are checked by the host tests and by reading, never by a
  simulator run, and no simulator result in this README should be read as
  covering them. The mask that is handed over is checked on the host and nowhere
  else: `test/test_sdpa_mask.py` asserts the operand, the stride and the
  reservation out of the blob, and then runs the host interpreter -- which walks
  the kernel's addressing -- against a reference for a mask that has to move the
  answer and for one that has to reproduce the unmasked one.
  `test/test_sdpa_single_token.py` pins the geometry rather than the verdict,
  which one assertion in that file did not: it handed the predicate a four-row
  mask for a one-row query, so the row-count clause is what refused the node and
  the assertion held however the query extent moved. It now hands over a mask
  whose rows are the query's -- the node the shortcut's own clause refuses --
  and the new file reads the refused side back from a lowering that admits it:
  the clause is rewritten in the predicate's own source and re-executed, every
  other refusal left as the backend's, and the command that comes out is
  asserted parameter by parameter, with one query row without a mask and a
  masked query of two rows as the controls. Closing the mask channel instead of
  the hole fails that file rather than passing it, and removing the clause turns
  exactly four tests red. Op 18 has no simulator tier, so this is host evidence
  plus the device measurement in the commit that added it: on a OnePlus 13 the
  one-row masked call answers byte for byte what the same command without a mask
  answers (V's row 0, broadcast), and a masked query of two rows answers bit for
  bit what the numpy reference does and 2.72 away from the answer that ignores
  the mask. Every other attention path here -- the vision
  kernel and the decomposed `scaled_dot_product_attention` -- is a plain loop and
  does run.

**Whether the simulator computes what silicon computes has now been measured, and
on the paths below it does, bit for bit.** This file had a fourth evidence tier
waiting on that question and it now has an answer: until this measurement, a case
that passed only on `hexagon-sim` was evidence about a model of the DSP, and on
these paths it is evidence about the DSP. Nine cases over seven families, one blob
each, extracted from the `.pte` the phone ran and handed to `hexagon_sim.run`
unchanged, so the two runs are the same bytes and not two exports that happen to
look alike, agree exactly on the reduction over a run-time-patched span, `amax`
over a 100-wide fold with a NaN inside it and without, the flat binary add, the
row gather, the max pool, the depthwise convolution and the 3x3 im2col
convolution. Note what that list is: nine separate one-delegate models, not the
graph in the Status list above. The depthwise convolution and the max pool here
are single-op blobs that *did* delegate, while the same two ops inside that graph
did not, and why is not established yet -- this file records the two observations
and not a cause. Three of the nine cases say more than "they agree":

- the reduction agrees with the phone *and the two differ from torch by the same
  5.86e-3*, which is what makes that difference the kernel's own accumulation
  order rather than anything FastRPC, the skel or the phone contributed;
- the rounding-sensitive convolution -- fp16 weights drawn from a normal
  distribution, where the integer weights the other convolution case uses cannot
  fail -- differs from torch at 1152 of its 4096 outputs by at most one ULP and
  agrees with the phone at all 4096, so the two HMX implementations agree about
  rounding and not only about the layout they walk;
- the activation staging tile is the other way round: the simulator
  reproduction and its padding fix are established, but the device discrepancy
  is open. The tile is allocated and read as `up_div(K, 64) * 64` elements per
  row while the copy that filled it used `K`, so a geometry whose `K` is not a
  multiple of 64 read its rows short of where they were written -- twelve rows of
  every 32-row tile came back zero from `K = 40` on the simulator. An earlier
  phone run reported the same signature, at the same positions and with the same
  64/40 ratio, in a Stable Diffusion cross-attention whose second product has
  SD's context length 77 as its `K`. A later run on that phone with a freshly
  built isolated skel did not reproduce the defect on either the old or the new
  skel: at S40, S32 and S77 the old-skel and new-skel outputs were bit-for-bit
  equal, and the reported 12.05 disagreement and 7392 exact zeros were absent.
  The copy now pads rows to the stride its reader uses, `K = 40 / 32 / 77` come
  back inside the fp16 accumulation tolerance with no row left empty, and the
  `K = 64 / 128` controls answer bit for bit what they answered before. The
  earlier phone skel still carried the old code, and that binary had never been
  diffed against the in-tree `mnn-htp-ops`, so it names no source revision. Why
  the earlier and later device observations differ is not established: the
  simulator reproduction remains valid, and the non-reproduction neither
  invalidates it nor establishes that the device path is right.
- the NaN `amax` disagrees with torch at exactly one element, in the NaN's
  payload: `0x7e00` from the phone and from the simulator, `0xffff` from torch.
  That is two NaNs, not two numbers.

**Keep the two claims apart when quoting them.** "The simulator answered the
phone's bits" is a statement about two implementations of a kernel agreeing.
"The phone answered torch's number" is a statement about the kernel itself. Where
both hold they hold for different reasons and neither implies the other.

What this does not do is upgrade the paths it did not measure -- the quantized
matrix forms, the attention kernels, `softmax`, the normalizations, `fmod` -- or
survive a change of arch or dtype, since every case here is v79 and fp16.
`test_blob_on_sim.py`'s own `amax` NaN cases are the standing caution: they reduce
the other axis, so `inside = 100`, and never reach the function the device takes
for the shape a user's `torch.amax(x, dim=1)` produces.

Not done yet:

- a second device and a second arch. An end-to-end run has now happened, once,
  and it covers the ops and shapes the device list above names. Everything else a
  device would settle is still open: the worker-pool kernels and `flash_attn`
  among them, the quantized paths, average pooling and every pool geometry but
  the single max-pool shape that has run, `fmod`, profiling and timing, every
  arch other than v79, and every dtype other than fp16. A kernel is otherwise
  checked by building a real blob, running it on hexagon-sim and comparing the
  result three ways against torch and a host model of the same command stream; the
  simulator is a functional model, so it says what the kernels compute and nothing
  about what the DSP costs;
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
  and unary families, `custom_sdpa`, `rms_norm`, `mul_silu`, `update_cache`,
  `mean`/`sum`/`amax`, `max_pool2d`/`avg_pool2d`,
  `embedding`/`index_select`/`index.Tensor`, `vision_attention`, the narrowing
  and transpose blits, and three whose command is not one of those families:
  `topk`'s values half (`TOPKV2_K1_FP16`), `where` (`SELECT`, whose condition is
  the one operand here that is not two bytes wide) and a zero-filling
  `constant_pad_nd` (a `ZERO` memset and one blit region). Each of the three is
  lowered from a real graph and read back out of the blob in its own test
  (`test_topk.py`, `test_bool_operands.py`, `test_pad.py`). The view, cast,
  getitem and dequantize emitters have run as well but emit nothing by design;
- **the row gather has now run on hexagon-sim** and agrees with torch
  bit-for-bit on the shape that makes its tiling visible: `ic=33`, `oc=35`, so no
  32x32 tile is whole. The tile order is checked by a control rather than by
  agreement alone -- the same blob with its weight bytes reordered row-major
  answers differently, so the case does depend on the order
  `pack_shared_gather_table` writes. The out-of-range index clearing is
  exercised too: two of the six indices are past the table. Still unverified on
  device: the `bytes = 4` (fp32 output) path, which no emitter here reaches; the
  int4 and int8 table paths; and what a 311 MB tiled table costs to read against
  a row-major one;
- attention delegates on fp32 operands that the runtime narrows to fp16 on the
  way into the arena, so the DSP runs fp16 attention. That trade is deliberate
  but unmeasured, and it means a working delegation is not yet a correct one;
- the fp32 normalisation segment is gone from the delegated graphs: the
  `mean`/`rsqrt`/`sigmoid` chain is fused into an `rms_norm` op and the casts
  around it are absorbed, since the arena holds fp16 and the vendored kernels
  take fp16 in and out;
- the int8 half of the quantized matmul path has both entries. The M>1
  prefill kernel (`MATMUL_W8A16_BLOCK_FP16`, 42) now uses the host HMX tile
  packer with its fp16 per-channel scale tail, and the pack64 activation and
  output repack blits are included. The int4 weight has both of its entries too;
  `## Status` records the evidence for each prefill half.
- **the vision attention has now run on hexagon-sim**, bit-for-bit against torch
  on `[1,4,2,64]` and `[2,3,4,64]` at `headDim` 64, with the scale passed as the
  fp32 bit pattern in the param slot it is read from, the mask absent and the
  workspace operand the second output the kernel requires. The layout is checked
  by separation rather than by agreement: the head-major reading (`[batch,
  heads, tokens, headDim]`) answers 1.2 away on these operands, so a case that
  could not tell the two apart would fail its own control. The params are
  positional and nothing in the ABI names them, so three more cases move one
  slot each and require the answer to follow: exchanging `tokens` with `heads`
  moves it 0.83, a scale of zero (uniform attention over the keys) moves it 0.41,
  and shrinking the workspace operand to one byte makes the kernel refuse -- the
  host model refuses it in its own words, and on `hexagon-sim` the kernel writes
  nothing, so that operand is the one the check is about and not merely one the
  answer ignores.
  Note which attention this is: the flash kernel behind `llama.custom_sdpa`
  starts a worker pool, which the simulated QuRT cannot (`qurt_cb_fwk_worker_init`
  returns -4), and that entry point stays out of the suite; the vision kernel is
  a plain loop over the head, so it runs. Still unverified on
  device: the masked path, which an emitter now reaches and which has run on the
  host alone -- the command, its operands and the workspace in one test and the
  kernel's arithmetic against a reference in another, with no tier above that;
  whether the workspace
  size the emitter reserves holds for every `tokens` the kernel's own 128-byte
  alignment asks for, since the simulator does not enforce the allocation; and
  the timing of a real tower;
- **no multimodal model runs end to end.** The vision attention is one block of a
  vision tower, and a tower still needs the patch embedding, the layer norms, the
  MLP and the projection into the text embedding space; on this backend the
  `DecomposePatchEmbed` pass handles the conv-to-matmul step only. Nothing in
  this checkout splices a tower's output into a language model's inputs;
- softmax is delegated on the last axis, and on a contiguous non-last axis when
  that axis can be moved last and back with three-level raster regions and its channel
  is below 64. The direct middle-axis stream is a blit, the existing last-axis
  `DSP_OP_SOFTMAX` with `inside == 1`, and the inverse blit. Non-contiguous,
  non-permuteable, and channel-at-or-above-64 middle-axis cases stay portable;
- **The softmax command is wrong for rows longer than one HVX vector, and the
  emitter now gates on the width.** A `(3, 197, 197)` softmax over uniform logits
  came back wrong on 116284 of 116427 elements, worst 5.9e-3 against a row maximum
  of 1.3e-2, mean relative error 9.6%, row sums still one. At 63 columns the same
  kernel is within an fp16 ulp and at 64 it is not, and within a wrong row the
  tail the kernel copies in and masks -- the last `channel % 64` elements -- is
  clean: its share of the error is the inflated normaliser the vector loop's
  exponentials give the whole row. The vector loop's exponential is up to 1.76x
  the correctly rounded value, as a function of the argument's fractional part
  only and identically for every octave, and the `UNARY` exp command over the same
  arguments is within an ulp of torch, so the defect is the softmax kernel's own
  path and not the shared polynomial. Rows shorter than a vector keep the command;
  longer ones are the shifted sum of exponentials, five commands and no new
  kernel. Measured on a OnePlus 13 (SM8750, Android 15, CDSP, v79) with the
  vendored skel the `executor_runner` beside it loads;
- **Convolutions have now run on the device, and the gate the emitter puts in
  front of them is the gate the device has.** A 3x3 conv and a 1x1 conv went
  through one three-delegate run on a OnePlus 13 (SM8750, Android 15, CDSP, v79)
  -- the depthwise convolution in that graph did not, and the Status list above
  says why that matters -- and a single 3x3 conv through another, each round
  returning `exit dN: ok` and each matching the fp16 reference computed from the
  same module instance that produced the `.pte` to one fp16 ULP (worst case
  4.88e-4 against a reference whose largest element is 0.57). That 4.88e-4 is the
  whole graph's output, and the depthwise walk's share of it came out of the
  portable kernels, so the number is not evidence about that kernel either. The
  kernels still agree
  with torch bit for bit on every case above; what changed is that the FastRPC
  transport, the skel deployment and the VTCM budget under a real arena are no
  longer part of the question. The VTCM gate is no longer paper arithmetic:
  `conv_vtcm_bytes` adds up the four allocations the kernel makes
  (`im2col_convolution_fp16.cc:1783`-`:1786`) and compares the total with the
  8192 KiB the simulator's manager reports, which refuses a reduction wider than
  `kp = 1364` -- 4832 input channels over a 3x3 window. Run on the phone with
  `oc = 64`, so that `np` reaches 2 and `np_chunk` really is 2: `ic` of 64, 512,
  2048 and 4832 answer within half an ULP of their largest reference element
  (9.77e-4 at `ic = 4832`, against 3.08), and `ic` of 4864, 5120 and 8192 --
  which the emitter refuses, so the AOT gate was raised to ship them -- fail in
  0.09-0.18 s with `execute_command_group failed: 0x8000040d`, write no output,
  and leave the DSP healthy enough that the `ic = 64` case passes immediately
  afterwards. So a device hands out the 8,354,560 bytes the widest delegated
  window asks for and not the 8,409,856 the next one asks for, which is the
  boundary the arithmetic draws. Note the code: `0x8000040d` is the one the
  RUNBOOK attributes to a skel built without `-O2`, and this is a second,
  unrelated way to earn it. `DSP_OP_ZERO` (`hexagon_ops.py:2618`-`:2625`, in
  front of any convolution whose `in_channels % 64` is non-zero) is now in a
  stream a device has executed -- the `ic = 4832` blob opens with one,
  `params=[155648]` bytes -- but that is not a check of the zeroing itself: the
  lanes it clears are multiplied by the zero weights the tiles carry for them,
  so an input with no NaN in those lanes cannot tell a cleared lane from a stale
  one. The im2col kernel's
  other entry points (`CONV1X1_DIRECT_FP16`, the weight-only quantized
  convolutions), its scale-block parameters (`scaleBlockNum`, `scaleAsymmetric`)
  and the `outputBytes` bound check, which this emitter turns off by passing 0,
  are unread beyond the fields the fp16 path uses. The depthwise walk's `relu`
  and `relu6` params are pinned off, because `to_edge` leaves a relu as its own
  node, so the kernel's fused activations are untested here.
- **pooling has now run on hexagon-sim** -- max pooling bit-for-bit against torch
  and average pooling within 1.95e-3, the fp16 `1/count` divisor, on both
  `countType` forms -- and `sum` has run through the run-time patch above. A
  control asserts that the packed layout matters: exchanging the two inner axes
  of the same buffer answers differently on these operands. `amax` has run there
  too, in two cases -- one whose span is a compile-time constant and one whose
  span arrives as a run-time patch -- each carrying a NaN in both halves of the
  fold. `fmod` has not: its layout and arithmetic are a second implementation of
  the same source, so the test agrees with the reading and not with the hardware.
  Since then a max pool has run on a device too: the single-op 64-channel case in
  the simulator comparison below is one delegate whose 1024 outputs are
  bit-identical to `hexagon-sim`'s and equal to torch's. The list that follows is
  about which path the dispatcher takes and what the kernel does at its edges,
  none of which that one number separates, and about average pooling and every
  geometry other than that one, which have not run.
  Unverified on device: that the two pool blits really take the DSP's pack-area
  fast path (`try_pack_area_blit` takes the geometry the
  emitter checks for, but the fallback's numbers were never compared against the
  fast path on hardware, and this suite models the fallback deliberately); that
  `hvx_pool2d_fp16`'s window origin and its out-of-range handling beyond the two
  cases above, and its `countType` divisor, are what the tests model, including
  its rounding of the fp16 reciprocal and its fp16 accumulation; `amax`'s signed
  zero and NaN tie-break, which is now measured on the device for the one shape
  this checkout emits most -- see the reduction paragraph below -- while the
  signed zero and the tie-break between two NaNs are still unread; that
  `HTP_OPS_BINARY_MOD`'s int32 truncation and its
  zero-divisor guard agree with the host model, and which of the two paths
  (`htp_ops_binary_elementwise`'s scalar `apply_fp16` or the fp16 vector tail) the
  dispatcher takes for subtype 12; and that the reduction's accumulator really is
  fp32 with an fp16 store, which the `bytes` param asserts and no test can
  observe;
- **the reduction's NaN contract is now measured on the device, and it is a
  function of how wide the fold is.** Take the shape this backend emits most, a
  reduction over the innermost axis, so the wire command is
  `params=[outside, reduce, 1, kind, ...]` with `inside = 1` and the kernel takes
  `htp_ops_reduce_max_fp16_inside1_hvx`: a NaN in the fold comes back as the NaN
  when `reduce` is a multiple of 64, and as the largest of the *other* elements
  when it is not. A `(2, 32)` fold answered `0x3c00`, which is `1.0`, for a
  sign-clear NaN at its first element and for one at its last, where torch
  answers `0x7e00` at the same position; `(2, 64)`, `(2, 128)`, `(2, 192)` and
  `(2, 256)` answered the NaN; `(2, 96)` and `(2, 100)` answered `1.0` again.
  Where the NaN sits does not matter -- a NaN at column 3 and a NaN at column 95
  of the same 96-wide fold both came back as `1.0`, so it is the presence of a
  scalar tail and not the tail's contents that decides. The sum over the same
  `(2, 100)` shape answers `0x7c00`, `+inf`, where torch answers the NaN. Read
  the simulator's half of this with care, because it is about a different
  function: `test_blob_on_sim.py`'s NaN reductions (`_AmaxAt`, cases `BJ` and
  `BL`) reduce the *other* axis, so `inside = 100 >= 64` and the kernel takes
  `htp_ops_reduce_fp16_inside_vector_range`; those pass on `hexagon-sim` and say
  nothing about `inside = 1`, which is what a `torch.amax(x, dim=<last>)` on a
  multi-dimensional tensor produces. Neither half covers the other. The signed
  zero, and the tie-break between two NaNs of different payloads, are still
  unread on a device;
- **the fused rectified sum (`add_relu`, subtype 8) reaches the DSP through a pass
  rather than through an ATen op**, because no ATen op produces `max(a + b, 0)`:
  `relu(x + y)` reaches the graph as an add and a relu, and `FuseAddReluPass` in
  the caller's `transform_passes` is what rewrites them into the node the emitter
  table has the subtype for. Without the pass the pair still reaches the DSP, as
  a binary add and a unary clamp, so the pass buys a command rather than a round
  trip -- a smaller claim than it made before `aten.relu.default` had an emitter.
  `test/test_add_relu.py` pins the rewrite, the command and its numbers, and
  `test_blob_on_sim.py` now runs the command itself through the kernel on a NaN
  that lands on both sides of its vector boundary -- a NaN whose sign bit the
  fp16 add sets, which is the value the scalar half used to answer with `0.0` and
  no longer does (`VENDORING.md`, modification 4). A NaN that arrives with its
  sign bit already set is still a wrong number at a vector lane: driven that way
  on the simulator, the lane answers `0.0` where torch answers the NaN, because
  `Q6_Vhf_vmax_VhfVhf` sends a sign-bit NaN to the other operand, and that
  instruction is behind the rectifier as well as behind the reduction. **That
  sign-bit lane is now answered, and it is the same on silicon as on the
  simulator.** Driven on the phone with a 128-element input -- every lane inside
  the vector loop, no tail -- the fused `relu(x + y)` answered `0x0000`, which is
  `+0.0`, at the three lanes whose sum was a sign-set NaN, where torch answered
  the NaN; the two lanes whose sum was a sign-clear NaN came back `0x7fff`, a NaN
  with a different payload, so the instruction keeps a NaN it can see and sends
  a sign-bit one to the other operand on hardware exactly as `hexagon-sim` does.
  What is left unverified on a device is that an add whose sum has another reader
  still gets the command, which is a partition question rather than a kernel one.
- **the clamp entry point has run on hexagon-sim, and the NaN path it was feared
  for is not one path but two.** The kernel walks `size & -64` elements a vector
  at a time and the rest one at a time, and until this branch the two halves
  answered a NaN differently. The vector loop restored one by the
  `(|x| & 0x7fff) > 0x7c00` bit test; the tail (`unary_ops.cc:536-539`,
  `x < lo ? lo : x > hi ? hi : x`) let the unordered compare send it to the upper
  bound, so `relu(NaN)` came back as `+inf` -- `6.0` for `relu6`, `1.0` for
  `hardtanh(-1, 1)` -- where torch and the host model return the NaN. No
  exception, no shape change, one wrong number, and it was any length that is not
  a multiple of 64 rather than only a short tensor: 32, 65, 100, 127 and 129 all
  lose the NaNs in their tails on the simulator, and 64 and 128 do not. The tail
  now carries the same bit test (`VENDORING.md`, modification 4), and
  `test_blob_on_sim.py` asserts it on 32, 64, 100, 127 and 129 elements, all three
  bounds, bit for bit against torch, rather than reporting it as an `xfail`
  (`test_the_clamp_tail_answers_a_nan_with_the_nan`). The two bounds are read as
  fp16 bit patterns out of params[3] and params[4] in that order. **The vector
  half has now run on the device, and silicon keeps the NaN.** Driven on the
  phone at 128 elements -- two whole vectors, no tail -- and at 100 -- 64 vector
  lanes and 36 tail lanes -- on an input carrying `0x7e00` and `0xfe00` in both
  halves and an `0x7c01` and an `0xfc01`, all three bounds returned the NaN at
  every lane torch returned one, sign bit and all: `relu`, `relu6` and
  `hardtanh(-1, 1)` each answered `0xfe00` for `0xfe00` and `0x7e00` for
  `0x7e00`, which is the case the simulator gets wrong and the one this paragraph
  was waiting on. The signaling forms are where the two part company on payload,
  and in both directions: `0x7c01` came back as `0x7c01` from the device and as
  `0x7e01` from torch at index 2 of the 128-element input, and as `0xfe01` from
  the device and as `0xfc01` from torch at index 99 of the 100-element one. Every
  one of those is a NaN on both sides, so it is a payload that differs and not a
  number. So `relu` is now trustworthy on
  data that can be NaN, subject to the silhouette of every device result here:
  one phone, one commit, fp16.
- **`torch.mean(x)` / `torch.max(x)` / `x ** 2` have never run anywhere but on
  the host either**, in the same sense as the other reductions: the span
  `[1][numel][1]` and the unary square are transcriptions. The mean's span is
  the same one the `dim=None` form already produced, so it adds no new kernel
  question; `max` adds the signed-zero and NaN-payload tie-break `amax` does not
  exercise; and `x ** 2` adds only that `(float)x * (float)x` rounds the way
  fp16 multiplication does, which the host model agrees with.

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
