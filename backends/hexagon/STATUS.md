# Hexagon backend status

Worktree: `/home/yydh/wt/executorch` (branch `hexagon-mm`).
The directory must be named exactly `executorch` or CMake refuses to configure.
`git worktree` does not bring submodules: 23 of them are symlinked from
`/home/yydh/executorch`, and `src/executorch/exir/_serialize/program.fbs` is a
build artifact that also has to be copied over or export fails at serialization.

## Verified

Delegation coverage on Qwen3-0.6B, measured with `/tmp/qwen3_mm_export.py`
(swaps the XNNPack partitioner for `HexagonPartitioner` inside the stock export
pipeline):

| op | before | after |
| --- | --- | --- |
| `aten.mm.default` | 0 | 197 |
| `aten.mul.Tensor` | 29 | 253 |
| `aten.add.Tensor` | 112 | 112 |
| `aten.neg.default` | 56 | 56 |
| `aten.view_copy.default` | 0 | 392 |
| `aten.unsqueeze_copy.default` | 0 | 56 |
| `aten.slice_copy.Tensor` | 2 | 112 |
| `aten.cat.default` | 0 | 112 |
| `aten.permute_copy.default` | 0 | 197 |
| `llama.custom_sdpa.default` | 0 | 28 |
| `et_hexagon.rms_norm.default` | 0 | 113 |
| `et_hexagon.update_cache.default` | 0 | 56 |
| **Total** | **394** | **1796** |

1796 delegated nodes in **86** subgraphs: 646 in 365 at the start, 982 in 225
after the views, 1207 in 142 after the norms, 1375 in 114 after the slices, 1487
in 86 after the concatenations, 1684 in 86 after the transposes and 1796 in 86
after the cache advances. The subgraph count is what costs FastRPC round trips,
and it has come down by a factor of four; 1191 nodes stay on the portable
kernels.

The two `slice_copy` left behind are the RoPE frequency tables, cut to the
sequence length: a blit region copies one constant run per row and those bounds
are not known until the call.

Numerical verification so far is **kernel transcription, not on-device**: each
DSP kernel was re-implemented from its source and compared against torch element
by element. Nothing has run on a DSP yet.

- **`aten.mm` -> `DSP_OP_BATCH_MATMUL` (38)**, 7 shapes up to K=777: float64
  comparison is exactly zero, so the addressing is provably right; fp16 with a
  float32 accumulator agrees to <= 1 ULP. `skipped=0` everywhere, meaning no
  element hit the DSP's bounds check and got silently dropped.
- **broadcast binary -> `DSP_OP_BINARY_ELEMENTWISE` (19)**, 5 shapes including
  1-D right-alignment and 4-D mixed broadcast: exact.
- **`aten.mean.dim` -> `DSP_OP_REDUCTION` (29)**, 4 shapes including a
  middle-dim reduction: exact.

Layout cross-checks: `HtpOpsLoopParam` is 100 bytes and byte-identical between
C and Python; `sizeof(HexagonOp)` == `OP_SIZE` == 480, asserted on both sides
(`static_assert` in `hexagon_schema.h`, compared against `blob.OP_SIZE` by
`test/test_blob_roundtrip.py`).

C++ builds with `EXECUTORCH_BUILD_HEXAGON=ON` from the worktree; confirm with
`CMAKE_HOME_DIRECTORY` in the build dir's `CMakeCache.txt`, not the exit code.

## Emitters

`EMITTERS` in `hexagon_ops.py`; `SUPPORTED_TARGETS = EMITTERS`, so adding an
emitter and a mirrored check in `partition/hexagon_partitioner.py` is all a new
op needs. **The mirror rule matters**: a node rejected in `is_node_supported`
falls back to a portable kernel, but one rejected inside an emitter fails the
whole export.

Of the 24 registered emitters, 5 have produced a DSP command on a real graph --
`mm`, `mul`, `add`, `neg`, `custom_sdpa` -- and 4 more have run without
emitting one (the views below). The rest are unexercised.
`layer_norm` is registered through the `native_layer_norm` that `to_edge`
rewrites it to, plus the `getitem 0` that carries its first output; its kernel
reads gamma and beta as fp32, which no delegate input can hold, so the norm runs
without them and the affine is a mul and an add (see `_emit_layer_norm`).

## Views

`alias_copy`, `unsqueeze_copy`, `view_copy` and `select_copy` do not move
bytes, so their emitter returns the operand's `TensorRef` and emits nothing. Two
nodes sharing one ref is what a view is: the kernels index the same buffer under
another shape, and the grid they walk does not change.

Three conditions hold, and each is a check rather than an assumption:

- **The result is the operand's bytes in order.**
  `_alias_keeps_the_same_bytes` requires both sides contiguous with the same
  element count. `slice_copy` looks like a view (`[1,s,16,128] -> [1,s,16,64]`)
  and fails this, which is why it is not registered: the half it takes skips
  bytes the result's own layout does not describe.
- **The node is not a partition boundary.** A view emits no command, so a
  partition that has to hand one out has nothing to write into its output slot.
  `_pack_section` raises `method output N has no slot` when that happens, and
  that is how this was found -- loudly, in the export, not as wrong numbers
  later. `partition()` drops a view whose consumers are not all delegated,
  walking in reverse topological order so that a chain of views is handled too.
  252 of the 588 views on Qwen3 are boundaries and keep their result on the CPU.
- **`select_copy` is the `int` overload, and 85 of its 86 occurrences are not
  views at all** -- they read the `int64` position tensor down to a scalar for
  `start_pos`. The dtype check already rejects them, which is the right
  outcome: the patch mechanism needs that value where it is.

`alias_copy` still shows 0. All 56 of its results feed `slice_copy`, which is
unsupported, so every one of them is a boundary. That number should move when
the KV cache path lands.

`MAX_OP_PARAMS` is 40 (raised twice: 16 -> 32 for BATCH_MATMUL's 26, then 32 ->
40 for the broadcast tail's 33). It is our own constant, not a DSP limit -- the
DSP reads a variable-length FlatBuffers vector and guards with
`params->size() > N`.

Units, settled by `htp_ops_loop_check_element` dividing the stride terms by the
element size: **strides are in bytes; sizes, cmdSteps and view offsets are in
elements**.

## Why `mean`, `rsqrt` and `sigmoid` show 0 in Qwen3

The emitters are correct, but the graph feeds them **fp32**. `RMSNorm` does
`x.float()` or equivalent around `upcast -> pow -> mean -> +eps -> rsqrt ->
downcast`, so the whole chain lands outside the delegate.

The counts identify the source exactly: `mean` 113, `rsqrt` 113 and the
epsilon `add` 113 per forward pass, and **113 = 28 layers x 4 norms + the final
norm** (attention_norm, ffn_norm, q_norm_fn, k_norm_fn). `mul` is 254 = 113 x 2
plus 28 for the gated variant, and `sigmoid` 28 is that variant. All of it
traces to `examples/models/llama/norm.py`, so it is model source, not a
`dtype_override` mistake.

Delegating it is not an emitter problem, and not a small one:

    htp_ops_flash_attn   no bytes parameter at all; tokens advance by
                         n_kv_heads * head_dim * sizeof(__fp16).
    htp_ops_unary        unary_ops.cc:576 rejects anything but 2 and 4, and :582
                         rejects 4 unless the op is ABS or NEG. bytes=4 means
                         INT32, not float.
    htp_ops_reduction    bytes=4 goes through htp_ops_reduce_int32, which adds and
                         divides as integers.
    htp_ops_cast         has no FP32 conversions at all.

The library's idiom is fp16 tensor I/O with fp32 accumulators, and HMX is
fp16-only. The way to take this segment is therefore one fused fp16-in/fp16-out
RMSNorm kernel that accumulates in fp32, plus an export-time transform replacing
`RMSNorm.forward` -- not emitters for `mean`/`rsqrt` -- because the cost here
is graph fragmentation (113 splits), not FLOPs (roughly 0.06% per layer).

## Solved: `llama.custom_sdpa`

**The blocker was not dtype.** `hexagon_ops.py` resolved
`exir_ops.edge.llama.custom_sdpa` at *import* time, but the LLM extension
registers its ops later, so `_sdpa_op` was `None`, `SDPA_TARGETS` was empty and
the `EMITTERS[_target] = _emit_sdpa` loop never ran once. The emitter was never
registered and the node never reached the dtype gate. Resolution is now lazy
(`sdpa_targets()`, called from `is_node_supported`).

Nine faults kept sdpa from delegating, every one of them silent -- a wrong
registration produces exactly the same observable as an unsupported op:

  1. `SDPA_TARGETS` held `sdpa_with_kv_cache`; the graph uses `custom_sdpa`.
  2. Registered from `torch.ops`; the graph carries `exir_ops.edge` overloads.
     Same schema, different objects -- membership never matches.
  3. The emitter was written for the cached layout; `custom_sdpa` has no caches.
  4. `start_pos` was required to be a tensor read; it can be a constant, which
     must be baked into params instead of patched.
  5. `len(node.args) < 6`, copied from the cached layout; `custom_sdpa` needs 4.
  6. `is_node_supported`'s fallback rejected the node because `None` is not in
     its `(int, float, bool, list, tuple)` exclusion, so `attn_mask=None` failed
     the `isinstance(arg, torch.fx.Node)` test and `all()` returned False.
  7. `preprocess` called `.numel()` on every placeholder, but `start_pos`
     arrives as a SymInt. Scalars now get one int64 slot, which is also what the
     patch mechanism reads.
  8. `SDPA_TARGETS` was resolved at import time and was therefore empty.
  9. `_scalar_source` matched `torch.ops.aten._local_scalar_dense.default` by
     object identity, which never fires on edge overloads, so it returned the
     extractor node itself instead of the tensor it read from.

**Layout: they match.** `attention_entry.cc:242` derives the token-to-token
stride as

    const int kv_stride_bytes = n_kv_heads * head_dim * sizeof(__fp16);

so the cache is `[.., token, n_kv_heads, head_dim]` -- token before heads, the
same order ExecuTorch documents (`op_sdpa.cpp:287-289`, BSNH). K and V advance
identically (`attention_entry.cc:258-259`), and `value_c4` is an optional
packing flag the default path ignores. No permute is needed in front of the op.

**Numerics caveat.** The operands are fp32 and the runtime narrows them to fp16
as they enter the arena, so the DSP runs fp16 attention. That is a deliberate
trade, and it means a working delegation is not yet a correct one.

## RMSNorm

`norm.py` computes the norm in fp32 and casts back, so every layer contributed
`mean`, `rsqrt`, `add` and two `mul` nodes with no DSP kernel behind them. That
was 621 nodes, and it was not the FLOPs that cost anything -- it was that each
norm cut the graph in two. `rms_norm.py` rewrites the pattern to one
`et_hexagon.rms_norm` node, which the emitter lowers to `DSP_OP_LAYER_NORM` with
its RMSNorm flag.

The result: `mean.dim` and `rsqrt` leave the graph entirely, `add.Tensor` goes
to 112 delegated and 0 left, `rms_norm` is 113/0, and 83 subgraphs disappear.

Three things had to be true, and the first two were only found by running it:

- **It cannot be done in the partitioner.** `to_backend` asserts that a
  partitioner returns the graph it was handed, so restructuring there fails with
  `should not modify the graph module`. It has to be an `ExportPass` supplied
  through `transform_passes`, which is why `FuseRmsNormPass` exists and why the
  export harness appends it.
- **The weight cannot be folded in.** The kernel reads gamma as fp32, but a
  weight reaches a subgraph as an fp16 placeholder, not a constant, so passing it
  as gamma would read half-precision bytes as floats. The scale keeps its own
  `mul`, which is fp16 once the fuse has run and already has a kernel. gamma
  reaches the kernel as ABSENT, which it handles by skipping the affine step.
  The 28 `mul` and 28 `sigmoid` still outstanding are `RMSNormGated`, a
  different pattern.
- **Both cast spellings have to be walked through.** Edge carries the pattern as
  `aten._to_copy` or as `dim_order_ops._to_dim_order_copy` depending on the path
  that built the graph, and only the second appeared in the synthetic test. The
  dim-order copy can also reshape, so the walk stops when the shape changes.

Fusing is exact here: on a synthetic norm the fused graph matches the unfused one
and eager bit for bit, because the DSP accumulates in fp32 exactly as the
decomposed graph did.

## What the KV cache and permute steps actually are

Probing the graph before writing either one changed the plan, so the finding is
recorded here rather than the plan.

`slice_copy` (114) is not the KV cache. Two of them slice `b_rope_freqs_cos`
and `b_rope_freqs_sin` by the sequence range; the other 112 are RoPE's
`rotate_half`, splitting the last dim of q and k into halves that `cat` rejoins
after a `neg`. The RoPE pair is 112 slices, 112 cats and 56 negs, all in service
of `DSP_OP_ROPE`, which already exists. Reading those names as "the KV cache
path" was a guess from the op names and it was wrong.

The KV cache is `auto_functionalized_v2(llama.update_cache)` -- 56 nodes, two
per layer -- plus the `getitem` that reads it back. This is the one step still
open, and the decision the step asked for is below.

`update_cache` is a mutating custom op whose tuple result is read by exactly one
`getitem(update_cache, 1)`, producing `[1, 2048, 8, 128]` -- the whole K cache
-- which feeds a graph **output** as well as a cast. That output exists *because*
the cache lives outside the delegate. Two things follow.

**Folding the update into attention is not available.** The cache is not written
by the attention kernel: `DSP_OP_FLASH_ATTN` reads k and v, and the 28
`llama.custom_sdpa` nodes that would have to absorb the write are not delegated
either. There is also no kernel to reach for -- none of the 41 `DSP_OP_*` cases
in `execute_command.cc` pushes into a cache -- so "a DSP kernel" would mean
writing a new one, and folding it into attention would mean teaching the
attention kernel about sequence positions as well.

**The cache is a method input that gets mutated in place, and the runtime already
does this.** `HexagonOp.in_place` is a bitmask: bit j means `inputs[j]` is
mutated and has to be copied back to the caller. The runtime collects those bits
and, after the command group, copies the buffer back -- with a comment that names
this exact case:

    // A subgraph that advances a KV cache in place has to hand the updated buffer
    // back, or the next execute() copies the old one in and every step after the
    // first attends over stale keys.

So no state section is needed and no allocation has to outlive a call: the cache
stays the method input it already is, and the delegate declares that it writes to
it. An earlier reading of this file concluded the opposite, from the arena being
per-call; the arena is per-call, but an in-place method input is not.

What is actually missing is smaller and stranger: **nothing ever sets the bit.**
`in_place` is in the schema, serialized by `blob.py`, covered by the blob
round-trip test with two different patterns, and honoured by the runtime, and
`grep in_place` over the emitters, the backend and the partitioner returns
nothing. It is a wired-up path with no producer -- the same shape as
`ctx.constant` earlier in this file.

That left the write itself, which is a blit into the cache at `input_pos`, and
one thing that did not fit: the destination offset is `input_pos * kv_heads *
head_dim`, while the patch copied the input through unscaled. Reading the
runtime rather than the comment made the shape of the problem exact. The patch
is a raw four-byte move, and nothing consults the operand's dtype:

    std::memcpy(&value, base + patch.input_offset, sizeof(value));
    std::memcpy(base + patch.param_offset, &value, sizeof(value));

`params` are static in the blob, so that copy is the only way any param can
become dynamic. There is no second channel, which means the scaling had to go
somewhere, and every way of avoiding it fails: the position can only reach a
region linearly through `dstOffset`; patching a stride does nothing because
`size[0]` is one and the z term is always zero; patching a size writes rows
`0..p-1` instead of row `p`; and `bytes` is both the operand size and the
offset unit, so it cannot carry the factor either.

So `HexagonOp` grew a `patch_scale`, and the param is now the input's first
element times it. That takes the struct from 476 bytes to 480, which is the
kind of change this file's own guard rails exist for: `blob.py` derives
`OP_SIZE` from its struct format so it follows automatically, the round-trip
test compares both sides field by field, and the header pins the same numbers
with `static_assert`. Both mutations were tried -- a trailer four bytes short,
and a scale silently forced to one -- and both failed the test, so the check is
one that can fail. The blob version went to 2 with it, because the runtime
validates it and a v1 blob read as v2 would take an `in_place` bit for a scale.

The default is one, so the attention path that already patched a position is
untouched.

**The emitter was then written, measured, and backed out.** Matching
`getitem[1]` against `auto_functionalized_v2(llama.update_cache, ...)` is
enough to lower the write: copy the cache out, then blit the new rows into the
copy at `dstOffset`, patched from the position tensor with `patch_scale` set
to one cached position's element count. The predicate called the same layout
function the emitter did, so the two could not disagree. It made the graph
worse:

    before  1684 delegated  1359 not  86 subgraphs
    after   1684 delegated  1443 not  114 subgraphs

with `auto_functionalized_v2` still `0 / 56`. The reason is that the node
holding the cache is not itself delegable, so claiming its `getitem` promotes a
CPU node into a subgraph boundary: the partition loses 28 groups' worth of
merging, and the delegate would perform a write the CPU then performs again.

So the emitter was backed out and the fix done where the measurement pointed:
`kv_cache.py` ships `FuseKvCachePass`, which rewrites the pair into a single
`et_hexagon.update_cache` node. The wrapper then has no users, and
`eliminate_dead_code` deletes it along with the scalar read whose only user it
was, so nothing is left stranded to become a boundary:

    before  1684 delegated  1359 not  86 subgraphs
    after   1796 delegated  1191 not  86 subgraphs

The subgraph count does not move at all, which is the point: the cache advance
is delegated, `auto_functionalized_v2` is gone from the graph, and the 56
`getitem` reads of it with it.

The fused node is a pure function of the old cache rather than a mutation. The
graph already surfaces the updated cache as an output, and the lowering copies
that cache out and writes the new rows into the copy, so a mutation would only
add a second path for the same bytes. The reference implementation in
`kv_cache.py` needs a fake kernel for the same reason: it reads the position to
size a slice, and a symbol cannot be read, so the shape-only registration is
what keeps tracing from calling it.

Two blits do the work, and the first attempt at the second one was wrong in a
way only a check could catch. A region is `[srcIndex, srcOffset, dstOffset,
size0..2, srcStride0..2, dstStride0..2]`, and the row count belongs in
`size[1]`, not `size[0]`: writing `size = [rows, 1, run]` with the strides in
positions 1 and 2 leaves `dstStride[0]` at zero, so the outer loop re-copies
the same run `rows` times and the cache is silently wrong everywhere but the
first row. Transcribing `blit_ops.cc` into Python and comparing against the
tensor op at four positions fails on that region and passes on
`size = [1, rows, run]`, which is the form the other emitters already use.

The op is declared `update_cache(Tensor value, Tensor(a!) cache, ...)`, so the
cache is an in-place mutation on the PyTorch side too, and the wrapper is
readable: `auto_functionalized_v2` takes the overload as `args[0]` and the
operands as **kwargs**, with the mutated tensors listed in `kwargs["_all_bases"]`.
That is not a guess -- `backends/mlx/custom_kernel_ops/gated_delta_rule.py`
already handles this exact shape, including matching on
`getitem[1] from auto_functionalized_v2(...)`, which is the form the cache read
appears in here. A handler for this backend can follow it directly.

So the shape of the work is settled: match the `getitem`, take the cache from
`_all_bases`, emit a blit into it with the `in_place` bit set for that input,
and settle the offset scaling one of the two ways above. What it is not is a
missing kernel or a missing runtime feature, which is what an earlier reading of
this file said.

`permute_copy` (197) is not a per-token transpose either. Every one sits
between a weight and the `mm` that consumes it:

    permute_copy(p_layers_0_attention_wq_weight, [1, 0]) -> mm

so it transposes something that never changes, on every token. Folding it looked
like the cheap win and it does not work, for a reason worth writing down:
**`torch.export` already lifts parameters to graph inputs**. The raw export graph
has no `get_attr` node at all, `p_w` is a placeholder, and
`ExportedProgram.constants` is empty. A pass cannot fold the transpose of an
input, because the value is not in the program.

That leaves the three options this step was meant to choose between, now with the
cost of each visible: keep the transpose as a real DSP command (197 per token, on
data that never changes), transpose the weights before export so the graph never
holds the permute, or give the delegate a load-time transposed weight. Only the
second removes the work instead of moving it.

## Strided copies: slices, concatenations and transposes

`slice_copy` is not a view, so it cannot become an alias emitter: the result
holds fewer elements than the operand and the operand's TensorRef still describes
the longer row. It is a strided read, and `DSP_OP_RASTER_BLIT` already does
exactly that -- one region is `srcIndex, srcOffset, dstOffset, size[3],
srcStride[3], dstStride[3]`, with strides in elements and `size[2]` the
innermost run.

`slice_region` computes that region and both the support check and the emitter
call it, so they cannot disagree about which slices the DSP can run. That matters
more than usual here: a slice the emitter refused would fail the whole export
rather than fall back, so the refusal has to happen in the support check.

The 112 that delegate are RoPE's `rotate_half` splits, with constant bounds. The
2 that do not are the frequency tables cut to the sequence length, where the end
is a `_local_scalar_dense` result.

`cat` is the same command read the other way: one region per operand, each
writing its own slice of the result, which is how the two 64-element halves of a
RoPE rejoin become one 128-element row. `cat_region` does for it what
`slice_region` does for slices, including the refusal, and all 112 delegate.

A region is twelve ints and the header is three, while an op's params vector
holds `blob.MAX_OP_PARAMS` (40) of them, so at most three operands fit.
`MAX_CAT_INPUTS` makes that bound explicit rather than letting a wider
concatenation reach `add_op` and be rejected there, which would fail the export
instead of leaving the node on a portable kernel.

`permute_copy` is the same command a third time, and all 197 of them are
`permute(w, [1, 0])` between a weight and its `mm`: one matrix transpose per
outer index. A region describes that directly -- the inner run reads a contiguous
source row and writes a strided destination column -- and it is the exact shape
`htp_ops_prepare_transpose` recognises, so it routes to the HVX transpose rather
than the generic loop. Permutations that move any other axis are refused: the row
index stops being linear in the destination, which one region cannot describe.

## Verifying a region

The region encoding has no room for a wrong answer to be visible: a blit that
reads the wrong offsets produces plausible numbers. Transcribing the kernel's
loop into a scratch script and running one region through it beats reading the
numbers, and that is how every region here was checked until now. It has two
weaknesses. The loop is retyped per check, so it can drift from `blit_ops.cc`
unnoticed, and it never sees what the emitters actually emitted.

`test/blob_interpreter.py` closes both. It decodes a blob the real
`HexagonBackend.preprocess` produced -- header, ops, and the weights, with the
other three sections present as header sizes -- builds
the arena the runtime builds, applies the patch slot, executes the command
stream, and compares against torch. Sizes, offsets and strides come out of the
blob, so the check cannot agree with an emitter it is not reading.

Writing it turned up two ways the first version was wrong, either of which would
have left it passing for the wrong reason:

The innermost axis is not always a contiguous run. `blit_ops.cc` takes that
shortcut only when both inner strides are one; otherwise it calls `blitProc`,
which advances by `srcStride[2]` and `dstStride[2]` per element. The permute
region emits `dstStride[2] = rows`, so a model that copies `size[2]` elements
as one run quietly transposes nothing.

A slice assignment in Python resizes a bytearray when the two sides differ in
length, and they differ whenever the strides do. Running the permute step shrank
the arena from 640 bytes to 556 and the output came back four bytes short.
`_copy` checks bounds and raises instead.

Neither was a bug in the emitters; the emitters were right both times. That is
the point. A host model of a command stream is as capable of being wrong as the
stream is, and comparing against torch is the only way to tell them apart.

The check can fail, and that was shown against the emitters rather than only
against a mutated blob. Setting `permute_region`'s `dstStride[2]` to 1 instead
of `rows` fails the blit tests, and asking `_emit_rms_norm` for a LayerNorm
instead of an RMSNorm fails the norm tests; both revert clean.

Thirteen tests run over the command stream, and five of the eight DSP ops the
emitters can produce are modelled: the blit, the fused norm, the products, and
the element-wise unary and binary paths. The blit is covered through `cat`,
`slice_copy` and `permute_copy` at three shapes. The cache advance is covered at
three positions, which is what exercises the patch slot: the position reaches the
region only as `patch_scale * position`, so that test unscales the patch and
requires the written rows to move. The fused norm is covered at three shapes, and
its last param is the RMS flag -- a LayerNorm there would still print plausible
numbers, and clearing the flag is what has to move them. The products are covered
at three shapes against a fp32 accumulation, which is the delegate's own contract
rather than what torch's fp16 `mm` does. The element-wise path is covered with and
without broadcast, and a broadcast stride is exactly the kind of encoding that
degrades quietly: zeroing the short operand's innermost stride leaves the shape
right and the product wrong.

Three things reading the kernel turned up. `DSP_OP_BATCH_MATMUL` is 38, not 39 --
a miscounted enum stays invisible until a blob refuses the op, which is how it
was caught. The unary `rsqrt`, `log`, `cos` and `sin` are the DSP's own fast
approximations, so a host model cannot reproduce them and refuses rather than
guessing. And no emitter here produces `DSP_OP_TENSOR_CONVERT` at all, so
modelling it would have been work with nothing on the other end.

What none of it establishes is that the DSP computes the same thing. Modelling
the general path rather than the HVX fast paths is deliberate -- they are
supposed to agree with it -- so agreement says the encoding and the arena are
right, and says nothing about the kernels. Softmax, the reductions and flash
attention are not modelled, so a subgraph reaching one raises instead of being
compared, which leaves all 28 attention nodes unverified.

## Running the kernels on hexagon-sim

Everything above checks a transcription of the C. This runs the C.

The SDK's tools 19.0.04 simulator supports v79, and the flow that works is the
one \`run_main_on_hexagon_sim\` sets up: build a normal Hexagon **shared object
with a \`main()\`** and let the simulated QuRT load it.

    hexagon-clang++ -mv79 -mhvx -mhvx-length=128b -mhmx -O2 -fPIC -shared ...
    hexagon-sim -mv79na_1 --usefs <dir> --rtos osam.cfg <sdk>/rtos/qurt/computev79/sdksim_bin/runelf.pbn \
        -- <sdk>/libs/run_main_on_hexagon/ship/hexagon_toolv19_v79/run_main_on_hexagon_sim -- runner.so

Four things have to be right, and each cost a run to find:

- **A shared object, not an executable.** The \`qurt_*\` symbols are undefined on
  purpose, because the simulated RTOS resolves them at load time. An executable
  has to satisfy them at link time and cannot.
- **\`-nostdlib++\` with the static \`libc++.a\` and \`libc++abi.a\`.** Linking \`-lc++\`
  leaves a \`NEEDED\` entry for \`libc++.so.1\` and the loader fails with "dlopen
  failed for needed library". This is the same trap the skel's own build
  documents, and it catches the test harness too.
- **\`-L\` on the \`pic\` directory.** Without it the linker takes \`libc_eh.a\` from
  the non-PIC directory and every exception-handling relocation fails.
- **\`libncurses.so.5\`.** hexagon-sim predates the ncurses 6 soname bump, so a
  current distribution needs a directory providing it, named through
  \`HEXAGON_SIM_LD_LIBRARY_PATH\`.

\`test_hexagon_sim.py\` is then an ordinary pytest: it builds the same vendored
sources the skel is built from, runs them, and compares fp16 **bit patterns**
against torch. It takes about six seconds. With no \`HEXAGON_SDK_ROOT\` it skips
with a reason rather than passing.

As it stands three op families are exact on the real kernels: the fused norm,
the batch matmul and the raster blit, each agreeing with torch to the last bit.
Getting there needed one correction that was ours and not the kernel's --
`htp_ops_layer_norm` reads **gamma and beta as fp32** while src and dst are fp16,
so a harness handing it fp16 tables gets garbage and it looks like a kernel bug.
RMSNorm has no affine, which is why it matched first and pointed at the affine as
the suspect. No emitter reaches that path today, and the host interpreter
declines it rather than modelling it, so the fact was only recorded rather than
used.

The checks fail for the right reason. Deleting the mean correction from
`htp_ops_layer_norm_one_batch` -- the one step that separates LayerNorm from
RMSNorm -- fails the LayerNorm test and leaves RMSNorm passing. Writing one lane
into every column of the HVX transpose's tail store fails the blit. Making
`htp_ops_batch_matmul` refuse its own loop bound fails the matmul. All three
revert byte-identical.

Two things came out of mutating the kernels that were not the point of mutating
them. The matmul does **not** take the generic accumulation in
`htp_ops_loop_matmul_region`, and the blit does **not** take the row copy at the
end of `htp_ops_raster_blit`; breaking either leaves the result untouched,
because the matmul goes through the HMX path and the blit through the HVX
transpose. That is worth stating plainly, because the host interpreter models the
*generic* paths -- so what has just been measured is the one claim its design
rests on and could not test: **the fast paths agree with the generic ones**, bit
for bit, on this input. Disabling both HMX paths for the matmul leaves the result
unchanged, which is the same statement made from the other side.

The runner above takes hand-written descriptors. `test_blob_on_sim.py` removes
that seam: it produces a real blob for a real subgraph, embeds its bytes in the
runner, and has the DSP parse the header, derive the section bases and the arena
from it, and execute the command stream through the same kernels. The op layout
comes from the backend's own `hexagon_schema.h` rather than a C-side
transcription, and the section bases are recomputed on the DSP rather than passed
in, so the alignment rule is an independent implementation and not a shared
constant.

That makes three executions of one artifact -- torch, the host interpreter, and
the DSP -- and the artifact itself is whatever the emitter wrote. Breaking the
permute region's destination column stride, or zeroing the matmul's contraction
stride, fails the DSP comparison; the emitter is inside the check, not beside it.

Seven blobs now run this way, and they are the same subgraphs the host
interpreter is checked on: the concatenate, slice and transpose chain, the fused
norm, the cache advance at three positions and two value shapes, a product and a
broadcast. Tensor convert is named in the DSP's op enum but no emitter produces
it, so nothing here meets it and the interpreter raises rather than models it. The fused norm is the one
case compared with a tolerance rather than by bits, because the kernel reduces in
a different order than numpy does and fp16 carries about three digits.

Two things the DSP side taught that the host model could not. The cache advance
lowers to two blits and not to a blit plus an element-wise op, so the position
arrives purely as a region offset through the patch slot. And the count passed
alongside the broadcast tail is a fixed 25 rather than a function of the rank:
the kernel rejects anything shorter, takes the non-broadcast path instead, and
returns success having written nothing. A runner that passes the natural
1 + 3 * rank gets a zeroed output and no error at all, which is the kind of quiet
failure this whole exercise exists to catch. The extents and the two stride lists
are each padded to eight words whether or not the rank needs them.

The checks fail for the right reason here too. Unscaling the cache's patch_scale
fails the DSP comparison, and so does shortening the runner's broadcast count
from 25 to 10, at each of the three cache positions. Swapping the two operands'
stride lists makes the host model index
past the end of its operand -- a wrong stride is a read out of bounds rather than
a wrong number -- and the five blobs it feeds then error instead of running.

Softmax and the reductions are modelled now, so the interpreter no longer stops
at nine ops but at one: flash attention. The reduction agrees with numpy to
within one ulp of a denormal -- the kernel sums in fp32 exactly as numpy does and
only the final rounding differs -- which makes it the second family, after the
blits, that is effectively bit-exact. Getting softmax_ops.cc to compile in this
runner needed one concession: it includes htp_ops.h, the IDL header the backend's
CMake generates with QAIC, and nothing that header declares is used by that
translation unit -- the only identifiers the two share are AEEResult and the AEE_
result codes, which come from AEEStdErr.h. The runner supplies an empty file of
that name rather than depending on a configured build tree, and if the include
ever becomes load bearing the compile fails rather than quietly losing a
declaration.

The softmax is a different story and worth stating plainly. htp_ops_softmax
exponentiates with hvx_my_exp2_vhf, a degree-six polynomial evaluated in fp16
with qfloat multiplies, and divides by an approximate reciprocal refined once. On
the input above its answer is off the exact softmax by 0.063, while still summing
to one along the reduced axis -- the shape of the distribution moves and not its
mass. Transcribing that polynomial into ordinary fp16 arithmetic lands further
from the kernel, 0.095, than the exact exponential does, which says the qfloat
rounding is load bearing and the kernel's answer is not reconstructible on the
host from the source alone. The check keeps a tolerance for it; what that
tolerance still catches is a wrong axis, a wrong decomposition or a wrong offset,
and swapping the two inner extents fails it.

Every op an emitter can produce is modelled except one: flash attention. This
was first recorded here as "declared rather than modelled" on the grounds that
the emitter was unreachable, and that reason was wrong -- the check below it was
right, the justification was not. `_register_llama_sdpa` registers the attention
emitter only when `exir_ops.edge.llama.custom_sdpa` exists, and in a bare
interpreter it does not: `sdpa_targets()` returns an empty frozenset. But the op
is a C++ custom op in extension/llm/custom_ops/op_sdpa_aot.cpp, and loading the
library that defines it -- torch.ops.load_library on libcustom_ops_aot_lib.so --
makes `torch.ops.llama.custom_sdpa` appear, resolves the edge op, and hangs
`_emit_sdpa` on both overloads. Any real LLM export loads that library, because
that is how the op enters the graph at all, so a delegated subgraph can contain
FLASH_ATTN and the interpreter would refuse it. It is a gap, not a declaration.

What the test around this does still hold. It parses the emitters' source for
their `type=` arguments and fails when one names an op that is neither modelled
nor excused, so the interpreter cannot quietly fall behind the emitter set.
Removing the layer norm model, or the softmax model, fails it. The set it pins is
3, 4, 8, 18, 19, 28, 29 and 38 -- tensor convert is in the DSP's enum but no
emitter produces it, so refusing that one is correct.

What this does not cover is the rest. Flash attention has no host model and no
runner, and modelling it means modelling an online softmax over the same
approximated exponential the softmax test showed is not reconstructible on the
host, so that check would be a tolerance too. Softmax is checked to a tolerance
rather than by bits. The shapes are small, so the blocked and multi-worker paths
are not exercised, and the norm runs over two outer rows and eight channels
rather than a shape a real model would use. And the simulator is still not the
device: the skel load that fails there with `0xffffffff` is untouched, so nothing
here says the FastRPC path or the skel's own deployment works.

## The fp32 segment, measured

The one-layer replay (`/tmp/qwen3_1layer.json`, real 0.6B shapes, 26 s a run)
makes every refusal legible, because the lowered graph keeps each node's dtype
beside its producers':

    view        fp32 (1,3,2048) <- getitem:fp16
    _to_copy    fp16 (1,3,2048) <- view:fp32
    view        fp16 (1,3,3072) <- getitem:fp16
    _to_copy    fp32 (1,3,3072) <- view:fp16
    sigmoid     fp32 (1,3,3072) <- _to_copy:fp32
    mul         fp32 (1,3,3072) <- _to_copy:fp32, sigmoid:fp32
    _to_copy    fp16 (1,3,3072) <- mul:fp32

3072 is hidden_dim, so this is the gated MLP: `sigmoid(x) * x` is SiLU written
out, and the model upcasts to fp32 around it the same way norm.py upcasts around
RMSNorm. The two `view` nodes that stay behind sit on this chain and nowhere
else, which is what makes them a symptom rather than a missing op.

The tempting fix -- accept fp32 in `is_node_supported` and let the runtime
narrow it on the way into the arena, the exception sdpa already enjoys -- does
not work here, and the section above says why: `htp_ops_cast` has no FP32
conversions at all, so a delegated `_to_copy` to fp32 has nothing to call, and
`htp_ops_unary` rejects bytes=4 outside ABS and NEG. Narrowing the gate to fp16
would change the arithmetic rather than the placement. This segment is the same
shape of problem RMSNorm was and wants the same answer: a fused fp16-in/fp16-out
op plus an export-time transform replacing the fp32 round trip. `silu` already
has an emitter and `htp_ops_silu_learned_bias` is linked into the skel, so the
kernel side exists.

What is left that is genuinely implementable, in the order the evidence supports:
`copy_` (fp16 cache writeback, no emitter), `embedding` (fp16 gather,
`shared_gather_ops.cc` exists, needs a new op type), the two RoPE `slice_copy`
nodes (fp16, bounds known only at call time, which is the patch mechanism's job),
and the fp16 `select_copy`; the int64 ones are start_pos scalar reads and belong
on the host.

## select_copy, and a correction about copy_

The one-layer replay put a `select_copy` on the left of the delegate boundary
holding `[1, 1024]` from an operand with more than one entry along that axis. It
was refused for a reason that is easy to miss: `select_copy.int` is already in
`ALIAS_TARGETS`, and that path requires the result to hold the operand's bytes,
which a narrowing select does not. `slice_region` refuses the same node for a
different reason -- it is not a slice -- so neither path took it and the node had
no emitter at all.

`select_region` treats `select_copy(source, dim, index)` as the slice
`[index, index + 1)`, which is what it is, and `_emit_select_copy` sends a
narrowing select down the blit and a same-bytes select down `_emit_alias`. On the
replay the boundary select is taken over, the subgraph count falls from five to
four, and the nodes left behind are int64 reads of `attn_options_input_pos`
feeding `_local_scalar_dense` and the start_pos patch, which belong on the host.
A blob case (J) runs the narrowing form three-way against torch and the DSP, and
aiming the region one row past the one picked fails it.

A correction to the plan this replaced. `copy_` looked like the cheapest win at
56 nodes, but its two instances per layer have no users at all: they are the
`auto_functionalized_v2` writeback into the cache buffer, and delegating them
would still move the same bytes back to the host through `in_place`. Nothing is
gained, and a check could not tell a working writeback from a broken one without
a buffer to read afterwards.

## The gated activation, which was already there twice

The fp32 chain the replay exposed is SiLU written out -- `x * sigmoid(x)`, upcast
to fp32 the way norm.py upcasts around RMSNorm. Everything needed to run it was
already present, in three places, and none of them were connected:

    MNN's kernel        HTP_OPS_BINARY_MUL_SILU = 7, eltwise_ops.cc:502,
                        `v0 * silu(v1)`, dispatched through the ordinary binary
                        elementwise path.
    hexagon_ops.py      BINARY_OP_TYPES has carried "mul_silu": 7 all along.
    blob_interpreter.py _BINARY[7] is `a * b / (1 + exp(-b))`.

So no kernel, no emitter and no host model were needed. What was missing was
something to produce the op, because a graph says `sigmoid` then `mul` and
nothing maps those two onto one command. `mul_silu.py` adds the fused op and the
pass that recognises the decomposition, which is the same shape as the fused
norm: the operand casts are walked through so the kernel gets fp16 while the
check keeps the arithmetic the model wrote.

On the replay the subgraph count falls from four to three and the portable nodes
from 55 to 47. `sigmoid` disappears from the table entirely, the gating `mul`
leaves the portable column, two of the fp32 casts go with it, and one of the two
leftover `view` nodes follows -- which is what makes the "a view is a symptom"
claim concrete rather than a guess.

Two things about the plumbing are worth knowing. Only one TORCH_LIBRARY DEF block
is allowed per namespace, so this one adds a FRAGMENT to the namespace
`rms_norm.py` opens, which makes the import order in hexagon_ops.py
load-bearing. And the numeric check cannot be by bits in principle: the kernel's
silu is piecewise linear. In practice, over gate values within +-24 and up values
within +-12, it is bit-identical to the exact one -- the tolerance is set at 1e-6
rather than 1e-9 only because two different fp16 values are an ulp apart, so
below that the test would be claiming exactness. How the approximation behaves
outside that range is unverified.

Pointing `BINARY_OP_TYPES["mul_silu"]` at 3 -- plain multiply -- fails the blob
test, and so does dropping the silu from `_BINARY[7]` in the interpreter, so
both the kernel and the host model are genuinely being compared.

## A bug the attribution could not see: fp32 results have no way out

Reading the copy-out side of the runtime next to the copy-in side turned up an
asymmetry that nothing currently exercises. The chain:

    llama.custom_sdpa declares torch.float32      measured on the one-layer graph
    _emit_sdpa calls result_for(node, numel)      hexagon_ops.py:837, two arguments,
                                                  so the dtype takes its fp16 default
    method_output(index, bytes_for(numel, fp16))  the slot the blob records is numel*2
    delegate->outputs[i].size = ref.size          hexagon_backend.cpp:362
    the caller hands back an fp32 tensor          4 bytes per element, so 4*numel
    if (tensor.nbytes() > out.size) return InvalidArgument

So a delegated attention fails at the first execute() with "hexagon: output N is
4N bytes, slot holds 2N". The input side already handles the same situation the
other way round -- a fp32 input whose slot is half its size is narrowed on the
way in, hexagon_backend.cpp:551 -- but the copy-out only compares sizes and has
no conversion at all. attention is the subgraph that lands there today, and it
is the one op with no host model and no blob test, which is why the attribution
table could not show this.

The widening has to happen on the host. The DSP cannot do it: htp_ops_cast has
no FP32 conversion in either direction, so TENSOR_CONVERT cannot be the answer
here even though the enum has a slot for it.

Fixed by mirroring the input path, with a half_bits_to_float written next to
float_to_half_bits. What that buys and what it does not: the conversion
algorithm was checked in Python against numpy over all 63488 finite fp16
patterns and matched bit for bit, and the file compiles with the flags the build
tree recorded for it. What is not verified is the behaviour -- no device has
ever completed an execute(), and the sim runner does not run this code. Treat the
fix as reasoned, not demonstrated.

## Casts between the two widths, which the arena already performs

A fp16 to fp32 cast has nothing to do inside a subgraph: every kernel reads and
writes fp16, the runtime narrows a fp32 operand on the way in and widens a fp32
result on the way out, and both directions are exact. fp16 to fp32 is exact
because the target has more bits; fp32 to fp16 is exact in the sense that the
narrowing the runtime performs on entry is the same rounding the cast would do.
So absorbing one changes no value, and the subgraph boundary the cast used to sit
on disappears with it.

What made this a small change rather than a careful one is that the partitioner
already had the safety property. A node whose emitter only re-points its
operand's TensorRef cannot fill an output slot, so a partition that would have to
hand one out keeps it portable instead; that loop walks backwards from the
outputs and drops such a node whenever any consumer is not itself delegated.
Casts join that loop through one predicate, `_emits_no_command`, which is true
for the alias targets and for a cast between fp16 and fp32.

The guard that matters is on the operand, not the result. A cast whose result is
fp16 passes the dtype gate by itself, so an int64 operand would be absorbed and
the kernel would read eight-byte integers as half floats. `_cast_stays_in_fp16`
is that guard, and it is the one thing here that could have gone quietly wrong.

On the replay: subgraphs fall from three to two, the four `_to_copy` nodes move
from portable to delegated, and the portable count falls from 47 to 42.

## The unary family had never run on the simulator

Case L, which is the cast chain folded down to one node, was the first blob in
this suite to contain a UNARY op. It came back all zeros, and the reason was not
the backend: the sim runner's dispatch handled six op types and printed
`UNSUPPORTED` for the rest, so the unary command was never issued and the output
slot kept whatever it started with. `htp_ops_unary` is now linked and called,
which brings abs, neg, gelu, sigmoid, exp, log, silu, tanh, sqrt and rsqrt into
reach of a three-way check for the first time. The host interpreter modelled them
all along, so only the DSP side was dark.

Dropping the branch again fails case L, which is what makes the new coverage
worth something.

## The RoPE tables, where only the offset moves

The two slices left over on the replay cut the RoPE frequency tables to the
prompt: a table of 256 rows, a fixed number of rows out, beginning wherever the
sequence is. The old support check refused them because it wanted an int for the
start, and the comment there said a dynamic end has no parameters. Measuring the
node showed the end is not the problem. The extent comes from the result, which
is the number that has to fit the buffer anyway, so the only part that moves is
the source offset -- one word, which is exactly what the patch mechanism fills.

So `slice_region` now takes a start that is either an int or a node, derives the
run from the result shape rather than from the end argument, and reports the node
and the row's worth in elements when the offset has to be patched. The emitter
sends that node along as an extra input the blit never reads and puts a patch on
params[4]. The replay takes both nodes over: `slice_copy` goes from four
delegated and two portable to six and none, and the portable count falls from 42
to 40.

## Two things the slice turned up

The first is an interpreter bug the new case caught immediately. The blit model
picked its destination with `refs[params[2]]`, where params[2] is the source
count and refs is inputs followed by outputs. That is the output only while there
is exactly one source and nothing else riding along, which was true of every blit
until a patched slice arrived; with the start row appended, the destination
became the row tensor. The destination is now the op's own output, which is what
the DSP writes to.

The second is worth recording because it nearly passed for verification. The new
case uses a start of 2 and a row length of 8, so a patch that fails to scale the
row number reads element 2 where it should read element 16. The fixture was
`_small`, whose values repeat every seven elements -- and 16 minus 2 is 14, two
whole periods, so both offsets read identical values and the mutation passed.
The table is now built from `arange`, where every row differs, and the same
mutation fails. A check that agrees with a wrong implementation is not a check,
and the only reason this was caught is that the mutation was run before the
result was believed.

## embedding, and why it is staying on the host

The last gap on the list was the embedding lookup, expected to need a new op type.
It does not: the op type is already there and already dispatched. The enum in
htp_command.h has DSP_OP_SHARED_GATHER at 23, and execute_command.cc:759 calls
htp_ops_shared_gather through it. What it cannot do is read the table this graph
carries.

    htp_ops_shared_gather(dst, indices, weight, selectSize, ic, oc, bytes,
                          isInt4, scaleBlockNum, scaleAsymmetric)

The int4 and int8 branches want quantized weights. The fp16 branch, which is the
one that would apply here, indexes the table as

    const int tileY = index / 32;
    const __fp16 *tile = weight + ((size_t) tileY * icP + x) * 32 * 32;

so the weight has to be in 32 by 32 blocks, the layout MNN's weight reorder
produces. A plain row-major table read through that address arithmetic returns
other rows' values, silently. The only other candidate was DSP_OP_SELECT, and
htp_ops_select is `cond ? a : b` -- four pointers, none of them an index.

The alternative a blit would offer is one region per token, which is
arithmetically right. It fails on the patch mechanism, which fills exactly one
word per op and reads it from a fixed offset inside an input slot, so the row
numbers for a prompt's worth of tokens cannot all arrive that way.

Repacking the table is the one route that would work, and it is not this
backend's to take: the weight arrives at a subgraph as a placeholder rather than
a constant, so the packing would have to happen outside the exported program and
the .pte would stop being self-contained.

Leaving it portable costs almost nothing, which is the part that decides it. The
lookup copies one row of 1024 fp16 values per token -- 6 KB for a three-token
prompt -- beside an attention that reads the whole 256-row cache per layer. The
node stays where it is on purpose, not for want of looking.

## The whole model, against the same model without the passes

The one-layer replay is a cheap stand-in, so the passes were also measured on all
twenty-eight layers of Qwen3-0.6B, against the same export with the passes
switched off.

                        passes off     passes on
    delegated subgraphs        169            29
    nodes delegated           1714          1967
    nodes left on the host    2121           825

Twenty-nine subgraphs for twenty-eight layers is a single partition per layer,
where the export without them chops each layer into six. The host is left with
sixty-one percent fewer nodes.

What is left is accounted for. `_assert_scalar` (227), `getitem` (114), `le`
(113), `_local_scalar_dense` (85), `lt` (57), `ge` (57), `add` (57) and
`sym_size` (1) are shape guards, 711 of the 825, and none of them is a gap. The
85 `select_copy` are int64 reads of the sequence position; the arena holds fp16,
so they belong on the host. That leaves two things. The one `embedding` is
covered above. The 28 `view_copy` are one per layer, and a view adds no command,
so the boundary rule keeps it out whenever anything downstream of it is not
delegated -- the symptom the plan predicted, not an op that is missing. The
one-layer replay shows the same shape: one `view_copy` against seventeen
delegated, which is exactly the twenty-eighth of the full model.

## Attention: the null cache, and how far the fix is verified

The emitter's input list was

    [q, k, v, mask, ABSENT, ABSENT]

and the two slots at the end are not spare. htp_ops_flash_attn takes the past
keys and values there -- mapped_ptrs[4] and [5], with the output and workspace
after them at mapped_ptrs[inputs->size()] -- and it uses them twice before it
looks at the query: htp_ops_push_kv writes the new rows into the cache through
the first two, with no null check anywhere in it, and sync_attention passes both
to the score and output matmuls as their key and value. So the command this
backend emitted wrote through a null pointer and then read its keys and values
from address zero. It could not have run, and no delegation count could show
that.

The cache belongs in those slots. It is an operand of this op rather than
something the kernel keeps: the graph carries its own update_cache, which the
KV-cache pass fuses into one node and which runs before attention, so the keys
and values the op is handed are the whole cache -- 128 rows of 8 heads for a
three-token prefill, which is what max_kv_len, read off kv_shape[1], already
assumed. The kernel's push then copies cache rows onto themselves and changes
nothing.

The scratch was wrong too. attention_entry.cc sizes a different kernel's
workspace; the one this op runs is one buffer of qo_len scores and one of qo_len
probabilities, each over the sequence rounded up to 32 and 128-aligned, taken
once per worker. The worker count is chosen on the DSP from g_max_num_workers,
which the host cannot know, so the emitter now asks for the widest count any
head could take: one slot per head.

Neither fix is demonstrated. The host model for FLASH_ATTN is written --
causal by clamping the row rather than by masking, kv head h / (n_heads /
n_kv_heads), the scale from params[6] read as float bits -- and the simulator
runner dispatches the op, but the case that would compare the three results does
not run yet, and the reason is not the one this section first gave. The op does
register: llama.custom_sdpa is a C++ custom op, and importing
executorch.extension.llm.custom_ops.custom_ops puts it in the dialect, after
which sdpa_targets() is non-empty and the case builds its blob.

What stops it is the run. The attention kernel brings up a worker pool on the
DSP -- worker_pool_global_init, which sizes itself from qurt_hvx_get_units -- and
hexagon-sim cannot start it: qurt_cb_fwk_worker_init returns -4, QuRT reports
0x7103, and the whole runner aborts. That takes the run down with it, so with the
import in place every case in test_blob_on_sim.py stopped being checked, not just
this one. The import is therefore out of the test and the case sits behind
sdpa_targets(), which is empty without it. A skip is not a pass, and a run that
skips everything is worse than one case that does not run: attention is still the
least verified op in the model, now with a known bug fixed behind it.

## clamp, and the compare that orders a NaN above its bound

`aten.clamp.default` and `clamp.out` are delegated now, with the two bounds the
graph carries rather than a tensor. The op rides the unary machinery -- the same
worker pool, the same fp16 chunking -- but its params[3] and params[4] hold the
fp16 bit patterns of the bounds instead of sitting unused, so it has an entry
point of its own in the skel, `htp_ops_unary_clamp`, rather than widening
`htp_ops_unary` for every other type as well. Narrowing a Python float bound to
fp16 happens at export, which is the narrowing torch's portable kernel applies to
the same bound before comparing.

The kernel is `min(max(x, lo), hi)` in that order, torch's order, which is why a
range whose lower bound sits above its upper one returns the upper bound. The
vector path is two fp16 compares and two selects. Its first device run returned
exactly one wrong element out of 4096: the NaN. The Hexagon fp16 compare is not
ordered -- a NaN input compares greater than its upper bound -- so that element
came back as the bound where torch returns the input. The fix asks the compare
unit nothing about NaN: a NaN is an all-ones exponent with a non-zero mantissa,
so `|x| > 0x7c00` as an integer compare, and the input is muxed back over the
clamped value, payload included. The scalar tail uses the C comparison, which
propagates NaN on its own.

Six probe exports carry the check, each one clamp and one delegate: 4096 elements
so every one is in a vector, 4161 so the scalar tail runs, both bounds, one bound
either way -- a one-bound clamp arrives as `args = (x, bound)` and means `min`,
which is the form `clamp(x, min=0.5)` and `clamp(x, 0.5)` both lower to, and
which the emitter used to index past the end of -- an omitted bound, and the full
fp16 range as bounds. All six come back bit for bit identical to torch's own fp16
result on the device, with an infinity and a NaN in each input. The four-layer tower with its
clamps delegated reproduces, bit for bit, the export whose clamps ran on the host
(md5 73e5df61 on the same input), and the 24-layer split export is still
18b6997d.

One consequence belongs with the fence measurements in `VISION_TOWER.md`: with
clamp supported, the two no-op clamps that were holding the score matmul and the
softmax apart are delegated like every other node, so that graph comes back as a
single delegate and its DSP time per execution for four layers goes from 94.3 ms
to 177.7 ms. A supported op cannot be a fence.
## Not verified

The bar every op taken over here had to meet was a real blob on hexagon-sim
agreeing with torch. What follows is what did not meet it, so that a green suite
is not read as more than it is.

Flash attention has never run on the DSP. `llama.custom_sdpa` has an emitter,
the partitioner delegates it, the blob it produces carries the command, and there
is now a host model of the op in blob_interpreter.py -- but no case issues one,
because hexagon-sim aborts in the kernel's worker pool before the kernel runs. It
is the single most expensive op in the model and it is the one with the least
evidence behind it. The host model itself is therefore unverified too: a model
nothing has been compared against is a guess with arithmetic in it. The patch
path attention depends on for start_pos is in the same position: the blit patch
is exercised by case N, the attention patch by nothing.

Nothing here has run on hardware. Every result comes from hexagon-sim, which is a
functional model: it says what the kernels compute, not what the DSP costs or
whether the FastRPC path and the skeleton deployment work. The skeleton still
fails to load with 0xffffffff and that has not been chased.

The fp32 widening fix is argued rather than shown. The runtime narrows inputs into
the fp16 arena and did not widen outputs back; the change that does so was read
out of the copy-out path and never demonstrated with a blob that returns fp32,
because the simulator does not exercise the RPC copy that it lives in.

The tolerances are what they are. mul_silu is bit-identical to an exact silu only
over gate values inside 24 and up values inside 12, and outside that range
nothing was said. Softmax is not bit-exact at 0.08 and transcribing the kernel's
exp2 approximation lands further off, so the qfloat rounding is load-bearing and
the agreement is with the algorithm rather than with the arithmetic. Each kernel
was checked at the shapes its case builds, which are not every shape this model
produces.

## Traps hit along the way

- `pgrep -f <script>` matches the calling shell's own command line and can kill
  it mid-command. Use `'prefix'"suffix"` quoting.
- A build script that `cd`s to a path that does not exist keeps running in the
  previous directory, so `cmake -S .` configures the wrong tree and still exits
  0. Verify via `CMAKE_HOME_DIRECTORY`, and have the export harness print the
  `hexagon_ops.__file__` it resolved.
- `is_node_supported` used to require every arg to be a `torch.fx.Node`, which
  silently rejected anything taking an int, bool or list (`mean.dim`'s `dims`
  and `keepdim`). Non-tensor args are now filtered out.
- **Test the real pipeline, not a copy of its conditions.** A minimal
  `custom_sdpa` module delegated fine while Qwen3 delegated none, because the
  minimal test happened to load the LLM extension first and so got a non-empty
  `SDPA_TARGETS`. The difference between the test and the real path was exactly
  the condition under test.
- **The partitioner ORs your support check back in.**
  `generate_partitions_from_list_of_nodes` builds
  `any_chain(op_support, MatchTag())`, and `MatchTag` accepts every node the
  pattern list tagged. So removing a node from the list handed in does **not**
  reject it -- the support object has to return the same answer, or the OR lets
  it straight back through. This is why `HexagonOperatorSupport.boundary_views`
  exists instead of the rule standing alone in `partition()`. Finding it cost
  three full export runs, because the symptom was a subgraph that still contained
  nodes the partitioner had just dropped.
- **A code path that no emitter reaches is not tested, however long it has been
  there.** `ctx.constant` only ever ran for `layer_norm`, which is unreachable,
  so its `get_parameter` assumption had never met a real subgraph. The first op
  to actually use it failed on the first run. "It is already written" and "it
  has run" are different claims.
- **An alias emitter cannot fill an output slot.** `_emit_alias` re-points the
  operand's TensorRef and writes no command, so when such a node is handed out as
  a subgraph output nothing ever claimed its slot and `build` failed with "method
  output 1 has no slot" -- an index, not a node. A view that is handed out now
  writes its bytes with a blit instead. The general lesson is that the failure
  named the symptom, and only dumping the output list against the claimed slots
  named the cause.
- Exit codes carry almost no information in this pipeline. Prefer printing the
  invariant you actually depend on (the resolved target set, the emitted params)
  over checking that a command succeeded.
