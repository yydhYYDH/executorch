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
| `llama.custom_sdpa.default` | 0 | 28 |
| **Total** | **394** | **646** |

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
C and Python; `sizeof(HexagonOp)` == `OP_SIZE` == 476, asserted on both sides
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

Of the 20 registered emitters only 5 have produced a command on a real graph --
`mm`, `mul`, `add`, `neg`, `custom_sdpa`. The rest are unexercised, and
`aten.layer_norm.default` is unreachable as written because `to_edge` rewrites
it to `native_layer_norm`, which returns three tensors.

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
- Exit codes carry almost no information in this pipeline. Prefer printing the
  invariant you actually depend on (the resolved target set, the emitted params)
  over checking that a command succeeded.
