# Hexagon backend op gaps

<!-- Hand-written. `OP_SUPPORT.md` is the generated half: it is rendered from
     `hexagon_ops.EMITTERS` by `scripts/gen_op_support.py`, and
     `test/test_op_support.py` fails when the two drift. This file covers what
     that table cannot: kernels the vendored library has and nothing emits,
     operators that would need a kernel to exist before an emitter is a
     question, and what either costs a real model. Every claim below names the
     file it was read from, so a reader can tell a measurement from a reading. -->

Computed against `53369ff` on 2026-09-24. The library inventory below is the
vendored tree at that commit.

## The three verdicts

| verdict | what it means |
|---|---|
| **wired** | `EMITTERS` has an entry and the support predicate accepts the node: it reaches a DSP command |
| **refused** | the predicate rejects this node, which falls back to a portable kernel. That is the intended outcome, not a failure: the alternative is an emitter reading an operand it does not understand |
| **unwired** | no emitter at all. `is_node_supported` returns `False` before it reads anything, so no line of the graph says why |

The distinction matters because only the last one is a gap that hides. A refused
node costs speed; an unwired op costs speed and leaves nothing to grep for.
`unwired_overload_census()` (`partition/hexagon_partitioner.py`) records every
target absent from the table while its family is present, and
`test_overload_census.py` + `test_overload_census2.py` pin 144 rows whose 184 op
verdicts are 88 wired, 41 refused and 55 unwired.

## 1. The vendored library has no kernel at all

These are not emitter problems. The two subtype tables and the op enum do not
contain the operator, so an emitter would have nothing to call.

| op | why there is no kernel |
|---|---|
| `aten.amin.default`, `aten.min.default`, `aten.min.dim` | `HtpOpsReductionOpType` is `SUM`(1), `MAXIMUM`(2), `MEAN`(3) -- `eltwise_ops.cc:2456-2459`. There is no minimum to select, and the dispatcher rejects anything else (`eltwise_ops.cc:2441-2445`) |
| `aten.argmax.default`, `aten.argmin.default` | Every reduction here walks values; `argmax`'s output is positions. `topk` is the one kernel that writes both -- §2 |
| `aten.prod.default`, `aten.var.correction`, `aten.cumsum.default` | A running product, a second moment and a prefix scan are each a different walk from the three reductions that exist |
| `aten.erf.default` | No `HtpOpsUnaryOpType` entry: the enum is 1..17 (`unary_ops.cc:14-31`) |
| `aten.leaky_relu.default`, `aten.elu.default` | The binary table has no slope form (`eltwise_ops.cc:27-38`) and the unary table no `elu` |
| `aten._log_softmax.default` | Composes a log with a softmax in a way no single command describes |
| `aten.convolution.default with transposed=True` | The im2col kernel walks its window forward over the input; a transposed convolution is a scatter read |
| `aten.repeat.default`, `aten.flip.default`, `aten.constant_pad_nd.default` | Each is a different region walk from the blits that exist |
| `aten._adaptive_avg_pool2d.default` | The pool command takes one fixed window and stride; an adaptive output sizes the window per output position |
| `aten.pow.Tensor_Tensor` | No pow kernel, and the unary table's `SQUARE` is exponent 2 only |
| `aten.clamp.Tensor` | The clamp entry point carries its bounds as two fp16 params, so an operand bound has nowhere to go |
| `aten.full`, `aten.full_like`, `aten.arange`, `aten.scalar_tensor` | Not kernels: nothing emits a tensor that was not read from memory |

## 2. Kernels present, nothing on the host emits them

`htp_command.h:32-81` declares 48 op types. 43 have a `case` in
`execute_command.cc`, and the host side emits 18 of them. The other 24 are
kernels the skel links and exports with nothing that can reach them:

| value | DSP op | kernel | what an emitter would serve |
|---|---|---|---|
| 5 | `DSP_OP_BINARY_BLIT` | `htp_ops_binary_blit` | unread |
| 6 | `DSP_OP_LOOP_BLIT` | `htp_ops_loop_blit` | unread |
| 7 | `DSP_OP_TENSOR_CONVERT` | `htp_ops_tensor_convert` | deliberately unused: the cast table has no FP32 forms, and the runtime converts at the arena boundary instead |
| 9 | `DSP_OP_LAYER_NORM_PACKED` | `htp_ops_layer_norm_packed` | a packed-input form of a norm that is already wired |
| 10 | `DSP_OP_WEIGHT_REORDER` | `htp_ops_weight_reorder` | see §6: the README says init uses this; nothing calls it |
| 11 | `DSP_OP_WEIGHT_REORDER_INT4` | `htp_ops_weight_reorder_int4` | same |
| 13 | `DSP_OP_SCALE` | `htp_ops_scale` | unread. `aten.mul.Scalar` goes through `UNARY`'s `SCALE`(17) subtype, not this op type |
| 15 | `DSP_OP_ROPE_FUSE_LAYERNORM` | `htp_ops_rope_fuse_layernorm` | a fused rope + norm, one command where two are emitted now |
| 17 | `DSP_OP_CONV1X1_DIRECT_FP16` | `htp_ops_conv1x1_direct_fp16` | the 1x1 fast path. A 1x1 convolution goes through im2col today |
| 22 | `DSP_OP_MATMUL_Q4A16_FP16` | `htp_ops_matmul_q4a16_fp16` | **quantized prefill** (`M > 1`) |
| 25 | `DSP_OP_CAST` | `htp_ops_cast` | deliberately unused, as 7 |
| 26 | `DSP_OP_SELECT` | `htp_ops_select` | `cond ? a : b`. The only ATen node that maps is `aten.where.self`, refused today for the bool reason below |
| 27 | `DSP_OP_TOPKV2_K1_FP16` | `htp_ops_topkv2_k1_fp16(values, indices, input, rowSize, rows)` | **`topk`'s two outputs**: this kernel produces both, so the gap is the emitter and the getitem rule, not a kernel |
| 30 | `DSP_OP_RELU6` | `htp_ops_relu6` | deliberately unused: `relu6` is `hardtanh(0, 6)` and goes through `UNARY`/`CLAMP`, the entry point that restores a NaN by a bit test |
| 31 | `DSP_OP_MASKED_REDUCTION` | `htp_ops_masked_reduction` | a reduction under a mask |
| 32 | `DSP_OP_TMAC_A16W1` | `htp_ops_tmac_a16w1_fp16` | 1-bit weights |
| 33 | `DSP_OP_FLASH_ATTENTION_BLOCK` | `htp_ops_flash_attention_block` | a blockwise attention |
| 34 | `DSP_OP_MATMUL_Q4A16_BLOCK_FP16` | `htp_ops_matmul_q4block_a16_fp16` | quantized prefill, block-scaled form |
| 36 | `DSP_OP_LSTM` | `htp_ops_lstm` | an LSTM cell. The recurrent models are unrolled before the partitioner, so nothing reaches it |
| 37 | `DSP_OP_RELU` | `htp_ops_relu` | deliberately unused, as 30 |
| 39 | `DSP_OP_PRELU` | `htp_ops_prelu` | `prelu`, which decomposes into `gt` + `mul` + `where` before the partitioner sees it (§3) |
| 40 | `DSP_OP_CONV1X1_DIRECT_W8A16_SYM_PER_CHANNEL` | `hmx_conv1x1_direct_w8a16_sym_per_channel` | quantized 1x1 convolution |
| 42 | `DSP_OP_MATMUL_W8A16_BLOCK_FP16` | `hmx_matmul_w8a16_block_fp16` | quantized prefill, int8 weights |
| 44 | `DSP_OP_VISION_FLASH_ATTENTION_FP16` | `htp_ops_vision_flash_attention_fp16` | a vision attention variant |

Of the 48, 43 have a dispatcher case: the 18 the host emits, the 24 above, and
`GET_INFO`(20), which the host never emits either. The 5 without one are
`RESERVED_0`(0), `RESERVED_21`(21), `COMMAND_GROUP`(99) and `MAX`(100) --
protocol, not gaps -- and `DSP_OP_POST_ATTN_REDUCE_FUSE`(35), which §6 is about.

## 3. One emitter away, inside a family that is already wired

These are the cheapest real gaps: the kernel is there, the command is there, and
the only thing missing is the line that produces it.

- **`aten.sin`, `aten.cos`, `aten.expm1`**: subtypes `SIN`(14), `COS`(13),
  `EXPM1`(12) exist and have kernels, and `UNARY_OP_TYPES` already carries all
  three (`hexagon_ops.py:129-131`). What is missing is the emitter, and the
  reason it is worth hesitating is the arithmetic rather than the plumbing:
  these run the DSP's own approximations, whose error has never been measured.
  `test_blob_on_sim.py` cannot settle that by agreement alone, which is what
  made `x ** 2` a one-line change and these three not.
- **`aten.gt`, `aten.ge`, `aten.lt`, `aten.le`, `aten.eq`, `aten.ne`**:
  `GREATER`(9) and `LESS`(10) exist. They write int32 1/0 or fp16 1.0/0.0, and
  the table has no one-byte mode, so a node declaring `torch.bool` cannot be
  handed one without an out-of-bounds write. This is a kernel change, not an
  emitter change. `SQUARED_DIFFERENCE`(11) has no ATen node at all.
- **`aten.where.self`**: see `DSP_OP_SELECT` above. The condition operand would
  have to carry a comparison result, and whether it can is not established.
- **`aten.prelu`**: `DSP_OP_PRELU`(39) exists and no emitter uses it, because the
  node does not survive export at all -- `F.prelu` arrives as `gt` + `mul` +
  `where` (`test_overload_census.py`, row "prelu decomposes, and the where stays
  put"). Reaching the kernel would take a fusion pass of the `mul_silu.py` kind,
  not an emitter.
- **`aten.topk.default`**: the kernel answers both outputs; the missing pieces
  are the emitter and the all-readers-are-getitem-0 rule (§4).

## 4. Shape and parameter limits of the ops that are supported

These are `refused` rather than `unwired`. There is an emitter, and the
predicate keeps the node off the DSP, so the fallback works. They are listed by
*limit* rather than by op because that is the form in which they cost a model:
the same node delegates or does not depending on a shape.

| limit | ops it holds back | what it would take |
|---|---|---|
| The arena holds two bytes per element | every op: fp16 only. int64 casts, `select_copy` on int64 (85 of them per Qwen3 forward pass, all sequence-position reads), and any `torch.bool` operand | a wider dtype on the DSP, or keeping these on the host, which is where they belong |
| The DSP's broadcast path needs 25 more params than a command carries | every binary op: operands must be the result's shape or a scalar | a command form with a region list, or a layout transform at export |
| One contiguous `[outside][reduce][inside]` span per reduction | `mean.dim` and `sum.dim_IntList` with non-adjacent reduced dims, `amax` the same way | a strided reduction, which is the same kernel the softmax row below needs |
| Softmax is last-axis only | `_softmax` over any other axis. Measured: `[1,2,4,8]` over dim 1 came back with 8 of 64 elements past 1e-2 (worst 1.1e-1) where the last-axis form is exact to 4.9e-4 | the kernel's strided reduction path has to agree with torch before the emitter can take it |
| Pooling needs `C == 64` and static shapes | `max_pool2d`, `avg_pool2d` | the blocking the pool kernel assumes is one 64-channel block |
| Convolution needs static extents, a constant weight and `groups == 1` or `groups == C_in == C_out` | `convolution` | the general group count has no kernel; a run-time weight has no command |
| The gather table is a constant, tiled at export, indexed by int32, along axis 0 | `embedding` (int64 indices are refused by dtype), `index_select` (dim must be 0), `index.Tensor` (one index only) | a run-time table, or an index the command can describe. A `tokens.to(torch.int32)` in the model is enough for the common case |
| The GEMV entries read one activation row linearly | the weight-only quantized matmul: `M == 1`, `K % 64 == 0`, `N % 32 == 0` | pack64 activation + output repack, i.e. §2's `MATMUL_*_BLOCK_FP16` entries |
| A command holds three 12-int regions | `cat` (at most 3 operands), `permute_copy` (at most 3 ordered groups, no reversal inside one) | a command form with more regions |
| A blit describes one run per row | `slice_copy` with `step != 1` | a strided run, or a different command |
| A two-output node is placed only when every reader takes the values output | `max.dim` (indices), `max_pool2d_with_indices`, `native_layer_norm` (`getitem 0`), `add_rms_norm` (`getitem 0` or 1) | the kernels compute values, not positions. `topk`'s kernel is the exception that proves it, and the rule would have to be extended for it |
| The narrow view rule: a view that would be a partition boundary is copied instead | `view_copy`, `unsqueeze_copy`, `squeeze_copy`, `expand_copy`, `alias_copy` | nothing -- this is the rule working, and its symptom is a delegate one op shorter than it looks |

## 5. What the gaps cost a model

Qwen3-0.6B, from the README's Status section: 1967 nodes reach 29 delegates (one
per layer -- a count that predates the mask refusal in `_sdpa_fits_dsp_limits`,
so an export whose attention nodes carry a mask delegates fewer than 28) and 825
stay on the portable kernels. 711 of those 825 are shape
guards (`_assert_scalar` 227, `getitem` 114, `le` 113, `_local_scalar_dense` 85,
`lt` 57, `ge` 57, `add` 57, `sym_size` 1), which is what a dynamic-shape export
looks like rather than a gap. The real remainder is small:

- **`embedding`**, because the graph carries int64 indices. One cast in the model
  source moves it.
- **one `view_copy` per layer**, which is the boundary rule above.
- **85 int64 `select_copy`**, the sequence-position reads, which belong on the
  host because the patch mechanism needs the value where it is.

The two gaps that are not of that shape:

- **Quantized prefill.** The weight-only path is `M == 1` only, so a quantized
  model can decode and cannot prefill. §2 lists three kernels that would change
  that, and both GEMV entries that are wired have run on hexagon-sim but never
  against a timing.
- **Multimodal.** `vision_attention` is one block of a tower: the patch
  embedding, the layer norms, the MLP and the projection into the text embedding
  space are all still needed, and `DecomposePatchEmbed` covers only the
  conv-to-matmul step. No multimodal model has run end to end on this backend.

## 6. Discrepancies noticed while compiling this

- **`README.md:40` says init delegates a runtime weight reorder to
  `htp_ops_weight_reorder`.** Outside the vendored tree the name appears in
  prose only -- `README.md:40` and `README.md:458` -- and nothing in
  `runtime/`, the emitters or the partitioner emits `DSP_OP_WEIGHT_REORDER` or
  `DSP_OP_WEIGHT_REORDER_INT4`. Either the sentence describes a fallback that has
  never had a caller, or the reorder this backend relies on happens at export in
  the Python packers and the sentence is stale. Worth a decision rather than a
  silent fix.
- **`DSP_OP_POST_ATTN_REDUCE_FUSE`(35) is declared and has no `case`** in
  `execute_command.cc`. Which of the two is missing has not been established.
- **The q4a16 and w8a16 packers have never run on a DSP.** The upstream host
  reorders are not in the vendored tree (`src/host/` was dropped at vendoring),
  so the tile order is agreed with by reading the kernel header and the read
  path, not by a device (`hexagon_ops.py:1497-1550`).
- **`OP_SUPPORT.md` has no row for `aten.prelu`**, although the census pins its
  decomposition and the library carries a kernel for it (§3). Nothing is wrong
  today -- the nodes that reach the partitioner are `gt`, `mul` and `where`, and
  `where` is refused with a reason that is written down -- but a reader looking
  for prelu in the generated table finds nothing, and the census row is where the
  answer lives instead.
- **The weight-only quantizer annotated `mm` and `addmm` but not the 2-D
  `aten.matmul`**, so a model written `a @ b` never reached the quantized kernels
  and both schemes produced identical output. Fixed in `53369ff`; recorded here
  because the same shape of gap -- a spelling the emitter table does not know --
  is the one `unwired_overload_census()` now counts.

## 7. Priority

1. **Quantized prefill (`M > 1`)** -- the only functional gap: it is the
   difference between a quantized model that runs and one that only decodes.
2. **`TOPKV2_K1_FP16` and `MASKED_REDUCTION`** -- sampling and masked reduction
   are what a decoder needs next, and both have kernels.
3. **`aten.where` via `DSP_OP_SELECT`** -- `prelu` decomposes into `gt` + `mul` +
   `where`, so this one command is what a fusable prelu (and every masked write)
   is waiting on. The open question is whether the kernel's condition operand
   accepts what a comparison writes, since a `torch.bool` node cannot be handed
   an int32 buffer.
4. **`sin`/`cos`/`expm1`** -- the emitters are trivial; what they need first is a
   measurement of the approximations' error.
5. Everything else in §1 would need a kernel that does not exist, and everything
   in §2 is a kernel no graph has asked for yet. Neither list is a defect.

## Appendix: reproducing the numbers

```sh
# the generated half, and the guard that keeps it equal to EMITTERS
PYTHONPATH=src python -m pytest backends/hexagon/test/test_op_support.py
PYTHONPATH=src python backends/hexagon/scripts/gen_op_support.py

# 144 rows, 184 verdicts
PYTHONPATH=src python -m pytest \
  backends/hexagon/test/test_overload_census.py \
  backends/hexagon/test/test_overload_census2.py

# op types the host never emits: names in htp_command.h with no case in
# execute_command.cc's dispatcher and no reference outside third-party/
git grep -n 'DSP_OP_' -- backends/hexagon ':!backends/hexagon/third-party'
```
