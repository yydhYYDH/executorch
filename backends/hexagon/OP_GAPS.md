# Hexagon backend op gaps

<!-- Hand-written. `OP_SUPPORT.md` is the generated half: it is rendered from
     `hexagon_ops.EMITTERS` by `scripts/gen_op_support.py`, and
     `test/test_op_support.py` fails when the two drift. This file covers what
     that table cannot: kernels the vendored library has and nothing emits,
     operators that would need a kernel to exist before an emitter is a
     question, and what either costs a real model. Every claim below names the
     file it was read from, so a reader can tell a measurement from a reading. -->

Computed against `53369ff` on 2026-09-24 and re-measured as the parallel branches
landed -- `topk`'s values half before `b7183d9`, then the element-wise select, the
quantized prefill entry, the zero-filling pad, the split, the transposed
convolution and the three norms, and then the second batch: the run-time convolution
extents, the batch-norm fold, the mask an SDPA node carries, the permutation
gate, the integer nearest resize, and the prefill ceiling the device work measured
(`K <= 12672` under the dispatcher's `M <= 32` split). The library inventory below is
the vendored tree, which none of them changed. Section 2 counts two commands more
than `b7183d9` did, because two kernels have left its table: `DSP_OP_SELECT`(26)
and `DSP_OP_MATMUL_Q4A16_FP16`(22). Section 1 lost a row for a different reason:
`_log_softmax` gained no kernel, but it is a composition of commands that already
exist, which is also why §3's last entry is a composition rather than a kernel.

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
`test_overload_census.py` + `test_overload_census2.py` pin 126 census rows whose
187 op verdicts are 112 wired, 37 refused and 38 unwired. That count is one
verdict per op listed in the row tables those two files are parametrized over --
the rule the appendix writes out, so the numbers can be re-derived rather than
believed; the same two files collect 149 tests, because 23 of them assert inside
a body instead of once per row. The row figure is the one that rule yields --
103 + 17 + 6, the two `_ROWS` tables and `_QUANTIZED_ROWS` -- and the sentence it
replaces read 127, which is three more rows than those tables hold at any revision
measured: 122 at `7dc3368` and 124 at `b7183d9`, the two counting the same way.

## 1. The vendored library has no kernel at all

These are not emitter problems. The two subtype tables and the op enum do not
contain the operator, so an emitter would have nothing to call.

| op | why there is no kernel |
|---|---|
| `aten.min.dim` reading `.indices` | The values are the minimum reduction, but a reader of the indices puts the node out of reach on the values side too: the reduction kernel computes values, not positions. Same all-readers-are-getitem-0 rule as `aten.max.dim` and `max_pool2d`. The value-only form is covered -- see `OP_SUPPORT.md` |
| `aten.argmax.default`, `aten.argmin.default` | Every reduction here walks values; `argmax`'s output is positions. `topk` is the one kernel that writes both, and its positions are refused on purpose -- §3 |
| `aten.prod.default`, `aten.var.correction` | A running product and a second moment are each a different walk from the four reductions that exist (sum, maximum, mean, minimum -- minimum arrived with merge 13). `aten.cumsum.default` moved off this row: a prefix scan is not a reduction the kernel has but a product the kernel does have, against a host mask, so it is wired through `et_hexagon.cumsum.default` |
| `aten.erf.default` | No `HtpOpsUnaryOpType` entry: the enum is 1..17 (`unary_ops.cc:14-31`) |
| `aten.leaky_relu.default`, `aten.elu.default` | The binary table has no slope form (`eltwise_ops.cc:27-38`) and the unary table no `elu` |
| `aten.convolution.default` with a group count above 1, not the depthwise form, and a height that is a run-time length | A group count needs no channel mapping on the kernel at all once the host partitions it, so the rest of this family is a supported row: a plain grouped convolution is one dense im2col command per group (`_emit_grouped_convolution` in `hexagon_ops.py`; 48 commands at `groups == 8` over 32 input channels to 64 outputs, `test_grouped_conv.py`), a grouped transposed one is the same partition behind the zero-insert (24 commands at `groups == 4`), and a dilated transposed window keeps its dilation on the kernel, since the im2col walk consumes the fields directly (`conv_spec`, 4 commands at `dilation == 2`). A run-time height is affine in the length for a single walk, which is why a stride-1 plain convolution and the depthwise form both still delegate with one (`test_conv_dynamic_height.py`). It is not affine per group: each group's plane extents and its own slice offsets reach a different command, so the patch would need a record per command rather than the one record the run-time length gets, and the case stays on the portable kernels rather than being emitted with the export's example extent baked in |
| `aten.conv3d.default` with a genuinely three-dimensional window, `aten.conv_transpose3d.input` | `Im2ColParameter` is `padX`, `padY`, `dilateX`, `dilateY`, `strideX`, `strideY`, `kernelX`, `kernelY`, `iw`, `ih`, `ow`, `oh` -- no depth axis exists (`ops.h:45`), so nothing here walks a volume. 3-D is a different kernel rather than another parameter set, and the two-dimensional identity above cannot be stretched to it. A rank-5 input does reach the two-dimensional kernel, but only through the rank-5 fold `conv_spec` already has (`hexagon_ops.py:4327-4370`): the window must be 1x1xK with stride 1, padding 0 and dilation 1 on the dropped axis, that axis must be 1 in input and output, and the other kept extent must be 1 as well -- a 1x1x3 window over `[1, C, 1, 1, W]` delegates as `[24, 3, 12, 3]`, while a 1xKx1 window (`Conv3d(3, 5, (3, 1, 1))` on `[1, 3, 5, 1, 9]`) is refused. `aten.conv_transpose3d.input` is refused for every geometry: the transposed branch returns before the fold. `test_conv_3d.py` pins the refusals and `test_decompose_conv3d.py` the opt-in 1x1xK depthwise rewrite |
| `aten.upsample_bilinear2d.vec` | No sampling command: `DSPOpType` (`htp_command.h:32`) has no upsample case and no kernel source mentions interpolation, and bilinear is arithmetic rather than an index map -- its taps alternate with the output row's parity, so it is neither a shift-invariant filter nor the constant-kernel transposed convolution that would let the convolution walk carry it. Nearest was in this row and is not any more: at an exact integer ratio its replication is a set of `s * s` destination-phase regions, each an affine map that reads the whole plane at the same strides, so `aten.upsample_nearest2d.vec` is a supported row now (`upsample_regions` in `hexagon_ops.py`). The ratio that is not an integer multiple is a refusal rather than a gap -- the runs of repeated source elements are then of unequal length, so the phases stop being a constant stride apart -- and `test_upsample.py` pins both sides. `nearest-exact` and `upsample_bicubic2d` are not this row at all: export decomposes them into arithmetic, and the nearest-exact spelling reaches the portable kernels on two `aten.arange.start_step` nodes that have no emitter (measured on `interpolate(..., mode="nearest-exact")`) |
| `aten.repeat.default` with two or more axes above one | One repeated axis is `cat([x] * factor, dim=axis)`: two loops, so one region, with the factor on a level's extent rather than on a region per phase -- which is why a factor of 64 costs the same single region a factor of two does, and why nothing here needs a region count past the one a serialised command already holds. Two repeated axes are four loops against a region's three levels. The host composes them from successive single-axis repeats exactly, so the shape is a refusal rather than a gap, and a refusal rather than a half-applied axis. A zero-filling `constant_pad_nd` was listed here and is not of this kind: it is a `DSP_OP_ZERO` memset plus one region, both already emitted for other ops, so it is a supported row now (`constant_pad_region` in `hexagon_ops.py`). A pad on a third axis from the end, a nonzero `value` and a negative pad still stay on the portable kernels -- the region is three levels and `htp_ops_zero` writes zero and nothing else |
| `aten.pow.Tensor_Tensor` with a base that is not a static value | No pow kernel: the supported exponents lower to the existing unary or binary elementwise commands, and the gate that decides which (`pow_tensor_tensor_is_emittable`) asks for a base the export can read as a value, so a base computed at run time is refused. With a static base and a uniform integral exponent in `{-1, 0, 1, 2, 3, 4}` the node is a supported row; anything outside that set, a non-integral or non-uniform exponent, and a non-finite base stay portable. `test_pow_tensor_tensor.py` covers both sides at the command stream |
| `aten.clamp.Tensor` | The clamp entry point carries its bounds as two fp16 params, so an operand bound has nowhere to go |
| `aten.full`, `aten.full_like`, `aten.arange`, `aten.scalar_tensor` | Not kernels: nothing emits a tensor that was not read from memory |

`aten._log_softmax.default` was a row in this table and is not any more, and
neither `aten.native_group_norm.default` nor
`aten._native_batch_norm_legit.no_stats` -- the batch-of-one view InstanceNorm
exports -- belongs in it. The enum has no kernel for any of the three, but "no
kernel" and "nothing an emitter can call" are different claims: each of them is a
composition of commands that do exist, so an emitter *can* call something. §3's
last entry is what that composition is and why it is not the obvious one.

## 2. Kernels present, nothing on the host emits them

`htp_command.h:32-81` declares 48 op types. 43 have a `case` in
`execute_command.cc`, and the host side emits 22 of them. The other 20 are
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
| 17 | `DSP_OP_CONV1X1_DIRECT_FP16` | `htp_ops_conv1x1_direct_fp16` | wired, and it adds no kernel: the name is a second entry for `hmx_im2col_convolution_fp16` (`im2col_convolution_fp16.cc:1840`), and the emitter's `conv_1x1_direct_applies` picks 17 only for the geometry whose 1x1 activation fill is the plane copy or the strided gather, falling back to 12 for a padded, dilated, batched, ragged, depthwise, transposed or non-1x1 window. §6.1 |
| 25 | `DSP_OP_CAST` | `htp_ops_cast` | deliberately unused, as 7 |
| 30 | `DSP_OP_RELU6` | `htp_ops_relu6` | deliberately unused: `relu6` is `hardtanh(0, 6)` and goes through `UNARY`/`CLAMP`, the entry point that restores a NaN by a bit test |
| 31 | `DSP_OP_MASKED_REDUCTION` | `htp_ops_masked_reduction` | not an emitter: the kernel wants a separate fp16 `[O][R]` predicate operand (`eltwise_ops.cc:2959`) that no ATen node carries, and the graphs that come closest already answer correctly as two commands. A fusion target, not a missing line -- §7 |
| 32 | `DSP_OP_TMAC_A16W1` | `htp_ops_tmac_a16w1_fp16` | 1-bit weights |
| 33 | `DSP_OP_FLASH_ATTENTION_BLOCK` | `htp_ops_flash_attention_block` | a blockwise attention |
| 34 | `DSP_OP_MATMUL_Q4A16_BLOCK_FP16` | `htp_ops_matmul_q4block_a16_fp16` | quantized prefill, block-scaled form |
| 36 | `DSP_OP_LSTM` | `htp_ops_lstm` | an LSTM cell. The recurrent models are unrolled before the partitioner, so nothing reaches it |
| 37 | `DSP_OP_RELU` | `htp_ops_relu` | deliberately unused, as 30 |
| 39 | `DSP_OP_PRELU` | `htp_ops_prelu` | `prelu`, which decomposes into `view_copy` + `gt` + `mul` + `select` before the partitioner sees it. Two of those four now delegate and the kernel is not reached, because nothing builds the prelu node back -- §3 |
| 40 | `DSP_OP_CONV1X1_DIRECT_W8A16_SYM_PER_CHANNEL` | `hmx_conv1x1_direct_w8a16_sym_per_channel` | left unwired on purpose, and still unwired after the int8 prefill landed. The entry forwards to `hmx_matmul_w8a16_block_fp16` and reads the int8 HMX tile order `reorderInt8SymWeightForHmx` produces, which is not in this tree. The forward reference this row used to carry -- that the packer and the command 42 wiring live on `hexagon-int8prefill` -- has been satisfied: command 42 and the int8 prefill packer are in the tree now, wired by `_emit_w8a16_prefill` for `bits == 8`. What is still missing is this row's own command, the quantized 1x1 convolution, which is a different entry and has no emitter: the name appears only in the vendored `htp_command.h` and `execute_command.cc` and in this table -- §6.1 |
| 42 | `DSP_OP_MATMUL_W8A16_BLOCK_FP16` | `hmx_matmul_w8a16_block_fp16` | quantized prefill, int8 weights, **wired**. Alongside the int4 prefill entry (22), which the block-scaled int4 form extends, the int8 entry emits command 42 for `M > 1` with a tiled int8 weight and an fp16 per-channel scale tail. The ceiling beside it is the one in §4: `K % 64 == 0`, `N % 32 == 0`, and `K <= 12672` below the dispatcher's `M <= 32` split |
| 44 | `DSP_OP_VISION_FLASH_ATTENTION_FP16` | `htp_ops_vision_flash_attention_fp16` | a vision attention variant |

Of the 48, 43 have a dispatcher case: the 22 the host emits, the 20 above, and
`GET_INFO`(20), which the host never emits either. The 5 without one are
`RESERVED_0`(0), `RESERVED_21`(21), `COMMAND_GROUP`(99) and `MAX`(100) --
protocol, not gaps -- and `DSP_OP_POST_ATTN_REDUCE_FUSE`(35), which §6 is about.

Three rows have left this table, one per merge: `DSP_OP_TOPKV2_K1_FP16`(27) when
`topk`'s values half landed, `DSP_OP_MATMUL_Q4A16_FP16`(22) with the int4
quantized prefill entry, and `DSP_OP_SELECT`(26) with `aten.where.self`. The
counts above read 21 and 21 where `b7183d9` read 19 and 23: the same table with
two fewer rows and the command set with two more. Each is worth reading in §3, and
the two that carry a qualification are there: the prefill entry reaches
neither the int4 nor the int8 weight above its measured ceiling, and `where` is
placed while the comparison that produces its condition is not.

## 3. One emitter away, inside a family that is already wired

These are the cheapest real gaps: the kernel is there, the command is there, and
the only thing missing looks like the line that produces it. What each entry
records is what that turned out to mean once it was checked. Three are decided
rather than open -- `sin` and `cos` are wired, `where` is wired, and `topk`'s
values half is wired -- and they are kept because what each decided is the shape
the next op in its family will have to decide too. Two were looked at and not
taken: `expm1`, on a measurement, and `prelu`, which turns out not to be one
emitter away after all. One is still open and needs a kernel rather than an
emitter: a one-byte bool output for the comparisons -- but that is the last piece of
its family and not what is holding the nodes, which is what the entry below now
says.

- **`aten.sin` and `aten.cos`**: subtypes `SIN`(14) and `COS`(13) exist, and
  `UNARY_OP_TYPES` carried both before anything emitted them. This was the entry
  that said the hesitation was the arithmetic rather than the plumbing: these run
  the DSP's own approximations, and agreement with the simulator is not evidence
  about them, since both sides run the same approximation. The error was measured
  against the mathematical value instead, and both are wired at it: `sin` worst
  3.9e-4 absolute inside four pi and 2.2e-3 past a thousand radians, `cos` worst
  8.9e-4 at `abs(x) == pi/2` and 1.2e-3 past a radians. The band that decided the
  acceptance is the one this backend already delegates: `test_unary_sim.py`'s
  module docstring records `tanh` as wired at a 2.1e-2 relative error and `log` as
  wired at 1.7e1 near its own zero, so neither `sin`'s nor `cos`'s number is
  disqualifying, and the generated table carries both rows with their error
  written into the row.
- **`aten.expm1`**: measured and **not** wired, and it is the counterpart to the
  two above rather than an omission. The kernel's HVX walk computes exp2 and
  subtracts 1 in fp16, so for small arguments the result carries no correct
  digits: the measured relative error for `abs(x)` below about 1e-3 is 1.000,
  which is the whole interval the op exists for. The same values in a 21-element
  array fall through to the kernel's fp32 `expf(x) - 1` and come back exact, so
  which of the two an array takes depends only on its length and no emitter can
  decide it at export. A measurement, not a preference: rewiring it would move a model's
  `expm1(x)` from torch's answer to a constant.
- **`aten.gt`, `aten.ge`, `aten.lt`, `aten.le`, `aten.eq`, `aten.ne`**: open, and
  not for the reason this entry used to give. The one-byte mode is real and still
  missing -- `htp_ops_binary_elementwise` returns -1 for a `bytes` that is not 2 or 4, and
  the compare arms write fp16 1.0/0.0, fp32 1.0/0.0 or int32 1/0 -- but it is the
  *last* missing piece, not the first. Two others are ahead of it. One is that the
  DSP has two comparison op types and not six: `HtpOpsBinaryOpType` carries `GREATER`(9) and `LESS`(10),
  `htp_ops_binary_is_compare` admits nothing else, and no host emitter ever passes either,
  so eq, ne, ge and le have no kernel at any width. The other is that the six
  targets are in neither `EMITTERS` nor `SUPPORTED_TARGETS`, which is where the
  partitioner actually refuses them -- the first gate in the predicate, ahead of
  the fp16/fp32 dtype rule and of `operand_dtypes_are_readable`, which is about bool
  *operands* and returns True on a fp16-operand comparison whose result is bool.
  `where`'s arrival did not need the one-byte mode because a comparison's result is a
  `where` input rather than a command's output. `SQUARED_DIFFERENCE`(11) has no ATen node at all.
- **`aten.where.self`**: **done.** This was a pure emitter gap plus one
  partitioner predicate rather than a kernel change, and the entry used to ask the
  question the wrong way round -- whether the kernel's condition operand accepts
  what a comparison writes. It does: `htp_ops_select` carries a `condBytes` and
  `htp_ops_select_cond_at` reads flagwise when it is one
  (`eltwise_ops.cc:2116-2125`). What had to be settled was the runtime's side of
  it, whether a `torch.bool` reaches the arena one byte per element, and it does,
  because an input's slot is the size the blob declares for it; that was measured
  on hexagon-sim, with a control that declares the condition two bytes wide and
  requires the answer to move. The qualification is the half that stayed: no
  kernel here *writes* a bool, so the comparison that produces a condition is
  still on the portable kernels and its result enters the delegate as an input or
  as a bool constant weight. The condition's extent is also bounded by the
  emitter: it must be the output's element count or a single element, so a
  broadcast shape that is neither still falls back whole.
- **`aten.prelu`**: `DSP_OP_PRELU`(39) exists and no emitter uses it, because the
  node does not survive export at all -- `F.prelu` arrives as `view_copy` (the
  weight re-pointed, not read) + `gt` + `mul` + `where`
  (`test_overload_census.py`, row "prelu decomposes, and the where stays
  put"). Reaching the kernel would take a fusion pass of the `mul_silu.py` kind,
  not an emitter, and **two things are missing, not one**: the pass builds the
  node, and the emitter that consumes it needs `plane` / `channel` / `pack` /
  `batch` (`execute_command.cc:693-699`) -- the channel geometry that
  `where_is_emittable` deliberately does not compute and that exists nowhere else
  to borrow. So this is a further step out than the `sin`/`cos` entries above,
  which really were one emitter each. The decomposition itself is exact and two
  of its four nodes are now in the delegate: with `where` wired, `F.prelu` with a
  one-element weight lowers to a single delegate whose two commands are the
  multiply and the select, and leaves `gt.Scalar` on the host. `DSP_OP_PRELU` is
  still the only route to the fused form, and the kernel's own contract is
  narrower than the decomposition it would replace -- it takes an fp16,
  per-channel slope and nothing else -- so the cheap subset, `slopeCount == 1`,
  is the one to start with if anyone takes it.
- **`aten.topk.default`**: this was the fourth, and it is done. The kernel
  answers both outputs -- `htp_ops_topkv2_k1_fp16(values, indices, input,
  rowSize, rows)` -- and what it took was the emitter plus the
  all-readers-are-getitem-0 rule (§4), not a kernel. `k == 1` over the last axis
  now emits one command whose params are the row length and the row count; the
  values are bit-identical to torch on the host, in `blob_interpreter.py` and on
  hexagon-sim. The positions are refused, and the reason is measured rather than
  assumed: this kernel writes the **first** occurrence of the row maximum while
  torch's own kernel writes whichever index its partial sort stops on, which
  over 200 rows of quantized values is neither the first nor the last occurrence
  175 times. Handing the graph a position torch never produced is the one thing
  a refusal is unambiguously for, so `topk_is_emittable` keeps the whole node on
  the portable kernels whenever anything reads it. (The width would refuse it
  anyway: the node declares int64 and the kernel writes one int32 a row.)
- **The three norms**: `aten.native_group_norm.default`,
  `aten._native_batch_norm_legit.no_stats` and `aten._log_softmax.default` all
  landed the same way `topk` did not: no kernel was added and nothing under
  `third-party/` changed. Each is a view of an op that is already wired. A group
  norm is one mean and one variance per group over that group's channels and the
  whole spatial block, which is one row per (batch, group) of the input's own
  `[N*G][(C/G)*H*W]` view -- the row `DSP_OP_LAYER_NORM` already reduces -- plus
  the two elementwise commands a per-channel affine needs, because the kernel
  takes no weight. InstanceNorm exports as the no-stats batch norm over a
  `[1, N*C, spatial]` view for the same reason, one row per (batch, channel),
  and the batch-of-one gate that keeps a wider batch portable is in §4.
  `log_softmax` is six commands rather than the two the name suggests: the row's
  maximum, the shift, the exponentials, their sum, the log of it, and the
  subtraction that removes the shift. `log(softmax(x))` is the composition that
  does not work -- the softmax command stores its small probabilities as fp16
  zeroes, and the log of a zero is the kernel's -65504, which
  `test/test_log_softmax.py` measures against torch's finite answer on a row
  that saturates. The shift also bounds what the sum holds at the row length,
  which is why a span past 65504 elements is refused rather than saturated.
- **`aten.glu`**: it is in neither the support table nor the refusal table because it
  needs neither: `F.glu` does not survive export, and what the partitioner sees is
  two `slice_copy` nodes, a `sigmoid` and a `mul` -- all three already wired. A
  GLU is therefore **supported by composition**, in five DSP commands: a blit for
  each of the two halves the channel axis is split at, the sigmoid over the gated
  half, the elementwise multiply, and the blit that writes the answer out. The
  Conformer convolution module is the geometry that needs it, and the whole block
  -- LayerNorm, two pointwise convolutions each followed by a GLU, a depthwise
  convolution, LayerNorm, residual -- lowers to one delegate with both GLUs
  inside it (`test/test_glu.py`). The numerics the model sees are the sigmoid's
  PWL band scaled by the values in the half the multiply is fed: the sigmoid's
  own band is 2.4e-3 and the product's is 4.0e-3 at one seed and 6.8e-3 across
  eight, on a `[1, 64, 1, 48]` input with a normal tail. That is a property of the
  sigmoid, not of the GLU, and it is why no emitter of its own is wanted: a fused
  kernel would have to match the PWL rather than improve on it.

## 4. Shape and parameter limits of the ops that are supported

These are `refused` rather than `unwired`. There is an emitter, and the
predicate keeps the node off the DSP, so the fallback works. They are listed by
*limit* rather than by op because that is the form in which they cost a model:
the same node delegates or does not depending on a shape.

| limit | ops it holds back | what it would take |
|---|---|---|
| The arena holds two bytes per element | every op: fp16 only. int64 casts, `select_copy` on int64 (85 of them per Qwen3 forward pass, all sequence-position reads), and any `torch.bool` operand | a wider dtype on the DSP, or keeping these on the host, which is where they belong |
| Binary broadcast has an 8-D representation | every binary op at rank 9 | a wider broadcast table; the 25-entry tail plus the 9-int command head uses 33 or 34 of the 40-int budget, so rank 8 fits but rank 9 has no representation |
| One contiguous `[outside][reduce][inside]` span per reduction | `mean.dim` and `sum.dim_IntList` with non-adjacent reduced dims, `amax` the same way | a strided reduction, which is the same kernel the softmax row below needs |
| Softmax takes only a contiguous middle axis below channel 64, plus the last axis | `_softmax` over a non-last axis with a non-contiguous source, a permutation that needs more than three advancing region groups, or a channel at or above 64 | a wider safe middle-axis command or a more expressive permutation region; the old strided reduction is not used |
| Pooling needs static shapes | `max_pool2d`, `avg_pool2d` with a symbolic extent | the blit regions are derived from the shape, and a symbolic one is refused rather than emitted with the example's stride |
| Convolution needs static extents and a constant weight | `convolution` | a run-time weight has no command; a grouped convolution over a run-time height has no per-group dynamic patch |
| The gather table is a constant, tiled at export, indexed by int32, along axis 0 | `embedding` (int64 indices are refused by dtype), `index_select` (dim must be 0), `index.Tensor` (one index only) | a run-time table, or an index the command can describe. A `tokens.to(torch.int32)` in the model is enough for the common case |
| The weight-only quantized matmul needs `K % 64 == 0` and `N % 32 == 0`, and then either a single activation row (the GEMV entries) or an int4/int8 weight (the prefill entry, command 22 and command 42); the prefill entry also needs `K <= 12672` while `M <= 32` (`PREFILL_M32_MAX_M` / `PREFILL_M32_MAX_K` in `hexagon_ops.py`). The small-M DSP source heap-allocates the per-tile descriptors (`matmul_q4fp16_mle32.c:443-447`, freed at `:691`); the historical VLA put the same count on the stack | a batch axis, a dynamic (`SymInt`) `M`, and an int4 or int8 prefill with `M <= 32` at `K > 12672` -- before the heap fix `K = 12736` at `M = 4` aborted the DSP process (`0x8000040d`, no output, under a second) while `K = 12672` answered within 3.9e-4 to 6.4e-4. A later measurement-only A/B on the same phone, with the refusal bypassed in memory, had the heap skel answer `K = 25216` for `M = 2, 4, 32`; the stale VLA skel still aborts | general prefill support for those additional geometries. The ceiling is a refusal past a measured bound rather than a claim about what the kernel can compute, and it is kept because deployment does not encode source provenance, so a stale skel can still carry the VLA. That evidence is one phone, one skel build and one session with a monkeypatched predicate, which is not a basis for widening the bound and not a basis for calling it necessary either |
| `where`'s three operands are each the result's element count or a single element | a `where` whose condition broadcasts, e.g. `where(cond[2,1,4], a[2,3,4], b)` | the kernel's per-channel value mode, which needs a `plane` / `channel` / `pack` / `batch` this emitter does not compute -- so the same `where` delegates in one model and not in the next |
| A constant pad fills its border with a `ZERO` memset, so the fill is zero and nothing else, and it needs a three-level region | a nonzero `value`; a pad on a third axis from the end (a fourth level); a negative pad (that is a slice); an all-zero pad, whose region is the operand at its own strides and which the kernel drops as a self-write | a fill other than zero is a kernel that writes a value; anything past the last two axes is a different region form. `mode='reflect'`/`'replicate'`/`'circular'` are not this node at all: torch lowers them to `arange`/`abs`/`clamp`/`index` programs, so no pad node reaches the partitioner |
| A command holds three 12-int regions | `permute_copy` (at most 3 groups that advance, no reversal inside one) | a command form with more regions. The permute count is of the groups that spend a loop, not of the axis groups: a batch of one in front of a head split is a fourth group that iterates once and advances nothing, so `[1, tokens, heads, dim] -> [1, heads, tokens, dim]` is three loops and delegates, while the same split on a batch of two is refused -- the difference is an extent, and `permute_region` in `hexagon_ops.py` is where it is decided. `cat` used to sit here too, at most 3 operands, and is not of this kind any more: it splits a longer list over as many blits as the budget needs, so four inputs is two commands and seven is three, every input writing its own disjoint slice |
| A blit describes one run per row | `slice_copy` with `step != 1` | a strided run, or a different command |
| A two-output node is placed only when every reader takes the values output | `max.dim` (indices), `max_pool2d_with_indices`, `topk` (indices), `native_layer_norm` (`getitem 0`), `add_rms_norm` (`getitem 0` or 1) | not a rule waiting to be relaxed: `max.dim` and the pool have kernels that compute values rather than positions, and `topk` is the one kernel here that answers both -- and its positions are refused anyway, because the position it writes is the first occurrence of the row maximum while torch writes whichever index its partial sort stops on, which over 200 rows of quantized values is neither the first nor the last 175 times |
| The narrow view rule: a view that would be a partition boundary is copied instead | `view_copy`, `unsqueeze_copy`, `squeeze_copy`, `expand_copy` (the form that grows nothing), `alias_copy` | nothing -- this is the rule working, and its symptom is a delegate one op shorter than it looks |
| A broadcasting `expand_copy` is a blit whose region holds the source still on the axes that repeat, so each grown axis must be an exact multiple and the shape's plain/broadcast pattern must make at most three runs | a contiguous operand, and one run of plain axes around or between the runs that repeat -- the RoPE half-rotary `(1,8,1,32,128) -> (1,8,2,32,128)` is exactly three runs and emits one `DSP_OP_RASTER_BLIT` with source stride `[4096,0,1]` | a pattern that alternates more than twice, e.g. `(1,2,1,2) -> (3,2,5,2)`, which is four runs | a region with more than three levels, or a gather. The vendored `htp_ops_raster_blit` already indexes each level with that level's source stride, so a zero is a re-read and not a special case, and `htp_ops_try_broadcast_inner_fp16_blit` is already a fast path for the innermost-broadcast shape |

The convolution and gather rows above are admission geometry rather than
missing support: each op has an emitter, and a model that produces the geometry
delegates. What each gate takes, at the line that decides it (`hexagon_ops.py` and
`hexagon_partitioner.py`; line numbers move with the file, the function names are
the stable part):

| gate | accepts | measured refusals | line |
|---|---|---|---|
| `pool_spec`: channel count | any `C`, and for a 3-D `(C, H, W)` input it is `shape[0]` that is read | nothing. The DSP blocking is a count of 64-lane blocks, `c4` carries `ceil(C/64)` and the kernel walks every one of them, so a width is `ceil(C/64)` blit regions a side and never a new parameter. A ragged last block is a narrower region plus a `ZERO`; the lanes it pads pool independently and the unpack never reads them. Measured on a phone: C in {64, 96, 128, 192, 256} at batch {1, 2, 3} bit-exact on maxima | `hexagon_ops.py`, `pool_spec` |
| `pool_spec`: window and counting | any static kernel and stride, `pad >= 0`, the output size floor division gives, every window inside its row | `dilation != (1, 1)`, `ceil_mode` (windows past the input, which the one-block geometry does not describe), `divisor_override` | `:2125`, `:2136`, `:2142` |
| `conv_spec`: group count | `groups == 1`, the im2col path, at any channel count (3->16 and 64->64 both delegate, with a leading `ZERO` command when `C_in % 64 != 0`); `groups == in_channels == out_channels` with one channel per group, the depthwise kernel; and any count in between as one dense im2col command per group, measured on the host, the simulator and the phone for 2, 3, 4, 8, 16 and 32 groups and for 63, 64 and 65 channels per group | a group count that does not divide the input channels, and a grouped convolution whose height is a run-time length, because the per-group partition emits static extents and the per-group dynamic record is not produced | `hexagon_ops.py` `_emit_grouped_convolution`, `test_grouped_conv.py` |
| `max_pool2d_with_indices`: readers | every reader takes `getitem 0` | reading the indices leaves the pool on the portable kernels, whatever its shape | `hexagon_partitioner.py:607` |

The gather's index width was a gate like that and is not any more. `embedding`,
`index_select` and `index.Tensor` take an int32 index tensor or an int64 one, and
an int64 tensor is narrowed into the int32 slot the kernel reads on the way in,
value by value and refused against the vocabulary rather than truncated
(`narrow_indices_to_int32` in `runtime/hexagon_backend.cpp`, beside the
fp32-to-fp16 narrowing the runtime already did for the same reason). The width
the caller handed over is `params[7]` of the command, and what a negative index
means is `params[8]`, because torch's own answer is not uniform: `embedding` and
`index_select` raise on a negative index and advanced indexing counts back from
the last row. A table past an int32 is still refused at export, which is a
different check and a different reason: the vocabulary is a command param and
every element offset in the kernel is derived from it. A `tokens.to(torch.int32)`
in the model is no longer needed, and the portable `_to_copy` would not have run
it anyway -- its type switch names no integer type
(`kernels/portable/cpu/op_to_copy.cpp`), so the documented workaround was a graph
the runtime could not execute.

## 5. What the gaps cost a model

Qwen3-0.6B, from the README's Status section: 1967 nodes reach 29 delegates (one
per layer -- a count that predates the mask refusal in `_sdpa_fits_dsp_limits`,
so an export whose attention nodes carry a mask delegates fewer than 28) and 825
stay on the portable kernels. 711 of those 825 are shape
guards (`_assert_scalar` 227, `getitem` 114, `le` 113, `_local_scalar_dense` 85,
`lt` 57, `ge` 57, `add` 57, `sym_size` 1), which is what a dynamic-shape export
looks like rather than a gap. The real remainder is small:

- **one `view_copy` per layer**, which is the boundary rule above.
- **85 int64 `select_copy`**, the sequence-position reads, which belong on the
  host because the patch mechanism needs the value where it is.

The two gaps that are not of that shape:

- **Quantized prefill.** This was the only functional gap -- the weight-only path
  was `M == 1` only, so a quantized model could decode and not prefill -- and it
  is now closed for both weight widths: an int4 weight above one row reaches
  `DSP_OP_MATMUL_Q4A16_FP16`(22) and an int8 weight
  `DSP_OP_MATMUL_W8A16_BLOCK_FP16`(42), which is the second row that left §2,
  and both go through `_quantized_prefill_fits`, whose first clause admits 4 and
  8. What stays beside it is a refusal past a measured bound, not a gap:
  `K <= 12672` below the dispatcher's `M <= 32` split (§4). Prefill's own
  evidence is host lowering plus hexagon-sim, at every layer below the device:
  no timing, and no device.
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
  silent fix. The decision, after a grep for `WEIGHT_REORDER` and
  `weight_reorder` over `backends/hexagon` with the vendored tree and the sim
  runners excluded, which returns this file and `README.md` alone: stale, and
  narrowly so. The README's own init list
  (`README.md:44-53`) names compile-spec, the resident block, the weight copy and
  the group array, with no reorder step in it, and the reorder that does happen is
  host-side at export (`hexagon_ops.py:2292`, `pack_q4a16_prefill`), which is what
  `README.md:479` already says about the int4 tile order. So the line at 40
  describes a runtime init step this backend never performs, while line 479's
  claim about the int4 order is true with the location implied by 40 being wrong.
  The sentence is left as written, deliberately: this file records the verdict.
- **`DSP_OP_POST_ATTN_REDUCE_FUSE`(35) is declared and has no `case`** in
  `execute_command.cc`. Which of the two is missing has not been established.
- **The q4a16 and w8a16 packers have not been checked against a DSP here.** The
  upstream host reorders are not in the vendored tree (`src/host/` was dropped at
  vendoring), so there is no host-side authority for the order to be compared
  against. The q4a16 **prefill** tile order is no longer agreed with by reading
  alone: `test_prefill_on_sim.py::test_the_packers_bytes_are_the_vendored_reorders_bytes`
  runs the vendored int4 reorder (`htp_ops_weight_reorder_int4`) on hexagon-sim
  and requires the packer's bytes to equal its output byte for byte, with the
  GEMV packer as the negative control. That is the simulator and not silicon.
  The int8 tile order the w8a16 prefill kernel reads was checked by nothing until
  the prefill entry was wired: an emitter writes it now, and
  `test_hexagon_quantizer.py::test_a_w8a16_prefill_matmul_lowers_to_command_42` (M = 8, K = 64, N = 128)
  pins the 29-parameter command 42 body, the tiled weight and the fp16 per-channel
  scale tail, with the int4 geometry as the control that must still emit 22.
  Those are host-lowering facts. The int8 prefill *kernel* is separately reported
  device-tested at `M = 4, K = 12800` on a OnePlus 13 by the int8prefill
  workstream (d6b2cf5, whose comment records the corrected heap-backed skel
  clearing the measured K boundary); that result is the workstream's report and
  was not reproduced during this integration.
- **`OP_SUPPORT.md` had no row for `aten.prelu`.** It has one now, and the
  discrepancy is worth keeping for what changed: the row used to be absent because
  the node does not survive export, so the generated table had nothing to render,
  and a reader looking for prelu found nothing although the census pinned its
  decomposition and the library carries a kernel for it. The row now names the
  kernel, says the decomposition's `mul` and `where` delegate as a multiply and a
  select, and says that fusing the three nodes back into one command is a pass of
  the `mul_silu.py` kind rather than another emitter. The `where` clause the old
  bullet carried is out of date too: `where` is no longer refused with a reason
  written down elsewhere, it is placed, and §3 says what stayed behind.
- **The weight-only quantizer annotated `mm` and `addmm` but not the 2-D
  `aten.matmul`**, so a model written `a @ b` never reached the quantized kernels
  and both schemes produced identical output. Fixed in `53369ff`; recorded here
  because the same shape of gap -- a spelling the emitter table does not know --
  is the one `unwired_overload_census()` now counts.

### 6.1 The 1x1 command, and what wiring it was worth

- **`DSP_OP_CONV1X1_DIRECT_FP16`(17) is now emitted, and it is the same kernel.**
  `htp_ops_conv1x1_direct_fp16` forwards to `hmx_im2col_convolution_fp16`
  (`im2col_convolution_fp16.cc:1840`), and the C's fill selection is by
  geometry, not by the command id: `fill_im2col_activation_tiles`
  (`:1593`) takes `fill_im2col_activation_1x1_pack64_tiles` whenever
  `packCUnit == 64`, the window is 1x1, both dilations are 1 and
  `kp == ceil(ic/32)`. A 1x1 convolution therefore already ran that fill
  under command 12, which is what row 17 meant by "a 1x1 convolution goes
  through im2col today".
- **What the change buys is the stream saying which fill runs, not less work.**
  The two names are the same function, so the element count, the VTCM
  footprint and the arithmetic are unchanged; measured on the simulator, the
  two entry points return bit-identical output for the same 64-to-96 1x1
  5x5 geometry. Inside the 1x1 fill, the paths command 17 is gated on are the
  vector plane copy (unit stride, `out plane == in plane`) and the strided
  gather (`use_pack64_1x1_strided_fast`, `:548`); a padded or differently
  shaped 1x1 reaches the same fill's per-position walk, which is why the
  emitter's predicate is a strict subset of the C's selection and a refused
  geometry stays on 12 rather than being mislabelled 17.
- **The gate is `conv_1x1_direct_applies`**, a host transcription of that
  selection: no transposed convolution, no depthwise, no upsample, a 1x1
  window, `in_channels` a whole number of 32-channel blocks, unit dilation,
  no padding, batch 1, and either unit stride with `out == in` or a stride
  above 1. It is tested against the blob rather than against a string: a 1x1
  graph counts one 17 and no 12, and the 3x3 graph on the same operands
  counts one 12 and no 17.
- **`DSP_OP_CONV1X1_DIRECT_W8A16_SYM_PER_CHANNEL`(40) stays unwired.** Its
  entry is a one-line forward to `hmx_matmul_w8a16_block_fp16`, so what an
  emitter would have to bring is the int8 HMX tile order, not a convolution.
  `reorderInt8SymWeightForHmx` is not in this tree; the int8 packing and the
  command 42 emitter did come with `hexagon-int8prefill` (`9b9e8bf`, merged as
  `d6b2cf5`), so what an emitter for this row still lacks is the tile reorder
  itself, and a layout guessed from the kernel's comments would produce
  plausible wrong numbers, which is the one failure mode this file exists to
  prevent. It belongs on top of a port of `reorderInt8SymWeightForHmx`.

## 7. Priority

1. **Quantized prefill (`M > 1`) for the int8 weight** -- **done.** This entry
   used to be the only functional gap left: the int4 half had landed and the int8
   half was waiting on a packer for kernel 42's tile order rather than on a
   kernel. Both halves are now wired (commands 22 and 42), so the packer exists
   and its row has left §2. What stays beside it is the shape limit §4 states,
   `K % 64 == 0` and `N % 32 == 0`, which is the granularity the block-scaled form
   (34) exists for -- and the block-scaled int4 prefill now carries its scale
   block count to the kernel, while the int8 prefill refuses block-wise scales
   because its kernel does not read them. §4 also carries the ceiling the wired
   entries hold (`K <= 12672` below the dispatcher's `M <= 32` split): that one is a
   refusal past a measured bound, not a missing packer. It is kept because a stale
   VLA skel can still be deployed even though the current source heap-allocates the
   descriptors, so widening it needs source provenance and a broader measured
   matrix rather than the one A/B that cleared it. The part of this item that
   was never about the host is still open and is not a priority entry: the packers
   have not been compared against a DSP from this tree, which §5 records.
2. **`aten.where` via `DSP_OP_SELECT`** -- **done.** This entry used to say the
   open question was whether the kernel's condition operand accepts what a
   comparison writes; that question was the wrong way round and is now closed. The
   kernel side holds -- `htp_ops_select` takes a `condBytes` and reads a one-byte
   flag per element (`eltwise_ops.cc:2116-2125`) -- and what had to be settled was
   the runtime's side of it, which was measured on hexagon-sim. Note that closing
   this does **not** by itself reach `DSP_OP_PRELU`; see the `aten.prelu` entry in
   §3. The half that is left is the producer rather than the consumer: no kernel
   here writes a one-byte bool, so a comparison stays portable. That is a kernel
   change, but it is not the whole of the producer gap: the comparisons are
   refused before the width is ever a question (see the comparison entry in §3),
   and a bool producer would not by itself reach the `where` either, because
   `htp_ops_select`'s own guard admits a condition that is the whole output or a
   single element and the SDPA guard's condition is one flag per query row.
3. **A masked reduction as a fusion** -- `DSP_OP_MASKED_REDUCTION` is not a node
   that cannot be placed. The shape that comes closest,
   `(a * m.unsqueeze(-1)).sum(-2)`, already lowers to a `mul` and a `sum` that
   answer correctly on the DSP, so this kernel would replace two commands with
   one: worth doing where a profile says the merge pays, and nowhere else. It is
   also the only item here with a soundness precondition to prove, because the
   kernel tests `mask != 0` rather than `mask == 1`: the operand has to be exactly
   0.0/1.0, which a lifted constant can be checked for and a run-time mask
   cannot.
4. **`sin`/`cos`/`expm1`** -- measured and decided, in §3: `sin` and `cos` are
   wired at the error their generated rows carry, and `expm1` was rejected
   because its vector path loses every digit for the arguments the op exists for.
   Nothing is queued on this item.
5. Everything else in §1 would need a kernel that does not exist, and everything
   in §2 is a kernel no graph has asked for yet. Neither list is a defect.

## Appendix: reproducing the numbers

```sh
# the generated half, and the guard that keeps it equal to EMITTERS
PYTHONPATH=src python -m pytest backends/hexagon/test/test_op_support.py
PYTHONPATH=src python backends/hexagon/scripts/gen_op_support.py

# the census counts in the header: one verdict per op listed in the row tables
# these two files are parametrized over (_ROWS, _QUANTIZED_ROWS), not per
# collected test -- 126 rows and 187 verdicts against 149 collected, where the
# 126 is len(_ROWS) in each file plus _QUANTIZED_ROWS: 103 + 17 + 6
PYTHONPATH=src python -m pytest \
  backends/hexagon/test/test_overload_census.py \
  backends/hexagon/test/test_overload_census2.py

# the op-type counts in section 2. The emitted set is the one the interpreter
# test pins -- 21 op types after the select and prefill entries, which is what
# makes the table 21 rows; the 43 are htp_command.h's types that have a case
# in the dispatcher. The grep below is a superset of the emitted set: it counts
# the names in these docs too, which is why it is not the count itself.
PYTHONPATH=src python -m pytest \
  backends/hexagon/test/test_blob_interpreter.py::test_the_ops_actually_emitted_are_the_ones_we_think
git grep -n 'DSP_OP_' -- backends/hexagon ':!backends/hexagon/third-party'
```
