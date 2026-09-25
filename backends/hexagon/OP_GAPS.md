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
187 op verdicts are 111 wired, 37 refused and 39 unwired. That count is one
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
| `aten.amin.default`, `aten.min.default`, `aten.min.dim` | `HtpOpsReductionOpType` is `SUM`(1), `MAXIMUM`(2), `MEAN`(3) -- `eltwise_ops.cc:2456-2459`. There is no minimum to select, and the dispatcher rejects anything else (`eltwise_ops.cc:2441-2445`) |
| `aten.argmax.default`, `aten.argmin.default` | Every reduction here walks values; `argmax`'s output is positions. `topk` is the one kernel that writes both, and its positions are refused on purpose -- §3 |
| `aten.prod.default`, `aten.var.correction`, `aten.cumsum.default` | A running product, a second moment and a prefix scan are each a different walk from the three reductions that exist |
| `aten.erf.default` | No `HtpOpsUnaryOpType` entry: the enum is 1..17 (`unary_ops.cc:14-31`) |
| `aten.leaky_relu.default`, `aten.elu.default` | The binary table has no slope form (`eltwise_ops.cc:27-38`) and the unary table no `elu` |
| `aten.convolution.default with transposed=False` and a group count other than 1, except the depthwise form | A transposed convolution *is* a window walk once its input is interleaved with zeros: `conv_transpose(x, w, s, p, op) == conv2d(zero_insert(x, s, op), flip(w).transpose(ic, oc), d * (k - 1) - p, dilation=d)`, which `conv_spec` emits on the same two kernels. The host now partitions a grouped transposed input and weight into one dense im2col command per group, while the DSP kernel reads dilation directly. A plain grouped convolution still has no channel mapping in either kernel, apart from the existing depthwise walk |
| `aten.conv3d.default`, `aten.conv_transpose3d.input` | `Im2ColParameter` is `padX`, `padY`, `dilateX`, `dilateY`, `strideX`, `strideY`, `kernelX`, `kernelY`, `iw`, `ih`, `ow`, `oh` -- no depth axis exists (`ops.h:45`), so nothing here walks a volume. 3-D is a different kernel rather than another parameter set, and the two-dimensional identity above cannot be stretched to it: EXIR spells both 3-D forms as their own targets, which `CONV_TARGETS` never names |
| `aten.upsample_bilinear2d.vec` | No sampling command: `DSPOpType` (`htp_command.h:32`) has no upsample case and no kernel source mentions interpolation, and bilinear is arithmetic rather than an index map -- its taps alternate with the output row's parity, so it is neither a shift-invariant filter nor the constant-kernel transposed convolution that would let the convolution walk carry it. Nearest was in this row and is not any more: at an exact integer ratio its replication is a set of `s * s` destination-phase regions, each an affine map that reads the whole plane at the same strides, so `aten.upsample_nearest2d.vec` is a supported row now (`upsample_regions` in `hexagon_ops.py`). The ratio that is not an integer multiple is a refusal rather than a gap -- the runs of repeated source elements are then of unequal length, so the phases stop being a constant stride apart -- and `test_upsample.py` pins both sides. `nearest-exact` and `upsample_bicubic2d` are not this row at all: export decomposes them into arithmetic, and the nearest-exact spelling reaches the portable kernels on two `aten.arange.start_step` nodes that have no emitter (measured on `interpolate(..., mode="nearest-exact")`) |
| `aten.repeat.default` with two or more axes above one | One repeated axis is `cat([x] * factor, dim=axis)`: two loops, so one region, with the factor on a level's extent rather than on a region per phase -- which is why a factor of 64 costs the same single region a factor of two does, and why nothing here needs a region count past the one a serialised command already holds. Two repeated axes are four loops against a region's three levels. The host composes them from successive single-axis repeats exactly, so the shape is a refusal rather than a gap, and a refusal rather than a half-applied axis. A zero-filling `constant_pad_nd` was listed here and is not of this kind: it is a `DSP_OP_ZERO` memset plus one region, both already emitted for other ops, so it is a supported row now (`constant_pad_region` in `hexagon_ops.py`). A pad on a third axis from the end, a nonzero `value` and a negative pad still stay on the portable kernels -- the region is three levels and `htp_ops_zero` writes zero and nothing else |
| `aten._adaptive_avg_pool2d.default` | The pool command takes one fixed window and stride; an adaptive output sizes the window per output position |
| `aten.pow.Tensor_Tensor` | No pow kernel, and the unary table's `SQUARE` is exponent 2 only |
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
| 17 | `DSP_OP_CONV1X1_DIRECT_FP16` | `htp_ops_conv1x1_direct_fp16` | the 1x1 fast path. A 1x1 convolution goes through im2col today |
| 25 | `DSP_OP_CAST` | `htp_ops_cast` | deliberately unused, as 7 |
| 30 | `DSP_OP_RELU6` | `htp_ops_relu6` | deliberately unused: `relu6` is `hardtanh(0, 6)` and goes through `UNARY`/`CLAMP`, the entry point that restores a NaN by a bit test |
| 31 | `DSP_OP_MASKED_REDUCTION` | `htp_ops_masked_reduction` | not an emitter: the kernel wants a separate fp16 `[O][R]` predicate operand (`eltwise_ops.cc:2959`) that no ATen node carries, and the graphs that come closest already answer correctly as two commands. A fusion target, not a missing line -- §7 |
| 32 | `DSP_OP_TMAC_A16W1` | `htp_ops_tmac_a16w1_fp16` | 1-bit weights |
| 33 | `DSP_OP_FLASH_ATTENTION_BLOCK` | `htp_ops_flash_attention_block` | a blockwise attention |
| 34 | `DSP_OP_MATMUL_Q4A16_BLOCK_FP16` | `htp_ops_matmul_q4block_a16_fp16` | quantized prefill, block-scaled form |
| 36 | `DSP_OP_LSTM` | `htp_ops_lstm` | an LSTM cell. The recurrent models are unrolled before the partitioner, so nothing reaches it |
| 37 | `DSP_OP_RELU` | `htp_ops_relu` | deliberately unused, as 30 |
| 39 | `DSP_OP_PRELU` | `htp_ops_prelu` | `prelu`, which decomposes into `view_copy` + `gt` + `mul` + `select` before the partitioner sees it. Two of those four now delegate and the kernel is not reached, because nothing builds the prelu node back -- §3 |
| 40 | `DSP_OP_CONV1X1_DIRECT_W8A16_SYM_PER_CHANNEL` | `hmx_conv1x1_direct_w8a16_sym_per_channel` | quantized 1x1 convolution |
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
the two that carry a qualification are there: the prefill entry reaches the int4
weight and not the int8 one, and `where` is placed while the comparison that
produces its condition is not.

## 3. One emitter away, inside a family that is already wired

These are the cheapest real gaps: the kernel is there, the command is there, and
the only thing missing looks like the line that produces it. What each entry
records is what that turned out to mean once it was checked. Three are decided
rather than open -- `sin` and `cos` are wired, `where` is wired, and `topk`'s
values half is wired -- and they are kept because what each decided is the shape
the next op in its family will have to decide too. Two were looked at and not
taken: `expm1`, on a measurement, and `prelu`, which turns out not to be one
emitter away after all. One is still open and needs a kernel rather than an
emitter: a one-byte bool output for the comparisons.

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
- **`aten.gt`, `aten.ge`, `aten.lt`, `aten.le`, `aten.eq`, `aten.ne`**:
  `GREATER`(9) and `LESS`(10) exist. They write int32 1/0 or fp16 1.0/0.0, and
  the table has no one-byte mode, so a node declaring `torch.bool` cannot be
  handed one without an out-of-bounds write. This is a kernel change, not an
  emitter change, and it is still open: `where`'s arrival did not need it, since a
  comparison's result is a `where` input rather than a command's output.
  `SQUARED_DIFFERENCE`(11) has no ATen node at all.
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
| Convolution needs static extents and a constant weight; plain convolution additionally needs `groups == 1` or a genuine depthwise shape | `convolution` | an intermediate group count in a plain forward convolution has no channel mapping; grouped transposed convolution is handled by host-side per-group walks; a run-time weight has no command |
| The gather table is a constant, tiled at export, indexed by int32, along axis 0 | `embedding` (int64 indices are refused by dtype), `index_select` (dim must be 0), `index.Tensor` (one index only) | a run-time table, or an index the command can describe. A `tokens.to(torch.int32)` in the model is enough for the common case |
| The weight-only quantized matmul needs `K % 64 == 0` and `N % 32 == 0`, and then either a single activation row (the GEMV entries) or an int4/int8 weight (the prefill entry, command 22 and command 42); the prefill entry also needs `K <= 12672` while `M <= 32` (`PREFILL_M32_MAX_M` / `PREFILL_M32_MAX_K` in `hexagon_ops.py`) | a batch axis, a dynamic (`SymInt`) `M`, and an int4 or int8 prefill with `M <= 32` at `K > 12672` -- before the heap fix `K = 12736` at `M = 4` aborted the DSP process (`0x8000040d`, no output, under a second) while `K = 12672` answered within 3.9e-4 to 6.4e-4, measured on one phone and one skel | general prefill support for those additional geometries. The ceiling is a refusal rather than a repair: the corrected heap-backed skel answers `K = 12800` at `M = 4`, but that is one device and one skel, so the host keeps the conservative bound until a broader device matrix justifies widening it |
| `where`'s three operands are each the result's element count or a single element | a `where` whose condition broadcasts, e.g. `where(cond[2,1,4], a[2,3,4], b)` | the kernel's per-channel value mode, which needs a `plane` / `channel` / `pack` / `batch` this emitter does not compute -- so the same `where` delegates in one model and not in the next |
| A constant pad fills its border with a `ZERO` memset, so the fill is zero and nothing else, and it needs a three-level region | a nonzero `value`; a pad on a third axis from the end (a fourth level); a negative pad (that is a slice); an all-zero pad, whose region is the operand at its own strides and which the kernel drops as a self-write | a fill other than zero is a kernel that writes a value; anything past the last two axes is a different region form. `mode='reflect'`/`'replicate'`/`'circular'` are not this node at all: torch lowers them to `arange`/`abs`/`clamp`/`index` programs, so no pad node reaches the partitioner |
| A command holds three 12-int regions | `cat` (at most 3 operands), `permute_copy` (at most 3 groups that advance, no reversal inside one) | a command form with more regions. The permute count is of the groups that spend a loop, not of the axis groups: a batch of one in front of a head split is a fourth group that iterates once and advances nothing, so `[1, tokens, heads, dim] -> [1, heads, tokens, dim]` is three loops and delegates, while the same split on a batch of two is refused -- the difference is an extent, and `permute_region` in `hexagon_ops.py` is where it is decided |
| A blit describes one run per row | `slice_copy` with `step != 1` | a strided run, or a different command |
| A two-output node is placed only when every reader takes the values output | `max.dim` (indices), `max_pool2d_with_indices`, `topk` (indices), `native_layer_norm` (`getitem 0`), `add_rms_norm` (`getitem 0` or 1) | not a rule waiting to be relaxed: `max.dim` and the pool have kernels that compute values rather than positions, and `topk` is the one kernel here that answers both -- and its positions are refused anyway, because the position it writes is the first occurrence of the row maximum while torch writes whichever index its partial sort stops on, which over 200 rows of quantized values is neither the first nor the last 175 times |
| The narrow view rule: a view that would be a partition boundary is copied instead | `view_copy`, `unsqueeze_copy`, `squeeze_copy`, `expand_copy`, `alias_copy` | nothing -- this is the rule working, and its symptom is a delegate one op shorter than it looks |

The pool, convolution and gather rows above are admission geometry rather than
missing support: each op has an emitter, and a model that produces the geometry
delegates. What each gate takes, at the line that decides it (`hexagon_ops.py` and
`hexagon_partitioner.py`; line numbers move with the file, the function names are
the stable part):

| gate | accepts | measured refusals | line |
|---|---|---|---|
| `pool_spec`: channel count | `C == 64` exactly, and for a 3-D `(C, H, W)` input it is `shape[0]` that is read | 32, 63, 65, 128 and every other count. Over C in {1,3,32,63,64,65,128} x kernel {2,3} x stride {1,2} x {max, avg}: only `C == 64` delegates, at any kernel and stride | `hexagon_ops.py:2110` |
| `pool_spec`: window and counting | any static kernel and stride, `pad >= 0`, the output size floor division gives, every window inside its row | `dilation != (1, 1)`, `ceil_mode` (windows past the input, which the one-block geometry does not describe), `divisor_override` | `:2125`, `:2136`, `:2142` |
| `conv_spec`: group count | plain `groups == 1` or `groups == in_channels == out_channels` with one channel per group; transposed any group count that divides the input and output channels, lowered as one dense walk per group | intermediate counts in a plain forward convolution only: `16->32 groups=16` is two outputs per group and remains portable; the same count is accepted when transposed because the host slices each group | `:3464-3474`, `:4136-4236` |
| `max_pool2d_with_indices`: readers | every reader takes `getitem 0` | reading the indices leaves the pool on the portable kernels, whatever its shape | `hexagon_partitioner.py:607` |

The gather's index width is the same kind of gate: `embedding`, `index_select` and
`index.Tensor` take an int32 index tensor and refuse an int64 one (`:3733`), which
is why a `tokens.to(torch.int32)` in the model is enough and an `nn.Embedding`
spelled with `torch.long` indices is not. A graph can hold a delegate either way --
the ops after the refused one form their own -- so a delegate count says nothing
about which of the two happened, and the command stream does.

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

- **Quantized prefill.** This was the only functional gap -- the weight-only path
  was `M == 1` only, so a quantized model could decode and not prefill -- and it
  is now half closed: an int4 weight above one row reaches
  `DSP_OP_MATMUL_Q4A16_FP16`(22), which is the row that left §2. The half that
  stays is the int8 weight, whose prefill kernel (42) reads a tile order this tree
  does not write. Prefill's own evidence is host lowering plus hexagon-sim, at
  every layer below the device: no timing, and no device.
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
  `test_hexagon_quantizer.py::test_a_w8a16_prefill_matmul_lowers_to_command_42`
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
   refusal past a measured bound, not a missing packer. The part of this item that
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
   here writes a one-byte bool, so a comparison whose node declares `torch.bool`
   still stays portable, and that is a kernel change.
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
