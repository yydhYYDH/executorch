# PARTITION_GATES — what refuses a node, how many nodes each clause holds, and whether removing it is safe

Every node `HexagonOperatorSupport._verdict` turns away is a node on the portable kernels. That
is a correct program and a quiet one: nothing prints, the model still returns torch's answer, and
the only symptom is a delegate one op shorter than the graph reads. The two counters that exist,
`refused_overload_census()` and `unwired_overload_census()`, both say in their own comments that
they carry no reason, so this file is the reason, one clause at a time.

**Evidence tier for the whole file: host lowering.** Export, `to_edge_transform_and_lower` with
the Hexagon partitioner, the support predicate, and the AOT blob read back with the runtime's own
`read_blob`. No kernel ran, `hexagon-sim` did not run, and no phone was involved. A row that says
"unsafe" says what the tree believes would go wrong, not what was observed going wrong.

**Base.** `hexagon-integration-a47` at `cef244b`, branch `hexagon-gateinv`, worktree
`.tmp/wt/gateinv/executorch`. Line numbers are this checkout's; `test_partition_gates.py`
asserts the table below against the source, so a line that moves fails the test rather than making
this file quietly wrong.

## 1. The three clauses that were already measured, and where they sit at a47

The facts hold. The line numbers in the brief are from another tree and do not, so they are
re-derived here.

| brief | at a47 | the fact |
|---|---|---|
| `SUPPORTED_TARGETS` IS `EMITTERS`, hexagon_backend.py:21 | `hexagon_backend.py:37` | `SUPPORTED_TARGETS: Dict[Callable, Callable] = EMITTERS`, so "the partitioner does not know this target" and "no emitter exists" are one clause, line 678 |
| the width gate reads the RESULT dtype, hexagon_partitioner.py:316 | `hexagon_partitioner.py:679-686` | `dtype = _dtype_of(node)` reads `node.meta["val"]`, the result. The operand width is a *different* clause, line 691 |
| `_require_arena_dtype` raises, hexagon_ops.py:192-202 | `hexagon_ops.py:546-556` | `raise RuntimeError(...)`, not a fallback, which is why admitting a width at line 686 without touching it converts a working export into a failed one |

The other half of the third fact is line 987, which is in this inventory and is the partitioner's
own copy of the same width rule for constant operands. It is marked measured because the pairing is
the finding.

## 2. How a reason was read, and the three controls that say the reading is right

A support-predicate counter counts calls, and `_verdict` asks about a node more than once per
lowering, so a count read after a lowering is a call count unless the counter keeps the nodes. The
reason problem is worse: the two censuses deliberately carry no reason, and the earlier recorder in
`census.py` wraps a hand-kept list of predicate names, which cannot see an inline clause and goes
unattributed for a getitem whose producer was refused.

So the instrument here rewrites the source instead. Every `return False` inside `_verdict` is
replaced, in the text `inspect.getsource` returns for that function, by a call that records the
line it is on and the node it is deciding; the rewritten copy is executed in the module's own
namespace and installed as the method. A node refused by that copy has been refused by the tree's
own code, in the tree's own order, and the last line recorded is the clause that decided it.

Three controls, all in `test_partition_gates.py`:

1. **Set equality.** The table's lines are compared to the `return False` lines found in the
   source, both ways. A clause the tree grows fails; a clause that moves or disappears fails.
2. **Twenty-three designed graphs**, each pinned to one line, spread across the table, together
   with the control that the same graph's ordinary fp16 chain is *accepted*. A support object that
   refused everything would satisfy all twenty-three at once, so the controls are what make the
   rows mean something.
3. **Parts to total.** Over both corpora below, the per-clause counts sum to exactly the number of
   refused nodes, and zero refused nodes carry no clause. A scan that finds nothing and a scan that
   is broken return the same thing, so a census is only worth reading once it has been shown to
   find the refusals that are known to be there.

## 3. The two corpora

**Models.** 21 exported graphs, 6545 `call_function` nodes, 594 refused, 0 unattributed. Nineteen
hand-written geometries (resnet18 and mobilenet_v2 shapes, a 12-layer deit-shaped encoder, a VAE
up-block square and non-square, nearest-exact upsample, a tiny causal LM in prefill and decode, the
sdpa spelling, a vision tower, an audio encoder, and the batch-norm-folded and norm/silu-fused
variants) plus `Qwen3ForCausalLM` built from a `transformers` config at one and at 28 layers
(hidden 1024, 28 layers, vocab 151936, 32 tokens, eager attention, fp16, random weights). Every
graph is run through `transform_for_pre_decomposition` and then `to_edge`, so the census sees the
nodes the partitioner sees.

**Rows.** Every row of `_ROWS` and `_QUANTIZED_ROWS` in `test_overload_census.py` and
`test_overload_census2.py` — 131 designed rows — exported and judged the same way. 58 refused, 0
unattributed, 0 unexportable. This corpus is thin on refusals by construction (most rows assert that
something *does* delegate) but it is the only corpus in the tree that deliberately aims at the
refusal clauses.

A count that falls while coverage rises is not a regression, and neither corpus is a share of
production traffic. The models corpus is dominated by two families, resnet/mobilenet and Qwen3, and
the rows corpus is a designed sweep, so the ranking below is *how many nodes each clause held on
these two corpora*, and nothing more.

## 4. The inventory, ordered by nodes held

`head-on` counts nodes whose producer the partitioner **accepted**: those are the nodes the clause
itself is holding up. `shadow` counts nodes whose producer was itself refused: those are
consequences, and fixing them means fixing the other end of the same island.

| line | clause | models | rows | total | head-on | shadow | what it turned away |
|---|---|---|---|---|---|---|---|
| 678 | `TARGET_NOT_IN_TABLE` | 298 | 30 | 328 | – | 298 | _native_batch_norm_legit_no_training 160, logical_not 42, full_like 24, eq 21, any 21 |
| 908 | `GETITEM_PRODUCER` | 160 | 2 | 162 | 0 | 160 | getitem 162 — every one behind a refused batch norm |
| 913 | `ALIAS_BYTES_OR_SELECT` | 58 | 1 | 59 | 58 | 0 | expand_copy 59 |
| 686 | `RESULT_DTYPE` | 52 | 3 | 55 | 2 | 50 | unsqueeze_copy 25, add 6, slice_copy 6, sub 5, index 4 |
| 697 | `WHERE_EMITTABLE` | 21 | 0 | 21 | 0 | 21 | where 21, all inside the sdpa fully-masked-row guard |
| 803 | `POOL_SPEC` | 2 | 2 | 4 | 2 | 0 | max_pool2d_with_indices 3, avg_pool2d 1 |
| 767 | `TOPK_EMITTABLE` | 0 | 3 | 3 | – | – | topk 3 |
| 827 | `GATHER_TABLE` | 1 | 2 | 3 | 1 | 0 | index.Tensor 2, embedding 1 |
| 958 | `DIM_ORDER_KEEPS_BYTES` | 2 | 1 | 3 | 0 | 2 | _to_dim_order_copy 3 |
| 750 | `REDUCTION_DIMS` | 0 | 2 | 2 | – | – | sum.dim_IntList 1, amax 1 |
| 691 | `OPERAND_DTYPES_READABLE` | 0 | 1 | 1 | – | – | _to_dim_order_copy 1 |
| 712 | `ADDMM_FITS_FLAT_PATH` | 0 | 1 | 1 | – | – | addmm 1 |
| 722 | `MEAN_REDUCES_ONE_SPAN` | 0 | 1 | 1 | – | – | mean.dim 1 |
| 729 | `MEAN_RESULT_WIDTH` | 0 | 1 | 1 | – | – | mean.dim 1 |
| 733 | `POW_IS_SQUARE` | 0 | 1 | 1 | – | – | pow.Tensor_Scalar 1 |
| 744 | `POW_TENSOR_TENSOR_EMITTABLE` | 0 | 1 | 1 | – | – | pow.Tensor_Tensor 1 |
| 752 | `SUM_DIM_EMITTABLE` | 0 | 1 | 1 | – | – | sum.dim_IntList 1 |
| 754 | `MAX_DIM_EMITTABLE` | 0 | 1 | 1 | – | – | max.dim 1 |
| 760 | `MAX_POOL_EMITTABLE` | 0 | 1 | 1 | – | – | max_pool2d_with_indices 1 |
| 858 | `LOG_SOFTMAX_WITHIN_ARENA` | 0 | 1 | 1 | – | – | _log_softmax 1 |
| 915 | `SLICE_REGION` | 0 | 1 | 1 | – | – | slice_copy 1 |
| 954 | `CLONE_DIM_ORDER_CONTIGUOUS` | 0 | 1 | 1 | – | – | _clone_dim_order 1 |
| 672 | `NOT_CALL_FUNCTION` | 0 | 0 | 0 | – | – | not an op clause; see §6 |
| 704, 706, 708, 710, 719, 736, 756, 777, 787, 796, 818, 833, 835, 837, 839, 847, 849, 866, 872, 886, 917, 919, 922, 928, 930, 933, 935, 943, 960, 968, 975, 987 | 32 clauses | 0 | 0 | 0 | – | – | nothing in either corpus; see §6 for which of them can be reached at all |

The other 32 rows, in `_verdict`'s own order, with what each clause is and whether it is safe to
remove:

| line | clause | what it is | verdict |
|---|---|---|---|
| 704 | `SDPA_FITS_DSP_LIMITS` | a mask the FLASH_ATTN kernel will apply, 4-D operands, batch 1, start_pos readable | unsafe; a mask the emitter got wrong is a different function, not a no-op |
| 706 | `BINARY_BROADCAST_FITS` | the binary descriptor has output extents and two stride tables but no per-operand extents | unsafe; it can repeat singleton axes, not address a smaller non-singleton tile |
| 708 | `MM_OPERANDS_FLAT` | contiguous 2-D operands with the contraction lined up | unsafe; unexercised — `to_edge` folds a batch into M and `FoldConstantTransposes` makes a transposed weight contiguous, so on this path the clause is nearly unreachable |
| 710 | `BMM_OPERANDS_FLAT` | contiguous 3-D stacks, no broadcast batch | unsafe; the emitter derives one tile geometry from the shapes |
| 719 | `QUANTIZED_MATMUL_REFUSED` | a weight-only matmul whose conditions the flat path does not decide | unsafe |
| 736 | `POW_TENSOR_TENSOR_NO_PROGRAM` | a support object built without the program | unsafe, but the failure it prevents is an `AttributeError` on `None.graph_module`, not a wrong number. `partition` always passes the program, so this clause is unreachable from the partitioner; the two in-tree callers that omit it are test helpers |
| 756 | `MIN_DIM_EMITTABLE` | every reader takes the values, not the positions | unsafe; same rule as 754 |
| 777 | `REPEAT_FLIP_EMITTABLE` | the region walk exists for this repeat or flip | unsafe; a shape with no region reaches a blit that reads the wrong elements |
| 787 | `NEAREST_UPSAMPLE_EMITTABLE` | one region per destination phase, only at an exact integer ratio | unsafe |
| 796 | `SPLIT_EMITTABLE` | pieces that add up to the axis, each piece's offset and extent baked into the command | unsafe |
| 818 | `CONV_SPEC` | neither kernel here can run this weight order, group count or staging | unsafe; a middle group count like `Conv2d(16, 32, groups=16)` is the case |
| 833 | `LAYER_NORM_EPS_CONST` | a run-time epsilon is not a number the command carries | unsafe |
| 835 | `LAYER_NORM_TRAILING_DIMS` | the kernel's outer-times-inner view of the input | unsafe |
| 837 | `NATIVE_LAYER_NORM_EMITTABLE` | every reader takes the output the kernel writes | unsafe |
| 839 | `ADD_RMS_NORM_EMITTABLE` | every reader takes one of the two outputs it writes | unsafe |
| 847 | `VISION_ATTENTION_EMITTABLE` | batch, head count and head width are params the run-time length cannot refresh | unsafe |
| 849 | `SOFTMAX_INNER_AXIS` | a last-axis softmax always; another axis only when the row is under `SOFTMAX_VECTOR_WIDTH` and both permute regions exist | unsafe, and the emitter re-checks it — see §5 |
| 866 | `GROUP_NORM_ONE_GROUP_PER_ROW` | the groups divide the channels and the normalized shape is the trailing dims | unsafe |
| 872 | `BATCH_NORM_ONE_SPAN` | one contiguous span per channel | unsafe; a batch wider than one normalizes over the batch too |
| 886 | `GETITEM_NORM_PRODUCER` | the norm this getitem reads is itself emittable | unsafe by construction: it is the norm's own gate, one node over |
| 917 | `CAT_PLAN` | one blit per piece, extents baked in | unsafe |
| 919 | `PERMUTE_REGION` | a permutation the three-level region can describe | unsafe; §8.44's rule, a fourth level for a third-from-last axis |
| 922 | `LEAKY_RELU_SLOPE_CONST` | the slope is a number the command can carry | **unexercised and, as far as this torch goes, unreachable**: `F.leaky_relu(x, tensor)` raises `TypeError: negative_slope must be Number` at export, measured |
| 928 | `PRELU_SOURCE` | a contiguous source of rank 2 or more | unexercised; a `permute`d source measured as accepted, so the copy the alias emitter needs is emitted first |
| 930 | `PRELU_SLOPE_RANK` | the slope is 1-D and contiguous | unsafe; a 0-D slope measured refused at this line |
| 933 | `PRELU_SLOPE_NUMEL` | the slope is one value or one per channel | unexercised; the only construction that violates it is one `F.prelu` itself rejects |
| 935 | `REFLECT_PAD_REGIONS` | a contiguous source, a positive pad narrower than the axis it reflects | unsafe; a pad at least as wide as the axis wraps around |
| 943 | `CONSTANT_PAD_REGION` | the region's three levels reach a pad on the last two axes, and the memset writes zero and nothing else | unsafe; a third-axis pad, a negative pad, a symbolic extent or a nonzero value |
| 960 | `UPDATE_CACHE_LAYOUT` | the cache-advance lowering's operands and geometry | unexercised: **the corpus builds no KV cache at all** (`use_cache=False` throughout), and a cached decode is the common LLM deployment shape |
| 968 | `CUMSUM_EMITTABLE` | a contiguous fp16 operand with a 64-aligned static last extent, so the mask the two commands read exists | unsafe, and only reachable by a caller that invokes the custom op: `CUMSUM` is `et_hexagon.cumsum.default`, while `torch.cumsum` exports as `aten.cumsum.default` and is refused at 678 instead, measured |
| 975 | `ARGUMENT_NOT_A_NODE` | after the literal check, anything left is not a node | unsafe, but again against a crash: removing it makes `arg.op` raise on a `torch.Size` or a dtype |
| 987 | `GET_ATTR_NOT_FP16` | a constant operand at the width the arena holds, except where the emitter converts it | unsafe, and paired with `_require_arena_dtype` — see §1 |

## 5. Both ends of the largest island, named separately

**908 `GETITEM_PRODUCER` holds 162 nodes and none of them is work.** All 160 in the models corpus
sit behind a producer refused at 678, and 160 of 298 of 678's nodes are themselves
`aten._native_batch_norm_legit_no_training.default` — inference-mode batch norm, one
`running_mean`/`running_var` getitem each, across resnet18 (4 graphs × 20) and mobilenet_v2
(2 graphs × 40). So the 162 and the 160 are the same fact read from two ends: the getitem is a
consequence of the batch-norm gap and disappears with it. **Aiming a fix at 908 is aiming at the
shadow.** The fix is `FoldBatchNormIntoConv` in the caller's `transform_passes`, which on
`cnn_mobilenetv2` takes 80 refused nodes to 0 and changes no command, because the batch norm was
never reaching a command either way.

**913 `ALIAS_BYTES_OR_SELECT` holds 59 nodes and every one of them is head-on.** Measured
geometry, on a one-layer Qwen3 at the census's hidden 1024: the RoPE frequency tensor is
`(1, 8, 1, 32, 128)` fp16, contiguous, and expanded to `(1, 8, 2, 32, 128)` — the half-rotary
`repeat_interleave`, once for `cos` and once for `sin`. The producer is accepted. The alias
emitter re-points the operand's `TensorRef`, and `_alias_keeps_the_same_bytes` requires equal
element counts (32768 against 65536), so the clause refuses. 2 per layer × 28 = 56 in the 28-layer
census and 2 in the 1-layer, which is the count exactly.

This is the largest clause in the inventory that is holding a node up on its own, and it is the
cheapest kind of gap: the blit region's per-side stride already expresses a zero source stride, so
a broadcast along one axis is a region and no kernel. **Removing 913 without adding that region
would not be safe** — a `TensorRef` re-pointed from a 32768-element buffer to a 65536-element
strided result reads the wrong elements, and nothing downstream would notice.

## 6. No clause here is safe to remove as written, and the three that are closest

Stated plainly because the brief asked for it and the answer is not the expected one. Of the 55
clauses, 52 are the inventory, and **none of the 52 is safe to remove**: every one either stands
between a geometry a program can produce and a kernel that would compute something else, or stands
between that geometry and a crash. Three are worth separating out, because the failure they prevent
is loud rather than silent, which changes their priority and nothing else:

- **672 `NOT_CALL_FUNCTION`** is not an op clause at all. It refuses the graph's own scaffolding —
  placeholders, `get_attr`, `output` — and the census only ever asks about `call_function`
  nodes, so its count is zero by construction rather than by measurement. Its two `return False`
  neighbours are the two entry gates, so removing this one moves every non-op node into the target
  table check.
- **736** and **975** convert a clean fallback into an `AttributeError` if removed. They are the
  two clauses where the cost of keeping is zero and the cost of removing is a stack trace.

Four clauses are *unexercised and, as far as this measurement goes, unreachable by a program torch
can execute* rather than merely unexercised: **922** (a tensor `negative_slope` does not export),
**933** (the only violating slope is one `F.prelu` rejects), **968** (the target is a custom op) and
**928** (a permuted source measures as accepted). Those four are where a removal is most likely to
be safe in practice and are also where the claim would be cheapest to get wrong, so they are called
out rather than folded into the unsafe list.

## 7. Two stale claims this census corrected

Both were found by measuring what the tree does today rather than by reading a document, and both
are claims a future reader would otherwise believe. Neither is in this tree: `README.md`,
`OP_SUPPORT.md` and `OP_GAPS.md` already say what the measurement says, and the two stale copies
are the Hexagon skill's per-op notes and the `PORTABLE-CENSUS` workstream report, the latter taken
on an older rev. Recorded here so the correction travels with the measurement.

- **"Pooling delegates only on exactly 64 channels."** `pool_spec` no longer says that. Its own
  comment now states that the channel axis is a packing granularity and that the walk loops over
  all the 64-lane blocks, so any channel count is a number of blocks. **Measured by decoding the
  blob:** `max_pool2d(2)` over 32, 17, 64 and 96 channels is one delegate each, carrying
  `RASTER_BLIT, POOL2D_FP16, RASTER_BLIT` and — for every count that is not a multiple of 64 — a
  `ZERO` ahead of the pack, because the fill reads whole 64-lane vectors and the lanes past the
  last channel would otherwise hold whatever the arena last held. What does refuse is
  `ceil_mode=True`, at 0 delegates for both `max_pool2d` and `avg_pool2d`, which is what the
  published resnet18 stem pool does and why `cnn_resnet18` and `cnn_resnet18_staticpool` differ by
  exactly that node.
- **"A 197-wide last-axis softmax emits one `DSP_OP_SOFTMAX`."** Stale, and the correction has the
  same shape as the pool one. Measured at a47 by decoding the blob with the runtime's own reader:
  a `(1, 33)` last-axis softmax is one `DSP_OP_SOFTMAX [1, 33, 1, 2]`, while `(1, 197)` and
  `(1, 4096)` are a five-command composition — `REDUCTION`, `BINARY_ELEMENTWISE`, `UNARY`,
  `REDUCTION`, `BINARY_ELEMENTWISE`. That is the same fact the README's own device finding states
  from the other side: the standalone command is wrong for rows longer than one HVX vector, and
  the emitter now gates on the width. A middle-axis softmax is a blit, a `DSP_OP_SOFTMAX` over the
  permuted inner width, and the inverse blit, which is what clause 849 is for. The emitter branches
  on `channel < SOFTMAX_VECTOR_WIDTH` itself, so the clause and the emitter agree and neither is
  the single line of defence — which is why 849 is listed as unsafe but not as a hole.

## 8. What this file is not

It is not a claim that any of these clauses should go. It is a count of what each clause holds on
two corpora, the reason each gives, and the reachability of each — the map an op gap needs before it
is a project. A clause that holds zero nodes on 6545 model nodes and 131 designed rows is not
shown to be unnecessary; it is shown to be unexercised by the only two corpora in this tree, and
960 is the clearest case, since a census with no KV cache cannot say anything about the clause that
governs one.

## 9. Where each number comes from, and what a reader has to rebuild

The rows corpus is in this tree: it is the `_ROWS` and `_QUANTIZED_ROWS` tables of
`test_overload_census.py` and `test_overload_census2.py`, driven through the same instrument.
The models corpus is not — the nineteen geometries are hand-written and live in a scratch
directory, and `Qwen3ForCausalLM` is built from a `transformers` config, so a reader has to
rebuild it to re-derive §4. That is a real limitation of a census whose point is to be re-derived,
and it is worth fixing before the next round rather than after: the geometries belong next to the
other model fixtures, and the instrument and both drivers belong beside
`test_partition_gates.py`, which already carries the instrument and the controls so that a
reader does not have to rebuild those to know the census is sound.
