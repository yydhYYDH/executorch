# Refusal audit — the non-per-layer refusals, measured

**Tree: `1e59f07` (`hexagon-mergeq`, detached in
`.tmp/wt/refusal28/executorch`, branch `hexagon-refusal28`).**
Every file:line below was re-derived on that tree; the numbers other documents
quote for them are from `cef244b`/`a47` and are **not** interchangeable.

Tier: **host lowering** unless stated. Nothing here is a statement about a DSP.

## 1. The gate clause lines, on this tree

| clause | line | what it is |
|---|---|---|
| `TARGET_NOT_IN_TABLE` | `hexagon_partitioner.py:685` (test at `:687`) | `node.target not in SUPPORTED_TARGETS` |
| `RESULT_DTYPE` | `:696` (test at `:704`) | `result_dtype_is_emittable` — reads the **result** dtype |
| `WHERE_EMITTABLE` | `:710` (test at `:715`) | `where_is_emittable` |
| `GATHER` | `:859` | `gather_table` |
| `GETITEM_PRODUCER` | `:941` | `add_rms_norm_getitem` |
| `DIM_ORDER` | `:992` (test at `:995`) | `dim_order_keeps_the_bytes` |
| `UPDATE_CACHE_LAYOUT` | `:996` (test at `:997`) | `update_cache_layout(node) is None` |

The census instrument records the line of the `return False`, so its numbers
are the second number in each row.

## 2. The count, measured

28-layer Qwen3-0.6B, `transformers 5.17.0`, `to_edge_transform_and_lower` with
`HexagonPartitioner()`, every target in the emitter table watched.

| clause | nodes | targets |
|---|---|---|
| `715 WHERE_EMITTABLE` | **28** | `aten::where` x 28 |
| `704 RESULT_DTYPE` | **23** | `unsqueeze_copy` 11, `add` 3, `slice_copy` 3, `index` 2, `sub` 2, `cat` 1, `expand_copy` 1 |
| `995 DIM_ORDER` | **1** | `dim_order_ops::_to_dim_order_copy` |
| **sum** | **52** | |

**Parts-to-total: census sum 52 = refused records 52 = unique node names 52.
It closes.** The 28 `where` split **28 refused / 28 accepted**, and the graph
carries **28 `DSP_OP_SELECT`** commands, so both populations are real.

The 23 width refusals are **22 `int64` + 1 `bool`**, and the width gate is
confirmed load-bearing on every one of them (`width_gate_accepts=False` 23/23).
It is **not** the reason on any of the other 29 (`width_gate_accepts=True` 29/29).

**`aten::cumsum` x 1 is not in any of those numbers.** It is refused at the
membership test `687` because only the fused `et_hexagon::cumsum` is
registered, so the width gate is never read — and it is invisible to *both*
census counters. Measured: `refused_overload_census` has no `cumsum` key and
`unwired_overload_census()` is `{}`.
