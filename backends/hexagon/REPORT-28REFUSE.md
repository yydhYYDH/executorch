# 28REFUSE — the 28 non-per-layer refusals, enumerated and ranked

Base for every line number and every number below: hexagon-mergeq @ 165482b,
worktree .tmp/wt/28refuse/executorch, DIRTY=0. Evidence tier: host lowering
unless a line says otherwise. No hexagon-sim, no device.

## 0. Re-derived line numbers on THIS tree (a file:line belongs to its tree)

| thing | brief / handover says | at 165482b |
|---|---|---|
| UPDATE_CACHE_LAYOUT clause | hexagon_partitioner.py:968 (960 at cef244b) | **hexagon_partitioner.py:996** |
| update_cache_layout | hexagon_ops.py:2328-2355 | **hexagon_ops.py:2812-2838** |
| TARGET_NOT_IN_TABLE | 678 / 687 | **hexagon_partitioner.py:687** |
| RESULT_DTYPE (width gate) | 679-686 / 704 | **hexagon_partitioner.py:704** |
| OPERAND_DTYPES_READABLE | 709 | **hexagon_partitioner.py:709** |
| WHERE_EMITTABLE | 697 / 715 | **hexagon_partitioner.py:715** |
| ALIAS_BYTES_OR_SELECT | 913 / 950 | **hexagon_partitioner.py:950** |
| DIM_ORDER_KEEPS_BYTES | 995 | **hexagon_partitioner.py:992** |
| GATHER_TABLE | 958 / 859 | **hexagon_partitioner.py:859** |
| GETITEM_PRODUCER | 908 / 941 | **hexagon_partitioner.py:941** |

Line 968 on this tree is constant_pad_region (the PAD_TARGETS clause) -- a
different clause entirely, so the brief's 968 does not point at the cache.

## 1. Highest-value item: a cached-decode step writes past the end of the arena

**Re-derived, not inherited. 12 geometries, 6 of them overrun, 12 of them accepted.**

    case                   verdict write[start,end)   arena     OOB    reg  commands
    in-bound   seq5 pos4 r1 True    (256, 320)         320       False  1    [3, 3]
    OOB       seq5 pos5 r1 True    (320, 384)         320       True   1    [3, 3]
    OOB       seq5 pos6 r1 True    (384, 448)         320       True   1    [3, 3]
    OOB       seq5 pos7 r1 True    (448, 512)         320       True   1    [3, 3]
    OOB r=2   seq5 pos4 r2 True    (256, 384)         320       True   1    [3, 3]
    in-bnd r2 seq5 pos3 r2 True    (192, 320)         320       False  1    [3, 3]
    OOB dim48 seq5 pos5 r1 True    (480, 576)         480       True   1    [3, 3]
    OOB 4head seq6 pos5 r1 True    (640, 768)         768       False  1    [3, 3]
    in-bound  seq8 pos5 r1 True    (320, 384)         512       False  1    [3, 3]
    OOB       seq8 pos8 r1 True    (512, 576)         512       True   1    [3, 3]
    OOB 1head seq4 pos3 r1 True    (96, 128)          128       False  1    [3, 3]
    in-bound  seq4 pos2 r1 True    (64, 96)           128       False  1    [3, 3]

    all accepted by the predicate  : True
    cases writing past the arena   : 6
    verdict True while writing OOB  : 6
    seq5 pos5, position dtype      : i32 True, i64 True

The write geometry is the tree's own. update_cache_layout returns
run, inner = numel // shape[-3], rows = value.numel() // run, and
_emit_update_cache emits patch=(5, 2), patch_scale=inner, so the blit
destination offset is pos * inner and it covers rows * run elements. The arena
is numel = seq * inner. The bound that is missing is
pos * inner + rows * run <= numel, which is pos + (rows*run)//inner <= seq.

update_cache_layout reads position exactly twice, both before the geometry:
the three-way unpack of node.args, and the isinstance(part, fx.Node) test. It
never reads position.meta["val"]. grep -rn APPEND_BOUND over backends/hexagon/
returns 0 hits; so do pos + rows and pos+rows. The bound does not exist anywhere
in the backend. Every case above, in-bounds and out, lowers to one delegate of
two RASTER_BLIT (3).

**Why it is latent.** match_kv_cache matches the getitem[1] of
auto_functionalized_v2(...), and that op raises on export; the mutable (a!) schema
is refused by to_edge; FuseKvCachePass and use_stored_caches have zero callers.
So et_hexagon::update_cache is reachable only when a graph names it directly,
which is what this probe does. The moment FuseKvCachePass gets a caller this is
memory corruption -- and the clause that would have caught it is the one clause
in the tree whose docstring does not say why it exists.

**What predicate would fix it, in the honest form of that answer.** No
export-time predicate can close this, because position is a runtime tensor and
the overrun depends on its value rather than on any shape. A predicate that
reads node.args[2].meta["val"] sees the same value for pos=4 and for pos=5, so
a static check either refuses every decode step -- which is the whole point of
the node -- or accepts every overrun. The two changes that do close it:

(a) **Runtime clamp. Cheapest, and the one to do.** The runtime already rewrites
    inputs before the blob runs and the position arrives as a one-element integer
    tensor. Clamp it to [0, seq - rows_tokens] there, where seq is the cache's
    own shape[-3] and rows_tokens is the value's token count; both are already in
    hand at that point. About three lines in hexagon_backend.cpp, next to
    narrow_indices_to_int32, which is the same kind of fix for the same tensor.
    It is correct for every position, and it turns a step that silently runs off
    the arena into one that writes the last valid row.

(b) **Emitter-side clamp.** Emit a min against a scalar constant into the arena
    before the blit. DSP_OP_BINARY_ELEMENTWISE min already exists (aten.clamp.Tensor
    emits two of them), so this is zero new DSP C++, but it needs a scratch region
    sized by hand against a VTCM budget that is not free. Strictly more work than
    (a) for the same result.

There is a third, smaller predicate change that is worth making today and that
belongs in update_cache_layout: refuse the geometry outright when rows * run
exceeds numel, i.e. when the value is larger than the entire cache and no
position can hold it. That is a shape-only fact, it is one line, and it closes
the degenerate case a hand-built graph can still reach. pos * inner + rows * run
<= numel is the complete bound; the runtime clamp is what enforces it.
