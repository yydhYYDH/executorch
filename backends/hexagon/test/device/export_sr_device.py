"""Super-resolution (EDSR) on the Hexagon backend: host lowering + census.

Adds four things the family's census script did not have:

  1. the command stream is decoded with \`enum DSPOpType\` parsed out of
     backends/hexagon/third-party/mnn-htp-ops/include/htp_command.h:32 --
     the C++ enum, not the partial Python mirror (skill §0 / §8.95).
  2. every refused node is attributed to the exact predicate line, by
     line-tracing the predicate that the partitioner consults.
  3. the reference is computed in the SAME process and the SAME module
     instance as the export (skill §5), and written to disk next to the
     input so the device run can be compared against it.
  4. the portable nodes are classified, not counted: refused / accepted
     but dragged out by a refused source / getitem of a delegate.

usage: python exsr.py <outdir> [H W]
"""
import collections
import json
import operator
import os
import re
import resource
import sys

import torch

# The tree under test is the one this script lives in, found by walking up to
# the marker rather than by naming a worktree: a driver that only runs from the
# branch it was written on cannot answer whether the integrated tree still works.
LAND = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
assert os.path.isdir(LAND + "/backends/hexagon"), LAND
FAMDEPS = "/home/yydh/executorch/.tmp/scratch/famdeps"
sys.path.insert(0, LAND + "/src")
sys.path.insert(0, LAND + "/backends/hexagon/test")
sys.path.insert(0, FAMDEPS)

import executorch.backends.hexagon.partition.hexagon_partitioner as _P
from blob_interpreter import read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    SUPPORTED_TARGETS,
    HexagonOperatorSupport,
    HexagonPartitioner,
    _data_placeholders,
    refused_overload_census,
    reset_refused_overload_census,
    reset_unwired_overload_census,
    unwired_overload_census,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

# The tree under test must be the tree that is imported. Load-bearing, and
# checked by identity of the tree rather than by the name of one worktree.
assert _P.__file__.startswith(LAND), (_P.__file__, LAND)
assert hexagon_ops.__file__.startswith(LAND), (hexagon_ops.__file__, LAND)

CFG = EdgeCompileConfig(_check_ir_validity=False)
DELEGATE = torch.ops.higher_order.executorch_call_delegate
HTP_HEADER = LAND + "/backends/hexagon/third-party/mnn-htp-ops/include/htp_command.h"
HEXOPS_SRC = LAND + "/backends/hexagon/hexagon_ops.py"
PART_SRC = LAND + "/backends/hexagon/partition/hexagon_partitioner.py"


# ---------------------------------------------------------------- C++ enum
def cpp_dsp_op_types(path=HTP_HEADER):
    """enum DSPOpType, read from the C++ header, implicit values filled in.

    The header is contiguous from 0 and then assigns explicitly from 40, so a
    naive name-only parse would put every value at 0. Both forms are handled.
    """
    text = open(path).read()
    body = text.split("enum DSPOpType {", 1)[1].split("};", 1)[0]
    out, nxt = {}, 0
    for line in body.splitlines():
        line = re.sub(r"//.*", "", line).strip().rstrip(",").strip()
        if not line or not line.startswith("DSP_OP_"):
            continue
        m = re.match(r"(DSP_OP_\w+)\s*(?:=\s*(\d+))?$", line)
        if not m:
            continue
        if m.group(2) is not None:
            nxt = int(m.group(2))
        out[nxt] = m.group(1)
        nxt += 1
    return out


# --------------------------------------------------------------- fx helpers
def nm(t):
    if t is operator.getitem:
        return "operator.getitem"
    sch = getattr(t, "_schema", None)
    return str(sch.name) if sch is not None else str(t)


def _sub(program, node):
    arg = node.args[0] if node.args else None
    if not isinstance(arg, torch.fx.Node):
        return None
    return dict(program.graph_module.named_children()).get(arg.target)


def _inner_gm(program, node):
    s = _sub(program, node)
    if s is None:
        return None
    gm = getattr(s, "graph_module", None)
    if gm is not None:
        return gm
    orig = getattr(s, "original_module", None)
    return getattr(orig, "graph_module", None) if orig is not None else None


def delegates(program):
    return [n for n in program.graph_module.graph.nodes if n.target is DELEGATE]


def all_op_nodes(program):
    rows = []
    for n in program.graph_module.graph.nodes:
        if n.op != "call_function":
            continue
        if n.target is DELEGATE:
            gm = _inner_gm(program, n)
            if gm is not None:
                for m in gm.graph.nodes:
                    if m.op == "call_function":
                        rows.append((m, "delegate", n.name))
            continue
        rows.append((n, "outer", None))
    return rows


def tv(v):
    if not hasattr(v, "shape"):
        return "?"
    return "%s %s" % (tuple(v.shape), str(v.dtype).replace("torch.", ""))


# ----------------------------------------------------- refusal attribution
def trace_last_line(fn, *a, **kw):
    """Run fn, return (result, [(file, line, text), ...] of lines it executed).

    Only the two source files under test are traced, so the cost is the
    predicate's own body. The last traced line is where the predicate bailed.
    """
    # co_filename is whatever path the interpreter was handed, so an absolute
    # watch set silently matches nothing; compare basenames. The caller's own
    # frame has to be watched too, because a global trace function is only
    # consulted for frames created after it is installed -- without this the
    # chain stops here and the callee is never seen at all.
    watching = {"hexagon_ops.py", "hexagon_partitioner.py", os.path.basename(__file__)}
    cur = {"last": None, "entered": 0, "funcs": set()}

    def local_trace(frame, event, arg):
        base = os.path.basename(frame.f_code.co_filename)
        if event == "line" and base in ("hexagon_ops.py", "hexagon_partitioner.py"):
            cur["last"] = (base, frame.f_lineno)
            cur["funcs"].add(base + ":" + frame.f_code.co_name)
        return local_trace

    def glob_trace(frame, event, arg):
        base = os.path.basename(frame.f_code.co_filename)
        if event == "call" and base in watching:
            if base != os.path.basename(__file__):
                cur["entered"] += 1
            return local_trace
        return None

    old = sys.gettrace()
    sys.settrace(glob_trace)
    try:
        out = fn(*a, **kw)
    finally:
        sys.settrace(old)
    where = None
    if cur["last"] is not None:
        base, ln = cur["last"]
        src = HEXOPS_SRC if base == "hexagon_ops.py" else PART_SRC
        where = (base, ln, open(src).read().splitlines()[ln - 1].strip())
    return out, where, cur["entered"]


def attribute(node):
    """Which predicate refused this node, and at which line."""
    tgt = node.target
    info = {"target": nm(tgt)}
    src = node.args[0] if node.args else None
    sv = src.meta.get("val") if isinstance(src, torch.fx.Node) else None
    info["in_shape"] = tv(sv)
    info["out_shape"] = tv(node.meta.get("val"))
    info["args"] = repr([a for a in node.args[1:]])[:200]
    info["kwargs"] = repr(dict(node.kwargs))[:200]
    # Key on the schema name, never on object identity: the lowered program may
    # hold a re-created OpOverload for the same schema, and a membership test on
    # identity then reports "off table" for a target the table holds.
    def has(frozenset_targets, overload):
        want = str(getattr(overload, "_schema", ""))
        return any(str(getattr(x, "_schema", "")) == want for x in frozenset_targets)

    is_permute = has(hexagon_ops.PERMUTE_TARGETS, tgt)
    is_dimorder = has(hexagon_ops.DIM_ORDER_TARGETS, tgt)
    info["is_permute_target"] = is_permute
    info["is_dim_order_target"] = is_dimorder
    info["identity_in_table"] = bool(tgt in SUPPORTED_TARGETS)
    info["schema_in_table"] = has(SUPPORTED_TARGETS, tgt)
    if is_permute:
        fn = lambda: hexagon_ops.permute_region(node)
        info["predicate"] = "hexagon_ops.permute_region"
        info["refused_when"] = "returns None"
    elif is_dimorder:
        fn = lambda: not hexagon_ops.dim_order_keeps_the_bytes(node)
        info["predicate"] = "hexagon_ops.dim_order_keeps_the_bytes"
        info["refused_when"] = "returns False"
    else:
        info["predicate"] = None
        return info
    res, where, entered = trace_last_line(fn)
    info["predicate_result"] = repr(res)
    info["predicate_entered"] = entered
    info["bail_line"] = "%s:%d" % (where[0], where[1]) if where else None
    info["bail_text"] = where[2] if where else None
    # for permute: also report what the group split was
    if is_permute and sv is not None and hasattr(sv, "shape"):
        ranks = [sv.dim(), node.meta["val"].dim()]
        info["rank"] = ranks
    return info


# ------------------------------------------------------------------- main
def build(h, w):
    from torchsr.models import edsr_r16f64
    m = edsr_r16f64(2, True).eval()
    return m, (torch.randn(1, 3, h, w),)


def main():
    outdir = sys.argv[1]
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    os.makedirs(outdir, exist_ok=True)
    ENUM = cpp_dsp_op_types()
    print("ENUM_FROM_CPP_HEADER n=%d" % len(ENUM))
    print("ENUM_SAMPLE %s" % {k: ENUM[k] for k in sorted(ENUM)[:8]})

    torch.manual_seed(0)
    module, inputs = build(h, w)

    # Reference in THIS process and THIS module instance (skill §5).
    with torch.no_grad():
        ref = module(*inputs)
    ref_np = ref.numpy().astype("float32")
    print("INPUT %s" % (tuple(inputs[0].shape),))
    print("REFERENCE_SHAPE %s absmax=%.6f" % (tuple(ref.shape), float(ref.abs().max())))

    program = export(module, inputs)
    reset_unwired_overload_census()
    reset_refused_overload_census()
    # NOTE: the census is read BEFORE to_executorch(), on purpose.
    # manager.to_executorch() runs further passes that rewrite the graph in place
    # (it turns the dim-order / permute ops into their out= form and adds alloc
    # nodes), so a census read after it counts nodes the partitioner never saw.
    manager = to_edge_transform_and_lower(
        program, partitioner=[HexagonPartitioner()], compile_config=CFG
    )
    ep = manager.exported_program()
    nodes_before = len(all_op_nodes(ep))
    unwired = dict(unwired_overload_census())
    refused = dict(refused_overload_census())
    print("STATUS LOWERED")
    print("UNWIRED_CENSUS %s" % unwired)
    print("REFUSED_CENSUS %s" % {k: dict(v) if hasattr(v, "items") else v for k, v in refused.items()})

    support = HexagonOperatorSupport(_data_placeholders(program))
    rows = all_op_nodes(ep)
    per = collections.OrderedDict()
    details = []
    for node, where, dname in rows:
        t = nm(node.target)
        rec = per.setdefault(t, {
            "et": getattr(node.target, "__name__", str(node.target)),
            "n": 0, "in_table": bool(node.target in SUPPORTED_TARGETS),
            "accept": 0, "reject": 0, "deleg": 0, "outer": 0,
            "shapes": [], "nodes": [],
        })
        rec["n"] += 1
        rec["deleg" if where == "delegate" else "outer"] += 1
        ok = bool(support.is_node_supported(None, node))
        rec["accept" if ok else "reject"] += 1
        v = node.meta.get("val")
        if len(rec["shapes"]) < 4:
            rec["shapes"].append(tv(v))
        entry = {
            "name": node.name, "target": t, "where": where,
            "delegate": dname, "supported": ok, "in_table": bool(node.target in SUPPORTED_TARGETS),
            "out": tv(v), "src": tv(node.args[0].meta.get("val")) if node.args and isinstance(node.args[0], torch.fx.Node) else None,
        }
        if not ok:
            entry.update(attribute(node))
        rec["nodes"].append(entry)
        details.append(entry)

    cmds = collections.Counter()
    per_delegate = []
    for d in delegates(ep):
        s = _sub(ep, d)
        raw = getattr(s, "_processed_bytes", None)
        names = []
        if raw is not None:
            for c in read_blob(bytes(raw))[1]:
                nmv = ENUM.get(c.type, "UNMAPPED_%d" % c.type)
                cmds[nmv] += 1
                names.append(nmv)
        per_delegate.append({"node": d.name, "n_cmds": len(names), "bytes": len(bytes(raw)) if raw else 0})
        gm = _inner_gm(ep, d)
        inner = [nm(m.target) for m in gm.graph.nodes if m.op == "call_function"] if gm else []
        per_delegate[-1]["inner_ops"] = inner

    print("CMD_TOTAL %s" % dict(cmds))
    for pd in per_delegate:
        print("DELEGATE %s n_cmds=%d blob_bytes=%d inner=%s" % (pd["node"], pd["n_cmds"], pd["bytes"], pd["inner_ops"]))

    total = sum(r["n"] for r in per.values())
    deleg = sum(r["deleg"] for r in per.values())
    off = sorted(k for k, r in per.items() if not r["in_table"])
    rej = sorted(k for k, r in per.items() if r["in_table"] and r["reject"])
    print("SUMMARY unique=%d in_table=%d off=%d nodes=%d delegated=%d portable=%d delegates=%d"
          % (len(per), len(per) - len(off), len(off), total, deleg, total - deleg, len(delegates(ep))))
    print("OFF_TABLE %s" % off)
    print("REJECTED_TARGETS %s" % rej)
    for k in sorted(per, key=lambda k: -per[k]["n"]):
        r = per[k]
        print("OP %s n=%d table=%s accept=%d reject=%d deleg=%d outer=%d geom=%s"
              % (k, r["n"], "IN" if r["in_table"] else "OFF", r["accept"], r["reject"],
                 r["deleg"], r["outer"], " | ".join(r["shapes"])))

    print("---- REFUSED NODES (attributed) ----")
    for e in details:
        if not e["supported"]:
            print("REFUSED name=%s target=%s where=%s in_table=%s" % (e["name"], e["target"], e["where"], e["in_table"]))
            print("   src=%s -> out=%s args=%s kwargs=%s" % (e.get("in_shape"), e.get("out_shape"), e.get("args"), e.get("kwargs")))
            print("   predicate=%s -> %s (refused when it %s)" % (e.get("predicate"), e.get("predicate_result"), e.get("refused_when")))
            print("   BAIL %s : %s" % (e.get("bail_line"), e.get("bail_text")))
    print("---- PORTABLE NODES (all, classified) ----")
    for e in details:
        if e["where"] == "outer":
            print("PORTABLE name=%s target=%s supported=%s in_table=%s src=%s out=%s"
                  % (e["name"], e["target"], e["supported"], e["in_table"], e.get("src"), e.get("out")))
    print("PEAK_RSS_MB %d" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024))

    # The .pte is written last, and the node count is taken either side of it, so
    # the report can say how many nodes to_executorch() added to the graph the
    # census was read from.
    pte_path = os.path.join(outdir, "edsr.pte")
    with open(pte_path, "wb") as fh:
        manager.to_executorch().write_to_file(fh)
    print("PTE %s bytes=%d" % (pte_path, os.path.getsize(pte_path)))
    print("GRAPH_MUTATION op_nodes_before_to_executorch=%d after=%d"
          % (nodes_before, len(all_op_nodes(ep))))

    # artefacts
    torch.save(inputs[0], os.path.join(outdir, "edsr_in0.pt"))
    import numpy as np
    np.save(os.path.join(outdir, "edsr_ref.npy"), ref_np)
    inputs[0].numpy().astype("float32").tofile(os.path.join(outdir, "edsr_in0.bin"))
    with open(os.path.join(outdir, "edsr_census.json"), "w") as f:
        json.dump({
            "input_shape": list(inputs[0].shape),
            "ref_shape": list(ref.shape),
            "enum_from_cpp": {str(k): v for k, v in ENUM.items()},
            "commands": dict(cmds),
            "delegates": per_delegate,
            "per_target": {k: {kk: vv for kk, vv in v.items()} for k, v in per.items()},
            "nodes": details,
        }, f, indent=1, default=str)
    print("WROTE %s" % outdir)


if __name__ == "__main__":
    main()
