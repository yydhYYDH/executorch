"""TTS (Voxtral-4B-TTS CodecDecoder) on the Hexagon backend: host census + export.

Adapted from backends/hexagon/test/device/export_sr_device.py.  The properties
that script was written to guarantee are kept:

  1. the command stream is decoded with enum DSPOpType parsed out of the C++
     header (skill section 0 / 8.95), not the partial Python mirror;
  2. every refused node is attributed to the exact predicate line;
  3. the reference is computed in the SAME process and the SAME module instance
     as the export (skill section 5) and written next to the input;
  4. the census is read BEFORE to_executorch(), which rewrites the graph in place;
  5. input itemsize is read from the ExportedProgram signature, not assumed.

usage: python export_tts_device.py <outdir> [T] [scale]
"""
import collections
import json
import operator
import os
import re
import resource
import sys
import types

import torch

def _repo_root():
    # Walk up to the tree root by looking for the two things every tree has,
    # rather than counting directory levels -- the script's own depth has
    # already been miscounted once.
    d = os.path.dirname(os.path.abspath(__file__))
    while d != "/":
        if os.path.isdir(os.path.join(d, "backends", "hexagon")) and os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root not found above " + __file__)


WT = _repo_root()
sys.path.insert(0, WT + "/src")
sys.path.insert(0, WT + "/backends/hexagon/test")
sys.path.insert(0, WT + "/examples/models/voxtral_tts")

# model.py line 23 imports custom_ops only for its side effect on the llama op
# namespace (# noqa: F401).  That import needs a prebuilt .so no worktree has.
_stub = types.ModuleType("executorch.extension.llm.custom_ops.custom_ops")
sys.modules["executorch.extension.llm.custom_ops.custom_ops"] = _stub

import executorch.backends.hexagon.partition.hexagon_partitioner as _P  # noqa: E402
from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    SUPPORTED_TARGETS,
    HexagonOperatorSupport,
    HexagonPartitioner,
    _data_placeholders,
    refused_overload_census,
    reset_refused_overload_census,
    reset_unwired_overload_census,
    unwired_overload_census,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

import rsscap  # noqa: E402

rsscap.start()

# The tree under test must be the tree that is imported.  Load-bearing.
assert _P.__file__.startswith(WT), _P.__file__
assert hexagon_ops.__file__.startswith(WT), hexagon_ops.__file__

import model as M  # noqa: E402

CFG = EdgeCompileConfig(_check_ir_validity=False)
DELEGATE = torch.ops.higher_order.executorch_call_delegate
HTP_HEADER = WT + "/backends/hexagon/third-party/mnn-htp-ops/include/htp_command.h"
HEXOPS_SRC = WT + "/backends/hexagon/hexagon_ops.py"
PART_SRC = WT + "/backends/hexagon/partition/hexagon_partitioner.py"
ENUM = None


def cpp_dsp_op_types(path=HTP_HEADER):
    """enum DSPOpType, read from the C++ header, implicit values filled in."""
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


def trace_last_line(fn, *a, **kw):
    watching = {"hexagon_ops.py", "hexagon_partitioner.py", os.path.basename(__file__)}
    cur = {"last": None, "entered": 0}

    def local_trace(frame, event, arg):
        base = os.path.basename(frame.f_code.co_filename)
        if event == "line" and base in ("hexagon_ops.py", "hexagon_partitioner.py"):
            cur["last"] = (base, frame.f_lineno)
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
    tgt = node.target
    info = {"target": nm(tgt)}
    src = node.args[0] if node.args else None
    sv = src.meta.get("val") if isinstance(src, torch.fx.Node) else None
    info["in_shape"] = tv(sv)
    info["out_shape"] = tv(node.meta.get("val"))
    info["args"] = repr([a for a in node.args[1:]])[:220]
    info["kwargs"] = repr(dict(node.kwargs))[:220]

    def has(fr, overload):
        want = str(getattr(overload, "_schema", ""))
        return any(str(getattr(x, "_schema", "")) == want for x in fr)

    is_permute = has(hexagon_ops.PERMUTE_TARGETS, tgt)
    is_dimorder = has(hexagon_ops.DIM_ORDER_TARGETS, tgt)
    info["is_permute_target"] = is_permute
    info["is_dim_order_target"] = is_dimorder
    info["identity_in_table"] = bool(tgt in SUPPORTED_TARGETS)
    info["schema_in_table"] = has(SUPPORTED_TARGETS, tgt)
    if is_permute:
        fn = lambda: hexagon_ops.permute_region(node)  # noqa: E731
        info["predicate"] = "hexagon_ops.permute_region"
        info["refused_when"] = "returns None"
    elif is_dimorder:
        fn = lambda: not hexagon_ops.dim_order_keeps_the_bytes(node)  # noqa: E731
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
    return info


def build(T, scale=1.0):
    cfg = M.VoxtralTTSConfig()
    if scale != 1.0:
        cfg.codec_dim = max(64, int(cfg.codec_dim * scale))
        cfg.codec_hidden_dim = max(128, int(cfg.codec_hidden_dim * scale))
        cfg.semantic_dim = max(32, int(cfg.semantic_dim * scale))
        cfg.semantic_codebook_size = max(128, int(cfg.semantic_codebook_size * scale))
    torch.manual_seed(0)
    dec = M.CodecDecoder(cfg).eval()
    n_cb = 1 + cfg.acoustic_dim
    codes = torch.randint(0, min(cfg.semantic_codebook_size, 512), (1, n_cb, T), dtype=torch.int64)
    return dec, (codes,), cfg


def main():
    global ENUM
    outdir = sys.argv[1]
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    os.makedirs(outdir, exist_ok=True)
    ENUM = cpp_dsp_op_types()
    print("ENUM_FROM_CPP_HEADER n=%d" % len(ENUM))
    print("TREE %s" % _P.__file__)

    module, inputs, cfg = build(T, scale)
    nparam = sum(p.numel() for p in module.parameters())
    print("CONFIG scale=%s codec_dim=%d codec_hidden=%d sem_dim=%d sem_cb=%d n_params=%d (%.1f MB fp32)"
          % (scale, cfg.codec_dim, cfg.codec_hidden_dim, cfg.semantic_dim,
             cfg.semantic_codebook_size, nparam, nparam * 4 / 1e6))
    print("INPUT shape=%s dtype=%s itemsize=%d" % (tuple(inputs[0].shape), inputs[0].dtype, inputs[0].element_size()))

    with torch.no_grad():
        ref = module(*inputs)
    import numpy as np
    ref_np = ref.detach().numpy().astype("float32")
    print("REFERENCE_SHAPE %s absmax=%.6f rms=%.6f" % (tuple(ref.shape), float(ref.abs().max()), float(ref.pow(2).mean().sqrt())))

    program = export(module, inputs)
    reset_unwired_overload_census()
    reset_refused_overload_census()

    # The input itemsize is READ from the exported program, never assumed
    # (skill trap: bytes = numel x itemsize, and itemsize is not always 4).
    placeholders = {n.name: n for n in program.graph.nodes if n.op == "placeholder"}
    sig = []
    for sp in program.graph_signature.input_specs:
        kind = str(getattr(sp, "kind", ""))
        if "USER_INPUT" not in kind:
            continue
        node = placeholders.get(sp.arg.name)
        val = node.meta.get("val") if node is not None else None
        if val is not None:
            sig.append({"name": sp.arg.name, "dtype": str(val.dtype),
                        "itemsize": val.element_size(), "shape": list(val.shape),
                        "bytes": int(val.numel()) * int(val.element_size())})
        else:
            sig.append({"name": sp.arg.name, "dtype": "UNREADABLE"})
    print("PROGRAM_INPUT_SPEC %s" % sig)

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
            "accept": 0, "reject": 0, "deleg": 0, "outer": 0, "shapes": [], "nodes": [],
        })
        rec["n"] += 1
        rec["deleg" if where == "delegate" else "outer"] += 1
        ok = bool(support.is_node_supported(None, node))
        rec["accept" if ok else "reject"] += 1
        v = node.meta.get("val")
        if len(rec["shapes"]) < 4:
            rec["shapes"].append(tv(v))
        entry = {
            "name": node.name, "target": t, "where": where, "delegate": dname,
            "supported": ok, "in_table": bool(node.target in SUPPORTED_TARGETS),
            "out": tv(v),
            "src": tv(node.args[0].meta.get("val")) if node.args and isinstance(node.args[0], torch.fx.Node) else None,
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
        print("DELEGATE %s n_cmds=%d blob_bytes=%d" % (pd["node"], pd["n_cmds"], pd["bytes"]))
        print("   inner=%s" % pd["inner_ops"])

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
    print("---- PORTABLE NODES (all) ----")
    for e in details:
        if e["where"] == "outer":
            print("PORTABLE name=%s target=%s supported=%s in_table=%s src=%s out=%s"
                  % (e["name"], e["target"], e["supported"], e["in_table"], e.get("src"), e.get("out")))
    print("PEAK_RSS_MB %d" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024))

    pte_path = os.path.join(outdir, "tts.pte")
    with open(pte_path, "wb") as fh:
        manager.to_executorch().write_to_file(fh)
    print("PTE %s bytes=%d" % (pte_path, os.path.getsize(pte_path)))
    print("GRAPH_MUTATION op_nodes_before_to_executorch=%d after=%d" % (nodes_before, len(all_op_nodes(ep))))

    torch.save(inputs[0], os.path.join(outdir, "tts_in0.pt"))
    np.save(os.path.join(outdir, "tts_ref.npy"), ref_np)
    inputs[0].numpy().astype("int64").tofile(os.path.join(outdir, "tts_in0.bin"))
    with open(os.path.join(outdir, "tts_census.json"), "w") as f:
        json.dump({
            "T": T, "scale": scale, "input_shape": list(inputs[0].shape),
            "input_dtype": str(inputs[0].dtype), "ref_shape": list(ref.shape),
            "ref_absmax": float(ref.abs().max()), "n_params": nparam,
            "program_input_spec": sig,
            "enum_from_cpp": {str(k): v for k, v in ENUM.items()},
            "commands": dict(cmds), "delegates": per_delegate,
            "per_target": per, "nodes": details,
        }, f, indent=1, default=str)
    print("WROTE %s" % outdir)


if __name__ == "__main__":
    main()
