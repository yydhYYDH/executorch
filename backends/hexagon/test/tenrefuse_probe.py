"""Read the actual refusal reason for the ten nodes on supported rows.

The 28-layer Qwen3 census counted `aten.add` 3, `aten.sub` 2, `aten.cat` 2,
`aten.index` 2 and `aten.cumsum` 1 as portable. A bucket count records no
reasons, so this asks the partitioner itself: every `is_node_supported` call is
wrapped in a line tracer, and the line the verdict frame returns from is the
clause that refused the node. Nothing here re-implements a gate.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import torch
from torch.export import export
from transformers import Qwen3Config, Qwen3ForCausalLM

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition import hexagon_partitioner as HP
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonOperatorSupport,
    HexagonPartitioner,
    _data_placeholders,
)
from executorch.backends.hexagon.hexagon_backend import SUPPORTED_TARGETS, EMITTERS
from executorch.exir import to_edge_transform_and_lower
from blob_interpreter import read_blob

import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from second_opinion import own_gate_report  # noqa: E402
from own_clause import own_gate_clause  # noqa: E402

DELEGATE = torch.ops.higher_order.executorch_call_delegate

WATCHED = {"aten::add", "aten::sub", "aten::cat", "aten::index", "aten::cumsum"}

#: Set to "all" to watch every target in the emitter table rather than the five
#: the census named, which is how the sweep asks whether the ten are the whole
#: set of supported-row refusals or a sample of one.
WATCH = os.environ.get("PROBE_WATCH", "five")


def target_name(node) -> str:
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        return schema.name
    return (
        getattr(node.target, "_name", None)
        or getattr(node.target, "__name__", None)
        or str(node.target)
    )


def is_delegate(node) -> bool:
    return node.target is DELEGATE or "call_delegate" in str(node.target)


def val_of(arg):
    if isinstance(arg, torch.fx.Node):
        v = arg.meta.get("val")
        if isinstance(v, torch.Tensor):
            return v
    return None


def shape_repr(v) -> str:
    if v is None:
        return "None"
    return "%s %s contig=%s" % (list(v.shape), str(v.dtype).replace("torch.", ""), v.is_contiguous())


def arg_repr(node) -> str:
    out = []
    for a in node.args:
        if isinstance(a, torch.fx.Node):
            out.append("%s:%s[%s]" % (a.name, target_name(a), shape_repr(val_of(a))))
        elif isinstance(a, (list, tuple)):
            inner = []
            for x in a:
                if isinstance(x, torch.fx.Node):
                    inner.append("%s:%s" % (x.name, shape_repr(val_of(x))))
                else:
                    inner.append(repr(x)[:60])
            out.append("[" + ", ".join(inner) + "]")
        else:
            out.append(repr(a)[:60])
    return " ; ".join(out)


class RefusalTrace:
    """Record the clause line each refusal came from.

    A line tracer rather than a second copy of the gates: the question is which
    `if` in the real `_verdict` turned the node away, and a re-implementation
    could disagree with the one under test.
    """

    def __init__(self) -> None:
        self.verdict_code = HexagonOperatorSupport._verdict.__code__
        self.records: dict = {}
        self.calls = 0

    def __enter__(self):
        self._orig = HexagonOperatorSupport.is_node_supported
        trace = self

        def wrapped(support, submodules, node):
            trace.calls += 1
            name = target_name(node)
            watched = node.op == "call_function" and (
                name in WATCHED if WATCH == "five" else node.target in SUPPORTED_TARGETS
            )
            lines = []
            if watched:

                def tracer(frame, event, arg):
                    if frame.f_code is trace.verdict_code:
                        if event in ("line", "return"):
                            lines.append(frame.f_lineno)
                        if event == "return":
                            return None
                    return tracer

                sys.settrace(tracer)
            try:
                verdict = trace._orig(support, submodules, node)
            finally:
                if watched:
                    sys.settrace(None)
            if watched:
                rec = trace.records.setdefault(
                    id(node),
                    {
                        "name": node.name,
                        "target": name,
                        "in_supported_targets": node.target in SUPPORTED_TARGETS,
                        "in_emitters": node.target in EMITTERS,
                    },
                )
                # Which of the partitioner's two censuses counted this node:
                # a REFUSAL is a supported row turned away, an UNWIRED entry is
                # a target with no emitter at all. The ten cannot be read without
                # telling them apart, because the second kind is not a gate.
                rec["args"] = arg_repr(node)
                rec["_node"] = node
                rec["out"] = shape_repr(val_of(node))
                rec["calls"] = rec.get("calls", 0) + 1
                rec["verdict"] = verdict
                if not verdict and lines:
                    rec["refuse_line"] = lines[-1]
            return verdict

        HexagonOperatorSupport.is_node_supported = wrapped
        return self

    def __exit__(self, *exc):
        HexagonOperatorSupport.is_node_supported = self._orig
        return False


def line_text(n) -> str:
    try:
        with open(HP.__file__) as fh:
            lines = fh.readlines()
        return lines[int(n) - 1].strip()
    except Exception:
        return "?"


def build_model(layers, vocab, hidden, intermediate):
    heads = 16
    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads // 2,
        head_dim=hidden // heads,
        max_position_embeddings=32768,
        tie_word_embeddings=True,
        use_cache=False,
    )
    return Qwen3ForCausalLM(config).half().eval()


def main():
    layers = int(os.environ.get("CENSUS_LAYERS", "28"))
    geometry = {
        "layers": layers,
        "vocab": int(os.environ.get("CENSUS_VOCAB", "151936")),
        "hidden": int(os.environ.get("CENSUS_HIDDEN", "1024")),
        "intermediate": int(os.environ.get("CENSUS_INTERMEDIATE", "3072")),
    }
    torch.manual_seed(11)
    model = build_model(**geometry)
    ids = torch.tensor([[17, 91, 5]], dtype=torch.int64)
    exported = export(model, (ids,))

    HP.reset_refused_overload_census()
    HP.reset_unwired_overload_census()
    trace = RefusalTrace()
    with trace:
        lowered = to_edge_transform_and_lower(
            exported, partitioner=[HexagonPartitioner()]
        ).exported_program()
    refused_census = HP.refused_overload_census()
    unwired_census = HP.unwired_overload_census()

    # The traced verdict is the first reason. This asks each refused node\'s OWN
    # gate directly, so a width that excuses a node on its own is not confused
    # with a width that happens to arrive first in front of a second problem.
    support = HexagonOperatorSupport(
        frozenset(_data_placeholders(exported)), exported
    )
    for rec in trace.records.values():
        node = rec.get("_node")
        if node is not None and not rec["verdict"]:
            rec.update(own_gate_report(node, support))
            clause = own_gate_clause(node, support)
            if clause is not None:
                rec["own_gate_clause"] = clause

    gm = lowered.graph_module
    graph = gm.graph
    names = {v: k for k, v in vars(hexagon_ops).items() if k.startswith("DSP_OP_")}
    commands = Counter()
    dsp_nodes = 0
    delegates = []
    portable = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        (delegates if is_delegate(node) else portable).append(node)
    for node in delegates:
        sub = gm.get_submodule(str(node.args[0].target))
        _header, ops = read_blob(bytes(sub.processed_bytes))
        for op in ops:
            commands[names.get(op.type, str(op.type))] += 1
        inner = sub.original_module.graph_module.graph
        dsp_nodes += sum(1 for n in inner.nodes if n.op == "call_function")

    portable_targets = Counter(target_name(n) for n in portable)
    census_buckets = {k: portable_targets.get(k, 0) for k in WATCHED}

    # Which watched node ended up in a delegate, by name: the lowered inner
    # graph keeps the outer node names, so this is a lookup and not a guess by
    # target. Without it a refusal is indistinguishable from a node the
    # partitioner accepted that grouping then did not place.
    inner_names = set()
    for node in delegates:
        sub = gm.get_submodule(str(node.args[0].target))
        inner_names.update(
            n.name for n in sub.original_module.graph_module.graph.nodes
        )

    out = os.environ.get(
        "PROBE_OUT", "/home/yydh/executorch/.tmp/hex_workstreams/tenrefuse/probe.json"
    )
    records = []
    for rec in trace.records.values():
        r = {k: v for k, v in rec.items() if k != "_node"}
        r["refuse_source"] = line_text(rec["refuse_line"]) if "refuse_line" in rec else None
        r["in_delegate"] = rec["name"] in inner_names
        records.append(r)
    records.sort(key=lambda r: (r["target"], not r["verdict"], r["name"]))
    payload = {
        "geometry": geometry,
        "boundary": {
            "lowered_call_nodes": len(portable) + len(delegates),
            "delegates": len(delegates),
            "portable": len(portable),
            "dsp_call_nodes_inside_delegates": dsp_nodes,
            "commands": dict(commands),
            "portable_targets": dict(portable_targets),
        },
        "census_buckets_recomputed": census_buckets,
        "refused_overload_census": refused_census,
        "unwired_overload_census": unwired_census,
        "support_calls": trace.calls,
        "watched_records": records,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
