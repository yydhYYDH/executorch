"""Corpus B again, recording the producer each refused node is standing behind.

A getitem the partitioner turns away is usually not refused for anything about
itself: it is refused because the op it reads is. So the census needs both ends
of every island -- the node that holds the graph up, and the shadow it drops --
or a count of shadows reads as a count of work when it is a count of
consequence.
"""
import collections
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/yydh/executorch/.tmp/scratch/portcensus")

import torch
from torch.export import export

import gatecensus as gc
from executorch.backends.hexagon.partition import hexagon_partitioner as hpart
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
)
from executorch.exir import EdgeCompileConfig, to_edge

CFG = EdgeCompileConfig(_check_ir_validity=False)


def edge_program(module, inputs):
    ep = export(module, inputs)
    ep = hpart.HexagonPartitioner().transform_for_pre_decomposition(ep)
    return to_edge(ep, compile_config=CFG).exported_program()


def producer(node):
    for arg in node.all_input_nodes:
        return arg
    return None


def run(module, inputs, label, out):
    _REFUSED = gc._REFUSED
    _REFUSED.clear()
    try:
        program = edge_program(module, inputs)
    except Exception:
        out.setdefault("unexportable", []).append((label, traceback.format_exc()[-300:]))
        return 0, 0
    support = HexagonOperatorSupport(_data_placeholders(program), program)
    calls = 0
    refused = 0
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        calls += 1
        if support.is_node_supported({}, node):
            continue
        refused += 1
        up = producer(node)
        up_line = _REFUSED.get(up) if up is not None else None
        out.setdefault("refused", []).append(
            {
                "label": label,
                "target": gc.name_of(node),
                "line": _REFUSED.get(node),
                "in_table": bool(node.target in hpart.SUPPORTED_TARGETS),
                "up_target": gc.name_of(up) if up is not None else None,
                "up_line": up_line,
            }
        )
    return calls, refused


def qwen3(layers, vocab=151936, hidden=1024, inter=3072, heads=16, kv=8, head_dim=128):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        max_position_embeddings=4096,
        use_cache=False,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).eval().to(torch.float16), (
        torch.randint(0, 1024, (1, 32)),
    )


def main():
    import models as models_mod

    inst = gc.Instrumented()
    inst.install()
    out = {}
    bounds = []
    try:
        for key, builder in models_mod.REGISTRY.items():
            inst.refresh()
            entry = builder()
            if entry is None:
                continue
            calls, refused = run(*entry, label=key, out=out)
            bounds.append({"label": key, "calls": calls, "refused": refused})
            print("done", key, calls, refused, flush=True)
        inst.refresh()
        calls, refused = run(*qwen3(1), label="qwen3_1L", out=out)
        bounds.append({"label": "qwen3_1L", "calls": calls, "refused": refused})
        print("done qwen3_1L", calls, refused, flush=True)
        inst.refresh()
        calls, refused = run(*qwen3(28), label="qwen3_28L", out=out)
        bounds.append({"label": "qwen3_28L", "calls": calls, "refused": refused})
        print("done qwen3_28L", calls, refused, flush=True)
    finally:
        inst.remove()
    out["gate_lines"] = inst.lines
    out["bounds"] = bounds
    with open(os.path.join(HERE, "islands.json"), "w") as f:
        json.dump(out, f, indent=1)
    rows = out.get("refused", [])
    print("refused:", len(rows), "unattributed:", sum(1 for r in rows if r["line"] is None))
    print("calls:", sum(b["calls"] for b in bounds))


if __name__ == "__main__":
    main()
