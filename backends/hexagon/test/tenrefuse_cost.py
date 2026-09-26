"""What the int64 position arithmetic costs the backend today.

The census counted these nodes; the partitioner explains them. Neither says what
they COST. A portable node in the middle of a delegate region is not one wasted
node, it is a region boundary, so the quantity that matters is how many
boundaries the int64 prologue puts between the fp16 work around it.

This measures it on the same lowered graph: for each refused int64 node, whether
its nearest supported neighbour on either side is in a delegate, which is what
makes it a boundary rather than an island.
"""

from __future__ import annotations

import json
import os

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition import hexagon_partitioner as HP
from executorch.backends.hexagon.partition.hexagon_partitioner import HexagonPartitioner
from executorch.exir import to_edge_transform_and_lower

DELEGATE = torch.ops.higher_order.executorch_call_delegate

WATCHED = {"aten::add", "aten::sub", "aten::cat", "aten::index", "aten::cumsum"}


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


def is_delegate(n) -> bool:
    return n.target is DELEGATE or "call_delegate" in str(n.target)


def main():
    ids = torch.tensor([[17, 91, 5]], dtype=torch.int64)
    model = build_model(
        layers=int(os.environ.get("CENSUS_LAYERS", "28")),
        vocab=int(os.environ.get("CENSUS_VOCAB", "151936")),
        hidden=int(os.environ.get("CENSUS_HIDDEN", "1024")),
        intermediate=int(os.environ.get("CENSUS_INTERMEDIATE", "3072")),
    )
    exported = torch.export.export(model, (ids,))
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[HexagonPartitioner()]
    ).exported_program()
    graph = lowered.graph_module.graph

    order = {n: i for i, n in enumerate(graph.nodes)}
    rows = []
    for n in graph.nodes:
        if n.op != "call_function" or is_delegate(n):
            continue
        t = getattr(getattr(n.target, "_schema", None), "name", None)
        if t not in WATCHED:
            continue
        before = after = None
        for other in graph.nodes:
            if other.op != "call_function" or is_delegate(other):
                continue
            if order[other] < order[n] and (before is None or order[other] > order[before]):
                before = other
            if order[other] > order[n] and (after is None or order[other] < order[after]):
                after = other
        rows.append(
            {
                "target": t,
                "name": n.name,
                "dtype": str(HP._dtype_of(n)),
                "prev_portable": before.name if before is not None else None,
                "next_portable": after.name if after is not None else None,
                "users": [u.name for u in n.users if u.op == "call_function"],
            }
        )
    print(json.dumps({"portable_watched": rows}, indent=2))


if __name__ == "__main__":
    main()
