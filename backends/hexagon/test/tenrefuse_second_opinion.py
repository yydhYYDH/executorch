"""Is the clause that refused each node the one it would refuse for anyway?

The ten nodes all land on one line of `_verdict`. That line is the width gate,
which is checked before the op's own geometry. So the reason recorded is the
FIRST reason, and the question it cannot answer on its own is whether the op's
own gate would also have refused -- that is what separates "the width alone
excuses this" from "the width is hiding a second problem the row would have
stated anyway".

This re-runs each refused node's own gate directly on the node, with the width
gate out of the way, and reports what that gate says. It is a second opinion,
not a replacement for the traced verdict: the traced verdict is what the
partitioner did, and this only says what it would have done next.
"""

from __future__ import annotations

import json

import torch
from executorch.backends.hexagon.partition import hexagon_partitioner as HP
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonOperatorSupport,
    _data_placeholders,
)
from executorch.exir.dialects._ops import ops as exir_ops

INDEX_TENSOR = exir_ops.edge.aten.index.Tensor
CAT = exir_ops.edge.aten.cat.default


def own_gate(node, support):
    """The op's own gate, asked directly. None means the op has no such gate."""
    t = node.target

    if t is exir_ops.edge.aten.cat.default:
        from executorch.backends.hexagon.hexagon_ops import cat_plan
        return "cat_plan", cat_plan(node) is not None
    if t is INDEX_TENSOR:
        from executorch.backends.hexagon.hexagon_ops import gather_table
        return "gather_table", gather_table(node, support.is_data_placeholder) is not None
    if t in HP.BINARY_TARGETS:
        return "_broadcast_fits_dsp_limits", HP._broadcast_fits_dsp_limits(node)
    return None, None


def width_gate_says(support, node) -> bool:
    """The width gate alone, evaluated on this node."""
    return HP._dtype_of(node) in (torch.float16, torch.float32)


def operand_width_gate(support, node) -> bool:
    from executorch.backends.hexagon.hexagon_ops import operand_dtypes_are_readable
    return operand_dtypes_are_readable(node)


def report(node, support):
    name, verdict = own_gate(node, support)
    return {
        "target": getattr(getattr(node.target, "_schema", None), "name", str(node.target)),
        "node": node.name,
        "result_dtype": str(HP._dtype_of(node)),
        "width_gate_accepts": width_gate_says(support, node),
        "operand_width_gate_accepts": operand_width_gate(support, node),
        "own_gate": name,
        "own_gate_accepts": verdict,
    }


def own_gate_report(node, support):
    return report(node, support)


if __name__ == "__main__":
    raise SystemExit("import-only; driven by probe.py")
