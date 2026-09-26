"""Which clause inside the op's OWN gate turned the node away.

The traced verdict names the first reason. For `cat` and `index` the width gate
fires before the op's own gate, so the record so far says only that the width
refused it, which hides whether the op's own gate would have agreed anyway.
This traces INSIDE the real gate function and returns the line that returned
None, so the answer is a clause in the function under test and not a second
copy of it that could disagree.
"""

from __future__ import annotations

import sys

import torch

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition import hexagon_partitioner as HP
from executorch.exir.dialects._ops import ops as exir_ops


def trace_return_line(fn, *args):
    """The line of the `return` that produced this value, and what it returned."""
    code = fn.__code__
    last = {}

    def tracer(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "return":
            last["line"] = frame.f_lineno
            last["arg"] = arg
            return None
        return tracer

    sys.settrace(tracer)
    try:
        value = fn(*args)
    finally:
        sys.settrace(None)
    with open(code.co_filename) as fh:
        lines = fh.readlines()
    line = last.get("line")
    return {
        "function": fn.__qualname__,
        "file": code.co_filename.split("/")[-1],
        "return_line": line,
        "return_source": lines[line - 1].strip() if line else None,
        "returned": repr(last.get("arg"))[:60],
        "result_is_none": value is None,
    }


def own_gate_clause(node, support):
    """The op's own gate, traced to the clause that returned."""
    t = node.target
    if t is exir_ops.edge.aten.cat.default:
        return trace_return_line(hexagon_ops.cat_plan, node)
    if t is exir_ops.edge.aten.index.Tensor:
        return trace_return_line(
            hexagon_ops.gather_table, node, support.is_data_placeholder
        )
    if t in (exir_ops.edge.aten.add.Tensor, exir_ops.edge.aten.sub.Tensor):
        return {
            "function": "_broadcast_fits_dsp_limits",
            "file": "hexagon_partitioner.py",
            "accepts": HP._broadcast_fits_dsp_limits(node),
            "note": "no clause to trace: the function returns a bool",
        }
    return None

