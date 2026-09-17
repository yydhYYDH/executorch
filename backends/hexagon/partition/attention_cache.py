"""Point custom_sdpa's cache operands at the storage they view.

The kernel sizes and indexes a key or value operand as [batch, seq, heads, dim] --
attention_entry.cc strides it by tokens * heads * headDim -- which is the layout
the upstream caller builds by transposing a head-major cache. A model that stores
its cache as [batch, seq, heads, dim] and transposes it anyway hands the op a
head-major view instead, and that view is what the delegate ends up reading.

The transpose is gone by the time a delegate sees its inputs, so the operand has
to be chosen before partitioning, on a graph where the transpose is still
visible. Run use_stored_caches() on the program before to_backend().
"""

from typing import Optional

import torch
from torch.fx import GraphModule, Node


def _swaps_last_two(node: Node) -> bool:
    name = str(getattr(node, "target", ""))
    if node.op != "call_function":
        return False
    if "transpose.int" in name:
        return tuple(node.args[1:3]) in ((1, 2), (2, 1))
    if "permute_copy" in name or "permute.default" in name or "permute." in name:
        dims = tuple(node.args[1])
        return len(dims) == 4 and dims[1] == 2 and dims[2] == 1
    return False


def _stored_cache(arg) -> Optional[Node]:
    """The cache behind a caller's transposed view, or None when there is none.

    Only a tensor the graph already holds can stand in for the view. Anything
    else is left alone: the emitter refuses the delegation rather than compute
    from a layout it cannot identify.
    """
    if not isinstance(arg, Node) or not _swaps_last_two(arg):
        return None
    source = arg.args[0]
    if isinstance(source, Node) and source.op in ("get_attr", "placeholder"):
        return source
    return None


def _graph_module(program) -> GraphModule:
    if isinstance(program, GraphModule):
        return program
    graph_module = getattr(program, "graph_module", None)
    if graph_module is None:
        raise TypeError(f"use_stored_caches: no graph module on {type(program).__name__}")
    return graph_module


def use_stored_caches(program):
    """Rewrite custom_sdpa's cache operands to the seq-major storage itself.

    Accepts an ExportedProgram, an EdgeProgramManager, or a GraphModule, and
    rewrites in place; the program is returned for convenience.
    """
    programs = getattr(program, "_edge_programs", None)
    graph_modules = (
        [_graph_module(p) for p in programs.values()] if programs else [_graph_module(program)]
    )
    for graph_module in graph_modules:
        for node in graph_module.graph.nodes:
            if "custom_sdpa" not in str(getattr(node, "target", "")):
                continue
            args = list(node.args)
            for index in (1, 2):
                if index >= len(args):
                    continue
                source = _stored_cache(args[index])
                if source is not None:
                    args[index] = source
            node.args = tuple(args)
        graph_module.graph.lint()
    return program
