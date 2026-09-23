# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turns the KV-cache advance into one node the DSP can be asked for.

`llama.update_cache` arrives wrapped in `auto_functionalized_v2`, with one
`getitem` reading the updated cache back. That wrapper is not delegable, and
claiming only the `getitem` is worse than leaving both on the CPU: the
unsupported node becomes a subgraph boundary, so partitions that used to merge
stop merging. Measured on Qwen3-0.6B, 86 subgraphs became 114 and the
non-delegated count rose from 1359 to 1443, while the cache advance itself did
not move at all.

Rewriting the pair into one node removes the wrapper instead of stranding it:
the fused node has no unsupported operand, and `eliminate_dead_code` drops the
wrapper along with the scalar read whose only user it was.

The fused node is spelled as a function of the old cache rather than as a
mutation. The graph already surfaces the updated cache as an output, and the
delegate lowers this by copying that cache out and writing the new rows into the
copy, so a mutation would only add a second path for the same bytes.
"""

from typing import NamedTuple, Optional

import torch
from executorch.backends.hexagon.rms_norm import _library
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


def _update_cache(cache, value, position):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    result = cache.clone()
    start = int(position.reshape(-1)[0].item())
    rows = value.shape[-3]
    result[..., start : start + rows, :, :] = value
    return result


# One TORCH_LIBRARY defines a namespace; rms_norm.py owns this one, and both
# op definitions go into it.
_library.define("update_cache(Tensor cache, Tensor value, Tensor position) -> Tensor")
_library.impl("update_cache", _update_cache, "CompositeExplicitAutograd")


@torch.library.register_fake("et_hexagon::update_cache")
def _update_cache_fake(cache, value, position):
    """The shape never depends on the position, so tracing must not read it.

    Without this the fallback is the reference implementation above, which
    reads the position to size its slice and cannot do that on a symbol.
    """
    return torch.empty_like(cache)


UPDATE_CACHE = exir_ops.edge.et_hexagon.update_cache.default


class KvCacheMatch(NamedTuple):
    cache: torch.fx.Node
    value: torch.fx.Node
    position: torch.fx.Node


def match_kv_cache(node: torch.fx.Node) -> Optional[KvCacheMatch]:
    """`getitem[1]` of `auto_functionalized_v2(llama.update_cache, ...)`.

    The operands arrive as kwargs, with the mutated tensors listed separately in
    `_all_bases`; the position is a scalar by then, and the one-element tensor
    it was read from is what the fused node takes, because that is what the
    arena can hand the DSP.
    """
    if node.op != "call_function" or "getitem" not in str(node.target):
        return None
    if len(node.args) < 2 or node.args[1] != 1:
        return None
    auto = node.args[0]
    if not isinstance(auto, torch.fx.Node) or "auto_functionalized" not in str(
        auto.target
    ):
        return None
    if not auto.args or "update_cache" not in str(auto.args[0]):
        return None

    kwargs = auto.kwargs
    bases = kwargs.get("_all_bases") or []
    index = kwargs.get("_cache_base_index", 0)
    value = kwargs.get("value")
    pos = kwargs.get("start_pos")
    if not bases or not 0 <= index < len(bases):
        return None
    cache = bases[index]
    if not all(isinstance(part, torch.fx.Node) for part in (cache, value, pos)):
        return None
    if not pos.args or not isinstance(pos.args[0], torch.fx.Node):
        return None

    position = pos.args[0]
    if position.meta["val"].dtype not in (torch.int32, torch.int64):
        return None
    if cache.meta["val"].dtype != torch.float16:
        return None
    if value.meta["val"].dtype != torch.float16:
        return None
    return KvCacheMatch(cache, value, position)


def fuse_kv_cache(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every cache advance with one node and returns how many."""
    graph = graph_module.graph
    found = [
        (node, match)
        for node in graph.nodes
        if node.op == "call_function" and (match := match_kv_cache(node)) is not None
    ]

    for node, match in found:
        with graph.inserting_before(node):
            fused = graph.create_node(
                "call_function",
                UPDATE_CACHE,
                args=(match.cache, match.value, match.position),
            )
        fused.meta["val"] = node.meta["val"]
        node.replace_all_uses_with(fused)

    if found:
        # The wrapper and the scalar read are unreachable once nothing reads
        # their results, and leaving them in would defeat the whole exercise.
        graph.eliminate_dead_code()
        graph_module.recompile()
    return len(found)


class FuseKvCachePass(ExportPass):
    """`fuse_kv_cache` as a pass, which is the only place it can live.

    `to_backend` checks that a partitioner returns the graph it was given, so a
    backend cannot do this while partitioning. It belongs in the caller's
    `transform_passes`, ahead of the split.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        return PassResult(graph_module, fuse_kv_cache(graph_module) > 0)
