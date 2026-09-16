# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses the RMSNorm decomposition back into one DSP command.

`norm.py` computes the norm in fp32 and casts the result back. That is the right
arithmetic, but it leaves a mean, an rsqrt and a multiply per layer with no DSP
kernel behind them, which cuts the graph into pieces at every layer. The DSP
does the same work in one command -- `htp_ops_layer_norm` with its RMSNorm flag,
fp16 in and out, accumulating in fp32 -- so this keeps the arithmetic and drops
the fragmentation.

The weight is left alone. The kernel wants gamma as fp32, and a weight reaches a
subgraph as an fp16 placeholder rather than a constant, so folding it in would
read half-precision bytes as floats. The scale keeps its own `mul`, which is
fp16 once this has run and already has a kernel.

The fused node is created after `to_edge`, which is the only place it can be:
the edge decomposition table takes `aten.rms_norm` back apart, so a fused norm
built before that pass does not survive it.
"""

from typing import NamedTuple, Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

NAMESPACE = "et_hexagon"

_library = torch.library.Library(NAMESPACE, "DEF")


def _rms_norm(x, eps):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


_library.define("rms_norm(Tensor x, float eps) -> Tensor")
_library.impl("rms_norm", _rms_norm, "CompositeExplicitAutograd")

RMS_NORM = exir_ops.edge.et_hexagon.rms_norm.default

# Edge keeps two spellings of the same cast: aten's, and the dim-order variant
# the LLM export runs. Which one appears depends on the path that built the graph.
_CASTS = frozenset(
    {
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.aten.to.dtype,
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
    }
)


def _source_of(node: torch.fx.Node) -> torch.fx.Node:
    """The value a chain of dtype casts was made from.

    `RMSNorm.forward` upcasts to fp32 and back, so the tensor the fused op wants
    sits underneath the casts. A copy that reshapes is not a cast and ends the
    walk, which matters because the dim-order variant can do both.
    """
    while node.target in _CASTS:
        source = node.args[0]
        cast, value = node.meta.get("val"), source.meta.get("val")
        if cast is None or value is None or cast.shape != value.shape:
            break
        node = source
    return node


def _pick(node: torch.fx.Node, target) -> Optional[tuple]:
    """Splits a binary node into the operand that unwraps to `target` and the rest.

    Returns the operand both as written and with its casts removed, because the
    replacement happens at the written one -- that is the node whose dtype the
    fused op has to take -- while the match is made against the unwrapped one.
    """
    if len(node.args) < 2:
        return None
    for candidate, other in ((node.args[0], node.args[1]), (node.args[1], node.args[0])):
        if isinstance(candidate, torch.fx.Node) and _source_of(candidate).target is target:
            return candidate, _source_of(candidate), other
    return None


def _as_float(value) -> Optional[float]:
    """A literal epsilon, whether it stayed a python number or became a tensor."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, torch.fx.Node):
        held = value.meta.get("val")
        if isinstance(held, torch.Tensor) and held.numel() == 1:
            return float(held.item())
    return None


class RmsNormMatch(NamedTuple):
    # The fp16 node the fused op takes over from, which is the last value the
    # fp32 chain produces and the thing the weight scale is applied to.
    replace: torch.fx.Node
    source: torch.fx.Node
    eps: float


def match_rms_norm(anchor: torch.fx.Node) -> Optional[RmsNormMatch]:
    """The decomposition `RMSNorm.forward` leaves, anchored on the weight scale.

    `RMSNorm.forward` ends in `output * weight`, so that last multiply is the
    anchor and the rest hangs off its first operand:

        x -> cast(fp32) -> x32
        x32 -> mul(x32, x32) -> mean(-1, keepdim) -> add(eps) -> rsqrt -> rstd
        mul(x32, rstd) -> cast(fp16) -> mul(weight)   <- anchor
    """
    scaled = _pick(anchor, exir_ops.edge.aten.mul.Tensor)
    if scaled is None:
        return None
    norm_cast, times_rstd, _weight = scaled
    if norm_cast.target not in _CASTS:
        return None

    with_rstd = _pick(times_rstd, exir_ops.edge.aten.rsqrt.default)
    if with_rstd is None:
        return None
    rsqrt, _unused, source_used = with_rstd

    added = rsqrt.args[0] if rsqrt.args else None
    if not isinstance(added, torch.fx.Node) or added.target is not (
        exir_ops.edge.aten.add.Tensor
    ):
        return None
    with_mean = _pick(added, exir_ops.edge.aten.mean.dim)
    if with_mean is None:
        return None
    mean, _unused, eps_operand = with_mean

    eps = _as_float(eps_operand)
    if eps is None:
        return None

    if len(mean.args) < 3 or list(mean.args[1]) != [-1] or not mean.args[2]:
        return None
    squared = mean.args[0]
    if not isinstance(squared, torch.fx.Node):
        return None
    if squared.target is exir_ops.edge.aten.mul.Tensor:
        if squared.args[0] is not squared.args[1]:
            return None
    elif squared.target is exir_ops.edge.aten.pow.Tensor_Scalar:
        if squared.args[1] not in (2, 2.0):
            return None
    else:
        return None

    # The square and the final multiply have to read the same tensor, or this is
    # something that only looks like a norm.
    source = _source_of(squared.args[0])
    if source is not _source_of(source_used):
        return None
    return RmsNormMatch(norm_cast, source, eps)


def fuse_rms_norm(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every matched norm with one node and returns how many.

    The dead fp32 chain is left for `eliminate_dead_code`, because a node the
    match walked through may still have a user outside the pattern.
    """
    graph = graph_module.graph
    found = [
        (node, match)
        for node in graph.nodes
        if node.op == "call_function"
        and (match := match_rms_norm(node)) is not None
    ]

    for _anchor, match in found:
        with graph.inserting_before(match.replace):
            fused = graph.create_node(
                "call_function",
                RMS_NORM,
                args=(match.source, match.eps),
            )
        fused.meta["val"] = match.replace.meta["val"]
        match.replace.replace_all_uses_with(fused)

    if found:
        graph.eliminate_dead_code()
        graph_module.recompile()
    return len(found)


class FuseRmsNormPass(ExportPass):
    """`fuse_rms_norm` as a pass, which is the only place it can live.

    `to_backend` checks that a partitioner returns the graph it was given, so a
    backend cannot do this while partitioning. It belongs in the caller's
    `transform_passes`, ahead of the split.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        return PassResult(graph_module, fuse_rms_norm(graph_module) > 0)
