# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses a residual `add` with the RMSNorm that reads it into one DSP command.

Qwen3's decoder layer computes `x = x + branch` and then normalizes `x` for the
next projection. The DSP has one command for both halves --
`DSP_OP_ADD_FUSE_LAYERNORM` -- and the RMSNorm flavor of `htp_ops_add_fuse_layernorm`
adds the two operands, keeps the sum as the new residual, and normalizes it,
all in fp32 accumulation. Left separate the add is one more
`DSP_OP_BINARY_ELEMENTWISE` command per layer, which is the top prefill cost.

The pass runs after `FuseRmsNormPass`, so the pattern it looks for is the fused
`et_hexagon.rms_norm(add(a, b), weight, eps)`. The fused node it builds returns
both tensors the command writes -- the normalized output first, then the residual
sum -- and two getitems name them, because a torch graph can only read one output
at a time. The residual getitem takes over every use of the original add, so the
add disappears entirely rather than being left behind for its other readers.
"""

import operator
from typing import NamedTuple, Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from executorch.backends.hexagon.rms_norm import RMS_NORM, _as_float

NAMESPACE = "et_hexagon"

# rms_norm opens this namespace with a DEF block and only one of those is
# allowed, so this one has to add a fragment to it instead.
_library = torch.library.Library(NAMESPACE, "FRAGMENT")


def _add_rms_norm(residual, branch, weight, eps):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    added = residual + branch
    variance = added.float().pow(2).mean(-1, keepdim=True)
    normalized = (added.float() * torch.rsqrt(variance + eps)).to(added.dtype)
    return normalized * weight, added


_library.define(
    "add_rms_norm(Tensor residual, Tensor branch, Tensor weight, float eps) "
    "-> (Tensor, Tensor)"
)
_library.impl("add_rms_norm", _add_rms_norm, "CompositeExplicitAutograd")

ADD_RMS_NORM = exir_ops.edge.et_hexagon.add_rms_norm.default

ADD = exir_ops.edge.aten.add.Tensor
GETITEM = operator.getitem


class AddRmsNormMatch(NamedTuple):
    # The separate add and norm the fused command replaces.
    add: torch.fx.Node
    norm: torch.fx.Node
    # The operands the kernel wants: the add's two inputs as written, not their
    # unwrapped sources, because their fp16 bytes are what the command reads.
    residual: torch.fx.Node
    branch: torch.fx.Node
    weight: torch.fx.Node
    eps: float
    add_value: torch.Tensor
    norm_value: torch.Tensor


def match_add_rms_norm(norm_node: torch.fx.Node) -> Optional[AddRmsNormMatch]:
    """The residual add read by a fused RMSNorm, anchored on the norm.

    `FuseRmsNormPass` builds `rms_norm(x, weight, eps)` with `x` the tensor the
    decomposition squared, so a residual add shows up as the norm's first
    argument:

        add(residual, branch) -> rms_norm(., weight, eps)   <- anchor

    The kernel adds in fp16 and normalizes in fp32, so both operands and the sum
    have to be fp16 and the same shape; a broadcast operand or a wider add is a
    different command than the one this emits.
    """
    if norm_node.target is not RMS_NORM:
        return None
    if len(norm_node.args) < 3:
        return None
    add, weight, eps_operand = norm_node.args[0], norm_node.args[1], norm_node.args[2]
    if not isinstance(add, torch.fx.Node) or add.target is not ADD:
        return None
    if not isinstance(weight, torch.fx.Node):
        return None
    eps = _as_float(eps_operand)
    if eps is None:
        return None

    # alpha is the only part of torch's add the kernel cannot express, and it
    # defaults to 1. Anything else is a scaled add the command would drop.
    alpha = add.kwargs.get("alpha")
    if alpha is None and len(add.args) > 2:
        alpha = add.args[2]
    if alpha not in (None, 1, 1.0):
        return None

    if len(add.args) < 2:
        return None
    residual, branch = add.args[0], add.args[1]
    if not (isinstance(residual, torch.fx.Node) and isinstance(branch, torch.fx.Node)):
        return None

    add_value = add.meta.get("val")
    norm_value = norm_node.meta.get("val")
    if not isinstance(add_value, torch.Tensor) or not isinstance(
        norm_value, torch.Tensor
    ):
        return None
    if add_value.dtype is not torch.float16 or norm_value.dtype is not torch.float16:
        return None
    if tuple(add_value.shape) != tuple(norm_value.shape) or add_value.dim() < 1:
        return None
    for operand in (residual, branch):
        held = operand.meta.get("val")
        if not isinstance(held, torch.Tensor):
            return None
        if (
            held.dtype is not torch.float16
            or tuple(held.shape) != tuple(add_value.shape)
        ):
            return None
    return AddRmsNormMatch(
        add, norm_node, residual, branch, weight, eps, add_value, norm_value
    )


def fuse_add_rms_norm(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every matched add+norm with one node and returns how many.

    The getitems and the dead norm are left for `eliminate_dead_code` after the
    uses have been moved, because the add may still have readers outside the
    pattern -- the next layer's residual path, for one.
    """
    graph = graph_module.graph
    found = [
        match
        for node in graph.nodes
        if node.op == "call_function"
        and (match := match_add_rms_norm(node)) is not None
    ]

    for match in found:
        with graph.inserting_before(match.add):
            fused = graph.create_node(
                "call_function",
                ADD_RMS_NORM,
                args=(match.residual, match.branch, match.weight, match.eps),
            )
            # Output order is the command's: normalized first, residual sum
            # second, matching the kernel's dst/add_out pointers.
            normalized = graph.create_node("call_function", GETITEM, args=(fused, 0))
            residual = graph.create_node("call_function", GETITEM, args=(fused, 1))
        fused.meta["val"] = (match.norm_value, match.add_value)
        normalized.meta["val"] = match.norm_value
        residual.meta["val"] = match.add_value
        match.norm.replace_all_uses_with(normalized)
        match.add.replace_all_uses_with(residual)

    if found:
        graph.eliminate_dead_code()
        graph_module.recompile()
    return len(found)


class FuseAddRmsNormPass(ExportPass):
    """`fuse_add_rms_norm` as a pass, after `FuseRmsNormPass`.

    The pattern anchors on the fused `et_hexagon.rms_norm`, so the RMSNorm pass
    has to have run already. Like that pass this belongs in the caller's
    `transform_passes`, ahead of the split: `to_backend` checks that a
    partitioner returns the graph it was given, so a backend cannot do it while
    partitioning.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        return PassResult(graph_module, fuse_add_rms_norm(graph_module) > 0)
