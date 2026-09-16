# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses the fp32 gated activation back into one DSP command.

Qwen3's MLP writes SiLU out as `x * sigmoid(x)`, and the model does it in fp32
the way norm.py does its norm in fp32. The DSP has both halves -- sigmoid is a
unary op and `HTP_OPS_BINARY_MUL_SILU` is `a * silu(b)` -- but not as two fp32
steps: the kernels are fp16 and `htp_ops_cast` has no FP32 conversion at all, so
a delegated cast to fp32 has nothing to call. Writing the activation out costs
five nodes per layer and cuts the graph there.

The fused node is created after `to_edge`, for the same reason the fused norm
is: a fused op built before that pass does not survive the decomposition table.
"""

from typing import NamedTuple, Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from executorch.backends.hexagon.rms_norm import _CASTS, _pick, _source_of

NAMESPACE = "et_hexagon"

# rms_norm opens this namespace with a DEF block and only one of those is
# allowed, so this one has to add a fragment to it instead. That makes the
# import order in hexagon_ops.py load-bearing.
_library = torch.library.Library(NAMESPACE, "FRAGMENT")


def _mul_silu(a, b):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    return a * torch.nn.functional.silu(b)


_library.define("mul_silu(Tensor a, Tensor b) -> Tensor")
_library.impl("mul_silu", _mul_silu, "CompositeExplicitAutograd")

MUL_SILU = exir_ops.edge.et_hexagon.mul_silu.default

_SIGMOID = exir_ops.edge.aten.sigmoid.default


class MulSiluMatch(NamedTuple):
    # The fp16 node the fused op stands in for, which is the cast that brings the
    # activation back down after the fp32 round trip.
    replace: torch.fx.Node
    # The operands the kernel wants: the fp16 values underneath the casts, not
    # the fp32 ones the multiplication was written against.
    lhs: torch.fx.Node
    rhs: torch.fx.Node


def match_mul_silu(node: torch.fx.Node) -> Optional[MulSiluMatch]:
    """The gated activation, anchored on the fp16 cast that ends it.

    `mul(a, sigmoid(b))` with both operands unwrapping to fp16 through casts:

        a -> cast(fp32) -> mul(a32, sigmoid(b32)) -> cast(fp16)   <- anchor
        b -> cast(fp32) -> sigmoid

    `a` and `b` may be the same tensor, which is how `silu(x)` is written out,
    or differ, which is the gated form. Both shapes have to match the result: the
    kernel's broadcast path is not part of this.
    """
    if node.target not in _CASTS:
        return None
    result = node.meta.get("val")
    if not isinstance(result, torch.Tensor) or result.dtype is not torch.float16:
        return None

    product = node.args[0] if node.args else None
    if not isinstance(product, torch.fx.Node):
        return None
    if product.target is not exir_ops.edge.aten.mul.Tensor:
        return None

    picked = _pick(product, _SIGMOID)
    if picked is None:
        return None
    _gated_operand, sigmoid, other = picked
    if not sigmoid.args:
        return None

    lhs, rhs = _source_of(other), _source_of(sigmoid.args[0])
    for value in (lhs, rhs):
        held = value.meta.get("val") if hasattr(value, "meta") else None
        if not isinstance(held, torch.Tensor):
            return None
        if held.dtype is not torch.float16 or tuple(held.shape) != tuple(result.shape):
            return None
    return MulSiluMatch(node, lhs, rhs)


def fuse_mul_silu(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every matched activation with one node and returns how many.

    The dead fp32 chain is left for `eliminate_dead_code`, because a node the
    match walked through may still have a user outside the pattern.
    """
    graph = graph_module.graph
    found = [
        (node, match)
        for node in graph.nodes
        if node.op == "call_function"
        and (match := match_mul_silu(node)) is not None
    ]

    for _anchor, match in found:
        with graph.inserting_before(match.replace):
            fused = graph.create_node(
                "call_function",
                MUL_SILU,
                args=(match.lhs, match.rhs),
            )
        fused.meta["val"] = match.replace.meta["val"]
        match.replace.replace_all_uses_with(fused)

    if found:
        graph.eliminate_dead_code()
        graph_module.recompile()
    return len(found)


class FuseMulSiluPass(ExportPass):
    """`fuse_mul_silu` as a pass, which is the only place it can live.

    `to_backend` checks that a partitioner returns the graph it was given, so a
    backend cannot do this while partitioning. It belongs in the caller's
    `transform_passes`, ahead of the split.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        return PassResult(graph_module, fuse_mul_silu(graph_module) > 0)
