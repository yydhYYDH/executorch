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

from executorch.backends.hexagon.rms_norm import _pick, _source_of
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

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
    # The fp16 multiply the fused op replaces: the gated MLP's `silu(gate) * up`,
    # not the fp32 silu that feeds it.
    replace: torch.fx.Node
    # The operands the kernel wants, in kernel order `a * silu(b)`. Both are the
    # fp16 values underneath the fp32 upcasts.
    lhs: torch.fx.Node
    rhs: torch.fx.Node


def _silu_source(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The fp16 `x` when `node` is the fp32 SiLU `x * sigmoid(x)`, else None.

    `_source_of` strips the cast that writes the activation back down, which
    lands on the fp32 multiply. Its two operands then have to unwrap to the same
    fp16 value, or the expression is a general `a * sigmoid(b)` and no `mul_silu`
    covers it -- the kernel would add a `b` factor that the source never had.
    """
    inner = _source_of(node)
    if inner.target is not exir_ops.edge.aten.mul.Tensor:
        return None
    picked = _pick(inner, _SIGMOID)
    if picked is None:
        return None
    _operand, sigmoid, other = picked
    if not sigmoid.args or not isinstance(other, torch.fx.Node):
        return None
    source = _source_of(other)
    if source is not _source_of(sigmoid.args[0]):
        return None
    return source


def match_mul_silu(node: torch.fx.Node) -> Optional[MulSiluMatch]:
    """The gated MLP, anchored on the fp16 multiply that consumes the activation.

    `silu(gate) * up` reaches the split as the fp32 activation written out and
    cast back down:

        gate -> cast(fp32) -> mul(gate32, sigmoid(gate32)) -> cast(fp16) --.
                                                                          mul -> anchor
        up --------------------------------------------------------------'

    The kernel is `a * silu(b)`, so it stands in for the whole pattern with
    `a = up` and `b = gate`. Anchoring on the fp32 silu instead -- as the first
    version of this pass did -- makes the fused node `gate * silu(gate)`: the
    kernel's `a` becomes `gate` as well, multiplying in a `gate` the source never
    had, because `gate * sigmoid(gate)` is `silu(gate)` and not
    `mul_silu(gate, gate)`.

    A bare `silu(x)` with no gated multiply around it is left alone; the kernel
    has no unary form.
    """
    if node.target is not exir_ops.edge.aten.mul.Tensor or len(node.args) < 2:
        return None
    result = node.meta.get("val")
    if not isinstance(result, torch.Tensor) or result.dtype is not torch.float16:
        return None

    for activation, up in ((node.args[0], node.args[1]), (node.args[1], node.args[0])):
        if not isinstance(activation, torch.fx.Node) or not isinstance(
            up, torch.fx.Node
        ):
            continue
        gate = _silu_source(activation)
        if gate is None:
            continue
        for value in (up, gate):
            held = value.meta.get("val") if hasattr(value, "meta") else None
            if not isinstance(held, torch.Tensor):
                return None
            if held.dtype is not torch.float16 or tuple(held.shape) != tuple(
                result.shape
            ):
                return None
        return MulSiluMatch(node, up, gate)
    return None


def fuse_mul_silu(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every matched gated multiply with one node, returns how many.

    The dead fp32 chain is left for `eliminate_dead_code`, because a node the
    match walked through may still have a user outside the pattern.
    """
    graph = graph_module.graph
    found = [
        (node, match)
        for node in graph.nodes
        if node.op == "call_function" and (match := match_mul_silu(node)) is not None
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
