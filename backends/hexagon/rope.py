# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses the RoPE decomposition back into one DSP command.

The HF rotary embedding is written out as a rotate-half pair:

    q1 = q[..., :half]
    q2 = q[..., half:]
    rotated = cat([-q2, q1], -1)
    out = q * cos + rotated * sin

which is four multiplies and two adds per tensor, and cuts the graph into pieces
where the DSP has nothing. The DSP has `DSP_OP_ROPE`, which applies the same
rotate-half convention -- `out[x] = in[x] * cos[x] - in[x+half] * sin[x]` and
`out[x+half] = in[x+half] * cos[x+half] + in[x] * sin[x+half]` -- reading a table
whose row is the first half of cos followed by the second half, exactly the
`[seq, head_dim]` table this graph already carries.

The DSP command takes q and k together. The graph applies RoPE to them in
separate partitions, so the fused op is per-tensor and the emitter hands the
same tensor to the command's k operand with `kv_num_head = 0`, which the kernel
skips. That keeps the change inside the graph the exporter already produced.

The fused node is created after `to_edge`, for the same reason the fused norm
is: a fused op built before that pass does not survive the decomposition table.
"""

from typing import NamedTuple, Optional

import torch

from executorch.backends.hexagon.rms_norm import _source_of
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

NAMESPACE = "et_hexagon"

# rms_norm opens this namespace with a DEF block and only one of those is
# allowed, so this one adds a fragment to it instead.
_library = torch.library.Library(NAMESPACE, "FRAGMENT")


def _rope(x, cos, sin):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


_library.define("rope(Tensor x, Tensor cos, Tensor sin) -> Tensor")
_library.impl("rope", _rope, "CompositeExplicitAutograd")

ROPE = exir_ops.edge.et_hexagon.rope.default

_ADD = exir_ops.edge.aten.add.Tensor
_MUL = exir_ops.edge.aten.mul.Tensor
_CAT = exir_ops.edge.aten.cat.default
_NEG = exir_ops.edge.aten.neg.default
_SLICE = exir_ops.edge.aten.slice_copy.Tensor


def _node(value) -> Optional[torch.fx.Node]:
    return value if isinstance(value, torch.fx.Node) else None


def _operand_with(node: torch.fx.Node, target) -> Optional[torch.fx.Node]:
    for arg in node.args[:2]:
        if isinstance(arg, torch.fx.Node) and arg.target is target:
            return arg
    return None


def _slice_span(node: torch.fx.Node):
    """The (source, dim, start, end) a slice_copy carries, or None.

    `end` stays as written -- None is the open end -- because the checks below
    want to tell `[0, half)` and `[half, None)` apart.
    """
    if node.target is not _SLICE or len(node.args) < 4:
        return None
    source, dim, start, end = node.args[:4]
    if not isinstance(source, torch.fx.Node):
        return None
    if not isinstance(dim, int) or isinstance(dim, bool):
        return None
    if not isinstance(start, int) or isinstance(start, bool):
        return None
    if end is not None and (not isinstance(end, int) or isinstance(end, bool)):
        return None
    return source, dim, start, end


class RopeMatch(NamedTuple):
    # The add the fused op replaces.
    anchor: torch.fx.Node
    # The tensor the rotation is applied to, and the two tables it rotates by.
    base: torch.fx.Node
    cos: torch.fx.Node
    sin: torch.fx.Node


def match_rope(anchor: torch.fx.Node) -> Optional[RopeMatch]:
    """The rotate-half embedding, anchored on the add that ends it.

        base -> slice -> neg -> cat([-hi, lo]) -> mul(cat, sin)  \\
        base -----------------------------------> mul(base, cos)  -> add  <- anchor

    The two multiplies are the only shape an add of two muls can take here; the
    rotate-half one is told apart by the cat on one of its operands, and the
    slices that cat consumes have to be the two halves of the other multiply's
    own base.
    """
    if anchor.target is not _ADD or len(anchor.args) < 2:
        return None
    left, right = _node(anchor.args[0]), _node(anchor.args[1])
    if left is None or right is None:
        return None
    if left.target is not _MUL or right.target is not _MUL:
        return None

    left_cat = _operand_with(left, _CAT)
    right_cat = _operand_with(right, _CAT)
    if (left_cat is None) == (right_cat is None):
        return None
    rotated, plain = (left, right) if left_cat is not None else (right, left)
    cat = left_cat if left_cat is not None else right_cat

    tensors = cat.args[0] if cat.args else None
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 2:
        return None
    neg = [t for t in tensors if isinstance(t, torch.fx.Node) and t.target is _NEG]
    positive = [
        t for t in tensors if isinstance(t, torch.fx.Node) and t.target is _SLICE
    ]
    if len(neg) != 1 or len(positive) != 1:
        return None
    hi_node = _node(neg[0].args[0]) if neg[0].args else None
    if hi_node is None:
        return None
    hi = _slice_span(hi_node)
    lo = _slice_span(positive[0])
    if hi is None or lo is None:
        return None
    hi_source, hi_dim, hi_start, hi_end = hi
    lo_source, lo_dim, lo_start, lo_end = lo
    if hi_source is not lo_source or hi_dim != lo_dim:
        return None

    result = anchor.meta.get("val")
    if not isinstance(result, torch.Tensor):
        return None
    rank = result.dim()
    head_dim = int(result.shape[-1])
    half = head_dim // 2
    if head_dim <= 0 or head_dim % 2 or half <= 0:
        return None
    dim = hi_dim + rank if hi_dim < 0 else hi_dim
    if dim != rank - 1:
        return None
    # rotate_half is exactly `[-base[half:], base[:half]]`. The open end of the
    # second slice is materialized as the largest int rather than None.
    if lo_start != 0 or lo_end != half:
        return None
    if hi_start != half or (hi_end is not None and hi_end < head_dim):
        return None

    base = lo_source
    plain_operands = [_node(arg) for arg in plain.args[:2]]
    if None in plain_operands:
        return None
    cos = None
    base_operand = None
    for arg in plain_operands:
        if _source_of(arg) is _source_of(base):
            base_operand = arg
        else:
            cos = arg
    if cos is None or base_operand is None:
        return None

    sin = None
    for arg in (_node(rotated.args[0]), _node(rotated.args[1])):
        if arg is not None and arg is not cat:
            sin = arg
    if sin is None or cos is sin:
        return None

    base_value = base.meta.get("val")
    if not isinstance(base_value, torch.Tensor):
        return None
    if base_value.dtype is not torch.float16:
        return None
    if base_value.dim() not in (3, 4):
        return None
    # The kernel indexes the sequence as the leading axis and walks heads inside
    # a token, so a batch axis can only be there if it is one wide.
    if base_value.dim() == 4 and int(base_value.shape[0]) != 1:
        return None
    if int(base_value.shape[-1]) != head_dim:
        return None
    for trig in (cos, sin):
        trig_value = trig.meta.get("val")
        if not isinstance(trig_value, torch.Tensor):
            return None
        if trig_value.dtype is not torch.float16:
            return None
        # The kernel reads one rope-dim row of the table per token.
        if int(trig_value.shape[-1]) != head_dim:
            return None
    return RopeMatch(anchor, base_operand, cos, sin)


def fuse_rope(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every matched embedding with one node and returns how many.

    The dead elementwise chain is left for `eliminate_dead_code`, because a node
    the match walked through may still have a user outside the pattern.
    """
    graph = graph_module.graph
    found = [
        (node, match)
        for node in graph.nodes
        if node.op == "call_function" and (match := match_rope(node)) is not None
    ]

    for _anchor, match in found:
        with graph.inserting_before(match.anchor):
            fused = graph.create_node(
                "call_function",
                ROPE,
                args=(match.base, match.cos, match.sin),
            )
        fused.meta["val"] = match.anchor.meta["val"]
        match.anchor.replace_all_uses_with(fused)

    if found:
        graph.eliminate_dead_code()
        graph_module.recompile()
    return len(found)


class FuseRopePass(ExportPass):
    """`fuse_rope` as a pass, which is the only place it can live.

    `to_backend` checks that a partitioner returns the graph it was given, so a
    backend cannot do this while partitioning. It belongs in the caller's
    `transform_passes`, ahead of the split.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        return PassResult(graph_module, fuse_rope(graph_module) > 0)
