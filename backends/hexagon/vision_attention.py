# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses a vision tower's bidirectional attention into one DSP command.

`htp_ops_vision_attention_fp16` is the DSP's unmasked, non-causal attention: it
reads query, key and value as `[batch, tokens, heads, headDim]` -- heads inside
a token's row, not a separate axis -- and writes the result in the same layout
(`attention_entry.cc:36-41,57`). A vision tower computes exactly that, but an
export writes it as the usual decomposition: split the heads out with a view and
a transpose, `bmm(q, k^T) * scale`, `softmax`, `bmm(probs, v)`, transpose back.
Lowered as it stands, a ViT block is three partitions: the projections, the
attention core (two BATCH_MATMULs and a SOFTMAX), and the output projection --
with all four head transposes left on the portable kernels, because a permute at
a partition boundary is not something the backend can keep.

The fused op takes the three `[batch, seq, heads, headDim]` tensors the head
splits were built from and states the attention once. The layout is the reason
the op exists rather than an emitter on the softmax: the operands the DSP wants
are the tensors *under* the transposes, so the transposes have to be part of the
pattern that gets replaced and cannot be left in the graph.

Only the exact pattern is rewritten. A mask, a causal flag, a row guard, a scale
that is not a constant, operands whose head split is anything but a view
followed by a `(0, 2, 1, 3)` transpose, or three operands that do not agree on
their shape all leave the node alone -- a looser match would compute a different
function. In particular a masked attention never matches: the mask reaches the
DSP as an absent operand here, and `attention_entry.cc:49` reads a mask only
when it is given a positive stride, so binding one is the silent wrong answer
the FLASH_ATTN emitter already refuses to commit.

The layout also has to be the arena's. The arena holds row-major two-byte
elements, so the fused op's operands are contiguous tensors whose *shape* is the
layout the kernel walks; a graph that already holds `[batch, heads, seq, dim]`
in that order is a different function and is left alone.
"""

from typing import NamedTuple, Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

NAMESPACE = "et_hexagon"

# rms_norm opens this namespace with a DEF block and only one of those is
# allowed, so this one has to add a fragment to it instead.
_library = torch.library.Library(NAMESPACE, "FRAGMENT")


def _vision_attention(q, k, v, scale):
    """The same attention, written out for anywhere the DSP cannot reach.

    The operands are `[batch, seq, heads, headDim]`; `transpose(1, 2)` is what
    turns one into the `[batch, heads, seq, headDim]` torch's matmul wants, and
    the last transpose is what turns the result back.
    """
    query, key, value = (tensor.transpose(1, 2).float() for tensor in (q, k, v))
    scores = (query @ key.transpose(-1, -2)) * scale
    weights = scores.softmax(dim=-1)
    return (weights @ value).transpose(1, 2).to(q.dtype)


_library.define("vision_attention(Tensor q, Tensor k, Tensor v, float scale) -> Tensor")
_library.impl("vision_attention", _vision_attention, "CompositeExplicitAutograd")

#: The op the emitter table registers, exactly as the pass inserts it.
VISION_ATTENTION = exir_ops.edge.et_hexagon.vision_attention.default

_VIEW = exir_ops.edge.aten.view_copy.default
_EXPAND = exir_ops.edge.aten.expand_copy.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default
_BMM = exir_ops.edge.aten.bmm.default
_SOFTMAX = exir_ops.edge.aten._softmax.default
_MUL_TENSOR = exir_ops.edge.aten.mul.Tensor
_MUL_SCALAR = exir_ops.edge.aten.mul.Scalar
_DIV_TENSOR = exir_ops.edge.aten.div.Tensor

#: The multipliers between the logits and the softmax. The constant is on either
#: side of a mul, and a division says the same thing the other way round.
_MULS = frozenset({_MUL_TENSOR, _MUL_SCALAR})
_DIVISIONS = frozenset({_DIV_TENSOR})

#: The copies an export puts around a scalar constant. They name a layout for a
#: tensor with one element, so following one to its source keeps the value.
_DIM_ORDER_COPIES = frozenset(
    {
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
        exir_ops.edge.dim_order_ops._clone_dim_order.default,
    }
)

#: The two scalars `attention_entry.cc` is handed: query tokens and key tokens
#: are the same `tokens`, and the scale reaches the kernel as one float.
_TRANSPOSE_HEADS = (0, 2, 1, 3)
_TRANSPOSE_LAST = (0, 1, 3, 2)
_TRANSPOSE_INNER = (0, 2, 3, 1)


class VisionAttention(NamedTuple):
    """One matched attention, with the node the fused op replaces."""

    #: `[batch, seq, heads, headDim]`, which is what the DSP reads.
    query: torch.fx.Node
    key: torch.fx.Node
    value: torch.fx.Node
    #: The transpose that puts the heads back inside the token row. Replacing
    #: this node keeps the result in the layout the graph already expects, and
    #: takes the three head splits with it, since nothing else reads them.
    output: torch.fx.Node
    #: The multiplier the graph applies to the logits, as the kernel wants it.
    scale: float
    shape: tuple


def _value(node):
    """The tensor a node stands for, or None."""
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _shape(node):
    value = _value(node)
    return None if value is None else tuple(value.shape)


def _order(node) -> Optional[tuple]:
    """The axis order a transpose names, or None."""
    if not node.args or len(node.args) < 2:
        return None
    order = node.args[1]
    if not isinstance(order, (list, tuple)):
        return None
    return tuple(order)


def _copies_nothing(node: torch.fx.Node) -> bool:
    """Whether this node re-reads its operand's elements in the same order.

    Three forms say nothing about the values. An expand that widens no axis is
    the export putting a batch axis back on an operand it had squeezed out for
    `bmm`. A dim-order copy names a layout rather than giving strides, and the
    arena is row-major, so only the identity order is the same bytes --
    `hexagon_ops.dim_order_keeps_the_bytes` accepts the same ones. A view of a
    contiguous tensor is the other half of that: row-major order is fixed by the
    element count, so any shape with the same count reads the same elements,
    which is how the export merges the batch and head axes back into one `bmm`
    batch. A view whose operand is not contiguous is a transpose-and-copy and
    leaves the walk.
    """
    if node.target in _DIM_ORDER_COPIES:
        return _dim_order_is_identity(node)
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return False
    here, there = _shape(node), _shape(source)
    if here is None or there is None or len(here) == 0 or len(there) == 0:
        return False
    if node.target is _EXPAND:
        return here == there
    if node.target is _VIEW:
        value = _value(source)
        return (
            value is not None
            and value.is_contiguous()
            and value.numel() == _value(node).numel()
        )
    return False


def _dim_order_is_identity(node: torch.fx.Node) -> bool:
    """Whether a dim-order copy leaves the element order alone.

    The layout the arena holds is row-major, so the only order a copy can name
    and still be the same bytes is the axes in their own order -- which is also
    what an omitted argument means.
    """
    value = _value(node)
    source = node.args[0] if node.args else None
    if value is None or not isinstance(source, torch.fx.Node):
        return False
    if (
        value.dim() != len(_shape(source) or ())
        or value.numel() != _value(source).numel()
    ):
        return False
    order = node.kwargs.get("dim_order")
    return order is None or list(order) == list(range(value.dim()))


def _underneath(node):
    """The node a run of value-preserving copies was made from."""
    while isinstance(node, torch.fx.Node) and _copies_nothing(node):
        node = node.args[0]
    return node


def _reaches(node, target, hops: int = 6):
    """The first `target` this value runs into through value-preserving copies.

    The export wraps each matmul operand in a squeeze-shaped view, so the node a
    walk wants is under one or two copies. A value with two readers is not on a
    chain this can follow, and neither is one that meets something other than a
    copy.
    """
    for _ in range(hops):
        if node.target is target:
            return node
        if len(node.users) != 1:
            return None
        node = next(iter(node.users))
        if node.target is target:
            return node
        if not _copies_nothing(node) or len(node.users) != 1:
            return None
    return None


def _head_split(node) -> Optional[torch.fx.Node]:
    """The `[batch, seq, heads, headDim]` tensor under one head transpose.

    `view(B, S, H, D).transpose(1, 2)` is how a projection's output becomes a
    head-major attention operand, and the view is the tensor the DSP wants.
    """
    node = _underneath(node)
    if not isinstance(node, torch.fx.Node) or node.target is not _PERMUTE:
        return None
    if _order(node) != _TRANSPOSE_HEADS:
        return None
    source = node.args[0]
    shape = _shape(source)
    if shape is None or len(shape) != 4:
        return None
    return source


def _transposed_key(node) -> Optional[torch.fx.Node]:
    """The `[batch, seq, heads, headDim]` key under `k.transpose(-2, -1)`.

    Both spellings reach the same place: the last two axes swapped on the
    head-major operand, or the interleave that does both transposes in one.
    """
    node = _underneath(node)
    if not isinstance(node, torch.fx.Node) or node.target is not _PERMUTE:
        return None
    order = _order(node)
    if order == _TRANSPOSE_INNER:
        source = node.args[0]
        shape = _shape(source)
        return source if shape is not None and len(shape) == 4 else None
    if order == _TRANSPOSE_LAST:
        return _head_split(node.args[0])
    return None


def _dot(node) -> Optional[torch.fx.Node]:
    """The operand of a binary op that is a computed tensor rather than a scalar.

    A scalar multiplier can be on either side, so the side that carries a
    non-empty shape is the one to keep walking from.
    """
    if not isinstance(node, torch.fx.Node):
        return None
    shape = _shape(node)
    return node if shape is not None and len(shape) > 0 else None


def _constant_scalar(ep: Optional[ExportedProgram], node) -> Optional[float]:
    """The float a constant operand holds, or None when it is not a constant.

    `torch.export` lifts a python float that multiplies a tensor into a buffer,
    so the value is in the program's own tables rather than in the graph.
    """
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return float(node)
    if not isinstance(node, torch.fx.Node):
        return None
    # A half-precision export wraps the lifted scalar in a dim-order copy, which
    # for a one-element tensor is the value itself.
    while (
        node.target in _DIM_ORDER_COPIES
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
        and (value := _value(node)) is not None
        and value.numel() == 1
    ):
        node = node.args[0]
    value = None
    if node.op == "get_attr":
        value = ep.state_dict.get(str(node.target)) if ep is not None else None
    elif node.op == "placeholder" and ep is not None:
        for spec in ep.graph_signature.input_specs:
            if spec.arg.name != node.name:
                continue
            if spec.kind not in (
                InputKind.PARAMETER,
                InputKind.BUFFER,
                InputKind.CONSTANT_TENSOR,
            ):
                return None
            for table in (ep.state_dict, ep.constants):
                candidate = table.get(spec.target)
                if isinstance(candidate, torch.Tensor):
                    value = candidate
                    break
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.reshape(-1)[0])
    return None


def _scaled_logits(ep, node):
    """The logits under the logits-times-scale, and what that scale was."""
    if node.target in _MULS:
        left, right = node.args[0], node.args[1]
        tensor, scalar = (
            (left, right)
            if _dot(left) is not None and _dot(right) is None
            else (right, left)
        )
        if _dot(tensor) is None:
            return None
        scale = _constant_scalar(ep, scalar)
        return None if scale is None else (tensor, float(scale))
    if node.target in _DIVISIONS:
        left, right = node.args[0], node.args[1]
        scalar = _constant_scalar(ep, right)
        if _dot(left) is None or scalar is None or scalar == 0.0:
            return None
        return left, 1.0 / float(scalar)
    return None


def _tail(node, shape: tuple) -> Optional[torch.fx.Node]:
    """The transpose that puts the heads back inside a token row.

    The matmul's result is `[batch, heads, seq, dim]`, and the graph turns it
    back into `[batch, seq, heads, dim]` -- under a value-preserving copy,
    because the export had to squeeze the batch axis out to reach `bmm`.
    """
    for _ in range(4):
        if node.target is _PERMUTE and _order(node) == _TRANSPOSE_HEADS:
            return node if _shape(node) == shape else None
        if len(node.users) != 1:
            return None
        node = next(iter(node.users))
        if node.target is _PERMUTE:
            continue
        if not _copies_nothing(node) or len(node.users) != 1:
            return None
    return None


def vision_attention_pattern(
    node: torch.fx.Node, exported_program: Optional[ExportedProgram] = None
) -> Optional[VisionAttention]:
    """The operands of one decomposed vision attention, or None.

    The anchor is the softmax, and every step backwards has to be the one the
    pattern needs: a `bmm` whose operands are head splits of two tensors that
    agree on their shape, a scale that is a constant this can read, and a second
    `bmm` reading the probabilities. Anything else -- a mask added to the
    logits, a causal flag, a row guard around the softmax -- is not this
    function and is not rewritten.
    """
    if node.op != "call_function" or node.target is not _SOFTMAX:
        return None
    if len(node.args) < 2:
        return None
    axis = node.args[1]
    if not isinstance(axis, int) or axis not in (-1, 3):
        return None
    probabilities = node.meta.get("val")
    if not isinstance(probabilities, torch.Tensor) or probabilities.dim() != 4:
        return None
    batch, heads, query_len, key_len = tuple(probabilities.shape)
    # The token count is the one dimension an export may leave symbolic, so it is
    # a symbol here as often as it is a number; the rest have to be numbers,
    # because the emitter hands them to the command as params.
    for size in (batch, heads, query_len, key_len):
        if isinstance(size, torch.SymInt):
            continue
        if not isinstance(size, int) or size <= 0:
            return None

    logits = node.args[0]
    if not isinstance(logits, torch.fx.Node):
        return None
    scale = 1.0
    scored = _scaled_logits(exported_program, logits)
    if scored is not None:
        logits, scale = scored
    dot = _underneath(logits)
    if not isinstance(dot, torch.fx.Node) or dot.target is not _BMM:
        return None

    query = _head_split(dot.args[0])
    key = _transposed_key(dot.args[1])
    if query is None or key is None:
        return None
    shape = _shape(query)
    if shape is None or shape != _shape(key):
        return None
    # The probabilities are `[batch, heads, tokens, tokens]` and the operands
    # are the same attention read token-major, so the head count and the token
    # count have to be the ones the kernel is handed -- and the kernel has a
    # single `tokens` for both sides, which is what a square query x key run is.
    if query_len != key_len:
        return None
    if (shape[0], shape[1], shape[2]) != (batch, query_len, heads):
        return None

    second = _reaches(node, _BMM)
    if second is None or len(second.args) < 2:
        return None
    value = _head_split(second.args[1])
    if value is None or _shape(value) != shape:
        return None

    output = _tail(second, shape)
    if output is None:
        return None
    return VisionAttention(query, key, value, output, scale, shape)


class FuseVisionAttention(ExportedProgramPassBase):
    """Replaces every decomposed vision attention in a program with one op.

    A program without the pattern comes back untouched. Callers that are not a
    pass manager reach the same code through ``FuseVisionAttention()(ep)``.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        matches = [
            (node, match)
            for node in graph.nodes
            if (match := vision_attention_pattern(node, exported_program)) is not None
        ]
        if not matches:
            return ExportedProgramPassResult(exported_program, False)

        for _node, match in matches:
            with graph.inserting_before(match.output):
                fused = graph.call_function(
                    VISION_ATTENTION, (match.query, match.key, match.value, match.scale)
                )
            # The transpose this replaces had already materialized, so the result
            # is contiguous and the copies after it stay aliases.
            fused.meta["val"] = match.output.meta["val"]
            match.output.replace_all_uses_with(fused)
        # The head splits and both matmul wrappers have no readers now: the walk
        # went through them, and the fused op reads their inputs directly.
        graph.eliminate_dead_code()
        exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)
