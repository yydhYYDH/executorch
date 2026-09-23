# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses the masked-row guard around attention logits into one op.

The SDPA these graphs were exported from zeroes every attention row that is
entirely masked, reading the mask off the logits and writing the softmax's
output:

    mask = not(any(not(logits == -inf), -1, keepdim=True))
    out  = where(mask, zeros, softmax_output)

As written that is nine host ops per attention block -- the comparison, both
negations, the reduction, the fill and the select, plus the views and dim-order
wrappers the export puts around them -- and on a phone each costs about a
millisecond of framework time whatever its size: measured over the 24-layer
tower, 97 ms per image, against a whole attention block's worth of DSP time. It
is also what cuts every layer into four partitions, because the partitioner
stops at the first op it has no emitter for.

Rewritten as one op, the guard runs on the DSP and the block stays in one
partition. The rewrite is exact: the kernel walks the same rows, compares the
same value, and copies the same bytes.

Only the exact pattern is rewritten, and a graph that does not carry it comes
back untouched, so a model exported by a different SDPA keeps its own ops.
"""

from typing import Optional, Tuple

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult

NAMESPACE = "et_hexagon"

# rms_norm opens this namespace with a DEF block and only one of those is
# allowed, so this one has to add a fragment to it instead.
_library = torch.library.Library(NAMESPACE, "FRAGMENT")


def _row_guard(mask, values, pad):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    all_pad = (mask == pad).all(dim=-1, keepdim=True)
    return torch.where(all_pad, torch.zeros_like(values), values)


_library.define("row_guard(Tensor mask, Tensor values, float pad) -> Tensor")
_library.impl("row_guard", _row_guard, "CompositeExplicitAutograd")

#: The op the emitter table registers, exactly as the pass inserts it.
ROW_GUARD = exir_ops.edge.et_hexagon.row_guard.default


def masked_row_guard(
    node: torch.fx.Node,
) -> Optional[Tuple[torch.fx.Node, torch.fx.Node, float]]:
    """The mask, values and pad of a masked-row guard, or None if this is not one.

    Structural, not by name alone: every step has to be the one the pattern
    needs, including a keepdim reduction, because without it the mask broadcasts
    along a different axis and the fused op would not mean the same thing. The
    mask and the values are deliberately not required to be the same node: the
    export puts a view between the logits and the comparison, and the two sides
    can come from different subgraphs as long as they describe the same rows.
    """
    if "aten.where" not in str(node.target) or len(node.args) != 3:
        return None
    condition, zeros, values = node.args
    if not all(isinstance(arg, torch.fx.Node) for arg in (condition, zeros, values)):
        return None
    if "aten.full_like" not in str(zeros.target) or zeros.args[0] is not values:
        return None
    fill = zeros.args[1] if len(zeros.args) > 1 else None
    if not isinstance(fill, (int, float)) or float(fill) != 0.0:
        return None
    if "aten.logical_not" not in str(condition.target):
        return None
    any_row = condition.args[0]
    if not isinstance(any_row, torch.fx.Node) or "aten.any" not in str(any_row.target):
        return None
    if len(any_row.args) < 2 or int(any_row.args[1]) != -1:
        return None
    values_value = values.meta.get("val")
    reduced_value = any_row.meta.get("val")
    if values_value is None or reduced_value is None:
        return None
    if tuple(reduced_value.shape) != tuple(values_value.shape[:-1]) + (1,):
        return None
    inner = any_row.args[0]
    if not isinstance(inner, torch.fx.Node) or "aten.logical_not" not in str(
        inner.target
    ):
        return None
    comparison = inner.args[0]
    if not isinstance(comparison, torch.fx.Node) or "aten.eq" not in str(
        comparison.target
    ):
        return None
    if len(comparison.args) < 2:
        return None
    mask, pad = comparison.args[0], comparison.args[1]
    if not isinstance(mask, torch.fx.Node):
        return None
    if not isinstance(pad, (int, float)) or float(pad) != float("-inf"):
        return None
    mask_value = mask.meta.get("val")
    if mask_value is None or tuple(mask_value.shape) != tuple(values_value.shape):
        return None
    return mask, values, float(pad)


class FuseMaskedRowGuard(ExportedProgramPassBase):
    """Replaces every masked-row guard in a program with one op.

    A program without the pattern comes back untouched. Callers that are not a
    pass manager reach the same code through ``FuseMaskedRowGuard()(ep)``.
    """

    def call(self, exported_program) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        matches = [
            (node, match)
            for node in graph.nodes
            if (match := masked_row_guard(node)) is not None
        ]
        if not matches:
            return ExportedProgramPassResult(exported_program, False)

        for node, (mask, values, pad) in matches:
            with graph.inserting_before(node):
                fused = graph.call_function(ROW_GUARD, (mask, values, pad))
            fused.meta["val"] = node.meta["val"]
            node.replace_all_uses_with(fused)
        graph.eliminate_dead_code()
        exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)
