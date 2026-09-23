# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrites a patch-embedding convolution into a matmul.

Qwen3-VL cuts an image into patches with a Conv3d whose kernel and stride are
the same triple, which has no padding and one group, so every output element is
an inner product over one disjoint window of the input:

    out[n, o] = sum_k x[n, k] * Wt[k, o] + b[o],   k over in_channels * prod(kernel)

Written that way the node is view -> mm -> add over emitters the DSP already
has, rather than a convolution it has no emitter for and whose five-dimensional
form the portable kernel refuses outright. The rewrite is exact: in fp32 the
two differ by about 2e-06, which is the matmul's own summation order, and the
whole patch embed costs one matmul over the flattened patches.

Call it from `to_edge_transform_and_lower`, as a transform pass, so it runs on
the edge program and before the partitioner sees the graph:

    to_edge_transform_and_lower(
        ep,
        transform_passes=[DecomposePatchEmbed()],
        partitioner=[HexagonPartitioner()],
    )

or, without a pass manager, as `DecomposePatchEmbed()(ep).exported_program`.

Only the exact pattern is rewritten. A stride that is not the kernel, padding,
dilation, groups, a transposed convolution, a rank other than five, an operand
the caller supplies instead of a constant, or an output window wider than one
patch all leave the node alone -- a looser match would silently rewrite models
this was not written for.
"""

from typing import List, NamedTuple, Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.remove_unused_parameters_pass import (
    remove_unused_parameters_pass,
)
from executorch.exir.program._program import lift_constant_tensor_pass
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

# The dialect this pass rewrites. `to_edge` turns the ATen conv3d that
# torch.export produces into this edge convolution, and the transform passes of
# to_edge_transform_and_lower run after that, so this is the name to match.
_CONVOLUTION = exir_ops.edge.aten.convolution.default
_VIEW = exir_ops.edge.aten.view_copy.default
_MM = exir_ops.edge.aten.mm.default
_ADD = exir_ops.edge.aten.add.Tensor

# The ATen convolution targets, matched only to refuse them: a caller who hands
# this pass a program from before to_edge would otherwise get a graph rewritten
# nowhere and no error saying so.
_ATEN_CONVOLUTIONS = frozenset(
    {
        torch.ops.aten.conv3d.default,
        torch.ops.aten.convolution.default,
    }
)


class PatchEmbed(NamedTuple):
    """The operands of one matched convolution, with the weight transposed."""

    source: torch.fx.Node
    bias: torch.fx.Node
    weight: torch.Tensor
    channels: int
    inner: int


def _arg(node: torch.fx.Node, index: int, name: str):
    if index < len(node.args):
        return node.args[index]
    return node.kwargs.get(name)


def _as_ints(value) -> Optional[List[int]]:
    """A convolution argument as python ints, or None when it is not one.

    The strides and extents have to be known at trace time to be judged at all,
    so anything computed is not this pattern.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return list(value)
    return None


def _constant(ep: ExportedProgram, node) -> Optional[torch.Tensor]:
    """The value of an operand the graph does not compute.

    A weight arrives either as an attribute or as a lifted parameter, and only
    those are constants this pass may fold into a transposed copy: an operand
    the caller passes in can be a different tensor on the next call.
    """
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "get_attr":
        value = ep.state_dict.get(str(node.target))
        return value if isinstance(value, torch.Tensor) else None
    if node.op != "placeholder":
        return None
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
            value = table.get(spec.target)
            if isinstance(value, torch.Tensor):
                return value
    return None


def patch_embed(node: torch.fx.Node, ep: ExportedProgram) -> Optional[PatchEmbed]:
    """The operands of this node as a patch embedding, or None.

    Both the support check and the rewrite call this, so the two cannot
    disagree about which convolutions are rewritten.
    """
    if node.op != "call_function" or node.target is not _CONVOLUTION:
        return None
    source = _arg(node, 0, "input")
    weight = _arg(node, 1, "weight")
    bias = _arg(node, 2, "bias")
    stride = _as_ints(_arg(node, 3, "stride"))
    padding = _as_ints(_arg(node, 4, "padding"))
    dilation = _as_ints(_arg(node, 5, "dilation"))
    transposed = _arg(node, 6, "transposed")
    output_padding = _arg(node, 7, "output_padding")
    groups = _arg(node, 8, "groups")
    if not isinstance(source, torch.fx.Node):
        return None
    if stride is None or padding is None or dilation is None:
        return None
    if transposed or groups != 1 or any(padding) or any(d != 1 for d in dilation):
        return None
    output_padding = _as_ints(output_padding)
    if output_padding is not None and any(output_padding):
        return None

    result = node.meta.get("val")
    if result is None or result.dim() != 5:
        return None
    # One window per output element is what makes this a projection: with the
    # stride equal to the kernel the windows tile the input, and every output
    # spatial extent is one patch.
    if any(size != 1 for size in result.shape[2:]):
        return None

    weight_value = _constant(ep, weight)
    bias_value = _constant(ep, bias)
    if weight_value is None or bias_value is None:
        return None
    if list(weight_value.shape[2:]) != stride:
        return None
    channels = int(weight_value.shape[0])
    inner = int(weight_value.shape[1])
    for size in stride:
        inner *= size
    if bias_value.numel() != channels:
        return None
    return PatchEmbed(source, bias, weight_value, channels, inner)


def _val_of_shape(val: torch.Tensor, shape) -> torch.Tensor:
    """A val carrying the shape and dtype a later check reads off this node.

    Only the shape and dtype are ever asked for, so an empty tensor of the same
    kind is enough, and a fake one costs nothing.
    """
    try:
        return val.new_empty(shape)
    except (RuntimeError, TypeError):
        return torch.empty(shape, dtype=val.dtype)


def _rewrite(ep: ExportedProgram, node: torch.fx.Node, match: PatchEmbed) -> None:
    graph = ep.graph_module.graph
    result = node.meta["val"]
    out_shape = list(result.shape)
    # mm wants the contraction last, so the constant is the transposed weight.
    # Leaving a permute in the graph to do it would be a second pass over the
    # weight on the DSP, and the transposed copy is a constant at export time.
    transposed = match.weight.reshape(match.channels, match.inner).t().contiguous()
    name = f"_hexagon_patch_embed_{node.name}"
    ep.graph_module.register_parameter(name, torch.nn.Parameter(transposed))

    with graph.inserting_before(node):
        weight = graph.create_node("get_attr", name)
        weight.meta["val"] = _val_of_shape(result, transposed.shape)
        # The convolution's batch *is* the row count: every output window is
        # one patch. Naming it keeps the val exact when it is symbolic.
        rows = result.shape[0]
        flat = graph.call_function(_VIEW, (match.source, [-1, match.inner]))
        flat.meta["val"] = _val_of_shape(result, (rows, match.inner))
        product = graph.call_function(_MM, (flat, weight))
        product.meta["val"] = _val_of_shape(result, (rows, match.channels))
        shifted = graph.call_function(_ADD, (product, match.bias))
        shifted.meta["val"] = _val_of_shape(result, (rows, match.channels))
        # The convolution's own shape is what its consumers were traced
        # against, so the last window is folded back into it.
        widened = graph.call_function(
            _VIEW, (shifted, [-1, match.channels] + [1] * (len(out_shape) - 2))
        )
        widened.meta["val"] = result

    node.replace_all_uses_with(widened)
    graph.erase_node(node)


class DecomposePatchEmbed(ExportedProgramPassBase):
    """Rewrites every patch-embedding convolution in a program into a matmul.

    A program with no such convolution comes back untouched: an earlier pass in
    the same list has no reason to see it rewritten. Callers that are not a pass
    manager reach the same code through `DecomposePatchEmbed()(ep)`.
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """Rewrites the matches in place, and says whether it found any."""
        graph = exported_program.graph_module.graph
        if any(node.target in _ATEN_CONVOLUTIONS for node in graph.nodes):
            raise RuntimeError(
                "hexagon: DecomposePatchEmbed runs on the edge program, after "
                "to_edge; this program still holds the ATen convolution, so it "
                "would be rewritten nowhere. Pass it through "
                "to_edge_transform_and_lower's transform_passes instead."
            )

        matches = [
            (node, match)
            for node in graph.nodes
            if (match := patch_embed(node, exported_program)) is not None
        ]
        if not matches:
            return ExportedProgramPassResult(exported_program, False)

        for node, match in matches:
            _rewrite(exported_program, node, match)
        # The weight the convolution read is dead now that its transposed copy
        # is the operand of the matmul.
        remove_unused_parameters_pass(exported_program)
        # The transposed weight is an attribute until this turns it into the
        # buffer the pipeline would have made of it a step later; a caller who
        # runs this pass by hand gets the same coherent program that way, and in
        # the pipeline the lift finds nothing left to do.
        lift_constant_tensor_pass(exported_program)
        exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)
