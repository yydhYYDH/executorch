# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lower the exact per-frame 1x1xK depthwise Conv3d to the DSP's 2-D path."""

from typing import List, NamedTuple, Optional

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.remove_unused_parameters_pass import remove_unused_parameters_pass
from executorch.exir.program._program import lift_constant_tensor_pass
from executorch.exir._program_utils import (
    _get_updated_graph_signature,
    _get_updated_range_constraints,
)
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

_CONVOLUTION = exir_ops.edge.aten.convolution.default
_CONV2D = exir_ops.edge.aten.conv2d.default
_PERMUTE = exir_ops.edge.aten.permute_copy.default
_VIEW = exir_ops.edge.aten.view_copy.default
_ATEN_CONVOLUTIONS = frozenset(
    {torch.ops.aten.conv3d.default, torch.ops.aten.convolution.default}
)


class _FrameConv(NamedTuple):
    source: torch.fx.Node
    weight: torch.fx.Node
    weight_value: torch.Tensor
    bias: Optional[torch.fx.Node]
    shape: List[int]
    output_shape: List[int]
    stride: List[int]
    padding: List[int]
    dilation: List[int]


def _arg(node: torch.fx.Node, index: int, name: str):
    if index < len(node.args):
        return node.args[index]
    return node.kwargs.get(name)


def _as_ints(value) -> Optional[List[int]]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return list(value)
    return None


def _constant(ep: ExportedProgram, node) -> Optional[torch.Tensor]:
    if not isinstance(node, torch.fx.Node):
        return None
    if node.op == "get_attr":
        value = ep.state_dict.get(str(node.target))
        return value if isinstance(value, torch.Tensor) else None
    if node.op != "placeholder":
        return None
    for spec in ep.graph_signature.input_specs:
        if spec.arg.name != node.name or spec.kind not in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
        ):
            continue
        for table in (ep.state_dict, ep.constants):
            value = table.get(spec.target)
            if isinstance(value, torch.Tensor):
                return value
    return None


def _val_of_shape(val: torch.Tensor, shape) -> torch.Tensor:
    try:
        return val.new_empty(tuple(shape))
    except (RuntimeError, TypeError):
        return torch.empty(tuple(shape), dtype=val.dtype)


def _match(node: torch.fx.Node, ep: ExportedProgram) -> Optional[_FrameConv]:
    if node.op != "call_function" or node.target is not _CONVOLUTION:
        return None
    source = _arg(node, 0, "input")
    weight_node = _arg(node, 1, "weight")
    bias = _arg(node, 2, "bias")
    stride = _as_ints(_arg(node, 3, "stride"))
    padding = _as_ints(_arg(node, 4, "padding"))
    dilation = _as_ints(_arg(node, 5, "dilation"))
    transposed = _arg(node, 6, "transposed")
    output_padding = _as_ints(_arg(node, 7, "output_padding"))
    groups = _arg(node, 8, "groups")
    if not isinstance(source, torch.fx.Node) or not isinstance(weight_node, torch.fx.Node):
        return None
    if not isinstance(bias, (torch.fx.Node, type(None))):
        return None
    if stride is None or padding is None or dilation is None:
        return None
    if len(stride) != 3 or len(padding) != 3 or len(dilation) != 3:
        return None
    if transposed or groups is None or not isinstance(groups, int):
        return None
    if output_padding is None or any(output_padding):
        return None
    if any(value <= 0 for value in stride + dilation) or any(
        value < 0 for value in padding
    ):
        return None
    value = source.meta.get("val")
    result = node.meta.get("val")
    weight = _constant(ep, weight_node)
    bias_value = _constant(ep, bias)
    if not all(isinstance(item, torch.Tensor) for item in (value, result, weight)):
        return None
    if bias is not None and bias_value is None:
        return None
    if value.dim() != 5 or result.dim() != 5 or weight.dim() != 5:
        return None
    if any(isinstance(dim, torch.SymInt) for dim in value.shape + result.shape):
        return None
    if value.dtype not in (torch.float16, torch.float32) or weight.dtype != value.dtype:
        return None
    if groups != value.shape[1] or groups != result.shape[1]:
        return None
    if weight.shape[0] != groups or weight.shape[1] != 1:
        return None
    if tuple(weight.shape[2:4]) != (1, 1) or int(weight.shape[4]) <= 1:
        return None
    if stride[0] != 1 or stride[1] != 1 or padding[0] != 0 or padding[1] != 0:
        return None
    if dilation[0] != 1 or dilation[1] != 1:
        return None
    shape = [int(dim) for dim in value.shape]
    output_shape = [int(dim) for dim in result.shape]
    expected_w = (
        shape[4] + 2 * padding[2] - dilation[2] * (int(weight.shape[4]) - 1) - 1
    ) // stride[2] + 1
    if output_shape != [shape[0], groups, shape[2], shape[3], expected_w]:
        return None
    return _FrameConv(
        source,
        weight_node,
        weight,
        bias,
        shape,
        output_shape,
        stride,
        padding,
        dilation,
    )


def _call(graph, target, args, val):
    node = graph.call_function(target, args=tuple(args))
    node.meta["val"] = val
    return node


def _frame_weight(
    ep: ExportedProgram, graph, match: _FrameConv, node: torch.fx.Node
) -> torch.fx.Node:
    name = f"_hexagon_frame_weight_{node.name}"
    value = match.weight_value[:, :, 0, 0, :].unsqueeze(2).contiguous()
    ep.graph_module.register_parameter(name, torch.nn.Parameter(value))
    weight = graph.create_node("get_attr", name)
    weight.meta["val"] = _val_of_shape(match.source.meta["val"], value.shape)
    return weight


def _rewrite(ep: ExportedProgram, node: torch.fx.Node, match: _FrameConv) -> None:
    graph = ep.graph_module.graph
    result = node.meta["val"]
    source_value = match.source.meta["val"]
    n, channels, depth, height, width = match.shape
    out_width = match.output_shape[4]
    with graph.inserting_before(node):
        to_frames = _call(
            graph,
            _PERMUTE,
            (match.source, [0, 2, 1, 3, 4]),
            _val_of_shape(source_value, (n, depth, channels, height, width)),
        )
        frames = _call(
            graph,
            _VIEW,
            (to_frames, [n * depth, channels, height, width]),
            _val_of_shape(source_value, (n * depth, channels, height, width)),
        )
        convolved = _call(
            graph,
            _CONV2D,
            (
                frames,
                _frame_weight(ep, graph, match, node),
                match.bias,
                [1, match.stride[2]],
                [0, match.padding[2]],
                [1, match.dilation[2]],
                channels,
            ),
            _val_of_shape(result, (n * depth, channels, height, out_width)),
        )
        volume = _call(
            graph,
            _VIEW,
            (convolved, [n, depth, channels, height, out_width]),
            _val_of_shape(result, (n, depth, channels, height, out_width)),
        )
        output = _call(
            graph,
            _PERMUTE,
            (volume, [0, 2, 1, 3, 4]),
            result,
        )
    node.replace_all_uses_with(output)
    graph.erase_node(node)
    remove_unused_parameters_pass(ep)
    lift_constant_tensor_pass(ep)
    ep.graph_module.recompile()
    ep._graph_signature = _get_updated_graph_signature(ep.graph_signature, ep.graph_module)
    ep._range_constraints = _get_updated_range_constraints(ep.graph_module)


class DecomposeFrameConv3d(ExportedProgramPassBase):
    """Rewrite a 1x1xK depthwise Conv3d to one DSP-supported Conv2d."""

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        if any(node.target in _ATEN_CONVOLUTIONS for node in graph.nodes):
            raise RuntimeError(
                "hexagon: DecomposeFrameConv3d runs on the edge program, after to_edge; "
                "pass it through to_edge_transform_and_lower's transform_passes instead."
            )
        matches = [
            (node, match)
            for node in graph.nodes
            if (match := _match(node, exported_program)) is not None
        ]
        if not matches:
            return ExportedProgramPassResult(exported_program, False)
        for node, match in matches:
            _rewrite(exported_program, node, match)
        exported_program.graph_module.recompile()
        return ExportedProgramPassResult(exported_program, True)
