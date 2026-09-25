# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from blob_interpreter import read_blob

from executorch.backends.hexagon.decompose_conv3d import DecomposeFrameConv3d
from executorch.backends.hexagon.partition.hexagon_partitioner import HexagonPartitioner
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


def _program(model, shape, transform=True):
    kwargs = {"transform_passes": [DecomposeFrameConv3d()]} if transform else {}
    return to_edge_transform_and_lower(
        export(model.half().eval(), (torch.randn(shape).half(),)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        **kwargs,
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _lowered(program):
    (delegate,) = _delegates(program)
    return program.graph_module.get_submodule(delegate.args[0].target)


def _frame_model(channels=3, width=3):
    model = torch.nn.Conv3d(
        channels, channels, (1, 1, width), groups=channels, bias=True
    )
    with torch.no_grad():
        model.weight.copy_(
            torch.arange(channels * width, dtype=torch.float16).reshape_as(model.weight)
            % 5
            - 2
        )
        model.bias.copy_(torch.arange(channels, dtype=torch.float16) - 1)
    return model


def test_frame_conv3d_is_rewritten_to_dsp_conv2d():
    program = _program(_frame_model(), (1, 3, 2, 4, 9))
    lowered = _lowered(program)
    nodes = lowered.original_module.graph_module.graph.nodes
    assert not any(node.target is exir_ops.edge.aten.convolution.default for node in nodes)
    assert any(node.target is exir_ops.edge.aten.conv2d.default for node in nodes)
    command_types = [command.type for command in read_blob(bytes(lowered._processed_bytes))[1]]
    assert command_types == [3, 3, 2, 3, 3]


@pytest.mark.parametrize("width", [63, 64, 65])
def test_frame_conv3d_width_boundary(width):
    program = _program(_frame_model(), (1, 3, 2, 4, width))
    lowered = _lowered(program)
    nodes = lowered.original_module.graph_module.graph.nodes
    assert any(node.target is exir_ops.edge.aten.conv2d.default for node in nodes)
    assert [command.type for command in read_blob(bytes(lowered._processed_bytes))[1]] == [
        3,
        3,
        2,
        3,
        3,
    ]


def test_frame_conv3d_rewrite_matches_torch():
    model = _frame_model().half().eval()
    x = torch.randn(1, 3, 2, 4, 9, dtype=torch.float16)
    ep = to_edge(export(model, (x,))).exported_program()
    result = DecomposeFrameConv3d()(ep).exported_program
    args = []
    for node in result.graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        if node.name == "p_bias":
            args.append(model.bias.detach())
        elif "lifted" in node.name:
            args.append(model.weight.detach()[:, :, 0, 0, :].unsqueeze(2))
        else:
            args.append(x)
    actual = result.graph_module(*args)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert torch.equal(actual, model(x))


@pytest.mark.parametrize(
    "shape,kwargs",
    [
        ((1, 3, 2, 4, 9), {"kernel_size": (1, 1, 3), "stride": (2, 1, 1)}),
        ((1, 3, 2, 4, 9), {"kernel_size": (1, 1, 3), "dilation": (2, 1, 1)}),
        ((1, 3, 2, 4, 9), {"kernel_size": (1, 1, 3), "padding": (1, 0, 0)}),
        ((1, 3, 2, 4, 9), {"kernel_size": (1, 3, 3)}),
        ((1, 3, 4, 5, 5), {"kernel_size": (3, 3, 3)}),
        ((1, 3, 4, 5, 5), {"kernel_size": (3, 3, 3), "stride": (2, 1, 1)}),
        ((1, 3, 7, 9, 9), {"kernel_size": (3, 3, 3), "dilation": (2, 1, 1)}),
        ((1, 3, 4, 5, 5), {"kernel_size": (3, 3, 3), "padding": (1, 1, 1)}),
    ],
)
def test_non_frame_conv3d_is_refused(shape, kwargs):
    model = torch.nn.Conv3d(3, 3, groups=3, **kwargs)
    program = _program(model, shape)
    assert not _delegates(program)


def test_frame_conv3d_refuses_dense_channels():
    model = torch.nn.Conv3d(3, 3, (1, 1, 3))
    program = _program(model, (1, 3, 2, 4, 9))
    assert not _delegates(program)


def test_pass_refuses_a_raw_aten_program():
    model = torch.nn.Conv3d(3, 3, (1, 1, 3), groups=3)
    with pytest.raises(RuntimeError, match="edge program"):
        DecomposeFrameConv3d()(export(model, (torch.randn(1, 3, 2, 4, 9),)))
