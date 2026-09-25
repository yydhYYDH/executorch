# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from executorch.backends.hexagon import hexagon_ops
from blob_interpreter import read_blob
from executorch.backends.hexagon.partition.hexagon_partitioner import HexagonPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export


def _program(model, shape):
    return to_edge_transform_and_lower(
        export(model.half().eval(), (torch.randn(shape).half(),)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _conv_nodes(program):
    return [node for node in program.graph_module.graph.nodes if "convolution" in str(node.target)]


def test_conv1d_is_delegated_and_uses_the_im2col_command():
    program = _program(torch.nn.Conv1d(3, 5, 3, padding=1), (1, 3, 9))
    calls = [node for node in program.graph_module.graph.nodes if node.target is torch.ops.higher_order.executorch_call_delegate]
    assert len(calls) == 1
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert any(
        node.target is hexagon_ops.CONVOLUTION
        for node in lowered.original_module.graph_module.graph.nodes
    )
    commands = read_blob(bytes(lowered._processed_bytes))[1]
    assert [command.type for command in commands] == [24, 3, 12, 3]


def test_a_genuinely_three_dimensional_kernel_is_refused():
    program = _program(torch.nn.Conv3d(3, 5, (3, 3, 3), padding=1), (1, 3, 5, 5, 5))
    node = _conv_nodes(program)[0]
    assert hexagon_ops.conv_spec(node, lambda _: True) is None
    assert not [
        node for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
