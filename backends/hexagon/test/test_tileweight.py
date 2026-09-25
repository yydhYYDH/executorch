# Copyright (c) Meta Platforms, Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from blob_interpreter import execute, read_blob

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from torch.export import export

CONFIG = EdgeCompileConfig(_check_ir_validity=False)
OUT_SHAPE = (2, 3, 5, 7)


class _UpperShape:
    @staticmethod
    def upper_shape(shape):
        return tuple(int(extent) for extent in shape)


class _Multiply(torch.nn.Module):
    def forward(self, x, weight):
        return x * weight


def _integer_tensor(shape):
    numel = int(np.prod(shape))
    return (torch.arange(numel) % 11 - 5).to(torch.float16).reshape(shape)


def _mul_program(weight_shape, out_shape=OUT_SHAPE):
    x = _integer_tensor(out_shape)
    weight = _integer_tensor(weight_shape)
    program = to_edge(
        export(_Multiply(), (x, weight)), compile_config=CONFIG
    ).exported_program()
    node = next(
        node for node in program.graph_module.graph.nodes if node.op == "call_function"
    )
    return program, node, x, weight


@pytest.mark.parametrize(
    "weight_shape,strides",
    (
        ((2, 3, 5, 7), (105, 35, 7, 1)),
        ((1, 3, 5, 7), (0, 35, 7, 1)),
        ((1, 1, 5, 7), (0, 0, 7, 1)),
        ((1, 1, 1, 7), (0, 0, 0, 1)),
        ((1, 1, 1, 1), (0, 0, 0, 0)),
    ),
)
def test_stride_rule_accepts_zero_through_four_singleton_axes(weight_shape, strides):
    got = hexagon_ops._broadcast_strides(weight_shape, OUT_SHAPE, _UpperShape())
    assert got == list(strides)


@pytest.mark.parametrize(
    "weight_shape",
    (
        (2, 3, 5, 7),
        (1, 3, 5, 7),
        (1, 1, 5, 7),
        (1, 1, 1, 7),
        (1, 1, 1, 1),
    ),
)
def test_supported_broadcasts_emit_one_named_binary_and_match_fp64(weight_shape):
    program, node, x, weight = _mul_program(weight_shape)
    lowered = to_edge_transform_and_lower(
        program,
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    delegates = [
        node
        for node in lowered.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(delegates) == 1
    delegate = lowered.graph_module.get_submodule(delegates[0].args[0].target)
    inner_names = {
        node.name for node in delegate.original_module.graph_module.graph.nodes
    }
    assert node.name in inner_names
    blob = bytes(delegate._processed_bytes)
    commands = read_blob(blob)[1]
    names = {
        value: key
        for key, value in vars(hexagon_ops).items()
        if key.startswith("DSP_OP_")
    }
    assert [names[command.type] for command in commands] == [
        "DSP_OP_BINARY_ELEMENTWISE"
    ]
    assert len(commands) == 1
    got = np.frombuffer(execute(blob, (x.numpy(), weight.numpy()))[0], dtype=np.float16)
    expected = (x.double() * weight.double()).half().numpy().reshape(-1)
    assert np.array_equal(got, expected)


def _forced_metadata_program(weight_shape, out_shape):
    valid_shape = (1,) * len(out_shape)
    program, node, _, _ = _mul_program(valid_shape, out_shape)
    node.args[1].meta["val"] = torch.empty(weight_shape, dtype=torch.float16)
    node.meta["val"] = torch.empty(out_shape, dtype=torch.float16)
    return program, node


@pytest.mark.parametrize(
    "weight_shape,out_shape",
    (
        ((1, 3, 5, 2), (2, 3, 5, 8)),
        ((1, 1, 2, 2), (2, 3, 8, 8)),
        ((1, 2, 2, 2), (2, 3, 8, 8)),
        ((2, 1, 2, 1), (4, 3, 8, 8)),
        ((3, 3, 5, 7), (2, 3, 5, 7)),
        ((2, 3), (3,)),
    ),
)
def test_non_singleton_mismatch_stays_refused_without_a_delegate(
    weight_shape, out_shape
):
    program, node = _forced_metadata_program(weight_shape, out_shape)
    with pytest.raises(
        ValueError, match="neither singleton nor output-sized|more axes than its output"
    ):
        hexagon_ops._broadcast_strides(weight_shape, out_shape, _UpperShape())
    support = HexagonOperatorSupport(_data_placeholders(program))
    assert not support.is_node_supported({}, node)
    result = HexagonPartitioner().partition(program)
    assert result.partition_tags == {}

    valid_shape = (1,) * len(out_shape)
    valid_program, valid_node, _, _ = _mul_program(valid_shape, out_shape)
    valid_support = HexagonOperatorSupport(_data_placeholders(valid_program))
    assert valid_support.is_node_supported({}, valid_node)
    assert len(HexagonPartitioner().partition(valid_program).partition_tags) == 1
