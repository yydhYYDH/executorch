# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A comparison produces a bool, and no kernel here can read one.

`aten.gt.Tensor` and `aten.lt.Tensor` are refused for the plain reason that the
dtype gate admits fp16 and fp32 and a comparison's result is bool: the kernel
would write one element per value where the arena holds two, and the element
count is decided by the graph. What a comparison's *output* then does is the
other half of the question -- `x + (y > 0)` is an ordinary thing to write -- and
it is answered by the promotion torch puts in the way: the bool is converted to
the arena's width by a cast, and `dim_order_keeps_the_bytes` refuses that cast
for a non-fp16/fp32 source, so the comparison and its conversion stay on the
portable kernels and the element-wise op is handed a fp16 tensor.
"""

import os
import pathlib
import sys

import numpy as np
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    operand_dtypes_are_readable,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402


class _Comparison(torch.nn.Module):
    """`x + (y > 0)`, which torch promotes for us."""

    def forward(self, x, y):
        return x + (y > 0.0)


class _ComparisonWithCast(torch.nn.Module):
    """The same thing with the conversion written out."""

    def forward(self, x, y):
        return x + (y > 0.0).to(torch.float16)


def _program(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def test_the_comparison_and_its_conversion_stay_outside_the_delegate():
    """The bool never reaches the arena; the element-wise op still does.

    Both spellings lower the same way, which is the point: whichever the graph
    author writes, the bool is converted outside the delegate and the op that
    consumes it is handed fp16. The delegate still holds the add, so this is not
    a refusal of the whole graph.
    """
    x = torch.randn(4, 8, dtype=torch.float16)
    y = torch.randn(4, 8, dtype=torch.float16)
    for model in (_Comparison(), _ComparisonWithCast()):
        program = _program(model, (x, y))
        outer = {
            str(node.target)
            for node in program.graph_module.graph.nodes
            if node.op == "call_function"
        }
        assert any("gt" in target for target in outer), outer
        assert any("dim_order_copy" in target for target in outer), outer
        calls = _delegates(program)
        assert (
            len(calls) == 1
        ), f"the element-wise op did not reach the delegate: {calls}"
        lowered = program.graph_module.get_submodule(calls[0].args[0].target)
        blob = bytes(lowered._processed_bytes)
        _, commands = read_blob(blob)
        converted = (y > 0.0).to(torch.float16)
        got = np.frombuffer(
            execute(blob, [x.numpy(), converted.numpy()])[0], dtype=np.float16
        )
        expected = x + converted
        assert got.tobytes() == expected.numpy().reshape(-1).tobytes()


def _node_with_operand(dtype, size=(4, 8)):
    """An add whose second operand carries `dtype`."""
    graph = torch.fx.Graph()
    lhs = graph.placeholder("x")
    lhs.meta["val"] = torch.empty(*size, dtype=torch.float16)
    rhs = graph.placeholder("y")
    rhs.meta["val"] = torch.empty(*size, dtype=dtype)
    node = graph.call_function(exir_ops.edge.aten.add.Tensor, args=(lhs, rhs))
    node.meta["val"] = torch.empty(*size, dtype=torch.float16)
    return node


def test_a_bool_operand_is_refused_at_the_gate():
    """The guard, on the node it reads.

    The exporter does not produce this shape today -- it converts the bool first,
    as the test above shows -- but a hand-built graph can, and a kernel reading
    it would take the neighbouring slot's bytes as the value: no error, wrong
    numbers. The control is the same node with a fp16 operand, which is taken.
    """
    assert not operand_dtypes_are_readable(_node_with_operand(torch.bool))
    assert not HexagonOperatorSupport().is_node_supported(
        {}, _node_with_operand(torch.bool)
    )
    assert operand_dtypes_are_readable(_node_with_operand(torch.float16))
