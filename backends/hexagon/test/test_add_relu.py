# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`relu(x + y)` as one command, and only when the caller asks for the fusion.

The kernel's element-wise op type 8 is `max(a + b, 0)`, and no ATen op carries
it: the graph writes an add and a rectifier, and there is no unary relu on the
DSP for the second one. `FuseAddReluPass` rewrites the pair into the single node
the emitter table has the subtype for; these tests pin the rewrite, the command
it produces, its numbers, and the fact that without the pass the two nodes stay
two nodes.
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
from executorch.backends.hexagon.add_relu import (  # noqa: E402
    fuse_add_relu,
    FuseAddReluPass,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.export import export  # noqa: E402

#: DSP_OP_BINARY_ELEMENTWISE and HTP_OPS_BINARY_ADD_RELU.
_BINARY = 19
_ADD_RELU = 8
_ADD = 1


class _RectifiedSum(torch.nn.Module):
    def forward(self, x, y):
        return torch.nn.functional.relu(x + y)


class _RectifiedSumWithReader(torch.nn.Module):
    """The sum read twice: once rectified, once not."""

    def forward(self, x, y):
        total = x + y
        return torch.nn.functional.relu(total) + total


def _program(model, inputs, passes=()):
    return to_edge_transform_and_lower(
        export(model, inputs),
        transform_passes=list(passes),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _commands(program):
    calls = _delegates(program)
    assert len(calls) == 1, f"the fused node did not reach one delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    blob = bytes(lowered._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def _run(blob, inputs):
    return np.frombuffer(
        execute(blob, [x.numpy() for x in inputs])[0], dtype=np.float16
    )


def _edge(model, inputs):
    """The edge graph on its own, which is where a transform pass runs."""
    return to_edge(
        export(model, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _call_targets(program):
    return [
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def test_the_pass_rewrites_the_pair_into_one_node():
    """The rewrite itself, and that running it twice changes nothing."""
    x = torch.randn(4, 8, dtype=torch.float16)
    program = _edge(_RectifiedSum(), (x, x))
    assert any("aten.relu.default" in target for target in _call_targets(program))
    assert fuse_add_relu(program.graph_module) == 1
    targets = _call_targets(program)
    assert any("et_hexagon.add_relu.default" in target for target in targets), targets
    assert not any("aten.relu.default" in target for target in targets), targets
    assert fuse_add_relu(program.graph_module) == 0


def test_a_rectified_sum_becomes_one_element_wise_command():
    """The subtype, the operands, and torch's own numbers."""
    x = torch.randn(4, 8, dtype=torch.float16)
    y = torch.randn(4, 8, dtype=torch.float16)
    blob, commands = _commands(
        _program(_RectifiedSum(), (x, y), passes=[FuseAddReluPass()])
    )
    assert [command.type for command in commands] == [_BINARY]
    assert list(commands[0].params[:8]) == [32, 32, 32, _ADD_RELU, 2, 2, 0, 0]
    got = _run(blob, (x, y))
    expected = torch.nn.functional.relu(x + y)
    assert got.tobytes() == expected.numpy().reshape(-1).tobytes()


def test_the_rectifier_stays_on_the_host_without_the_pass():
    """The pass is the caller's, so this is what a graph gets without it.

    The add still reaches the DSP -- it is an element-wise op like any other --
    and the rectifier runs on the host, which is a round trip through the arena
    for the same arithmetic the fused command does in one step.
    """
    x = torch.randn(4, 8, dtype=torch.float16)
    blob, commands = _commands(_program(_RectifiedSum(), (x, x)))
    assert [command.type for command in commands] == [_BINARY]
    assert commands[0].params[3] == _ADD
    assert any(
        "aten.relu.default" in target
        for target in _call_targets(_program(_RectifiedSum(), (x, x)))
    )


def test_an_add_with_another_reader_keeps_it():
    """Only the rectifier's use is replaced; the sum stays for everyone else.

    `eliminate_dead_code` would drop it if nothing read it, and the pass has to
    leave it alone when something does -- otherwise the second reader would be
    left holding a node that no longer exists.
    """
    x = torch.randn(4, 8, dtype=torch.float16)
    y = torch.randn(4, 8, dtype=torch.float16)
    model = _RectifiedSumWithReader()
    program = _edge(model, (x, y))
    assert fuse_add_relu(program.graph_module) == 1
    targets = _call_targets(program)
    assert any("add.Tensor" in target for target in targets), targets
    assert any("et_hexagon.add_relu.default" in target for target in targets), targets
    # And the model still computes what it did, on the host side of the split.
    expected = torch.nn.functional.relu(x + y) + (x + y)
    assert model(x, y).numpy().tobytes() == expected.numpy().tobytes()
