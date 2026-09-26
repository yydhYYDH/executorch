# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The precisely bounded argmax/argmin path and its refusal boundary.

These tests keep the command descriptor, the host model, and the Torch index
answer together.  The simulator and phone runs are separate evidence tiers;
this file is the host-side contract those tiers are measured against.
"""

import os
import pathlib
import struct
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import arg_reduction_spec  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import Dim, export  # noqa: E402

F16 = torch.float16
_ARG = 47
_ARGMAX = exir_ops.edge.aten.argmax.default
_ARGMIN = exir_ops.edge.aten.argmin.default


class _Arg(torch.nn.Module):
    def __init__(self, operation, dim=None, keepdim=False):
        super().__init__()
        self.operation = operation
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            return getattr(torch, self.operation)(x, keepdim=self.keepdim)
        return getattr(torch, self.operation)(x, dim=self.dim, keepdim=self.keepdim)


def _program(model, x, **kwargs):
    return to_edge_transform_and_lower(
        export(model, (x,), **kwargs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _blob(model, x, **kwargs):
    program = _program(model, x, **kwargs)
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"expected one delegate, got {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    raw = bytes(lowered._processed_bytes)
    return raw, read_blob(raw)[1]


def _indices(raw, x):
    return np.frombuffer(execute(raw, [x.numpy()])[0], dtype="<i8").copy()


def _param_at(blob, op_index, param_index):
    """The byte offset of one int parameter in a serialized command."""
    # The params follow the four int32 header words in the command record.
    from executorch.backends.hexagon.serialization import blob

    return blob.HEADER_SIZE + op_index * blob.OP_SIZE + 16 + 4 * param_index


def _set_param(blob, op_index, param_index, value):
    data = bytearray(blob)
    struct.pack_into("<i", data, _param_at(data, op_index, param_index), value)
    return bytes(data)


@pytest.mark.parametrize(
    "operation, mode",
    [("argmax", 0), ("argmin", 1)],
)
def test_last_axis_emits_one_int64_command(operation, mode):
    x = torch.tensor([[1, 3, 3, 2], [4, 1, 0, 1]], dtype=F16)
    raw, commands = _blob(_Arg(operation, dim=-1), x)
    assert [command.type for command in commands] == [_ARG]
    assert list(commands[0].params[:3]) == [4, 2, mode]
    assert commands[0].outputs[0].size == 2 * 8
    assert commands[0].inputs[0].size == x.numel() * 2
    assert np.array_equal(
        _indices(raw, x),
        getattr(torch, operation)(x, dim=-1).numpy().astype("<i8"),
    )


@pytest.mark.parametrize("operation", ["argmax", "argmin"])
def test_no_dim_flattens_and_keepdim_preserves_output_shape(operation):
    x = torch.arange(12, dtype=F16).reshape(3, 4)
    raw, commands = _blob(_Arg(operation), x)
    assert list(commands[0].params[:3]) == [12, 1, int(operation == "argmin")]
    assert commands[0].outputs[0].size == 8
    assert np.array_equal(
        _indices(raw, x),
        getattr(torch, operation)(x).numpy().astype("<i8").reshape(-1),
    )

    raw, commands = _blob(_Arg(operation, dim=-1, keepdim=True), x)
    assert list(commands[0].params[:3]) == [4, 3, int(operation == "argmin")]
    assert commands[0].outputs[0].size == 3 * 8
    assert np.array_equal(
        _indices(raw, x),
        getattr(torch, operation)(x, dim=-1, keepdim=True)
        .numpy()
        .astype("<i8")
        .reshape(-1),
    )


@pytest.mark.parametrize("operation", ["argmax", "argmin"])
def test_cpu_index_semantics_cover_ties_nan_and_signed_zero(operation):
    # The first NaN is selected for both operations on the CPU reference.  A
    # signed zero is one tie, so the first zero is selected rather than whichever
    # sign the vector instruction happens to retain.
    x = torch.tensor(
        [
            [1, 5, 5, 2],
            [5, 1, 1, 5],
            [float("nan"), 9, float("nan"), 8],
            [-0.0, 0.0, 1.0, -1.0],
            [float("inf"), 1, float("-inf"), 2],
        ],
        dtype=F16,
    )
    raw, _ = _blob(_Arg(operation, dim=-1), x)
    got = _indices(raw, x)
    want = getattr(torch, operation)(x, dim=-1).numpy().astype("<i8")
    assert np.array_equal(got, want), (got, want)


@pytest.mark.parametrize("length", [63, 64, 65])
@pytest.mark.parametrize("operation", ["argmax", "argmin"])
def test_structural_boundary_lengths_exercise_both_halves(operation, length):
    # The winning position is the last element for 63/64/65, so the vector and
    # scalar halves are both observable and a wrong tail bound cannot pass.
    x = torch.arange(length, dtype=torch.float32).reshape(1, length).half()
    x[0, length - 1] = 1 if operation == "argmax" else -1
    if operation == "argmax":
        x[0, : length - 1] = torch.linspace(-4, 0, length - 1)
    else:
        x[0, : length - 1] = torch.linspace(4, 0, length - 1)
    raw, commands = _blob(_Arg(operation, dim=-1), x)
    assert list(commands[0].params[:3]) == [length, 1, int(operation == "argmin")]
    assert np.array_equal(_indices(raw, x), np.array([length - 1], dtype=np.int64))


def test_large_index_is_written_as_int64_not_int32():
    length = 70000
    x = torch.zeros((1, length), dtype=F16)
    x[0, length - 1] = 1
    raw, commands = _blob(_Arg("argmax", dim=-1), x)
    assert commands[0].outputs[0].size == 8
    assert _indices(raw, x).tolist() == [length - 1]


def test_wrong_mode_and_geometry_controls_move_the_host_answer():
    x = torch.tensor([[1, 4, 2, 3], [8, 1, 7, 2]], dtype=F16)
    raw, commands = _blob(_Arg("argmax", dim=-1), x)
    want = _indices(raw, x)
    wrong_mode = _set_param(raw, 0, 2, 1)
    wrong_size = _set_param(raw, 0, 0, 3)
    assert _indices(wrong_mode, x).tolist() != want.tolist()
    assert _indices(wrong_size, x).tolist() != want.tolist()
    assert _indices(wrong_size, x).shape == want.shape


@pytest.mark.parametrize(
    "model, x",
    [
        (_Arg("argmax", dim=0), torch.randn(2, 3, dtype=F16)),
        (_Arg("argmax", dim=1), torch.randn(2, 3, 4, dtype=torch.float32)),
        (_Arg("argmin", dim=0), torch.randn(2, 3, dtype=F16)),
    ],
)
def test_unsupported_axis_and_dtype_are_refused(model, x):
    program = _program(model, x)
    assert not any(
        node.target is torch.ops.higher_order.executorch_call_delegate
        for node in program.graph_module.graph.nodes
    )


def test_noncontiguous_input_is_refused():
    x = torch.randn(3, 4, dtype=F16).t()
    program = _program(_Arg("argmax", dim=-1), x)
    assert not any(
        node.target is torch.ops.higher_order.executorch_call_delegate
        for node in program.graph_module.graph.nodes
    )


def test_symbolic_extent_is_refused():
    x = torch.randn(2, 3, dtype=F16)
    program = _program(
        _Arg("argmax", dim=-1),
        x,
        dynamic_shapes={"x": {1: Dim("width", min=3, max=8)}},
    )
    assert not any(
        node.target is torch.ops.higher_order.executorch_call_delegate
        for node in program.graph_module.graph.nodes
    )


def _fake_node(target, source_value, result_value, args):
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = source_value
    node = graph.call_function(target, args=(source, *args))
    node.meta["val"] = result_value
    return node


def test_predicate_rejects_scalar_empty_and_out_of_range_dimensions():
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake = FakeTensorMode()
    scalar = fake.from_tensor(torch.tensor(1, dtype=F16))
    empty = fake.from_tensor(torch.empty((0,), dtype=F16))
    row = fake.from_tensor(torch.empty((2, 3), dtype=F16))
    for value, args in (
        (scalar, (None,)),
        (empty, (-1,)),
        (row, (2,)),
        (row, (-3,)),
        (row, (False,)),
    ):
        node = _fake_node(
            _ARGMAX,
            value,
            fake.from_tensor(torch.empty((), dtype=torch.int64)),
            args,
        )
        assert arg_reduction_spec(node) is None
