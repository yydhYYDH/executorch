# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimum reductions and their value-only boundary sweep."""

import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import HexagonPartitioner  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

_MINIMUM = 4
_REDUCTION = hexagon_ops.DSP_OP_REDUCTION


class _Amin(torch.nn.Module):
    def forward(self, x):
        return torch.amin(x)


class _MinBinary(torch.nn.Module):
    def forward(self, x, y):
        return torch.minimum(x, y)


class _MinDimValues(torch.nn.Module):
    def forward(self, x):
        return torch.min(x, dim=1).values


class _MinDimIndices(torch.nn.Module):
    def forward(self, x):
        return torch.min(x, dim=1).indices


def _program(model, args):
    return to_edge_transform_and_lower(
        export(model, tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _blob(program):
    calls = [n for n in program.graph_module.graph.nodes if n.target is torch.ops.higher_order.executorch_call_delegate]
    assert len(calls) == 1
    return bytes(program.graph_module.get_submodule(calls[0].args[0].target)._processed_bytes)


def _command(blob):
    return read_blob(blob)[1][0]


def test_amin_is_one_minimum_reduction_and_uses_the_first_element_identity():
    for values in (
        torch.tensor([4.0, 7.0, 2.0], dtype=torch.float16),
        torch.tensor([-4.0, -7.0, -2.0], dtype=torch.float16),
        torch.tensor([0.0, 3.0, 8.0], dtype=torch.float16),
        torch.tensor([9.0], dtype=torch.float16),
    ):
        x = values.reshape(1, -1)
        blob = _blob(_program(_Amin(), (x,)))
        command = _command(blob)
        assert command.type == _REDUCTION
        assert list(command.params[:5]) == [1, x.numel(), 1, _MINIMUM, 2]
        got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)[0]
        assert got == x.to(torch.float64).min().numpy()


def test_min_binary_is_binary_elementwise_minimum():
    x = torch.tensor([[-3.0, 2.0], [5.0, -1.0]], dtype=torch.float16)
    y = torch.tensor([[1.0, 4.0], [-2.0, 7.0]], dtype=torch.float16)
    blob = _blob(_program(_MinBinary(), (x, y)))
    command = _command(blob)
    assert command.type == hexagon_ops.DSP_OP_BINARY_ELEMENTWISE
    assert command.params[3] == 6
    got = np.frombuffer(execute(blob, [x.numpy(), y.numpy()])[0], dtype=np.float16)
    np.testing.assert_array_equal(got.reshape(x.shape), torch.minimum(x, y).numpy())


def test_min_dim_values_sweep_reduction_length_and_output_length_at_64():
    for reduce in (63, 64, 65):
        x = torch.arange(1, reduce + 1, dtype=torch.float16).reshape(1, reduce)
        blob = _blob(_program(_MinDimValues(), (x,)))
        command = _command(blob)
        assert command.type == _REDUCTION
        assert list(command.params[:5]) == [1, reduce, 1, _MINIMUM, 2]
        got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)
        np.testing.assert_array_equal(got, torch.amin(x, dim=1).numpy())

    for output in (63, 64, 65):
        x = torch.arange(2 * output, dtype=torch.float16).reshape(output, 2)
        blob = _blob(_program(_MinDimValues(), (x,)))
        command = _command(blob)
        assert command.type == _REDUCTION
        assert list(command.params[:5]) == [output, 2, 1, _MINIMUM, 2]
        got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)
        np.testing.assert_array_equal(got, torch.amin(x, dim=1).numpy())


def test_min_dim_indices_remain_refused_without_a_reduction_command():
    x = torch.randn(2, 64, dtype=torch.float16)
    program = _program(_MinDimIndices(), (x,))
    delegates = [n for n in program.graph_module.graph.nodes if n.target is torch.ops.higher_order.executorch_call_delegate]
    assert delegates == []


def test_min_dim_values_never_reads_the_indices():
    x = torch.randn(2, 64, dtype=torch.float16)
    program = _program(_MinDimValues(), (x,))
    assert len([n for n in program.graph_module.graph.nodes if n.target is torch.ops.higher_order.executorch_call_delegate]) == 1
