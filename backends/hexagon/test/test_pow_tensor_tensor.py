# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded tensor-tensor power coverage for the Hexagon backend."""

import numpy as np
import pytest
import torch
from torch.export import export

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.backends.hexagon.test.blob_interpreter import execute, read_blob
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

POWER = hexagon_ops.POW_TENSOR_TENSOR


class StaticPower(torch.nn.Module):
    def __init__(self, base, exponent):
        super().__init__()
        self.base = base
        self.exponent = exponent

    def forward(self):
        return torch.pow(
            torch.tensor(self.base, dtype=torch.float16),
            torch.tensor(self.exponent, dtype=torch.float16),
        )


class RuntimeBasePower(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, base):
        return torch.pow(
            base,
            torch.tensor(self.exponent, dtype=torch.float16),
        )


def _lowered(model, args=()):
    return to_edge_transform_and_lower(
        export(model, tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegate(program):
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1
    return program.graph_module.get_submodule(calls[0].args[0].target)


def _blob_and_commands(model, args=()):
    program = _lowered(model, args)
    inner = _delegate(program)
    blob = bytes(inner._processed_bytes)
    header, commands = read_blob(blob)
    return blob, header, commands


def _mapping():
    return {v: k for k, v in vars(hexagon_ops).items() if k.startswith("DSP_OP_")}


def _power_is_delegated(program):
    for node in program.graph_module.graph.nodes:
        if node.target is torch.ops.higher_order.executorch_call_delegate:
            inner = program.graph_module.get_submodule(node.args[0].target)
            if any(
                inner_node.target is POWER
                for inner_node in inner.original_module.graph_module.graph.nodes
            ):
                return True
    return False


@pytest.mark.parametrize("exponent", [-1, 0, 1, 2, 3, 4])
def test_tensor_tensor_power_uses_the_bounded_command_subset(exponent):
    base = [1.0, -2.0, 0.5, 3.0, -0.25, 4.0]
    blob, header, commands = _blob_and_commands(StaticPower(base, exponent))
    assert header.n_ops == len(commands)
    assert commands
    assert {_mapping()[command.type] for command in commands} <= {
        "DSP_OP_BINARY_ELEMENTWISE",
        "DSP_OP_RASTER_BLIT",
    }
    actual = np.frombuffer(execute(blob, [])[0], dtype=np.float16)
    reference = torch.pow(
        torch.tensor(base, dtype=torch.float64),
        torch.tensor(float(exponent), dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.from_numpy(actual.copy()).double(),
        reference,
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("length", [63, 64, 65])
def test_tensor_tensor_power_covers_vector_boundaries(length):
    base = (torch.arange(length, dtype=torch.float32) / 8.0 + 0.25).tolist()
    blob, _, commands = _blob_and_commands(StaticPower(base, 2))
    types = [command.type for command in commands]
    # The contiguous clone the elementwise branch admits writes the lifted
    # constants with raster blits before the power itself; the power command is
    # the one binary elementwise here, and nothing else computes.
    assert types.count(hexagon_ops.DSP_OP_BINARY_ELEMENTWISE) == 1
    assert set(types) <= {
        hexagon_ops.DSP_OP_BINARY_ELEMENTWISE,
        hexagon_ops.DSP_OP_RASTER_BLIT,
    }
    actual = np.frombuffer(execute(blob, [])[0], dtype=np.float16)[:length]
    reference = torch.pow(
        torch.tensor(base, dtype=torch.float64),
        torch.tensor(2.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.from_numpy(actual.copy()).double(),
        reference,
        rtol=2e-3,
        atol=2e-3,
    )


def test_runtime_base_identity_is_the_only_dynamic_base_case():
    model = RuntimeBasePower(1.0)
    program = _lowered(model, (torch.tensor([1.0, -2.0, 0.0]),))
    inner = _delegate(program)
    inner_nodes = {
        node.name for node in inner.original_module.graph_module.graph.nodes
    }
    assert any(node.target is POWER for node in inner.original_module.graph_module.graph.nodes)
    assert any(node.name.startswith("base") for node in inner.original_module.graph_module.graph.nodes)
    assert inner_nodes
    dynamic_types = [
        command.type for command in read_blob(bytes(inner._processed_bytes))[1]
    ]
    # An identity base with a runtime input is a copy, not a power: the delegate
    # holds the clone blits and the pow node itself stays portable, so the
    # stream is copies only and no binary elementwise is written for the power.
    assert dynamic_types
    assert set(dynamic_types) == {hexagon_ops.DSP_OP_RASTER_BLIT}

    refused = _lowered(
        RuntimeBasePower(2.0),
        (torch.tensor([1.0, -2.0, 0.0]),),
    )
    refused_inner = []
    for node in refused.graph_module.graph.nodes:
        if node.target is torch.ops.higher_order.executorch_call_delegate:
            refused_inner.extend(
                refused.graph_module.get_submodule(node.args[0].target)
                .original_module.graph_module.graph.nodes
            )
    assert not any(node.target is POWER for node in refused_inner)


@pytest.mark.parametrize(
    "base, exponent",
    [
        ([-2.0, 4.0], 0.5),
        ([0.0, 2.0], -1.0),
        ([0.0, 2.0], 0.0),
        ([1.0e5, 2.0], 2.0),
        ([1.0e-4, 2.0], 2.0),
        ([2.0, 3.0], 5.0),
    ],
)
def test_tensor_tensor_power_refuses_invalid_domains_and_numerics(base, exponent):
    program = _lowered(StaticPower(base, exponent))
    assert not _power_is_delegated(program)


class ScalarSquare(torch.nn.Module):
    def forward(self, value):
        return torch.pow(value, 2)


def test_scalar_power_overload_remains_the_unary_square_path():
    program = _lowered(ScalarSquare(), (torch.tensor([1.0, -2.0, 0.5]),))
    inner = _delegate(program)
    commands = read_blob(bytes(inner._processed_bytes))[1]
    assert [command.type for command in commands] == [hexagon_ops.DSP_OP_UNARY]
    assert POW_SCALAR_TARGET is not POWER
    assert POW_SCALAR_TARGET in hexagon_ops.EMITTERS


POW_SCALAR_TARGET = hexagon_ops.POW_TENSOR_SCALAR
