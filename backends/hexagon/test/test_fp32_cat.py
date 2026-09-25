# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fp32 cat path uses the existing fp16 arena blit unchanged."""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

_CONFIG = EdgeCompileConfig(_check_ir_validity=False)
_RASTER_BLIT = hexagon_ops.DSP_OP_RASTER_BLIT


class _Cat(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *values):
        return torch.cat(values, dim=self.dim)


class _ConstantCat(torch.nn.Module):
    def __init__(self, values, dim):
        super().__init__()
        self.dim = dim
        for index, value in enumerate(values):
            self.register_buffer(f"value_{index}", value)

    def forward(self):
        values = [
            getattr(self, f"value_{index}") for index in range(len(self._buffers))
        ]
        return torch.cat(values, dim=self.dim)


def _delegates(module, inputs, dynamic_shapes=None):
    return to_edge_transform_and_lower(
        export(module.eval(), inputs, dynamic_shapes=dynamic_shapes),
        partitioner=[HexagonPartitioner()],
        compile_config=_CONFIG,
    ).exported_program()


def _calls(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _inputs_for(axis, dtype=torch.float32, count=3):
    shapes = []
    for length in (1, 2, 3)[:count]:
        shape = [2] * (axis + 1)
        shape[axis] = length
        shapes.append(tuple(shape))
    return tuple(
        torch.arange(1, math_prod(shape) + 1, dtype=torch.float32)
        .reshape(shape)
        .to(dtype)
        for shape in shapes
    )


def math_prod(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def _blob_for(module, inputs):
    program = _delegates(module, inputs)
    calls = _calls(program)
    assert len(calls) == 1
    delegate = program.graph_module.get_submodule(calls[0].args[0].target)
    names = {node.name for node in delegate.original_module.graph_module.graph.nodes}
    assert "aten_cat_default" in names, names
    return bytes(delegate.processed_bytes)


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_fp32_cat_on_every_axis_uses_one_existing_blit(axis, count):
    inputs = _inputs_for(axis, count=count)
    blob = _blob_for(_Cat(axis), inputs)
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [_RASTER_BLIT]
    assert commands[0].params[:3] == [count, 2, count]
    assert len(commands[0].params) == 3 + count * 12 <= 39
    assert commands[0].outputs[0].size == torch.cat(inputs, axis).numel() * 2

    got = np.frombuffer(
        execute(blob, [value.half().numpy() for value in inputs])[0],
        dtype=np.float16,
    )
    reference = torch.cat(
        [value.double() for value in inputs],
        dim=axis,
    )
    expected = torch.cat([value.half() for value in inputs], dim=axis).double().numpy()
    assert np.array_equal(got, expected.reshape(-1))
    assert np.array_equal(got.astype(np.float64), reference.numpy().reshape(-1))


def test_fp16_and_fp32_spellings_emit_the_same_blob_bytes():
    axis = 2
    fp16_inputs = _inputs_for(axis, torch.float16)
    fp32_inputs = tuple(value.float() for value in fp16_inputs)
    fp16_blob = _blob_for(_Cat(axis), fp16_inputs)
    fp32_blob = _blob_for(_Cat(axis), fp32_inputs)
    assert fp16_blob == fp32_blob
    fp16_output = execute(fp16_blob, [value.numpy() for value in fp16_inputs])[0]
    fp32_output = execute(
        fp32_blob,
        [value.half().numpy() for value in fp32_inputs],
    )[0]
    assert fp16_output.tobytes() == fp32_output.tobytes()


@pytest.mark.parametrize(
    "values,dim",
    [
        ((torch.ones(2, dtype=torch.float16), torch.ones(2, dtype=torch.float32)), 0),
        (tuple(torch.ones(2, dtype=torch.float64) for _ in range(2)), 0),
    ],
)
def test_non_fp32_cats_still_emit_no_command_and_delegate_nothing(values, dim):
    assert _calls(_delegates(_ConstantCat(values, dim), ())) == []


def test_four_contiguous_fp32_inputs_delegate_over_two_blits():
    values = tuple(torch.ones((1, 2), dtype=torch.float32) for _ in range(4))
    blob = _blob_for(_ConstantCat(values, 0), ())
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [_RASTER_BLIT] * 2
    assert commands[0].params[:3] == [3, 2, 3]
    assert commands[1].params[:3] == [1, 2, 1]
    assert len(commands[0].params) == 3 + 3 * 12
    assert len(commands[1].params) == 3 + 12


def test_noncontiguous_cats_stay_portable_at_every_input_count():
    values = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        torch.arange(6, 12, dtype=torch.float32).reshape(2, 3).t(),
    )
    assert _calls(_delegates(_Cat(0), values)) == []
    assert _calls(_delegates(_Cat(0), values + values[:2])) == []
