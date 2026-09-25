# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP32 cat blobs on the real Hexagon raster-blit kernel."""

import math
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

import hexagon_sim  # noqa: E402
from blob_interpreter import read_blob  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
    _bits,
    _case,
    _fixture_header,
)
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from torch.export import export  # noqa: E402


class _Cat(torch.nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, *values):
        return torch.cat(values, dim=self.axis)


def _values(axis, count, dtype):
    values = []
    for length in (1, 2, 3)[:count]:
        shape = [2] * (axis + 1)
        shape[axis] = length
        values.append(
            torch.arange(1, math.prod(shape) + 1, dtype=torch.float32)
            .reshape(shape)
            .to(dtype)
        )
    return tuple(values)


def _blob(axis, count, dtype):
    values = _values(axis, count, dtype)
    program = to_edge(export(_Cat(axis).eval(), values)).exported_program()
    return bytes(HexagonBackend.preprocess(program, []).processed_bytes), values


def test_fp32_cat_shape_matrix_answers_on_hexagon_sim():
    hexagon_sim._check()
    fixtures = []
    expected_by_tag = {}

    for axis in range(4):
        for count in (1, 2, 3):
            fp32_blob, fp32_values = _blob(axis, count, torch.float32)
            fp16_blob, fp16_values = _blob(axis, count, torch.float16)
            assert fp32_blob == fp16_blob

            _, commands = read_blob(fp32_blob)
            assert [command.type for command in commands] == [3]
            assert commands[0].params[:3] == [count, 2, count]
            assert len(commands[0].params) == 3 + count * 12 <= 39

            expected = torch.cat(
                [value.double() for value in fp32_values],
                dim=axis,
            )
            expected_half = expected.half()
            assert np.array_equal(
                expected_half.numpy(),
                torch.cat(fp16_values, dim=axis).numpy(),
            )

            fp32_tag = f"F32A{axis}C{count}"
            fp16_tag = f"F16A{axis}C{count}"
            fixtures.extend(
                [
                    _case(
                        fp32_tag,
                        None,
                        tuple(value.half() for value in fp32_values),
                        _bits(expected),
                        blob=fp32_blob,
                    ),
                    _case(
                        fp16_tag,
                        None,
                        fp16_values,
                        _bits(expected),
                        blob=fp16_blob,
                    ),
                ]
            )
            expected_by_tag[fp32_tag] = expected_half
            expected_by_tag[fp16_tag] = expected_half

    simulated = hexagon_sim.run(
        _RUNNER,
        _SOURCES,
        headers={
            "blob_fixture.h": _fixture_header(fixtures),
            "htp_ops.h": _HT_P_OP_SHIM,
        },
        includes_more=[str(_SCHEMA)],
    )

    for tag, expected_half in expected_by_tag.items():
        output_tag = f"{tag}0"
        assert output_tag in simulated, hexagon_sim.LAST_STDOUT
        output_bits = np.asarray(simulated[output_tag], dtype=np.uint16)
        assert output_bits.size == expected_half.numel()
        expected_bits = expected_half.numpy().reshape(-1).view(np.uint16)
        assert np.array_equal(output_bits, expected_bits)

        output = output_bits.view(np.float16)
        reference = expected_half.double().numpy().reshape(-1)
        assert np.array_equal(output.astype(np.float64), reference)

    for axis in range(4):
        for count in (1, 2, 3):
            fp32_bits = np.asarray(
                simulated[f"F32A{axis}C{count}0"],
                dtype=np.uint16,
            )
            fp16_bits = np.asarray(
                simulated[f"F16A{axis}C{count}0"],
                dtype=np.uint16,
            )
            assert fp32_bits.tobytes() == fp16_bits.tobytes()
