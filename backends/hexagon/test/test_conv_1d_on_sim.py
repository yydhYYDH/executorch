# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the emitted Conv1d blob's command stream on the Hexagon simulator."""

import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import hexagon_sim  # noqa: E402
from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
    _fixture_header,
)
from torch.export import export  # noqa: E402


def _model_and_input():
    model = torch.nn.Conv1d(3, 5, 3, padding=1).half().eval()
    model.weight.data.copy_(torch.arange(45, dtype=torch.float16).reshape_as(model.weight) % 7 - 3)
    model.bias.data.copy_(torch.arange(5, dtype=torch.float16) - 2)
    value = torch.arange(27, dtype=torch.float16).reshape(1, 3, 9) % 5 - 2
    return model, value


def _fixture():
    model, value = _model_and_input()
    program = to_edge(export(model, (value,))).exported_program()
    blob = bytes(HexagonBackend.preprocess(program, []).processed_bytes)
    header, commands = read_blob(blob)
    assert [command.type for command in commands] == [24, 3, 12, 3]
    return SimpleNamespace(
        tag="CONV1D",
        blob=blob,
        inputs=value.numpy().astype(np.float16).tobytes(),
        length=0,
        arena_bytes=8 * 1024 * 1024,
    )


def test_conv1d_emitted_blob_executes_on_hexagon_sim():
    try:
        hexagon_sim._check()
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))

    case = _fixture()
    result = hexagon_sim.run(
        _RUNNER,
        _SOURCES,
        headers={
            "blob_fixture.h": _fixture_header([case]),
            "htp_ops.h": _HT_P_OP_SHIM,
        },
        includes_more=[str(_SCHEMA)],
    )

    assert "CONV1D0" in result, result
    output = np.asarray(result["CONV1D0"], dtype=np.uint16).view(np.float16)
    assert output.size == 45
    assert np.count_nonzero(output) == output.size

    model, value = _model_and_input()
    with torch.no_grad():
        reference = model.double()(value.double()).reshape(-1).numpy()
    np.testing.assert_array_equal(output, reference.astype(np.float16))
