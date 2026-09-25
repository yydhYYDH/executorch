# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the exact 1x1xK Conv3D rewrite on the Hexagon simulator."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hexagon_sim  # noqa: E402
from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon.decompose_conv3d import DecomposeFrameConv3d  # noqa: E402
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
    model = torch.nn.Conv3d(3, 3, (1, 1, 3), groups=3, bias=True).half().eval()
    with torch.no_grad():
        model.weight.copy_(
            torch.arange(9, dtype=torch.float16).reshape_as(model.weight) % 5 - 2
        )
        model.bias.copy_(torch.arange(3, dtype=torch.float16) - 1)
    value = torch.arange(216, dtype=torch.float16).reshape(1, 3, 2, 4, 9) % 5 - 2
    return model, value


def _fixture():
    model, value = _model_and_input()
    program = to_edge(export(model, (value,))).exported_program()
    program = DecomposeFrameConv3d()(program).exported_program
    blob = bytes(HexagonBackend.preprocess(program, []).processed_bytes)
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [3, 3, 2, 3, 3]
    return SimpleNamespace(
        tag="CONV3D",
        blob=blob,
        inputs=value.numpy().astype(np.float16).tobytes(),
        length=0,
        arena_bytes=8 * 1024 * 1024,
    )


def test_frame_conv3d_blob_executes_on_hexagon_sim():
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

    assert "CONV3D0" in result, hexagon_sim.LAST_STDOUT
    output = np.asarray(result["CONV3D0"], dtype=np.uint16).view(np.float16)
    assert output.size == 168
    assert output.size > 0

    model, value = _model_and_input()
    with torch.no_grad():
        reference = model.double()(value.double()).reshape(-1).numpy()
    np.testing.assert_array_equal(output, reference.astype(np.float16))
