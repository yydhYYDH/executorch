# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator evidence for amin, binary min, and min-dimension values."""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _RUNNER,
    _SOURCES,
    _fixture_header,
    _HT_P_OP_SHIM,
    _case,
)
from executorch.backends.hexagon.serialization import blob as _blob  # noqa: E402

_SCHEMAS = pathlib.Path(__file__).resolve().parents[1] / "serialization"


class _Amin(torch.nn.Module):
    def forward(self, x):
        return torch.amin(x)


class _MinBinary(torch.nn.Module):
    def forward(self, x, y):
        return torch.min(x, y)


class _MinDimValues(torch.nn.Module):
    def forward(self, x):
        return torch.min(x, dim=1).values


def _bits(tensor):
    return tensor.detach().half().numpy().view(np.uint16)


@pytest.fixture(scope="module")
def cases():
    amin = torch.tensor([[4, 7, 2, -3, 5]], dtype=torch.float16)
    left = torch.tensor([[-3, 2, 5, -1]], dtype=torch.float16)
    right = torch.tensor([[1, 4, -2, 7]], dtype=torch.float16)
    dim63 = torch.arange(1, 64, dtype=torch.float16).reshape(1, 63)
    dim64 = torch.arange(1, 65, dtype=torch.float16).reshape(1, 64)
    dim65 = torch.arange(1, 66, dtype=torch.float16).reshape(1, 65)
    return [
        _case("MA", _Amin(), (amin,), _bits(amin.min())),
        _case("MB", _MinBinary(), (left, right), _bits(torch.minimum(left, right))),
        _case("MC", _MinDimValues(), (dim63,), _bits(dim63.amin(dim=1))),
        _case("MD", _MinDimValues(), (dim64,), _bits(dim64.amin(dim=1))),
        _case("ME", _MinDimValues(), (dim65,), _bits(dim65.amin(dim=1))),
    ]


def test_minimum_overloads_run_on_hexagon_sim_and_match_fp64_reference(cases):
    try:
        hexagon_sim._check()
        answers = hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={"blob_fixture.h": _fixture_header(cases), "htp_ops.h": _HT_P_OP_SHIM},
            includes_more=[str(_SCHEMAS)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))

    assert not [line for line in hexagon_sim.LAST_STDOUT.splitlines() if "TAG" not in line and "hexagon" in line.lower() and "error" in line.lower()]
    for case in cases:
        result = answers[f"{case.tag}0"]
        assert result, f"{case.tag} returned no output words"
        got = np.frombuffer(np.asarray(result, dtype=np.uint16).tobytes(), dtype=np.float16)
        if case.tag == "MA":
            expected = torch.amin(case.args[0]).to(torch.float64).numpy()
        elif case.tag == "MB":
            expected = torch.minimum(case.args[0], case.args[1]).to(torch.float64).numpy()
        else:
            expected = torch.amin(case.args[0], dim=1).to(torch.float64).numpy()
        np.testing.assert_allclose(got.astype(np.float64).reshape(-1), expected.reshape(-1), rtol=0, atol=0, err_msg=case.tag)
        assert [command.type for command in case.commands] == [29 if case.tag != "MB" else 19]
