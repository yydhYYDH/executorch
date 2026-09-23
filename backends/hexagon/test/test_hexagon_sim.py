"""The vendored kernels themselves, executed on the Hexagon simulator.

Every other test in this directory compares torch against a transcription of
the C. This one runs the C: the same sources the skel is built from, compiled
for v79 by the Hexagon toolchain and executed under hexagon-sim, with the fp16
bit patterns compared against torch. It needs HEXAGON_SDK_ROOT, and a
libncurses5 for the simulator, and skips with a reason when either is missing.
"""

import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hexagon_sim

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/ops_runner.cpp"

#: The vendored sources the ops need; nothing here is reimplemented.
_SOURCES = [
    "blit_ops.cc",
    "loop_ops.cc",
    "matmul_ops.cc",
    "eltwise_ops.cc",
    "unary_ops.cc",
    "layer_norm_ops.cc",
    "worker_pool.cc",
    "vtcm_mgr.cc",
    "hmx_mgr.cc",
    "pwl.cc",
    "power.cc",
    "ops/matmul_q4fp16.c",
    "ops/matmul_q4fp16_mle32.c",
]

_ROWS, _INNER = 3, 8
_M, _K, _N = 4, 8, 3
_EPS = 1e-5


@pytest.fixture(scope="module")
def simulated():
    try:
        return hexagon_sim.run(_RUNNER, _SOURCES)
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))


def _bits(values):
    return values.half().numpy().view("uint16").reshape(-1).tolist()


def _norm_source():
    values = torch.arange(_ROWS * _INNER) * 7 % 13 - 6
    return (values.float().reshape(_ROWS, _INNER) * 0.5).half()


def _norm_references(eps=_EPS):
    x = _norm_source().float()
    sqsum = (x * x).sum(dim=1, keepdim=True)
    mean = x.sum(dim=1, keepdim=True) / _INNER
    rms = x * (1.0 / torch.sqrt(sqsum / _INNER + eps))
    variance = sqsum / _INNER - mean * mean
    # The kernel's own order: subtract the mean, scale, then apply the affine.
    normed = (x - mean) * (1.0 / torch.sqrt(variance + eps))
    gamma = torch.tensor([1.0 + 0.125 * i for i in range(_INNER)])
    beta = torch.tensor([0.25 * (i % 3) for i in range(_INNER)])
    return rms, normed * gamma + beta


def _mm_operands():
    a = torch.tensor([(i * 5) % 7 - 3 for i in range(_M * _K)], dtype=torch.float32)
    b = torch.tensor([(i * 3) % 5 - 2 for i in range(_K * _N)], dtype=torch.float32)
    return a.half().reshape(_M, _K), b.half().reshape(_K, _N)


def _blit_source():
    return torch.tensor([i * 2 - 5 for i in range(6)], dtype=torch.float32).half()


def test_the_runner_receives_the_operands_we_meant(simulated):
    """The descriptors are only meaningful if the bytes got there intact."""
    a, b = _mm_operands()
    assert simulated["IN"] == _bits(_norm_source())
    assert simulated["A"] == _bits(a)
    assert simulated["B"] == _bits(b)


def test_rms_norm_on_the_dsp_matches_torch(simulated):
    assert simulated["RMS"] == _bits(_norm_references()[0]), "not bit-exact"


def test_layer_norm_on_the_dsp_matches_torch(simulated):
    assert simulated["LN"] == _bits(_norm_references()[1]), "not bit-exact"


def test_batch_matmul_on_the_dsp_matches_torch(simulated):
    a, b = _mm_operands()
    expected = (a.float() @ b.float()).half()
    assert simulated["MM"] == _bits(expected), "not bit-exact"


def test_the_raster_blit_on_the_dsp_transposes(simulated):
    """The region is the one the emitter produces for a 2x3 permute."""
    source = _blit_source()
    expected = source.reshape(2, 3).t().reshape(-1)
    assert simulated["BLIT"] == _bits(expected), "not bit-exact"


def test_the_comparisons_can_fail():
    """Every equality above compares bit patterns that a wrong value would move.

    Without this, a comparison that always passed would look like agreement.
    """
    assert _bits(_norm_references(eps=_EPS * 10)[1]) != _bits(_norm_references()[1])
    assert _bits(_blit_source().reshape(2, 3).reshape(-1)) != _bits(
        _blit_source().reshape(2, 3).t().reshape(-1)
    )
