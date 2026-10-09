# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator evidence for aten.var.correction, and for the model gap it exposes.

This is the tier the arithmetic claims in the workstream report rest on. The
host suite reads the same blobs through `blob_interpreter`, which is a numpy
model: a green test there is evidence about the command stream, not about a
kernel. Here the compiled DSP sources really run, so a difference between this
file and the model is a real disagreement between two implementations and not a
rounding artefact of the reference.

Two things are asserted, and the second is the one worth having.

The composition returns the correctly rounded fp16 of the exact variance. That
is the claim the row makes about itself.

`REDUCTION mean` on its own does not. This is a property of the kernel as the
simulator computes it, and it means the model overstates every mean the backend
emits: the model is the correctly rounded value, and the kernel is several fp16
steps away from it once the span is wide. `REDUCTION sum` errs identically at the
same spans, which is what a power-of-two divide between the two would predict.
The error does not scale with the data and does scale with the span, at roughly
sqrt(n) * 2**-11, so it is an fp16 accumulator and not the fp32 one the source
comment describes.

The variance is unaffected, and the reason is worth writing down because it is
not obvious. A mean that is wrong by d gives a variance that is wrong by
exactly d**2 -- mean((x - m - d)**2) is var + d**2 once mean(x - m) is zero -- so
an error of 3e-04 in the mean contributes 9e-08 to a variance of order one. The
variance first command IS that mean, and the variance is still right. This test
asserts both halves of that, so the file cannot go green while either is wrong.

Simulator arithmetic is not silicon. The project has its own sim-vs-silicon
evidence for a reduction -- the simulator and a phone agree bit for bit, and both
differ from torch by the same 5.859375e-3 -- so the number below is most likely a
property of the DSP and not of the simulator, but a device run is the only tier
that establishes it and none is made here.
"""

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
    _HT_P_OP_SHIM,
    _RUNNER,
    _SOURCES,
    _case,
    _fixture_header,
)

_SCHEMAS = pathlib.Path(__file__).resolve().parents[1] / "serialization"
F16 = torch.float16

#: The shape every case here shares. Four rows so the reduction has four
#: answers, and a last axis that is a power of two so the span is a whole number
#: of vector registers -- the geometry the real node arrives in.
ROWS = 4


class _Var(torch.nn.Module):
    def __init__(self, correction):
        super().__init__()
        self.correction = correction

    def forward(self, x):
        return torch.var(x, dim=-1, correction=self.correction)


class _Mean(torch.nn.Module):
    def forward(self, x):
        return x.mean(-1, keepdim=True)


class _Sum(torch.nn.Module):
    def forward(self, x):
        return x.sum(-1, keepdim=True)


def _bits(tensor):
    return np.ascontiguousarray(tensor.detach().half().numpy().reshape(-1)).view(np.uint16)


def _x(seed, n, scale=1.0, offset=0.0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(ROWS, n, generator=generator, dtype=torch.float32) * scale + offset
    ).to(F16)


def _exact(x, correction=None):
    """The variance in fp64, from the fp64 data, the way a reference should be."""
    wide = x.float().double().numpy()
    if correction is None:
        return wide.mean(-1)
    n = wide.shape[-1]
    return ((wide - wide.mean(-1, keepdims=True)) ** 2).mean(-1) * (n / (n - correction))


#: (tag, module or module factory, n, correction, scale, offset). Six variance
#: cases and two mean cases at the same geometries, so every mean claim has a
#: variance beside it.
_LAYOUT = [
    ("VA", _Var(0), 16, 0, 1.0, 0.0),
    ("VB", _Var(1), 16, 1, 1.0, 0.0),
    ("VC", _Var(1), 16, 1, 1.0, 200.0),
    ("VD", _Var(1), 64, 1, 1.0, 0.0),
    ("VE", _Var(1), 16, 1, 1.0, 0.0),
    ("VF", _Var(1), 64, 1, 1.0, 200.0),
    ("MA", _Mean(), 64, None, 1.0, 0.0),
    ("MB", _Mean(), 1024, None, 0.01, 0.0),
]


@pytest.fixture(scope="module")
def cases():
    built = []
    for index, (tag, module, n, correction, scale, offset) in enumerate(_LAYOUT):
        x = _x(100 + index, n, scale, offset)
        net = module() if isinstance(module, type) else module
        built.append(
            (tag, n, correction, x, _case(tag, net, (x,), _bits(net(x).reshape(-1))))
        )
    return built


@pytest.fixture(scope="module")
def answers(cases):
    # _check() first, and on purpose before the run: run() raises the same
    # Unavailable for "there is no SDK" and for "the binary did not run", so a
    # skip on either would hide a runner that failed to load.
    try:
        hexagon_sim._check()
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))
    try:
        return hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={
                "blob_fixture.h": _fixture_header([c[4] for c in cases]),
                "htp_ops.h": _HT_P_OP_SHIM,
            },
            includes_more=[str(_SCHEMAS)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.fail("hexagon_sim.run failed after _check() passed: " + str(error))


def _got(answers, tag):
    result = answers[f"{tag}0"]
    assert result, f"{tag} returned no output words"
    return np.frombuffer(np.asarray(result, dtype=np.uint16).tobytes(), dtype=np.float16)


def test_the_command_stream_is_the_two_pass_form(cases, answers):
    # The shape check is at the simulator tier as well, so a claim about which
    # commands ran is not resting on the lowering alone.
    for tag, _n, correction, _x, case in cases:
        stream = [command.type for command in case.commands]
        if tag.startswith("M"):
            assert stream == [29], (tag, stream)
        else:
            assert stream == ([29, 19, 4, 29] if correction == 0 else [29, 19, 4, 29, 4]), (
                tag,
                stream,
            )
        assert _got(answers, tag).size > 0


def _ulp(answer, reference):
    """Distance in fp16 steps of the reference.

    The step is read out of the reference own bit pattern rather than out of
    numpy spacing, because numpy spacing of a float32 is a float32 step and a
    float32 has thirteen more mantissa bits than an fp16 -- so a float32 step is
    8192 fp16 steps, and a test written that way is off by a factor of 8192 and
    still passes everything below its own bound.
    """
    bits = np.asarray(reference, dtype=np.float16).view(np.uint16).astype(np.int32)
    step = np.ldexp(1.0, ((bits >> 10) & 0x1F) - 25).astype(np.float32)
    got = np.asarray(answer, dtype=np.float32)
    return np.abs(got - np.asarray(reference, dtype=np.float32)) / step


#: The variance cases, split by whether the data has a mean worth cancelling
#: against. Centred data lands on the correctly rounded exact value to within a
#: step; offset data is where the composition actually loses, and lumping the
#: two together under one tolerance is how a 1.3e-02 gap gets reported as a
#: 3.5e-04 one.
_CENTRED = ("VA", "VB", "VD", "VE")
_OFFSET = ("VC", "VF")


def test_the_variance_is_within_a_step_of_the_correctly_rounded_exact_value(
    cases, answers
):
    # Centred data. The exact variance in fp64, rounded once, is the answer to
    # within one fp16 step, and the bound is two so a second seed cannot make it
    # red. The strict lower bound is what keeps this from passing on nothing.
    worst = 0.0
    for tag, _n, correction, x, _case in cases:
        if tag not in _CENTRED:
            continue
        want = _exact(x, correction).astype(np.float16)
        ulps = _ulp(_got(answers, tag), want)
        assert ulps.max() <= 2.0, (tag, ulps)
        worst = max(worst, float(ulps.max()))
    assert 0.0 < worst <= 2.0, worst


def test_offset_data_is_where_the_composition_loses(cases, answers):
    # A mean of 200 against data of scale 1. The answer is a difference of two
    # numbers of the data own size, so the cancellation is where the fp16 steps
    # go, and on the DSP that costs up to 17 of them -- about 1.3e-02 relative.
    # The model costs 4. The gap between those two numbers is the point of the
    # next test, and both are bounded here so neither can drift unannounced.
    kernel_worst = 0.0
    model_worst = 0.0
    relative = 0.0
    for tag, _n, correction, x, case in cases:
        if tag not in _OFFSET:
            continue
        want = _exact(x, correction)
        rounded = want.astype(np.float16)
        kernel = _got(answers, tag).reshape(-1)
        model = np.frombuffer(case.host[0], dtype=np.float16).reshape(-1)
        kernel_worst = max(kernel_worst, float(_ulp(kernel, rounded).max()))
        model_worst = max(model_worst, float(_ulp(model, rounded).max()))
        relative = max(
            relative,
            float((np.abs(kernel.astype(np.float64) - want) / want).max()),
        )
    assert 0.0 < kernel_worst <= 32.0, kernel_worst
    assert 0.0 < model_worst <= 8.0, model_worst
    assert 0.005 < relative < 0.05, relative


def test_the_kernel_is_further_from_exact_than_the_model_on_offset_data(
    cases, answers
):
    # The finding that matters for a host suite. blob_interpreter is the better
    # of the two implementations on offset data, so a green host test on a
    # reduction over offset activations overstates what the DSP does. Asserted as
    # a comparison between two measured quantities rather than as a bare constant,
    # because that is the claim and a constant would not be.
    for tag, _n, correction, x, case in cases:
        if tag not in _OFFSET:
            continue
        want = _exact(x, correction).astype(np.float16)
        kernel = float(np.abs(_got(answers, tag).reshape(-1).astype(np.float32) - want).max())
        model = float(np.abs(np.frombuffer(case.host[0], dtype=np.float16).astype(np.float32) - want).max())
        assert kernel > model, (tag, kernel, model)


def test_the_reduction_is_not_the_correctly_rounded_mean(cases, answers):
    # The finding. The model IS the correctly rounded value, so the model is a
    # stand-in that overstates the DSP -- and the direction of that overstatement
    # is what makes this worth asserting rather than merely noting.
    for tag, _n, _c, x, case in cases:
        if not tag.startswith("M"):
            continue
        exact = _exact(x).astype(np.float16)
        model = np.frombuffer(case.host[0], dtype=np.float16)
        np.testing.assert_array_equal(model, exact, err_msg=tag + " model")
        kernel = _got(answers, tag).reshape(-1)
        assert not np.array_equal(kernel, exact), (
            tag,
            "the kernel is now the correctly rounded mean, so this test and the",
            "report about an fp16 accumulator are both out of date",
            kernel,
            exact,
        )
        rel = (
            np.abs(kernel.astype(np.float64) - _exact(x)).max()
            / max(abs(float(_exact(x).max())), 1e-30)
        )
        assert rel < 1e-1, (tag, rel)


def test_sum_errs_the_same_way_mean_does(cases, answers):
    # `mean` is `sum` divided by a power of two, so if the accumulation is the
    # thing that is wrong the two errors are the same error. If they stop being
    # the same, the divide is what changed and the fp16-accumulator reading in
    # the module docstring is wrong.
    x = _x(7, 64)
    cases = [
        _case("SA", _Sum(), (x,), _bits(_Sum()(x).reshape(-1))),
        _case("SB", _Mean(), (x,), _bits(_Mean()(x).reshape(-1))),
    ]
    try:
        hexagon_sim._check()
        answers = hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={"blob_fixture.h": _fixture_header(cases), "htp_ops.h": _HT_P_OP_SHIM},
            includes_more=[str(_SCHEMAS)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.fail("hexagon_sim.run failed after _check() passed: " + str(error))
    # Each against its own exact reference: the sum against the sum and the mean
    # against the mean. Comparing a sum to a mean would be a large "error" that
    # is entirely the comparison own mistake.
    exact_sum = x.float().double().numpy().sum(-1)
    exact_mean = _exact(x)
    errors = [
        float(np.abs(_got(answers, tag).reshape(-1).astype(np.float64) - want).max())
        for tag, want in (("SA", exact_sum), ("SB", exact_mean))
    ]
    assert errors[0] > 0.0 and errors[1] > 0.0, errors
    # 64 is a power of two, so the divide is exact and the errors must be in
    # exactly that ratio. If they stop being, the accumulation is not the thing
    # that is wrong and the reading in the module docstring is out of date.
    assert abs(errors[0] / 64.0 - errors[1]) < 1e-9, (errors, 64.0)
