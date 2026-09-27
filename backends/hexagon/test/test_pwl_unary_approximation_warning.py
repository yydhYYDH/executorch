# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""That an export which delegates a table-walked unary says so, and says what.

The DSP computes gelu, sigmoid, silu and tanh as a bank of fp16 chords
(htp_ops_unary_pwl_fp16_vec, unary_ops.cc:259-330) and takes the last numel % 64
elements through a scalar expression instead (unary_ops.cc:489-491). The support
table says that these ops take one input and preserve the element count, which is
true and mentions none of that, and the export prints nothing. So the two ways of
asking the same question -- a host reference through torch, and the device -- come
back with different numbers, by up to 6.16e-3 for gelu at one activation, and the
log cannot account for the difference. The host interpreter models the chords
correctly, which is what makes the silence the whole of the problem: the right
answer exists in the tree and nothing points a caller at it.

This is the part that is not a number. The three other PWL test files hold the
deviations and the grain, and they are only read by someone who already suspects
one. The four here hold the warning, which is what an export that contains a gelu
gets whether or not anyone suspects anything.

What is NOT asserted, and why:

- that the warning fires exactly once. It is raised from the emitter, so it fires
  once per delegated node, and warnings' own registry collapses the repeats for a
  caller who has not asked for them. A test that counted them would be testing the
  registry, not the emitter.
- that it is fatal. It is a warning because the DSP does compute the op; refusing
  would be a different backend. The escalation is a separate case below, and it is
  opt-in through the environment rather than through an argument, because the
  emitter has no argument to carry it.

The negative case is in every one of the four. A detector that fires for everything
is indistinguishable from one that fires for the right thing, so each test pairs a
kind that must warn with one that must not: abs and square are DSP_OP_UNARY too,
and the DSP computes both exactly.
"""

import os
import pathlib
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    PWL_UNARY_HOST_MODEL,
    PWL_UNARY_MAX_ERROR,
    HexagonApproximationWarning,
)
from executorch.exir import to_edge  # noqa: E402
from torch.export import export  # noqa: E402

#: The kinds htp_ops_unary_compute_fp16_chunk sends to the walk, against the two
#: unary kinds it evaluates exactly, so every case here has a control that shares
#: the emitter, the command type and the dtype and differs only in the function.
_WALKED = ("gelu", "sigmoid", "silu", "tanh")
_EXACT = ("abs", "square")


class _One(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


_FUNCTIONS = {
    "gelu": torch.nn.functional.gelu,
    "sigmoid": torch.sigmoid,
    "silu": torch.nn.functional.silu,
    "tanh": torch.tanh,
    "abs": torch.abs,
    "square": torch.square,
}


def _export(name):
    """The blob for one unary over 128 elements, which is above the grain."""
    x = torch.zeros(1, 128, dtype=torch.float16)
    program = to_edge(export(_One(_FUNCTIONS[name]), (x,))).exported_program()
    return HexagonBackend.preprocess(program, []).processed_bytes


def _approximations(name):
    """The warnings one export raised, with the registry set to record every call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _export(name)
    return [
        record
        for record in caught
        if issubclass(record.category, HexagonApproximationWarning)
    ]


def test_the_kinds_the_walk_computes_warn_and_the_ones_it_does_not_do_not():
    """Four that warn, two that share everything but the function and do not."""
    for name in _WALKED:
        raised = _approximations(name)
        assert raised, (
            f"an export of a bare {name} raised no approximation warning, so the"
            " caller is left with a host reference through torch and a device"
            " that answer two different functions"
        )
    for name in _EXACT:
        assert not _approximations(name), (
            f"a bare {name} raised an approximation warning: the DSP computes it"
            " exactly, so the warning is noise and a warning that cries wolf is"
            " a warning that gets switched off"
        )


def test_the_warning_names_the_kind_the_number_and_the_tail():
    """A warning nobody can act on is a log line, so it has to carry all three."""
    raised = _approximations("gelu")
    assert len(raised) == 1, [str(r.message) for r in raised]
    text = str(raised[0].message)
    assert "gelu" in text, text
    assert f"{PWL_UNARY_MAX_ERROR['gelu']:.3e}" in text, (
        f"the warning does not carry {PWL_UNARY_MAX_ERROR['gelu']:.3e}, so a"
        " caller cannot tell whether the gap it is about to see is the one"
        " being described"
    )
    assert "numel % 64" in text, (
        "the warning does not say that the last numel % 64 elements take a"
        " second form, which is the half of the answer that depends on the"
        " tensor length and is the half a caller debugging a mismatch needs"
    )
    assert "unary_ops.cc" in text, text


def test_the_warning_is_a_category_of_its_own_so_it_can_be_filtered():
    """-W ignore::... and -W error::... are the only way to turn it off or up."""
    assert issubclass(HexagonApproximationWarning, UserWarning)
    raised = _approximations("gelu")
    assert raised[0].category is HexagonApproximationWarning
    assert f"{HexagonApproximationWarning.__module__}." in text_of(raised)


def text_of(raised):
    """The fully qualified name a -W filter would be given."""
    return f"{raised[0].category.__module__}.{raised[0].category.__qualname__}"


def test_the_environment_variable_turns_the_warning_into_a_refusal(monkeypatch):
    """The opt-in escalation, and that it is off unless it is asked for."""
    monkeypatch.delenv("EXECUTORCH_HEXAGON_ERROR_ON_APPROX_UNARY", raising=False)
    assert _export("abs"), "the control export failed, so the refusal below is untested"
    monkeypatch.setenv("EXECUTORCH_HEXAGON_ERROR_ON_APPROX_UNARY", "1")
    with pytest.raises(RuntimeError) as refusal:
        _export("gelu")
    assert "gelu" in str(refusal.value)
    with pytest.raises(RuntimeError):
        _export("sigmoid")
    # and the kinds the DSP computes exactly still export, which is what makes
    # this a gate on the approximation and not a gate on the backend
    assert _export("abs")
    assert _export("square")


def test_every_walked_kind_has_a_number_and_a_host_model_answer():
    """The tables the warning quotes, checked to be filled in."""
    assert sorted(PWL_UNARY_MAX_ERROR) == sorted(_WALKED)
    assert sorted(PWL_UNARY_HOST_MODEL) == sorted(_WALKED)
    for name, error in PWL_UNARY_MAX_ERROR.items():
        assert 0.0 < error < 1.0e-2, (name, error)
        assert PWL_UNARY_HOST_MODEL[name], name
    assert any(
        "NOT modelled" in PWL_UNARY_HOST_MODEL[name] for name in _WALKED
    ), (
        "no kind is recorded as unmodellable on the host, so either the host"
        " interpreter grew one since this was written or a kind was added to the"
        " warning without saying what a caller can do about it"
    )
