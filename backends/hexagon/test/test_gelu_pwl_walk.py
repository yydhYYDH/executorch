# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""That the gelu the host model answers is the gelu the DSP computes, re-derived.

The failure this guards against is not a wrong constant. It is a host model that
answers the fp32 tanh form -- which is closer to torch, and therefore the thing a
reference wants -- while the DSP answers a twelve-chord fp16 table. The two
disagree on the majority of the finite fp16 values in [-9, 9] and by up to
6.16e-3, so a comparison run through torch's gelu measures the gap between two
things that are both called gelu and reports it as a DSP defect. The export is
silent, the run is silent, and nothing in the log says which of the two is wrong.

blob_interpreter already transcribes the table, and the transcription is right:
measured here against a second implementation built from unary_ops.cc and
pwl.h without looking at the module under test, _pwl_body answers the same bits
as that implementation on every one of the 63488 finite fp16 values, and the
same bits as the phone's own recorded answers in test_unary_pwl._GOLDEN. So this
file is not here to replace it. It is here because a transcription with no
independent check is a transcription nobody has read, and the whole class of bug
is invisible from the inside: the model and the DSP being the same function is
not observable from the model.

What is re-derived here and why:

- the twelve chords, read out of the C++ rather than taken from _PWL_SLOPE, so a
  second copy of the table cannot agree with the first one by construction. It is
  parsed, not written out, which is test_pwl_bank_single_source's rule about where
  a bank may be stated and the reason for it.
- the companded index and the eval, written out from pwl.h:47-71 and :90-93, for
  the reason test_sigmoid_grain.py gives: a wrong index has to be able to hide
  inside a wrong table if the table is the module's own.
- the split at numel & ~63, through execute() rather than through _pwl_body, so
  what is tested is the dispatch the DSP performs and not the body alone.

The control is the fourth test. A test that says the interpreter answers the
chords passes on an interpreter that answers the chords, and would also pass on
one that answers anything the chords happen to agree with; the count of inputs
where the two forms separate is what makes the other three worth reading.
"""

import os
import pathlib
import re
import sys

import numpy as np
import torch

# The checkout directory is itself named executorch, so putting its parent on the
# path makes `import executorch` resolve to this tree.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import test_unary_pwl  # noqa: E402  the phone's own recorded bits
from blob_interpreter import _pwl_body, execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from torch.export import export  # noqa: E402

_TEST_DIR = pathlib.Path(__file__).resolve().parent
_UNARY_OPS = (
    _TEST_DIR.parent
    / "third-party"
    / "mnn-htp-ops"
    / "src"
    / "dsp"
    / "unary_ops.cc"
)

#: HTP_OPS_UNARY_GELU, and the magnitude the table stops at (unary_ops.cc:269-270,
#: where range_is_eight is false for everything but sigmoid).
_GELU = 3
_RANGE = 4.0

#: htp_ops_unary_compute_fp16_chunk's vec_len: 128 bytes over fp16
#: (unary_ops.cc:334), and the index the walk is measured in.
_GRAIN = 64


def _cpp_gelu_bank():
    """{slope: bits, bias: bits} read out of the branch the skel is built with."""
    source = _UNARY_OPS.read_text()
    branch = source.split("#if HTP_OPS_PWL_COMPANDED16")[1].split("#else")[0]
    tables = {
        name: tuple(
            int(literal, 16)
            for literal in re.findall(r"0x[0-9a-fA-F]+", body)
        )
        for name, body in re.findall(
            r"HTP_OPS_PWL_TABLE\((\w+),\s*([^;]*?)\);", branch, re.S
    )
    }
    return {"slope": tables["gelu_slope"], "bias": tables["gelu_bias"]}


def _lookup16(index, bits):
    """htp_ops_pwl_lookup16 (pwl.h:38-45): the index reads four bits."""
    table = np.zeros(32, dtype=np.uint16)
    for position, value in enumerate(bits):
        table[position] = np.uint16(value)
    return table[index & np.uint16(0x000F)].view(np.float16)


def _companded_index16(abs_v):
    """htp_ops_pwl_companded_index16 (pwl.h:57-71) over htp_ops_pwl_index16 (:47-55)."""
    scaled = (abs_v * np.float16(4.0)).astype(np.float16)
    scaled = np.where(scaled > np.float16(15.0), np.float16(15.0), scaled)
    low = ((scaled + np.float16(16.0)).astype(np.float16).view(np.uint16) >> np.uint16(6))
    wide = ((abs_v.view(np.uint16) >> np.uint16(8)) & np.uint16(7)) + np.uint16(8)
    return np.where(abs_v >= np.float16(2.0), wide, low).astype(np.uint16)


def chords(values):
    """htp_ops_unary_pwl_fp16_vec for opType == GELU (unary_ops.cc:259-330)."""
    bank = _cpp_gelu_bank()
    x = np.ascontiguousarray(np.asarray(values, dtype=np.float16))
    negative = np.float16(0.0) > x
    abs_v = np.where(negative, (-x).astype(np.float16), x).astype(np.float16)
    index = _companded_index16(abs_v)
    slope = _lookup16(index, bank["slope"])
    bias = _lookup16(index, bank["bias"])
    # htp_ops_pwl_eval (pwl.h:90-93): the product is fp16 before the bias is added.
    positive = (abs_v * slope).astype(np.float16) + bias
    folded = (positive - abs_v).astype(np.float16)
    result = np.where(negative, folded, positive).astype(np.float16)
    limit = np.where(negative, np.float16(0.0), abs_v).astype(np.float16)
    return np.where(abs_v < np.float16(_RANGE), result, limit).astype(np.float16)


def scalar_form(values):
    """htp_ops_unary_apply_fp16 for opType == GELU (unary_ops.cc:167-172)."""
    x = np.ascontiguousarray(np.asarray(values, dtype=np.float16)).astype(np.float32)
    inner = np.float32(0.79788456) * (x + np.float32(0.044715) * x * x * x)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))).astype(np.float16)


def _every_finite_fp16():
    """All 63488, so a disagreement is a count and not a probe."""
    every = np.arange(0, 1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
    return np.ascontiguousarray(every[np.isfinite(every)])


class _Gelu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)


def _dsp_answer(values):
    """What the command stream computes for these values, through execute.

    The blob is built at this length and run at this length, so the length the
    emitter writes into the command and the length the walk splits on are the
    same number and cannot drift apart behind the assertion.
    """
    length = len(values)
    x = torch.tensor([values], dtype=torch.float16)
    program = to_edge(export(_Gelu(), (x,))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _header, commands = read_blob(blob)
    assert len(commands) == 1, f"{length} elements: {len(commands)} commands"
    assert commands[0].params[0] == length and commands[0].params[1] == _GELU, (
        f"the command is {list(commands[0].params)[:2]}, not {length}-element gelu"
    )
    outputs = execute(blob, [np.ascontiguousarray(x.numpy())])
    return np.frombuffer(outputs[0], dtype=np.float16)


def test_the_walk_over_every_finite_fp16_is_the_dsp_table_bit_for_bit():
    """63488 values, two implementations, no value where they differ."""
    every = _every_finite_fp16()
    assert every.size == 63488, every.size
    mine = chords(every).view(np.uint16)
    theirs = _pwl_body(every, _GELU).view(np.uint16)
    differing = int((mine != theirs).sum())
    assert differing == 0, (
        f"{differing} of {every.size} finite fp16 come out of the two walks with"
        " different bits, so blob_interpreter is no longer transcribing the table"
        " in unary_ops.cc -- or the index, the fold, the saturation or the rounding"
        " of the product has moved"
    )


def test_the_walk_also_reproduces_the_phone_bits_that_were_recorded():
    """The device tier, reached through an implementation that is not the model."""
    recorded = test_unary_pwl._GOLDEN["gelu"]
    assert set(recorded) == {63, 64, 100}, sorted(recorded)
    for length, cases in recorded.items():
        for at, value, bits in cases:
            probe = np.array([value], dtype=np.float16)
            # The three recorded lengths are 63, 64 and 100, so they are the
            # boundary itself: below the grain the whole buffer is the scalar
            # form, at it the whole buffer is the chords, and 100 puts both in
            # one buffer. The position, not the value, is what says which.
            # The length picks the form, not the index: a 63-element buffer is
            # the scalar form throughout because the walk covers [0, numel & ~63)
            # and that is empty. Index 1 of 63 is the scalar form and index 1 of
            # 64 is a chord.
            body = length - length % _GRAIN
            form = chords if at < body else scalar_form
            got = test_unary_pwl._fold(int(form(probe).view(np.uint16)[0]))
            assert got == test_unary_pwl._fold(bits), (
                f"{length} elements, index {at}, input {value!r}: this file says"
                f" 0x{got:04X} and the phone said 0x{bits:04X}"
            )


def test_the_scalar_form_is_a_different_function_and_the_model_does_not_use_it():
    """The control: what makes the three assertions above worth reading."""
    every = _every_finite_fp16()
    walk = chords(every).view(np.uint16)
    scalar = scalar_form(every).view(np.uint16)
    differing = int((walk != scalar).sum())
    assert differing > 0, (
        "the chords and the scalar tanh form answer every finite fp16 alike, so"
        " an interpreter running either would satisfy the tests above and the"
        " control would have nothing to hold"
    )
    assert differing > every.size // 2, (
        f"the two forms separate on {differing} of {every.size} finite fp16, not"
        " more than half, so this is not the shape of a rounding difference"
    )
    gap = np.abs(chords(every).astype(np.float32) - scalar.astype(np.float32)).max()
    assert gap > 6.0e-3, (
        f"the two forms come within {gap:.3e} of each other, so the wrong one"
        " would not be the 6.16e-3 the export now warns about"
    )
    # and the model is on the chords' side of the disagreement, by name
    at = int(np.argmax(np.abs(chords(every).astype(np.float32) - scalar.astype(np.float32))))
    assert _pwl_body(every[at : at + 1], _GELU)[0] == chords(every[at : at + 1])[0], (
        f"at the input {float(every[at])!r} where the two forms are furthest apart"
        " the model answers the scalar form, so the model is not the DSP's"
    )


def test_the_tail_is_the_scalar_form_and_the_body_is_not():
    """The split, through execute: both halves, at every length around the grain."""
    for length in (1, 33, 63, 64, 65, 127, 128, 129, 256):
        values = np.linspace(-9.0, 9.0, length).astype(np.float16)
        body = length - length % _GRAIN
        got = _dsp_answer(values)
        assert np.array_equal(got[:body], chords(values[:body])), (
            f"{length} elements: the first {body} are not the chords"
        )
        assert np.array_equal(got[body:], scalar_form(values[body:])), (
            f"{length} elements: the last {length - body} are not the scalar form"
        )
    # -0.125, because the two forms separate there and -0.25 does not: one
    # buffer, one value, two answers, and the position is what says which. The
    # same case is recorded from the phone in test_unary_pwl at lengths 63, 64
    # and 100, so the assertion below is about the sweep above rather than about
    # the value -- it has to hold at every length the sweep reaches, not only at
    # the three the device answered.
    for length in (64, 65, 100):
        filled = np.full(length, np.float16(-0.125), dtype=np.float16)
        got = _dsp_answer(list(filled))
        assert got[0] == chords(filled[:1])[0], (
            f"{length} elements: index 0 is not the chords"
        )
        assert got[_GRAIN - 1] == chords(filled[_GRAIN - 1 : _GRAIN])[0], (
            f"{length} elements: index {_GRAIN - 1} is not the chords"
        )
        assert got[0] != scalar_form(filled[:1])[0], (
            "-0.125 answers the same number in both forms, so this buffer cannot"
            " tell the walk from the tail and the sweep above proves nothing"
        )
