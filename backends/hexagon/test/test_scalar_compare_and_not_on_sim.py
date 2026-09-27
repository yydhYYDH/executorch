# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The two routes a literal comparison and a mask negation take, run in hexagon-sim.

Everything here is the SIMULATOR tier: the vendored eltwise_ops.cc compiled for
v79 by the SDK's own hexagon-clang and executed under a simulated QuRT, so this
is the DSP's arithmetic and not a model of it. It says nothing about a phone and
nothing about the runtime that would carry a blob out to a caller.

Four questions the host cannot answer about the commands its own emitters write.
The first is which branch a zero stride reaches: `_emit_compare` passes the
binary broadcast descriptor, and the kernel's `in1Size == 1` arm is separate
code with its own IEEE behaviour on a NaN -- the two are compared here rather
than assumed equivalent, and they are not. The second is the negotiation:
`logical_not` is an all-1 select, and the only thing that could get it wrong is
the widths in the call, so two mis-spellings are measured beside the right one.
The third is whether a route measured at sixteen elements holds at 32. The fourth
is what a conjunction of two masks costs, which is the question a bool and/or
asks and which the host cannot answer by reading its own emitter.

The sixteen values are the corner list the comparison-route file uses, read
here against a threshold rather than against each other: the two signed zeros, a
NaN, each infinity, and the fp16 extremes.
"""

import pathlib

import pytest
import torch

import hexagon_sim
import test_blob_on_sim as blob_sim

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/scalar_not_runner.cpp"

#: The values the runner is handed, in the order it prints them.
_A = [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), float("-inf"),
      2.0, -2.0, 0.5, -0.5, 3.0, 1e-4, -1e-4, 65504.0, -65504.0]

_TAGS = (
    "A", "GT0", "LT0", "GT1", "LT1", "ARM0", "ARM1", "TIE0", "TIEVEC", "RC0",
    "PACK", "NOT", "NOTAGAIN", "NOTTWO", "NOTWIDE", "MIXEDIN", "MIXEDOUT",
    "GTPKG", "LTBIG", "AND", "OR", "ANDWIDE",
)


def _half_bits(values):
    return [int(v) for v in torch.tensor(values, dtype=torch.float16).view(torch.uint16)]


def _as_half(words):
    return torch.tensor(words, dtype=torch.int64).bitwise_and(0xFFFF).to(torch.uint16).view(
        torch.float16
    )


def _flags(words):
    """A printed flag run as 0/1, the way the packing select reads it."""
    return [1 if float(x) else 0 for x in _as_half(words)]


@pytest.fixture(scope="module")
def ran():
    try:
        hexagon_sim._check()
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))
    sources = [source for source in blob_sim._SOURCES if isinstance(source, str)]
    try:
        results = hexagon_sim.run(
            _RUNNER,
            sources,
            headers={"htp_ops.h": blob_sim._HT_P_OP_SHIM},
        )
    except hexagon_sim.Unavailable as error:
        # _check answered above, so the SDK and the simulator are both here and
        # this is the simulator failing to run the runner -- a failure, and the
        # only thing between this file and a green no-op.
        raise AssertionError(f"hexagon-sim did not run the runner: {error}")
    for tag in _TAGS:
        # A missing tag and an empty result are the same dict lookup, so every
        # tag is asked for by name before any answer is read.
        assert tag in results, f"the runner printed no {tag}"
    return results


def test_the_runner_was_handed_the_sixteen_values_this_file_names(ran):
    """The inputs first, because a route that disagreed could not be one it never saw."""
    assert ran["A"] == _half_bits(_A), (ran["A"], _half_bits(_A))
    assert ran["RC0"] == [0], f"the command returned {ran['RC0']}"


def test_a_literal_comparison_answers_torch_through_the_broadcast_descriptor(ran):
    """`x > 0` as the emitter spells it, at two thresholds, against torch.

    The signed zeros, the NaN and the infinities are the load-bearing elements.
    A pack that tested the raw flag bits for nonzero would call -0.0 set; a
    compare that was not IEEE would call the NaN set. Both would be different
    implementations of this same route, and the corner list is where they show.
    """
    a = torch.tensor(_A, dtype=torch.float16)
    for tag, threshold, op in (
        ("GT0", 0.0, torch.gt),
        ("LT0", 0.0, torch.lt),
        ("GT1", 1.0, torch.gt),
        ("LT1", 1.0, torch.lt),
    ):
        want = [1 if bool(x) else 0 for x in op(a, torch.tensor(threshold, dtype=torch.float16))]
        assert _flags(ran[tag]) == want, (tag, _flags(ran[tag]), want)
    # Named rather than left inside the vector comparison, because these two are
    # what an IEEE compare has to get right and a bit test does not.
    assert not bool(a[1] > 0) and not bool(a[4] > 0) and not bool(a[4] < 0)
    assert bool(a[5] > 0) and not bool(a[6] > 0)


def test_the_kernels_own_scalar_arm_is_a_different_branch_and_not_an_equivalent_one(ran):
    """The spelling the emitter did not choose, and why that is the right answer.

    `htp_ops_binary_elementwise` has an `in1Size == 1` arm that needs no
    descriptor, and a host that passed in1Size alone would take it. Over these
    sixteen it agrees with the descriptor on fifteen and disagrees on the NaN,
    where it answers 1.0 and torch answers false: the scalar arm's comparison is
    not IEEE on an unordered pair. So "a literal reaches the same compare" is
    true of the descriptor and not of the arm, and the emitter writes the
    descriptor. The less direction is the control: the same arm, the other
    op type, no disagreement.
    """
    diff = [i for i, (a, b) in enumerate(zip(_flags(ran["ARM0"]), _flags(ran["GT0"]))) if a != b]
    assert diff == [4], (diff, _flags(ran["ARM0"]), _flags(ran["GT0"]))
    assert _as_half(ran["ARM0"])[4] == 1.0, "the arm calls a NaN greater than zero"
    assert _flags(ran["ARM0"]) != [1 if bool(x) else 0 for x in
                                   torch.tensor(_A, dtype=torch.float16) > 0.0]
    assert _flags(ran["ARM1"]) == _flags(ran["LT1"]), "the less arm agrees"


def test_the_runner_is_comparing_rather_than_returning_a_constant(ran):
    """A control that the sixteen are not all the same answer.

    `TIE0` compares the input against itself with the same op type through the
    descriptor and `TIEVEC` through the elementwise arm, so every element must
    be zero in both. That is what makes `GT0`'s ones a comparison rather than a
    buffer the kernel filled.
    """
    assert _flags(ran["TIE0"]) == [0] * len(_A)
    assert _flags(ran["TIEVEC"]) == [0] * len(_A)
    assert {float(x) for x in _as_half(ran["GT0"])} == {0.0, 1.0}


def test_a_packing_select_reads_the_two_byte_flags_it_is_given(ran):
    """The first half of a comparison's two commands, on its own.

    `_emit_compare` declares the condition two bytes wide because the flags
    are fp16, and packs to one byte a result. The runner does the same call
    with a two-byte condition and prints the destination, so the packing is a
    measurement rather than something read off the emitter.
    """
    assert ran["PACK"] == _flags(ran["GT0"]), (ran["PACK"], _flags(ran["GT0"]))
    assert set(ran["PACK"]) <= {0, 1} and len(ran["PACK"]) == len(_A)


def test_a_negation_is_a_one_byte_select_that_reads_a_one_byte_condition(ran):
    """`logical_not` end to end: one command, and the bytes say which arm ran.

    The condition here is the *packed* mask. Reading the flags as one byte
    instead is the same call with a two-byte buffer behind it, and `NOTTWO`
    records what that answers -- so the assertion is about the emitter's
    spelling rather than about a select in general.
    """
    assert ran["NOT"] == [1 - b for b in ran["PACK"]], (ran["NOT"], ran["PACK"])
    assert len(ran["NOT"]) == len(_A) and set(ran["NOT"]) <= {0, 1}
    # Two negations are the identity, which is a statement about the pairing of
    # the two constants rather than about either of them alone.
    assert ran["NOTAGAIN"] == ran["PACK"] and ran["NOTAGAIN"] != ran["NOT"]


def test_the_flags_read_as_a_one_byte_condition_answer_something_else(ran):
    """The misreading, recorded so the right spelling above is not a tautology.

    A byte of an fp16 1.0 is 0x00, so a one-byte read of the flag run sees a
    row of zeroes in one half of the buffer and 0x3C in the other. The negation
    then answers from those, and it is not the negation of the mask. This is the
    failure a packed-bool slot is there to avoid, and it is why the select after
    a comparison declares the condition two bytes wide and the select after a
    comparison's result declares it one.
    """
    assert ran["NOTTWO"] != ran["NOT"], (ran["NOTTWO"], ran["NOT"])


def test_a_condition_declared_two_bytes_wide_answers_differently(ran):
    """The other mis-spelling, in the other direction: pairs the elements up.

    If this answered the same as `NOT` then the widths in the call would not be
    what decides the answer, and every other assertion in the file would be about
    a parameter that does nothing.
    """
    assert ran["NOTWIDE"] != ran["NOT"], (ran["NOTWIDE"], ran["NOT"])


def test_the_condition_test_is_nonzero_and_not_equal_to_one(ran):
    """torch's logical_not is "nonzero becomes False", so 0xff is False too.

    A torch.bool slot only ever carries 0 or 1, so this is a robustness claim
    rather than a correctness requirement -- and it is checked because the
    alternative spelling, `== 1`, would be wrong for any producer that hands
    the select a byte the graph did not write.
    """
    assert ran["MIXEDIN"][:3] == [0x00, 0xFF, 0x01]
    assert ran["MIXEDOUT"][:4] == [1, 0, 0, 1], ran["MIXEDOUT"][:4]


def test_two_masks_conjoin_over_the_flags_the_comparisons_left(ran):
    """`and` and `or` as MIN and MAX over fp16 0.0 and 1.0, and no kernel.

    The question a bool and/or asks has two answers and this is the first. Two
    comparisons in the same island leave their results as fp16 1.0 and 0.0 before
    the select packs them, and MIN(6) and MAX(5) combine those elementwise, so a
    conjunction of two *unpacked* masks is two more commands over flags that
    already exist rather than a new op type. The thresholds are a zero and the
    fp16 extreme, so the sixteen elements carry six trues for the conjunction
    and seven for the disjunction: neither is a constant.
    """
    a = torch.tensor(_A, dtype=torch.float16)
    gt, ltb = a > 0.0, a < 65504.0
    assert ran["GTPKG"] == [1 if bool(x) else 0 for x in gt]
    assert ran["LTBIG"] == [1 if bool(x) else 0 for x in ltb]
    assert ran["AND"] == [1 if (bool(g) and bool(l)) else 0 for g, l in zip(gt, ltb)]
    assert ran["OR"] == [1 if (bool(g) or bool(l)) else 0 for g, l in zip(gt, ltb)]
    # Neither answer is a constant, and the conjunction is a subset of the
    # disjunction: a route that filled the buffer with one bit would fail both.
    assert 0 < sum(ran["AND"]) < len(_A) and 0 < sum(ran["OR"]) < len(_A)
    assert all(a <= o for a, o in zip(ran["AND"], ran["OR"])), (ran["AND"], ran["OR"])


def test_the_same_conjunction_over_packed_masks_answers_wrongly(ran):
    """The second answer: once a mask is one byte, there is nothing to combine.

    The binary command declares two-byte operands, so a packed mask read at that
    width is not a flag at all: a byte of 0x01 inside the half-float 0x0001 is a
    subnormal, and the byte after it belongs to the *next* element, so the
    conjunction reads pairs and answers neither operand. There is no command that
    unpacks a bool, so `bitwise_and` over two packed masks is a new op type and
    not a composition, and this is the measurement that says so.
    """
    a = torch.tensor(_A, dtype=torch.float16)
    want = ran["AND"]
    assert want != [0] * len(_A), "the control has to have a true in it"
    assert ran["ANDWIDE"] != want, (ran["ANDWIDE"], want)
    assert ran["ANDWIDE"] != ran["OR"], "it is not the disjunction either"
