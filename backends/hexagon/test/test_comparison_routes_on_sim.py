# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Can a comparison be two commands instead of a new kernel? Run the two.

The question is whether `htp_ops_select` or `htp_ops_reduction` can stand in for a
binary comparison kernel, and the answer differs per op, so the cases here are
per-op rather than per-family. Both kernels are real DSP code: this file
compiles the vendored sources and runs them in hexagon-sim, so what it measures
is what the kernels do and not a model of them.

The sixteen pairs are chosen for their order relations rather than for their
values. Between the two signed zeros, a NaN against itself, each infinity
against itself, and the fp16 extremes is every case where a decomposition of a
comparison into arithmetic plus a nonzero test can be wrong, and each of the
routes below fails on a different one. That is the finding: a route that is
right on random data can still be wrong here, so the corners are the test.
"""

import pathlib

import pytest
import torch


import hexagon_sim
import test_blob_on_sim as blob_sim

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/cmp_route_runner.cpp"

#: (a, b) per element. Chosen for the order relation, not the magnitude: 0 the
#: two zeros, 1 the signed zero against the unsigned one, 2 and 11 two ordinary
#: equalities, 3/7/8/9/10/13/15 seven orderings, 4 a NaN against itself, 5 and 6
#: each infinity against itself, 14 the fp16 extremes.
_A = [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), float("-inf"),
      2.0, -2.0, 0.5, -0.5, 3.0, 1e-4, -1e-4, 65504.0, -65504.0]
_B = [0.0, 0.0, 1.0, 1.0, float("nan"), float("inf"), float("-inf"),
      1.0, -1.0, 0.25, -0.25, 3.0, 1e-4, 1e-4, -65504.0, 65504.0]

#: AEE_EBADPARM, which is what the reduction returns for a width it does not
#: take. AEE_EOFFSET + 0x00E from the SDK's own incs/stddef/AEEStdErr.h, so the
#: number is read off that header rather than guessed at from the shape of it.
_AEE_EBADPARM = 0x8000040E


def _half_bits(values):
    return [int(v) for v in torch.tensor(values, dtype=torch.float16).view(torch.uint16)]


def _as_half(words):
    return torch.tensor(words, dtype=torch.int64).bitwise_and(0xFFFF).to(torch.uint16).view(
        torch.float16
    )


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
        # this is the simulator failing to run the runner. That is a failure and
        # not a skip, and the distinction is the only thing between this file and
        # a green no-op.
        raise AssertionError(f"hexagon-sim did not run the runner: {error}")
    for tag in (
        "A", "B", "SUB", "GT", "LT", "NEBYTE", "EQBYTE", "GTBYTE", "GEBYTE",
        "LEBYTE", "SUBABS", "NEABS", "EQABS", "SUBABSV", "EQABSV",
        "REDONE", "REDTWO", "REDMAX", "REDIN", "REDOUT",
    ):
        # A missing tag and an empty result are the same dict lookup, so every
        # tag is asked for by name before any answer is read.
        assert tag in results, f"the runner printed no {tag}"
    return results


def test_the_runner_and_the_reference_were_handed_the_same_sixteen_pairs(ran):
    """The operands are checked before any answer is, on both sides.

    A route that disagreed with torch on a pair the runner never received would
    look exactly like a route that disagrees on a pair it did, so the printed
    inputs are compared against the reference's own bytes first.
    """
    assert ran["A"] == _half_bits(_A), (ran["A"], _half_bits(_A))
    assert ran["B"] == _half_bits(_B), (ran["B"], _half_bits(_B))


def test_select_at_one_byte_really_writes_a_one_byte_result(ran):
    """The one-byte arm is a path, not a claim about it.

    Four destinations, four one-byte-per-element results, and each of them is
    either 0 or 1 over the whole span. A select that had fallen through to its
    two-byte arm would have written sixteen bytes and left the first eight
    looking like fp16 garbage rather than flags.
    """
    for tag in ("NEBYTE", "EQBYTE", "GTBYTE", "GEBYTE", "LEBYTE", "NEABS", "EQABS"):
        assert len(ran[tag]) == len(_A), tag
        assert set(ran[tag]) <= {0, 1}, (tag, ran[tag])
    # The condition is tested on the raw bits, so the expected value is derived
    # from the printed difference rather than from torch. Asking torch instead
    # would hide the one element where the two disagree, and that element is the
    # whole reason the route for equality does not work.
    nonzero_bits = [1 if word != 0 else 0 for word in ran["SUB"]]
    assert ran["NEBYTE"] == nonzero_bits
    assert ran["EQBYTE"] == [1 - word for word in nonzero_bits]


def test_the_dsp_greater_and_less_answer_torch_on_every_pair(ran):
    """The two op types the enum already has are exact, corners included.

    Zero disagreements over the sixteen, which is the whole case for routing gt
    and lt through what the library already ships: the compare is IEEE, so a NaN
    compares false and the two signed zeros compare equal.
    """
    a, b = torch.tensor(_A, dtype=torch.float16), torch.tensor(_B, dtype=torch.float16)
    assert [float(x) for x in _as_half(ran["GT"])] == [
        1.0 if bool(x) else 0.0 for x in a > b
    ]
    assert [float(x) for x in _as_half(ran["LT"])] == [
        1.0 if bool(x) else 0.0 for x in a < b
    ]
    assert ran["GTBYTE"] == [1 if bool(x) else 0 for x in a > b]
    assert len(ran["GTBYTE"]) == len(_A)


def test_the_negation_of_greater_is_not_greater_or_equal(ran):
    """De Morgan does not hold on an unordered pair, and that is the load-bearing case.

    a >= b written as "not (b > a)" is right on fifteen of the sixteen and wrong
    on the sixteenth, where both operands are NaN: neither is greater, so the
    negation is true, and torch says the comparison is false. A route checked on
    random data passes, which is why this case is in the suite and why the
    assertion names the index rather than the count alone.
    """
    a, b = torch.tensor(_A, dtype=torch.float16), torch.tensor(_B, dtype=torch.float16)
    want = [1 if bool(x) else 0 for x in a >= b]
    assert want != ran["GEBYTE"]
    mismatch = [i for i, (w, g) in enumerate(zip(want, ran["GEBYTE"])) if w != g]
    assert mismatch == [4], mismatch
    assert torch.isnan(a[4]) and torch.isnan(b[4])


def test_a_nonzero_test_of_the_difference_is_not_equality(ran):
    """a - b != 0 is not a != b, and the two places it fails are the corners.

    Three of the sixteen, and each for its own reason. The signed zero pair
    because the difference is -0.0 and select's test is "!= 0" on the raw bits,
    so -0.0 reads as set; the two infinities because the difference of two equal
    infinities is a NaN, which is also "!= 0". No number of extra commands in
    this route repairs the second, which is why equality is the one comparison
    that does not fit.
    """
    a, b = torch.tensor(_A, dtype=torch.float16), torch.tensor(_B, dtype=torch.float16)
    want_eq = [1 if bool(x) else 0 for x in a == b]
    want_ne = [0 if bool(x) else 1 for x in want_eq]
    assert [i for i, (w, g) in enumerate(zip(want_eq, ran["EQBYTE"])) if w != g] == [1, 5, 6]
    assert [i for i, (w, g) in enumerate(zip(want_ne, ran["NEBYTE"])) if w != g] == [1, 5, 6]
    difference = _as_half(ran["SUB"])
    assert int(difference[1].view(torch.uint16)) == 0x8000, "the signed zero pair"
    assert torch.isnan(difference[5]) and torch.isnan(difference[6])





def test_less_or_equal_is_the_mirror_and_misses_the_same_pair(ran):
    """le measured, not inferred from ge.

    Same command, operands swapped, same negation, same single miss on the NaN
    pair. Saying so without running it would be the one number in this file
    that nobody looked at.
    """
    a, b = torch.tensor(_A, dtype=torch.float16), torch.tensor(_B, dtype=torch.float16)
    want = [1 if bool(x) else 0 for x in a <= b]
    mismatch = [i for i, (w, g) in enumerate(zip(want, ran["LEBYTE"])) if w != g]
    assert mismatch == [4], mismatch
    assert torch.isnan(a[4]) and torch.isnan(b[4])


def test_the_repair_for_the_signed_zero_depends_on_the_length(ran):
    """The third command of the subtract-based route, run at two lengths.

    abs of the difference is the only unary op that reaches the signed zero, and
    it does not reach it the same way twice. Below 64 elements the unary kernel
    uses its scalar tail, which is `x < 0 ? -x : x` and leaves -0.0 alone; from
    64 it uses a mask of 0x7fff, which does not. So the same three commands on
    the same operands answer differently at 16 and at 128, and at 16 the repair
    repairs nothing. The two infinities survive at both lengths, so equality is
    not a subtraction at any length.
    """
    a, b = torch.tensor(_A, dtype=torch.float16), torch.tensor(_B, dtype=torch.float16)
    want_eq = [1 if bool(x) else 0 for x in a == b]
    want_ne = [0 if bool(x) else 1 for x in want_eq]

    # At sixteen elements the unary kernel takes its scalar tail, and that tail
    # is `x < 0 ? -x : x`, which leaves -0.0 alone because -0.0 is not less than
    # zero. So the repair fixes nothing here.
    assert int(_as_half(ran["SUBABS"])[1].view(torch.uint16)) == 0x8000
    assert [i for i, (w, g) in enumerate(zip(want_eq, ran["EQABS"])) if w != g] == [1, 5, 6]
    assert [i for i, (w, g) in enumerate(zip(want_ne, ran["NEABS"])) if w != g] == [1, 5, 6]
    # The one it fixed, named, so a later edit that loses it again is visible.
    assert [i for i, (w, g) in enumerate(zip(want_eq, ran["EQBYTE"])) if w != g] == [1, 5, 6]

    # At 128 the same three commands take the vector path, which is a mask of
    # 0x7fff rather than a comparison, and there -0.0 does become +0.0. The two
    # answers differ on the same elements of the same operands, so the repair is
    # a function of the tensor's length. Neither list is the equality, because
    # the two infinities survive both.
    assert int(_as_half(ran["SUBABSV"])[1].view(torch.uint16)) == 0x0000
    assert [i for i, (w, g) in enumerate(zip(want_eq, ran["EQABSV"])) if w != g] == [5, 6]
    assert torch.isnan(_as_half(ran["SUBABSV"])[5])

def test_the_reduction_refuses_one_byte_and_takes_two(ran):
    """The route that is refused, refused by the kernel and not by a clause.

    The same call at two widths: one byte is AEE_EBADPARM and two bytes is zero.
    A reduction has one source and no predicate, so it cannot compare two tensors
    at any width, and at the width a bool would need it does not run at all.
    """
    assert ran["REDONE"][0] == _AEE_EBADPARM
    assert ran["REDTWO"][0] == 0
    assert ran["REDMAX"][0] == 0


def test_any_over_zero_and_one_is_the_reduction_the_library_already_has(ran):
    """The identity the reduction does support, measured rather than quoted.

    any over a 0/1 row is MAXIMUM over the same row, and the DSP's answer agrees
    with torch's on all four rows. This is a fact about a reduction of a value,
    not about a comparison of two, and it is the shape the route would take if a
    comparison were available: a reduction consumes the bool, it does not make
    one.
    """
    values = _as_half(ran["REDIN"]).reshape(4, 4)
    assert [float(x) for x in _as_half(ran["REDOUT"])] == [
        float(x) for x in values.amax(dim=-1)
    ]
    assert [float(x) for x in _as_half(ran["REDOUT"])] == [
        1.0 if bool(x) else 0.0 for x in values.bool().any(dim=-1)
    ]
