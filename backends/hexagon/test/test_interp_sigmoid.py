# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The two sigmoids the DSP has, and the rounding that tells them apart.

htp_ops_unary_compute_fp16_chunk is two implementations of gelu, sigmoid and
tanh, and the length of the buffer decides which one answers an element. The
first _PWL_GRAIN elements are walked by htp_ops_unary_pwl_fp16_vec, which
evaluates a chord as a companded16 table lookup and an fp16 multiply into an fp16
add; the remainder goes element by element through htp_ops_unary_apply_fp16,
where sigmoid is the exact 1/(1+expf(-x)) in fp32 (unary_ops.cc:174-177,
:489-491). They do not approximate one another, and at -9 the chords return
exactly 0 where the scalar form returns 1.2341e-4. So the two are compared to
each other here and never to np.sigmoid: a band measured against the ideal
function certifies a comparison the DSP never makes.

This file used to assert the opposite rounding, and the record is worth keeping
because the test was worse than no test. Its
test_chords_round_once_at_the_end_of_an_fp16_multiply_accumulate opened by
saying the kernel evaluates a chord as "an fp16 multiply accumulated into an fp16
add (pwl.h:90-93)" and that "a model that never rounded, or that carried fp32
across the multiply, would pass it and be wrong". It then closed by asserting
np.allclose(chords, fp32_path), where fp32_path was

    (abs(x).astype(np.float32) * slope.astype(np.float32) + bias.astype(np.float32)).astype(np.float16)

which is the fp32-across-the-multiply form its own comment had just named as the
wrong one. It was not a tautology; it pinned the defect. Re-measured against
_pwl_body, the form the phone's recorded tanh answers agree with, all three of
its discriminating assertions fail: the allclose differs on 10 of 129 elements,
its delta <= ulp/2 bound is exceeded at 2.861023e-04 against a 2.441406e-04
allowance, and its delta <= 0.5 * 4.8828125e-4 ceiling is exceeded at the same
value. A test that rejects the correct arithmetic is a lock on the wrong one, so
the case is rebuilt below to demand the two roundings and to fail if the fp32
product is ever substituted.

What is here is the unit view of the two halves. The grain itself, end to end
through execute() on a real blob, is test_sigmoid_grain.py's, because a boundary
nobody exercises through the entry point is not a boundary.
"""

import unittest

import numpy as np

from blob_interpreter import _PWL_BIAS, _PWL_SLOPE, _UNARY, _pwl_body

#: vec_len is 128 bytes over fp16 (unary_ops.cc:334) and the walk covers
#: [0, numel & ~63).
_GRAIN = 64

#: Where sigmoid's chords stop, and the only one of the three that is eight
#: rather than four (unary_ops.cc:269-270).
_RANGE = 8.0

#: The ceiling test_unary_pwl.py records for sigmoid's own distance from the
#: definition, 2.4e-3 measured, sitting just under a rounder number.
_BAND = 2.5e-3

_SLOPE = np.asarray(_PWL_SLOPE[4], np.uint16).view(np.float16)
_BIAS = np.asarray(_PWL_BIAS[4], np.uint16).view(np.float16)


def _scalar(x):
    """htp_ops_unary_apply_fp16's SIGMOID: fp32, rounded once to fp16.

    Kept independent of the module under test so the two halves cannot be
    compared against a shared mistake.
    """
    x = np.asarray(x, dtype=np.float32)
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float16)


def _chord_index(abs_x):
    """htp_ops_pwl_companded_index16 (pwl.h:57-71), re-derived from the source.

    Deliberately not imported: the rounding case below has to be able to tell a
    wrong index from a wrong rounding, and reusing the implementation's own index
    would let one hide inside the other.
    """
    abs_x = np.ascontiguousarray(np.asarray(abs_x, np.float16))
    scaled = np.minimum((abs_x * np.float16(4.0)).astype(np.float16), np.float16(15.0))
    low = (scaled + np.float16(16.0)).astype(np.float16).view(np.uint16).astype(np.int64) >> 6
    wide = ((abs_x.view(np.uint16).astype(np.int64) >> 8) & 7) + 8
    return np.where(abs_x >= np.float16(2.0), wide, low) & 15


def _chord_fp16_product(x):
    """The kernel's chord: fp16 multiply, then fp16 add (pwl.h:90-93)."""
    x = np.ascontiguousarray(np.asarray(x, np.float16))
    negative = x < np.float16(0.0)
    abs_v = np.where(negative, (-x).astype(np.float16), x)
    index = _chord_index(abs_v)
    positive = (abs_v * _SLOPE[index]).astype(np.float16) + _BIAS[index]
    folded = (np.float16(1.0) - positive).astype(np.float16)
    limit = np.where(negative, np.float16(0.0), np.float16(1.0))
    return np.where(
        abs_v < np.float16(_RANGE), np.where(negative, folded, positive), limit
    ).astype(np.float16)


def _chord_fp32_product(x):
    """The form the kernel is not: the same multiply carried in fp32."""
    x = np.ascontiguousarray(np.asarray(x, np.float16))
    negative = x < np.float16(0.0)
    abs_v = np.where(negative, (-x).astype(np.float16), x)
    index = _chord_index(abs_v)
    positive = (
        abs_v.astype(np.float32) * _SLOPE[index].astype(np.float32)
        + _BIAS[index].astype(np.float32)
    ).astype(np.float16)
    folded = (np.float32(1.0) - positive.astype(np.float32)).astype(np.float16)
    limit = np.where(negative, np.float16(0.0), np.float16(1.0))
    return np.where(
        abs_v < np.float16(_RANGE), np.where(negative, folded, positive), limit
    ).astype(np.float16)


def _ideal(x):
    """The model this file replaced: sigmoid as a function of the value alone."""
    return _scalar(x)


class ChordArithmeticTest(unittest.TestCase):
    def test_a_chord_is_an_fp16_multiply_into_an_fp16_add(self):
        """The kernel's two roundings, and the fp32 form refused by name.

        Q6_Vqf16_vmpy_VhfVhf returns an fp16 product (pwl.h:91), so the bias is
        added to a value that has already been rounded. The two forms are not
        equal and this says so on both sides: the chords match the fp16 product
        everywhere, and differ from the fp32 product somewhere.
        """
        x = np.linspace(-7.5, 7.5, 129).astype(np.float16)
        chords = _pwl_body(x, 4)
        np.testing.assert_array_equal(chords, _chord_fp16_product(x))

        fp32_form = _chord_fp32_product(x)
        differing = int((chords.view(np.uint16) != fp32_form.view(np.uint16)).sum())
        self.assertGreater(
            differing, 0,
            "the fp32 product agrees with the fp16 product on this sample, so "
            "this case cannot tell the two roundings apart",
        )
        distance = np.abs(chords.astype(np.float64) - fp32_form.astype(np.float64)).max()
        self.assertGreater(distance, 0.0)

    def test_the_two_roundings_separate_at_minus_three(self):
        """One named input, so the failure says which form is installed."""
        x = np.array([-3.0], dtype=np.float16)
        np.testing.assert_array_equal(
            _pwl_body(x, 4), np.array([0.0478515625], dtype=np.float16)
        )
        np.testing.assert_array_equal(
            _chord_fp32_product(x), np.array([0.04736328125], dtype=np.float16)
        )

    def test_the_deviation_from_float64_is_two_roundings_and_not_one(self):
        """Bounded by the kernel's arithmetic rather than by a loose tolerance.

        Evaluated in float64 the same slope and bias, the chords must differ
        somewhere: both forms round at least once, and the fp16 product rounds
        twice. What separates them is that the fp16 form's inner rounding is
        visible -- it cannot be inside the half-ulp the final rounding alone
        would allow, which is what the old version of this case measured and
        measured wrongly, against the fp32 form.
        """
        x = np.linspace(-7.5, 7.5, 129).astype(np.float16)
        index = _chord_index(np.abs(x))
        wide = (
            np.abs(x).astype(np.float64) * _SLOPE[index].astype(np.float64)
            + _BIAS[index].astype(np.float64)
        )
        delta = np.abs(_pwl_body(x, 4).astype(np.float64) - wide)
        self.assertGreater(int((delta > 0).sum()), 0, "the chords never rounded")
        # The fp32 product rounds once, so its deviation is inside half an ulp of
        # the value it rounds. The kernel's does not, and the margin is recorded
        # so a refitted table cannot quietly move it back inside.
        rounded = wide.astype(np.float16).astype(np.float64)
        half_ulp = np.abs(
            np.nextafter(rounded, np.float16(np.inf)).astype(np.float64) - rounded
        ) / 2.0
        self.assertGreater(
            float((delta > half_ulp + 1e-12).sum()),
            0,
            "every chord is inside a single rounding, so the inner fp16 "
            "multiply is not happening",
        )


class TwoImplementationsTest(unittest.TestCase):
    def test_minus_nine_saturates_to_zero_where_the_scalar_path_does_not(self):
        x = np.array([-9.0], dtype=np.float16)
        self.assertEqual(float(_pwl_body(x, 4)[0]), 0.0)
        self.assertAlmostEqual(float(_scalar(x)[0]), 1.2341e-4, places=7)
        self.assertNotEqual(float(_scalar(x)[0]), 0.0)

    def test_eight_saturates_to_one_where_the_scalar_path_returns_just_under(self):
        x = np.array([8.0], dtype=np.float16)
        self.assertEqual(float(_pwl_body(x, 4)[0]), 1.0)
        self.assertLess(float(_scalar(x)[0]), 1.0)

    def test_the_saturation_is_the_vector_paths_alone(self):
        """_UNARY[4] has no comparison in it, which is the kernel's own shape.

        htp_ops_unary_apply_fp16's sigmoid is three lines and returns the true
        sigmoid at any magnitude (unary_ops.cc:174-177); the clamp at 8 belongs
        to the chord walk (unary_ops.cc:327-329). Reading a saturated value out
        of _UNARY[4] would be a host model inventing a guard the kernel has not
        got in that path.
        """
        for value in (-9.0, 9.0):
            x = np.array([value], dtype=np.float16)
            self.assertEqual(float(_pwl_body(x, 4)[0]), 0.0 if value < 0 else 1.0)
            self.assertNotEqual(float(_UNARY[4](np.float32(x))[0]), 0.0 if value < 0 else 1.0)

    def test_the_chords_stay_inside_the_recorded_band(self):
        """Distance from the definition, against the ceiling the device set."""
        x = np.linspace(-_RANGE, _RANGE, 4096).astype(np.float16)
        chords = _pwl_body(x, 4).astype(np.float64)
        want = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
        worst = float(np.abs(chords - want).max())
        self.assertLessEqual(
            worst, _BAND, f"sigmoid's chords are off by {worst:.3e}, past {_BAND:.3e}"
        )
        self.assertGreater(worst, _BAND / 64, "the chords are exact, which they are not")

    def test_the_scalar_entry_is_the_kernel_s_scalar_form(self):
        """_UNARY[4] on its own, against a reference written from the C++."""
        x = np.linspace(-40.0, 40.0, 1024).astype(np.float32)
        np.testing.assert_array_equal(
            np.asarray(_UNARY[4](x), np.float16).view(np.uint16),
            _scalar(x).view(np.uint16),
        )

    def test_the_split_is_the_grain_and_nothing_else(self):
        """vec_end = numel & ~63, stated as arithmetic rather than end to end.

        The end-to-end version, which is the one that would catch a dispatch
        change, is test_sigmoid_grain.py's.
        """
        for numel in (1, 33, 63, 64, 65, 127, 128, 129, 256):
            self.assertEqual(numel & ~(_GRAIN - 1), numel - numel % _GRAIN)


class IdealModelWasWrongTest(unittest.TestCase):
    """The red that motivated the transcription, kept as a test.

    Each case is a statement the previous model could not satisfy. They are not
    tolerances loosened to make a suite green; they are the comparisons that
    were certifying an ideal function the DSP never computes.
    """

    def test_the_ideal_model_could_not_reach_zero_at_minus_nine(self):
        x = np.array([-9.0] * _GRAIN, dtype=np.float16)
        self.assertEqual(float(_ideal(x)[0]), float(_scalar(x)[0]))
        self.assertNotEqual(float(_pwl_body(x, 4)[0]), float(_ideal(x)[0]))

    def test_the_ideal_model_agreed_with_the_scalar_path_on_chord_elements(self):
        """The failure in one sentence: a short-tensor-only suite stays green."""
        rng = np.random.default_rng(7)
        x = rng.uniform(-_RANGE, _RANGE, 128).astype(np.float16)
        ideal = _ideal(x).astype(np.float32)
        scalar = _scalar(x).astype(np.float32)
        chords = _pwl_body(x, 4).astype(np.float32)
        self.assertEqual(int((ideal != scalar).sum()), 0)
        self.assertGreater(int((chords != scalar).sum()), 0)
        self.assertGreater(float(np.abs(chords - ideal).max()), 1e-3)

    def test_the_ideal_model_had_no_chord_error_at_all(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(-_RANGE, _RANGE, 4096).astype(np.float16)
        ideal_error = np.abs(
            _ideal(x).astype(np.float32) - _scalar(x).astype(np.float32)
        ).max()
        chord_error = np.abs(
            _pwl_body(x, 4).astype(np.float32) - _scalar(x).astype(np.float32)
        ).max()
        self.assertLess(float(ideal_error), 1e-7)
        self.assertGreater(float(chord_error), 1e-3)


if __name__ == "__main__":
    unittest.main()
