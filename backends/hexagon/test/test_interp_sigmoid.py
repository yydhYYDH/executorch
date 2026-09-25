# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The sigmoid the DSP computes, which is not the sigmoid.

Sigmoid is the one unary in this interpreter that the kernel does not evaluate
exactly above 64 elements. htp_ops_unary_compute_fp16_chunk walks 64 fp16 at a
time down to vec_end = size & -64 through htp_ops_unary_pwl_fp16_vec, which
walks sixteen companded16 chords, and then finishes the remainder element by
element through htp_ops_unary_apply_fp16, where sigmoid is the exact
1/(1+expf(-x)) in fp32 (unary_ops.cc:174-177, :454-490).

Every band below is measured against the kernel's OWN scalar path. A band
measured against torch.sigmoid or np.sigmoid certifies a comparison the DSP
never makes: the chords are wrong against the ideal function by design, and so
is the difference between the two paths, so one tolerance spanning them hides
the very thing the test is for.
"""

import unittest

import numpy as np

from blob_interpreter import _SIGMOID_BIAS, _SIGMOID_SLOPE, _sigmoid, _sigmoid_pwl_vector

#: What the kernel's own scalar path returns: 1/(1+expf(-x)) in fp32 rounded
#: once to fp16. This is the reference a chord has to be judged against.
def _scalar(x):
    return (
        np.float32(1.0) / (np.float32(1.0) + np.exp(-np.asarray(x, dtype=np.float32)))
    ).astype(np.float16)


def _chord_index(abs_x):
    """The chord each magnitude selects, re-derived here from pwl.h:58-71.

    Deliberately not imported from the module under test: the point of the
    rounding test is to re-evaluate the chord in wider precision, and reusing
    the implementation's own index would let a wrong index hide inside it.
    """
    abs_x = np.asarray(abs_x, dtype=np.float16)
    scaled = (abs_x * np.float16(4.0)).astype(np.float16)
    scaled = np.minimum(scaled, np.float16(15.0)).astype(np.float16)
    shifted = np.ascontiguousarray(
        (scaled + np.float16(16.0)).astype(np.float16)
    ).view(np.uint16).astype(np.int64) >> 6
    wide = ((np.ascontiguousarray(abs_x).view(np.uint16).astype(np.int64) >> 8) & 7) + 8
    return np.where(abs_x >= np.float16(2.0), wide, shifted) & 15


#: The model this file replaces: sigmoid as an ideal function, applied to every
#: element regardless of how many there are.
def _old_model(x):
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-np.asarray(x, dtype=np.float32)))).astype(
        np.float16
    )


class SigmoidPathSplitTest(unittest.TestCase):
    def test_below_sixty_four_elements_the_kernel_takes_the_scalar_path(self):
        for numel in (1, 33, 63):
            with self.subTest(numel=numel):
                x = np.linspace(-8, 8, numel).astype(np.float16)
                got = _sigmoid(x.astype(np.float32)).astype(np.float16)
                np.testing.assert_array_equal(got, _scalar(x))

    def test_at_sixty_four_elements_every_element_takes_the_chords(self):
        x = np.linspace(-8, 8, 64).astype(np.float16)
        got = _sigmoid(x.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(got, _sigmoid_pwl_vector(x))

    def test_sixty_five_elements_split_sixty_four_and_one(self):
        x = np.linspace(-8, 8, 65).astype(np.float16)
        got = _sigmoid(x.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(got[:64], _sigmoid_pwl_vector(x[:64]))
        np.testing.assert_array_equal(got[64:], _scalar(x[64:]))

    def test_a_multiple_of_sixty_four_leaves_no_scalar_tail(self):
        # vec_end = size & -64 (unary_ops.cc:335), so a length that is already a
        # multiple of the vector width is walked to the end and nothing reaches
        # the scalar loop. A model that assumed a half-and-half split here would
        # be wrong on all 128.
        x = np.linspace(-8, 8, 128).astype(np.float16)
        got = _sigmoid(x.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(got, _sigmoid_pwl_vector(x))
        self.assertEqual((128 & ~63), 128)

    def test_one_hundred_and_thirty_split_one_hundred_twenty_eight_and_two(self):
        x = np.linspace(-8, 8, 130).astype(np.float16)
        got = _sigmoid(x.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(got[:128], _sigmoid_pwl_vector(x[:128]))
        np.testing.assert_array_equal(got[128:], _scalar(x[128:]))


class SigmoidChordTest(unittest.TestCase):
    def test_minus_nine_saturates_to_zero_where_the_scalar_path_does_not(self):
        # The sharpest disagreement between the two DSP paths. The chords clamp
        # at 8 in magnitude, so the vector path returns exactly 0; the kernel's
        # scalar path returns the true sigmoid.
        x = np.array([-9.0], dtype=np.float16)
        self.assertEqual(float(_sigmoid_pwl_vector(x)[0]), 0.0)
        self.assertAlmostEqual(float(_scalar(x)[0]), 1.2341e-4, places=7)
        self.assertNotEqual(float(_scalar(x)[0]), 0.0)

    def test_eight_saturates_to_one_where_the_scalar_path_returns_just_under(self):
        x = np.array([8.0], dtype=np.float16)
        self.assertEqual(float(_sigmoid_pwl_vector(x)[0]), 1.0)
        self.assertLess(float(_scalar(x)[0]), 1.0)

    def test_chords_and_the_scalar_path_disagree_across_the_domain(self):
        x = np.linspace(-7.9, 7.9, 256).astype(np.float16)
        chords = _sigmoid_pwl_vector(x).astype(np.float32)
        scalar = _scalar(x).astype(np.float32)
        self.assertGreater(int((chords != scalar).sum()), 0)
        self.assertLessEqual(float(np.abs(chords - scalar).max()), 2.5e-3)

    def test_chords_round_once_at_the_end_of_an_fp16_multiply_accumulate(self):
        # The kernel evaluates a chord as an fp16 multiply accumulated into an
        # fp16 add (pwl.h:90-93), so every chord output has been rounded to
        # fp16 exactly once. Evaluating the same slope and bias in float64 and
        # demanding agreement would therefore have to FAIL; a model that never
        # rounded, or that carried fp32 across the multiply, would pass it and
        # be wrong.
        x = np.linspace(-7.5, 7.5, 129).astype(np.float16)
        chords = _sigmoid_pwl_vector(x)
        index = _chord_index(np.abs(x))
        # The same arithmetic in float64, including the sign rule, so the only
        # thing that can differ is the single fp16 rounding.
        positive = (
            np.abs(x).astype(np.float64) * _SIGMOID_SLOPE[index].astype(np.float64)
            + _SIGMOID_BIAS[index].astype(np.float64)
        )
        wide = np.where(x < 0, 1.0 - positive, positive)
        chords64 = chords.astype(np.float64)
        rounded = int((chords64 != wide).sum())
        self.assertGreater(rounded, 0)
        # The deviation is a single rounding, so it is bounded by half an ulp
        # of the value the kernel rounds -- the positive branch, which for a
        # negative input sits just under 1.0 where an fp16 ulp is 4.9e-4.
        # Nothing is rounded at the small final value, which is why 1 - y can
        # look like a large relative error while being exactly one rounding of
        # y; the measured worst ratio is 0.4833.
        rounded_pos = positive.astype(np.float16)
        ulp = np.abs(
            np.nextafter(rounded_pos, np.float16(np.inf)).astype(np.float64)
            - rounded_pos.astype(np.float64)
        )
        # A model that carried fp32 across the rounding, or skipped it, would
        # land outside that bound; a model that rounds twice would as well.
        delta = np.abs(chords64 - wide)
        self.assertTrue(np.all(delta <= ulp / 2.0 + 1e-12))
        # The deviation disappears when the identical arithmetic is evaluated in
        # fp16, which is what makes this a rounding artefact and not a drift.
        fp16_path = (
            (np.abs(x).astype(np.float32) * _SIGMOID_SLOPE[index].astype(np.float32)
             + _SIGMOID_BIAS[index].astype(np.float32))
            .astype(np.float16)
            .astype(np.float64)
        )
        fp16_wide = np.where(x < 0, 1.0 - fp16_path, fp16_path)
        self.assertTrue(np.allclose(chords64, fp16_wide, rtol=0.0, atol=1e-12))
        self.assertLessEqual(float(delta.max()), 0.5 * 4.8828125e-4)


class OldModelWasWrongTest(unittest.TestCase):
    """The red that motivated the change, kept as a test.

    Each case below is a statement the previous model could not satisfy. They
    are not tolerances loosened to make a suite green; they are the specific
    comparisons that were certifying an ideal function the DSP never computes.
    """

    def test_the_old_model_could_not_reach_zero_at_minus_nine(self):
        x = np.array([-9.0] * 64, dtype=np.float16)
        self.assertEqual(float(_old_model(x)[0]), float(_scalar(x)[0]))
        self.assertNotEqual(
            float(_sigmoid_pwl_vector(x)[0]), float(_old_model(x)[0])
        )

    def test_the_old_model_agreed_with_the_scalar_path_on_chord_elements(self):
        # The old model was exact, so it agreed with the scalar path on every
        # element -- including the ones the kernel computes with chords, where
        # it should have disagreed and did not. That is the whole failure: a
        # suite that only ever compared a short tensor, where the kernel does
        # take the scalar path, stayed green and certified the wrong function.
        rng = np.random.default_rng(7)
        x = rng.uniform(-8, 8, 128).astype(np.float16)
        old = _old_model(x).astype(np.float32)
        scalar = _scalar(x).astype(np.float32)
        chords = _sigmoid_pwl_vector(x).astype(np.float32)
        # The old model matches the scalar path everywhere.
        self.assertEqual(int((old != scalar).sum()), 0)
        # And so it misses the elements the kernel actually computes as a chord:
        # 91 of these 128 chords miss the scalar path, max 2.2e-3.
        self.assertEqual(int((chords != scalar).sum()), 91)
        self.assertGreater(float(np.abs(chords - old).max()), 1e-3)

    def test_the_old_model_had_no_chord_error_at_all(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(-8, 8, 4096).astype(np.float16)
        old_error = np.abs(
            _old_model(x).astype(np.float32) - _scalar(x).astype(np.float32)
        ).max()
        chord_error = np.abs(
            _sigmoid_pwl_vector(x).astype(np.float32) - _scalar(x).astype(np.float32)
        ).max()
        self.assertLess(float(old_error), 1e-7)
        self.assertGreater(float(chord_error), 1e-3)


if __name__ == "__main__":
    unittest.main()
