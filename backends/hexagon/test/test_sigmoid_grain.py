# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The grain at which a sigmoid stops being the sigmoid, and the one rounding
that decides it.

Two branches transposed the same sixteen chords of htp_ops_unary_pwl_fp16_vec and
the merge kept both: _pwl_body with the banks under _PWL_SLOPE, and
_sigmoid_pwl_vector with its own copy under _SIGMOID_SLOPE. The tables are
bit-identical, the index arithmetic is the same function written twice, and both
saturate at 8 and fold the negative half through 1 - y. They differ in one
place, and it is the place the whole table turns on.

htp_ops_pwl_eval (pwl.h:90-93) is

    product = Q6_Vqf16_vmpy_VhfVhf(x, slope);
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(product, bias));

The qf16 prefix is the result's type, so the product comes back rounded to
fp16 before the bias is added: two fp16 roundings, not one. _pwl_body
reproduces that. _sigmoid_pwl_vector widened the multiply to fp32 and rounds
once. Over every finite fp16 in [-8, 8) the two disagree on 978 of 36866, by
at most 4.8828125e-04, which is one fp16 step at 0.5.

Two independent things say the fp32 form is the wrong one. The source above is
the kernel. And the device agrees: test_unary_pwl.py records the phone's bits
for this walk, and swapping the fp32 product into _pwl_body turns
test_the_host_model_reproduces_the_phone[tanh-64] red at tanh(-3.5), where the
model then says -0.99853515625 and the phone said -0.998046875. So the question
the two branches disagreed about is settled from the device tier, not settled by
preferring the shorter function.

What that leaves is a gap rather than a merge. Of the three activations the
recorded golden values cover, only tanh's straddle a rounding boundary, so the
sigmoid of the device golden is a test that passes on either transcription. And
the boundary is recorded for gelu alone. A sigmoid tensor is therefore free to
be walked by the wrong arithmetic and every test in the tree stays green, which
is what these tests exist to stop: the first pins the grain for sigmoid through
the interpreter's own entry point, the second pins the discontinuity at it, the
third pins the rounding the walk hands the elements below it, and the fourth is
the statement of what the recorded answers do and do not decide on their own.
"""

import numpy as np
import torch

import blob_interpreter as B
from blob_interpreter import execute, read_blob
from executorch.backends.hexagon.hexagon_backend import HexagonBackend
from executorch.exir import to_edge
from torch.export import export

#: HTP_OPS_UNARY_SIGMOID, the op type DSP_OP_UNARY carries in params[1].
_SIGMOID = 4

#: htp_ops_unary_compute_fp16_chunk's vec_len: 128 bytes over fp16
#: (unary_ops.cc:334), and the grain the walk is measured in.
_GRAIN = 64

#: Where the chords stop, for sigmoid alone (unary_ops.cc:269-270).
_RANGE = 8.0

#: The values the recorded device answers were taken at
#: (test_unary_pwl.py's _GOLDEN["sigmoid"][64]).
_GOLDEN_PROBE = (
    -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.5, -3.0,
    -2.0, -1.0, -0.5, -0.125, -0.03125, 0.0, 0.03125,
    0.125, 0.5, 1.0, 2.0, 3.5, 4.0, 9.0,
)


class _Sigmoid(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


def _index16(abs_v):
    """htp_ops_pwl_companded_index16 (pwl.h:57-71), re-derived from the source.

    Not imported from the module under test: the point of the rounding test is
    that a wrong index cannot hide inside a wrong rounding, so the segment is
    rebuilt here from the same three lines the kernel has.
    """
    abs_v = np.ascontiguousarray(np.asarray(abs_v, np.float16))
    scaled = np.minimum((abs_v * np.float16(4.0)).astype(np.float16), np.float16(15.0))
    low = (scaled + np.float16(16.0)).astype(np.float16).view(np.uint16).astype(np.int64) >> 6
    wide = ((abs_v.view(np.uint16).astype(np.int64) >> 8) & 7) + 8
    return np.where(abs_v >= np.float16(2.0), wide, low) & 15


def _chord(x, product_in):
    """htp_ops_unary_pwl_fp16_vec's body, with the product's type chosen by hand.

    product_in is "fp16" for the kernel's Q6_Vqf16_vmpy_VhfVhf and "fp32" for
    the transcription that widened it. Everything else -- the segment, the fold,
    the saturation at 8 -- is unary_ops.cc:259-330 exactly.
    """
    x = np.ascontiguousarray(np.asarray(x, np.float16))
    zero, one = np.float16(0.0), np.float16(1.0)
    negative = zero > x
    abs_v = np.where(negative, (-x).astype(np.float16), x)
    index = _index16(abs_v)
    slope = np.asarray(B._PWL_SLOPE[_SIGMOID], np.uint16)[index].view(np.float16)
    bias = np.asarray(B._PWL_BIAS[_SIGMOID], np.uint16)[index].view(np.float16)
    if product_in == "fp16":
        positive = (abs_v * slope).astype(np.float16) + bias
    else:
        positive = (
            abs_v.astype(np.float32) * slope.astype(np.float32) + bias.astype(np.float32)
        ).astype(np.float16)
    folded = (one - positive).astype(np.float16)
    limit = np.where(negative, zero, one)
    return np.where(
        abs_v < np.float16(_RANGE), np.where(negative, folded, positive), limit
    ).astype(np.float16)


def _scalar(x):
    """htp_ops_unary_apply_fp16's SIGMOID, unary_ops.cc:174-177, in fp32."""
    x = np.asarray(x, np.float32)
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float16)


def _dsp_answer(length, values):
    """What the command stream computes for values, through the interpreter."""
    x = torch.zeros(1, length, dtype=torch.float16)
    program = to_edge(export(_Sigmoid(), (x,))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _header, commands = read_blob(blob)
    assert len(commands) == 1, f"{length} elements: {len(commands)} commands"
    assert commands[0].params[0] == length and commands[0].params[1] == _SIGMOID, (
        f"the command is {list(commands[0].params)[:2]}, not a {length}-element sigmoid"
    )
    outputs = execute(blob, [np.ascontiguousarray(np.asarray(values, np.float16))])
    return np.frombuffer(outputs[0], dtype=np.float16)


def test_the_grain_is_where_a_sigmoid_stops_being_the_sigmoid():
    """63 is exact throughout, 64 is chords throughout, 65 splits 64 and 1.

    This is the boundary neither branch recorded for sigmoid. The walk covers
    [0, numel & ~63) and the remainder is the fp32 scalar form
    (unary_ops.cc:489-491), so a buffer one element short of the grain is not
    approximated at all and a buffer one element past it is almost entirely
    chords with a scalar remainder the model has to place by position.
    """
    for length in (1, 33, 63, 64, 65, 127, 128, 129, 256):
        values = np.linspace(-9.0, 9.0, length).astype(np.float16)
        body = length - length % _GRAIN
        got = _dsp_answer(length, values)
        assert np.array_equal(got[:body], _chord(values[:body], "fp16")), (
            f"{length} elements: the first {body} are not the chords"
        )
        assert np.array_equal(got[body:], _scalar(values[body:])), (
            f"{length} elements: the last {length - body} are not the scalar form"
        )


def test_the_same_value_answers_two_numbers_and_its_position_is_what_says_which():
    """The discontinuity is in the position, not in the value, and it sits at 64.

    -0.25 fills a 65-element buffer. Its first 64 copies take the chords and
    the last takes the scalar form, so one input vector carries two answers for
    one value; and appending a 65th element leaves the 64th alone, which is what
    makes the split a property of the length rather than of the contents. The
    value is chosen because the two forms do not agree there: 0.437988281 against
    0.437744141, where a value the two forms happen to share could not tell them
    apart at all.
    """
    got = _dsp_answer(65, np.full(65, np.float16(-0.25), dtype=np.float16))
    assert got[0] != got[64], (
        f"-0.25 answers {float(got[0])!r} at index 0 and {float(got[64])!r} at "
        "index 64, so this case cannot tell the walk from the scalar tail"
    )
    assert got[64] == _scalar(np.float32(-0.25)), "index 64 is not the scalar form"
    assert got[0] == _chord(np.float16(-0.25), "fp16"), "index 0 is not the chords"

    at_64 = _dsp_answer(64, np.full(64, np.float16(-0.25), dtype=np.float16))
    assert at_64[63] == got[63], (
        "the 64th element changed answer when a 65th was appended, so the walk "
        "is not covering a fixed prefix"
    )


def test_the_chord_rounds_its_product_to_fp16_and_the_two_forms_disagree():
    """Which of the two transcriptions is the kernel's, and by how much.

    -3.0 is one of the 978 inputs among the finite fp16 in [-8, 8) on which the
    two separate: the chords give 0.0478515625 and the fp32 product gives
    0.04736328125, one fp16 step at that magnitude, 4.8828125e-04 apart. The
    kernel's is the first, because Q6_Vqf16_vmpy_VhfVhf returns an fp16 product
    (pwl.h:91), and the phone's recorded tanh answers agree.
    """
    x = np.array([-3.0], dtype=np.float16)
    fp16_form = _chord(x, "fp16")
    fp32_form = _chord(x, "fp32")
    np.testing.assert_array_equal(fp16_form, np.array([0.0478515625], np.float16))
    np.testing.assert_array_equal(fp32_form, np.array([0.04736328125], np.float16))
    distance = abs(float(fp16_form[0]) - float(fp32_form[0]))
    assert distance > 0, (
        "the two transcriptions agree at -3.0, so this case cannot tell the "
        "product's rounding from anything else"
    )

    got = _dsp_answer(64, np.full(64, np.float16(-3.0), dtype=np.float16))
    assert np.all(got == fp16_form), (
        f"the interpreter answered {float(got[0])!r} at -3.0, not the kernel's "
        f"{float(fp16_form[0])!r}; the fp32 product would have said "
        f"{float(fp32_form[0])!r}, a distance of {distance:.6e}"
    )

    every = np.arange(0, 1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
    every = every[np.isfinite(every) & (np.abs(every.astype(np.float32)) <= np.float32(_RANGE))]
    a = _chord(every, "fp16").view(np.uint16)
    b = _chord(every, "fp32").view(np.uint16)
    disagreeing = int((a != b).sum())
    assert disagreeing == 978, (
        f"{disagreeing} of {every.size} finite fp16 in [-8, 8) separate the two "
        "transcriptions, not the 978 this records: the disagreement has moved, "
        "so the number quoted here is stale"
    )


def test_the_recorded_sigmoid_answers_do_not_choose_between_the_two_forms():
    """The device golden is a real test and it does not settle this on its own.

    Swapping the fp32 product into the chord body leaves every recorded sigmoid
    answer passing and turns only tanh's red. That is no reason to distrust the
    recorded values; it is the reason the rounding needs a case of its own,
    because a test that passes on either transcription is not evidence about the
    thing they disagree on.
    """
    probe = np.array(_GOLDEN_PROBE, dtype=np.float16)
    a = _chord(probe, "fp16").view(np.uint16)
    b = _chord(probe, "fp32").view(np.uint16)
    differing = [float(v) for v, u, w in zip(probe, a, b) if u != w]
    assert differing == [-3.0], (
        f"the recorded sigmoid probe now separates the two forms at {differing}, "
        "not at [-3.0] alone; this records where the discrimination is not"
    )
