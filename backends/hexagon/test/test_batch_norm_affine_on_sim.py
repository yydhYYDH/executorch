# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A batch norm's affine, run on hexagon-sim as one multiply and one add.

The host interpreter answers `x * s + b` from Python and the simulator answers
it from the real kernel behind `DSP_OP_BINARY_ELEMENTWISE`, so this covers the
descriptor as well as the arithmetic: the strides in the tail are what decides
where each element's coefficient is read from, and the only way to see that is to
run the command that carries them.

The shapes are the ones where the walk in the kernel is a boundary. HVX takes the
vector part down to `size & -64` and finishes the remainder in the scalar tail, so
an extent of 63 is all scalar, 64 is all vector, and 65 is 64 plus one: the last
element of a 65-long axis is the one a kernel that rounds the extent down answers
wrong, and the first is the one a kernel that rounds up answers wrong. 128 and
129 are the same boundary at the second vector, with 128 as the control between
them, and 66 is 64 plus two so that a walk which got 65 right by accident has two
to get right rather than one.

The inputs are all ones and the scale is a power of two, so every answer is
exactly its own coefficient and a wrong stride is a whole power of two rather
than a rounding step. The control is the same fixture with the constant's
channel-axis stride zeroed: the right shape, and every element reading the first
channel's coefficient instead of its own.

`hexagon_sim._check()` runs before anything else, so a simulator that is present
but has not built its kernel is a failure here rather than a skip later.
"""

import os
import pathlib
import struct
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
import test_blob_on_sim as blob_sim  # noqa: E402
from executorch.backends.hexagon.batch_norm import RewriteBatchNormToAffine  # noqa: E402
from executorch.backends.hexagon.serialization import blob as _blob  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from torch.export import export  # noqa: E402

_BINARY = 19
#: The channel-axis stride of the second operand. The broadcast tail is
#: `[rank] + outDims[rank] + in0Strides[rank] + in1Strides[rank]` after the eight
#: int params, so the channel of the second operand is at 8 + 1 + rank + rank + 1
#: -- 26 at rank four, which is where the host-side control writes.
_CHANNEL_STRIDE_AT = 26

#: (tag, channels, height, width). The B cases walk the 64-element boundary with a
#: single channel so that the boundary and the channel axis cannot be confused;
#: the BN cases put a real channel count against a real plane.
_CONFIGS = (
    ("B63", 1, 63, 1),
    ("B64", 1, 64, 1),
    ("B65", 1, 65, 1),
    ("B66", 1, 66, 1),
    ("B128", 1, 128, 1),
    ("B129", 1, 129, 1),
    ("BN1", 3, 5, 7),
    ("BN16", 16, 4, 4),
    ("BN63", 63, 2, 1),
    ("BN64", 64, 2, 1),
    ("BN65", 65, 2, 1),
    ("BN1D", 8, 1, 1),
)


class _Norm(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.norm(x)


def _expected(model, x):
    """The affine twice over: once as the kernel computes it and once as it is.

    Three roundings happen on the way to a number and this names all of them.
    The pass stores `s` and `b` as fp16, which is the width the weight section
    holds and the only width the kernel can read. The kernel then rounds the
    product and rounds the sum, each once, which is not the same as rounding
    their sum once. The first return value is the kernel's own order; the second
    is the same two constants evaluated in fp64 and rounded once, which is the
    definition rather than a description of the implementation.

    A reference that went through `model(x.half())` would be a third thing
    again -- it rounds the normalized value, the scale and the bias separately
    on the way -- and would be measuring torch's expression rather than the
    affine this pass wrote.
    """
    norm = model.norm
    scale = torch.rsqrt(norm.running_var.double() + norm.eps)
    if norm.affine:
        scale = scale * norm.weight.double()
        shift = norm.bias.double() - norm.running_mean.double() * scale
    else:
        shift = -norm.running_mean.double() * scale
    stored_scale = scale.half().reshape((1, -1, 1, 1))
    stored_shift = shift.half().reshape((1, -1, 1, 1))
    with torch.no_grad():
        narrow = (x.half() * stored_scale + stored_shift).numpy()
        wide = (x.double() * stored_scale.double() + stored_shift.double()).numpy()
    return narrow.astype(np.float16), wide.astype(np.float16)


def _case_for(tag, channels, height, width):
    generator = torch.Generator().manual_seed(23)
    weight = (torch.rand((channels,), generator=generator) + 0.5).half()
    bias = (torch.rand((channels,), generator=generator) - 0.5).half()
    mean = torch.rand((channels,), generator=generator) * 6 - 3
    var = torch.rand((channels,), generator=generator) * 4 + 0.01
    model = _Norm(channels).eval()
    with torch.no_grad():
        model.norm.weight.copy_(weight)
        model.norm.bias.copy_(bias)
        model.norm.running_mean.copy_(mean)
        model.norm.running_var.copy_(var)

    x = torch.ones((1, channels, height, width), dtype=torch.float16)
    expected, wide = _expected(model, x)
    program = to_edge(export(model, (x,))).exported_program()
    assert RewriteBatchNormToAffine()(program).modified, f"{tag} was not rewritten"
    case = blob_sim._case(tag, None, (x,), expected, kind="bits", blob=_emit(program))
    case.scale = weight
    case.wide = wide
    return case


def _emit(program):
    from executorch.backends.hexagon.hexagon_backend import HexagonBackend

    return HexagonBackend.preprocess(program, []).processed_bytes


def _zero_the_channel_stride(blob):
    """The control: the constant's channel stride is zero, so all channels share.

    params[26] is where the second operand's channel stride sits, and zeroing it
    says every channel reads the first channel's coefficient. The shape of the
    answer is unchanged and the values are real, which is what makes this a
    control rather than a second happy path: a tolerance that accepts the right
    answer has to reject these by the width of the coefficients themselves.
    """
    data = bytearray(blob)
    struct.pack_into(
        "<i", data, _blob.HEADER_SIZE + 16 + 4 * _CHANNEL_STRIDE_AT, 0
    )
    return bytes(data)


def _cases():
    out = []
    for tag, channels, height, width in _CONFIGS:
        case = _case_for(tag, channels, height, width)
        case.shape = (1, channels, height, width)
        out.append(case)
    return out


@pytest.fixture(scope="module")
def cases():
    """Built lazily, so a blob that cannot be produced fails a test, not collection."""
    hexagon_sim._check()
    return _cases()


@pytest.fixture(scope="module")
def simulated(cases):
    try:
        return hexagon_sim.run(
            blob_sim._RUNNER,
            blob_sim._SOURCES,
            headers={
                "blob_fixture.h": blob_sim._fixture_header(cases),
                "htp_ops.h": blob_sim._HT_P_OP_SHIM,
            },
            includes_more=[str(blob_sim._SCHEMA)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))


def _dsp(case, simulated):
    bits = simulated[f"{case.tag}0"]
    return np.asarray(blob_sim._from_bits(bits), dtype=np.float16).reshape(
        case.expected.shape
    )


def test_every_case_is_a_multiply_and_an_add(cases):
    """A case that lowered to anything else would be testing something else."""
    for case in cases:
        kinds = [command.type for command in case.commands]
        assert kinds == [_BINARY, _BINARY], f"{case.tag} emitted {kinds}"


def test_the_constant_is_broadcast_on_the_spatial_axes_only(cases):
    """The strides are the claim, so they are read rather than inferred."""
    for case in cases:
        for command in case.commands:
            rank = command.params[8]
            assert rank == 4, f"{case.tag} carries a rank-{rank} tail"
            assert command.params[2] == case.shape[1], (
                f"{case.tag}'s constant holds {command.params[2]} values, not one "
                f"per channel"
            )
            # Zero on the batch and the two spatial axes, one on the channel:
            # the constant's channel extent equals the output's, so the walk
            # advances by one element per channel exactly as a dense read would.
            # A single channel is the exception and not a special case of it --
            # one element repeats on every axis, the channel included.
            channel_stride = 1 if case.shape[1] > 1 else 0
            assert list(command.params[25:29]) == [0, channel_stride, 0, 0], (
                f"{case.tag} carries {list(command.params[25:33])}"
            )


def test_the_simulator_answers_each_coefficient(cases, simulated):
    """The DSP's own bytes against the affine, bit for bit."""
    for case in cases:
        dsp = _dsp(case, simulated)
        assert dsp.size > 0, f"{case.tag} came back empty"
        if not np.array_equal(dsp, case.expected):
            wrong = np.flatnonzero(
                dsp.reshape(-1) != case.expected.reshape(-1)
            )[:4]
            raise AssertionError(
                f"{case.tag} is wrong at {wrong.tolist()}: the DSP gave "
                f"{[float(dsp.reshape(-1)[i]) for i in wrong]} and the affine is "
                f"{[float(case.expected.reshape(-1)[i]) for i in wrong]}"
            )


def test_the_walk_ends_where_the_extent_ends(cases, simulated):
    """The last element of every axis, which is the one a rounded walk drops.

    Stated separately from the equality above because it is the assertion that
    names the boundary: a kernel that took the vector part to `size - 64` would
    leave the final lane of 65, 66 and 129 holding the lane before it.
    """
    for case in cases:
        channels, height, width = case.shape[1:]
        flat = _dsp(case, simulated).reshape(-1)
        last = [height * width - 1, width - 1, channels * height * width - 1]
        for offset in last:
            assert float(flat[offset]) != 0.0, (
                f"{case.tag} left the element at {offset} at zero, which is what "
                "a walk that stopped a vector short leaves behind"
            )

def test_the_narrow_reference_is_the_wide_affine_within_one_step(cases, simulated):
    """The reference the comparison uses is itself pinned to the definition.

    The DSP's answer is compared against the kernel's own order of operations,
    which on its own would accept a kernel that computed something else in the
    same order. This is the other half: the same answer against the fp64 affine,
    allowed the one step that the two roundings can spend between them and
    nothing else. A coefficient read from the wrong channel is a different
    number rather than a different last bit, so this is the bound a wrong
    broadcast placement cannot get near.
    """
    for case in cases:
        dsp = _dsp(case, simulated).astype(np.float64)
        wide = case.wide.astype(np.float64)
        step = np.abs(
            np.nextafter(case.expected, np.float16(np.inf)).astype(np.float64)
            - case.expected.astype(np.float64)
        )
        drift = np.abs(dsp - wide) / np.where(step > 0, step, np.finfo(np.float16).tiny)
        assert drift.max() <= 1.0, (
            f"{case.tag} is {drift.max():.1f} steps from the fp64 affine, and the "
            f"two roundings can only spend one"
        )


def test_a_wrong_channel_stride_is_not_within_any_of_them(cases):
    """The control, run on the host because it needs no simulator.

    Zeroing the constant's channel stride leaves the shape, the arity and the
    magnitudes all right and makes every channel read the first one's
    coefficient. The two bounds above accept a difference of one fp16 step; this
    answer differs from the affine by the width of the coefficient spread, which
    is three orders of magnitude over the tightest bound in the file.
    """
    for case in cases:
        blob = _zero_the_channel_stride(case.blob)
        _, commands = blob_sim.read_blob(blob)
        assert len(commands) == 2, f"{case.tag} lost a command to the patch"
        answered = blob_sim.execute(blob, [case.args[0].numpy()])[0]
        wrong = np.asarray(answered, dtype=np.float16).reshape(-1)
        # The control has one job: its answer is not the affine. A stride that
        # is zero where the channel walk wants one changes the walk, and the
        # two ways it can show are a different number of elements and a
        # different value inside the right number. Either is a failure; neither
        # may be a shape the bounds above would accept.
        if wrong.size != case.expected.size:
            continue
        step = np.abs(
            np.nextafter(case.expected, np.float16(np.inf)).astype(np.float64)
            - case.expected.astype(np.float64)
        )
        inside = (
            np.abs(wrong.astype(np.float64) - case.wide.astype(np.float64))
            <= np.where(step > 0, step, np.finfo(np.float16).tiny)
        )
        assert not inside.all(), (
            f"{case.tag}'s control is inside the fp64 bound at "
            f"{int(inside.sum())} of {inside.size} elements, so the bound in the "
            "previous test would accept a wrong broadcast placement"
        )