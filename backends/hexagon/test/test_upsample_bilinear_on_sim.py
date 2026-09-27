# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Does the bilinear upsample command execute on the DSP?

This is the hexagon-sim tier and nothing more. The runner compiles
`upsample_ops.cc` -- the translation unit the skel builds for this op -- for
v79 and runs it inside the simulator's QuRT, so HVX and all, with no device and
no FastRPC. What a green run establishes is that the command stream the emitter
produced reaches the kernel and the kernel computes it. It says nothing about a
phone, and nothing here was run on one.

The runner is the shared blob runner rather than a purpose-built one, so the
blob under test is the same bytes the backend hands the skel: the argument is
handled by parsing the real `HexagonBlobHeader` and `HexagonOp` structs out of
it, the way the device does, rather than by being told the geometry. The
output is the shared runner's 16-bit word format, which the host decodes as
fp16 without a float conversion in between.

The fixtures are chosen so the thing this kernel could get wrong is the thing
the run decides:

  * 1x1, where every weight is zero and both taps are the single source
    element, so a kernel that accumulated instead of selecting cannot pass;
  * 3x3, where the reciprocal of the scale is not representable and the tap
    weights are not periodic in the output index, which is the case the
    per-axis table exists for;
  * 1x4, an axis with no vertical change at all, so a kernel that mixed the
    two axes up answers something plausible;
  * a single-column input, where the upper tap clamps onto the lower one and
    the kernel reads the same element twice;
  * the last output column, where the upper tap clamps for the same reason;
  * three planes, so the plane loop is walked;
  * values at the top of the fp16 range, so an accumulation that overflowed
    would show as an infinity;
  * and a mutated command, without which every comparison above would also be
    satisfied by a kernel that ignores its parameters.
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
from blob_interpreter import UnsupportedOp  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
    _case,
    _fixture_header,
)

_BILINEAR = 46

#: The bound the host test states and the reason it is a bound: a factor whose
#: reciprocal is not representable leaves the last fp32 rounding before the fp16
#: store to chance. Two fp16 steps at one, where the spacing is 2^-10.
_ATOL = 2.0**-9


class _Bilinear(torch.nn.Module):
    def __init__(self, scale_factor, align_corners=False) -> None:
        super().__init__()
        self.kwargs = {
            "mode": "bilinear",
            "align_corners": align_corners,
            "scale_factor": [float(scale_factor[0]), float(scale_factor[1])],
        }

    def forward(self, x):
        import torch.nn.functional as F

        return F.interpolate(x, **self.kwargs)


def _bits(tensor):
    return tensor.detach().half().numpy().view(np.uint16)


def _float_bits(tensor):
    return tensor.detach().numpy().reshape(-1).view(np.uint16)


def _set_param(data, param_index, value, op_index=0):
    """One command param rewritten in place, from the struct own layout."""
    from blob_interpreter import read_blob

    _, commands = read_blob(data)
    command = commands[op_index]
    prefix = struct.pack(
        "<IIII",
        command.type,
        len(command.inputs),
        len(command.outputs),
        len(command.params),
    )
    at = data.find(prefix)
    assert at >= 0 and data.find(prefix, at + 1) < 0, "the command block is not unique"
    slot = at + len(prefix) + 4 * param_index
    patched = data[:slot] + int(value).to_bytes(4, "little", signed=True) + data[slot + 4 :]
    assert read_blob(patched)[1][op_index].params[param_index] == value
    return patched


@pytest.fixture(scope="module")
def cases():
    torch.manual_seed(0)
    plane = torch.randn(1, 1, 3, 4, dtype=torch.float16)
    single_column = torch.randn(1, 1, 3, 1, dtype=torch.float16)
    three_planes = torch.randn(1, 3, 2, 3, dtype=torch.float16)
    top_of_range = torch.full((1, 1, 2, 2), 60000.0, dtype=torch.float16)

    def reference(module, tensor):
        return _float_bits(module(tensor))

    c1 = _Bilinear((1, 1))
    c2 = _Bilinear((2, 2))
    c3 = _Bilinear((3, 3))
    c14 = _Bilinear((1, 4))
    c41 = _Bilinear((4, 1))
    c4 = _Bilinear((4, 4))

    # A ramp rather than a constant, so no case can be answered by a kernel that
    # wrote a single value. The 1x1 case is the one where every weight is exactly
    # zero and both taps are the same element, and the claim is that the answer
    # is then the input byte for byte -- which the scale factor alone would not
    # guarantee, since the source coordinate is still floor(1 * (dst + 0.5) - 0.5)
    # and has to come out as dst.
    ramp = torch.arange(12, dtype=torch.float16).reshape(1, 1, 3, 4) - 5.0

    return [
        _case("UB_S1", c1, (ramp,), reference(c1, ramp)),
        _case("UB_RAMP", c2, (ramp,), reference(c2, ramp)),
        _case("UB_S2", c2, (plane,), reference(c2, plane)),
        _case("UB_S3", c3, (plane,), reference(c3, plane)),
        _case("UB_S1X4", c14, (plane,), reference(c14, plane)),
        _case("UB_S4X1", c41, (plane,), reference(c41, plane)),
        _case("UB_S4", c4, (plane,), reference(c4, plane)),
        _case("UB_COLUMN", c4, (single_column,), reference(c4, single_column)),
        _case("UB_PLANES", c2, (three_planes,), reference(c2, three_planes)),
        _case("UB_RANGE", c2, (top_of_range,), reference(c2, top_of_range)),
    ]


@pytest.fixture(scope="module")
def simulated(cases):
    try:
        hexagon_sim._check()
        return hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={
                "blob_fixture.h": _fixture_header(cases),
                "htp_ops.h": _HT_P_OP_SHIM,
            },
            includes_more=[str(_SCHEMA)],
        )
    except hexagon_sim.Unavailable as error:
        # _check() above already proved the toolchain is here, so a later
        # Unavailable is the simulator refusing to run the binary, which is a
        # failure and not a missing toolchain.
        raise


def _dsp_words(simulated, case):
    """The runner's 16-bit words, decoded as fp16 without a float in between."""
    result = simulated[f"{case.tag}0"]
    assert result, f"{case.tag} returned no words"
    return np.frombuffer(np.asarray(result, dtype=np.uint16).tobytes(), dtype=np.float16)

def test_each_blob_carries_the_bilinear_command_and_nothing_else(cases):
    for case in cases:
        assert [command.type for command in case.commands] == [_BILINEAR], case.tag
        assert len(case.commands[0].params) == 5, case.tag
        assert len(case.blob) > 0


def test_the_dsp_and_the_host_model_and_torch_agree_on_every_fixture(cases, simulated):
    for case in cases:
        result = simulated[f"{case.tag}0"]
        assert result, f"{case.tag} returned no words"
        got = np.frombuffer(np.asarray(result, dtype=np.uint16).tobytes(), dtype=np.float16)
        want = np.frombuffer(case.expected.tobytes(), dtype=np.float16)
        assert got.size == want.size, case.tag
        deviation = np.abs(got.astype(np.float64) - want.astype(np.float64))
        assert deviation.max() <= _ATOL, (case.tag, deviation.max())
        host = np.frombuffer(case.host[0].tobytes(), dtype=np.float16)
        assert host.size == want.size, case.tag
        assert np.abs(host.astype(np.float64) - got.astype(np.float64)).max() <= _ATOL, (
            case.tag,
            "host and DSP",
        )


def test_no_kernel_refused_a_geometry_the_gate_admitted(cases, simulated):
    """A refusal is a line on the runner stdout, and an empty answer is not one.

    The runner prints `TAG upsample_bilinear2d returned N` when a kernel returns
    non-zero, so a refusal the decoded words would otherwise hide shows up here.
    The word count is pinned to the expected size for the same reason: a kernel
    that wrote nothing and a kernel that wrote the wrong thing both leave a
    plausible array behind unless the size is checked.
    """
    for case in cases:
        for line in hexagon_sim.LAST_STDOUT.splitlines():
            if line.startswith(case.tag) and "returned" in line:
                raise AssertionError(line.strip())
        words = simulated[f"{case.tag}0"]
        assert len(words) == int(np.prod(case.expected.shape)), case.tag


def test_the_1x1_case_returns_its_input_byte_for_byte(cases, simulated):
    """The zero-weight path: every output weight is exactly 0 and both taps are
    the same source element, so the answer is the input and a kernel that had
    mixed the two up could not produce it."""
    case = next(c for c in cases if c.tag == "UB_S1")
    assert list(case.commands[0].params) == [1, 3, 4, 3, 4]
    got = _dsp_words(simulated, case)
    source = case.args[0].numpy().reshape(-1)
    assert got.size == source.size
    np.testing.assert_array_equal(got.view(np.uint16), source.view(np.uint16))


def test_every_case_that_can_move_did_move(cases, simulated):
    """A kernel that copied its input passes a comparison at scale one, so the
    cases where the output differs in size have to be shown to differ in bytes."""
    moved = 0
    for case in cases:
        got = _dsp_words(simulated, case)
        source = case.args[0].numpy().reshape(-1)
        same = got.size == source.size and np.array_equal(got.view(np.uint16), source.view(np.uint16))
        moved += 0 if same else 1
    assert moved == 9, f"only {moved} of {len(cases)} fixtures changed the input bytes"
def test_the_dsp_result_would_move_if_the_command_did(cases):
    """A comparison that also passes on the wrong parameters is not a check."""
    case = next(c for c in cases if c.tag == "UB_S2")
    assert list(case.commands[0].params) == [1, 3, 4, 6, 8]
    good = case.host[0]
    corrupted = _set_param(case.blob, 1, 2)
    from blob_interpreter import execute

    moved = execute(corrupted, [t.numpy() for t in case.args])[0]
    assert moved.tobytes() != good.tobytes()


def test_a_non_integer_ratio_blob_is_not_answered_rather_than_guessed(cases):
    """The kernel refuses the geometry the gate refuses; the host model says so."""
    case = next(c for c in cases if c.tag == "UB_S2")
    odd = _set_param(_set_param(case.blob, 3, 7), 4, 9)
    from blob_interpreter import execute

    with pytest.raises(UnsupportedOp):
        execute(odd, [t.numpy() for t in case.args])
