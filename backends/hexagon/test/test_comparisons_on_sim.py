# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The two-command comparison route on the DSP, with the emitter's own parameters.

This is the SIMULATOR tier: hexagon-sim running the vendored eltwise_ops.cc,
compiled for v79 with the SDK's own bin/hexagon-clang. It is real DSP code under
emulation, which is the right tier for the question -- does the command stream the
host writes compute torch's answer -- and it is not a phone. Nothing here is
device evidence and nothing here is a claim about a phone's DSP.

The distinction this file exists for: hexagon-cmpgate measured that
htp_ops_binary_elementwise at op type 9 followed by htp_ops_select at one byte
answers torch on sixteen corner pairs. That measurement drove a hand-written
call. What was not established is that the parameters _emit_compare writes are the
parameters that measurement used, and a transposed pair, a wrong op type or a
value width declared at two bytes would agree with itself perfectly. So the
parameters here are read out of the blob a real export produced and handed to the
kernels through a generated header. If the emitter and the measurement disagree,
the disagreement is the failure this test reports.
"""

import pathlib

import pytest
import torch


import hexagon_sim
import test_blob_on_sim as blob_sim
from blob_interpreter import read_blob

from test_comparisons import _blob_of, _corners

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/gtlt_runner.cpp"
_BINARY = 19
_SELECT = 26


def _ints(values):
    return "{" + ",".join(str(int(v)) for v in values) + "}"


def _emitted(op, a, b):
    """The blob's own parameters and weight bytes for one comparison, as C.

    The compare command's params are copied whole -- head and broadcast tail --
    and the select's eight are copied whole. The two one-byte sources are not
    re-declared: their offsets come out of the blob and their bytes are sliced
    out of its weight section, so a selector that had materialized them as fp16
    is caught here instead of being papered over by a literal in this file.
    """
    from executorch.backends.hexagon.serialization import blob as schema

    blob, commands = _blob_of(op, a, b)
    assert [c.type for c in commands] == [_BINARY, _SELECT], op
    compare, select = commands
    header, _ = read_blob(blob)
    start = schema.HEADER_SIZE + header.n_ops * schema.OP_SIZE
    section = blob[start: start + header.weights_bytes]
    return compare, select, section


def _header(a, b):
    gt_compare, select, section = _emitted(torch.gt, a, b)
    lt_compare, lt_select, lt_section = _emitted(torch.lt, a, b)
    # The two exports are independent, so the LESS vector is the lt export's own
    # and not the gt one written twice. The section is the gt blob's, and the
    # two constants are the same two bytes either way -- the offsets and the
    # bytes are both asserted equal here so that is a measurement and not an
    # assumption.
    assert list(lt_select.params) == list(select.params), "the select is one shape"
    assert (lt_select.inputs[1].offset, lt_select.inputs[2].offset) == (
        select.inputs[1].offset,
        select.inputs[2].offset,
    )
    assert lt_section == section, "the two constants are the same two bytes"
    return (
        "#pragma once\n"
        f"static const int32_t kCompareParams[] = {_ints(gt_compare.params)};\n"
        f"static const int32_t kLessParams[] = {_ints(lt_compare.params)};\n"
        f"static const int32_t kSelectParams[] = {_ints(select.params)};\n"
        f"static const int32_t kOnOffset = {select.inputs[1].offset};\n"
        f"static const int32_t kOffOffset = {select.inputs[2].offset};\n"
        f"static const uint8_t kSelectSources[] = {_ints(section)};\n"
    )


@pytest.fixture(scope="module")
def ran():
    try:
        hexagon_sim._check()
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))
    a, b = _corners()
    sources = [source for source in blob_sim._SOURCES if isinstance(source, str)]
    try:
        results = hexagon_sim.run(
            _RUNNER,
            sources,
            headers={
                "htp_ops.h": blob_sim._HT_P_OP_SHIM,
                "gtlt_params.h": _header(a, b),
            },
        )
    except hexagon_sim.Unavailable as error:
        # _check answered above, so the SDK and the simulator are both here and
        # this is the simulator failing to run the runner. That is a failure, not
        # a skip, and the distinction is the only thing between this file and a
        # green no-op.
        raise AssertionError(f"hexagon-sim did not run the runner: {error}")
    for tag in ("A", "B", "FLAGS", "GT", "LT", "BRCMP", "SELRC", "LSCMP", "LSSEL", "SRC"):
        assert tag in results, f"the runner printed no {tag}"
    return results


def _as_half(words):
    return torch.tensor(words, dtype=torch.int64).bitwise_and(0xFFFF).to(torch.uint16).view(
        torch.float16
    )


def test_the_dsp_was_handed_the_sixteen_pairs_the_reference_has(ran):
    """The operands are checked before any answer is, on both sides.

    A route that disagreed with torch on a pair the kernel never received would
    look exactly like a route that disagrees on one it did, so the printed inputs
    are compared against the reference's own bytes first.
    """
    a, b = _corners()
    assert ran["A"] == [int(v) for v in a.view(torch.uint16).flatten().tolist()]
    assert ran["B"] == [int(v) for v in b.view(torch.uint16).flatten().tolist()]


def test_both_commands_reported_success(ran):
    """The return codes, which are the only thing a wrong parameter vector does
    not always change: a bytes the select refuses is -1 and a tail it cannot
    walk is -2, and a run that returned either would still print an OUT.
    """
    assert ran["BRCMP"] == [0x00000000], ran["BRCMP"]
    assert ran["SELRC"] == [0x00000000], ran["SELRC"]
    assert ran["LSCMP"] == [0x00000000], ran["LSCMP"]
    assert ran["LSSEL"] == [0x00000000], ran["LSSEL"]


def test_the_two_sources_the_dsp_copied_between_are_one_and_zero(ran):
    """Read out of the blob's weight section, so the byte the kernel saw is the
    byte the emitter wrote rather than a literal this file also holds.
    """
    assert ran["SRC"] == [0x01, 0x00], ran["SRC"]


def test_the_flag_the_binary_command_leaves_is_fp16_one_and_zero(ran):
    """The intermediate, which is the whole premise of the second command.

    At bytes 2 the compare arm writes 1.0 and 0.0 whatever outputIsFloat says,
    so the flag is a half float. A command that wrote int32 1 and 0 instead would
    make the select read four bytes an element and answer nonsense, and the flag
    is the only place that shows it.
    """
    flags = _as_half(ran["FLAGS"])
    assert set(int(v) for v in flags.view(torch.uint16)) <= {0x3C00, 0x0000}, (
        [f"{v}" for v in flags.view(torch.uint16)]
    )
    a, b = _corners()
    assert ran["FLAGS"] == [
        int(v) for v in torch.where(a > b, torch.ones((), dtype=torch.float16),
                                    torch.zeros((), dtype=torch.float16))
        .flatten()
        .view(torch.uint16)
        .tolist()
    ]


def test_the_dsp_greater_and_less_answer_torch_on_every_corner_pair(ran):
    """Zero disagreements over the sixteen, for both op types, on the DSP.

    The corners are the reason the route exists: a NaN compares false and the two
    signed zeros compare equal, which is what a subtraction-based route cannot
    do at any length. Index 1 is -0.0 against 0.0, index 4 is a NaN against
    itself, and 5 and 6 are each infinity against itself.
    """
    a, b = _corners()
    a, b = a.flatten(), b.flatten()
    assert ran["GT"] == [1 if bool(x) else 0 for x in a > b], (
        [i for i, (g, w) in enumerate(zip(ran["GT"], (a > b).tolist()))
         if g != (1 if w else 0)]
    )
    assert ran["LT"] == [1 if bool(x) else 0 for x in a < b], (
        [i for i, (g, w) in enumerate(zip(ran["LT"], (a < b).tolist()))
         if g != (1 if w else 0)]
    )
    # The two answers are not the same vector, so a runner that ran one command
    # twice could not pass both.
    assert ran["GT"] != ran["LT"]
    assert len(ran["GT"]) == len(ran["A"]) == 16


def test_the_result_is_one_byte_per_element_and_all_of_it_is_zero_or_one(ran):
    """The output width the blob declared, checked on the bytes the DSP wrote.

    Sixteen booleans is sixteen bytes. A command that had fallen through to the
    two-byte arm would have written thirty-two and left the first sixteen looking
    like fp16 garbage rather than flags.
    """
    for tag in ("GT", "LT"):
        assert len(ran[tag]) == 16, tag
        assert set(ran[tag]) <= {0, 1}, (tag, ran[tag])
