# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Does the zero-insert region zero-insert, on the DSP?

A transposed convolution is lowered to a plain convolution over a
zero-interleaved input, and the interleaving is emitted as `DSP_OP_ZERO` plus
one `DSP_OP_RASTER_BLIT` region. This test runs `htp_ops_zero` and
`htp_ops_raster_blit` -- the vendored C, compiled for v79 and executed under
hexagon-sim -- on exactly that region and asks whether the destination came out
as the zero-insert the emitter intends, or as whatever one of the kernel's fast
paths writes instead.

Each case is checked three ways. The runner counts, element by element, how
many destination elements differ from the intent it computes itself, and that
count has to be zero. It also hashes the raw destination bytes, and the host
independently rebuilds the same zero-insert in numpy and hashes it too; the two
digests have to agree, so the check is not merely "the C agrees with a C-side
copy of the intent". A third line carries the source's own hash, so the host can
see that both sides built the same input before comparing outputs.

The simulator's stdout is parsed here rather than through `hexagon_sim.run`'s
return value: that return value is the harness's own `TAG <hex words...>`
protocol, and the `HASH=`/`MISMATCH=` line this runner prints per case is
deliberately not in it. `hexagon_sim.LAST_STDOUT` is the same run's whole
output.

It needs HEXAGON_SDK_ROOT and a libncurses5 for the simulator, and skips with a
reason when the toolchain is not there. A runner that does not compile fails
rather than skipping, as does a simulator that refuses to run a binary it was
handed -- the `_check()` below is the same pre-flight `run()` does, so anything
after it is a defect and not a missing toolchain.
"""

import os
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hexagon_sim  # noqa: E402

_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/zero_insert_runner.cpp"

#: The vendored sources the blit needs; nothing here is reimplemented. The
#: worker pool comes up because several of the kernel's fast paths are gated on
#: there being more than one worker.
_SOURCES = [
    "blit_ops.cc",
    "worker_pool.cc",
    "power.cc",
]

#: htp_ops.h is generated from the IDL, so the runner gets a shim with the one
#: AEEStdDef declaration the vendored headers expect.
_HEADERS = {"htp_ops.h": '#pragma once\n#include "AEEStdDef.h"\n'}

#: The runner's own table: tag, planes, source H, source W, interleave factor.
#: ZI_W16_S4 is the one geometry `htp_ops_try_interleave_c64_single_blit` claims,
#: and ZI_W16_S4_CUT scatters it with the width cut into two regions so that path
#: cannot match -- which the emitter does not do, and which is kept because it
#: shows the destination does not depend on how the scatter is split. The rest
#: are shapes that keep every fast path out of reach.
CASES = [
    ("ZI_SMALL", 1, 3, 5, 2),
    ("ZI_OX", 1, 4, 7, 3),
    ("ZI_PLANES", 3, 3, 5, 2),
    ("ZI_W16_S4", 1, 2, 16, 4),
    ("ZI_W16_S4_CUT", 1, 2, 16, 4),
    ("ZI_W64", 2, 3, 64, 2),
    ("ZI_S1H", 2, 9, 4, 2),
    ("ZI_H64", 1, 64, 4, 2),
    # Replications: not a zero-insert at all, but the same region walk with a
    # destination offset per phase, which is the one thing the cases above never
    # put in front of the kernel's fast paths.
    ("ZI_REP_S2", 1, 3, 5, 2),
    ("ZI_REP_S2_P3", 3, 3, 7, 2),
    ("ZI_REP_S3", 2, 2, 4, 3),
]

#: The cases the runner builds as a replication rather than as a zero-insert.
REPLICATE = {"ZI_REP_S2", "ZI_REP_S2_P3", "ZI_REP_S3"}

_IDS = [case[0] for case in CASES]

#: The zero-insert cases, for the expectations that are a zero-insert. The
#: replication cases share the runner and the parametrization but not that
#: expectation, and they carry their own test below.
_ZERO_CASES = [case for case in CASES if not case[0].startswith("ZI_REP")]
_ZERO_IDS = [case[0] for case in _ZERO_CASES]

_RESULT = re.compile(r"^(ZI_[A-Z0-9_]+) HASH=([0-9a-f]{8}) MISMATCH=(-?\d+)$")
_SOURCE_DIGEST = re.compile(r"^(ZI_[A-Z0-9_]+)_SRC HASH=([0-9a-f]{8})$")
_RETURN = re.compile(r"^(ZI_[A-Z0-9_]+)_RET (-?\d+) (-?\d+)$")
_DESTINATION = re.compile(r"^(ZI_[A-Z0-9_]+)_DEST((?: [0-9a-f]{4})+)$")

_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619


@pytest.fixture(scope="module")
def simulated():
    """The runner's own output, as tag -> whatever that tag printed.

    Skips when the toolchain is missing and fails when it is not: a machine
    without a simulator has nothing to say here, while a shared object that
    does not build or that the simulator declines to run is the defect this
    test exists to catch.
    """
    try:
        hexagon_sim._check()  # noqa: SLF001 - the same pre-flight run() does
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))
    try:
        hexagon_sim.run(_RUNNER, _SOURCES, headers=_HEADERS)
    except hexagon_sim.BuildFailed as error:
        pytest.fail(str(error), pytrace=False)
    except hexagon_sim.Unavailable as error:
        pytest.fail(f"the runner built but did not run: {error}", pytrace=False)

    printed = {"result": {}, "source": {}, "return": {}, "destination": {}}
    for line in hexagon_sim.LAST_STDOUT.splitlines():
        match = _RESULT.match(line)
        if match:
            printed["result"][match.group(1)] = (
                int(match.group(2), 16),
                int(match.group(3)),
            )
            continue
        match = _SOURCE_DIGEST.match(line)
        if match:
            printed["source"][match.group(1)] = int(match.group(2), 16)
            continue
        match = _RETURN.match(line)
        if match:
            printed["return"][match.group(1)] = (
                int(match.group(2)),
                int(match.group(3)),
            )
            continue
        match = _DESTINATION.match(line)
        if match:
            printed["destination"][match.group(1)] = [
                int(word, 16) for word in match.group(2).split()
            ]
    assert printed["result"], "the simulator ran but printed no case lines"
    return printed


def _source(planes, h, w):
    """1 .. P*H*W: every element distinct, all of them exactly fp16."""
    count = planes * h * w
    values = np.arange(1, count + 1, dtype=np.int64)
    # The runner writes the same integers, and the digest comparison below only
    # means anything while both sides can represent them exactly.
    assert np.array_equal(values.astype(np.float16).astype(np.int64), values)
    return values.astype(np.float16).reshape(planes, h, w)


def _intent(planes, h, w, s):
    """The zero-insert the emitter's region describes, in numpy.

    The region's strides put element (p, y, x) of the source at flat offset
    `p*uh*uw + y*s*uw + x*s` of the destination, which is a
    `(planes, uh, uw)` array with the interleave on both axes and zeros
    everywhere the scatter does not land.
    """
    uh, uw = (h - 1) * s + 1, (w - 1) * s + 1
    source = _source(planes, h, w)
    destination = np.zeros((planes, uh, uw), dtype=np.float16)
    for p in range(planes):
        for y in range(h):
            for x in range(w):
                destination[p, y * s, x * s] = source[p, y, x]
    return destination


def _digest(values):
    """The runner's own FNV-1a, over the same bytes it hashed.

    The simulator is little-endian and so is the array's fp16 encoding here,
    which is what makes the two digests comparable at all.
    """
    hash_value = _FNV_OFFSET
    for byte in np.asarray(values, dtype="<f2").tobytes():
        hash_value = ((hash_value ^ int(byte)) * _FNV_PRIME) & 0xFFFFFFFF
    return hash_value


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_the_runner_receives_the_source_both_sides_built(simulated, case):
    """The digests below compare outputs, so the inputs have to agree first."""
    tag, planes, h, w, _ = case
    assert simulated["source"][tag] == _digest(_source(planes, h, w))


@pytest.mark.parametrize("case", _ZERO_CASES, ids=_ZERO_IDS)
def test_the_region_zero_inserts_on_the_dsp(simulated, case):
    """`MISMATCH == 0` and the bytes are the host's own zero-insert.

    The second half is the independent one: a runner that agreed with a
    C-side copy of the intent while both were wrong would still be a mismatch
    here, because numpy built the destination from the region's strides rather
    than from anything the runner computed.
    """
    tag, planes, h, w, s = case
    digest, mismatches = simulated["result"][tag]
    assert mismatches == 0, f"{tag}: the destination is not the zero-insert"
    assert digest == _digest(
        _intent(planes, h, w, s)
    ), f"{tag}: the destination is not the host's zero-insert"


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_both_commands_reported_success(simulated, case):
    """`htp_ops_zero` and `htp_ops_raster_blit` returned zero.

    A refusal is worth separating from a wrong answer: a nonzero status with an
    unchanged destination and a wrong mapping that returned success are
    different findings.
    """
    tag = case[0]
    zero_ret, blit_ret = simulated["return"][tag]
    assert (zero_ret, blit_ret) == (0, 0)


def test_the_zero_command_cleared_the_poisoned_buffer(simulated):
    """`DSP_OP_ZERO` is load-bearing, and the destination proves it ran.

    The runner fills the destination with 0xAB before the clear, so a gap that
    is zero is a gap something cleared rather than one nothing ever wrote. In
    ZI_SMALL (H=3, W=5, S=2) the destination plane is 5 rows of 9: source rows
    land on destination rows 0, 2 and 4, and every odd column is a gap.
    """
    words = simulated["destination"]["ZI_SMALL"]
    assert words[0] == 0x3C00  # 1.0, the first source element
    assert words[1] == 0x0000  # the gap between it and the next one, poisoned
    assert words[2] == 0x4000  # 2.0
    assert words[9] == 0x0000  # destination row 1, which no source row lands on
    assert words[18] == 0x4600  # 6.0, the first element of source row 1


def test_the_w16_s4_case_the_interleave_path_claims_is_the_hosts_zero_insert(
    simulated,
):
    """The geometry `htp_ops_try_interleave_c64_single_blit` matches.

    That path is guarded by ``size[2] == 16 && srcStride[2] == 1 &&
    dstStride[2] == 4``, which a one-region zero-insert at W=16, S=4 satisfies
    exactly, so this case is the one that would show a 64-lane interleave
    written where a plain scatter was asked for.
    """
    tag = "ZI_W16_S4"
    digest, mismatches = simulated["result"][tag]
    assert mismatches == 0
    assert digest == _digest(_intent(1, 2, 16, 4))
    # The first destination row, read out: the sixteen source values at stride
    # four with a zero between each pair, which is both what the region asks for
    # and what that path writes.
    words = simulated["destination"][tag]
    assert words[:64:4] == np.arange(1, 17, dtype=np.float16).view(np.uint16).tolist()
    assert words[1:64:4] == [0] * 16


def test_splitting_the_scatter_into_two_regions_says_the_same_thing(simulated):
    """The destination does not depend on how the scatter is split.

    `ZI_W16_S4` is one region, which the interleave fast path claims; this case
    is the same scatter as two, which it cannot. Both are checked against the
    host's zero-insert above, so this only says the two encodings agree -- and
    that is the point, because the fast path is reached on the first and not on
    the second, so agreement is evidence the two paths write the same bytes
    rather than a comparison of one path with itself.
    """
    assert simulated["result"]["ZI_W16_S4"] == simulated["result"]["ZI_W16_S4_CUT"]
    assert (
        simulated["destination"]["ZI_W16_S4"]
        == simulated["destination"]["ZI_W16_S4_CUT"]
    )


def test_the_comparisons_can_fail():
    """Reverse controls: the digests above are equalities a change would move.

    Without these, a comparison that always passed -- or an intent that
    happened to be the source -- would read as agreement.
    """
    for _, planes, h, w, s in CASES:
        source = _source(planes, h, w)
        intent = _intent(planes, h, w, s)
        assert _digest(source) != _digest(intent), "the zero-insert is the identity"
        assert _digest(np.zeros_like(intent)) != _digest(
            intent
        ), "the destination is all zeros, so nothing was scattered"
    intent = _intent(1, 3, 5, 2)
    # A destination that ignores the interleave -- the input copied straight into
    # the plane -- is a different buffer, so a blit that dropped the strides
    # would be caught.
    unexpanded = np.zeros_like(intent)
    unexpanded[0, :3, :5] = _source(1, 3, 5)[0]
    assert _digest(unexpanded) != _digest(intent)
    # And the digest is sensitive to a single moved element, which is what makes
    # its agreement with the host's worth anything.
    moved = intent.copy()
    moved[0, 0, 0], moved[0, 2, 4] = moved[0, 2, 4], moved[0, 0, 0]
    assert moved.tobytes() != intent.tobytes()
    assert _digest(moved) != _digest(intent)


def _replicated(planes, h, w, s):
    """What a nearest upsample of the source is: every element repeated s times."""
    return np.repeat(np.repeat(_source(planes, h, w), s, axis=1), s, axis=2)


@pytest.mark.parametrize(
    "case", [c for c in CASES if c[0] in REPLICATE], ids=lambda c: c[0]
)
def test_a_replication_is_the_region_walk_at_an_offset(case, simulated):
    """The phases the upsample emitter builds, on the DSP.

    Every phase reads the source whole and starts at its own destination offset,
    which is a shape none of the zero-insert cases reach. The destination is
    compared against the replication itself, and the two commands are the two
    the emitter's parameter block splits four regions into.
    """
    tag, planes, h, w, s = case
    digest, mismatches = simulated["result"][tag]
    assert mismatches == 0, f"{tag}: the destination is not the replication"
    assert digest == _digest(_replicated(planes, h, w, s)), f"{tag}: wrong bytes"
    _, blit_ret = simulated["return"][tag]
    assert blit_ret == 0, f"{tag}: the blit returned {blit_ret}"
