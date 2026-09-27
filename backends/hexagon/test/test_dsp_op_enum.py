# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`hexagon_ops.py`'s op-type mirror cannot drift from the DSP's enum.

The command type is a number, and the number is defined once, in C++:
`third-party/mnn-htp-ops/include/htp_command.h`. Everything on the host -- the
emitters, `blob_interpreter`, the device tooling -- reads a *Python copy* of that
table, and the copies were written by hand, one entry per op somebody needed at
the time.

The failure mode is silent and it is the reason this file exists. A device model
whose 19 commands used 7 distinct types was reported with 3 of them recognised
and the other 4 as *unmapped*, because the Python dictionary was a strict subset
of the header. An unmapped reading reads as "this kernel does not exist", while
the truth is "it ran on the DSP ten seconds ago", and 4 of the 5 that went
missing were among the four real compute kernels. Nothing raised; the number was
just absent from a dict.

So this is a test and not a note, because a note has to be read and a test cannot
be skipped by accident:

* every member of `enum DSPOpType` has a `DSP_OP_*` name in `hexagon_ops` with
  the **same value**, including the reserved slots and the two sentinels;
* `hexagon_ops` has no `DSP_OP_*` name the header does not have;
* the head of the enum auto-increments and the tail is explicitly numbered, so
  the test resolves the auto-increment itself rather than assuming a convention;
* and the command types `blob_interpreter` executes are pinned to `hexagon_ops`,
  because that is the second hand-written copy and the one that decides whether a
  blob is answered.

Reading the header here is the point. The C++ enum is the single source of truth;
every other copy is a convenience, and a convenience that is checked is a
convenience.
"""

import pathlib
import re

import pytest

from executorch.backends.hexagon import hexagon_ops

_HEADER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "third-party"
    / "mnn-htp-ops"
    / "include"
    / "htp_command.h"
)


def _dsp_op_types_from_the_header():
    """Every member of `enum DSPOpType` as {name: value}.

    The head of the enum runs on without a value and the tail is numbered, so
    the next value is carried along and only overridden where one is written. A
    member written with no value after one written with a value would otherwise
    silently take the wrong number and this is the one place that has to be right.
    """
    source = _HEADER.read_text()
    start = source.index("enum DSPOpType")
    end = source.index("};", start)
    body = re.sub(r"//[^\n]*", "", source[start:end])
    members = {}
    following = 0
    for name, written in re.findall(r"\b(DSP_OP_[A-Z0-9_]+)\s*(?:=\s*([0-9]+))?", body):
        following = int(written) if written else following + 1
        members[name] = following
    return members


@pytest.fixture(scope="module")
def header_types():
    return _dsp_op_types_from_the_header()


def test_the_header_is_where_the_numbers_come_from(header_types):
    """The parse has to be reading a real table, not an empty or partial one."""
    assert len(header_types) >= 45, len(header_types)
    assert header_types["DSP_OP_POOL2D_FP16"] == 1
    assert header_types["DSP_OP_RASTER_BLIT"] == 3
    assert header_types["DSP_OP_IM2COL_CONVOLUTION_FP16"] == 12
    assert header_types["DSP_OP_MAX"] == 100


def test_every_command_type_in_the_header_has_a_python_name(header_types):
    mirror = {
        name: value
        for name, value in vars(hexagon_ops).items()
        if name.startswith("DSP_OP_")
    }
    missing = sorted(set(header_types) - set(mirror))
    assert not missing, (
        "these command types have no name in hexagon_ops, so anything reading the",
        "mirror reports them as unmapped:",
        missing,
    )
    wrong = sorted((name, mirror[name], value) for name, value in header_types.items() if mirror[name] != value)
    assert not wrong, ("the same name at two different numbers:", wrong)


def test_python_has_no_command_type_the_header_does_not(header_types):
    mirror = {
        name
        for name in vars(hexagon_ops)
        if name.startswith("DSP_OP_")
    }
    extra = sorted(mirror - set(header_types))
    assert not extra, ("a command type that the DSP's enum does not have:", extra)


def test_the_upsample_command_keeps_its_number(header_types):
    """46 was the free slot between 45 and 47, and 47 is a device reading."""
    assert header_types["DSP_OP_UPSAMPLE_BILINEAR2D_FP16"] == 46
    assert header_types["DSP_OP_ARGMAX_FP16"] == 47
    assert header_types["DSP_OP_MATMUL_W8A16_GEMV_I8"] == 45
    assert hexagon_ops.DSP_OP_UPSAMPLE_BILINEAR2D_FP16 == 46


def test_the_interpreter_copy_does_not_contradict_the_header(header_types):
    """The second hand-written copy, on the side that decides whether a blob runs.

    `blob_interpreter` names what it executes without the `DSP_OP_` prefix, so
    the comparison is on the name suffix: where both copies carry the same name,
    the numbers have to agree. A disagreement is a renumbered command, and it is
    the failure this test exists to make loud -- it would otherwise surface as a
    blob that builds, reaches the host, and is refused for no stated reason.
    """
    import blob_interpreter

    contradicted = []
    for name, value in vars(blob_interpreter).items():
        if not isinstance(value, int) or name.startswith("_"):
            continue
        header_name = "DSP_OP_" + name
        if header_name in header_types and header_types[header_name] != value:
            contradicted.append((name, value, header_types[header_name]))
    assert not contradicted, contradicted


def test_the_interpreter_executes_the_command_this_branch_added(header_types):
    import blob_interpreter

    assert blob_interpreter.UPSAMPLE_BILINEAR2D_FP16 == header_types[
        "DSP_OP_UPSAMPLE_BILINEAR2D_FP16"
    ]
    assert blob_interpreter.UPSAMPLE_BILINEAR2D_FP16 in blob_interpreter._EXECUTORS
