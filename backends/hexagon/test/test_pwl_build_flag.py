# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which of the two chord tables in unary_ops.cc the skel is actually built with.

unary_ops.cc holds two gelu banks, two sigmoid banks and two tanh banks, and
they are not the same numbers: the HTP_OPS_PWL_COMPANDED16 branch gives gelu
twelve chords ending 0x3c3e, 0x3c17, 0x3c06, 0x3c01 and the #else branch gives
it sixteen ending 0x3c08, 0x3c04, 0x3c02, 0x3c01. They agree for their first
eight entries and separate from there, and the index that reaches them is a
different function as well, so the choice is a whole second arithmetic and not
a detail of the table.

So the question is not which branch is right -- both compile, and the skel picks
one with a compile definition (skel/CMakeLists.txt:184) -- but whether anything
ties that choice to the transcription. Today nothing does. Every test that reads
the C++ reads the #if branch, because that is the branch the build sets; move the
definition to 0 and the tests go on reading the #if branch, the host interpreter
goes on answering twelve chords, and the device goes on computing sixteen. Every
one of them stays green. The failure is not a wrong answer in the tree, it is a
correct answer to a question the tree has quietly stopped asking the device.

That is worth a guard for the same reason test_pwl_bank_single_source.py is: one
definition, checked against the thing that makes it true. The two tests are the
reader and the reason the reader can fail. The first asserts that the build and
the transcription name the same branch; the second asserts that the branch they
do NOT name is a different table, so the first is not green because both branches
happen to be the same.
"""

import pathlib
import re

from blob_interpreter import _PWL_BIAS, _PWL_SLOPE

_TEST_DIR = pathlib.Path(__file__).resolve().parent
_BACKEND = _TEST_DIR.parent

_UNARY_OPS = (
    _BACKEND
    / "third-party"
    / "mnn-htp-ops"
    / "src"
    / "dsp"
    / "unary_ops.cc"
)
_PWL_H = _BACKEND / "third-party" / "mnn-htp-ops" / "include" / "dsp" / "pwl.h"
_SKEL_CMAKE = _BACKEND / "skel" / "CMakeLists.txt"

#: HTP_OPS_UNARY_GELU, the kind whose two banks differ in exactly the entries the
#: companded16 index can reach past the eighth.
_GELU = 3


def _branches():
    """The two #if branches as {name: bits}, plus the header's own default."""
    source = _UNARY_OPS.read_text()
    _head, rest = source.split("#if HTP_OPS_PWL_COMPANDED16", 1)
    on, rest = rest.split("#else", 1)
    off = rest.split("#endif", 1)[0]

    def tables(text):
        return {
            name: tuple(
                int(literal, 16)
                for literal in re.findall(r"0x[0-9a-fA-F]+", body)
            )
            for name, body in re.findall(
                r"HTP_OPS_PWL_TABLE\((\w+),\s*([^;]*?)\);", text, re.S
            )
        }

    header = _PWL_H.read_text()
    default = int(
        re.search(r"#\s*define\s+HTP_OPS_PWL_COMPANDED16\s+(\d+)", header).group(1)
    )
    return tables(on), tables(off), default


def _skel_defines_companded16():
    """The value the skel compiles with, or None when the line is gone."""
    match = re.search(
        r"HTP_OPS_PWL_COMPANDED16=(\d+)", _SKEL_CMAKE.read_text()
    )
    return None if match is None else int(match.group(1))


def test_the_skel_and_the_host_transcription_are_the_same_branch():
    """The link: the build sets the macro, and the macro is what selects the table."""
    on, _off, header_default = _branches()
    compiled = _skel_defines_companded16()
    assert compiled is not None, (
        f"{_SKEL_CMAKE} no longer sets HTP_OPS_PWL_COMPANDED16, so which bank the"
        f" skel runs is decided by the header default ({header_default}) and by"
        " nothing in this tree that a test can read"
    )
    assert compiled == 1, (
        f"the skel is built with HTP_OPS_PWL_COMPANDED16={compiled} while the host"
        " banks are transcribed from the #if HTP_OPS_PWL_COMPANDED16 branch, so the"
        " host and the device now compute different functions of one input"
    )
    assert compiled == header_default, (
        f"the skel compiles with {compiled} and the header defaults to"
        f" {header_default}, so a build that dropped the definition would pick a"
        " different bank from one that keeps it"
    )
    for cpp_name, bank in (
        ("gelu_slope", _PWL_SLOPE[_GELU]),
        ("gelu_bias", _PWL_BIAS[_GELU]),
        ("sigmoid_slope", _PWL_SLOPE[4]),
        ("sigmoid_bias", _PWL_BIAS[4]),
        ("tanh_slope", _PWL_SLOPE[8]),
        ("tanh_bias", _PWL_BIAS[8]),
    ):
        assert on[cpp_name] == bank[: len(on[cpp_name])], (
            f"{cpp_name} in the branch the skel is built with is not the bank the host"
            " interpreter answers from, so the model below is not the DSP's"
        )


def test_the_other_branch_is_a_different_table_so_the_link_can_break():
    """The control: a guard that cannot fail is not a guard."""
    on, off, _default = _branches()
    for name in ("gelu_slope", "gelu_bias"):
        assert off[name] != on[name], (
            f"the #else branch carries the same {name} as the companded16 one, so"
            " this file's claim that the build flag picks a different arithmetic no"
            " longer describes the source"
        )
    shared = 0
    for a, b in zip(on["gelu_slope"], off["gelu_slope"]):
        if a != b:
            break
        shared += 1
    assert shared == 8, (
        f"the two gelu slopes agree for their first {shared} entries and separate"
        " after, not the eight this records; a bank refitted upstream would land"
        " here as a red test rather than as a silent swap"
    )
    padded = tuple(off["gelu_slope"]) + (0,) * (16 - len(off["gelu_slope"]))
    assert padded != _PWL_SLOPE[_GELU], (
        "the host answers from the #else bank as well, so the test above cannot tell"
        " which of the two branches it read"
    )