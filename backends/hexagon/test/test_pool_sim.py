# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-block pooling on hexagon-sim, against an fp64 torch reference.

test_blob_on_sim.py carries the pool at one 64-channel block, which was for a
long time the only channel count the gate let through. This file carries the
widths: two whole blocks, a ragged tail either side of the boundary, a batch
above one, and an average whose divisor is a padded window. What is being tested
is the one claim the host tests cannot make -- that the DSP walks every block the
command names, writes each of them back, and leaves a lane that belongs to no
channel out of the answer.

This tier is the simulator, not silicon. A case passing here is evidence about
the kernels and their dispatch; it says nothing about a phone until the same blob
is run on one. The boundary is worth naming for the pool in particular, because
the DSP carries its own pack-area blit fast paths and which of those, or of the
generic three-level loop in htp_ops_raster_blit, ran depends on the region the
emitter wrote for that width.

The reference is computed in fp64 and rounded to fp16 once, so a disagreement has
to be the kernel rather than the reference. The control is the same data pooled
as one block and as two, which has to agree bit for bit: that is the check that
an extra block does not move the first one.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on the
# path makes "import executorch" resolve to this tree; the editable install in
# this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
    _Pool,
    _bits,
    _case,
    _fixture_header,
    _from_bits,
    _host_bits,
)

#: hvx_pool2d_fp16 narrows 1/count to fp16 before the multiply
#: (pool_fp16.c:83-85), so an average's divisor is not exact and neither is the
#: answer. A maximum returns one of the window's own values and is exact.
_AVERAGE_TOLERANCE = 5e-3

#: DSP_OP_POOL2D_FP16, DSP_OP_RASTER_BLIT and DSP_OP_ZERO: the commands a pool
#: of any width is built from, and the third only when the last block is ragged.
_POOL = 1
_BLIT = 3
_ZERO = 24

#: (tag, channels, batch, kind, kernel, stride, padding, count_include_pad).
#: Two letters each, in a range this file owns.
_SHAPES = [
    ("P0", 64, 1, "max", 2, 2, 0, True),   # the one width the gate used to take
    ("P1", 128, 1, "max", 2, 2, 0, True),  # two whole blocks
    ("P2", 128, 3, "max", 2, 2, 0, True),  # two whole blocks, a batch of three
    ("P3", 65, 1, "max", 2, 2, 0, True),   # a tail one channel short
    ("P4", 63, 1, "max", 2, 2, 0, True),   # a tensor one channel short of a block
    ("P5", 96, 1, "max", 2, 2, 0, True),   # a ragged block at a real SE width
    ("P6", 192, 1, "max", 2, 2, 0, True),  # three whole blocks
    ("P7", 128, 1, "avg", 3, 2, 1, True),  # divisor is the whole window
    ("P8", 128, 1, "avg", 3, 2, 1, False),  # divisor is what landed inside it
    ("P9", 96, 2, "avg", 3, 1, 1, True),   # ragged, a batch, stride one
    ("PA", 256, 1, "max", 2, 2, 0, True),  # four blocks: two blits a side
    ("PB", 33, 1, "avg", 2, 2, 0, True),   # a narrow ragged tail
]


def _operands(channels, batch, seed, size=8):
    """Values in [-4, 4) from a generator this file pins.

    The reference and the blob have to see the same bytes, which they do because
    both are built from this one call rather than from two torch calls.
    """
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.rand(batch, channels, size, size, generator=gen) * 8 - 4
    ).to(torch.float16)


def _out_shape(ih, iw, kernel, stride, padding, ceil_mode=False):
    """torch's output size, which is the geometry the command carries.

    Ceil mode is torch's own arithmetic and not the numerator's ceiling: it adds
    stride - 1 and then drops the last output position once, when that window
    would start at or past size + pad.
    """
    out_h = (ih + 2 * padding - kernel + (stride - 1 if ceil_mode else 0)) // stride + 1
    out_w = (iw + 2 * padding - kernel + (stride - 1 if ceil_mode else 0)) // stride + 1
    if ceil_mode:
        if (out_h - 1) * stride >= ih + padding:
            out_h -= 1
        if (out_w - 1) * stride >= iw + padding:
            out_w -= 1
    return out_h, out_w


def _reference(x, kind, kernel, stride, padding, count_include_pad, ceil_mode=False):
    """torch's answer, computed in fp64 and rounded once.

    This is not the kernel's arithmetic -- the kernel accumulates in fp16 and
    multiplies by an fp16 reciprocal -- it is the number the kernel is trying to
    reach, and the tolerance above is the width of the gap between the two.
    """
    wide = x.to(torch.float64)
    if kind == "max":
        out = torch.nn.functional.max_pool2d(
            wide, kernel, stride, padding, ceil_mode=ceil_mode
        )
    else:
        out = torch.nn.functional.avg_pool2d(
            wide,
            kernel,
            stride,
            padding,
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
        )
    return out.to(torch.float16)


def _pool_cases():
    cases = []
    for tag, channels, batch, kind, kernel, stride, padding, cip in _SHAPES:
        x = _operands(channels, batch, 1000 + channels + batch)
        model = _Pool(kind, kernel, stride, padding, count_include_pad=cip)
        expected = _reference(x, kind, kernel, stride, padding, cip)
        cases.append(
            _case(
                tag,
                model,
                (x,),
                _bits(expected),
                kind="close" if kind == "avg" else "bits",
                tolerance=_AVERAGE_TOLERANCE if kind == "avg" else None,
            )
        )
    return cases


@pytest.fixture(scope="module")
def cases():
    # Built lazily, so a blob that cannot be produced fails a test rather than
    # aborting collection with a traceback nobody can attribute.
    return _pool_cases()


def _run(cases):
    # _check() first, and no catch anywhere: run() raises the same Unavailable for
    # "no SDK" and for "the simulator did not come up", so a fixture that
    # swallows it turns a simulator that failed to run into a green skip.
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


@pytest.fixture(scope="module")
def simulated(cases):
    return _run(cases)


def test_the_blobs_carry_the_blocks_their_widths_ask_for(cases):
    """A blob that lowered to something else would test nothing.

    The claim under test is about the channel count, so it is read back out of
    the command rather than inferred from the model that produced it.
    """
    tags = [case.tag for case in cases]
    assert len(set(tags)) == len(tags), "two fixtures share a tag"
    for (tag, channels, batch, kind, k, s, p, cip), case in zip(_SHAPES, cases):
        types = [command.type for command in case.commands]
        pool = next(c for c in case.commands if c.type == _POOL)
        blocks = -(-channels // 64)
        assert int(pool.params[0]) == batch, f"{tag}: the batch is not the command's"
        assert int(pool.params[5]) == blocks, f"{tag}: c4 is not {channels} channels"
        assert int(pool.params[1]) == 8 and int(pool.params[2]) == 8
        blits = [c for c in case.commands if c.type == _BLIT]
        assert sum(int(c.params[0]) for c in blits) == 2 * blocks, (
            f"{tag}: the blits do not move one region per block a side"
        )
        # A ragged last block is one the pack does not fill, so a memset has to
        # precede the pool; a whole-block width has no lanes to clear.
        assert (_ZERO in types) is (channels % 64 != 0), f"{tag}: the tail is not zeroed"
        assert types.count(_POOL) == 1


@pytest.mark.parametrize(
    "tag, channels, batch, kind, kernel, stride, padding, cip",
    _SHAPES,
    ids=[shape[0] for shape in _SHAPES],
)
def test_the_dsp_pools_every_block(
    simulated, cases, tag, channels, batch, kind, kernel, stride, padding, cip
):
    """One width at a time, against the fp64 reference, with a non-empty answer.

    The non-empty part is the point: an invisible runner leaves the slot as the
    arena's fill and answers zeros, which is finite and entirely wrong. The key
    is tag+index because that is what run() returns, and the answer is compared
    with the reference rather than with the host model, so this is a statement
    about the kernels and not about two models of them agreeing.
    """
    case = next(c for c in cases if c.tag == tag)
    shape = (batch, channels) + _out_shape(8, 8, kernel, stride, padding)
    dsp = simulated[tag + "0"]
    assert dsp, f"{tag}: the simulator returned nothing for this fixture"
    got = _from_bits(dsp).reshape(shape)
    expected = _from_bits(
        [int(v) for v in case.expected.view(np.uint16).tolist()]
    ).reshape(shape)
    if kind == "max":
        # A maximum returns one of the window's own values; there is nothing to
        # be inexact about, so this is an equality.
        assert got.tobytes() == expected.tobytes(), (
            f"{tag}: the DSP disagrees with the fp64 reference"
        )
    else:
        np.testing.assert_allclose(
            got.astype(np.float64),
            expected.astype(np.float64),
            rtol=_AVERAGE_TOLERANCE,
            atol=_AVERAGE_TOLERANCE,
            err_msg=f"{tag}: the DSP disagrees with the fp64 reference",
        )
    # The host model has to agree too, or the disagreement would be between two
    # implementations of the layout rather than against the reference.
    host = _from_bits([int(v) for v in _host_bits(case.host[0])]).reshape(shape)
    if kind == "max":
        assert host.tobytes() == expected.tobytes(), f"{tag}: the host model disagrees"
    else:
        np.testing.assert_allclose(
            host.astype(np.float64),
            expected.astype(np.float64),
            rtol=_AVERAGE_TOLERANCE,
            atol=_AVERAGE_TOLERANCE,
            err_msg=f"{tag}: the host model disagrees with the fp64 reference",
        )


def test_one_block_and_two_blocks_are_the_same_kernel_on_the_dsp():
    """The control, run on the simulator rather than on the host.

    The same bytes pooled as one 64-block and as two have to answer identically
    over the first block, or a second block is not merely more of the same walk.
    The two blobs are in one run rather than two so both are answered by the
    same build, which is what makes the comparison a statement about the block
    count and not about two different binaries.
    """
    data = _operands(64, 1, 4242)
    wide = torch.cat([data, data], dim=1).contiguous()
    expected = _reference(data, "max", 2, 2, 0, True)
    narrow = _case(
        "PC", _Pool("max", 2, 2, 0), (data,), _bits(expected)
    )
    doubled = _case(
        "PD",
        _Pool("max", 2, 2, 0),
        (wide,),
        _bits(torch.cat([expected, expected], dim=1)),
    )
    answer = _run([narrow, doubled])
    one = _from_bits(answer["PC0"]).reshape(1, 64, 4, 4)
    two = _from_bits(answer["PD0"]).reshape(1, 128, 4, 4)
    assert one.size and two.size, "the control returned nothing"
    np.testing.assert_array_equal(one, expected.numpy())
    assert two[:, :64].tobytes() == one.tobytes(), (
        "pooling the same data as two blocks moved the first one"
    )
    assert two[:, 64:].tobytes() == one.tobytes(), (
        "the second block is not the first one"
    )


#: The ceil-mode shapes, one letter each in a range this file owns. Every one of
#: them is a size where ceil mode changes the answer, which is the point: the
#: floor cases above already pin the window walk, and a case where ceil equals
#: floor would test the spelling rather than the geometry.
#:
#: (tag, channels, kind, kernel, stride, padding, count_include_pad, size).
_CEIL_SHAPES = [
    ("Q0", 64, "max", 3, 3, 0, True, 7),   # the last window holds 1x1 of 3x3
    ("Q1", 64, "max", 3, 2, 1, True, 10),  # a stride of two at a resnet stem size
    ("Q2", 128, "max", 2, 2, 0, True, 9),  # two whole blocks
    ("Q3", 96, "max", 3, 2, 1, True, 10),  # a ragged block and a padding
    ("Q4", 64, "avg", 3, 3, 0, False, 8),  # the divisor is the 2x2 that is left
    ("Q5", 64, "avg", 3, 2, 1, False, 10),
    ("Q6", 96, "avg", 2, 3, 0, False, 7),  # a stride wider than the kernel stride
    ("Q7", 65, "avg", 3, 4, 1, False, 12), # a narrow tail at a wide stride
    ("Q8", 128, "avg", 3, 2, 1, False, 10), # the same window over two blocks
]

#: The one average that counts the padding *and* delegates. It is here because
#: the refusal above is exactly this row with a size where ceil mode changes the
#: answer: the divisor test holds precisely when ceil mode does not change the
#: shape, so an average over the whole kernel is portable iff it would have been
#: a different answer. At 9x9 with a 3x3 window at stride 4, ceil mode and floor
#: both give 3, the divisor is the whole kernel, and the node runs -- which is
#: the half of the equivalence no case above reaches.
_CEIL_NOOP_SHAPES = [
    ("Q9", 64, "avg", 3, 4, 1, True, 9),
]


def _ceil_cases():
    cases = []
    for tag, channels, kind, kernel, stride, padding, cip, size in _CEIL_SHAPES:
        x = _operands(channels, 1, 3000 + size * 10 + kernel, size=size)
        model = _Pool(
            kind, kernel, stride, padding, count_include_pad=cip, ceil_mode=True
        )
        expected = _reference(
            x, kind, kernel, stride, padding, cip, ceil_mode=True
        )
        floor = _Pool(kind, kernel, stride, padding, count_include_pad=cip)
        assert expected.shape != floor(x).shape, (
            f"{tag}: ceil mode does not change this shape, so it is not a case"
        )
        cases.append(
            _case(
                tag,
                model,
                (x,),
                _bits(expected),
                kind="close" if kind == "avg" else "bits",
                tolerance=_AVERAGE_TOLERANCE if kind == "avg" else None,
            )
        )
    for tag, channels, kind, kernel, stride, padding, cip, size in _CEIL_NOOP_SHAPES:
        x = _operands(channels, 1, 3000 + size * 10 + kernel, size=size)
        model = _Pool(
            kind, kernel, stride, padding, count_include_pad=cip, ceil_mode=True
        )
        expected = _reference(
            x, kind, kernel, stride, padding, cip, ceil_mode=True
        )
        floor = _Pool(kind, kernel, stride, padding, count_include_pad=cip)
        assert expected.shape == floor(x).shape, (
            f"{tag}: ceil mode changes this shape, so it is not a control"
        )
        cases.append(
            _case(
                tag,
                model,
                (x,),
                _bits(expected),
                kind="close",
                tolerance=_AVERAGE_TOLERANCE,
            )
        )
    return cases


@pytest.fixture(scope="module")
def ceil_cases():
    return _ceil_cases()


@pytest.fixture(scope="module")
def ceil_simulated(ceil_cases):
    hexagon_sim._check()
    return _run(ceil_cases)


def test_a_ceil_mode_blob_carries_torchs_own_extents(ceil_cases):
    """The command is the one floor mode emits, at a different size.

    Ceil mode is not a second command: it is the same fifteen params with oh and
    ow replaced by torch's numbers, so a case whose blob carried anything else
    would be measuring the emitter rather than the kernel. The divisor is read
    rather than assumed, and the two tables between them cover both of its
    values -- the averages above are all countType 0 and this one is countType 1.
    """
    for case, shape in zip(ceil_cases, _CEIL_SHAPES + _CEIL_NOOP_SHAPES):
        tag, channels, kind, kernel, stride, padding, cip, size = shape
        pool = next(c for c in case.commands if c.type == _POOL)
        want = _reference(
            _operands(channels, 1, 3000 + size * 10 + kernel, size=size),
            kind,
            kernel,
            stride,
            padding,
            cip,
            ceil_mode=True,
        )
        assert int(pool.params[1]) == size and int(pool.params[2]) == size
        assert (int(pool.params[3]), int(pool.params[4])) == tuple(
            want.shape[-2:]
        ), f"{tag}: the command does not carry torch's ceil extent"
        assert int(pool.params[6]) == kernel and int(pool.params[7]) == kernel
        assert int(pool.params[13]) == (0 if kind == "avg" and not cip else 1), (
            f"{tag}: the divisor is not the one this shape asks for"
        )


@pytest.mark.parametrize(
    "tag, channels, kind, kernel, stride, padding, cip, size",
    _CEIL_SHAPES + _CEIL_NOOP_SHAPES,
    ids=[shape[0] for shape in _CEIL_SHAPES + _CEIL_NOOP_SHAPES],
)
def test_the_dsp_pools_a_window_that_runs_off_the_edge(
    ceil_simulated, ceil_cases, tag, channels, kind, kernel, stride, padding, cip, size
):
    """This tier's whole reason to exist.

    The host interpreter models pool_fp16.c, so a green host case says the
    command is shaped the way the kernel's source reads and nothing about the
    kernel. Here the DSP walks a window whose last row or column is past the
    input, and the answer is torch's: the clip at pool_fp16.c:36-43, the
    divisor at :74-79, and the output extent are all three exercised by a case
    where floor mode would have produced a different tensor, plus the one case
    where an average over the whole kernel delegates because it does not.
    """
    case = next(c for c in ceil_cases if c.tag == tag)
    dsp = ceil_simulated[tag + "0"]
    assert dsp, f"{tag}: the simulator returned nothing for this fixture"
    shape = (1, channels) + _out_shape(size, size, kernel, stride, padding, True)
    got = _from_bits(dsp).reshape(shape)
    expected = _from_bits(
        [int(v) for v in case.expected.view(np.uint16).tolist()]
    ).reshape(shape)
    if kind == "max":
        assert got.tobytes() == expected.tobytes(), (
            f"{tag}: the DSP disagrees with the fp64 reference"
        )
    else:
        np.testing.assert_allclose(
            got.astype(np.float64),
            expected.astype(np.float64),
            rtol=_AVERAGE_TOLERANCE,
            atol=_AVERAGE_TOLERANCE,
            err_msg=f"{tag}: the DSP disagrees with the fp64 reference",
        )
