# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The shape table behind `select_condition_plane`, and why each row is what it is.

Two questions are asked of every pair, and the second is the one a happy-path case
cannot answer. The first is what plane the predicate names. The second is whether
that plane is the one the kernel would use: the kernel derives each operand mode
from that operand size and consults the plane only in the per-channel mode, so a
plane that disagreed with the mode the size implies would be a descriptor the
kernel rejects with -2 -- the predicate saying yes to anything the DSP refuses.
And with a plane in hand, the walk it names has to be the broadcast, which is
computed here by numpy rather than restated by this file, so a walk that tiled
the condition along the wrong axis would show up as a difference instead of
agreeing with a bad reference.

The rows are pairs, not cases: each refusal sits next to the shape that is the
same graph one axis away, so a predicate that refused every broadcast would fail
and a predicate that admitted every broadcast would fail too.
"""


import numpy as np
import pytest

from executorch.backends.hexagon.hexagon_ops import (
    NO_SELECT_PLANE,
    select_condition_plane,
)


def _numel(shape):
    total = 1
    for extent in shape:
        total *= extent
    return total


def _kernel_mode(size, out_size):
    """The walk htp_ops_select takes for an operand of this size, from its size."""
    if size == 1:
        return 0
    if size == out_size:
        return 1
    return 2


def _broadcast(cond_shape, out_shape):
    """The torch broadcast of a non-repeating condition, computed by numpy."""
    flags = np.array(
        [(i * 7 + 3) % 2 == 0 for i in range(_numel(cond_shape))], dtype=bool
    )
    padded = (1,) * (len(out_shape) - len(cond_shape)) + tuple(cond_shape)
    return np.broadcast_to(flags.reshape(padded), out_shape).reshape(-1), flags


#: (condition shape, output shape, what the predicate is expected to say). None
#: means the node is refused, and the DSP would refuse the descriptor too; a pair
#: is the plane the command has to carry.
PAIRS = [
    # The two walks that name no plane, which the backend already emitted.
    ((2, 3, 4), (2, 3, 4), NO_SELECT_PLANE),
    ((1, 2, 3, 3), (1, 2, 3, 3), NO_SELECT_PLANE),
    ((1,), (1, 2, 3, 4), NO_SELECT_PLANE),
    ((1, 1, 1), (2, 3, 4), NO_SELECT_PLANE),
    ((1, 1, 1, 1), (2, 3, 4, 5), NO_SELECT_PLANE),
    # The staircase, at every repeat the context axis can have and at every rank.
    ((1, 2, 3, 1), (1, 2, 3, 3), (6, 3)),
    ((1, 2, 3, 1), (1, 2, 3, 8), (6, 8)),
    ((1, 2, 3, 1), (1, 2, 3, 4), (6, 4)),
    ((2, 3, 1), (2, 3, 4), (6, 4)),
    ((1, 2, 6, 1), (1, 2, 6, 2), (12, 2)),
    ((2, 3, 1, 1), (2, 3, 4, 4), (6, 16)),
    ((1, 2, 3, 1, 1), (1, 2, 3, 4, 4), (6, 16)),
    ((1, 2, 1, 1, 1), (1, 2, 3, 4, 4), (2, 48)),
    ((2, 1, 1), (2, 3, 4), (2, 12)),
    ((1, 2, 3, 1), (1, 2, 3, 1), NO_SELECT_PLANE),
    # A condition narrow on no axis at its own size, which is a whole-output walk.
    ((1, 2, 3), (1, 2, 3), NO_SELECT_PLANE),
    # The middle axis, which the walk cannot express. These four are the same
    # graph one axis away from a staircase: a condition narrow on an outer axis as
    # well as on the suffix has an index that is not a function of index/innerSize.
    ((1, 1, 3, 1), (1, 2, 3, 4), None),
    ((1, 3, 1), (2, 3, 4), None),
    ((1, 2, 1, 3), (1, 2, 4, 3), None),
    ((1, 2, 2, 3), (1, 2, 4, 3), None),
    # A condition narrow on no axis whose size is neither one nor the output: no
    # plane describes it, and the kernel would take the per-channel walk anyway.
    # This is the -2 this table exists to keep off the wire.
    ((1, 2, 3), (2, 2, 3), None),
    ((2, 1, 4), (2, 3, 4), None),
    ((3, 4), (2, 3, 4), None),
    ((1, 2, 3, 1), (1, 2, 4, 3), None),
    ((1, 2, 3, 1), (1, 2, 3, 3, 3), None),
    ((1, 2, 2, 1, 1), (1, 2, 3, 4, 4), None),
    # Not broadcastable at all, and the rank rule that catches it.
    ((2, 3, 4, 1), (2, 3, 4), None),
    ((2, 3, 4), (2, 3, 4, 1), None),
    ((5,), (2, 3, 4), None),
    ((3,), (1, 2, 3, 5), None),
]


@pytest.mark.parametrize("cond_shape,out_shape,expected", PAIRS)
def test_the_predicate_names_the_plane_the_kernel_would_walk(
    cond_shape, out_shape, expected
):
    assert select_condition_plane(cond_shape, out_shape) == expected


@pytest.mark.parametrize("cond_shape,out_shape,expected", PAIRS)
def test_a_named_plane_is_one_the_kernel_accepts(cond_shape, out_shape, expected):
    """The plane, checked against the kernel arithmetic rather than against itself.

    The guard is arithmetic: a per-channel operand has to name a channel size of
    its own, and a positive repeat. So a predicate that returned a plane whose
    channel was not the condition element count, or one for a condition the kernel
    would not take the per-channel walk for at all, would produce a descriptor the
    DSP answers -2 -- and this is the assertion that says so on the host rather
    than on the phone.
    """
    plane = select_condition_plane(cond_shape, out_shape)
    cond_numel, out_numel = _numel(cond_shape), _numel(out_shape)
    mode = _kernel_mode(cond_numel, out_numel)
    if mode != 2:
        # The kernel never consults the plane in the other two modes, so all that
        # matters is that the predicate did not name a channel.
        assert plane is None or plane == NO_SELECT_PLANE
        return
    if plane is None:
        return
    channel, inner = plane
    assert channel == cond_numel, (
        f"the kernel guard wants channelSize == {cond_numel}, got {channel}"
    )
    assert inner > 0, "a per-channel walk with a zero repeat reads element zero"


@pytest.mark.parametrize("cond_shape,out_shape,expected", PAIRS)
def test_the_walk_the_plane_names_is_torch_broadcast(
    cond_shape, out_shape, expected
):
    """And that the walk it names is the broadcast rather than a plausible neighbour.

    The condition is filled with a pattern no all-equal mask could imitate, and the
    reference is the numpy broadcast, so reading the condition at its own stride,
    tiling it along the wrong axis, or dividing by any repeat but the one the plane
    carries each answer something different here.
    """
    plane = select_condition_plane(cond_shape, out_shape)
    if plane is None:
        return
    cond_numel, out_numel = _numel(cond_shape), _numel(out_shape)
    full, flags = _broadcast(cond_shape, out_shape)
    mode = _kernel_mode(cond_numel, out_numel)
    if mode == 0:
        at = np.zeros(out_numel, dtype=np.intp)
    elif mode == 1:
        at = np.arange(out_numel, dtype=np.intp)
    else:
        at = (np.arange(out_numel, dtype=np.intp) // plane[1]) % plane[0]
    assert at.max(initial=0) < cond_numel, "the walk reads past the condition"
    assert np.array_equal(full, flags[at]), (
        f"the walk answers {flags[at].tolist()}, broadcast is {full.tolist()}"
    )
