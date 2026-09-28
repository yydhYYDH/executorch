# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CAMPPlus's SE pooling geometry, on hexagon-sim.

CAMPPlus (iic/speech_campplus_zh-cn_common) puts one squeeze-and-excitation
block in each of its 52 TDNN blocks, and every one of them pools with

    avg_pool2d(x, kernel=(1, 100), stride=(1, 100), padding=(0, 0),
               ceil_mode=True, count_include_pad=True)

over a 150-frame input. Ceil mode makes that two output positions, the second
of which is a window over columns 100..199 of a 150-column input, so it holds
50 real columns out of the 100 its kernel names. torch divides that position by
50; pool_fp16.c's countType 1 divides by kY*kX, which is 100. The emitter
refuses the node for that reason, and refuses correctly -- the answer it would
have produced is half the right one. 52 nodes, 52 breaks in the delegate chain.

This file asks the question that refusal raises, at this model's own geometry:
at padding=0 torch's two count_include_pad settings are the same average over the
same elements, so the spelling the rewrite would produce divides by what is
actually inside the window -- which is the count type the DSP already has, and
which this geometry already admits. The two halves of that are separate tests,
because one of them cannot be built into a blob at all:

* count_include_pad=True is what the model emits and pool_spec refuses it;
* count_include_pad=False is what a host-side rewrite would emit, it lowers to
  count_type=0, and the DSP's answer is the fp64 torch reference.

Q4 in test_pool_sim.py is the same claim on a 3x3 window over 8 columns. This
one is what a 52-layer network actually emits, at 128 channels, and the point of
running it is that the refusal is about this geometry and not about the 3x3.

This tier is the simulator, not silicon. A case passing here is evidence about
the kernel and its dispatch, and says nothing about a phone.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
import test_blob_on_sim as TB  # noqa: E402
from executorch.backends.hexagon import hexagon_ops as HO  # noqa: E402
from executorch.exir.program._program import to_edge  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
    _bits,
    _case,
    _fixture_header,
    _from_bits,
)
from test_pool_sim import (  # noqa: E402
    _AVERAGE_TOLERANCE,
    _operands,
    _out_shape,
    _reference,
)

#: What CAMPPlus emits, per SE block, per channel block. The squeeze runs
#: over time only: the height axis is 1 and keeps a 1-wide window, so the whole
#: of ceil mode's overhang is on the width axis.
_CHANNELS = 128
_HEIGHT = 1
_WIDTH = 150
_KERNEL = (1, 100)
_STRIDE = (1, 100)
_PADDING = 0

#: The spelling the model emits today, and the one a rewrite at padding=0 would
#: emit. Only the second can be built into a blob.
_MODEL_CIP = True
_REWRITE_CIP = False

_TAG = "C0"


def _se_operands(seed):
    """CAMPPlus's own operand shape: the squeeze pools time and not height."""
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.rand(1, _CHANNELS, _HEIGHT, _WIDTH, generator=gen) * 8 - 4
    ).to(torch.float16)


def _pool_node(cip):
    model = TB._Pool(
        "avg", _KERNEL, _STRIDE, _PADDING, count_include_pad=cip, ceil_mode=True
    ).eval()
    x = torch.randn(1, _CHANNELS, _HEIGHT, _WIDTH)
    program = to_edge(torch.export.export(model, (x,), strict=False)).exported_program()
    return next(
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and "pool" in str(node.target)
    )


@pytest.fixture(scope="module")
def case():
    x = _se_operands(7000 + _WIDTH)
    model = TB._Pool(
        "avg",
        _KERNEL,
        _STRIDE,
        _PADDING,
        count_include_pad=_REWRITE_CIP,
        ceil_mode=True,
    )
    expected = _reference(
        x, "avg", _KERNEL, _STRIDE, _PADDING, _REWRITE_CIP, ceil_mode=True
    )
    assert tuple(expected.shape) == (1, _CHANNELS, 1, 2), (
        f"not this model's geometry: {tuple(expected.shape)}"
    )
    return _case(
        _TAG, model, (x,), _bits(expected), kind="close", tolerance=_AVERAGE_TOLERANCE
    )


@pytest.fixture(scope="module")
def simulated(case):
    hexagon_sim._check()
    return hexagon_sim.run(
        _RUNNER,
        _SOURCES,
        headers={
            "blob_fixture.h": _fixture_header([case]),
            "htp_ops.h": _HT_P_OP_SHIM,
        },
        includes_more=[str(_SCHEMA)],
    )


def test_the_geometry_is_the_one_campplus_emits(case):
    """A 3x3 window would test the spelling rather than this model's shape.

    128 channels of a one-row input squeezed to two positions, so 256 values
    come back where a 3x3 window over 8 would be a different count entirely.
    The fixture has already asserted the fp64 reference's own shape; this pins
    the flattened form the runner reads.
    """
    assert case.expected.size == _CHANNELS * 2, case.expected.shape


def test_the_models_own_spelling_is_refused():
    """The gap, recorded at the geometry that has it.

    torch's divisor for the second position is 50 and the kernel's countType 1
    divides by 100, so this node is one the DSP cannot compute correctly and
    pool_spec is right to refuse it. 52 of these is 52 breaks in the chain.
    """
    assert HO.pool_spec(_pool_node(_MODEL_CIP)) is None


def test_the_rewrites_spelling_is_what_the_kernel_can_do():
    """The premise, and it is a premise about arithmetic rather than spelling."""
    spec = HO.pool_spec(_pool_node(_REWRITE_CIP))
    assert spec is not None, "padding=0 was supposed to be enough to admit this"
    assert spec.count_type == HO.POOL_COUNT_VALID
    assert (spec.oh, spec.ow) == (1, 2)


def test_the_two_settings_would_get_the_same_number():
    """What makes the rewrite sound: at padding=0 they are the same average."""
    x = _se_operands(7000 + _WIDTH).to(torch.float64)
    args = (_KERNEL, _STRIDE, _PADDING)
    with_pad = torch.nn.functional.avg_pool2d(
        x, *args, count_include_pad=_MODEL_CIP, ceil_mode=True
    )
    without_pad = torch.nn.functional.avg_pool2d(
        x, *args, count_include_pad=_REWRITE_CIP, ceil_mode=True
    )
    assert with_pad.shape[-1] == 2
    assert torch.equal(with_pad, without_pad), (
        "padding=0 is the condition this rewrite depends on"
    )


def test_the_dsp_se_pools_the_overhanging_window(simulated, case):
    """The claim a rewrite would rest on, measured rather than assumed.

    The window at the second position runs 50 columns past the end of what the
    kernel names, and the answer is still torch's.
    """
    dsp = simulated.get(_TAG + "0")
    assert dsp, "the simulator returned nothing for this fixture"
    shape = tuple(case.expected.shape)
    got = _from_bits(dsp).reshape(shape)
    expected = _from_bits(
        [int(v) for v in case.expected.view(np.uint16).tolist()]
    ).reshape(shape)
    np.testing.assert_allclose(
        got.astype(np.float64),
        expected.astype(np.float64),
        rtol=_AVERAGE_TOLERANCE,
        atol=_AVERAGE_TOLERANCE,
        err_msg="the DSP disagrees with the fp64 reference",
    )
