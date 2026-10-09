# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused masked reduction, and the width it is exact at, run on hexagon-sim.

`test_masked_reduction.py` pins the command stream and the host model. This file
is the other half of the claim: the one command the fusion emits runs on the
real kernel, and what comes back is the two-command lowering's own bits. The
fused blob and the unfused one are built from the same activation and the same
mask, so if the kernels differed by one fp16 step there would be nothing to buy.

The second thing measured here is the width. Reading the two kernels says the
masked kernel's scalar tail covers a row below 64 elements and that
`htp_ops_reduction` accumulates in fp32; what neither says, and what the widths
below are here to measure, is that the *plain* reduction walks a row with its
scalar path only below 32, and that its small-vector walk between 32 and 64 is
not fp32-exact -- an fp32 add there is the qf32 add-then-normalize pair
(eltwise_ops.cc:2737-2740). So the admitted width is 32 and not 64: above it the
merge moves the answer, and between 32 and 64 it moves it in the direction of
being *more* accurate than the two commands, which is a different lowering.

Every case runs one blob, so the commands, the geometry and the answers come out
of one simulator invocation. Nothing here is hand-written.

This file, like the other `*_on_sim` modules, says "the DSP" for the kernel as
the simulator runs it: the arithmetic and the dispatch are the device's, the
silicon is not. Nothing here has run on a device.
"""

import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on the
# path makes `import executorch` resolve to this tree. Without it the editable
# install wins, and in this environment that points at a different checkout.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import hexagon_sim  # noqa: E402
from blob_interpreter import Arena, execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _fixture_header,
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
)
from torch.export import export  # noqa: E402

_MASKED_REDUCTION = 31
_REDUCTION = 29
_BINARY = 19
_SUM = 1
_FP16_BYTES = 2

#: The geometry of the two cases the equality is claimed for, and the wider row
#: the plain kernel loses its last bits at. Three axes where the middle one is
#: the mask's own axis, so a transposed pair of strides cannot agree by accident.
#: Eight and thirty-one are the admitted end of the width; forty is a width the
#: plain kernel's small-vector walk covers and thirty-one is below it, and
#: sixteen and sixty-four are the widths of the scalar and HVX walks, which are
#: the two the boundary is measured against.
_OUTSIDE, _REDUCE = 6, 5
_ADMITTED = (31, 8)
#: One plain reduction per walk of `htp_ops_reduction`: the scalar path, its
#: small-vector path, and the HVX path the FP32 accumulate flag reaches.
_PLAIN = (("E", 16), ("F", 31), ("G", 40), ("H", 64))


def _inputs(inside, seed=4):
    generator = np.random.default_rng(seed)
    a = (generator.standard_normal((_OUTSIDE, _REDUCE, inside)) * 0.5).astype(
        np.float16
    )
    mask = (generator.random((_OUTSIDE, _REDUCE)) < 0.5).astype(np.float16)
    mask[0, :] = 1.0
    mask[1, :] = 0.0
    return a, mask


class _ConstantMask(torch.nn.Module):
    """The mask as a buffer, which is the shape the gate admits."""

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, a):
        return (a * self.mask.unsqueeze(-1)).sum(-2)


class _RuntimeMask(torch.nn.Module):
    """The same arithmetic with the mask a caller hands in, which is refused."""

    def forward(self, a, m):
        return (a * m.unsqueeze(-1)).sum(-2)


class _SumLast(torch.nn.Module):
    """The reduction on its own: no multiply, so its arithmetic is the kernel's."""

    def forward(self, x):
        return x.sum(-2)


def _blob_for(model, args):
    program = to_edge(
        export(model, args),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    return bytes(HexagonBackend.preprocess(program, []).processed_bytes)


def _bits(values):
    return np.ascontiguousarray(values).view(np.uint16)


def _fp32_sum(x, inside):
    """The row sum accumulated in fp32 and rounded once, which is the model the
    interpreter uses and the arithmetic two of the kernel's three paths do."""
    return x.astype(np.float32).sum(axis=1).astype(np.float16).reshape(-1)


class _Case:
    """One blob: its tag, the bytes, the method inputs and the mask it holds."""

    def __init__(self, tag, blob, args, kind, mask):
        self.tag = tag
        self.blob = blob
        self.args = args
        self.kind = kind
        self.mask = mask


def _cases():
    cases = []
    for letter, inside in zip("AC", _ADMITTED):
        a, mask = _inputs(inside)
        a_tensor, mask_tensor = torch.from_numpy(a), torch.from_numpy(mask)
        fused = _blob_for(_ConstantMask(mask_tensor), (a_tensor,))
        plain = _blob_for(_RuntimeMask(), (a_tensor, mask_tensor))
        cases.append(_Case(letter, fused, (a_tensor,), "fused", mask_tensor))
        cases.append(
            _Case(
                chr(ord(letter) + 1),
                plain,
                (a_tensor, mask_tensor),
                "plain",
                mask_tensor,
            )
        )
    for letter, inside in _PLAIN:
        x = torch.from_numpy(_inputs(inside, seed=25)[0])
        cases.append(_Case(letter, _blob_for(_SumLast(), (x,)), (x,), "reduction", None))
    return cases


@pytest.fixture(scope="module")
def measured():
    """The blobs, their commands, the host model's answers and the DSP's."""
    cases = _cases()
    tags = [case.tag for case in cases]
    assert tags == list("ABCDEFGH"), tags

    fixtures, commands, host, arenas = [], {}, {}, {}
    for case in cases:
        tag, blob, args = case.tag, case.blob, case.args
        header, decoded = read_blob(blob)
        commands[tag] = decoded
        arena = Arena(header, blob, "fixture")
        arenas[tag] = arena
        fixtures.append(
            SimpleNamespace(
                tag=tag,
                blob=blob,
                inputs=b"".join(t.numpy().tobytes() for t in args),
                length=0,
                arena_bytes=len(arena.bytes),
            )
        )
        host[tag] = np.frombuffer(
            execute(blob, [t.numpy() for t in args], arena=arena)[0],
            dtype=np.float16,
        )

    try:
        answers = hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={
                "blob_fixture.h": _fixture_header(fixtures),
                "htp_ops.h": _HT_P_OP_SHIM,
            },
            includes_more=[str(_SCHEMA)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))

    dsp = {
        tag: np.asarray(answers[f"{tag}0"], dtype=np.uint16).view(np.float16)
        for tag in tags
    }
    return SimpleNamespace(
        cases={case.tag: case for case in cases},
        commands=commands,
        host=host,
        arena=arenas,
        dsp=dsp,
    )


def _args(measured, tag):
    return measured.cases[tag].args


def test_the_blobs_carry_the_commands_the_gate_decided(measured):
    """One command where the gate fires, two where it stands down, and the same
    span either way -- the merge moves a command and not a number."""
    for tag in "AC":
        command = measured.commands[tag]
        assert [c.type for c in command] == [_MASKED_REDUCTION], (
            f"{tag}: {[c.type for c in command]} rather than one "
            f"{_MASKED_REDUCTION}"
        )
        inside = _args(measured, tag)[0].shape[-1]
        assert list(command[0].params) == [
            _OUTSIDE,
            _REDUCE,
            inside,
            _SUM,
            _FP16_BYTES,
        ], f"{tag}: the fused params are {list(command[0].params)}"
        # The mask really is the kernel's operand, read one entry per
        # (outside, reduce) pair: two inputs, and its bytes are the mask's.
        assert len(command[0].inputs) == 2
        assert (
            bytes(measured.arena[tag].view(command[0].inputs[1]))
            == measured.cases[tag].mask.numpy().tobytes()
        ), f"{tag}: the second operand is not the mask"
    for tag in "BD":
        command = measured.commands[tag]
        assert [c.type for c in command] == [_BINARY, _REDUCTION], (
            f"{tag}: the refused case is carrying {[c.type for c in command]}"
        )
        inside = _args(measured, tag)[0].shape[-1]
        assert list(command[1].params) == [
            _OUTSIDE,
            _REDUCE,
            inside,
            _SUM,
            _FP16_BYTES,
        ], f"{tag}: the reduction's params are {list(command[1].params)}"


@pytest.mark.parametrize("fused, plain", [("A", "B"), ("C", "D")])
def test_the_fused_command_is_the_two_command_answer_bit_for_bit(measured, fused, plain):
    """The primary claim, on the kernels rather than on a model of them.

    Both blobs hold the same activation and the same mask, so at the admitted
    width the two have to agree to the last bit: the masked kernel accumulates
    in fp32 over the rows the mask keeps, and the two commands accumulate in
    fp32 over the same rows with an exact 1.0 or 0.0 factor in front.
    """
    got, want = measured.dsp[fused], measured.dsp[plain]
    assert np.array_equal(_bits(got), _bits(want)), (
        f"the kernel's fused answer moved {int((_bits(got) != _bits(want)).sum())}"
        f" of {got.size} outputs by up to "
        f"{np.abs(got.astype(np.float32) - want.astype(np.float32)).max():.3e}"
    )


@pytest.mark.parametrize("tag", list("ABCD"))
def test_both_legs_are_torch_within_one_fp16_step(measured, tag):
    """The tolerance class, stated in the output's own fp16 step.

    A reduction accumulates, so this is a tolerance and not an equality, and the
    step of the largest output is the unit it is stated in: an absolute number
    would say nothing about a sum whose size is the row count.
    """
    inside = _args(measured, tag)[0].shape[-1]
    a, mask = _inputs(inside)
    want = (
        (a.astype(np.float32) * mask[:, :, None].astype(np.float32))
        .sum(axis=1)
        .astype(np.float16)
    )
    got = measured.dsp[tag].reshape(want.shape)
    worst = np.abs(got.astype(np.float32) - want.astype(np.float32)).max()
    step = np.spacing(np.abs(want).max().astype(np.float16))
    assert worst <= step, f"{tag}: worst {worst:.3e} past one fp16 step ({step:.3e})"


def test_the_mask_operand_is_what_moves_the_answer(measured):
    """A control on the operand, not on the verdict.

    Row 1 of this mask is all zeros and row 0 is all ones, so the fused kernel's
    row 1 has to be exactly zero and its row 0 has to be the plain row sum. A
    kernel reading the mask from the wrong place, or reading a product nobody
    wrote, moves both.
    """
    a, _mask = _inputs(31)
    got = measured.dsp["A"].reshape(_OUTSIDE, _ADMITTED[0])
    assert np.all(got[1] == 0.0), (
        f"the all-zero mask row answered {got[1][:6]} rather than zero"
    )
    plain_row = a[0].astype(np.float32).sum(axis=0).astype(np.float16)
    assert np.array_equal(_bits(got[0]), _bits(plain_row)), (
        "the all-one mask row is not the plain row sum: worst "
        f"{np.abs(got[0].astype(np.float32) - plain_row.astype(np.float32)).max():.3e}"
    )


@pytest.mark.parametrize("tag, inside", [("E", 16), ("F", 31), ("H", 64)])
def test_the_plain_reduction_is_an_fp32_sum_on_two_of_its_three_walks(
    measured, tag, inside
):
    """The measurement the admitted width rests on.

    The plain kernel has three walks, and only the middle one is not an fp32
    sum: below 32 it is the scalar path, at 64 and above the HVX path, and both
    are bit-equal to an fp32 accumulation rounded once. The widths asserted here
    are the ones where that holds, and they are what makes the boundary a
    measurement rather than a reading.
    """
    assert _args(measured, tag)[0].shape[-1] == inside
    x, _mask = _inputs(inside, seed=25)
    want = _fp32_sum(x, inside)
    got = measured.dsp[tag]
    assert np.array_equal(_bits(got), _bits(want)), (
        f"inside {inside}: the DSP is not the fp32 sum at "
        f"{int((_bits(got) != _bits(want)).sum())} of {got.size} outputs, worst "
        f"{np.abs(got.astype(np.float32) - want.astype(np.float32)).max():.3e}"
    )


def test_the_plain_reductions_small_vector_walk_is_not_an_fp32_sum(measured):
    """Where the boundary is, and which side of it is the less accurate one.

    The one width of the four this file runs the plain kernel at whose walk is
    not an fp32 sum, and the reason the gate stops at 32 rather than at the
    masked kernel's own 64. At 40 elements a row the kernel takes its
    small-vector walk, and the DSP's answer there is not the fp32 sum of the row
    at 100-odd outputs, by up to about 1.4 fp16 ULP of the largest output -- a
    rounding-level difference and not a wrong row, which the two bounds
    separate. Between 32 and 64 the merged command is therefore the *more*
    accurate of the two, so there the merge would be a numerical change rather
    than a merge. The interpreter models this walk as an fp32 sum like the other
    two, so it is also the one place its model of DSP_OP_REDUCTION is
    optimistic: no case in this file compares the two at this width.
    """
    inside = 40
    assert _args(measured, "G")[0].shape[-1] == inside
    x, _mask = _inputs(inside, seed=25)
    want = _fp32_sum(x, inside)
    got = measured.dsp["G"]
    moved = int((_bits(got) != _bits(want)).sum())
    worst = np.abs(got.astype(np.float32) - want.astype(np.float32)).max()
    step = np.spacing(np.abs(want).max().astype(np.float16))
    assert moved, (
        "the small-vector walk came out as an fp32 sum after all, which is what "
        "the small-vector walk is not what the gate's comment says it is -- if "
        "the kernel changed, the admitted width can be re-measured wider"
    )
    assert worst <= 1.5 * step, (
        f"the small-vector walk is off by {worst:.3e}, more than 1.5 fp16 steps "
        f"({step:.3e}): that is a wrong row rather than a lost last bit"
    )
