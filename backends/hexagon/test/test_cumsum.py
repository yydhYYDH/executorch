# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""A prefix scan, as the two commands the DSP already has.

The DSP has no scan. htp_command.h declares no cumulative command and the
dispatch in execute_command.cc has no case for one; what it has is
DSP_OP_BATCH_MATMUL, so a cumsum along the last axis is one product against
a constant mask, and a streaming one adds its carry with the binary
elementwise the graph was going to emit anyway.

These tests pin, in the order the questions come: the rewrite, the plan as
counted out of read_blob, the scan order, the accumulator, the boundary
brackets, the error ladder, and a negative control that moves the carry
without moving the plan.
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
import test_blob_on_sim as blob_sim  # noqa: E402
from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.cumsum import (  # noqa: E402
    FuseCumsumPass,
    MAX_SCAN_LENGTH,
    SCAN_ALIGNMENT,
    cumsum_is_emittable,
    fuse_cumsum,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.export import export  # noqa: E402

#: DSP_OP_BATCH_MATMUL, DSP_OP_BINARY_ELEMENTWISE, DSP_OP_RASTER_BLIT.
_BATCH_MATMUL = 38
_BINARY = 19
_BLIT = 3

#: The binary op type for HTP_OPS_BINARY_ADD, which is what a carry is.
_ADD = 1

#: hmxFlags as the plan records it (params[26]): the magic alone, and the
#: magic with the bit that says the weight is already in the tile order.
_PLAN_MAGIC = 0x484D58
_PLAN_PREPACKED = 0x484D59

#: The repository's fp16 product allowance, as test_matmul_staging_on_sim.py
#: states it: these are reductions, so the comparison is a tolerance.
TOLERANCE = 1e-2


class _StreamingCumsum(torch.nn.Module):
    """One chunk of a stream: scan it and add the previous total.

    The carry is a method argument because the graph holds no state between
    execute() calls, so the caller threads it and the DSP never keeps it.
    """

    def forward(self, x, carry):
        return torch.cumsum(x, dim=-1) + carry


class _StreamingCumsumWithNext(torch.nn.Module):
    """The same, plus the next carry as the caller reads it.

    out[..., -1] is a select_copy the graph already had; this workstream did
    not add it.
    """

    def forward(self, x, carry):
        scanned = torch.cumsum(x, dim=-1) + carry
        return scanned, scanned[..., -1]


class _PlainCumsum(torch.nn.Module):
    """The first chunk of a stream: a scan with nothing carried in yet."""

    def forward(self, x):
        return torch.cumsum(x, dim=-1)


def _edge(model, inputs):
    return to_edge(
        export(model, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _program(model, inputs, passes=()):
    return to_edge_transform_and_lower(
        export(model, inputs),
        transform_passes=list(passes),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _targets(program):
    return [
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def _blob(program):
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the scan did not reach one delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    blob = bytes(lowered._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def _run(blob, inputs, outputs=1):
    raw = execute(blob, [x.numpy() for x in inputs])
    return [
        np.frombuffer(bytes(raw[index]), dtype=np.float16)
        for index in range(outputs)
    ]


def _random(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=generator) * 2 - 1).half()


def _reference(x, carry=None):
    """The scan in fp64: the reference both sides are measured against.

    torch's fp16 cumsum rounds every step, so it is a second opinion and not
    the reference for a kernel that accumulates in fp32.
    """
    scanned = torch.cumsum(x.double(), dim=-1)
    return scanned if carry is None else scanned + carry.double()


def _step(values):
    """The fp16 step at the size of these values.

    A kernel that accumulates in fp32 and narrows once may spend this much on
    the store and no more, however long the scan is.
    """
    largest = float(np.abs(np.asarray(values, dtype=np.float64)).max())
    return float(np.spacing(np.float16(largest)))


def _fp16_sequential(x):
    """The scan that rounds to fp16 at every step, which is the one this is not.

    It is here so the difference between narrowing once and narrowing per step
    is a measured number rather than an assumption about either.
    """
    acc = torch.zeros(x.shape[0], dtype=torch.float32)
    out = []
    for index in range(x.shape[1]):
        acc = (acc + x[:, index].float()).half().float()
        out.append(acc.clone())
    return torch.stack(out, -1)


# --- the rewrite ------------------------------------------------------------


def test_the_pass_rewrites_the_carrying_pair_into_one_node():
    """The streaming pattern, and that a second pass changes nothing."""
    x = _random((4, 64), 3)
    carry = _random((4, 1), 4)
    program = _edge(_StreamingCumsum(), (x, carry))
    targets = _targets(program)
    assert any("aten.cumsum.default" in t for t in targets), targets
    assert any("aten.add.Tensor" in t for t in targets), targets
    assert fuse_cumsum(program.graph_module) == 1
    targets = _targets(program)
    assert any("et_hexagon.cumsum.default" in t for t in targets), targets
    assert not any("aten.cumsum.default" in t for t in targets), targets
    assert not any("aten.add.Tensor" in t for t in targets), targets
    assert fuse_cumsum(program.graph_module) == 0


def test_a_bare_scan_is_the_first_chunk_and_is_fused_too():
    """A scan nobody carries into is the computation with a zero carry.

    The carry arrives as an argument when there is one to arrive.
    """
    x = _random((4, 64), 5)
    program = _edge(_PlainCumsum(), (x,))
    assert fuse_cumsum(program.graph_module) == 1
    targets = _targets(program)
    assert any("et_hexagon.cumsum.default" in t for t in targets), targets
    assert not any("aten.cumsum.default" in t for t in targets), targets


def test_a_scan_over_another_axis_is_left_for_a_portable_kernel():
    """The mask scans the trailing axis, which is the contiguous one.

    A cumsum over dim 0 is a transpose away from this kernel and the pass
    does not own that transpose.
    """

    class _FirstAxis(torch.nn.Module):
        def forward(self, x):
            return torch.cumsum(x, dim=0)

    x = _random((64, 4), 6)
    program = _edge(_FirstAxis(), (x,))
    assert fuse_cumsum(program.graph_module) == 0
    assert any("aten.cumsum.default" in t for t in _targets(program))


def test_a_scan_read_twice_keeps_the_bare_one():
    """The fused op serves the add's readers; the scan stays for its other one.

    that wanted the scan itself, the same split add_relu makes.
    """

    class _Twice(torch.nn.Module):
        def forward(self, x, carry):
            scanned = torch.cumsum(x, dim=-1)
            return scanned + carry, scanned

    x = _random((4, 64), 7)
    carry = _random((4, 1), 8)
    program = _edge(_Twice(), (x, carry))
    assert fuse_cumsum(program.graph_module) == 1
    targets = _targets(program)
    assert sum("et_hexagon.cumsum.default" in t for t in targets) == 1, targets
    assert sum("aten.cumsum.default" in t for t in targets) == 1, targets


# --- the plan ---------------------------------------------------------------


def test_the_scan_is_a_batched_matmul_and_the_carry_a_binary_add():
    """The real command stream, counted out of the blob.

    The third command is the caller's out[..., -1] slice, a select_copy the
    graph already had, emitted as a blit. Nothing this workstream added.
    """
    x = _random((8, 64), 9)
    carry = _random((8, 1), 10)
    _, commands = _blob(
        _program(
            _StreamingCumsumWithNext(),
            (x, carry),
            passes=[FuseCumsumPass()],
        )
    )
    assert [c.type for c in commands] == [_BATCH_MATMUL, _BINARY, _BLIT]
    matmul, add = commands[0], commands[1]
    assert matmul.params[2:5] == [8, 64, 64], "the product is not (rows, L) @ (L, L)"
    assert add.params[3] == _ADD, "the carry is not an add"
    assert add.params[1] == 8 * 64, "the add does not read the whole scan"
    assert add.params[2] == 8, "the add does not see the carry as [rows, 1]"


def test_the_carry_broadcasts_over_the_scan_from_its_stride_tail():
    """A [rows, 1] carry reaches every column through a zero stride.

    the axis it is one element wide on, which is the whole mechanism.
    """
    x = _random((8, 64), 11)
    carry = _random((8, 1), 12)
    _, commands = _blob(
        _program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()])
    )
    add = commands[1]
    assert add.params[8] == 2, "the tail does not describe a two-dimensional add"
    assert add.params[9:11] == [8, 64], "the output extents are not (rows, L)"
    assert add.params[17:19] == [64, 1], "the scan operand is not read in order"
    assert add.params[25:27] == [1, 0], "the carry does not broadcast its column"


def test_the_mask_is_stored_in_the_units_tile_order():
    """A host constant, so the plan says the weight is already tiled.

    the unit reads, and the DSP streams it out of the weights section
    instead of rearranging a tile at a time per inference.
    """
    x = _random((16, 64), 13)
    carry = _random((16, 1), 14)
    assert 16 * 64 * 64 >= 32768, "this shape is below the general kernel's gate"
    _, commands = _blob(
        _program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()])
    )
    assert commands[0].params[26] == _PLAN_PREPACKED, hex(commands[0].params[26])


def test_a_scan_small_enough_to_miss_the_hmx_gate_is_stored_row_major():
    """The packed tile order is readable only by the kernel the plan commits to.

    A region below E*K*N = 32768 has no such kernel, so the mask is left
    row-major there. Packing unconditionally would hand the fallbacks a
    matrix in the wrong order, and this is the case that would be wrong bytes.
    """
    x = _random((4, 64), 15)
    assert 4 * 64 * 64 < 32768
    _, commands = _blob(_program(_PlainCumsum(), (x,), passes=[FuseCumsumPass()]))
    assert commands[0].params[26] == _PLAN_MAGIC, hex(commands[0].params[26])


def test_a_scan_with_no_carry_is_one_command():
    """The first chunk of a stream is the product alone: nothing to add.

    add until the second chunk has a total to carry.
    """
    x = _random((8, 64), 16)
    _, commands = _blob(_program(_PlainCumsum(), (x,), passes=[FuseCumsumPass()]))
    assert [c.type for c in commands] == [_BATCH_MATMUL]


# --- the scan itself --------------------------------------------------------


def test_the_scan_is_the_prefix_in_order_and_the_carry_adds_to_it():
    """Order and boundary on the host model.

    Every element is sum_{i<=j} x[i] plus the carry, and the last column is
    the total the next chunk starts from. The reference is fp64 because the
    kernel accumulates in fp32, so anything narrower would be measuring the
    reference's own rounding rather than the kernel's.
    """
    x = _random((8, 64), 17)
    carry = _random((8, 1), 18)
    blob, _ = _blob(
        _program(
            _StreamingCumsumWithNext(),
            (x, carry),
            passes=[FuseCumsumPass()],
        )
    )
    scanned, last = _run(blob, (x, carry), outputs=2)
    expected = _reference(x, carry)
    difference = np.abs(
        scanned.reshape(8, 64).astype(np.float64) - expected.numpy()
    )
    assert difference.max() <= _step(expected.numpy()), (
        f"the scan is off by {difference.max()}"
    )
    tail = np.abs(last.astype(np.float64) - expected[:, -1].numpy())
    assert tail.max() <= _step(expected[:, -1].numpy()), (
        f"the carried total is off by {tail.max()}"
    )


def test_the_lower_triangle_would_be_the_reverse_scan():
    """Why the mask is the upper triangle, as a statement the numbers carry.

    x @ tril(ones) is the same plan and the same two commands, and it sums
    each suffix, so nothing about the command stream can separate the right
    mask from the wrong one. This is the same product with the mask built the
    other way, which is what a lowering with the orientation backwards gives.
    """
    x = _random((8, 64), 19)
    right = torch.cumsum(x, dim=-1)
    wrong = (x.float() @ torch.tril(torch.ones(64, 64))).half()
    assert not torch.allclose(wrong, right)
    assert (wrong.float() - right.float()).abs().max() > 1.0


def test_the_carry_enters_once_and_on_every_column():
    """The difference between a chunk that carries and one that does not.

    It is the carry, everywhere, and nowhere twice. A lowering that added the
    carry to the mask, or read it on the wrong axis, would be off by a
    multiple of it on some columns only.
    """
    x = _random((4, 64), 20)
    carry = _random((4, 1), 21)
    with_carry = _run(
        _blob(_program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()]))[0],
        (x, carry),
    )[0].reshape(4, 64)
    alone = _run(
        _blob(_program(_PlainCumsum(), (x,), passes=[FuseCumsumPass()]))[0],
        (x,),
    )[0].reshape(4, 64)
    difference = with_carry.astype(np.float64) - alone.astype(np.float64)
    expected = carry.double().numpy()
    assert np.abs(difference - expected).max() <= _step(alone.astype(np.float64))


def test_a_carry_that_broadcasts_over_several_axes_reaches_every_element():
    """The stride tail is the broadcast mechanism, and it is not one axis.

    one axis, so the carry the gate accepts for a 2-D chunk is the same
    computation for a 3-D one: every row and every column.
    """
    x = _random((2, 3, 64), 22)
    carry = _random((2, 3, 1), 23)
    assert cumsum_is_emittable(x, carry)
    blob, commands = _blob(
        _program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()])
    )
    assert commands[0].params[2:5] == [6, 64, 64], "the leading axes do not fold"
    (got,) = _run(blob, (x, carry))
    expected = _reference(x, carry)
    difference = np.abs(
        got.reshape(2, 3, 64).astype(np.float64) - expected.numpy()
    )
    assert difference.max() <= _step(expected.numpy())


# --- the boundary -----------------------------------------------------------


@pytest.mark.parametrize("length", [31, 32, 33, 63, 65, 96, 127])
def test_a_scan_length_the_phone_cannot_stage_is_refused(length):
    """The scan length is the product K, and the brackets are on both rules.

    63/64/65 and 31/32/33 bracket the 64 rule, not the 32 tile rule: 32 is a
    whole number of HMX tiles but not a whole number of staged rows, and a
    K that is not a multiple of 64 is staged at ceil(K/64)*64 while the
    phone's skel copies the rows straight through, so every row after the
    first in a tile is read from the wrong place.
    test_matmul_staging_on_sim.py measured that at K=40 as 12.05 of max
    difference and 7392 zero elements. The tree and the simulator carry the
    fix, the phone does not, so the gate refuses rather than hand the device
    a blob it would compute wrongly.
    """
    x = _random((4, length), 24)
    carry = _random((4, 1), 25)
    assert not cumsum_is_emittable(x, carry), f"{length} was accepted"
    program = _program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()])
    targets = _targets(program)
    assert not any("et_hexagon.cumsum.default" in t for t in targets), targets
    delegates = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    for call in delegates:
        lowered = program.graph_module.get_submodule(call.args[0].target)
        if lowered.backend_id != "HexagonBackend":
            continue
        _, commands = read_blob(bytes(lowered._processed_bytes))
        assert _BATCH_MATMUL not in [c.type for c in commands], (
            f"a refused {length}-frame scan put a product on the DSP"
        )

    # The positive control, in this same test and through this same pipeline:
    # the identical graph at 64 frames does put a product on the DSP. Without
    # it, "no product in the blob" is also what a partitioner that refuses
    # everything would produce, and the refusal above would prove nothing.
    control_x = _random((4, SCAN_ALIGNMENT), 24)
    _, control_commands = _blob(
        _program(_StreamingCumsum(), (control_x, carry), passes=[FuseCumsumPass()])
    )
    assert _BATCH_MATMUL in [c.type for c in control_commands], (
        "the pipeline refuses every geometry, so the refusals above are vacuous"
    )


@pytest.mark.parametrize("length", [SCAN_ALIGNMENT, 128, 192, 256])
def test_a_scan_length_the_phone_can_stage_is_accepted(length):
    """The other side of the bracket.

    Every multiple of 64 is one product, and the descriptor names the same
    three extents at all of them.
    """
    x = _random((4, length), 26)
    carry = _random((4, 1), 27)
    assert cumsum_is_emittable(x, carry)
    _, commands = _blob(
        _program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()])
    )
    assert commands[0].params[2:5] == [4, length, length]


def test_the_gate_refuses_what_the_mask_cannot_be_built_for():
    """The mask side is the scan length, so the gate is about the operands.

    operands: fp16 both, contiguous, at least two dimensions, a static extent
    no larger than the largest mask worth packing.
    """
    carry = _random((4, 1), 28)
    chunk = _random((4, 64), 29)
    assert not cumsum_is_emittable(chunk.float(), carry), "an fp32 chunk"
    assert not cumsum_is_emittable(
        _random((4, 128), 30).transpose(0, 1), carry
    ), "a strided chunk"
    assert not cumsum_is_emittable(_random((128,), 31), carry), "one dimension"
    assert not cumsum_is_emittable(
        _random((4, MAX_SCAN_LENGTH + SCAN_ALIGNMENT), 32), carry
    ), "past the largest mask"
    assert not cumsum_is_emittable(
        _random((4, 128), 33), _random((4, 1), 34).float()
    ), "an fp32 carry"


# --- the error ladder -------------------------------------------------------


@pytest.mark.parametrize(
    "length", [64, 128, 256, 512, MAX_SCAN_LENGTH], ids=lambda n: str(n)
)
def test_the_error_is_one_step_at_every_length(length):
    """The ladder, over the scan length itself, and what it says.

    Measured at L = 64, 128, 256, 512, 1024 on the host interpreter with a
    2-row fp16 chunk: the DSP answer and torch.cumsum are bitwise identical at
    every one of them, and the difference from an exact (fp64) scan stays
    inside one fp16 step of the output scale. So the error does not grow with
    the length the way a per-step-rounded scan would, and the reason is that
    the accumulator is fp32 and narrows once.

    The comparison that does grow is against `_fp16_sequential`, the scan that
    rounds at every step: there the gap is 0.0078 at L=64 and 0.1875 at L=1024,
    which is one step per frame rather than one step in total.

    This is the measurement that corrected the design note. torch.cumsum on an
    fp16 CPU tensor accumulates in fp32 the same way this kernel does, so the
    two agree exactly; the handoff's "torch rounds at every step" is not true of
    this torch. The allowance below is stated against the exact scan because
    that is the reference a device result has to be compared against too.
    """
    x = _random((2, length), 100 + length)
    blob, commands = _blob(_program(_PlainCumsum(), (x,), passes=[FuseCumsumPass()]))
    assert [c.type for c in commands] == [_BATCH_MATMUL]
    assert commands[0].params[2:5] == [2, length, length]
    (got,) = _run(blob, (x,))
    answer = got.reshape(2, length).astype(np.float64)
    exact = _reference(x).numpy()
    step = _step(exact)
    ours = float(np.abs(answer - exact).max())

    assert ours <= step, f"L={length}: off an exact scan by {ours}, over one step ({step})"
    assert np.array_equal(answer, torch.cumsum(x, dim=-1).double().numpy()), (
        f"L={length}: the answer is not torch.cumsum bit for bit"
    )
    per_step = float(np.abs(answer - _fp16_sequential(x).numpy()).max())
    assert step < per_step <= step * length, (
        f"L={length}: the per-step scan differs by {per_step}, expected a step a frame"
    )


def test_a_dropped_carry_is_caught_while_the_plan_stays_the_same():
    """The negative control, in the form the handoff calls the failure shape.

    this failure: the plan is byte-identical and only the numbers move.

    A one-step carry perturbation moves the answer by less than one output
    step, because the fp16 store absorbs it at many columns -- so a control
    that only perturbs by a step is weak. This one substitutes a zero carry,
    which is what a lowering that dropped the carry would compute, and the
    reference still expects the real one: the gap is the carry's own size,
    two orders of magnitude past the allowance, with the command types and
    params untouched.
    """
    x = _random((16, 64), 37)
    carry = _random((16, 1), 137)
    blob, commands = _blob(
        _program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()])
    )
    assert [c.type for c in commands] == [_BATCH_MATMUL, _BINARY]
    reference = _reference(x, carry).numpy()
    step = _step(reference)

    zero = torch.zeros_like(carry)
    answer = (
        np.frombuffer(bytes(execute(blob, [x.numpy(), zero.numpy()])[0]), dtype=np.float16)
        .astype(np.float64)
        .reshape(16, 64)
    )
    gap = float(np.abs(answer - reference).max())
    magnitude = float(np.abs(carry.double().numpy()).max())
    assert gap > 10 * step, f"a dropped carry moved the answer by only {gap}"
    assert abs(gap - magnitude) <= step, (
        f"the gap {gap} is not the carry's own size {magnitude}"
    )


# --- the negative control ---------------------------------------------------


def _stream_blob(length, seed):
    x = _random((16, length), seed)
    carry = _random((16, 1), seed + 100)
    return x, carry, _blob(
        _program(_StreamingCumsum(), (x, carry), passes=[FuseCumsumPass()])
    )[0]


def _sim_cases():
    """One fixture per length, each with a control that is the same graph.

    the same chunk, differing only in the carry, moved by one fp16 step.

    This is the negative control the handoff asks for: a carry-over between
    chunks perturbed without disturbing the pipeline. The plan is identical
    (same commands, same params, same length, same blob size), so a
    comparison that stayed green on both fixtures would be saying nothing
    about the carry, and the two answers have to differ by the perturbation.
    """
    out = []
    for tag, length, seed in (("CU64", 64, 37), ("CU128", 128, 38)):
        x, carry, blob = _stream_blob(length, seed)
        out.append(
            blob_sim._case(
                tag,
                _StreamingCumsum(),
                (x, carry),
                _reference(x, carry),
                kind="close",
                tolerance=TOLERANCE,
                blob=blob,
            )
        )
        bumped = torch.nextafter(
            carry, torch.full_like(carry, float("inf"))
        ).half()
        assert not torch.equal(bumped, carry), "the control did not move the carry"
        out.append(
            blob_sim._case(
                tag + "C",
                _StreamingCumsum(),
                (x, bumped),
                _reference(x, bumped),
                kind="close",
                tolerance=TOLERANCE,
                blob=blob,
            )
        )
    return out


@pytest.fixture(scope="module")
def cases():
    """The four sim fixtures: two lengths, each with its carry control."""
    return _sim_cases()


@pytest.fixture(scope="module")
def simulated(cases):
    """The runner built once over the vendored DSP sources, or a skip."""
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
        tuple(case.expected.shape)
    )


@pytest.mark.parametrize("tag", ["CU64", "CU128", "CU64C", "CU128C"])
def test_the_simulator_reads_the_prefix_and_the_carry(cases, simulated, tag):
    """hexagon-sim, which executes the command stream rather than naming it.

    it. The answer is the fp64 prefix within the repository's fp16 product
    allowance, for the case and for its carry control alike.
    """
    case = blob_sim._tagged(cases, tag)
    dsp = _dsp(case, simulated).astype(np.float64)
    expected = np.asarray(case.expected, dtype=np.float64)
    difference = float(np.abs(dsp - expected).max())
    assert difference <= TOLERANCE, f"{tag}: the sim is off by {difference}"


def test_the_carry_control_moves_the_answer_and_not_the_plan(cases, simulated):
    """The criterion this case is load-bearing on.

    One fp16 step of carry changes the answer, and the plan does not move at
    all. The same property is what notices a lowering that dropped the carry
    (the two answers would be equal) or added the wrong one (they would
    differ by more than the step), and the plan comparison is what says the
    perturbation was in the data and not in the pipeline.
    """
    for tag in ("CU64", "CU128"):
        plain = blob_sim._tagged(cases, tag)
        control = blob_sim._tagged(cases, tag + "C")
        assert len(plain.blob) == len(control.blob), f"{tag}: the blob moved"
        assert [c.type for c in plain.commands] == [c.type for c in control.commands]
        assert [c.params for c in plain.commands] == [c.params for c in control.commands]
        before = _dsp(plain, simulated).astype(np.float64)
        after = _dsp(control, simulated).astype(np.float64)
        moved = np.abs(after - before)
        assert moved.max() > 0.0, f"{tag}: the carry never reached the answer"
        # One step of carry is the most that can move an fp16 answer, and at
        # some columns it moves nothing: the sum rounds back to the same fp16
        # value. So the bound is the output's own step rather than the
        # perturbation's, and a lowering that dropped the carry would be the
        # one that answers identically everywhere.
        assert moved.max() <= 2 * _step(before), f"{tag}: the answer moved by {moved.max()}"
