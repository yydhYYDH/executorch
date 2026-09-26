# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What gt and lt delegate as, and which of the three gates each of them stops at.

The route is the DSP's own GREATER(9) and LESS(10) op types followed by a
one-byte SELECT, so the whole cost is host side and the whole risk is host side:
two commands with positional, unchecked parameters, a result one byte wide, and
three gates between the table row and the export. A test that only says "gt
delegates" passes on a blob that computes something else entirely, so the
assertions here are on the command stream, the constants and the gates.

Everything in this file is the HOST tier: the lowering, the partitioner and the
blob. Nothing here executes a kernel, because nothing in this tree can -- the
numpy model in blob_interpreter raises on binary op 9 rather than guessing, and
the arithmetic on the DSP is test_comparisons_on_sim.py.
"""

import numpy as np
import torch

from blob_interpreter import read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_backend import SUPPORTED_TARGETS
from executorch.backends.hexagon.hexagon_ops import (
    COMPARISON_TARGETS,
    EMITTERS,
    LESS_THAN,
    GREATER_THAN,
    operand_dtypes_are_readable,
    result_dtype_is_emittable,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

_BINARY = 19
_SELECT = 26
_WEIGHTS = 0
_OUTPUTS = 2

#: (a, b) per element, chosen for the order relation rather than the value: the
#: two signed zeros, a NaN against itself, each infinity against itself, and the
#: fp16 extremes. Indices 1, 4, 5 and 6 are the ones a subtract-based route
#: would get wrong, which is why they are here and not random data.
_A = [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), float("-inf"),
      2.0, -2.0, 0.5, -0.5, 3.0, 1e-4, -1e-4, 65504.0, -65504.0]
_B = [0.0, 0.0, 1.0, 1.0, float("nan"), float("inf"), float("-inf"),
      1.0, -1.0, 0.25, -0.25, 3.0, 1e-4, 1e-4, -65504.0, 65504.0]


class _Compare(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, a, b):
        return self.op(a, b)


def _program(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegate(program):
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"expected one delegate, got {len(calls)}"
    return program.graph_module.get_submodule(calls[0].args[0].target)


def _blob_of(op, a, b):
    """The one delegate's blob and command list for a comparison of two tensors."""
    program = _program(_Compare(op), (a, b))
    lowered = _delegate(program)
    blob = bytes(lowered._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def _corners():
    a = torch.tensor([_A], dtype=torch.float16)
    b = torch.tensor([_B], dtype=torch.float16)
    return a, b


def _tensor_overload(name):
    """The two-tensor overload of a comparison op.

    The edge dialect groups these under an overload packet, so the dotted name
    is two getattrs rather than one; writing it as a single getattr is how a test
    ends up asserting on an AttributeError it never noticed.
    """
    return getattr(getattr(exir_ops.edge.aten, name), "Tensor")


def test_the_table_row_is_the_supported_target_row_too():
    """One change, not a pair to keep in step.

    SUPPORTED_TARGETS is EMITTERS rather than a copy of it, so a row added to one
    is a row the partitioner sees. That is the claim that makes "admit the
    comparison" a single edit, and it is asserted here rather than assumed,
    because a future refactor to a copy would turn this branch's one change into
    two and every test below would still pass.
    """
    assert SUPPORTED_TARGETS is EMITTERS
    for target in (GREATER_THAN, LESS_THAN):
        assert target in EMITTERS
        assert target in SUPPORTED_TARGETS
    assert COMPARISON_TARGETS == frozenset({GREATER_THAN, LESS_THAN})


def test_the_four_comparisons_left_out_are_absent_and_the_two_kept_are_present():
    """The stop-at-gt-and-lt boundary, with a control that is present.

    eq, ne, ge and le are out for arithmetic reasons rather than spelling ones,
    and a test that only asserted their absence would also be satisfied by a
    branch that had never tried. The control is the same table carrying the two
    that are in, so an EMITTERS that had somehow lost every comparison fails here
    rather than passing.
    """
    for name in ("eq", "ne", "ge", "le"):
        target = _tensor_overload(name)
        assert target not in EMITTERS, name
        assert target not in COMPARISON_TARGETS, name
    # The control, without which the four assertions above are satisfied by a
    # table that lost every comparison rather than by one that kept two.
    assert GREATER_THAN in EMITTERS and LESS_THAN in EMITTERS
    assert len(EMITTERS) >= 2


def test_a_comparison_is_two_commands_with_the_parameters_the_measurement_used():
    """The whole route, in the order the two commands run.

    The numbers are the ones the DSP was measured with, so a reader does not have
    to take that from another report: the binary command names GREATER(9) and
    writes two bytes an element, the select reads that condition at two bytes and
    writes one byte an element from two single-element sources, and the
    broadcast tail is the same 25-int descriptor the other binary rows carry.
    """
    a, b = _corners()
    for op, op_type in ((torch.gt, 9), (torch.lt, 10)):
        _, commands = _blob_of(op, a, b)
        assert [c.type for c in commands] == [_BINARY, _SELECT], op
        compare, select = commands
        n = a.numel()
        assert list(compare.params[:8]) == [n, n, n, op_type, 2, 2, 0, 0], op
        assert compare.params[8] == 2, "the broadcast descriptor's rank"
        assert list(compare.params[9:11]) == [1, n], "the output extents"
        assert list(select.params) == [n, n, 1, 1, 1, 2, 0, 0], op
        # The condition of the select is the activation the binary command wrote,
        # so the two commands are wired to each other rather than to two arenas.
        assert select.inputs[0].space == compare.outputs[0].space
        assert select.inputs[0].index == compare.outputs[0].index
        assert select.inputs[0].size == 2 * n, "the flag is two bytes an element"


def test_the_selects_two_sources_are_the_bytes_one_and_zero():
    """The constants are read out of the blob, not asserted about the emitter.

    bytes_for(n, torch.bool) is n, so a one-byte constant needs no schema change,
    and a bool result is bytes_for(n, torch.bool) as well. If the emitter had
    materialized the sources as fp16 -- which ctx.scalar does by default -- the
    command would still declare bytes 1 and the kernel would copy the low half of
    each half-float, which is a 0x3C and not a 0x01. The control is the same
    tensor one byte wide read as two.
    """
    a, b = _corners()
    blob, commands = _blob_of(torch.gt, a, b)
    select = commands[1]
    assert [ref.space for ref in select.inputs[1:]] == [_WEIGHTS, _WEIGHTS]
    assert [ref.size for ref in select.inputs[1:]] == [1, 1]
    section = _weight_section(blob)
    for ref, want in zip(select.inputs[1:], (0x01, 0x00)):
        got = section[ref.offset: ref.offset + ref.size]
        assert bytes(got) == bytes([want]), (ref, bytes(got).hex())
    # A source read at the two-byte width a half-float would use is a different
    # number, which is what makes the byte above a fact and not a tautology: fp16
    # 1.0 is 0x3C00, so a selector that had taken ctx.scalar's default would read
    # 0x00 for the "true" source and answer every element false.
    first = select.inputs[1]
    # Read at the two-byte width a half-float would use, the pair of bytes is
    # 0x0001 and fp16 1.0 is 0x3C00. A selector that had taken ctx.scalar's fp16
    # default would carry 0x3C00 here and copy 0x3C to a one-byte destination,
    # which is neither 1 nor 0.
    assert np.frombuffer(
        section[first.offset: first.offset + 2], dtype=np.uint16
    )[0] == 0x0001
    assert np.frombuffer(b"\x00\x3c", dtype=np.float16)[0] == 1.0
    # Two one-byte sources and the alignment the builder puts between them: the
    # section is longer than two bytes because each weight starts on a 128-byte
    # boundary, which costs nothing and would be a silent overlap if it did not.
    assert len(section) > 2


def _weight_section(blob):
    """The bytes of the blob's weight section, at the offsets the blob records.

    A weight's offset in the blob is what the runtime uses, so reading at it is
    the runtime's own addressing rather than a replay of the builder's packing
    order -- a replay would agree with a builder that had put the bytes in the
    wrong place only by accident.
    """
    from executorch.backends.hexagon.serialization import blob as schema

    header, _ = read_blob(blob)
    start = schema.HEADER_SIZE + header.n_ops * schema.OP_SIZE
    return blob[start: start + header.weights_bytes]


def test_a_bool_result_is_one_byte_per_element_at_every_length_measured():
    """The output slot, at the four lengths the width was measured at.

    n bytes for n bools is the whole reason the select's one-byte arm is usable
    at all, and it is a property of bytes_for rather than of this emitter, so it
    is checked at the lengths the claim was made for and at one that is not a
    multiple of anything.
    """
    from executorch.backends.hexagon.hexagon_backend import bytes_for

    for n in (1, 4, 64, 4096):
        assert bytes_for(n, torch.bool) == n
    a, b = _corners()
    for n in (1, 4, 64, 4096):
        left = torch.zeros(1, n, dtype=torch.float16)
        right = torch.full((1, n), -1.0, dtype=torch.float16)
        _, commands = _blob_of(torch.gt, left, right)
        out = commands[1].outputs[0]
        assert out.space == _OUTPUTS, (n, out)
        assert out.size == n, (n, out.size)


def test_the_width_gate_admits_a_comparison_and_refuses_every_other_bool():
    """Gate two, on the node it reads, with a control at the same geometry.

    _dtype_of reads node.meta["val"], which is the node's own OUTPUT, so this
    gate is about the result and not about the operands. The mirror is the
    crossed case: an fp16 result with a bool operand passes this gate and fails
    operand_dtypes_are_readable, which is what stops the one-line reading of the
    operand rule as the thing holding a comparison.
    """
    def node_with(target, out_dtype, arg_dtype=torch.float16, size=(4, 8)):
        graph = torch.fx.Graph()
        lhs = graph.placeholder("x")
        lhs.meta["val"] = torch.empty(*size, dtype=arg_dtype)
        rhs = graph.placeholder("y")
        rhs.meta["val"] = torch.empty(*size, dtype=arg_dtype)
        node = graph.call_function(target, args=(lhs, rhs))
        node.meta["val"] = torch.empty(*size, dtype=out_dtype)
        return node

    support = HexagonOperatorSupport()
    for target in COMPARISON_TARGETS:
        assert support.is_node_supported({}, node_with(target, torch.bool))
    # The control: the same object at the same geometry with an fp16 result is
    # taken, so the two assertions above are about the bool and not about an
    # object that refuses everything.
    assert support.is_node_supported(
        {}, node_with(exir_ops.edge.aten.add.Tensor, torch.float16)
    )
    for name in ("eq", "ne", "ge", "le"):
        target = _tensor_overload(name)
        assert not support.is_node_supported({}, node_with(target, torch.bool)), name
    # A bool *operand* is a different rule and still refuses: the kernels read two
    # bytes an element and a flag is one.
    assert not support.is_node_supported(
        {}, node_with(exir_ops.edge.aten.add.Tensor, torch.float16, torch.bool)
    )
    # And a comparison's operands are readable, which is why the operand rule
    # needed no change for this branch.
    assert operand_dtypes_are_readable(
        node_with(GREATER_THAN, torch.bool, torch.float16)
    )


def test_the_operand_rule_crosses_the_width_rather_than_matching_it():
    """The two gates are about two different tensors, and the mirror proves it.

    A bool result with fp16 operands fails the width gate and passes the operand
    rule; an fp16 result with bool operands does the opposite. Neither statement
    is load bearing on its own, because a gate that refused everything would
    satisfy the first, so the positive control is a fp16/fp16 node passing both.
    """
    def node_with(out_dtype, arg_dtype):
        graph = torch.fx.Graph()
        lhs = graph.placeholder("x")
        lhs.meta["val"] = torch.empty(4, 8, dtype=arg_dtype)
        rhs = graph.placeholder("y")
        rhs.meta["val"] = torch.empty(4, 8, dtype=arg_dtype)
        node = graph.call_function(exir_ops.edge.aten.add.Tensor, args=(lhs, rhs))
        node.meta["val"] = torch.empty(4, 8, dtype=out_dtype)
        return node

    assert operand_dtypes_are_readable(node_with(torch.bool, torch.float16))
    assert not operand_dtypes_are_readable(node_with(torch.float16, torch.bool))
    assert operand_dtypes_are_readable(node_with(torch.float16, torch.float16))


def test_the_two_gates_read_one_predicate_so_they_cannot_disagree():
    """The drift guard for the three-gate interaction.

    _require_arena_dtype RAISES on a width the arena does not hold, so a gate
    relaxed on its own turns a working export into a RuntimeError rather than a
    delegate. One predicate behind both is what makes that unreachable, and this
    walks the whole cross product of (dtype, target) so a later edit that gives
    the two gates separate rules is caught here rather than by an export.

    The control is in the same loop: for every (dtype, target) the support
    object refuses, the predicate must also say no. A predicate that said yes
    everywhere would pass the first half of this test and fail the second.
    """
    def node_with(target, dtype):
        graph = torch.fx.Graph()
        lhs = graph.placeholder("x")
        lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        rhs = graph.placeholder("y")
        rhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
        node = graph.call_function(target, args=(lhs, rhs))
        node.meta["val"] = torch.empty(4, 8, dtype=dtype)
        return node

    support = HexagonOperatorSupport()
    # Targets whose only gate past this one is a shape or a weight rule a node
    # built here satisfies, so an "emittable but not admitted" failure would be
    # the width gate's. Two are deliberately absent. `where` gates on its
    # condition's dtype as well, so a fp16 condition is refused for a reason this
    # predicate has nothing to say about, and `mm` gates on the contraction
    # dims, which two (4, 8) operands do not satisfy. Either would make this loop
    # assert a falsehood rather than test anything.
    targets = [GREATER_THAN, LESS_THAN, exir_ops.edge.aten.add.Tensor,
               exir_ops.edge.aten.sub.Tensor, exir_ops.edge.aten.mul.Tensor,
               exir_ops.edge.aten.relu.default, exir_ops.edge.aten.abs.default]
    dtypes = [torch.float16, torch.float32, torch.bool, torch.int64,
              torch.int32, torch.float64]
    for target in targets:
        for dtype in dtypes:
            node = node_with(target, dtype)
            emitted = result_dtype_is_emittable(dtype, target)
            admitted = support.is_node_supported({}, node)
            if emitted:
                assert admitted, (target, dtype, "emittable but not delegated")
            else:
                assert not admitted, (target, dtype, "delegated but not emittable")
    # The control the loop cannot supply on its own: a predicate that answered
    # True everywhere would pass every "emittable but not delegated" line and
    # fail only here, so this is the assertion that makes the rest mean something.
    assert not result_dtype_is_emittable(torch.bool, exir_ops.edge.aten.add.Tensor)
    assert result_dtype_is_emittable(torch.bool, GREATER_THAN)
    assert result_dtype_is_emittable(torch.float16, exir_ops.edge.aten.add.Tensor)


def test_a_bool_on_a_target_that_is_not_a_comparison_still_raises():
    """Gate three, as a backstop rather than as a path.

    The measurement that came with this branch was that a table row plus a
    relaxed width gate and an untouched _require_arena_dtype turns an export that
    works into RuntimeError: must be fp16 or fp32, got torch.bool. Both gates now
    read one predicate, so the emitter's raise is unreachable for a width the
    partitioner already refused -- and the message is kept here because a bool
    that reaches an emitter by any other route is a failed export rather than a
    wrong number, which is the better of the two.
    """
    graph = torch.fx.Graph()
    lhs = graph.placeholder("x")
    lhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
    rhs = graph.placeholder("y")
    rhs.meta["val"] = torch.empty(4, 8, dtype=torch.float16)
    node = graph.call_function(exir_ops.edge.aten.add.Tensor, args=(lhs, rhs))
    node.meta["val"] = torch.empty(4, 8, dtype=torch.bool)
    with np.testing.assert_raises(RuntimeError) as caught:
        hexagon_ops._require_arena_dtype(node, "add input")
    assert "must be fp16 or fp32, got torch.bool" in str(caught.exception)


def test_a_comparison_against_a_scalar_stays_portable():
    """The overload boundary, with the tensor form as its own control.

    F.prelu decomposes to view_copy + gt + mul + where, and that gt is the Scalar
    form, so widening to the Scalar overloads would be what moves the prelu
    decomposition rather than a separate convenience. The control is the same
    operands one line apart: a > b delegates, a > 0.5 does not.
    """
    a = torch.randn(2, 8, dtype=torch.float16)
    b = torch.randn(2, 8, dtype=torch.float16)
    tensor_form = _program(_Compare(torch.gt), (a, b))
    assert _delegate(tensor_form) is not None

    class Scalar(torch.nn.Module):
        def forward(self, x):
            return x > 0.5

    scalar_form = _program(Scalar(), (a,))
    calls = [
        node
        for node in scalar_form.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert calls == [], "the Scalar overload is not in EMITTERS, so nothing delegates"
    outer = {
        str(node.target)
        for node in scalar_form.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert any("gt" in target for target in outer), outer


def test_a_rank_nine_operand_stays_portable_rather_than_failing_the_export():
    """The fifth gate, which the route needs and the brief did not list.

    gt and lt take the binary broadcast descriptor, whose stride tables are
    eight entries wide, so a rank-nine operand has no representation. Leaving
    them out of BINARY_TARGETS would not have refused the node: the emitter would
    have raised, and a raise while emitting fails the whole export rather than
    falling back. The control is rank eight, which must delegate -- a gate that
    refused everything would make the first assertion vacuous.
    """
    rank8 = [torch.randn(1, 1, 1, 1, 1, 1, 1, 4, dtype=torch.float16),
             torch.randn(1, 1, 1, 1, 1, 1, 1, 4, dtype=torch.float16)]
    _, commands = _blob_of(torch.gt, *rank8)
    assert [c.type for c in commands] == [_BINARY, _SELECT]
    assert GREATER_THAN in hexagon_ops.BINARY_TARGETS
    assert LESS_THAN in hexagon_ops.BINARY_TARGETS

    rank9 = [torch.randn(1, 1, 1, 1, 1, 1, 1, 1, 4, dtype=torch.float16),
             torch.randn(1, 1, 1, 1, 1, 1, 1, 1, 4, dtype=torch.float16)]
    program = _program(_Compare(torch.gt), tuple(rank9))
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert calls == [], "a rank-nine broadcast is not walkable and must fall back"


def test_a_broadcast_operand_reaches_the_kernel_with_the_stride_the_gate_saw():
    """The descriptor is the binary one, so a broadcast actually broadcasts.

    A single-row right-hand side is the shape the stride table exists for: the
    compare arm walks it with the same htp_ops_binary_broadcast_offset every
    other binary op uses. Reading the strides out of the blob is the only way to
    tell a broadcast descriptor from an elementwise one written by hand.
    """
    a = torch.randn(3, 5, dtype=torch.float16)
    b = torch.randn(1, 5, dtype=torch.float16)
    _, commands = _blob_of(torch.gt, a, b)
    compare = commands[0]
    assert list(compare.params[9:11]) == [3, 5], "the output extents"
    # params[8] is the rank, [9:17] the eight output extents, [17:25] and [25:33]
    # the two stride tables. A (1, 5) right-aligned against (3, 5) repeats the
    # outer axis, which is stride 0 there and stride 1 on the inner one.
    assert list(compare.params[17:19]) == [5, 1], "the left operand's strides"
    assert list(compare.params[25:27]) == [0, 1], "the broadcast operand's strides"
    assert compare.params[8] == 2 and compare.params[16:17] == [0]
    assert list(commands[1].params[:2]) == [15, 15]


def test_the_comparison_reaches_the_delegate_where_a_where_reads_its_result():
    """The island this closes, measured rather than asserted from a decomposition.

    torch.where(a > b, x, y) is the shape the comparison gap was always about.
    Before this branch the where delegated and the comparison did not, so the
    bool entered the delegate as an argument; now both are inside it and the
    select's condition is the one-byte activation the comparison wrote. The
    control is the same graph with a bool condition passed in, which has to
    delegate as a single select.
    """
    a = torch.randn(4, 8, dtype=torch.float16)
    b = torch.randn(4, 8, dtype=torch.float16)
    x = torch.randn(4, 8, dtype=torch.float16)
    y = torch.randn(4, 8, dtype=torch.float16)

    class Island(torch.nn.Module):
        def forward(self, a, b, x, y):
            return torch.where(a > b, x, y)

    program = _program(Island(), (a, b, x, y))
    lowered = _delegate(program)
    inner = [
        str(node.target)
        for node in lowered.original_module.graph_module.graph.nodes
    ]
    assert any("gt.Tensor" in target for target in inner), inner
    _, commands = read_blob(bytes(lowered._processed_bytes))
    assert [c.type for c in commands] == [_BINARY, _SELECT, _SELECT], [
        c.type for c in commands
    ]
    # The where's condition is the one-byte activation, not a method input.
    assert commands[2].inputs[0].space == _OUTPUTS or commands[2].inputs[0].space == 3
    assert commands[2].inputs[0].size == a.numel()

    class Given(torch.nn.Module):
        def forward(self, cond, x, y):
            return torch.where(cond, x, y)

    control = _program(Given(), (a > b, x, y))
    _, control_commands = read_blob(bytes(_delegate(control)._processed_bytes))
    assert [c.type for c in control_commands] == [_SELECT], control_commands
