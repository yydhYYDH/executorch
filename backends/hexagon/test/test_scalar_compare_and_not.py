# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A comparison against a literal, and a mask's negation: what each one is and what it refuses.

Two rows that cost no DSP C++ at all. `x > 3` is the command gt against a
tensor already emits, with the literal's stride table at zero -- the kernel has
an `in1Size == 1` arm for it and the emitter does not use that arm, so the
measurement is on the descriptor, not on the kernel's own spelling of the same
idea. `logical_not` is the select already in the blob, between the two one-byte
constants, with the condition read at the width its slot is declared at.

Everything in this file is the HOST tier: the lowering, the partitioner and the
blob. Nothing here executes a kernel. The arithmetic on the DSP is
test_scalar_compare_and_not_on_sim.py, and this file is the half that says the
command stream is the one that file measures.

Every refusal below is paired with a positive control at the same geometry,
because a gate that refused everything would satisfy each of them.
"""

import numpy as np
import torch

from blob_interpreter import read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_ops import (
    BOOL_RESULT_TARGETS,
    COMPARISON_TARGETS,
    EMITTERS,
    GREATER_THAN_SCALAR,
    LESS_THAN_SCALAR,
    LOGICAL_NOT,
    BITWISE_NOT,
    ONE_BYTE_CONDITION_TARGETS,
    comparison_operands_are_readable,
    mask_negation_is_emittable,
    operand_dtypes_are_readable,
    result_dtype_is_emittable,
    where_is_emittable,
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
_EDGE = exir_ops.edge.aten

F16 = torch.float16


class _Op(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _program(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _stream(program, index=0):
    """The command list of one delegate, which is what a claim is about."""
    lowered = program.graph_module.get_submodule(_delegates(program)[index].args[0].target)
    return read_blob(bytes(lowered._processed_bytes))[1]


def _outer_targets(program):
    return {
        str(node.target)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    }


def _verdicts(model, inputs):
    """Per-target support verdict on the unpartitioned edge graph.

    The lowered graph is the wrong place to read this: partitioning has already
    replaced the accepted nodes with one delegate, so a count read there is a
    count of what was left behind.
    """
    from executorch.backends.hexagon.partition.hexagon_partitioner import (
        _data_placeholders,
    )
    from executorch.exir import to_edge

    edge = to_edge(
        export(model, inputs), compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).exported_program()
    support = HexagonOperatorSupport(_data_placeholders(edge))
    return {
        node.target: support.is_node_supported(edge.graph_module, node)
        for node in edge.graph_module.graph.nodes
        if node.op == "call_function"
    }


def _x(*shape):
    return torch.randn(*shape, dtype=F16)


# --- the three rows are wired, and the sets they are read through ------------


def test_the_three_rows_are_in_the_table_the_partitioner_reads():
    """One object, so "admit the row" is a single edit rather than a pair."""
    from executorch.backends.hexagon.partition import hexagon_partitioner as partition

    assert partition.SUPPORTED_TARGETS is EMITTERS
    for target in (GREATER_THAN_SCALAR, LESS_THAN_SCALAR, LOGICAL_NOT, BITWISE_NOT):
        assert target in EMITTERS, target
    # The two width rules read derived sets rather than restated ones, so a
    # comparison, a negation and a select cannot drift into three lists.
    assert BOOL_RESULT_TARGETS == COMPARISON_TARGETS | {LOGICAL_NOT, BITWISE_NOT}
    assert ONE_BYTE_CONDITION_TARGETS == frozenset(
        {_EDGE.where.self, LOGICAL_NOT, BITWISE_NOT}
    )
    for target in (GREATER_THAN_SCALAR, LESS_THAN_SCALAR):
        assert target in COMPARISON_TARGETS, target
        assert target in hexagon_ops.BINARY_TARGETS, (
            "the literal reaches the binary descriptor, so the rank gate applies"
        )


# --- what each one emits ----------------------------------------------------


def test_a_literal_comparison_is_the_tensor_route_with_a_zero_stride():
    """Same two commands, and the difference is one stride table.

    params[2] is the literal's element count and the second stride table is all
    zeroes, which is what makes the kernel read element zero at every index.
    An emitter that had widened the literal to the output would produce the
    same answers and a different blob, so both are read out.
    """
    a = _x(4, 8)
    n = a.numel()
    for fn, op_type in ((lambda x: x > 3, 9), (lambda x: x < 0.5, 10)):
        commands = _stream(_program(_Op(fn), (a,)))
        assert [c.type for c in commands] == [_BINARY, _SELECT], fn
        compare, select = commands
        assert list(compare.params[:8]) == [n, n, 1, op_type, 2, 2, 0, 0], fn
        assert compare.params[8] == 2, "the descriptor's rank"
        assert list(compare.params[9:11]) == [4, 8], "the output extents"
        assert list(compare.params[17:19]) == [8, 1], "the tensor's strides"
        assert list(compare.params[25:33]) == [0] * 8, (
            "a literal is a one-element source read at a zero stride"
        )
        # The select is the comparison route's own: two-byte condition, one-byte
        # result, and the two one-byte constants between which it copies.
        assert list(select.params) == [n, n, 1, 1, 1, 2, 0, 0], fn
        assert select.inputs[0].index == compare.outputs[0].index


def test_the_literal_itself_is_a_two_byte_half_in_the_weight_section():
    """The threshold the kernel reads, read at the width the command declares.

    A literal reaches the arena as an fp16 constant, because the binary command
    reads two bytes an element. If it were written at one byte the descriptor
    would still read element zero and would get the low half of the half-float,
    which for 3.0 is not 3.0.
    """
    a = _x(4, 8)
    commands = _stream(_program(_Op(lambda x: x > 3), (a,)))
    rhs = commands[0].inputs[1]
    assert rhs.space == _WEIGHTS and rhs.size == 2, (rhs.space, rhs.size)
    section = _weight_section(bytes_of(_program(_Op(lambda x: x > 3), (a,))))
    assert np.frombuffer(section[rhs.offset: rhs.offset + 2], dtype=np.float16)[0] == 3.0


def _weight_section(blob):
    from executorch.backends.hexagon.serialization import blob as schema

    header, _ = read_blob(blob)
    start = schema.HEADER_SIZE + header.n_ops * schema.OP_SIZE
    return blob[start: start + header.weights_bytes]


def bytes_of(program):
    lowered = program.graph_module.get_submodule(_delegates(program)[0].args[0].target)
    return bytes(lowered._processed_bytes)


def test_a_negation_is_one_select_between_two_one_byte_constants():
    """`logical_not` as the emitter writes it, and the swap that makes it a negation.

    The select picks its first source when the condition is nonzero, so the
    first source has to be the zero. Reading the constants out of the blob is
    what makes that a fact: a select that carried 1 and 0 the other way round
    would still be one command of the right shape and would answer the identity.
    """
    commands = _stream(_program(_Op(lambda x: torch.logical_not(x > 0)), (_x(4, 8),)))
    assert [c.type for c in commands] == [_BINARY, _SELECT, _SELECT]
    negation = commands[2]
    n = 4 * 8
    assert list(negation.params) == [n, n, 1, 1, 1, 1, 0, 0], (
        "one byte throughout, including the condition, which is this op's input"
    )
    # The condition is the bool the comparison wrote, and the bool is one byte
    # per element -- which is the slot the blob declared for it.
    assert negation.inputs[0].index == commands[1].outputs[0].index
    assert negation.inputs[0].size == n, "a bool slot is one byte an element"
    blob = bytes_of(_program(_Op(lambda x: torch.logical_not(x > 0)), (_x(4, 8),)))
    section = _weight_section(blob)
    assert [ref.space for ref in negation.inputs[1:]] == [_WEIGHTS, _WEIGHTS]
    for ref, want in zip(negation.inputs[1:], (0x00, 0x01)):
        assert bytes(section[ref.offset: ref.offset + ref.size]) == bytes([want]), (
            ref,
            bytes(section[ref.offset: ref.offset + ref.size]).hex(),
        )


def test_a_negation_of_a_mask_from_outside_the_delegate_is_one_command():
    """The mask need not come from a comparison to reach the same command.

    A bool arriving as a graph input is the other producer of a condition --
    `where` already took that route -- and it is the case that shows the op is
    not a decomposition of the comparison but a select in its own right.
    """
    mask = torch.zeros(4, 8, dtype=torch.bool)
    commands = _stream(_program(_Op(torch.logical_not), (mask,)))
    assert [c.type for c in commands] == [_SELECT]
    assert list(commands[0].params) == [32, 32, 1, 1, 1, 1, 0, 0]


# --- what is accepted, and what is refused, one line each -------------------


def test_a_literal_comparison_is_accepted_at_the_geometries_it_can_walk():
    """Rank 8 is the boundary and the gate is a fall back rather than a raise."""
    rank8 = (torch.randn(1, 1, 1, 1, 1, 1, 1, 4, dtype=F16),)
    assert [c.type for c in _stream(_program(_Op(lambda x: x > 3), rank8))] == [
        _BINARY,
        _SELECT,
    ]
    rank9 = (torch.randn(1, 1, 1, 1, 1, 1, 1, 1, 4, dtype=F16),)
    program = _program(_Op(lambda x: x > 3), rank9)
    assert _delegates(program) == [], "rank 9 has no representation in the descriptor"
    assert any("gt" in target for target in _outer_targets(program))


def test_a_comparison_against_an_operand_the_arena_does_not_hold_falls_back():
    """The operand half of the width rule, which the Scalar form needs too.

    A comparison's result is a bool that one command writes at a byte, and its
    operands are floats the binary command reads two at a time, so the two rules
    answer different questions. This one is the operand half: an fp64 or int
    comparison is refused at the partition so the graph keeps a portable kernel,
    where admitting it raised inside the emitter and failed the whole export.
    The control is the fp16 form at the same geometry.
    """
    assert [c.type for c in _stream(_program(_Op(lambda x: x > 3), (_x(4, 8),)))] == [
        _BINARY,
        _SELECT,
    ]
    for dtype in (torch.float64, torch.int32, torch.int64, torch.bfloat16):
        wide = torch.ones(4, 8, dtype=dtype)
        program = _program(_Op(lambda x: x > 3), (wide,))
        assert _delegates(program) == [], f"a {dtype} comparison must fall back"
        assert any("gt" in target for target in _outer_targets(program))


def test_a_negation_of_anything_but_a_bool_stays_portable():
    """The one operand, and the width it is read at.

    torch's logical_not answers "nonzero becomes False" for any dtype, and the
    kernel's test is nonzero on a one-byte read, so a float or int operand would
    be read at half the stride the graph declared it at. The control is the bool
    at the same shape, which is one command.
    """
    for dtype in (F16, torch.float32, torch.int32, torch.int64):
        value = torch.ones(4, 8, dtype=dtype)
        program = _program(_Op(torch.logical_not), (value,))
        assert _delegates(program) == [], f"a {dtype} operand must stay portable"
        assert any("logical_not" in target for target in _outer_targets(program))
    mask = torch.zeros(4, 8, dtype=torch.bool)
    assert len(_delegates(_program(_Op(torch.logical_not), (mask,)))) == 1


def test_a_bool_operand_is_refused_for_every_target_that_is_not_a_select():
    """The asymmetry the whole cluster turns on, read off one object.

    A bool result is admitted for the comparisons and the negation; a bool
    *operand* is admitted only for the two targets whose command reads a
    condition at one byte. Everything else reads two, so a flag reaching it is a
    wrong number rather than an error. The control is the same node with fp16
    operands, which every one of these admits.
    """
    def node_with(target, out_dtype, arg_dtype):
        graph = torch.fx.Graph()
        lhs = graph.placeholder("x")
        lhs.meta["val"] = torch.empty(4, 8, dtype=arg_dtype)
        rhs = graph.placeholder("y")
        rhs.meta["val"] = torch.empty(4, 8, dtype=arg_dtype)
        node = graph.call_function(target, args=(lhs, rhs))
        node.meta["val"] = torch.empty(4, 8, dtype=out_dtype)
        return node

    support = HexagonOperatorSupport()
    for target in COMPARISON_TARGETS:
        assert support.is_node_supported({}, node_with(target, torch.bool, F16)), target
    for target in (LOGICAL_NOT, BITWISE_NOT):
        assert support.is_node_supported({}, node_with(target, torch.bool, torch.bool)), target
        # The negation's only operand is its condition, read one byte at a time,
        # so the one-byte exemption answers True whatever the slot is declared
        # and the *width* of that slot is the other predicate's question. Both
        # halves of that are asserted, because either alone would pass with the
        # emitter reading a float at a bool's stride.
        assert operand_dtypes_are_readable(node_with(target, torch.bool, F16)), target
        assert not mask_negation_is_emittable(node_with(target, torch.bool, F16)), target
        assert mask_negation_is_emittable(node_with(target, torch.bool, torch.bool)), target
        assert not support.is_node_supported({}, node_with(target, torch.bool, F16)), target
        assert not support.is_node_supported({}, node_with(target, torch.bool, torch.int32)), target
    # The control for the whole set: the same object at the same geometry with
    # fp16 operands, so a support object that refused everything fails here.
    assert support.is_node_supported({}, node_with(_EDGE.add.Tensor, F16, F16))
    assert not support.is_node_supported({}, node_with(_EDGE.add.Tensor, F16, torch.bool))
    # The mirror of the exemption: a bool is a legal *value* only if the result
    # is a bool, and this command has no way to declare a value's width apart
    # from the output's. Where's two-operand sibling is not a where, so the
    # three-argument form is built here rather than reusing the loop's.
    def where_with(cond_dtype, lhs_dtype, rhs_dtype, out_dtype):
        graph = torch.fx.Graph()
        made = []
        for name, dtype in (("c", cond_dtype), ("x", lhs_dtype), ("y", rhs_dtype)):
            placeholder = graph.placeholder(name)
            placeholder.meta["val"] = torch.empty(4, 8, dtype=dtype)
            made.append(placeholder)
        node = graph.call_function(_EDGE.where.self, args=tuple(made))
        node.meta["val"] = torch.empty(4, 8, dtype=out_dtype)
        return node

    assert where_is_emittable(where_with(torch.bool, F16, F16, F16))
    assert not where_is_emittable(where_with(torch.bool, torch.bool, F16, F16)), (
        "a one-byte value read at the result's two bytes is a misread"
    )
    assert not where_is_emittable(where_with(torch.bool, F16, torch.float32, F16))
    assert not where_is_emittable(where_with(F16, F16, F16, F16)), (
        "a two-byte condition read at one byte is the other misread"
    )


def test_the_result_width_and_the_operand_width_are_two_rules_and_they_are_crossed():
    """A bool result with fp16 operands, and the mirror, which is the point.

    The width rule reads `node.meta["val"]` -- the node's own output -- and the
    operand rule reads its arguments, so neither one implies the other. A
    comparison passes the first and would fail the second if the operands were
    one byte, and the mirror is an fp16 result carrying a bool, which is a
    different node entirely.
    """
    assert result_dtype_is_emittable(torch.bool, GREATER_THAN_SCALAR)
    assert result_dtype_is_emittable(torch.bool, LOGICAL_NOT)
    assert result_dtype_is_emittable(torch.bool, BITWISE_NOT)
    assert not result_dtype_is_emittable(torch.bool, _EDGE.add.Tensor)
    assert not result_dtype_is_emittable(torch.int64, GREATER_THAN_SCALAR)
    assert result_dtype_is_emittable(F16, _EDGE.add.Tensor)
    # A comparison's operands are the two-byte widths the binary command reads,
    # which is a rule of its own and not the result's rule restated.
    def operands(dtype):
        graph = torch.fx.Graph()
        lhs = graph.placeholder("x")
        lhs.meta["val"] = torch.empty(4, 8, dtype=dtype)
        node = graph.call_function(GREATER_THAN_SCALAR, args=(lhs, 3))
        node.meta["val"] = torch.empty(4, 8, dtype=torch.bool)
        return node

    assert comparison_operands_are_readable(operands(F16))
    assert comparison_operands_are_readable(operands(torch.float32))
    assert not comparison_operands_are_readable(operands(torch.float64))
    assert not comparison_operands_are_readable(operands(torch.int32))


def test_the_four_comparisons_with_no_op_type_still_refuse_and_say_which_gate():
    """eq, ne, ge and le, in both spellings, with the control beside them.

    `htp_ops_binary_is_compare` admits GREATER and LESS and nothing else, so
    the other four have no command at any width. The control is the same graph
    one spelling across, which must delegate -- otherwise a support object that
    refused everything would satisfy the four assertions.
    """
    a, b = _x(4, 8), _x(4, 8)
    for name in ("eq", "ne", "ge", "le"):
        for overload in ("Tensor", "Scalar"):
            target = getattr(getattr(_EDGE, name), overload)
            assert target not in EMITTERS, f"aten.{name}.{overload}"
            assert target not in BOOL_RESULT_TARGETS, f"aten.{name}.{overload}"
    program = _program(_Op(lambda x: x > 3), (a,))
    assert len(_delegates(program)) == 1
    for name in ("eq", "ge", "le", "ne"):
        program = _program(_Op(lambda x, y: getattr(torch, name)(x, y)), (a, b))
        assert _delegates(program) == [], name
        assert any(name in target for target in _outer_targets(program)), name


# --- the island these three close ------------------------------------------


def test_a_mask_built_on_the_dsp_stays_on_the_dsp_through_its_negation():
    """The chain the gap documents name, measured rather than asserted from it.

    `where(a > b, x, y)` was the shape the comparison gap was about: the where
    delegated and the comparison did not, so the bool crossed the boundary as an
    argument. With a literal comparison and a negation in the same graph, the
    three commands are one delegate and no bool is passed in. The control is the
    same island with the condition supplied from outside, which is one select
    and is how the argument arrived before.
    """
    a, b, x, y = _x(4, 8), _x(4, 8), _x(4, 8), _x(4, 8)
    program = _program(_Op(lambda x, y: torch.where(~(x > 3), x, y)), (a, y))
    delegates = _delegates(program)
    assert len(delegates) == 1, delegates
    commands = _stream(program)
    assert [c.type for c in commands] == [_BINARY, _SELECT, _SELECT, _SELECT], commands
    for node in delegates:
        lowered = program.graph_module.get_submodule(node.args[0].target)
        inner = [
            str(node.target)
            for node in lowered.original_module.graph_module.graph.nodes
        ]
        assert any("gt.Scalar" in target for target in inner), inner
        assert any("bitwise_not" in target for target in inner), inner
        assert all(
            node.meta["val"].dtype is not torch.bool
            for node in lowered.original_module.graph_module.graph.nodes
            if node.op == "placeholder"
        ), "a bool entered the delegate as an argument"

    # `~` and `torch.logical_not` are one command and two ATen nodes, and an
    # island written with one is not the same island as the other. The spelling
    # matters for the count: `~` exports as bitwise_not, and with that node
    # missing it stayed portable and split the island in two, so the control is
    # the same graph with torch.logical_not named outright.
    spelled = _program(_Op(lambda a, b, x, y: torch.where(torch.logical_not(a > b), x, y)),
                       (a, b, x, y))
    assert len(_delegates(spelled)) == 1, _delegates(spelled)
    assert [c.type for c in _stream(spelled)] == [_BINARY, _SELECT, _SELECT, _SELECT]

    mask = torch.zeros(4, 8, dtype=torch.bool)
    outside = _program(_Op(lambda m, x, y: torch.where(~m, x, y)), (mask, x, y))
    assert [c.type for c in _stream(outside)] == [_SELECT, _SELECT], (
        "the same two commands when the mask comes from outside"
    )
