# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which gate a comparison stops at, and what each of the three costs.

Four comparisons stop here with nothing on the other side: `aten.eq.Tensor`,
`aten.ne.Tensor`, `aten.ge.Tensor` and `aten.le.Tensor` name a DSP op type the
vendored enum does not have, so they have no row in `EMITTERS` at all.
`aten.gt.Tensor` and `aten.lt.Tensor` name one it does have -- the DSP's own
`HTP_OPS_BINARY_GREATER`(9) and `HTP_OPS_BINARY_LESS`(10) -- and the comparison
merge wired them, so of the six, two delegate and four stay portable. This file
measures where each of the four is refused, owns the count that says two of six
delegate, and records what the three gates did when the table was edited
underneath them.

The three are one fact and two clauses, not three facts. `SUPPORTED_TARGETS` is
the `EMITTERS` dict itself, so a target absent from one is absent from the other
and admitting one admits both. The width gate reads the result's dtype and
not its operands', and `operand_dtypes_are_readable` -- the clause below it
-- reads the operands and not the result, which is why a node can fail either
and pass the other.

The third gate no longer raises behind a relaxed one. `_require_arena_dtype`
and the width gate read one shared predicate, `result_dtype_is_emittable`, so a
comparison's bool result is admitted by both and a bool arriving anywhere else
is refused by both. That is what turned this file's own experiment into the
tree: `test_the_comparison_merge_performed_the_experiment_this_test_described`
used to walk refuse-then-accept against a probe emitter, and its third step --
relax the gate, and the emitter reports that a bool is not a width the arena
holds -- is no longer reachable, because the emitter knows the width. The raise
stays as a backstop for a bool that reaches an emitter by any other route.
"""

import inspect

import pytest
import torch

from blob_interpreter import read_blob

from executorch.backends.hexagon import hexagon_ops as ops
from executorch.backends.hexagon.partition import hexagon_partitioner as partition
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonOperatorSupport,
    HexagonPartitioner,
    _data_placeholders,
    _dtype_of,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

_EDGE = exir_ops.edge.aten

#: The six comparisons an `a OP b` lowers to, named by the overload the graph
#: actually carries. `Tensor` and not `default`: both exist for every one of
#: them, and a table written against the wrong overload asserts about an object
#: no node ever has, which is a test that cannot fail.
COMPARISON_TARGETS = {
    name: getattr(_EDGE, name).Tensor
    for name in ("gt", "lt", "eq", "ne", "ge", "le")
}

#: Targets that are wired, at the geometry the comparisons above are refused at.
#: A refusal assertion is vacuous against a support object that refuses
#: everything, so every count below is read beside these. Three of them reach
#: the graph in the census test; the other two are here for the table check.
CONTROL_TARGETS = {
    "add": _EDGE.add.Tensor,
    "sub": _EDGE.sub.Tensor,
    "relu": _EDGE.relu.default,
    "maximum": _EDGE.maximum.default,
    "where": _EDGE.where.self,
}

#: The three the census graph carries, so a control that is merely "some node was
#: accepted" cannot be satisfied by a placeholder.
_GRAPH_CONTROLS = (
    _EDGE.add.Tensor,
    _EDGE.relu.default,
    _EDGE.maximum.default,
)


def _node(result_dtype, operand_dtypes, numel=(4, 8)):
    """A call_function node carrying recorded values and nothing else."""
    graph = torch.fx.Graph()
    args = []
    for index, dtype in enumerate(operand_dtypes):
        placeholder = graph.placeholder("in%d" % index)
        placeholder.meta["val"] = torch.zeros(numel).to(dtype)
        args.append(placeholder)
    node = graph.call_function(_EDGE.add.Tensor, tuple(args))
    node.meta["val"] = torch.zeros(numel).to(result_dtype)
    return node


def test_supported_targets_is_the_emitter_table_and_not_a_second_one():
    """One object, so the wiring question is asked once and answered once."""
    assert partition.SUPPORTED_TARGETS is ops.EMITTERS
    # 90 when this branch was written, 96 after the comparison merge, 100 now:
    # the arg-reduction merge added two reductions, the comparison merge four
    # comparison targets, and the Scalar-comparison, logical_not and bitwise_not
    # rows four more -- gt.Scalar, lt.Scalar, logical_not and bitwise_not. The count
    # is a census rather than a constant of the design, so it is re-derived rather
    # than loosened.
    assert len(partition.SUPPORTED_TARGETS) == len(ops.EMITTERS) == 100
    for name, target in COMPARISON_TARGETS.items():
        # Two of the six are wired since this file was written, so the absence is
        # the narrower claim now, and the narrower one is the useful one: eq, ne,
        # ge and le have no kernel at any width and never will on this path.
        if name in ("gt", "lt"):
            assert target in ops.EMITTERS
        else:
            assert target not in ops.EMITTERS
    for target in CONTROL_TARGETS.values():
        assert target in ops.EMITTERS


def test_an_emitter_row_is_a_supported_target_without_a_second_edit():
    """The consequence of the identity, on a copy so the live table is intact.

    A copy because this is the edit a branch would make, and the point is that
    it needs no second one. Were these two tables, the row below would be found
    by grep and forgotten.
    """
    widened = dict(ops.EMITTERS)
    widened[COMPARISON_TARGETS["eq"]] = lambda node, ctx: None
    assert len(widened) == len(ops.EMITTERS) + 1
    assert COMPARISON_TARGETS["eq"] in widened
    assert COMPARISON_TARGETS["ne"] not in widened


def test_the_width_gate_reads_the_result_and_the_operand_rule_reads_the_operands():
    """The two clauses are crossed, and each is blind to what the other sees.

    A node with a bool result and fp16 operands is refused by the width gate
    and *passes* the operand rule, so the operand rule is not what holds a
    comparison. A node with an fp16 result and bool operands is the mirror: the
    width gate passes it and the operand rule refuses. The fp16/fp16 node is
    the positive control, and it has to pass both or the two assertions above
    say nothing.
    """
    bool_result = _node(torch.bool, (torch.float16, torch.float16))
    assert _dtype_of(bool_result) is torch.bool
    assert _dtype_of(bool_result) not in (torch.float16, torch.float32)
    assert ops.operand_dtypes_are_readable(bool_result) is True

    bool_operands = _node(torch.float16, (torch.bool, torch.bool))
    assert _dtype_of(bool_operands) is torch.float16
    assert _dtype_of(bool_operands) in (torch.float16, torch.float32)
    assert ops.operand_dtypes_are_readable(bool_operands) is False

    control = _node(torch.float16, (torch.float16, torch.float16))
    assert _dtype_of(control) in (torch.float16, torch.float32)
    assert ops.operand_dtypes_are_readable(control) is True


def test_require_arena_dtype_raises_on_a_bool_result_wherever_the_bool_is():
    """It reads the node's own value, so it is the result that is checked.

    A bool on an operand with an fp16 result is accepted, and a bool result is
    refused whether the operands are fp16 or bool. That is the clause the
    partitioner's width gate mirrors, reached from the emitter side, and it
    raises rather than returning False: every call site is a statement in an
    emitter, so a fallback is not what a caller would get.
    """
    with pytest.raises(RuntimeError, match="must be fp16 or fp32"):
        ops._require_arena_dtype(
            _node(torch.bool, (torch.float16, torch.float16)), "probe"
        )
    with pytest.raises(RuntimeError, match="must be fp16 or fp32"):
        ops._require_arena_dtype(_node(torch.bool, (torch.bool, torch.bool)), "probe")
    assert (
        ops._require_arena_dtype(
            _node(torch.float16, (torch.bool, torch.bool)), "probe"
        )
        is None
    )
    assert (
        ops._require_arena_dtype(
            _node(torch.float16, (torch.float16, torch.float16)), "probe"
        )
        is None
    )


class _SixComparisons(torch.nn.Module):
    def forward(self, a, b):
        return (a == b, a != b, a >= b, a <= b, a > b, a < b)


class _ComparisonsAndThreeControls(torch.nn.Module):
    """The same six plus three wired ops of the same geometry.

    One graph, so the support object that refuses the six is the same object
    that accepts the three, and a count of refusals cannot be an artifact of an
    object that refuses everything.
    """

    def forward(self, a, b):
        return (
            a == b,
            a != b,
            a >= b,
            a <= b,
            a > b,
            a < b,
            a + b,
            torch.relu(a),
            torch.maximum(a, b),
        )


def _lowered(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _verdicts():
    """Per-target support verdict, on the unpartitioned edge graph.

    The lowered graph is the wrong place to read this: partitioning has already
    replaced the three wired nodes with one `executorch_call_delegate`, so their
    verdicts are gone and a count read there is a count of leftovers. The keys
    are the nodes' own targets, so the table of six is checked against the graph
    rather than against a hand-written list that could name a different overload.
    """
    edge_program = to_edge(
        export(_ComparisonsAndThreeControls(), _inputs()),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    support = HexagonOperatorSupport(_data_placeholders(edge_program))
    out = {}
    for node in edge_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        out[node.target] = support.is_node_supported(edge_program.graph_module, node)
    return out


def _commands(program):
    commands = []
    for node in program.graph_module.graph.nodes:
        if node.target is not torch.ops.higher_order.executorch_call_delegate:
            continue
        lowered = program.graph_module.get_submodule(node.args[0].target)
        _, decoded = read_blob(bytes(lowered._processed_bytes))
        commands.extend(decoded)
    return commands


def _inputs():
    return (torch.randn(8, 8, dtype=torch.float16), torch.randn(8, 8, dtype=torch.float16))


def test_two_comparisons_are_wired_and_four_refuse_and_three_wired_ops_do_not():
    """The number for what the backend does with a comparison today.

    Was zero of six when this file was written and is two of six now: the
    comparison merge wired `gt` and `lt` against a tensor and left the other
    four alone, which is the split its own section 3 argues for from the DSP
    side. `test_comparisons.py` owns the detail of what the two emit; this
    file still owns the count, because the count is what a reader of the gap
    document is checking.

    Two of six and three of three, in one graph and against one support
    object, which is what makes the zero a measurement rather than an object
    that refuses everything. The control has to name the three by name: a
    control that is merely "some node was accepted" would also be satisfied by
    a placeholder.
    """
    verdicts = _verdicts()
    comparisons = {
        target: value
        for target, value in verdicts.items()
        if target in set(COMPARISON_TARGETS.values())
    }
    controls = {
        target: verdicts[target]
        for target in verdicts
        if target in set(_GRAPH_CONTROLS)
    }
    assert set(comparisons) == set(COMPARISON_TARGETS.values()), sorted(
        str(t) for t in comparisons
    )
    assert sum(comparisons.values()) == 2, comparisons
    assert sorted(
        str(t) for t, v in comparisons.items() if v
    ) == sorted(
        str(COMPARISON_TARGETS[name]) for name in ("gt", "lt")
    ), comparisons
    assert len(controls) == 3, controls
    assert all(controls.values()), controls


def test_a_graph_of_six_comparisons_now_delegates_and_still_omits_four_of_them():
    """Six comparisons, one delegate, and four of the six still portable.

    This was zero delegates and zero commands. The comparison merge made it one
    delegate, and the interesting half is that it is one: a delegate carrying
    only the two wired comparisons is a different thing from six delegates, and
    a graph whose four unwired comparisons had been absorbed would show up here
    as a stream naming a command the DSP has no route for.
    """
    program = _lowered(_SixComparisons(), _inputs())
    delegates = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(delegates) == 1, delegates
    names = {str(command.type) for command in _commands(program)}
    assert not any(
        name in names for name in ("eq", "ne", "ge", "le")
    ), f"an unwired comparison reached a command: {names}"


def test_the_control_graph_delegates_and_its_stream_carries_no_comparison():
    """The one thing a count of zero needs beside it: a stream that is not zero.

    The delegate decodes to a BINARY_ELEMENTWISE, and neither compare op type
    (9 and 10) appears -- so "no comparison command" is a statement about a
    stream that exists, not about a graph that produced nothing.
    """
    program = _lowered(_ComparisonsAndThreeControls(), _inputs())
    types = [command.type for command in _commands(program)]
    assert types, "the control graph produced no command at all"
    assert ops.DSP_OP_BINARY_ELEMENTWISE in types, types
    compare_types = {9: "GREATER", 10: "LESS"}
    assert not [t for t in types if t in compare_types], types


class _CompareAndAdd(torch.nn.Module):
    def forward(self, a, b):
        return a > b, a + b


class _AddOnly(torch.nn.Module):
    def forward(self, a, b):
        return a + b


def _probe_greater_emitter(node, ctx):
    """A comparison emitter of the shape one would actually write.

    The `_require_arena_dtype` line is not decoration: all nineteen other
    emitters in this backend open with it, and it is the line the experiment
    below turns into a failure.
    """
    ops._require_arena_dtype(node, "probe greater")
    numel = ops._numel(node)
    return ctx.emit(
        node,
        ops.Op(
            type=ops.DSP_OP_BINARY_ELEMENTWISE,
            inputs=[ctx.operand(node.args[0]), ctx.operand(node.args[1])],
            outputs=[ctx.result_for(node, numel, torch.bool)],
            params=[numel, numel, numel, 1, ops.FP16_BYTES, ops.FP16_BYTES, 0, 0]
            + [ops.BINARY_OP_TYPES["greater"]]
            + [8, 8, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0],
        ),
    )


def _lower(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegate_count(program):
    return sum(
        1
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    )


def test_the_width_gate_admits_a_bool_result_only_for_a_wired_comparison():
    """The clause that refused a bool result now admits it, and only where it should.

    This asserted the opposite and the comparison merge is what overturned it. The
    probe emitter it installed on `gt.Tensor` to get past the membership test is
    now a real one, and the width gate reads the same shared predicate the
    emitters do, so a bool result is admitted for a comparison and refused
    everywhere else. That is the narrower claim and the one the tree now makes:
    `result_dtype_is_emittable` answers True for torch.bool only when the target
    is in COMPARISON_TARGETS, so the gate is still a gate rather than a removal.

    What is no longer measurable here is the old sequence, refuse-then-accept on
    one node, because nothing in this graph is refused at the width gate any more.
    `test_comparisons.py` owns the forward-looking behaviour; the property this
    file still pins is that a bool result is a per-target decision.
    """
    assert ops.result_dtype_is_emittable(torch.bool, _EDGE.gt.Tensor) is True
    assert ops.result_dtype_is_emittable(torch.bool, _EDGE.lt.Tensor) is True
    for name in ("eq", "ne", "ge", "le"):
        target = COMPARISON_TARGETS[name]
        assert ops.result_dtype_is_emittable(torch.bool, target) is False, name
    assert (
        ops.result_dtype_is_emittable(torch.bool, _EDGE.add.Tensor) is False
    ), "a bool result on a non-comparison is still refused"


def test_the_comparison_merge_performed_the_experiment_this_test_described():
    """What this test walked through by hand is now just the tree.

    It was three steps measured against a probe emitter: no table row and the
    export falls back; a row and the width gate still refuses, because the gate
    runs before any emitter; relax the gate and the emitter runs and the first
    thing it says is that a bool is not a width the arena holds. The comparison
    merge then did exactly those three steps and kept the result, so the third
    state is no longer reachable here: the shared predicate admits the bool
    result for a comparison and the emitter knows the width.

    What is left to measure is the end state, and the control that made the
    original third step a claim about the bool rather than about the
    relaxation: an fp16-result add under the same shapes still delegates.

    The six-comparison count is the part that is easy to get wrong in the other
    direction. A width gate relaxed far enough to admit a bool everywhere would
    still pass both delegate counts above and still fail this one, because a
    delegate carrying all six would name a command the DSP has no route for.
    """
    inputs = _inputs()

    assert _delegate_count(_lower(_CompareAndAdd(), inputs)) == 1
    assert _delegate_count(_lower(_AddOnly(), inputs)) == 1

    program = _lower(_SixComparisons(), inputs)
    assert _delegate_count(program) == 1
    names = {str(command.type) for command in _commands(program)}
    assert not any(name in names for name in ("eq", "ne", "ge", "le"))


#: The clause the operand gate refuses an int64 operand on, spelled out so that a
#: reformat of the predicate makes this file fail loudly instead of quietly
#: turning the mutation below into a no-op that still passes.
_OPERAND_WIDTH_CLAUSE = """value.dtype not in (
            torch.float16,
            torch.float32,
        )"""
_MUTATED_OPERAND_WIDTH_CLAUSE = """value.dtype not in (
            torch.float16,
            torch.float32,
            torch.int64,
        )"""


class _Int64Comparison(torch.nn.Module):
    """The shape an ASR front end produces: an int64 arange against int64 lengths.

    Its result is a torch.bool, which is the width `result_dtype_is_emittable`
    admits on purpose for a wired comparison, so the result gate passes and the
    operand is the only thing left to refuse it.
    """

    def forward(self, lengths):
        steps = torch.arange(0, 8, dtype=torch.int64)
        return (steps[: lengths.shape[-1]] < lengths).to(torch.float16)


class _Fp16Comparison(torch.nn.Module):
    """The same geometry at the two widths the arena holds."""

    def forward(self, lengths):
        steps = torch.arange(0, 8, dtype=torch.float16)
        return (steps[: lengths.shape[-1]] < lengths).to(torch.float16)


class _ArgReduction(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=-1)


def _call_function(target, result_dtype, operand_dtypes, numel=(4,)):
    """A call_function node carrying recorded values and nothing else."""
    graph = torch.fx.Graph()
    args = []
    for index, dtype in enumerate(operand_dtypes):
        placeholder = graph.placeholder("in%d" % index)
        placeholder.meta["val"] = torch.zeros(numel).to(dtype)
        args.append(placeholder)
    node = graph.call_function(target, tuple(args))
    node.meta["val"] = torch.zeros(numel).to(result_dtype)
    return node


def _verdict(node):
    """One node's support verdict, on a support object built as the partitioner does."""
    program = to_edge(
        export(_Fp16Comparison(), (torch.zeros(4, dtype=torch.float16),)),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    return HexagonOperatorSupport(_data_placeholders(program)).is_node_supported(
        program.graph_module, node
    )


def _delegated_targets(program):
    names = set()
    for node in program.graph_module.graph.nodes:
        if node.target is not torch.ops.higher_order.executorch_call_delegate:
            continue
        lowered = program.graph_module.get_submodule(node.args[0].target)
        for inner in lowered.original_module.graph_module.graph.nodes:
            if inner.op != "call_function":
                continue
            text = str(inner.target)
            names.add(
                text[len("<EdgeOpOverload: ") :].split(">:")[0]
                if text.startswith("<EdgeOpOverload: ")
                else text
            )
    return names


def test_the_operand_gate_refuses_an_int64_operand_a_wired_comparison_carries():
    """The defect: the two gates disagree on exactly this shape.

    A bool result passes the result gate whatever it compared, so the operand is
    the only thing that can refuse this node -- and until the operand rule read
    widths and not only bools, nothing did. The emitter caught it instead, and
    `_require_arena_dtype` raising there is not a fallback: it takes the whole
    `to_edge_transform_and_lower` down, so an ASR model whose mask is
    `arange < lengths` could not be exported at all.
    """
    node = _call_function(
        _EDGE.lt.Tensor, torch.bool, (torch.int64, torch.int64)
    )
    assert _dtype_of(node) is torch.bool
    assert partition.result_dtype_is_emittable(_dtype_of(node), node.target) is True
    assert ops.operand_dtypes_are_readable(node) is False
    assert _verdict(node) is False


def test_the_width_clause_and_not_another_one_is_what_refuses_that_node():
    """The assertion above is load-bearing only if this clause is the reason.

    Relaxing it in the predicate's own source and re-executing it puts the int64
    node back on the DSP, while the fp16 twin is taken either way. A test naming
    a clause can be satisfied by whichever clause happens to fire, so the node
    this one claims to refuse has to flip when the clause is dropped -- and the
    control has to stay put, or the relaxed predicate is refusing everything
    instead of everything-but-int64.
    """
    source = inspect.getsource(ops.operand_dtypes_are_readable)
    assert _OPERAND_WIDTH_CLAUSE in source, "the clause this file mutates is gone"
    namespace = dict(vars(ops))
    exec(compile(source.replace(
        _OPERAND_WIDTH_CLAUSE, _MUTATED_OPERAND_WIDTH_CLAUSE
    ), "<mutated operand gate>", "exec"), namespace)
    relaxed = namespace["operand_dtypes_are_readable"]

    int64 = _call_function(_EDGE.lt.Tensor, torch.bool, (torch.int64, torch.int64))
    fp16 = _call_function(_EDGE.lt.Tensor, torch.bool, (torch.float16, torch.float16))
    assert relaxed(int64) is True, "the clause under test is not what refuses it"
    assert relaxed(fp16) is True
    assert ops.operand_dtypes_are_readable(int64) is False
    assert ops.operand_dtypes_are_readable(fp16) is True


def test_the_fp16_twin_of_that_geometry_still_delegates_the_comparison():
    """The positive control, and a refusal assertion is vacuous without one.

    A support object that refused everything would satisfy the two tests above, so
    the same geometry at fp16 has to still reach a delegate with the comparison
    inside it -- named, not counted, so an empty delegate cannot satisfy it.
    """
    program = _lower(
        _Fp16Comparison(), (torch.tensor([2, 5, 1, 3], dtype=torch.float16),)
    )
    assert _delegate_count(program) == 1
    assert _delegated_targets(program) == {
        "aten.lt.Tensor",
        "aten.slice_copy.Tensor",
    }


def test_the_int64_comparison_lowers_and_the_refusal_is_named():
    """The end state, measured on the refusal rather than on a missing delegate.

    "No delegate" is also what a graph of nothing but refusals gives, so the
    assertion is the count the census reports for the comparison itself.
    """
    partition.reset_refused_overload_census()
    program = _lower(
        _Int64Comparison(), (torch.tensor([2, 5, 1, 3], dtype=torch.int64),)
    )
    assert _delegate_count(program) == 0
    assert partition.refused_overload_census().get("aten.lt.Tensor") == 1


def test_an_arg_reduction_still_delegates_its_int64_result():
    """The negative control, on the case the special case in _verdict exists for.

    An arg reduction's result is int64 by design and its command writes that width
    directly, which is why it has to run before the shared width check. The
    operand gate is a different question: it reads the operand, which is the fp16
    data. Widening it to widths must not reach the index result, or every valid
    index in the model is refused.
    """
    program = _lower(_ArgReduction(), (torch.randn(2, 8, dtype=torch.float16),))
    assert _delegate_count(program) == 1
    assert _delegated_targets(program) == {"aten.argmax.default"}
    fp16 = _call_function(_EDGE.argmax.default, torch.int64, (torch.float16,))
    assert _dtype_of(fp16) is torch.int64
    assert ops.operand_dtypes_are_readable(fp16) is True
    assert _verdict(fp16) is True


def test_an_arg_reduction_over_an_int64_source_stops_delegating():
    """The one behaviour the widening changes, stated rather than left implied.

    The kernel reads its source as half floats, so an int64 arg reduction
    delegated today answered something other than the index torch computed. It is
    now portable. The fp16 case above is the control that stays on the DSP.
    """
    int64 = _call_function(_EDGE.argmax.default, torch.int64, (torch.int64,))
    assert ops.operand_dtypes_are_readable(int64) is False
    assert _verdict(int64) is False


def test_the_exemptions_are_operands_a_command_declares_its_own_width_for():
    """A gate that refused every operand would pass every test above.

    Each exemption is a width the kernel reads at a width of its own, keyed by
    argument position: a where's one-byte condition slot, a gather's four-byte
    index slot. The gather's indices are int32 *and* int64, because the slot is
    declared four bytes an element either way and the runtime narrows a wider one
    into it -- so exempting the position rather than the width is the whole
    reason int64 keeps delegating there.
    """
    assert ops.NON_ARENA_OPERAND_SLOTS[ops.WHERE] == (0,)
    assert ops.NON_ARENA_OPERAND_SLOTS[ops.EMBEDDING] == (1,)
    assert ops.NON_ARENA_OPERAND_SLOTS[ops.INDEX_SELECT] == (2,)
    assert ops.NON_ARENA_OPERAND_SLOTS[ops.UPDATE_CACHE] == (2,)

    where = _call_function(
        ops.WHERE, torch.float16, (torch.bool, torch.float16, torch.float16)
    )
    assert ops.operand_dtypes_are_readable(where) is True
    for indices in (torch.int32, torch.int64):
        assert ops.operand_dtypes_are_readable(
            _call_function(ops.EMBEDDING, torch.float16, (torch.float16, indices))
        ) is True
    # The index slot is what is exempted, so an int64 *table* is still refused.
    assert ops.operand_dtypes_are_readable(
        _call_function(ops.EMBEDDING, torch.float16, (torch.int64, torch.int64))
    ) is False


def test_the_emitter_still_raises_for_an_int64_operand_that_reaches_it():
    """The backstop is not dead code, which is why the fix belongs on the other side.

    `_require_arena_dtype` is what caught this defect, and it is kept: a node
    that arrives by a route the support check does not cover still raises rather
    than emits a command that reads eight bytes an element as two.
    """
    node = _call_function(_EDGE.lt.Tensor, torch.bool, (torch.int64, torch.int64))
    with pytest.raises(RuntimeError, match="must be fp16 or fp32"):
        ops._require_arena_dtype(node.args[0], "comparison less operand")
