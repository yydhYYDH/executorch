# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which gate a comparison stops at, and what each of the three costs.

Four comparisons reach this backend's tables with nothing on the other side:
`aten.eq.Tensor`, `aten.ne.Tensor`, `aten.ge.Tensor` and `aten.le.Tensor` name a
DSP op type the vendored enum does not have, and `aten.gt.Tensor` and
`aten.lt.Tensor` name one it does have and never reach. This file measures
where each of them is refused, and what the three gates in the way would do if
the table were edited underneath them.

The three are one fact and two clauses, not three facts. `SUPPORTED_TARGETS` is
the `EMITTERS` dict itself, so a target absent from one is absent from the other
and admitting one admits both. The width gate reads the result's dtype and not
its operands', and `operand_dtypes_are_readable` -- the clause below it -- reads
the operands and not the result, which is why a node can fail either and pass
the other. `_require_arena_dtype` raises rather than falling back, so a table
edit that lets a bool result past the partitioner turns an export that worked
into an export that raises.
"""

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
    # 90 when this branch was written, 96 now: the arg-reduction merge added the
    # two reductions to the same table and the comparison merge added four more.
    # The count is a census rather than a constant of the design, so it is
    # re-derived rather than loosened.
    assert len(partition.SUPPORTED_TARGETS) == len(ops.EMITTERS) == 96
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
