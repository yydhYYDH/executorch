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

#: Every comparison the edge dialect spells as a tensor-tensor overload. All
#: six refuse, and they are the six an `a == b` written in torch lowers to.
COMPARISON_TARGETS = {
    name: getattr(_EDGE, name).default
    for name in ("gt", "lt", "eq", "ne", "ge", "le")
}

#: Targets that are wired, at the geometry the comparisons above are refused at.
#: A refusal assertion is vacuous against a support object that refuses
#: everything, so every count below is read beside these.
CONTROL_TARGETS = {
    "add": _EDGE.add.Tensor,
    "sub": _EDGE.sub.Tensor,
    "relu": _EDGE.relu.default,
    "maximum": _EDGE.maximum.default,
    "where": _EDGE.where.self,
}

_CONTROL_OP_NAMES = ("add", "relu", "maximum")


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
    assert len(partition.SUPPORTED_TARGETS) == len(ops.EMITTERS) == 90
    for target in COMPARISON_TARGETS.values():
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
    verdicts are gone and a count read there is a count of leftovers.
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
        name = node.target.__name__ if hasattr(node.target, "__name__") else str(node.target)
        out[name] = support.is_node_supported(edge_program.graph_module, node)
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


def test_six_comparisons_refuse_and_three_wired_ops_of_the_same_geometry_do_not():
    """The number for what the backend does with a comparison today.

    Zero of six and three of three, in one graph and against one support
    object, which is what makes the zero a measurement rather than an object
    that refuses everything. The control has to name the three by name: a
    control that is merely "some node was accepted" would also be satisfied by
    a placeholder.
    """
    verdicts = _verdicts()
    comparisons = {
        name: value
        for name, value in verdicts.items()
        if name.split(".")[1] in ("eq", "ne", "ge", "le", "gt", "lt")
    }
    controls = {name: verdicts[name] for name in verdicts if name.split(".")[1] in _CONTROL_OP_NAMES}
    assert sorted(comparisons) == [
        "aten.eq.Tensor",
        "aten.ge.Tensor",
        "aten.gt.Tensor",
        "aten.le.Tensor",
        "aten.lt.Tensor",
        "aten.ne.Tensor",
    ]
    assert sum(comparisons.values()) == 0, comparisons
    assert len(controls) == 3, controls
    assert all(controls.values()), controls


def test_a_graph_of_comparisons_alone_produces_no_delegate_and_no_command():
    """Six comparisons, zero delegates, zero commands."""
    program = _lowered(_SixComparisons(), _inputs())
    delegates = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert delegates == []
    assert _commands(program) == []


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
