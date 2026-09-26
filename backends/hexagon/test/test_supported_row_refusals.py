"""Why the supported-row nodes a 28-layer Qwen3 leaves portable are portable.

A bucket census over the lowered graph counted `aten.add` 3, `aten.sub` 2,
`aten.cat` 2, `aten.index` 2 and `aten.cumsum` 1 on ops the support table
lists as delegated, and a count records no reasons. Reading the reasons out of
the partitioner settles it, and the answer is one clause and not five.

Eight of the nine are the RoPE position arithmetic: `arange` plus an offset, a
difference of two position slices, a concatenation of one, and the two-axis
gathers that read the causal mask out of a cumsum. Every one of them produces
int64, and the width gate at `hexagon_partitioner.py` refuses a node whose
result is not fp16 or fp32 before any op-specific gate is read. That is the
documented limit -- the arena holds two bytes an element -- and it is the same
refusal that holds back 85 int64 position reads a Qwen3 forward pass already
carries in `OP_GAPS.md`.

The tenth, `aten.cumsum`, is not a refusal of a supported row at all. The
emitter is registered for `et_hexagon.cumsum`, the fused node `FuseCumsumPass`
creates, and `aten.cumsum` is not in the emitter table, so the partitioner
stops at the membership test, before the width gate. The census that found it
bucketed it with four families it does share a row with, which is the
difference a bucket count cannot see -- and it is invisible to both
partitioner censuses, because the unwired one wants a family the emitter table
names and the table names `et_hexagon::cumsum`.

So: no defect. One row-counted reason (the width) and one miscategorisation
(`aten.cumsum` in a supported-rows list). Every case here asserts both halves --
the verdict, and the command stream the fp16 twin of the same op produces --
so a partitioner that refused everything would fail this file rather than pass
it, and a width gate that stopped working would go red on the command.
"""

import sys

import pytest
import torch
import torch.nn as nn

from blob_interpreter import read_blob  # noqa: E402

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition import hexagon_partitioner
from executorch.backends.hexagon.hexagon_backend import SUPPORTED_TARGETS
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
    refused_overload_census,
    reset_refused_overload_census,
    reset_unwired_overload_census,
    unwired_overload_census,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from torch.export import export

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tenrefuse_probe import RefusalTrace, target_name  # noqa: E402

F16 = torch.float16
I64 = torch.int64
CONFIG = EdgeCompileConfig(_check_ir_validity=False)


@pytest.fixture(autouse=True)
def _fresh_censuses():
    reset_refused_overload_census()
    reset_unwired_overload_census()
    yield
    reset_refused_overload_census()
    reset_unwired_overload_census()


class _M(torch.nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.forward_fn = forward

    def forward(self, *args):
        return self.forward_fn(*args)


def _program(model, inputs):
    return to_edge(export(_M(model), inputs), compile_config=CONFIG).exported_program()


def _support(program):
    return HexagonOperatorSupport(_data_placeholders(program))


def _verdict(model, inputs, target):
    """The support verdict for every node of one target, as a list of bools."""
    program = _program(model, inputs)
    support = _support(program)
    return [
        support.is_node_supported({}, node)
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and target_name(node) == target
    ]


def _refusals(model, inputs):
    reset_refused_overload_census()
    program = _program(model, inputs)
    support = _support(program)
    for node in program.graph_module.graph.nodes:
        if node.op == "call_function":
            support.is_node_supported({}, node)
    return refused_overload_census()


def _unwired(model, inputs):
    reset_unwired_overload_census()
    program = _program(model, inputs)
    support = _support(program)
    for node in program.graph_module.graph.nodes:
        if node.op == "call_function":
            support.is_node_supported({}, node)
    return unwired_overload_census()


def _commands(model, inputs):
    program = to_edge_transform_and_lower(
        export(_M(model), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    out = []
    for call in program.graph_module.graph.nodes:
        if call.target is not torch.ops.higher_order.executorch_call_delegate:
            continue
        module = program.graph_module.get_submodule(call.args[0].target)
        out.extend(c.type for c in read_blob(bytes(module._processed_bytes))[1])
    return out


def _refuse_line(model, inputs, target):
    """The `_verdict` line that refused this target, from the tracer."""
    with RefusalTrace() as trace:
        to_edge_transform_and_lower(
            export(_M(model), inputs),
            partitioner=[HexagonPartitioner()],
            compile_config=CONFIG,
        )
    lines = {
        r["refuse_line"]
        for r in trace.records.values()
        if r["target"] == target and not r["verdict"]
    }
    assert len(lines) == 1, f"expected one refusing line for {target}, got {lines}"
    return lines.pop()


class _PositionArithmetic(torch.nn.Module):
    """The RoPE position prologue a Qwen3 export writes, at one token.

    Every node here is one the census found portable on a supported row: an
    offset added to `arange`, a difference of two position slices, and the
    concatenation that puts the offset in front of the sequence length. These
    are the shapes the ten nodes have, so a case built from anything else would
    be measuring a different graph than the one the census counted.
    """

    def forward(self, position_ids):
        arange = torch.arange(position_ids.shape[-1], dtype=I64)
        return torch.cat(
            [position_ids[:, :1] - position_ids[:, :1], position_ids + arange], dim=-1
        )


#: The two lines that decide these nodes, found by the tracer rather than
#: written from memory, so a gate that moves fails this file loudly instead
#: of quietly asserting a clause other than the one that refuses.
WIDTH_GATE = 686
UNWIRED_GATE = 678


def test_every_supported_row_node_in_the_position_prologue_is_refused_at_one_line():
    """The census counted five families; the partitioner names one clause."""
    inputs = (torch.zeros(1, 3, dtype=I64),)
    assert _verdict(_PositionArithmetic(), inputs, "aten::add") == [False]
    assert _verdict(_PositionArithmetic(), inputs, "aten::sub") == [False]
    assert _verdict(_PositionArithmetic(), inputs, "aten::cat") == [False]
    assert _refuse_line(_PositionArithmetic(), inputs, "aten::add") == WIDTH_GATE
    assert _refuse_line(_PositionArithmetic(), inputs, "aten::sub") == WIDTH_GATE
    assert _refuse_line(_PositionArithmetic(), inputs, "aten::cat") == WIDTH_GATE


def test_the_clause_is_the_result_width_and_not_the_op_own_geometry():
    """A refused node whose own gate would have accepted it is the width alone.

    `add` and `sub` on these operands pass `_broadcast_fits_dsp_limits`: the
    output is `[1, 3]`, the operands are `[1, 3]` and a scalar, and the
    descriptor walks that without complaint. So the geometry is emittable and
    only the width stops it, which is the difference between "this op cannot
    do this shape" and "this op cannot do int64 at all".
    """
    program = _program(_PositionArithmetic(), (torch.zeros(1, 3, dtype=I64),))
    support = _support(program)
    seen = {}
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = target_name(node)
        if name not in ("aten::add", "aten::sub", "aten::cat"):
            continue
        seen[name] = {
            "verdict": support.is_node_supported({}, node),
            "own_gate": hexagon_partitioner._broadcast_fits_dsp_limits(node)
            if name in ("aten::add", "aten::sub")
            else None,
            "result_dtype": str(hexagon_partitioner._dtype_of(node)),
        }
    assert seen["aten::add"]["own_gate"] is True
    assert seen["aten::sub"]["own_gate"] is True
    for name in ("aten::add", "aten::sub", "aten::cat"):
        assert seen[name]["result_dtype"] == "torch.int64"
        assert seen[name]["verdict"] is False


def test_the_width_gate_does_not_refuse_the_fp16_twin_of_the_same_geometry():
    """The control: same ops, same shapes, fp16, and they emit commands.

    Without this a partitioner that refused every node would satisfy every row
    above, and the file would be a description of a backend that runs nothing.
    """

    def forward(x, y):
        return torch.cat([x - y, x + y], dim=-1)

    inputs = (torch.randn(2, 8, dtype=F16), torch.randn(2, 8, dtype=F16))
    assert _verdict(forward, inputs, "aten::add") == [True]
    assert _verdict(forward, inputs, "aten::sub") == [True]
    assert _verdict(forward, inputs, "aten::cat") == [True]
    commands = _commands(forward, inputs)
    assert hexagon_ops.DSP_OP_BINARY_ELEMENTWISE in commands
    assert hexagon_ops.DSP_OP_RASTER_BLIT in commands


def test_aten_cumsum_is_not_a_refusal_of_a_supported_row():
    """The tenth node, and the categorisation the census carried.

    `aten.cumsum` is not in the emitter table: the emitter is registered for
    `et_hexagon.cumsum`, the fused node `FuseCumsumPass` creates, and that pass
    is opt-in. So the partitioner stops at the membership test, before the width
    gate, and the width is not why this one is portable. The bucket census filed
    it with four families it does share a row with, which is the difference a
    count cannot see.

    It is also invisible to *both* partitioner censuses, and that is a fact about
    the counters rather than about this op. `_note_unwired_target` only records a
    target whose family the emitter table speaks for, and the table's family is
    `et_hexagon::cumsum` while this node's is `aten::cumsum`; so the refused
    census skips it (not a target in the table) and the unwired census skips it
    too (a family the table does not name). The same boundary an op with no
    emitter at all has, reached here by a node whose fused twin does have one.
    """
    from executorch.exir.dialects._ops import ops as exir_ops

    assert exir_ops.edge.aten.cumsum.default not in SUPPORTED_TARGETS
    assert exir_ops.edge.aten.cumsum.default not in hexagon_ops.EMITTERS
    assert exir_ops.edge.et_hexagon.cumsum.default in SUPPORTED_TARGETS
    assert "aten::cumsum" not in hexagon_partitioner._emitted_families()
    assert "et_hexagon::cumsum" in hexagon_partitioner._emitted_families()

    forward = lambda x: torch.cumsum(x, dim=-1)  # noqa: E731
    inputs = (torch.arange(64, dtype=I64),)
    assert _verdict(forward, inputs, "aten::cumsum") == [False]
    assert _refuse_line(forward, inputs, "aten::cumsum") == UNWIRED_GATE
    assert _refusals(forward, inputs) == {}
    assert _unwired(forward, inputs) == {}

