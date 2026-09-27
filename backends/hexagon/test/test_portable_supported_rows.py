"""The five supported-row families a 28-layer Qwen3 still leaves portable.

A bucket census over the lowered graph counted `aten.add` 3, `aten.sub` 2,
`aten.cat` 1, `aten.index` 2 and `aten.cumsum` 1 -- nine nodes on rows the
emitter table lists as delegated.  A count records no reason, and the three ways
a node can be portable are three different sizes of work: refused by a named
clause, accepted and then left out of every delegate, or in a delegate.  This
file asks the partitioner, so each row names its clause, and every row carries
the control that would fail if the partitioner answered a different question.

Two instruments read the same refusals and this file needs them to agree: a
rewrite of `_verdict`'s own text that tags every `return False` with the file
line it is on, and a `sys.settrace` over the unmodified function that records
the last line its frame ran.  The first measures a function this file replaced,
the second the function the tree ships.

Host tier: export, `to_edge`, the support predicate, and an AOT blob read back
with the runtime's own reader.  No kernel runs and no phone is involved; a
delegate here is a command in a blob, which is a statement about host lowering.

Measured at `bbe19c9`; every line number is derived from the source at import
time rather than written down, because a clause that moves should fail the
derivation instead of quietly asserting a different clause than the one that
decides.
"""

import inspect
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

from blob_interpreter import read_blob  # noqa: E402

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition import hexagon_partitioner as hpart
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonOperatorSupport,
    HexagonPartitioner,
    _data_placeholders,
)
from executorch.backends.hexagon.hexagon_backend import EMITTERS, SUPPORTED_TARGETS
from executorch.backends.hexagon.cumsum import cumsum_is_emittable
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

sys.path.insert(0, __file__.rsplit("/", 1)[0])

F16 = torch.float16
I64 = torch.int64
CONFIG = EdgeCompileConfig(_check_ir_validity=False)
DELEGATE = torch.ops.higher_order.executorch_call_delegate
WATCHED = ("aten::add", "aten::sub", "aten::cat", "aten::index", "aten::cumsum")

#: The census buckets, re-measured.  The prologue is per model, not per layer, so
#: a 28-layer Qwen3 and a 1-layer one carry the same nine nodes.
EXPECTED_BUCKETS = {
    "aten::add": 3,
    "aten::sub": 2,
    "aten::cat": 1,
    "aten::index": 2,
    "aten::cumsum": 1,
}

#: The families that reach the width clause and the one that stops before it.
WIDTH_FAMILIES = ("aten::add", "aten::sub", "aten::cat", "aten::index")
SPELLING_FAMILY = "aten::cumsum"


def _clause_line(path, name, anchor, ret="return False"):
    """The line of the first bare `ret` at or after `anchor` inside `def name`.

    Derived from the file rather than written down, because this tree has moved
    these lines three times as clauses were inserted above them and a written-down
    line is stale the day a merge lands.  Read from the file and not from
    `inspect.getsource` so the constant describes the TREE while the traced line
    describes the RUN: if the run is not the tree, the comparison is a failing
    assertion and not an import error, which is the difference between a control
    that can fail and one that merely stops the suite from starting.
    """
    with open(path) as handle:
        src = handle.readlines()
    start = None
    for index, line in enumerate(src, start=1):
        if line.lstrip().startswith("def %s(" % name):
            start = index
            break
    if start is None:
        raise AssertionError("no def %s in %s" % (name, path))
    end = start
    while end < len(src) and not (
        src[end].startswith("def ") or src[end].startswith("class ")
    ):
        end += 1
    for line_number in range(start, end + 1):
        if anchor not in src[line_number - 1]:
            continue
        for j in range(line_number, end + 1):
            if src[j - 1].strip() == ret:
                return j
    raise AssertionError("no %r after %r in %s" % (ret, anchor, name))


HPART = hpart.__file__
OPS = hexagon_ops.__file__
WIDTH_GATE = _clause_line(
    HPART, "_verdict", "elif not result_dtype_is_emittable"
)
UNWIRED_GATE = _clause_line(
    HPART, "_verdict", "if node.target not in SUPPORTED_TARGETS"
)
GATHER_GATE = _clause_line(
    HPART, "_verdict", "gather_table(node, self.is_data_placeholder)"
)
GATHER_RANK_CLAUSE = _clause_line(
    OPS, "gather_table", "if table_value.dim() != 2", ret="return None"
)
GATHER_TWO_INDEX_CLAUSE = _clause_line(
    OPS, "_gather_operands", "or len(names) != 1", ret="return None, None"
)
CAT_WIDTH_CLAUSE = _clause_line(
    OPS, "cat_plan", "if result.dtype not in (torch.float16, torch.float32)",
    ret="return None",
)


def _target_name(node):
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        return schema.name
    return getattr(node.target, "__name__", None) or str(node.target)


class _Prologue(nn.Module):
    """The causal-mask and position prologue a Qwen3 export writes.

    Every family this file is about is here, in the shape the census counted:
    a `cumsum` over a bool row, the two two-axis gathers that read the mask out
    of it, an `arange` plus its lifted offset, a difference of two position
    slices, and the concatenation that puts the offset in front of the sequence
    length.  A graph built from anything else would be measuring a different
    graph than the one the census counted.
    """

    def forward(self, position_ids, query_len, activations):
        ones = torch.ones(1, query_len, dtype=torch.bool)
        cumulative = torch.cumsum(ones, dim=-1)
        rows = torch.arange(query_len, dtype=I64).unsqueeze(0).unsqueeze(-1)
        cols = torch.arange(query_len, dtype=I64).unsqueeze(0).unsqueeze(-1)
        mask_row = cumulative[rows, cols]
        mask_col = cumulative[cols, rows]
        arange = torch.arange(query_len, dtype=I64)
        # the three adds and the two subs of a stock Qwen3 prologue: the mask is
        # read two ways and the position ids are offset, differenced and
        # concatenated.  The fp16 relu is the one node here the partitioner
        # accepts, so the graph has a delegate for the refusals to sit beside.
        mask = mask_row + mask_col + mask_row
        positions = torch.cat(
            [
                position_ids[:, :1] - position_ids[:, :1],
                position_ids[:, -1:] - position_ids[:, :1],
                position_ids + arange,
            ],
            dim=-1,
        )
        return mask, positions, torch.relu(activations)


class _M(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _program(module, inputs):
    return to_edge(export(module, inputs), compile_config=CONFIG).exported_program()


def _support(program):
    return HexagonOperatorSupport(_data_placeholders(program), program)


def _trace_return_line(fn, *args):
    code = fn.__code__
    last = {}

    def tracer(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "return":
            last["line"] = frame.f_lineno
            return None
        return tracer

    sys.settrace(tracer)
    try:
        fn(*args)
    finally:
        sys.settrace(None)
    return last.get("line")


class _Trace:
    """Wrap `is_node_supported` and record both refusal lines, per node."""

    def __init__(self):
        self.original = hpart.HexagonOperatorSupport._verdict
        self.code = self.original.__code__
        self.hits = {}
        self.rewrite_verdict = None

    def _hit(self, line, node):
        self.hits[node.name] = line

    def __enter__(self):
        start = self.original.__code__.co_firstlineno
        out = []
        for offset, line in enumerate(
            textwrap.dedent(inspect.getsource(self.original)).splitlines()
        ):
            if line.strip() == "return False":
                line = line.replace(
                    "return False", "_hit(%d, node); return False" % (start + offset)
                )
            out.append(line)
        namespace = dict(vars(hpart))
        namespace["_hit"] = self._hit
        exec(compile("\n".join(out), "<rewrite>", "exec"), namespace)
        namespace.update(vars(hpart))
        # the rewrite only replaces the SOURCE; instrument B needs the code the
        # tree ships, which is the one captured before the rewrite.
        self.rewrite_verdict = namespace["_verdict"]
        hpart.HexagonOperatorSupport._verdict = self.rewrite_verdict
        return self

    def ask(self, support, node):
        """The verdict, the rewrite's clause, and the tracer's, for one node.

        Instrument B runs with the rewrite REMOVED, so it reads the function the
        tree ships and its line numbers are the file's own.  A pair that
        disagreed would mean one of them is measuring its own instrumentation.
        """
        hpart.HexagonOperatorSupport._verdict = self.rewrite_verdict
        verdict_a = support.is_node_supported({}, node)
        gate_a = self.hits.get(node.name)
        hpart.HexagonOperatorSupport._verdict = self.original
        lines = []

        def tracer(frame, event, arg):
            if frame.f_code is self.code:
                if event in ("line", "return"):
                    lines.append(frame.f_lineno)
                if event == "return":
                    return None
            return tracer

        sys.settrace(tracer)
        try:
            verdict_b = support.is_node_supported({}, node)
        finally:
            sys.settrace(None)
        return {
            "verdict": bool(verdict_a),
            "verdict_B": bool(verdict_b),
            "gate_A": gate_a,
            "gate_B": lines[-1] if lines else None,
        }

    def __exit__(self, *exc):
        hpart.HexagonOperatorSupport._verdict = self.original
        return False


def _classify(module, inputs):
    """{family: [row, ...]} over one exported graph, with both instruments."""
    program = _program(module, inputs)
    support = _support(program)
    rows = {}
    with _Trace() as trace:
        for node in program.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            family = _target_name(node)
            if family not in WATCHED:
                continue
            row = trace.ask(support, node)
            row["node"] = node.name
            row["dtype"] = str(hpart._dtype_of(node))
            rows.setdefault(family, []).append(row)
    return program, support, rows


def _own_gate(node, support):
    t = node.target
    if t in (exir_ops.edge.aten.add.Tensor, exir_ops.edge.aten.sub.Tensor):
        return "_broadcast_fits_dsp_limits", bool(
            hpart._broadcast_fits_dsp_limits(node)
        )
    if t is exir_ops.edge.aten.cat.default:
        return "cat_plan", _trace_return_line(hexagon_ops.cat_plan, node)
    if t is exir_ops.edge.aten.index.Tensor:
        return "gather_table", _trace_return_line(
            hexagon_ops.gather_table, node, support.is_data_placeholder
        )
    return None, None


def _commands(module, inputs):
    program = to_edge_transform_and_lower(
        export(module, inputs), partitioner=[HexagonPartitioner()], compile_config=CONFIG
    ).exported_program()
    gm = program.graph_module
    inner_names = set()
    out = []
    for call in gm.graph.nodes:
        if call.op != "call_function" or call.target is not DELEGATE:
            continue
        sub = gm.get_submodule(str(call.args[0].target))
        out.extend(c.type for c in read_blob(bytes(sub._processed_bytes))[1])
        inner_names.update(
            n.name for n in sub.original_module.graph_module.graph.nodes
        )
    return out, inner_names


def _prologue_inputs():
    return (torch.zeros(1, 4, dtype=I64), 4, torch.randn(2, 8, dtype=F16))


def _refusals(module, inputs):
    program, support, rows = _classify(module, inputs)
    return {k: v for k, v in rows.items() if any(not r["verdict"] for r in v)}


def test_the_nine_nodes_of_the_prologue_are_all_refused_and_name_one_clause():
    """The census counted five families; the partitioner names two clauses.

    Four families land on the result-width clause and one on the membership
    test, and the width clause is the one that reads the RESULT: every one of
    these four produces int64, which the arena does not hold at two bytes an
    element.  A partitioner that refused everything would satisfy this file, so
    every row below carries the control that would not.
    """
    program, support, rows = _classify(_Prologue(), _prologue_inputs())
    counts = {k: len(v) for k, v in rows.items()}
    assert counts == EXPECTED_BUCKETS, counts
    for family in WIDTH_FAMILIES:
        for row in rows[family]:
            assert row["verdict"] is False, (family, row)
            assert row["dtype"] == "torch.int64", (family, row)
            assert row["gate_A"] == WIDTH_GATE, (family, row)
            assert row["gate_B"] == WIDTH_GATE, (
                "the two instruments disagree",
                family,
                row,
            )
            assert row["verdict_B"] is False
    for row in rows[SPELLING_FAMILY]:
        assert row["verdict"] is False
        assert row["gate_A"] == UNWIRED_GATE
        assert row["gate_B"] == UNWIRED_GATE
    assert WIDTH_GATE != UNWIRED_GATE, "the two clauses collapsed into one"


def test_the_prologue_has_no_node_that_is_accepted_and_left_out_of_a_delegate():
    """Category (b), which a bucket count cannot see.

    A node the support predicate accepts can still reach no delegate, and that
    is a different size of work from a refusal.  Every node of the five families
    is either refused with a named clause or inside a delegate whose blob this
    file decodes, so the middle category is empty here by measurement rather
    than by assumption.
    """
    _cmds, inner_names = _commands(_Prologue(), _prologue_inputs())
    program, support, rows = _classify(_Prologue(), _prologue_inputs())
    for family, family_rows in rows.items():
        for row in family_rows:
            in_delegate = row["node"] in inner_names
            if row["verdict"]:
                assert in_delegate, (
                    "accepted and in no delegate: category (b)",
                    family,
                    row,
                )
            else:
                assert row["gate_A"] is not None, (family, row)
                assert not in_delegate, (family, row)
    # and the graph really does produce delegates, so the empty middle is a fact
    assert _cmds, "no delegate in the control graph, so this proves nothing"


def test_the_width_clause_reads_the_result_and_the_op_own_gate_would_have_agreed():
    """add and sub pass their own broadcast gate; only the width stops them.

    That is the difference between "this op cannot do this shape" and "this op
    cannot do int64 at all", and it is the one thing a gate firing early hides.
    `cat` is the width restated inside `cat_plan`, and `index.Tensor` has its
    own reasons, all measured here rather than asserted from a document.
    """
    program, support, rows = _classify(_Prologue(), _prologue_inputs())
    by_name = {n.name: n for n in program.graph_module.graph.nodes}
    seen = {}
    for family, family_rows in rows.items():
        for row in family_rows:
            if family in ("aten::add", "aten::sub"):
                name, value = _own_gate(by_name[row["node"]], support)
                assert name == "_broadcast_fits_dsp_limits"
                assert value is True, (
                    "the width is not the whole reason for this node",
                    row,
                )
            elif family == "aten::cat":
                name, line = _own_gate(by_name[row["node"]], support)
                assert name == "cat_plan"
                assert line == CAT_WIDTH_CLAUSE, (
                    "cat_plan refuses for a reason other than the width",
                    line,
                )
            elif family == "aten::index":
                name, line = _own_gate(by_name[row["node"]], support)
                assert name == "gather_table"
                assert line is not None
    # the width clause is a distinct function from the emitters' own raise, so
    # admitting a width at the gate without touching the emitters would convert
    # a working export into a failed one
    assert hexagon_ops.result_dtype_is_emittable(I64, exir_ops.edge.aten.add.Tensor) is False
    assert hexagon_ops.result_dtype_is_emittable(F16, exir_ops.edge.aten.add.Tensor) is True
    assert hexagon_ops.result_dtype_is_emittable(torch.bool, exir_ops.edge.aten.add.Tensor) is False
    assert hexagon_ops.result_dtype_is_emittable(torch.bool, exir_ops.edge.aten.gt.Tensor) is True
    assert (
        hexagon_ops.result_dtype_is_emittable(torch.bool, exir_ops.edge.aten.eq.Tensor)
        is False
    ), "the one-byte exemption is keyed on the target, not on the width alone"


class _Table(nn.Module):
    """A gather whose table is a registered buffer, not a method input.

    `gather_table` needs a table whose bytes this layer can read now: the
    kernel reads a table tiled at export, and a table that only exists at run
    time is refused at GATHER_GATE for that reason alone.  Holding the table in
    a buffer is what makes the rank control below measure rank and not
    visibility.
    """

    def __init__(self, t):
        super().__init__()
        self.register_buffer("t", t)

    def forward(self, *idx):
        return self.t[idx]


def test_the_gather_row_does_not_say_the_table_has_to_be_two_dimensional():
    """The reason the OP_SUPPORT row leaves out, isolated.

    The row for `aten.index.Tensor` says: an fp16 table, a constant table, one
    index on axis 0.  It never says the table must be a MATRIX.  A one-
    dimensional fp16 table with a single int32 index produces an fp16 result,
    so the width clause is not involved, and it is still refused -- by
    `table_value.dim() != 2`.  This is the gap between the row and the gate,
    and the two neighbouring controls are what make it a statement about rank
    rather than about a partitioner that refuses gathers.
    """
    two_d = _Table(torch.randn(4, 8, dtype=F16))
    idx32 = (torch.tensor([0, 2, 3], dtype=torch.int32),)
    one_d = _Table(torch.randn(8, dtype=F16))

    accepted, support, rows = _classify(two_d, idx32)
    index_rows = rows["aten::index"]
    assert [r["verdict"] for r in index_rows] == [True], rows
    commands, _inner = _commands(two_d, idx32)
    assert hexagon_ops.DSP_OP_SHARED_GATHER in commands, commands

    refused, support, rows = _classify(one_d, idx32)
    index_rows = rows["aten::index"]
    assert [r["verdict"] for r in index_rows] == [False], rows
    assert index_rows[0]["dtype"] == "torch.float16", index_rows
    assert index_rows[0]["gate_A"] == GATHER_GATE
    node = next(
        n
        for n in refused.graph_module.graph.nodes
        if n.op == "call_function" and _target_name(n) == "aten::index"
    )
    assert _trace_return_line(
        hexagon_ops.gather_table, node, support.is_data_placeholder
    ) == GATHER_RANK_CLAUSE

    # the two indices the Qwen3 nodes carry are the reason the row DOES state,
    # and the two reasons are different clauses
    two_idx = (
        torch.tensor([0, 2, 3], dtype=torch.int32),
        torch.tensor([1, 0, 2], dtype=torch.int32),
    )
    _p, support, rows = _classify(two_d, two_idx)
    assert [r["verdict"] for r in rows["aten::index"]] == [False], rows
    node = next(
        n
        for n in _program(two_d, two_idx).graph_module.graph.nodes
        if n.op == "call_function" and _target_name(n) == "aten::index"
    )
    assert (
        _trace_return_line(hexagon_ops.gather_table, node, support.is_data_placeholder)
        is not GATHER_RANK_CLAUSE
    )
    assert GATHER_RANK_CLAUSE != GATHER_TWO_INDEX_CLAUSE


def test_the_gather_gate_reads_the_result_width_and_not_the_index_width():
    """The width clause is the table's width here, and the row says fp16.

    An int64 INDEX is fine -- the runtime narrows it into the four-byte slot the
    kernel reads -- and an int64 TABLE is not, because a gather's result has the
    table's width and the result is what the clause at WIDTH_GATE reads.  The
    two are separated by holding one of them fixed, which is the control that
    says the clause is about the result rather than about int64 anywhere.
    """
    two_d = _Table(torch.randn(4, 8, dtype=F16))
    _p, _s, rows = _classify(two_d, (torch.tensor([0, 2, 3], dtype=torch.int64),))
    assert [r["verdict"] for r in rows["aten::index"]] == [True], rows

    wide = _Table(torch.arange(32, dtype=I64).reshape(4, 8))
    _p, _s, rows = _classify(wide, (torch.tensor([0, 2, 3], dtype=torch.int32),))
    index_rows = rows["aten::index"]
    assert [r["verdict"] for r in index_rows] == [False], rows
    assert index_rows[0]["dtype"] == "torch.int64", index_rows
    assert index_rows[0]["gate_A"] == WIDTH_GATE


def test_the_position_arithmetic_fp16_twin_emits_commands():
    """The negative control for every refusal above, in one place.

    Same ops, same shapes, fp16: `add`, `sub` and `cat` are accepted and the
    blob carries the two command types.  Without this a partitioner that refused
    everything would satisfy the other rows in this file, and the file would be
    a description of a backend that runs nothing.
    """

    def forward(x, y):
        return torch.cat([x - y, x + y], dim=-1)

    inputs = (torch.randn(2, 8, dtype=F16), torch.randn(2, 8, dtype=F16))
    _p, _s, rows = _classify(_M(forward), inputs)
    for family in ("aten::add", "aten::sub", "aten::cat"):
        assert rows[family], family
        assert all(r["verdict"] for r in rows[family]), (family, rows[family])
        assert all(r["gate_A"] is None for r in rows[family]), (family, rows[family])
    commands, _inner = _commands(_M(forward), inputs)
    assert hexagon_ops.DSP_OP_BINARY_ELEMENTWISE in commands, commands
    assert hexagon_ops.DSP_OP_RASTER_BLIT in commands, commands


def test_aten_cumsum_is_a_spelling_and_not_a_gate():
    """The tenth node, and what the two censuses do with it.

    The cumsum emitter is registered for `et_hexagon.cumsum`, the node
    `FuseCumsumPass` creates, and that pass is opt-in -- so a stock export
    produces `aten.cumsum`, which is not in the emitter table, and the
    partitioner stops at the membership test before any width is read.  Three
    widths are tried and all three stop there, which is what separates "the
    spelling" from "the width": an fp16 operand of the shape the fused emitter
    accepts is refused at the same line as a bool one.
    """
    assert exir_ops.edge.aten.cumsum.default not in SUPPORTED_TARGETS
    assert exir_ops.edge.aten.cumsum.default not in EMITTERS
    assert exir_ops.edge.et_hexagon.cumsum.default in SUPPORTED_TARGETS
    assert "aten::cumsum" not in hpart._emitted_families()
    assert "et_hexagon::cumsum" in hpart._emitted_families()

    shapes = {
        "bool [1,4] (what Qwen3 cumsums)": torch.ones(1, 4, dtype=torch.bool),
        "int64 [1,64]": torch.arange(64, dtype=I64).reshape(1, 64),
        "fp16 [2,64] (the fused emitter's shape)": torch.randn(2, 64, dtype=F16),
    }
    for label, x in shapes.items():
        _p, _s, rows = _classify(_M(lambda a: torch.cumsum(a, dim=-1)), (x,))
        row = rows["aten::cumsum"][0]
        assert row["verdict"] is False, (label, row)
        assert row["gate_A"] == UNWIRED_GATE, (label, row)
        assert row["gate_B"] == UNWIRED_GATE, (label, row)

    # and the width the spelling is hiding: the fused emitter's own predicate
    # refuses the operand Qwen3 actually has, so this node would stay portable
    # with the pass turned on.
    assert cumsum_is_emittable(torch.randn(2, 64, dtype=F16)) is True
    assert cumsum_is_emittable(torch.ones(1, 4, dtype=torch.bool)) is False
    assert cumsum_is_emittable(torch.arange(64, dtype=I64).reshape(1, 64)) is False


def test_both_partitioner_censuses_are_blind_to_aten_cumsum():
    """The counters carry no reason, and here they carry nothing at all.

    `_note_refused_target` counts only a target in SUPPORTED_TARGETS, and
    `_note_unwired_target` only a target whose FAMILY the table names; this
    node's family is `aten::cumsum` and the table's is `et_hexagon::cumsum`, so
    neither counter reports it.  A census that returned {} for the same node and
    a broken census would look the same, so the control below is a node each
    counter DOES report.
    """
    hpart.reset_refused_overload_census()
    hpart.reset_unwired_overload_census()
    program = _program(_M(lambda a: torch.cumsum(a, dim=-1)), (torch.ones(1, 4, dtype=torch.bool),))
    support = _support(program)
    for node in program.graph_module.graph.nodes:
        if node.op == "call_function":
            support.is_node_supported({}, node)
    assert hpart.refused_overload_census() == {}
    assert hpart.unwired_overload_census() == {}
    hpart.reset_refused_overload_census()
    hpart.reset_unwired_overload_census()

    program = _program(
        _M(lambda a: a.to(torch.int32)), (torch.randn(2, 3, dtype=F16),)
    )
    support = _support(program)
    for node in program.graph_module.graph.nodes:
        if node.op == "call_function":
            support.is_node_supported({}, node)
    assert hpart.refused_overload_census() != {}, "the counter reports nothing at all"
    hpart.reset_refused_overload_census()
    hpart.reset_unwired_overload_census()
