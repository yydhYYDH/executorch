"""The refusals of targets the emitter table *does* have.

`test_unwired_targets.py` covers the targets the table is missing. This covers the
other half of the same complaint: an op with an emitter looks supported, so a
reader of `OP_SUPPORT.md` expects it to run, and a geometry or a dtype the kernels
do not take sends it back to the portable kernels with nothing printed. The
spelling census found that shape three times -- a pool that wants exactly 64
channels, a grouped convolution whose group count no kernel here covers, and a
gather whose indices are int64 -- and each of them lowers to a graph that still
returns torch's answer, so nothing but a delegate count tells them apart.

Every row asserts both directions: the count the census reports, and the delegate
count the graph actually has. The second half matters, because two of these cases
produce a delegate anyway (holding the ops around the refused one).
"""

import operator

import pytest
import torch
import torch.nn as nn

from blob_interpreter import read_blob  # noqa: E402

from executorch.backends.hexagon import hexagon_ops as hexagon_ops
from executorch.backends.hexagon.partition import hexagon_partitioner
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

F16 = torch.float16
CONFIG = EdgeCompileConfig(_check_ir_validity=False)

MAX_POOL = "aten.max_pool2d_with_indices.default"
AVG_POOL = "aten.avg_pool2d.default"
CONVOLUTION = "aten.convolution.default"
EMBEDDING = "aten.embedding.default"
DIM_ORDER_COPY = "dim_order_ops._to_dim_order_copy.default"


@pytest.fixture(autouse=True)
def _fresh_censuses():
    reset_refused_overload_census()
    reset_unwired_overload_census()
    yield
    reset_refused_overload_census()
    reset_unwired_overload_census()


class _M(torch.nn.Module):
    """One callable, so a case is a plain module over its arguments."""

    def __init__(self, forward):
        super().__init__()
        self.forward_fn = forward

    def forward(self, *args):
        return self.forward_fn(*args)


def _program(model, inputs):
    return to_edge(export(_M(model), inputs), compile_config=CONFIG).exported_program()


def _support(program):
    """The support object `HexagonPartitioner.partition` builds.

    A bare `HexagonOperatorSupport()` has an empty `data_names`, so a parameter
    reaches it as an ordinary placeholder and every gate that wants a constant
    weight reads it as "not one". The pool and the gather disagree with the
    partitioner in opposite directions that way, which
    `test_a_parameter_is_only_a_constant_with_the_partitioners_data_names` pins.
    """
    return HexagonOperatorSupport(_data_placeholders(program))


def _refusals(model, inputs):
    """{target: count} for one graph, from a census that started empty."""
    reset_refused_overload_census()
    program = _program(model, inputs)
    support = _support(program)
    for node in program.graph_module.graph.nodes:
        if node.op == "call_function":
            support.is_node_supported({}, node)
    return refused_overload_census()


def _delegates(model, inputs):
    program = to_edge_transform_and_lower(
        export(_M(model), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _commands(model, inputs):
    """The command type of every command in every delegate, in graph order."""
    program = to_edge_transform_and_lower(
        export(_M(model), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    out = []
    for call in _delegates(model, inputs):
        module = program.graph_module.get_submodule(call.args[0].target)
        out.extend(
            command.type for command in read_blob(bytes(module._processed_bytes))[1]
        )
    return out


class _Conv(nn.Module):
    def __init__(self, cin, cout, groups=1, k=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=padding, groups=groups).half()

    def forward(self, x):
        return self.conv(x)


class _Embedding(nn.Module):
    """`nn.Embedding` whose index width comes from the caller, as it does in use.

    No `.to()` in the model: a cast would put a `_to_dim_order_copy` node in the
    graph next to the gather, and the row would then be about two ops.
    """

    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(64, 32)

    def forward(self, indices):
        return self.table(indices)


class _Pool(nn.Module):
    """`nn.MaxPool2d(2)` or `nn.AvgPool2d(2)` over `channels` channels at 16x16.

    Built as the module and not as `functional.max_pool2d`, so the row is the
    spelling a model has: the two go through the same gate, but the module is the
    one the spelling census measured, and it has no indexing node of its own.
    """

    def __init__(self, channels, kind="max", **kwargs):
        super().__init__()
        self.channels = channels
        pool = nn.MaxPool2d if kind == "max" else nn.AvgPool2d
        self.pool = pool(2, 2, **kwargs)

    def forward(self, x):
        return self.pool(x)


def _pool(channels, kind="max", **kwargs):
    return _Pool(channels, kind, **kwargs), (
        torch.randn(1, channels, 16, 16, dtype=F16),
    )


#: (label, model, inputs, {target: count}, delegates)
_ROWS = [
    (
        "max pool at 3 channels: the kernel wants exactly 64",
        *_pool(3),
        {MAX_POOL: 1},
        0,
    ),
    ("max pool at 32 channels: still not 64", *_pool(32), {MAX_POOL: 1}, 0),
    (
        "max pool at 65 channels: one block and a remainder",
        *_pool(65),
        {MAX_POOL: 1},
        0,
    ),
    ("max pool at 64 channels", *_pool(64), {}, 1),
    ("average pool at 3 channels", *_pool(3, kind="avg"), {AVG_POOL: 1}, 0),
    ("average pool at 64 channels", *_pool(64, kind="avg"), {}, 1),
    (
        "max pool at 64 channels with dilation: the kernel has none",
        *_pool(64, dilation=2),
        {MAX_POOL: 1},
        0,
    ),
    (
        "max pool at 64 channels with ceil_mode",
        *_pool(64, ceil_mode=True),
        {MAX_POOL: 1},
        0,
    ),
    (
        "a max pool whose indices are read, at 64 channels",
        lambda x: torch.nn.functional.max_pool2d(x, 2, 2, return_indices=True)[1].to(
            F16
        ),
        (torch.randn(1, 64, 16, 16, dtype=F16),),
        {MAX_POOL: 1, DIM_ORDER_COPY: 1},
        0,
    ),
    (
        "a grouped convolution with two output channels per group",
        _Conv(16, 32, groups=16),
        (torch.randn(1, 16, 16, 16, dtype=F16),),
        {CONVOLUTION: 1},
        0,
    ),
    (
        "a grouped convolution with four input channels per group",
        _Conv(16, 16, groups=4),
        (torch.randn(1, 16, 16, 16, dtype=F16),),
        {CONVOLUTION: 1},
        0,
    ),
    (
        "a depthwise convolution: groups == in == out",
        _Conv(16, 16, groups=16),
        (torch.randn(1, 16, 16, 16, dtype=F16),),
        {},
        1,
    ),
    (
        "a plain convolution at 3 input channels",
        _Conv(3, 16),
        (torch.randn(1, 3, 16, 16, dtype=F16),),
        {},
        1,
    ),
    (
        "an embedding with int32 indices",
        _Embedding(),
        (torch.zeros(1, 8, dtype=torch.int32),),
        {},
        1,
    ),
]


@pytest.mark.parametrize(
    "label, model, inputs, refusals, delegates", _ROWS, ids=[row[0] for row in _ROWS]
)
def test_each_refusal_row_matches_its_recorded_census(
    label, model, inputs, refusals, delegates
):
    """Both halves per row: what the census says, and what the graph did.

    A row that stops refusing goes red on its expected dict; a row that starts
    refusing a shape it used to run goes red on its delegate count. The second is
    the direction that matters, since the census cannot tell a wrong acceptance
    from a right one.
    """
    assert _refusals(model, inputs) == refusals, f"{label}: the census moved"
    assert len(_delegates(model, inputs)) == delegates, f"{label}: the graph moved"


def test_a_delegate_can_hold_none_of_the_op_that_was_refused():
    """The illusion the census exists to break, in one graph.

    `MaxPool2d` into a graph that keeps going gets a delegate -- of the ops
    *after* the pool. Reading the delegate count alone says "part of this model
    runs on the DSP", which is true and tells a reader nothing about the pool.
    """
    inputs = (torch.randn(1, 32, 16, 16, dtype=F16),)

    def forward(x):
        return torch.relu(torch.nn.functional.max_pool2d(x, 2, 2)[0] + 1.0)

    assert _refusals(forward, inputs) == {MAX_POOL: 1}
    assert len(_delegates(forward, inputs)) == 1  # the add and the relu, not the pool
    assert _commands(forward, inputs) == [
        hexagon_ops.DSP_OP_BINARY_ELEMENTWISE,
        hexagon_ops.DSP_OP_UNARY,
    ]


def test_both_index_widths_emit_the_same_gather_command():
    """Same table, same graph, two index widths, and the command stream agrees.

    The int64 graph used to be refused here, which is the thing this change took
    away: the command is the same SHARED_GATHER in both, and what differs is one
    param the DSP never reads -- the width the caller's tensor has, which is
    what the host narrows with. A command that differed would mean the index slot
    differed, and the slot is four bytes an element in both.
    """

    def graph(model):
        def forward(indices):
            return torch.amax(model(indices) + 1.0, dim=1)

        return forward

    expect = [
        hexagon_ops.DSP_OP_SHARED_GATHER,
        hexagon_ops.DSP_OP_BINARY_ELEMENTWISE,
        hexagon_ops.DSP_OP_REDUCTION,
    ]
    wide = _commands(graph(_Embedding()), (torch.zeros(1, 8, dtype=torch.int64),))
    narrow = _commands(graph(_Embedding()), (torch.zeros(1, 8, dtype=torch.int32),))
    assert wide == narrow == expect
    assert _refusals(graph(_Embedding()), (torch.zeros(1, 8, dtype=torch.int64),)) == {}
    assert _refusals(graph(_Embedding()), (torch.zeros(1, 8, dtype=torch.int32),)) == {}


def test_an_op_with_no_emitter_is_neither_a_gap_nor_a_refusal():
    """`aten.erf` has no emitter and no family in the table: neither counter fires.

    This is the boundary of what the two censuses can say. The unwired counter
    wants a family the table speaks for, and the refused counter wants a target in
    the table; an op with neither is invisible to both, and is exactly the case
    `OP_SUPPORT.md` already answers by not listing it.
    """
    assert _refusals(lambda x: torch.erf(x), (torch.randn(8, dtype=F16),)) == {}
    reset_unwired_overload_census()
    program = _program(lambda x: torch.erf(x), (torch.randn(8, dtype=F16),))
    support = _support(program)
    for node in program.graph_module.graph.nodes:
        if node.op == "call_function":
            support.is_node_supported({}, node)
    assert unwired_overload_census() == {}
    assert _delegates(lambda x: torch.erf(x), (torch.randn(8, dtype=F16),)) == []


def test_an_unwired_target_is_not_counted_as_a_refusal():
    """`2.0 ** x` is the unwired case; the two censuses must not overlap."""
    forward = lambda x: 2.0**x  # noqa: E731
    inputs = (torch.randn(8, dtype=F16),)
    assert _refusals(forward, inputs) == {}
    reset_unwired_overload_census()
    program = _program(forward, inputs)
    support = _support(program)
    for node in program.graph_module.graph.nodes:
        if node.op == "call_function":
            support.is_node_supported({}, node)
    assert unwired_overload_census() == {"aten::pow": {"aten.pow.Scalar": 1}}


def test_the_census_only_holds_targets_the_table_has():
    """The invariant the two counters split on, checked against the table itself.

    A `getitem` is in the table and has no schema, and is left out: it is a
    lowering artifact of the node it reads, which is itself in the census.
    """
    names = {
        getattr(target, "__name__", str(target)) for target in hexagon_ops.EMITTERS
    }
    counted = set()
    for _label, model, inputs, _expected_refusals, _expected_delegates in _ROWS:
        counted |= set(_refusals(model, inputs))
    assert counted, "no row refused anything, so this proves nothing"
    assert counted <= names, counted - names
    assert not any(
        name in counted for name in ("getitem", getattr(operator.getitem, "__name__"))
    )


def test_the_debug_line_names_the_target_and_the_node(caplog):
    """The human surface: which op, and which node in the graph."""
    caplog.set_level(10, logger=hexagon_partitioner.__name__)
    _refusals(*_pool(3))
    assert MAX_POOL in caplog.text
    assert "refused" in caplog.text


def test_nothing_is_logged_at_the_default_level(caplog):
    """An export must not start printing because of this."""
    caplog.set_level(30, logger=hexagon_partitioner.__name__)
    for _label, model, inputs, _expected_refusals, _expected_delegates in _ROWS:
        _refusals(model, inputs)
    assert caplog.text == ""


def test_the_diagnostic_cannot_change_a_verdict(monkeypatch):
    """The control: with both counters replaced by no-ops, nothing moves.

    The reporting runs where the support check has already decided the node stays
    off the DSP, and this is the in-process statement of that: the verdicts and
    the delegates are identical with the counting removed.
    """
    corpus = [
        ("pool 3", *_pool(3)),
        ("pool 64", *_pool(64)),
        ("avg pool 3", *_pool(3, kind="avg")),
        (
            "conv grouped",
            _Conv(16, 32, groups=16),
            (torch.randn(1, 16, 16, 16, dtype=F16),),
        ),
        (
            "conv depthwise",
            _Conv(16, 16, groups=16),
            (torch.randn(1, 16, 16, 16, dtype=F16),),
        ),
        ("embedding int64", _Embedding(), (torch.zeros(1, 8, dtype=torch.int64),)),
        ("embedding int32", _Embedding(), (torch.zeros(1, 8, dtype=torch.int32),)),
        ("erf", lambda x: torch.erf(x), (torch.randn(8, dtype=F16),)),
        ("pow scalar", lambda x: 2.0**x, (torch.randn(8, dtype=F16),)),
        (
            "max.dim values",
            lambda x: torch.max(x, dim=1).values,
            (torch.randn(2, 3, 4, dtype=F16),),
        ),
    ]

    def fingerprint():
        out = []
        for label, model, inputs in corpus:
            program = _program(model, inputs)
            support = _support(program)
            out.append(
                (
                    len(_delegates(model, inputs)),
                    [
                        support.is_node_supported({}, node)
                        for node in program.graph_module.graph.nodes
                        if node.op == "call_function"
                    ],
                )
            )
        return out

    before = fingerprint()
    counted = refused_overload_census()
    unwired = unwired_overload_census()
    assert counted, "the control corpus refused nothing, so this proves nothing"
    assert unwired, "the control corpus has no unwired target, so this proves little"

    monkeypatch.setattr(hexagon_partitioner, "_note_refused_target", lambda node: None)
    monkeypatch.setattr(hexagon_partitioner, "_note_unwired_target", lambda node: None)
    assert fingerprint() == before
    assert refused_overload_census() == counted
    assert unwired_overload_census() == unwired


def test_a_parameter_is_only_a_constant_with_the_partitioners_data_names():
    """Why every test here builds its support object the way `partition` does.

    With an empty `data_names` a parameter reaches the check as an ordinary
    placeholder, so `is_data_placeholder` says no and every gate that needs a
    weight it can read refuses. A convolution and a gather are both measured with
    a bare `HexagonOperatorSupport()` in this tree's other tests -- fine there,
    since no row of theirs needs a constant -- and would have measured a refusal
    here that the partitioner does not make.
    """
    conv = _Conv(3, 16)
    inputs = (torch.randn(1, 3, 16, 16, dtype=F16),)
    program = _program(conv, inputs)
    node = next(
        node for node in program.graph_module.graph.nodes if node.op == "call_function"
    )
    assert HexagonOperatorSupport().is_node_supported({}, node) is False
    assert _support(program).is_node_supported({}, node) is True
    assert _refusals(conv, inputs) == {}
    assert len(_delegates(conv, inputs)) == 1

    # The gate reads no weight, so the two support objects agree about a pool --
    # which is why the pool rows would have measured the same either way.
    pool, pool_inputs = _pool(64)
    pool_node = next(
        node
        for node in _program(pool, pool_inputs).graph_module.graph.nodes
        if node.op == "call_function"
    )
    assert HexagonOperatorSupport().is_node_supported({}, pool_node) is True
    assert (
        _support(_program(pool, pool_inputs)).is_node_supported({}, pool_node) is True
    )


def test_a_lowering_counts_each_refused_node_once():
    """The count is nodes, not calls into the support check.

    `to_edge_transform_and_lower` asks the predicate about the same node twice
    (measured: this graph reports 2 before the dedupe), so a real lowering and the
    `_refusals` helper above -- one pass over the graph -- disagreed by that
    factor, and the number a caller reads after a lowering is the inflated one.
    """
    _delegates(*_pool(3))
    assert refused_overload_census() == {MAX_POOL: 1}


def test_two_nodes_of_the_same_target_count_twice():
    """The dedupe is by node and not by target: two refusals are two."""
    pool = _Pool(3)
    inputs = (torch.randn(1, 3, 16, 16, dtype=F16),)
    _delegates(_M(lambda x: pool(x) + pool(x * 0.5)), inputs)
    assert refused_overload_census() == {MAX_POOL: 2}
