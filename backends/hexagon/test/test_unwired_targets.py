# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The silent-gap diagnostic in the partitioner.

An op that is missing from `hexagon_ops.EMITTERS` falls back without a word: the
model still computes the right answer on the portable kernels, and the only
symptom is a delegate that is one op shorter than it looks. `torch.mean(x)` sat
that way while `torch.mean(x, dim=1)` was delegated, and the census in
`test_overload_census.py` found four more of the same kind by hand.

These are the mechanism's own tests, in both directions: a target that belongs to
a family the emitter table speaks for has to be visible, an op outside those
families has to stay out of the way, and the counting has to be provably unable
to change an answer.
"""

import logging

import pytest
import torch

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_backend import SUPPORTED_TARGETS
from executorch.backends.hexagon.partition import hexagon_partitioner
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
    reset_unwired_overload_census,
    unwired_overload_census,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

F16 = torch.float16

_X = torch.randn(8, dtype=F16)
_XY = (_X, _X)


class _M(torch.nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.forward_fn = forward

    def forward(self, *args):
        return self.forward_fn(*args)


@pytest.fixture(autouse=True)
def _empty_census():
    """Each test counts its own nodes, not the process's."""
    reset_unwired_overload_census()
    yield
    reset_unwired_overload_census()


def _lower(forward, inputs):
    return to_edge_transform_and_lower(
        export(_M(forward), inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return len(
        [
            node
            for node in program.graph_module.graph.nodes
            if node.target is torch.ops.higher_order.executorch_call_delegate
        ]
    )


def _shape():
    """The census as families to target names, without the repeat counts.

    A count is how many times the partitioner asked about a node rather than how
    many nodes there are, so it is not the part these tests state.
    """
    return {
        family: sorted(targets) for family, targets in unwired_overload_census().items()
    }


#: Expressions whose graph reaches an op that is absent from the emitter table
#: while a sibling overload is in it, with the family the census has to name.
_GAPS = [
    (
        "fmod with a scalar",
        lambda a, b: torch.fmod(a, 2.0),
        _XY,
        "aten::fmod",
        "aten.fmod.Scalar",
    ),
    (
        "roll, which fmods twice",
        lambda a, b: torch.roll(a, 1),
        _XY,
        "aten::fmod",
        "aten.fmod.Scalar",
    ),
    (
        "a scalar base raised to a tensor",
        lambda a, b: 2.0**a,
        _XY,
        "aten::pow",
        "aten.pow.Scalar",
    ),
    (
        "a tensor raised to a tensor",
        lambda a, b: a**b,
        _XY,
        "aten::pow",
        "aten.pow.Tensor_Tensor",
    ),
    (
        "truncated division",
        lambda a, b: torch.div(a, 2.0, rounding_mode="trunc"),
        _XY,
        "aten::div",
        "aten.div.Tensor_mode",
    ),
    (
        "clamp against a tensor bound",
        lambda a, b: torch.clamp(a, min=b),
        _XY,
        "aten::clamp",
        "aten.clamp.Tensor",
    ),
    (
        "a dtype reinterpretation",
        lambda a, b: a.view(torch.float32),
        _XY,
        "aten::view_copy",
        "aten.view_copy.dtype",
    ),
    (
        "rms_norm's decomposition",
        lambda a, b: torch.rms_norm(a, [8]),
        _XY,
        "aten::add",
        "aten.add.Scalar",
    ),
]


def test_a_sibling_overload_that_is_absent_is_counted():
    """The case this mechanism exists for: family in the table, target not in it.

    `aten::max` is a family the DSP runs -- max.default and max.dim both have
    emitters -- so a max.other node at the support check is a forgotten overload
    rather than an op this backend has decided against.
    """
    graph = torch.fx.Graph()
    node = graph.call_function(exir_ops.edge.aten.max.other, ())
    assert HexagonOperatorSupport().is_node_supported({}, node) is False
    assert _shape() == {"aten::max": ["aten.max.other"]}


def test_an_op_outside_the_emitted_families_is_not_counted():
    """The other direction, so the census stays a list of suspects.

    Nothing in the table speaks for `aten::full`. Its absence is a decision about
    the op, not a forgotten overload, and counting it would bury the real gaps
    under every op the backend does not implement.
    """
    graph = torch.fx.Graph()
    node = graph.call_function(exir_ops.edge.aten.full.default, ([2, 3], 1.0))
    assert HexagonOperatorSupport().is_node_supported({}, node) is False
    assert _shape() == {}


@pytest.mark.parametrize(
    "label,forward,inputs,family,target", _GAPS, ids=[row[0] for row in _GAPS]
)
def test_a_real_graph_names_the_gap_it_hits(label, forward, inputs, family, target):
    """The census, over graphs an exporter actually produces.

    Row by row rather than as a total, because the failure worth catching is a
    target that has quietly gained or lost its emitter, and a total would hide
    which one moved.
    """
    _lower(forward, inputs)
    names = [getattr(key, "__name__", key) for key in hexagon_ops.EMITTERS]
    assert target not in names, f"{target} now has an emitter; the row is stale"
    assert _shape() == {family: [target]}


def test_the_cost_of_a_gap_is_a_whole_graph_on_the_host():
    """Two of the rows above cost the whole model its delegate.

    The census reports a missing emitter, and this is what that means at the end
    of the pipeline: `2.0 ** a` and a tensor-bounded clamp produce no delegate at
    all, so a model built out of them runs every node on the portable kernels.
    """
    for forward in (lambda a, b: 2.0**a, lambda a, b: torch.clamp(a, min=b)):
        assert _delegates(_lower(forward, _XY)) == 0


def test_the_debug_line_names_the_target_and_the_family(caplog):
    """The human surface: one line, on this module's logger, at debug level."""
    graph = torch.fx.Graph()
    node = graph.call_function(exir_ops.edge.aten.max.other, ())
    with caplog.at_level(
        logging.DEBUG,
        logger="executorch.backends.hexagon.partition.hexagon_partitioner",
    ):
        HexagonOperatorSupport().is_node_supported({}, node)
    assert "aten.max.other" in caplog.text
    assert "aten::max" in caplog.text


def test_nothing_is_logged_at_the_default_level(caplog):
    """An export nobody is debugging pays no formatting cost."""
    with caplog.at_level(logging.WARNING):
        for _, forward, inputs, _, _ in _GAPS:
            _lower(forward, inputs)
    assert caplog.text == ""


def test_the_diagnostic_cannot_change_a_verdict(monkeypatch):
    """The control: with the counting replaced by a no-op, nothing moves.

    The mechanism runs only where the support check has already decided the node
    stays off the DSP, so the strongest statement available in-process is that
    the per-node verdicts and the delegates are identical with it removed -- which
    is what this measures over a corpus of graphs, partitioner probes and all.
    """
    corpus = [
        (lambda a: torch.fmod(a, 2.0), (torch.randn(8, dtype=F16),)),
        (lambda a: 2.0**a, (torch.randn(8, dtype=F16),)),
        (lambda a: torch.max(a, dim=1).values, (torch.randn(2, 3, 4, dtype=F16),)),
        (
            lambda a: torch.nn.functional.layer_norm(a, [6]),
            (torch.randn(4, 6, dtype=F16),),
        ),
        (
            lambda a: torch.nn.functional.max_pool2d(a, 2, 2),
            (torch.randn(1, 64, 8, 8, dtype=F16),),
        ),
        (lambda a: torch.mean(a), (torch.randn(2, 3, 4, dtype=F16),)),
        (lambda a: torch.erf(a), (torch.randn(8, dtype=F16),)),
    ]

    def verdicts():
        out = []
        for forward, inputs in corpus:
            program = _lower(forward, inputs)
            support = HexagonOperatorSupport(_data_placeholders(program))
            out.append(
                (
                    _delegates(program),
                    [
                        support.is_node_supported({}, node)
                        for node in program.graph_module.graph.nodes
                        if node.op == "call_function"
                    ],
                )
            )
        return out

    before = verdicts()
    counted = _shape()
    monkeypatch.setattr(hexagon_partitioner, "_note_unwired_target", lambda node: None)
    after = verdicts()
    assert before == after
    assert _shape() == counted, "the no-op was not the one being called"
    assert counted, "no gap was hit, so this compared nothing"


def test_the_gap_the_census_found_is_closed():
    """`torch.max(x, dim).values`, the one gap of the seven that this batch shut.

    The target now has an emitter and the values reader now delegates, which is
    what the census stopped reporting: it was `aten.max.dim` under `aten::max`
    before, and rewiring a sibling to it has to fail here.
    """
    assert hexagon_ops.MAX_DIM in SUPPORTED_TARGETS
    program = _lower(
        lambda a: torch.max(a, dim=1).values, (torch.randn(2, 3, 4, dtype=F16),)
    )
    assert _delegates(program) == 1
    assert _shape() == {}
