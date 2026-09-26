# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The census corpus, in the tree, and the unwired census read off it.

unwired_overload_census() is a per-process accumulator that starts empty. Nothing
in this tree filled it, so a fresh worktree reported {} and the number of
unwired targets had to be inherited from a run whose corpus was not in the tree
either. model_census_corpus.py is that corpus -- nineteen graphs, moved out of a
scratch directory -- and these tests lower every one of them through the real
partitioner, so the census is re-derived here rather than quoted from a report.

The count that comes out is a zero, and a zero from a scan is exactly what a
broken scan returns too. Three things pin that this one is real.

* The positive control lowers a graph the tree already pins an unwired node in,
  and requires the census to name it, so a driver that finds nothing cannot pass
  this file.
* The parts are added to the total here rather than asserted beside a header, and
  the same total is predicted a second time by walking the lowered graphs, so the
  two numbers come from different code and have to agree.
* The reason the corpus reads zero is stated as a checkable property rather than
  as prose: every target the corpus leaves off the emitter table is a
  whole-family absence, which _note_unwired_target declines by construction. The
  day one of them grows a wired sibling this fails and names it, instead of the
  corpus quietly going on being evidence for a zero that no longer holds.

Host tier: export, to_edge, the support predicate. No kernel runs, hexagon-sim
does not run, and nothing here says anything about a phone DSP. The row census
reads declared verdicts, which is a table and not a measurement of the partitioner.
"""

import collections

import pytest
import torch

import model_census_corpus as corpus
import test_overload_census as rows1
import test_overload_census2 as rows2
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
    SUPPORTED_TARGETS,
    _emitted_families,
    _schema_name,
    reset_unwired_overload_census,
    unwired_overload_census,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

F16 = torch.float16
_CFG = EdgeCompileConfig(_check_ir_validity=False)
_EMITTER_NAMES = {getattr(key, "__name__", str(key)) for key in hexagon_ops.EMITTERS}
_VERDICTS = ("wired", "refused", "unwired", "accepted")


class _M(torch.nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.forward_fn = forward

    def forward(self, *args):
        return self.forward_fn(*args)

def _lower(module, inputs, passes=None):
    return to_edge_transform_and_lower(
        export(module, inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=_CFG,
        transform_passes=passes,
    ).exported_program()

def _delegates(program):
    return sum(
        1
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    )

@pytest.fixture(scope="module")
def lowered():
    """The whole corpus, lowered once, with the census left as the tree left it."""
    reset_unwired_overload_census()
    graphs = {}
    for name, builder in corpus.REGISTRY.items():
        module, inputs = builder()
        graphs[name] = _lower(module, inputs, corpus.PASSES.get(name))
    census = unwired_overload_census()
    per_family = {family: sum(t.values()) for family, t in census.items()}
    return {
        "graphs": graphs,
        "census": census,
        "per_family": per_family,
        "total": sum(per_family.values()),
    }

def test_the_corpus_is_the_nineteen_geometries():
    """A geometry that cannot be built is a census that cannot be re-derived."""
    assert len(corpus.REGISTRY) == 19
    assert set(corpus.REGISTRY) == set(corpus.DESCRIPTIONS)
    assert set(corpus.PASSES) <= set(corpus.REGISTRY)
    for name, builder in corpus.REGISTRY.items():
        module, inputs = builder()
        assert module is not None, name
        assert isinstance(inputs, tuple) and inputs, name
        assert all(hasattr(t, "shape") for t in inputs), name

@pytest.mark.parametrize("name", sorted(corpus.REGISTRY))
def test_every_geometry_reaches_the_dsp(lowered, name):
    """The control on the corpus itself: a graph that delegated nothing proved nothing.

    A lowering can form no delegate without raising, so a corpus of graphs that
    all fell back whole would satisfy every census below vacuously.
    """
    assert _delegates(lowered["graphs"][name]) > 0, name

def test_the_driver_sees_an_unwired_node_it_is_built_to_see():
    """The positive control: same driver, a graph known to carry a sibling gap.

    torch.rms_norm(a, [8]) decomposes to an aten.add.Scalar beside a wired
    aten.add.Tensor, which is the one shape _note_unwired_target counts and which
    test_unwired_targets.py already pins by name. If this returns nothing, then
    the zero the corpus produces is the zero of a broken instrument.
    """
    reset_unwired_overload_census()
    x = torch.randn(8, dtype=F16)
    _lower(_M(lambda a: torch.rms_norm(a, [8])), (x,))
    census = unwired_overload_census()
    reset_unwired_overload_census()
    assert census, "the driver counted nothing on a graph with a known gap in it"
    assert census["aten::add"]["aten.add.Scalar"] == 1
    assert sum(sum(t.values()) for t in census.values()) == 1

def test_the_census_is_the_number_the_partitioner_produced(lowered):
    """The corpus census, and the parts added up to it rather than quoted beside it."""
    census = lowered["census"]
    assert isinstance(census, dict)
    assert lowered["total"] == sum(lowered["per_family"].values())
    assert lowered["total"] == sum(sum(t.values()) for t in census.values())
    for family, targets in census.items():
        assert lowered["per_family"][family] == sum(targets.values())
        assert all(count > 0 for count in targets.values())

def test_the_census_agrees_with_the_graphs_it_was_taken_from(lowered):
    """A second derivation of the same number, from the graphs and not the counter.

    The rule _note_unwired_target applies is a target absent from the emitter table
    whose schema family is present, so the same count can be predicted by walking
    the lowered graphs. The counter and this walk are different code reading
    different objects, and they have to come out equal: that is the check which
    says the zero belongs to the corpus and not to the counter.
    """
    families = _emitted_families()
    predicted = collections.Counter()
    off_table = 0
    for program in lowered["graphs"].values():
        for node in program.graph_module.graph.nodes:
            if node.op != "call_function" or node.target in SUPPORTED_TARGETS:
                continue
            off_table += 1
            if _schema_name(node.target) in families:
                predicted[_schema_name(node.target)] += 1
    assert off_table > 0, "no off-table node at all: the walk found nothing to check"
    assert sum(predicted.values()) == lowered["total"]
    assert dict(predicted) == lowered["per_family"]

def test_the_corpus_reads_zero_because_every_gap_is_a_whole_family(lowered):
    """The reason, as a property the tree can fail on rather than as a sentence.

    A node is counted only when a sibling overload of the same schema has an
    emitter. A family with no emitter at all is a decision about the op, so the
    census declines it by construction, and these nineteen geometries are full of
    them: aten::full, aten::full_like, aten::arange, aten::eq, aten::le,
    aten::logical_not, aten::any, aten::_native_batch_norm_legit_no_training.
    The moment one of them grows a wired sibling this fails, which is the day the
    corpus stops being evidence for a zero.
    """
    families = _emitted_families()
    whole_family = {}
    for program in lowered["graphs"].values():
        for node in program.graph_module.graph.nodes:
            if node.op != "call_function" or node.target in SUPPORTED_TARGETS:
                continue
            family = _schema_name(node.target)
            if family not in families:
                name = getattr(node.target, "__name__", str(node.target))
                whole_family.setdefault(family, set()).add(name)
    assert whole_family, "the corpus has no off-table node left to explain"
    for family, names in whole_family.items():
        assert family not in families, (family, sorted(names))

def _declared():
    """{verdict: count} over every row table, and the targets the unwired rows name."""
    counts = collections.Counter()
    targets = collections.Counter()
    for table in (rows1._ROWS, rows2._ROWS, rows2._QUANTIZED_ROWS):
        for row in table:
            for name, verdict in row[-1]:
                counts[verdict] += 1
                if verdict == "unwired":
                    targets[name] += 1
    return counts, targets

def test_the_row_census_adds_up_to_its_own_total():
    """Parts to total, done by the addition. A decomposition nobody summed is a list."""
    counts, targets = _declared()
    verdicts = sum(
        len(row[-1])
        for table in (rows1._ROWS, rows2._ROWS, rows2._QUANTIZED_ROWS)
        for row in table
    )
    assert set(counts) <= set(_VERDICTS), dict(counts)
    assert sum(counts.values()) == verdicts, (dict(counts), verdicts)
    assert sum(targets.values()) == counts["unwired"]
    assert len(targets) <= counts["unwired"]

def test_every_unwired_row_is_really_absent_from_the_table():
    """A row that says unwired while EMITTERS holds the name is a stale row."""
    _, targets = _declared()
    stale = {name: n for name, n in targets.items() if name in _EMITTER_NAMES}
    assert not stale, stale
