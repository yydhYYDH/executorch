"""The two gates a boolean node meets, and the one that is not a fallback.

aten.logical_not and aten.any are refused today at the partitioner's first
gate, which reads like an absent emitter and is only half of it. Both gates are
pinned here so that the claim cannot be carried by the first one alone, and the
third is pinned because it is the only one that is an export failure rather
than a fallback.
"""

import os
import pathlib
import sys

import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[1]))

import executorch.backends.hexagon.partition.hexagon_partitioner as part_mod  # noqa: E402
import hexagon_backend  # noqa: E402
import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonOperatorSupport,
)
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402

LOGICAL_NOT = exir_ops.edge.aten.logical_not.default
LOGICAL_AND = exir_ops.edge.aten.logical_and.default
ANY_DIM = exir_ops.edge.aten.any.dim
ADD = exir_ops.edge.aten.add.Tensor


def _bool_chain():
    """The shapes the census counts come from, as fx nodes with their values.

    eq.Scalar -> logical_not -> any.dim -> logical_not -> where.self is what a
    masked scaled_dot_product_attention decomposes into, and the two boolean
    nodes measured here are one logical_not over the mask and one over the
    per-row flag that any.dim produces.
    """
    graph = torch.fx.Graph()
    source = graph.placeholder("eq_out")
    source.meta["val"] = torch.empty(1, 2, 3, 3, dtype=torch.bool)
    first = graph.call_function(LOGICAL_NOT, args=(source,))
    first.meta["val"] = torch.empty(1, 2, 3, 3, dtype=torch.bool)
    reduced = graph.call_function(ANY_DIM, args=(first, [-1], True))
    reduced.meta["val"] = torch.empty(1, 2, 3, 1, dtype=torch.bool)
    second = graph.call_function(LOGICAL_NOT, args=(reduced,))
    second.meta["val"] = torch.empty(1, 2, 3, 1, dtype=torch.bool)
    return [first, reduced, second]


def _add(result_dtype):
    """aten.add.Tensor, a target that is in the table, at a chosen result width."""
    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    left.meta["val"] = torch.empty(2, 3, dtype=torch.float16)
    right.meta["val"] = torch.empty(2, 3, dtype=torch.float16)
    node = graph.call_function(ADD, args=(left, right))
    node.meta["val"] = torch.empty(2, 3, dtype=result_dtype)
    return node


def test_the_supported_set_is_the_emitter_table_so_a_target_cannot_be_added_alone():
    """There is no separate membership list to widen, which is the first gate.

    SUPPORTED_TARGETS is the same object as EMITTERS, so "the partitioner does
    not know this target" and "no emitter writes a command for it" are one fact
    and not two. A refusal at that gate is therefore never evidence that a
    second gate would have accepted the node.
    """
    assert hexagon_backend.SUPPORTED_TARGETS is hexagon_backend.EMITTERS
    assert part_mod.SUPPORTED_TARGETS is hexagon_backend.EMITTERS
    # logical_not is wired now, so the example of a target that is not has moved
    # to a neighbour in the same family that still is not; the assertion under
    # test is the identity above, not the choice of example.
    assert LOGICAL_AND not in hexagon_backend.SUPPORTED_TARGETS
    assert ANY_DIM not in hexagon_backend.SUPPORTED_TARGETS

    support = HexagonOperatorSupport()
    first, reduced, second = _bool_chain()
    # The two negations are wired, so they are admitted. any.dim is not in the
    # table, and it is refused there rather than anywhere later -- which is the
    # point of the identity above: nothing downstream ever saw the node.
    assert support.is_node_supported(None, first)
    assert reduced.target not in hexagon_backend.SUPPORTED_TARGETS
    assert not support.is_node_supported(None, reduced)
    assert support.is_node_supported(None, second)


def test_the_width_gate_reads_the_result_so_a_boolean_node_is_refused_in_both_tables(
    monkeypatch,
):
    """Putting the target in the table is not enough, and the operands do not help.

    The gate is on the result's dtype, not the operand's, so an fp16 mask that
    any.dim reduces to a bool is refused for the same reason a bool operand is,
    and the refusal survives the table entry that the first gate wanted. The
    control is the same target with an fp16 result, so a support object that
    refused everything could not make this pass.
    """
    support = HexagonOperatorSupport()
    assert support.is_node_supported(None, _add(torch.float16))
    assert not support.is_node_supported(None, _add(torch.bool))

    # any.dim only: a negation of a bool is a bool, which the arena holds, so it
    # is admitted on purpose and putting it here would assert the opposite of
    # what wiring it means.
    for target in (ANY_DIM,):
        monkeypatch.setitem(hexagon_backend.EMITTERS, target, lambda ctx, node, i: None)
        monkeypatch.setitem(
            part_mod.SUPPORTED_TARGETS, target, lambda ctx, node, i: None
        )
    # The aliasing is what makes one setitem reach both, and the assertion is
    # here so a future unbinding fails loudly instead of patching half the path.
    assert part_mod.SUPPORTED_TARGETS is hexagon_backend.SUPPORTED_TARGETS

    first, reduced, second = _bool_chain()
    # The negations are wired, so putting a target in the table is not what decides
    # them. The reduce is the node this is about: with any.dim now in the table it
    # is still refused, and the refusal is the width gate rather than the table,
    # which is the whole claim.
    assert reduced.target in part_mod.SUPPORTED_TARGETS
    assert support.is_node_supported(None, first)
    assert not support.is_node_supported(None, reduced)
    assert support.is_node_supported(None, second)


def test_an_emitter_for_those_targets_would_need_the_arena_width_it_refuses():
    """The emitter-side guard raises instead of falling back, so the order matters.

    A node the partitioner refuses falls back to a portable kernel; a node an
    emitter refuses fails the export. _require_arena_dtype is the second of
    those, and it is reached with the very same bool the width gate turns away,
    so admitting bool at the partitioner alone would convert a working export
    into a RuntimeError.
    """
    # The bool exemption is keyed on the target, and a comparison is on it, so
    # the node that still has to be refused is a bool arriving at a float target.
    bool_at_a_float_target = _add(torch.bool)
    with pytest.raises(RuntimeError, match="must be fp16 or fp32"):
        hexagon_ops._require_arena_dtype(bool_at_a_float_target, "add")

    add = _add(torch.float16)
    hexagon_ops._require_arena_dtype(add, "add")
