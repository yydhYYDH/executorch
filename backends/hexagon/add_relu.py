# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses `relu(x + y)` into the one command the DSP has for it.

The kernel's element-wise op type 8 is `max(a + b, 0)` and nothing in the graph
produces it: `torch.export` writes a rectified activation as two nodes, an add
and a `relu`, and `HtpOpsUnaryOpType` has no relu for the second one to reach. So
a graph that writes the pair keeps both on the host, or delegates the add alone
and pays a round trip through the arena for the clamp, which is the shape of
every `linear + bias` followed by a rectifier.

The fused node is what the emitter table already has an entry for -- subtype 8 of
`DSP_OP_BINARY_ELEMENTWISE` -- and it is created in the caller's
`transform_passes`, ahead of the split, for the same reason the fused norm and
the gated activation are: a fused op built before `to_edge` does not survive the
decomposition table. `FuseAddReluPass` is that pass; a caller that does not add
it gets the two nodes it wrote, delegated as far as each can go.
"""

from typing import NamedTuple, Optional

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

NAMESPACE = "et_hexagon"

# rms_norm opens this namespace with a DEF block and only one of those is
# allowed, so this one has to add a fragment to it instead. That makes the
# import order in hexagon_ops.py load-bearing.
_library = torch.library.Library(NAMESPACE, "FRAGMENT")


def _add_relu(a, b):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    return torch.nn.functional.relu(a + b)


_library.define("add_relu(Tensor a, Tensor b) -> Tensor")
_library.impl("add_relu", _add_relu, "CompositeExplicitAutograd")

ADD_RELU = exir_ops.edge.et_hexagon.add_relu.default

_ADD = exir_ops.edge.aten.add.Tensor
_RELU = exir_ops.edge.aten.relu.default


class AddReluMatch(NamedTuple):
    # The rectifier the fused op replaces. It is the node the graph's readers
    # hold, so it is the one to re-point.
    replace: torch.fx.Node
    # The addend and the augend, in the order the kernel takes them. The
    # activation is not commutative in general, but `max(a + b, 0)` is, so no
    # operand order has to be recovered: both inputs are read as fp16 and the
    # kernel rounds the sum once, which is what the two fp16 nodes did.
    lhs: torch.fx.Node
    rhs: torch.fx.Node


def match_add_relu(node: torch.fx.Node) -> Optional[AddReluMatch]:
    """The rectified sum, anchored on the rectifier.

        x --.
            add -> relu -> anchor
        y --'

    Both operands have to be fp16 tensors, because the kernel reads the arena's
    two-byte elements and nothing else: an operand of another width would be
    narrowed somewhere the pattern does not look. A python scalar operand is left
    alone rather than materialized here -- `ctx.scalar` would take it, but the
    pattern is about a graph torch wrote, not one this pass invents.
    """
    if node.target is not _RELU:
        return None
    result = node.meta.get("val")
    if not isinstance(result, torch.Tensor) or result.dtype is not torch.float16:
        return None
    added = node.args[0] if node.args else None
    if not isinstance(added, torch.fx.Node) or added.target is not _ADD:
        return None
    if len(added.args) != 2:
        return None
    lhs, rhs = added.args
    for value in (lhs, rhs):
        if not isinstance(value, torch.fx.Node):
            return None
        held = value.meta.get("val")
        if not isinstance(held, torch.Tensor) or held.dtype is not torch.float16:
            return None
    return AddReluMatch(node, lhs, rhs)


def fuse_add_relu(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every matched rectified sum with one node, returns how many.

    The add is left for `eliminate_dead_code`, which is what happens when
    something else reads it as well: the fused node serves the rectifier's
    readers and the add stays delegated on its own for the rest.
    """
    graph = graph_module.graph
    found = [
        (node, match)
        for node in graph.nodes
        if node.op == "call_function" and (match := match_add_relu(node)) is not None
    ]

    for _anchor, match in found:
        with graph.inserting_before(match.replace):
            fused = graph.create_node(
                "call_function",
                ADD_RELU,
                args=(match.lhs, match.rhs),
            )
        fused.meta["val"] = match.replace.meta["val"]
        match.replace.replace_all_uses_with(fused)

    if found:
        graph.eliminate_dead_code()
        graph_module.recompile()
    return len(found)


class FuseAddReluPass(ExportPass):
    """`fuse_add_relu` as a pass, which is the only place it can live.

    `to_backend` checks that a partitioner returns the graph it was given, so a
    backend cannot do this while partitioning. It belongs in the caller's
    `transform_passes`, ahead of the split.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        return PassResult(graph_module, fuse_add_relu(graph_module) > 0)
