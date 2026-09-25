# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuses a prefix scan into the two commands the DSP already has.

`torch.cumsum` is a scan, and the DSP has no scan: the reduction table carries
sum, amax and mean, and nothing walks a prefix. What it has is a batched
matmul, so a cumsum along the last axis of an fp16 input is one
`BATCH_MATMUL` against the constant `(L, L)` upper-triangular matrix of ones:
column j of the mask holds a one in rows 0..j and a zero in the rest, so
`out[r, j] = sum_{i <= j} x[r, i]`, and every leading axis folds into the
matmul's rows because the scan runs along the contiguous one.

The scan order is the mask, not the kernel. The k loop visits i = 0..K-1 left
to right, which is the order `torch.cumsum` defines, and the mask keeps
exactly the prefix ending at j: the lower triangle would sum the suffix and a
full ones matrix would sum the whole row. The accumulator is the matmul's:
fp32, narrowed to fp16 once at the store. torch.cumsum on an fp16 CPU tensor
accumulates in fp32 the same way, so the two agree bit for bit here; the
reference that matters is an exact (fp64) scan, and both sit within one
fp16 step of the output scale.

The weight is a host constant, so the export arranges it into the matrix
unit's tile order and the DSP streams it out of the weights section. The
masked zeros are ordinary packed zeros: `pack_hmx_weight` has no
triangular-specific path, and a zero entry is a zero in either layout. The
weight is packed only when `_weight_operand`'s own rule would pack a
matmul's constant, because a weight this layer cannot pack is read
row-major by the HMX fallbacks and a tiled one would be rubbish to them.

A streaming cumsum carries the previous chunk's total into the next one, and
that is the `add` the caller writes after the scan -- `out = cumsum(x, -1) +
carry`, then `carry = out[..., -1:]` for the next execute(). The add is the
existing `BINARY_ELEMENTWISE` emitter and the slice the existing
`select_copy` blit, so this pass states the scan and the chunk boundary
stays ordinary graph structure: no state is kept on the DSP and the carry
travels as a method argument.

The scan length is the matmul's K, and the old phone skel stages a matmul's
activation rows at `ceil(K/64)*64`, so a K that is not a multiple of 64
reads from misaligned rows there (the tree and the simulator have the fix,
the phone's .so does not). The support predicate therefore accepts only
`L % 64 == 0`.

The fused node is created after `to_edge`, for the same reason the fused norm
is: a fused op built before that pass does not survive the decomposition table.
"""

from typing import NamedTuple, Optional

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

NAMESPACE = "et_hexagon"

# rms_norm opens this namespace with a DEF block and only one of those is
# allowed, so this one adds a fragment to it instead.
_library = torch.library.Library(NAMESPACE, "FRAGMENT")

#: The scan length the support predicate accepts: the matmul's K and N and the
#: mask's side, and the row stride the phone's skel stages a K-wide row at.
SCAN_ALIGNMENT = 64

#: The largest scan the mask makes practical. A (L, L) fp16 weight is 2*L*L
#: bytes, so L = 1024 is 2 MB of weights for one fused node; a longer scan is
#: better written as a log-step pass (L adds over L/2 + L/4 + ... shifted
#: copies) than as a square mask, and this op refuses rather than hide that.
MAX_SCAN_LENGTH = 1024


def _cumsum(x, carry=None):
    """The same arithmetic, written out for anywhere the DSP cannot reach."""
    scanned = torch.cumsum(x, dim=-1)
    return scanned if carry is None else scanned + carry


_library.define("cumsum(Tensor x, Tensor? carry=None) -> Tensor")
_library.impl("cumsum", _cumsum, "CompositeExplicitAutograd")

CUMSUM = exir_ops.edge.et_hexagon.cumsum.default

_CUMSUM = exir_ops.edge.aten.cumsum.default
_ADD = exir_ops.edge.aten.add.Tensor


def _val(value) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, torch.fx.Node):
        return None
    meta = value.meta.get("val")
    return meta if isinstance(meta, torch.Tensor) else None


def cumsum_is_emittable(x, carry=None) -> bool:
    """Whether the two commands this op composes can run this scan.

    `x` and `carry` are graph arguments, and a bare tensor is accepted for the
    same question asked of a value that is not on a graph yet.

    The operand has to be a contiguous fp16 tensor of at least two dimensions
    with a static last extent that is a positive multiple of
    `SCAN_ALIGNMENT` and no larger than `MAX_SCAN_LENGTH`: it is the
    mask's side, the matmul's K and N, and a K that is not 64-aligned reads
    from misaligned rows on the phone's skel. The carry, when the graph passes
    one, has to be an fp16 tensor too, because the add is the ordinary binary
    emitter and the arena holds two bytes per element.
    """
    if x is None or carry is not None and _val(carry) is None:
        return False
    value = _val(x)
    if value is None or value.dtype is not torch.float16 or value.dim() < 2:
        return False
    if not value.is_contiguous():
        return False
    length = value.shape[-1]
    if isinstance(length, torch.SymInt) or not isinstance(length, int):
        return False
    if length <= 0 or length % SCAN_ALIGNMENT or length > MAX_SCAN_LENGTH:
        return False
    if carry is None:
        return True
    carried = _val(carry)
    return carried.dtype is torch.float16


class CumsumMatch(NamedTuple):
    """The three arguments one matched scan hands the fused node."""

    # The node the fused op takes over: the add when the graph threads a
    # carry, the scan itself when it is the first chunk of a stream.
    replace: torch.fx.Node
    # What the scan reads, and the carry the add contributes, if any.
    x: torch.fx.Node
    carry: Optional[torch.fx.Node]


def _scans_the_last_axis(scanned: torch.fx.Node) -> bool:
    """Whether this cumsum is the `dim=-1` form the mask computes."""
    if len(scanned.args) < 2:
        return False
    dim = scanned.args[1]
    if isinstance(dim, bool) or not isinstance(dim, int):
        return False
    value = _val(scanned.args[0])
    if value is None:
        return False
    return dim == -1 or dim == value.dim() - 1


def match_cumsum(node: torch.fx.Node) -> Optional[CumsumMatch]:
    """The streaming scan, anchored on the add that carries the last chunk.

        x --.    cumsum --> scan --> add --> anchor
        c --'                 carry'

    A scan with no carry is the first chunk of a stream, so it is matched on
    its own too; the fused op then holds the scan alone and a reader that
    adds a carry to it keeps doing so with the ordinary binary emitter.
    """
    if node.op != "call_function":
        return None
    if node.target is _ADD:
        if len(node.args) != 2:
            return None
        for scanned, carry in (node.args, node.args[::-1]):
            if (
                isinstance(scanned, torch.fx.Node)
                and scanned.target is _CUMSUM
                and _scans_the_last_axis(scanned)
                and cumsum_is_emittable(scanned.args[0], carry)
            ):
                return CumsumMatch(node, scanned.args[0], carry)
        return None
    if node.target is _CUMSUM and _scans_the_last_axis(node):
        if cumsum_is_emittable(node.args[0] if node.args else None):
            return CumsumMatch(node, node.args[0], None)
    return None


def fuse_cumsum(graph_module: torch.fx.GraphModule) -> int:
    """Replaces every matched scan with one node, returns how many.

    The cumsum is left for `eliminate_dead_code`, which is what happens when
    something else reads it as well: the fused op serves the add's readers
    and the bare scan stays delegated on its own for the rest.
    """
    graph = graph_module.graph
    carried = {
        scanned
        for node in graph.nodes
        if node.op == "call_function"
        and node.target is _ADD
        and len(node.args) == 2
        for scanned in node.args
        if isinstance(scanned, torch.fx.Node) and scanned.target is _CUMSUM
    }
    found = [
        (node, match)
        for node in graph.nodes
        if node.op == "call_function"
        and (match := match_cumsum(node)) is not None
        and (node.target is not _CUMSUM or node not in carried)
    ]

    for _anchor, match in found:
        with graph.inserting_before(match.replace):
            fused = graph.create_node(
                "call_function",
                CUMSUM,
                args=(match.x, match.carry),
            )
        fused.meta["val"] = match.replace.meta["val"]
        match.replace.replace_all_uses_with(fused)

    if found:
        graph.eliminate_dead_code()
        graph_module.recompile()
    return len(found)


class FuseCumsumPass(ExportPass):
    """`fuse_cumsum` as a pass, which is the only place it can live.

    `to_backend` checks that a partitioner returns the graph it was given, so
    a backend cannot do this while partitioning. It belongs in the caller's
    `transform_passes`, ahead of the split.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        """Every matched scan becomes one node, in place."""
        fuse_cumsum(graph_module)
        return PassResult(graph_module, True)
