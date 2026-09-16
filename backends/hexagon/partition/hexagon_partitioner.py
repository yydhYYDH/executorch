# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, final, Set

import torch
from executorch.backends.hexagon.hexagon_backend import HexagonBackend, SUPPORTED_TARGETS
from executorch.backends.hexagon.hexagon_ops import (
    _scalar_source,
    ALIAS_TARGETS,
    CAST_TARGETS,
    BINARY_TARGETS,
    MEAN_TARGETS,
    cat_region,
    CAT_TARGETS,
    MM_TARGETS,
    permute_region,
    PERMUTE_TARGETS,
    select_region,
    SELECT_TARGETS,
    slice_region,
    SLICE_TARGETS,
    sdpa_targets,
    update_cache_layout,
)
from executorch.backends.hexagon.kv_cache import UPDATE_CACHE
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from torch.export import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase


def _dtype_of(node: torch.fx.Node):
    return getattr(node.meta.get("val"), "dtype", None)


def _sdpa_fits_dsp_limits(node: torch.fx.Node) -> bool:
    """Whether the attention operands match the one FLASH_ATTN shape we emit.

    The emitter parks start_pos on an extra input slot, so the value has to come
    from a tensor read that survives into the graph.
    """
    # custom_sdpa has no caches, so only query, key, value and start_pos are
    # required; the rest carry defaults and may be omitted.
    if len(node.args) < 4:
        return False
    for arg in (node.args[0], node.args[1], node.args[2]):
        if not isinstance(arg, torch.fx.Node):
            return False
        val = arg.meta.get("val")
        # fp32 operands are accepted: the runtime narrows them to fp16 as they
        # enter the arena. The attention math is then fp16 on the DSP.
        if val is None or val.dtype not in (torch.float16, torch.float32):
            return False
        if val.dim() != 4:
            return False
    # Either a run-time read the runtime patches, or a constant we bake in.
    source = _scalar_source(node.args[3])
    return isinstance(source, torch.fx.Node) or isinstance(node.args[3], int)


def _broadcast_fits_dsp_limits(node: torch.fx.Node) -> bool:
    """Whether the operand broadcast is one the DSP can walk.

    The DSP's broadcast path takes at most 8 dims and right-aligns each operand
    against the output, which is what torch's own broadcasting does; anything
    wider has to stay off the delegate.
    """
    out = node.meta.get("val")
    if out is None or out.dim() > 8:
        return False
    for arg in node.args[:2]:
        if not isinstance(arg, torch.fx.Node):
            continue
        val = arg.meta.get("val")
        if val is None or val.dim() > 8:
            return False
    return True


def _mm_operands_fit_flat_path(node: torch.fx.Node) -> bool:
    """Whether mm's operands are the plain contiguous 2-D tiles the emitter assumes."""
    lhs, rhs = node.args[0], node.args[1]
    if not (isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)):
        return False
    lhs_val, rhs_val = lhs.meta.get("val"), rhs.meta.get("val")
    if lhs_val is None or rhs_val is None:
        return False
    return (
        lhs_val.dim() == 2
        and rhs_val.dim() == 2
        and lhs_val.is_contiguous()
        and rhs_val.is_contiguous()
        and lhs_val.shape[1] == rhs_val.shape[0]
    )


def _mean_reduces_one_span(node: torch.fx.Node) -> bool:
    """Whether mean's dims are the single contiguous span REDUCTION can collapse."""
    src = node.args[0]
    if not isinstance(src, torch.fx.Node):
        return False
    val = src.meta.get("val")
    if val is None or val.dim() == 0:
        return False
    dims = node.args[1]
    dims = [dims] if isinstance(dims, int) else list(dims)
    if not dims:
        return False
    rank = val.dim()
    norm = sorted(d % rank for d in dims)
    return norm == list(range(norm[0], norm[0] + len(norm)))


def _cast_stays_in_fp16(node: torch.fx.Node) -> bool:
    """Whether this cast is between the two widths the arena already holds.

    A cast from int64 is a conversion the kernels cannot do and must not be
    absorbed; one between fp16 and fp32 is the runtime's job on whichever side
    of the boundary it lands.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return False
    return {_dtype_of(source), _dtype_of(node)} <= {torch.float16, torch.float32}


def _emits_no_command(node: torch.fx.Node) -> bool:
    """Whether this node's emitter only re-points its operand's TensorRef."""
    if node.target in ALIAS_TARGETS:
        return True
    return node.target in CAST_TARGETS and _cast_stays_in_fp16(node)


def _alias_keeps_the_same_bytes(node: torch.fx.Node) -> bool:
    """True when the result is the operand's bytes in the same order.

    A contiguous operand and a contiguous result of the same element count is
    the whole condition: the emitter then re-points the operand's TensorRef, so
    whichever way the kernels walk the buffer they see the same numbers. A
    slice of the inner dimension fails it, because the result would skip bytes
    the layout it describes does not mention.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return False
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return False
    return (
        source_value.is_contiguous()
        and result_value.is_contiguous()
        and source_value.numel() == result_value.numel()
    )


class HexagonOperatorSupport(OperatorSupportBase):
    """Accepts the ops the DSP has a kernel for, at a dtype it can run.

    The target set is the DSP op table's keys, so support and codegen cannot
    drift: an op is delegated exactly when the runtime can emit a command for
    it. The dtype check belongs here rather than in the emitters: the DSP
    kernels are fp16, and a node rejected here falls back to a portable kernel,
    whereas one rejected while emitting fails the whole export.
    """

    def __init__(self) -> None:
        # Views that a partition would have to hand out, recorded by the
        # partitioner once the consumers are known. Empty means the per-node
        # answer, which is all this check can reach on its own.
        self.boundary_views: Set[torch.fx.Node] = set()

    def is_node_supported(self, _submodules, node: torch.fx.Node) -> bool:
        if node in self.boundary_views:
            return False
        if node.op != "call_function":
            return False
        # Resolving the attention overloads also registers their emitter, and the
        # extension that defines them loads after this module is imported.
        sdpa = sdpa_targets()
        if node.target not in SUPPORTED_TARGETS:
            return False
        dtype = _dtype_of(node)
        if dtype is not torch.float16:
            # Attention is the one op whose fp32 operands are narrowed to fp16
            # on the way into the arena, so it is the one op that may be fp32.
            # A cast between the two widths is the other: the runtime performs it
            # on whichever side of the boundary the cast ends up.
            absorbed = node.target in sdpa or (
                node.target in CAST_TARGETS and _cast_stays_in_fp16(node)
            )
            if dtype is not torch.float32 or not absorbed:
                return False
        if node.target in sdpa and not _sdpa_fits_dsp_limits(node):
            return False
        if node.target in BINARY_TARGETS and not _broadcast_fits_dsp_limits(node):
            return False
        if node.target in MM_TARGETS and not _mm_operands_fit_flat_path(node):
            return False
        if node.target in MEAN_TARGETS and not _mean_reduces_one_span(node):
            return False
        if _emits_no_command(node) and not _alias_keeps_the_same_bytes(node):
            # A narrowing select reaches the same emitter through its own region
            # rather than by re-pointing, so the alias test is not the last word.
            if node.target not in SELECT_TARGETS or select_region(node) is None:
                return False
        if node.target in SLICE_TARGETS and slice_region(node) is None:
            return False
        if node.target in CAT_TARGETS and cat_region(node) is None:
            return False
        if node.target in PERMUTE_TARGETS and permute_region(node) is None:
            return False
        if node.target is UPDATE_CACHE and update_cache_layout(node) is None:
            return False
        return all(
            isinstance(arg, torch.fx.Node)
            and (arg.op != "get_attr" or _dtype_of(arg) is torch.float16)
            for arg in node.args
            # None is a legitimate argument (custom_sdpa passes no mask), so it
            # must be excluded here rather than rejected by the Node test.
            if not isinstance(arg, (int, float, bool, list, tuple, type(None)))
        )


@final
class HexagonPartitioner(Partitioner):
    """Delegates every supported op, merging connected ones into one subgraph.

    Merging matters: each delegate subgraph costs one FastRPC round trip and its
    own arena, so a partition per node would be correct but pointless.
    """

    def __init__(self) -> None:
        self.delegation_spec = DelegationSpec(HexagonBackend.__name__, [])

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        graph_module = exported_program.graph_module
        support = HexagonOperatorSupport()

        supported = [
            node
            for node in graph_module.graph.nodes
            if support.is_node_supported(None, node)
        ]

        # A view re-points its operand's TensorRef and emits no command, so a
        # partition that has to hand one out would have nothing to fill the
        # output slot with. One whose consumers are all delegated stays inside
        # its partition; one that is a boundary keeps its own result instead.
        # Who the consumers are is only known here, so the question is asked
        # here rather than in the support check.
        # Reversed, because dropping a view makes the view feeding it a
        # boundary as well, and graph.nodes is topological, so walking backwards
        # settles every consumer before its producer. The answer is recorded on
        # the support object as well as removed from this list, because
        # generate_partitions_from_list_of_nodes asks the support object again
        # instead of trusting the list.
        supported_set = set(supported)
        for node in reversed(list(supported)):
            if _emits_no_command(node) and any(
                user not in supported_set for user in node.users
            ):
                supported.remove(node)
                supported_set.discard(node)
                support.boundary_views.add(node)

        partition_tags: Dict[str, DelegationSpec] = {}
        if not supported:
            return PartitionResult(
                tagged_exported_program=exported_program, partition_tags=partition_tags
            )

        # Splits the supported nodes into connected components, so an
        # unsupported op in the middle ends one partition and starts another.
        partitions = generate_partitions_from_list_of_nodes(
            graph_module, [supported], support
        )

        for partition in partitions:
            delegation_tag = f"hexagon_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = delegation_tag
                # Constants consumed by a delegated node have to come along, or
                # they stay in the graph as inputs the delegate never sees.
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                        arg.meta["delegation_tag"] = delegation_tag
            partition_tags[delegation_tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
