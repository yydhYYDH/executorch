# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, final, Set

import torch
from executorch.backends.hexagon.hexagon_backend import HexagonBackend, SUPPORTED_TARGETS
from executorch.backends.hexagon.hexagon_ops import (
    _scalar_arg,
    _scalar_source,
    ADDMM_TARGETS,
    ALIAS_TARGETS,
    BMM_TARGETS,
    CAST_TARGETS,
    BINARY_TARGETS,
    GETITEM,
    LAYER_NORM,
    layer_norm_getitem,
    layer_norm_is_emittable,
    layer_norm_normalizes_the_trailing_dims,
    MEAN_TARGETS,
    NATIVE_LAYER_NORM,
    softmax_reduces_the_inner_axis,
    SOFTMAX_TARGETS,
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


# Emitters that store a constant at the width their kernel reads rather than the
# graph's own, so the weight does not have to be fp16 for them.
FP32_CONSTANT_TARGETS = frozenset({LAYER_NORM, NATIVE_LAYER_NORM})


def _dtype_of(node: torch.fx.Node):
    val = node.meta.get("val")
    # A multi-output op carries the whole tuple as its value; the first tensor is
    # the one the kernels read, so that is the width the gate is about.
    if isinstance(val, tuple):
        val = val[0] if val else None
    return getattr(val, "dtype", None)


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


def _batched_operands_fit(node: torch.fx.Node) -> bool:
    """Whether bmm's operands are the contiguous 3-D stacks the emitter assumes.

    The batch axis has to be there on both operands and the contraction has to
    line up, because the emitter derives one tile geometry and one step from the
    shapes and cannot describe a broadcast batch.
    """
    lhs, rhs = node.args[0], node.args[1]
    if not (isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)):
        return False
    lhs_val, rhs_val = lhs.meta.get("val"), rhs.meta.get("val")
    if lhs_val is None or rhs_val is None:
        return False
    return (
        lhs_val.dim() == 3
        and rhs_val.dim() == 3
        and lhs_val.is_contiguous()
        and rhs_val.is_contiguous()
        and lhs_val.shape[0] == rhs_val.shape[0]
        and lhs_val.shape[2] == rhs_val.shape[1]
        and _fits_int32_offsets(node, lhs_val, rhs_val)
    )


def _fits_int32_offsets(node: torch.fx.Node, *operands) -> bool:
    """Whether the sizes and steps the descriptor carries fit an int32.

    The DSP computes an iteration's base as step * iter in int32, so a tile
    large enough to overflow it would land somewhere else in the arena rather
    than fail.
    """
    result = node.meta.get("val")
    if result is None:
        return False
    values = [operand.numel() for operand in operands]
    values.append(result.numel())
    return all(0 < value < 2**31 for value in values)


def _addmm_fits_flat_path(node: torch.fx.Node) -> bool:
    """Whether addmm is one matmul plus a bias the DSP's broadcast can repeat.

    alpha has no kernel here, so only 1 is emittable and the rest has to stay on
    a portable kernel. beta is the bias's own scale: 0 drops it, 1 keeps it, and
    anything else would need a multiply this does not emit. The bias is read
    right-aligned against the 2-D result, so it cannot have more axes than that
    and it must be a tensor the delegate can reach.
    """
    if len(node.args) < 3:
        return False
    mat1, mat2 = node.args[1], node.args[2]
    if not (isinstance(mat1, torch.fx.Node) and isinstance(mat2, torch.fx.Node)):
        return False
    alpha = _scalar_arg(node, "alpha", 4, 1.0)
    if alpha != 1.0:
        return False
    beta = _scalar_arg(node, "beta", 3, 1.0)
    if beta not in (0.0, 1.0):
        return False
    mat1_val, mat2_val = mat1.meta.get("val"), mat2.meta.get("val")
    if mat1_val is None or mat2_val is None:
        return False
    if not (
        mat1_val.dim() == 2
        and mat2_val.dim() == 2
        and mat1_val.is_contiguous()
        and mat2_val.is_contiguous()
        and mat1_val.shape[1] == mat2_val.shape[0]
    ):
        return False
    if beta == 0.0:
        return _fits_int32_offsets(node, mat1_val, mat2_val)
    bias = node.args[0]
    if not isinstance(bias, torch.fx.Node):
        return False
    bias_val = bias.meta.get("val")
    if bias_val is None or bias_val.dim() > 2:
        return False
    # Whatever torch broadcasts the bias to is what the DSP repeats, so the only
    # question left is whether the strides it walks can reach it.
    try:
        torch.broadcast_shapes(tuple(bias_val.shape), tuple(mat1_val.shape[:1] + mat2_val.shape[1:]))
    except RuntimeError:
        return False
    return _fits_int32_offsets(node, mat1_val, mat2_val, bias_val)


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
        if dtype not in (torch.float16, torch.float32):
            # Both widths the arena holds are emittable: every kernel reads and
            # writes two bytes per element, the runtime narrows a fp32 operand on
            # the way in and widens a fp32 result on the way out, so one declared
            # fp32 emits the same commands as its fp16 twin. Any other width
            # would leave the kernels reading int64 bits as half floats.
            return False
        if node.target in sdpa and not _sdpa_fits_dsp_limits(node):
            return False
        if node.target in BINARY_TARGETS and not _broadcast_fits_dsp_limits(node):
            return False
        if node.target in MM_TARGETS and not _mm_operands_fit_flat_path(node):
            return False
        if node.target in BMM_TARGETS and not _batched_operands_fit(node):
            return False
        if node.target in ADDMM_TARGETS and not _addmm_fits_flat_path(node):
            return False
        if node.target in MEAN_TARGETS and not _mean_reduces_one_span(node):
            return False
        if node.target in (LAYER_NORM, NATIVE_LAYER_NORM):
            # A run-time epsilon is not a number the command can carry, and a
            # normalized shape that is not the trailing dims is a span the kernel
            # would read wrong rather than refuse.
            if _scalar_arg(node, "eps", 4, 1e-5) is None:
                return False
            if not layer_norm_normalizes_the_trailing_dims(node):
                return False
            if node.target is NATIVE_LAYER_NORM and not layer_norm_is_emittable(node):
                return False
        if node.target in SOFTMAX_TARGETS and not softmax_reduces_the_inner_axis(node):
            return False
        if node.target is GETITEM:
            # The getitem that reads a layer norm's first output is the one this
            # backend can place; every other getitem (a split's, for one) has no
            # producer here.
            source = layer_norm_getitem(node)
            if source is None or not layer_norm_is_emittable(source):
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
        for arg in node.args:
            # None is a legitimate argument (custom_sdpa passes no mask), so a
            # literal is excluded here rather than rejected by the Node test.
            if isinstance(arg, (int, float, bool, list, tuple, type(None))):
                continue
            if not isinstance(arg, torch.fx.Node):
                return False
            # A weight has to be the width the arena holds, except where the
            # emitter converts it itself because the kernel reads that width:
            # layer norm's gamma and beta are fp32 on the DSP.
            if (
                arg.op == "get_attr"
                and node.target not in FP32_CONSTANT_TARGETS
                and _dtype_of(arg) is not torch.float16
            ):
                return False
        return True


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
