# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, final, Optional, Set

import torch
from executorch.backends.hexagon.hexagon_backend import (
    HexagonBackend,
    HexagonCompileOptions,
    SUPPORTED_TARGETS,
)
from executorch.backends.hexagon.hexagon_ops import (
    _dequantize_is_fused,
    _scalar_arg,
    _scalar_source,
    ADD_RMS_NORM,
    add_rms_norm_getitem,
    add_rms_norm_is_emittable,
    ADDMM_TARGETS,
    ALIAS_TARGETS,
    BINARY_TARGETS,
    BMM_TARGETS,
    CAST_TARGETS,
    cat_region,
    CAT_TARGETS,
    conv_spec,
    CONV_TARGETS,
    dim_order_keeps_the_bytes,
    DIM_ORDER_TARGETS,
    DQ_PER_CHANNEL,
    gather_table,
    GATHER_TARGETS,
    GETITEM,
    LAYER_NORM,
    layer_norm_getitem,
    layer_norm_is_emittable,
    layer_norm_normalizes_the_trailing_dims,
    MAX_POOL2D_WITH_INDICES,
    max_pool_getitem,
    max_pool_is_emittable,
    mean_result_width_is_emittable,
    MEAN_TARGETS,
    MM_TARGETS,
    NATIVE_LAYER_NORM,
    operand_dtypes_are_readable,
    permute_region,
    PERMUTE_TARGETS,
    pool_spec,
    POOL_TARGETS,
    pow_is_square,
    quantized_matmul_is_refused,
    reduction_dims,
    REDUCTION_TARGETS,
    sdpa_targets,
    select_region,
    SELECT_TARGETS,
    slice_region,
    SLICE_TARGETS,
    softmax_reduces_the_inner_axis,
    SOFTMAX_TARGETS,
    SQUARE_POW_TARGETS,
    sum_dim_is_emittable,
    SUM_TARGETS,
    update_cache_layout,
    vision_attention_is_emittable,
    VISION_ATTENTION_TARGETS,
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
from executorch.exir.sym_util import eval_shape_upper_bound
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
    if len(node.args) > 4 and node.args[4] is not None:
        # A mask is the one operand the kernel cannot be handed. It reads the
        # mask as rows of `mask_stride` two-byte elements right-aligned to the
        # keys and copies them into a fp32 region of the workspace it places
        # after the per-task rows -- a region this backend sizes for the
        # unmasked shape, out of a buffer `htp_ops_flash_attn` does not
        # bounds-check. The emitter can only say mask_stride = -1, which the
        # kernel reads as "no mask", so delegating one is silently computing
        # something else. (The exporter builds this form for
        # `use_custom_sdpa_with_attention_mask`.) Refused until the stride, the
        # dtype and the workspace are validated on hardware.
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

    def upper_numel(value) -> int:
        numel = 1
        for dim in eval_shape_upper_bound(value.shape):
            numel *= dim
        return numel

    values = [upper_numel(operand) for operand in operands]
    values.append(upper_numel(result))
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
        torch.broadcast_shapes(
            tuple(bias_val.shape), tuple(mat1_val.shape[:1] + mat2_val.shape[1:])
        )
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
    # aten.mean.dim carries the dim and aten.mean.default has none, so the
    # second slot is absent on the overload that reduces everything. A missing
    # dim means every dim on both, and every dim is one span: the kernel reduces
    # the whole buffer as [1][numel][1].
    dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dims is None:
        return True
    dims = [dims] if isinstance(dims, int) else list(dims)
    # An empty dim set is not "reduce nothing": `torch.mean(x, dim=())` is
    # `torch.mean(x)` to the last bit, and so is `torch.sum(x, dim=())` and
    # `torch.amax(x, dim=())`. The reduction table reads it that way already
    # (`reduction_dims`), so this function refusing it left `torch.mean(x,
    # dim=())` on the portable kernels while the other two delegated the
    # identical reduction.
    if not dims:
        return True
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
    if dim_order_keeps_the_bytes(node):
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

    def __init__(self, data_names: Optional[frozenset] = None) -> None:
        # The placeholders the program owns rather than the caller handing them
        # in, which is the question a row gather's table has to answer. Without
        # the program there is no way to tell one from a method input, and the
        # only operand provably a constant on its own is a get_attr.
        self.data_names = frozenset(data_names or ())

    def is_data_placeholder(self, node: torch.fx.Node) -> bool:
        """Whether this operand is a parameter, buffer or lifted constant."""
        return node.op == "get_attr" or (
            node.op == "placeholder" and node.name in self.data_names
        )

    def is_node_supported(self, _submodules, node: torch.fx.Node) -> bool:
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
        if not operand_dtypes_are_readable(node):
            # A bool operand is one byte per element where the kernels read two,
            # which is a wrong answer rather than an error. No emitter here takes
            # one, so any node that carries one stays portable.
            return False
        if node.target is DQ_PER_CHANNEL:
            # The weight-only pattern's dequantize. It is delegated only when
            # every reader is a quantized matmul the GEMV kernels can run; the
            # node itself emits nothing.
            return _dequantize_is_fused(node)
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
        if quantized_matmul_is_refused(node):
            # A weight-only matmul goes to a GEMV kernel, and the conditions the
            # flat path just checked are not the ones that decide it.
            return False
        if node.target in MEAN_TARGETS:
            if not _mean_reduces_one_span(node):
                return False
            if not mean_result_width_is_emittable(node):
                # The kernel accumulates in fp32 and stores fp16, so a mean that
                # names another dtype is a different op -- the rule
                # sum_dim_IntList already carries in sum_dim_is_emittable. Both
                # mean overloads used to skip it, and torch.mean(x, dim=1,
                # dtype=torch.float32) came back as fp16 values widened to fp32.
                return False
        if node.target in SQUARE_POW_TARGETS and not pow_is_square(node):
            # Only x ** 2 maps to the unary square kernel; another exponent is a
            # different function, not a narrower command.
            return False
        if node.target in REDUCTION_TARGETS:
            # One contiguous span, the same shape rule mean has; sum additionally
            # has to be the fp16 sum, since the kernel's accumulator is fp32 and
            # its result fp16.
            if reduction_dims(node) is None:
                return False
            if node.target in SUM_TARGETS and not sum_dim_is_emittable(node):
                return False
        if node.target is MAX_POOL2D_WITH_INDICES and not max_pool_is_emittable(node):
            # The indices come out of the same node and no kernel here produces
            # them, so a graph that reads them keeps the pool portable.
            return False
        if node.target in POOL_TARGETS and pool_spec(node) is None:
            # The kernel walks its activation in the DSP's 64-channel blocked
            # layout, and only the C == 64 form of it is one this backend
            # rearranges; a dilation, a ceil_mode window, a divisor this kernel
            # cannot take, or a shape whose windows would fall outside the input
            # all have to stay on a portable kernel rather than reach it.
            return False
        if (
            node.target in CONV_TARGETS
            and conv_spec(node, self.is_data_placeholder) is None
        ):
            # The two convolution kernels take one weight order and one weight
            # they can see at export, and neither of them walks a group count
            # in between 1 and the channel count: anything else stays on a
            # portable kernel rather than reach one that reads it wrong.
            return False
        if (
            node.target in GATHER_TARGETS
            and gather_table(node, self.is_data_placeholder) is None
        ):
            # The DSP reads a tiled table and four-byte indices, so a table whose
            # bytes this layer cannot see at export -- or an index tensor that is
            # not an int32 method input -- has no command form: those stay on a
            # portable kernel rather than reach a kernel that reads them wrong.
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
        if node.target is ADD_RMS_NORM and not add_rms_norm_is_emittable(node):
            return False
        if (
            node.target in VISION_ATTENTION_TARGETS
            and not vision_attention_is_emittable(node)
        ):
            # The batch, head count and head width are params the run-time length
            # cannot refresh, so a symbolic one would be the traced example at
            # run time: a wrong answer rather than a failure.
            return False
        if node.target in SOFTMAX_TARGETS and not softmax_reduces_the_inner_axis(node):
            return False
        if node.target is GETITEM:
            # The getitem that reads a layer norm's first output, or one of the
            # fused add+norm's two, is the one this backend can place; every
            # other getitem (a split's, for one) has no producer here.
            source = layer_norm_getitem(node)
            if source is not None:
                if not layer_norm_is_emittable(source):
                    return False
            elif max_pool_getitem(node) is not None:
                # The max pool's values; that node's own check has already
                # refused a pool whose indices anything else reads.
                pass
            else:
                source = add_rms_norm_getitem(node)
                if source is None or not add_rms_norm_is_emittable(source):
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
        if node.target in DIM_ORDER_TARGETS and not dim_order_keeps_the_bytes(node):
            # A dim-order copy the alias emitter cannot stand in for is one the
            # portable kernels run, not one the DSP should read wrong.
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


def _data_placeholders(exported_program: ExportedProgram) -> Set[str]:
    """Placeholders the program owns rather than the caller hands in.

    Parameters, buffers and lifted constants reach the partitioner as ordinary
    graph inputs. Their names are what says which is which.

    A mutated buffer is excluded: tagging it would hand its bytes to the backend
    as delegate-owned data, which EXIR then drops from the delegate's arguments.
    For a KV cache that means the empty initial buffer is baked in as a weight
    and every execute() after the first reads a cache nothing ever wrote. Left
    untagged it stays a user input the runtime copies in and reads back out.
    """
    signature = exported_program.graph_signature
    mutated = set(signature.buffers_to_mutate.values())
    owned_buffers = {
        name
        for name, target in signature.inputs_to_buffers.items()
        if target not in mutated
    }
    return (
        set(signature.inputs_to_parameters)
        | owned_buffers
        | set(signature.inputs_to_lifted_tensor_constants)
    )


@final
class HexagonPartitioner(Partitioner):
    """Delegates every supported op, merging connected ones into one subgraph.

    Merging matters: each delegate subgraph costs one FastRPC round trip and its
    own arena, so a partition per node would be correct but pointless.
    """

    def __init__(self, compile_options: Optional[HexagonCompileOptions] = None) -> None:
        # Stamped into the .pte with the delegate, so preprocess and the runtime
        # both see the same choices and the blob can be checked against them.
        self.compile_options = compile_options or HexagonCompileOptions()
        self.delegation_spec = DelegationSpec(
            HexagonBackend.__name__, self.compile_options.to_compile_specs()
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        graph_module = exported_program.graph_module
        support = HexagonOperatorSupport(_data_placeholders(exported_program))

        supported = [
            node
            for node in graph_module.graph.nodes
            if support.is_node_supported(None, node)
        ]

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

        data_names = _data_placeholders(exported_program)
        for partition in partitions:
            delegation_tag = f"hexagon_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = delegation_tag
                # Constants consumed by a delegated node have to come along, or
                # they stay in the graph as inputs the delegate never sees.
                # Tagging a parameter, buffer or lifted constant is also what
                # makes it the delegate's own weight rather than an argument:
                # EXIR then hands its bytes to the backend and deletes it from
                # the call, so the runtime stores it once instead of copying it
                # into the arena on every execute. A constant another partition
                # also reads is duplicated for us.
                for arg in node.all_input_nodes:
                    if arg.op == "get_attr" or arg.name in data_names:
                        arg.meta["delegation_tag"] = delegation_tag
            partition_tags[delegation_tag] = self.delegation_spec

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
