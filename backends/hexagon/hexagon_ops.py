# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Emitters that turn edge ops into DSP commands.

Each emitter appends one command and returns the ref holding its result. The
parameter orders here are the DSP's, read from the dispatch in
third-party/mnn-htp-ops/src/dsp/execute_command.cc, and they are positional and
unchecked: getting one wrong produces wrong numbers rather than an error.
"""

import operator
import os
import struct
from typing import Dict, List, NamedTuple, Optional

import torch
from executorch.backends.hexagon.add_rms_norm import ADD_RMS_NORM
from executorch.backends.hexagon.kv_cache import UPDATE_CACHE
from executorch.backends.hexagon.rms_norm import RMS_NORM

# After rms_norm, which opens the et_hexagon namespace this fragment joins.
from executorch.backends.hexagon.mul_silu import MUL_SILU
from executorch.backends.hexagon.rope import ROPE
from executorch.backends.hexagon.row_guard import ROW_GUARD
from executorch.backends.hexagon.serialization.blob import ABSENT, Op, TensorRef
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.sym_util import eval_upper_bound

# DSPOpType, from third-party/mnn-htp-ops/include/htp_command.h.
DSP_OP_RASTER_BLIT = 3
DSP_OP_UNARY = 4
DSP_OP_LAYER_NORM = 8
DSP_OP_ROPE = 14
DSP_OP_ADD_FUSE_LAYERNORM = 16
DSP_OP_BINARY_ELEMENTWISE = 19
DSP_OP_SOFTMAX = 28
DSP_OP_REDUCTION = 29
DSP_OP_BATCH_MATMUL = 38
DSP_OP_FLASH_ATTN = 18

# HtpOpsReductionType, from the DSP's eltwise_ops.cc.
REDUCTION_MEAN = 3

# HtpOpsUnaryOpType, declared in the DSP's unary_ops.cc. Transcribed whole so
# the numbering can be checked against one place; only the entries with an
# emitter in EMITTERS below are reachable.
UNARY_OP_TYPES: Dict[str, int] = {
    "abs": 1,
    "neg": 2,
    "gelu": 3,
    "sigmoid": 4,
    "exp": 5,
    "log": 6,
    "silu": 7,
    "tanh": 8,
    "square": 9,
    "sqrt": 10,
    "rsqrt": 11,
    "expm1": 12,
    "cos": 13,
    "sin": 14,
    # clamp carries its bounds rather than a single op type: params[3] and
    # params[4] are the fp16 bit patterns of them, and the DSP dispatches it to
    # an entry point of its own instead of htp_ops_unary.
    "clamp": 15,
    # row_guard is the same arrangement: params[3] is the row length and
    # params[4] the masked value, and its entry point walks whole rows.
    "row_guard": 16,
    # mul_scalar carries the scale the same way: params[3] is the fp32 bit
    # pattern of a python float, because widening to fp32 is the whole point of
    # the op (see the emitter).
    "mul_scalar": 17,
}

# HtpOpsBinaryOpType, declared in the DSP's eltwise_ops.cc. Same reasoning as
# UNARY_OP_TYPES: the whole enum, but only some entries are reachable.
BINARY_OP_TYPES: Dict[str, int] = {
    "add": 1,
    "sub": 2,
    "mul": 3,
    "div": 4,
    "max": 5,
    "min": 6,
    "mul_silu": 7,
    "add_relu": 8,
    "greater": 9,
    "less": 10,
    "squared_difference": 11,
    "mod": 12,
}

# The DSP's default element format for these ops.
FP16_BYTES = 2


LAYER_NORM = exir_ops.edge.aten.layer_norm.default
NATIVE_LAYER_NORM = exir_ops.edge.aten.native_layer_norm.default
GETITEM = operator.getitem
SOFTMAX_TARGETS = frozenset({exir_ops.edge.aten._softmax.default})


def softmax_reduces_the_inner_axis(node: torch.fx.Node) -> bool:
    """Whether this is the last-axis softmax the DSP kernel is right for.

    The kernel's inside-greater-than-one path reduction over a strided span does
    not agree with torch on hardware: [1,2,4,8] reduced over dim 1 came back with
    eight of 64 elements past 1e-2, the worst by 1.1e-1, where the last-axis form
    is exact to 4.9e-4. Until that path is checked against the kernel, a softmax
    over anything but the last axis stays on the portable kernels.
    """
    return int(node.args[1]) in (-1, node.meta["val"].dim() - 1)


def _value_of(node: torch.fx.Node) -> torch.Tensor:
    """A node's value, or its first when the node hands out several.

    native_layer_norm returns (out, mean, rstd), so the node's own value is the
    tuple; the tensor the kernel reads is its first element.
    """
    value = node.meta["val"]
    return value[0] if isinstance(value, tuple) else value


def _numel(node: torch.fx.Node) -> int:
    return _value_of(node).numel()


def _upper_product(values, ctx=None) -> int:
    result = 1
    for value in values:
        result *= ctx.upper_dim(value) if ctx is not None else (
            eval_upper_bound(value) if isinstance(value, torch.SymInt) else int(value)
        )
    return result


def _patch_dynamic_product(ctx, op_index: int, values, param_index: int) -> None:
    """Patch a product containing the one exported sequence symbol."""
    for value in values:
        if ctx.is_dynamic_dim(value):
            scale = 1
            for dim in values:
                if not ctx.is_dynamic_dim(dim):
                    scale *= ctx.upper_dim(dim)
            ctx.add_dynamic_patch(op_index, param_index, scale, 0)
            return


def _dynamic_scale(ctx, value) -> int:
    if not ctx.is_dynamic_dim(value) or ctx.dynamic_sequence is None:
        return 1
    return 1


def _patch_dynamic_rows(ctx, op_index: int, param_index: int, rows: int, factors) -> None:
    """Patch a row count that is the exported length times static factors.

    A blit region's row count is the product of the shape the copy runs over.
    When exactly one of those entries is the sequence symbol the count is linear
    in the run length, and the runtime's patch recomputes it as
    `rows // example` per token. Anything else -- a constant cache copy, say --
    is left alone.
    """
    if ctx.dynamic_sequence is None or not ctx.dynamic_example:
        return
    if sum(ctx.is_dynamic_dim(factor) for factor in factors) != 1:
        return
    example = int(ctx.dynamic_example)
    if rows <= 0 or rows % example:
        return
    ctx.add_dynamic_patch(op_index, param_index, rows // example, 0)


def _patch_dynamic_numel(ctx, op_index: int, node: torch.fx.Node, param_index: int = 0) -> None:
    _patch_dynamic_product(ctx, op_index, tuple(_value_of(node).shape), param_index)


def _require_arena_dtype(node: torch.fx.Node, what: str) -> None:
    """The two widths the arena holds, both of which reach the same fp16 command.

    Every kernel reads and writes two bytes per element, and the runtime narrows
    a fp32 operand and widens a fp32 result at the boundary, so a node the graph
    declares fp32 is emitted exactly as its fp16 twin. Any other width would have
    the kernels reading those bits as half floats.
    """
    dtype = _value_of(node).dtype
    if dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"hexagon: {what} must be fp16 or fp32, got {dtype}")


def _float_bits(value: float) -> int:
    """Packs a float into an int32 param slot.

    DSP_OP_LAYER_NORM reads params[2] through a float* cast, so epsilon is bit
    pattern rather than a rounded integer.
    """
    return struct.unpack("<i", struct.pack("<f", value))[0]


class SliceRegion(NamedTuple):
    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz].
    # The offsets are in elements, and the source's layout is the caller's own.
    region: List[int]
    # The value the start row is read from when the caller only knows it at run
    # time, and what one row of the sliced dimension is worth in elements. Both
    # are None and 1 when the region is complete as it stands.
    patch_source: Optional[torch.fx.Node] = None
    patch_scale: int = 1


def slice_region(node: torch.fx.Node) -> Optional[SliceRegion]:
    """The blit region for a narrowing slice, or None when it is not one.

    Both the support check and the emitter call this, so they cannot disagree
    about which slices the DSP can run -- a node the emitter would refuse has to
    be refused here instead, or the whole export fails rather than falling back.

    A run per row is what a region describes, so a step is out. The extent comes
    from the result rather than from the end argument: they are the same number,
    the result's is the one that has to fit the buffer that was allocated, and
    taking it from there means an end the caller computes cannot disagree with
    the shape. A start the caller computes is the one part that cannot be known
    here, and it is a single word -- the source offset -- which the patch
    mechanism already knows how to fill at run time. That is what the RoPE
    frequency tables need: a table of 256 rows cut to the prompt, where the
    extent is fixed and only the row it starts at moves.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node) or len(node.args) < 4:
        return None
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return None
    dim, start = node.args[1], node.args[2]
    if not isinstance(dim, int) or isinstance(dim, bool):
        return None
    if len(node.args) > 4 and node.args[4] not in (None, 1):
        return None
    shape = list(source_value.shape)
    result = list(result_value.shape)
    if dim < 0:
        dim += len(shape)
    if not 0 <= dim < len(shape) or len(result) != len(shape):
        return None
    if any(result[d] != shape[d] for d in range(len(shape)) if d != dim):
        return None

    inner = 1
    for size in shape[dim + 1 :]:
        inner *= size
    rows = 1
    for size in shape[:dim]:
        rows *= size

    patch_source = None
    if isinstance(start, torch.fx.Node):
        patch_source = _scalar_source(start)
        if not isinstance(patch_source, torch.fx.Node):
            return None
        offset = 0
    elif isinstance(start, int) and not isinstance(start, bool):
        start = max(start, 0)
        if start + result[dim] > shape[dim]:
            return None
        offset = start * inner
    else:
        return None

    run = result[dim] * inner
    return SliceRegion(
        [0, offset, 0, 1, rows, run, 0, shape[dim] * inner, 1, 0, run, 1],
        patch_source,
        inner,
    )


def select_region(node: torch.fx.Node):
    """The blit region for a select_copy that narrows, or None.

    `select_copy(source, dim, index)` picks one entry along `dim`, which is the
    slice `[index, index + 1)` and nothing else, so it copies the same run a
    narrowing slice does. When the source is one wide along `dim` the result
    holds the operand's bytes and the alias path is the right one, so this
    returns None there and `_alias_keeps_the_same_bytes` decides.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node) or len(node.args) < 3:
        return None
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return None
    dim, index = node.args[1], node.args[2]
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (dim, index)):
        return None
    shape = list(source_value.shape)
    if dim < 0:
        dim += len(shape)
    if index < 0:
        index += shape[dim]
    if not 0 <= index < shape[dim]:
        return None
    if shape[dim] == 1 or source_value.numel() == result_value.numel():
        return None

    inner = 1
    for size in shape[dim + 1 :]:
        inner *= size
    rows = 1
    for size in shape[:dim]:
        rows *= size
    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
    return [0, index * inner, 0, 1, rows, inner, 0, shape[dim] * inner, 1, 0, inner, 1]


def _emit_select_copy(node: torch.fx.Node, ctx) -> TensorRef:
    """A select that narrows writes a buffer; one that does not re-points.

    Both forms reach the same target, so the two decisions are made here rather
    than by splitting the target in the op table.
    """
    region = select_region(node)
    if region is None:
        return _emit_alias(node, ctx)
    out = ctx.result_for(node, _numel(node))
    ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(node.args[0])],
            outputs=[out],
            params=[1, FP16_BYTES, 1] + region,
        )
    )
    return ctx.record(node, out)


def _emit_slice_copy(node: torch.fx.Node, ctx) -> TensorRef:
    """A narrowing slice cannot re-point its operand the way an alias does.

    The result holds fewer elements than the operand, so the kernel writes the
    selected run into a buffer of its own. When the start row is a run-time
    value it rides along as an extra input the blit never reads, and the runtime
    writes it into the region's source offset on the way in, scaled by what one
    row is worth in elements.
    """
    region = slice_region(node)
    inputs = [ctx.operand(node.args[0])]
    patch = None
    if region.patch_source is not None:
        inputs.append(ctx.operand(region.patch_source))
        patch = (_SLICE_OFFSET_PARAM, len(inputs) - 1)
    out = ctx.result_for(node, _numel(node))
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=inputs,
            outputs=[out],
            # region count, element bytes, source count, then the region.
            params=[1, FP16_BYTES, 1] + region.region,
            patch=patch,
            patch_scale=region.patch_scale,
        )
    )
    # Rows are the sliced dimension's leading shape, at index 7 past the header
    # and the region's three offsets.
    source_shape = _value_of(node.args[0]).shape
    dim = node.args[1]
    if dim < 0:
        dim += len(source_shape)
    _patch_dynamic_rows(ctx, op_index, 7, region.region[4], source_shape[:dim])
    return ctx.record(node, out)


# The region's source offset, counted from the start of the params vector: the
# blit header is three ints, then srcIndex, then srcOffset.
_SLICE_OFFSET_PARAM = 4

# A blit header is three ints and each region twelve, and an op's params vector
# holds blob.MAX_OP_PARAMS (40) of them, which leaves room for three inputs.
MAX_CAT_INPUTS = 3


def cat_region(node: torch.fx.Node):
    """The blit parameters for a concatenation along one axis, or None.

    Each input becomes one region writing into its own slice of the result, which
    is what a RoPE rejoin needs: two 64-element halves becoming one 128-element
    row. The region list is fixed when the command is built, so the split has to
    be known here -- a concatenation whose lengths are only known at the call
    cannot be described by it.
    """
    if not node.args or len(node.args) > 2:
        return None
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else 0
    if not isinstance(tensors, (list, tuple)):
        return None
    if not isinstance(dim, int) or isinstance(dim, bool):
        return None
    if not 1 <= len(tensors) <= MAX_CAT_INPUTS:
        return None
    if not all(isinstance(tensor, torch.fx.Node) for tensor in tensors):
        return None

    result = node.meta.get("val")
    values = [tensor.meta.get("val") for tensor in tensors]
    if not isinstance(result, torch.Tensor) or not all(
        isinstance(value, torch.Tensor) for value in values
    ):
        return None
    if not result.is_contiguous() or not all(value.is_contiguous() for value in values):
        return None
    if any(value.dtype is not torch.float16 for value in values):
        return None

    shape = list(result.shape)
    if dim < 0:
        dim += len(shape)
    if any(len(value.shape) != len(shape) for value in values):
        return None
    # Only the concatenated axis may differ, and it has to add up.
    if any(
        value.shape[other] != shape[other]
        for value in values
        for other in range(len(shape))
        if other != dim
    ):
        return None
    if sum(value.shape[dim] for value in values) != shape[dim]:
        return None

    inner = 1
    for size in shape[dim + 1 :]:
        inner *= size
    rows = 1
    for size in shape[:dim]:
        rows *= size
    combined = shape[dim] * inner

    params = [len(values), FP16_BYTES, len(values)]
    offset = 0
    for index, value in enumerate(values):
        run = value.shape[dim] * inner
        # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
        params += [index, 0, offset * inner, 1, rows, run, 0, run, 1, 0, combined, 1]
        offset += value.shape[dim]
    return params


def _emit_cat(node: torch.fx.Node, ctx) -> TensorRef:
    """Each operand is copied into its own slice of a fresh buffer."""
    tensors = node.args[0]
    params = cat_region(node)
    out = ctx.result_for(node, _numel(node))
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(tensor) for tensor in tensors],
            outputs=[out],
            params=params,
        )
    )
    # Every region repeats the row count at its own size1 slot, so each one has
    # to move with the length; the regions that mirror it are patched with it.
    result_shape = _value_of(node).shape
    dim = node.args[1] if len(node.args) > 1 else 0
    if dim < 0:
        dim += len(result_shape)
    for region_index in range(params[0]):
        rows_param = 3 + region_index * 12 + 4
        _patch_dynamic_rows(
            ctx, op_index, rows_param, params[rows_param], result_shape[:dim]
        )
    return ctx.record(node, out)


def _row_major_strides(shape) -> List[int]:
    """The strides a contiguous tensor of this shape is read at, in elements."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * int(shape[axis + 1])
    return strides


def permute_region(node: torch.fx.Node):
    """The blit region for a permutation of the axes, or None.

    A region is three nested loops with a stride per side, so a permutation is
    describable exactly when the axes split into at most three groups, each a run
    of consecutive axes that keeps its order in both layouts: such a run is one
    loop whose size is the product of the run and whose strides are its last
    axis's. `permute(w, [1, 0])` between a weight and its `mm` is the two-group
    case, and it is the shape `htp_ops_prepare_transpose` recognises and routes
    to the HVX transpose -- the inner run reads a contiguous source row and
    writes a strided destination column. `permute(1, 0, 2, 3)` over a fused qkv
    is the same split with the head run carried whole.

    A permutation that reverses the axes inside a group, or that needs a fourth
    one, is refused rather than approximated: no single region describes it, and
    what it would emit reads the wrong elements rather than failing.
    """
    if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
        return None
    source_value = node.args[0].meta.get("val")
    result_value = node.meta.get("val")
    dims = node.args[1]
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return None
    if not isinstance(dims, (list, tuple)):
        return None
    # Both widths the arena holds reach the same two-byte command.
    if source_value.dtype not in (torch.float16, torch.float32):
        return None
    if not source_value.is_contiguous() or not result_value.is_contiguous():
        return None

    rank = len(source_value.shape)
    if rank < 2 or len(dims) != rank or sorted(dims) != list(range(rank)):
        return None
    if source_value.numel() != result_value.numel():
        return None

    source_strides = _row_major_strides(source_value.shape)
    result_strides = _row_major_strides(result_value.shape)
    # Where each source axis lands, and so the stride the destination keeps it at.
    positions = [0] * rank
    for position, axis in enumerate(dims):
        positions[axis] = position
    destinations = [result_strides[position] for position in positions]

    # A group ends where the destination stops advancing one axis at a time.
    groups = []
    first = 0
    for axis in range(1, rank):
        if positions[axis] != positions[first] + (axis - first):
            groups.append((first, axis - 1))
            first = axis
    groups.append((first, rank - 1))
    if len(groups) > 3:
        return None

    levels = []
    for start, last in groups:
        run = 1
        for axis in range(start, last + 1):
            run *= int(source_value.shape[axis])
        levels.append((run, source_strides[last], destinations[last]))
    while len(levels) < 3:
        levels.append((1, 0, 0))
    # A run both sides read contiguously is one copy where it is innermost and
    # one two-byte copy per element where it is not, and the loop order does not
    # change which elements the region covers.
    for index, level in enumerate(levels):
        if level[1] == 1 and level[2] == 1 and index != 2:
            levels[index] = levels[2]
            levels[2] = level
            break

    size = [level[0] for level in levels]
    src = [level[1] for level in levels]
    dst = [level[2] for level in levels]
    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
    return [0, 0, 0] + size + src + dst


def _fold_constant_transpose(node: torch.fx.Node, ctx):
    """Transposes a constant weight at export instead of on the DSP.

    Every permute_copy in these graphs turns a weight round for the matmul that
    consumes it, so its result is a constant as well. Computing it here costs one
    pass and saves the DSP the rearrange on every inference, along with the
    weight-sized tensor it would otherwise hand across a delegate boundary.
    """
    if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
        return None
    if tuple(node.args[1]) != (1, 0):
        return None
    tensor = ctx.constant_value(node.args[0])
    if tensor is None or tensor.dim() != 2:
        return None
    # The arena holds two-byte elements, so a fp32 weight is narrowed here the
    # same way the runtime narrows one that arrives as a method input.
    return ctx.folded_weight(
        node, tensor.to(torch.float16).transpose(0, 1).contiguous()
    )


def _emit_permute_copy(node: torch.fx.Node, ctx) -> TensorRef:
    """A transpose is a strided read and a strided write, which is one region."""
    folded = _fold_constant_transpose(node, ctx)
    if folded is not None:
        return folded
    out = ctx.result_for(node, _numel(node))
    ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(node.args[0])],
            outputs=[out],
            params=[1, FP16_BYTES, 1] + permute_region(node),
        )
    )
    return ctx.record(node, out)


def dim_order_keeps_the_bytes(node: torch.fx.Node) -> bool:
    """Whether a dim-order copy only re-reads its operand's bytes.

    `to_edge` carries a memory format through the graph as a dim-order op, which
    names the layout by listing the axes rather than by giving strides. The arena
    holds row-major two-byte elements, so one whose order is the identity is a
    plain copy of the same elements in the same order: a widening or a narrowing
    when the widths differ, which the runtime does at the boundary, and nothing
    at all when they do not, which is why the alias emitter is the right one.

    Any other order describes a layout the arena does not hold, and re-pointing
    the operand would hand the consumer the wrong numbers rather than fail, so
    the support check refuses it and the node stays on a portable kernel.
    """
    if node.target not in DIM_ORDER_TARGETS:
        return False
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return False
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return False
    if source_value.dtype not in (torch.float16, torch.float32):
        return False
    if result_value.dtype not in (torch.float16, torch.float32):
        return False
    if source_value.dim() != result_value.dim():
        return False
    if source_value.numel() != result_value.numel():
        return False
    if not source_value.is_contiguous() or not result_value.is_contiguous():
        return False
    order = node.kwargs.get("dim_order")
    return order is None or list(order) == list(range(result_value.dim()))


def _unary(op_name: str):
    def emit(node: torch.fx.Node, ctx) -> TensorRef:
        src = node.args[0]
        _require_arena_dtype(node, f"unary {op_name} input")
        numel = _numel(node)
        out = ctx.result_for(node, numel)
        op_index = ctx.builder.add_op(
            Op(
                type=DSP_OP_UNARY,
                inputs=[ctx.operand(src)],
                outputs=[out],
                # size is in elements, not bytes.
                params=[_upper_product(tuple(_value_of(node).shape), ctx), UNARY_OP_TYPES[op_name], FP16_BYTES],
            )
        )
        _patch_dynamic_numel(ctx, op_index, node)
        return ctx.record(node, out)

    return emit


def _clamp_bound_bits(bound, unbounded: float) -> int:
    """A clamp bound as the fp16 bit pattern the DSP compares against.

    torch narrows a Python float bound to the tensor's dtype before comparing,
    so narrowing here is the same comparison the portable kernel makes; an
    omitted bound is an infinity, which fp16 holds exactly.
    """
    value = unbounded if bound is None else float(bound)
    return int(torch.tensor(value, dtype=torch.float16).view(torch.uint16).item())


def _emit_clamp(node: torch.fx.Node, ctx) -> TensorRef:
    src = node.args[0]
    _require_arena_dtype(node, "clamp input")
    numel = _numel(node)
    out = ctx.result_for(node, numel)
    # The bounds arrive positionally and are not always both there: a one-bound
    # clamp is min= that bound, and a clamp with no bounds at all is the
    # identity, which the infinities below express on their own.
    lower = node.args[1] if len(node.args) > 1 else None
    upper = node.args[2] if len(node.args) > 2 else None
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_UNARY,
            inputs=[ctx.operand(src)],
            outputs=[out],
            # size is in elements, not bytes, as in _unary.
            params=[
                _upper_product(tuple(_value_of(node).shape), ctx),
                UNARY_OP_TYPES["clamp"],
                FP16_BYTES,
                _clamp_bound_bits(lower, float("-inf")),
                _clamp_bound_bits(upper, float("inf")),
            ],
        )
    )
    _patch_dynamic_numel(ctx, op_index, node)
    return ctx.record(node, out)


def _emit_mul_scalar(node: torch.fx.Node, ctx) -> TensorRef:
    """A fp16 tensor times a python float, widened to fp32 to multiply.

    torch and the portable kernel both promote the scalar to fp32, multiply in
    fp32 and round the product back to fp16; a product of two fp16 values
    differs from that on about one element in six. Reproducing the fp32 step is
    what lets the op stay inside a partition instead of being handed out, which
    is what the graph is split on everywhere the attention scales its queries
    and keys.
    """
    values, scale = node.args
    _require_arena_dtype(node, "mul scalar")
    numel = _numel(node)
    out = ctx.result_for(node, numel)
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_UNARY,
            inputs=[ctx.operand(values)],
            outputs=[out],
            params=[
                _upper_product(tuple(_value_of(node).shape), ctx),
                UNARY_OP_TYPES["mul_scalar"],
                FP16_BYTES,
                struct.unpack("<i", struct.pack("<f", float(scale)))[0],
                0,
            ],
        )
    )
    _patch_dynamic_numel(ctx, op_index, node)
    return ctx.record(node, out)


def _emit_row_guard(node: torch.fx.Node, ctx) -> TensorRef:
    """The masked-row guard as one op over the logits and the values it guards.

    It walks rows of the last dimension -- the mask says which of them are
    entirely the masked value, the second operand holds what they select -- so
    beyond the size the DSP needs a row's length and that value.
    """
    mask, values, pad = node.args
    _require_arena_dtype(node, "row guard")
    numel = _numel(node)
    row = int(values.meta["val"].shape[-1])
    out = ctx.result_for(node, numel)
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_UNARY,
            inputs=[ctx.operand(mask), ctx.operand(values)],
            outputs=[out],
            params=[
                _upper_product(tuple(_value_of(node).shape), ctx),
                UNARY_OP_TYPES["row_guard"],
                FP16_BYTES,
                row,
                _clamp_bound_bits(pad, float("-inf")),
            ],
        )
    )
    _patch_dynamic_numel(ctx, op_index, node)
    return ctx.record(node, out)


def _shape_of(operand) -> tuple:
    """An operand's shape: a node's value, a caller-given tuple, or a literal's ().

    A literal is one element, which right-aligns and broadcasts the way a scalar
    tensor does.
    """
    if isinstance(operand, torch.fx.Node):
        return tuple(operand.meta["val"].shape)
    if isinstance(operand, (tuple, list)):
        return tuple(operand)
    return ()


def _broadcast_strides(shape, out_shape, ctx):
    """Row-major strides of an operand of this shape, zero on broadcast dims.

    The DSP walks the output linearly and computes each operand's offset as
    sum(coord[d] * stride[d]), so a dimension of extent 1 contributes nothing.
    """
    shape = ctx.upper_shape(shape)
    out_shape = ctx.upper_shape(out_shape)
    rank = len(out_shape)
    padded = (1,) * (rank - len(shape)) + tuple(shape)
    strides = [0] * rank
    acc = 1
    for d in range(rank - 1, -1, -1):
        strides[d] = 0 if padded[d] == 1 else acc
        acc *= padded[d]
    return strides


def _broadcast_tail(lhs, rhs, out_shape, ctx):
    """The 25 params the DSP's broadcast path reads from params[8].

    Each operand is right-aligned against the output, which is torch's own rule,
    so a broadcast operand's offsets come out the same either way.
    """
    rank = len(out_shape)
    pad = 8 - rank
    return (
        [rank]
        + list(out_shape)
        + [0] * pad
        + _broadcast_strides(_shape_of(lhs), out_shape, ctx)
        + [0] * pad
        + _broadcast_strides(_shape_of(rhs), out_shape, ctx)
        + [0] * pad
    )


def _binary(op_name: str):
    def emit(node: torch.fx.Node, ctx) -> TensorRef:
        lhs, rhs = node.args[0], node.args[1]
        _require_arena_dtype(node, f"binary {op_name} input")

        lhs_numel = (
            _upper_product(tuple(_value_of(lhs).shape), ctx)
            if isinstance(lhs, torch.fx.Node)
            else 1
        )
        rhs_numel = (
            _upper_product(tuple(_value_of(rhs).shape), ctx)
            if isinstance(rhs, torch.fx.Node)
            else 1
        )
        out_numel = _numel(node)
        out_shape = tuple(node.meta["val"].shape)
        out_shape_plan = ctx.upper_shape(out_shape)

        out = ctx.result_for(node, out_numel)
        op_index = ctx.builder.add_op(
            Op(
                type=DSP_OP_BINARY_ELEMENTWISE,
                inputs=[ctx.operand(lhs), ctx.operand(rhs)],
                outputs=[out],
                params=[
                    _upper_product(out_shape, ctx),
                    lhs_numel,
                    rhs_numel,
                    BINARY_OP_TYPES[op_name],
                    FP16_BYTES,
                    FP16_BYTES,
                    0,  # inputs are not 4-byte floats
                    0,  # output is not a 4-byte float
                    *_broadcast_tail(lhs, rhs, out_shape_plan, ctx),
                ],
            )
        )
        _patch_dynamic_product(ctx, op_index, out_shape, 0)
        if isinstance(lhs, torch.fx.Node):
            _patch_dynamic_product(ctx, op_index, tuple(lhs.meta["val"].shape), 1)
        if isinstance(rhs, torch.fx.Node):
            _patch_dynamic_product(ctx, op_index, tuple(rhs.meta["val"].shape), 2)
        for axis, size in enumerate(out_shape):
            if ctx.is_dynamic_dim(size):
                ctx.add_dynamic_patch(op_index, 9 + axis, _dynamic_scale(ctx, size), 0)
        return ctx.record(node, out)

    return emit


# HtpOpsLoopParam, from the DSP's region_ops.h. The struct is packed, so its
# three int64 tails sit at byte offsets 76/84/92 with no padding: "<19i3q2i"
# reproduces that layout, including the two int32 fields the host plan lives in,
# and "<27i" re-reads it as the int32 words the param vector carries.
_LOOP_PARAM = struct.Struct("<19i3q2i")

# hmxFlags: the magic marks the two plan fields as present, so a command that
# predates them reads as unplanned rather than as garbage. Bit 0 says the weights
# are already in the unit's tile order.
_HMX_PLAN_MAGIC = 0x484D58
_HMX_PLAN_PREPACKED_WEIGHTS = 1


def _loop_param(
    loop_number: int,
    size_xyz,
    dst_stride,
    src0_stride,
    src1_stride,
    out_elems: int,
    in0_elems: int,
    in1_elems: int,
    steps=(0, 0, 0),
    hmx_prepacked: bool = False,
    hmx_tile_budget: int = 0,
):
    """The descriptor BATCH_MATMUL reads out of params[1:].

    Sizes are in elements and strides in bytes. The DSP settles that by
    dividing the stride terms by the element size before bounds-checking them.

    With the three iter operands absent the DSP numbers every iteration itself,
    so an iteration's base is steps[i] * iter, in elements: one stride per
    operand, which is how a batch walks from one tile to the next.
    """
    packed = _LOOP_PARAM.pack(
        loop_number,
        *size_xyz,
        *dst_stride,
        *src0_stride,
        *src1_stride,
        *steps,
        0,
        0,
        0,  # cmdViewOffset: every operand starts at its own base
        out_elems,
        in0_elems,
        in1_elems,
        _HMX_PLAN_MAGIC | (_HMX_PLAN_PREPACKED_WEIGHTS if hmx_prepacked else 0),
        hmx_tile_budget,
    )
    return list(struct.unpack("<27i", packed))


def _emit_alias(node: torch.fx.Node, ctx) -> TensorRef:
    """A node that only re-reads its operand's bytes emits no command.

    Sharing one TensorRef is what a view is: the kernels index the same buffer
    under a different shape. The support check admits a node here only when the
    operand and the result are both contiguous with the same element count, so
    the bytes are already in the order the result describes.
    """
    source = ctx.operand(node.args[0])
    if not ctx.is_method_output(node):
        return ctx.record(node, source)

    # An output slot is read as a buffer of its own, so re-pointing the operand
    # is not enough here: the bytes have to be written out.
    numel = _numel(node)
    out = ctx.result_for(node, numel)
    ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[source],
            outputs=[out],
            params=[1, FP16_BYTES, 1, 0, 0, 0, 1, 1, numel, 0, 0, 1, 0, 0, 1],
        )
    )
    return ctx.record(node, out)


def update_cache_layout(node: torch.fx.Node):
    """The cache-advance lowering's operands and geometry, or None.

    Shared by the support predicate and the emitter, so a node the predicate
    accepts cannot reach an emitter that does not know what to do with it.
    """
    if node.target is not UPDATE_CACHE:
        return None
    cache, value, position = node.args[0], node.args[1], node.args[2]
    if not all(isinstance(part, torch.fx.Node) for part in (cache, value, position)):
        return None
    cache_val, value_val = cache.meta["val"], value.meta["val"]
    if cache_val.dtype != torch.float16 or value_val.dtype != torch.float16:
        return None
    if not (cache_val.is_contiguous() and value_val.is_contiguous()):
        return None
    if cache_val.dim() < 3 or value_val.dim() != cache_val.dim():
        return None
    # Only the sequence axis may differ: the cache is written a position at a
    # time, so a head or a head width that does not line up is a different op.
    if value_val.shape[2:] != cache_val.shape[2:]:
        return None
    run = cache_val.shape[-1]
    inner = cache_val.numel() // cache_val.shape[-3]
    rows = value_val.numel() // run
    if inner % run:
        return None
    return cache, value, position, run, inner, rows


def _emit_update_cache(node: torch.fx.Node, ctx) -> TensorRef:
    """Advances a KV cache, as two blits.

    The graph reads the cache back whole, so an output has to hold all of it
    either way. Copying it out and writing the new rows into the copy is what
    this does; mutating the input and handing that back would move the same
    bytes twice, which is what in_place exists for and not what this needs.
    """
    cache_node, value_node, pos_node, run, inner, rows = update_cache_layout(node)
    cache = ctx.operand(cache_node)
    value = ctx.operand(value_node)
    pos = ctx.operand(pos_node)

    numel = _numel(node)
    out = ctx.result_for(node, numel)
    ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[cache],
            outputs=[out],
            params=[
                1,
                FP16_BYTES,
                1,
                0,
                0,
                0,
                1,
                numel // run,
                run,
                0,
                run,
                1,
                0,
                run,
                1,
            ],
        )
    )
    # dstOffset is the token position times one whole cached position, which is
    # what patch_scale is for. dst is the output, at index 3.
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[value, cache, pos],
            outputs=[out],
            params=[1, FP16_BYTES, 3, 0, 0, 0, 1, rows, run, 0, run, 1, 0, run, 1],
            patch=(5, 2),
            patch_scale=inner,
        )
    )
    # The rows this writes are the value's tokens, so they scale with the
    # length; the full-cache copy above does not and is left alone.
    value_shape = _value_of(value_node).shape
    _patch_dynamic_rows(ctx, op_index, 7, rows, value_shape[:-1])
    return ctx.record(node, out)


def hmx_prefers_general(k: int, n: int) -> bool:
    """Whether the DSP routes this matmul to the general HMX kernel.

    The DSP answers this with MNN_MATMUL_PREFER_HMX in loop_ops.cc, which this
    tree sets to 1, so every shape takes that kernel. It is the only kernel that
    can read a weight in the unit's own tile order, and that order is what lets
    the unit stream the weight out of DDR instead of the DSP copying it into
    VTCM first, so there is nothing to gain from leaving a shape out. Two places
    answer the same question; they have to agree.
    """
    del k, n
    return True


# The general kernel is the only one that reads the unit's tile order, and it
# takes a region only once the region's work, E*K*N, reaches this many elements:
# loop_ops.cc htp_ops_loop_matmul_hmx_general_eligible. Two places answer the
# same question, so they have to agree.
HMX_GENERAL_MIN_ELEMS = 32768


def hmx_general_eligible(m: int, k: int, n: int) -> bool:
    """Whether the DSP's general HMX kernel will take an (m, k) @ (k, n) region.

    A command whose weight is already tiled and whose shape this kernel refuses
    has no kernel left that can read its operand: the fallbacks walk the weight
    row-major, so the packed bytes are rubbish to them and the output keeps
    whatever its buffer held. The host has to ask before it packs, not after.
    """
    return m * k * n >= HMX_GENERAL_MIN_ELEMS


def _hmx_prepack_enabled() -> bool:
    """HEXAGON_HMX_PREPACK=0 leaves matmul weights in their row-major order.

    On by default. Packing a weight here forces that matmul onto the HMX route,
    because no other kernel reads that layout, and it costs the model nothing:
    the reordered bytes are the same size, so the blob is unchanged in size and
    the output is bit-identical.

    What it buys is the DSP's weight fill. A row-major (k, n) weight gives the
    unit 32 rows of 64 bytes at a pitch of n * 2, which is a stride no prefetcher
    likes; the tile order turns the same read into one straight run per 32-column
    group. On the 24-layer vision tower that is the difference between 430 and
    315 ms an image, and the DSP call drops from 465 to 310 ms.
    """
    import os

    return os.environ.get("HEXAGON_HMX_PREPACK", "") not in ("0", "false")


def _hmx_tile_budget() -> int:
    """HEXAGON_HMX_TILE_BUDGET caps the 32-column weight tiles held in VTCM.

    A cap, not a size: the DSP fits itself to the VTCM it actually finds, so
    planning for more than the part has cannot overflow it.
    """
    import os

    return int(os.environ.get("HEXAGON_HMX_TILE_BUDGET", "0"))


def _hmx_weight_operand(ctx, rhs, k: int, n: int):
    """The weight operand: packed at export when the graph stores it as one."""
    if getattr(rhs, "target", None) in ctx.packed_targets:
        return ctx.operand(rhs), True
    return ctx.operand(rhs), False


def pack_hmx_weight(weight, k: int, n: int) -> bytes:
    """A (k, n) fp16 weight in the tile order the DSP would pack it into VTCM.

    Tile (nt, kt) holds rows kt*32..+32 against columns nt*32..+32, and inside a
    tile the unit reads k interleaved in pairs: element (k, c) sits at
    (k // 2) * 64 + c * 2 + (k & 1), which is exactly what
    htp_ops_loop_hmx_pack_weight_tile writes. Tiles run nt-major, so the kp tiles
    of one 32-column group are contiguous and the DSP streams a group in one copy.
    """
    import numpy as np

    w = weight.astype(np.float16, copy=False)
    if w.shape != (k, n):
        raise RuntimeError(f"hexagon: weight is {w.shape}, expected ({k}, {n})")
    kp = -(-k // 32)
    nt_total = -(-n // 32)
    padded = np.zeros((kp * 32, nt_total * 32), dtype=np.float16)
    padded[:k, :n] = w
    tiles = padded.reshape(kp, 32, nt_total, 32).transpose(2, 0, 1, 3)
    tiles = tiles.reshape(nt_total, kp, 16, 2, 32).transpose(0, 1, 2, 4, 3)
    return np.ascontiguousarray(tiles).tobytes()


def _weight_operand(ctx, rhs, m: int, k: int, n: int):
    """The matmul's weight operand, in the unit's tile order where that pays.

    Returns (ref, prepacked) so the command can say which one its operand holds.
    A weight whose value this layer can see is stored rearranged and read
    straight out of the weights section by the unit itself; one it cannot see is
    left alone and the DSP rearranges a tile at a time per inference.

    The region the DSP judges is one (m, k, n) tile. A batched matmul is one
    command per batch element -- loopNumber is the batch and E the tile's rows --
    so a batch of small tiles is a batch of small regions, and the bar is m*K*N
    rather than the batch's total work.
    """
    value = ctx.constant_value(rhs)
    if (
        _hmx_prepack_enabled()
        and hmx_prefers_general(k, n)
        and hmx_general_eligible(m, k, n)
        and value is not None
        and value.dim() == 2
        and tuple(value.shape) == (k, n)
    ):
        return ctx.packed_weight(rhs, value, k, n), True
    return ctx.operand(rhs), False


def _matmul_command(
    ctx, lhs, rhs, out, batches: int, m: int, k: int, n: int,
    hmx_prepacked: bool = False, hmx_tile_budget: int = 0,
) -> None:
    """One BATCH_MATMUL over contiguous (m, k) @ (k, n) tiles.

    The DSP takes dst from mapped_ptrs[inputs->size()] and reads iter0..2 out of
    slots 2..4, so the three unused iterators still have to be present as absent
    operands rather than dropped. A batch is the same tile geometry repeated,
    which the steps in the descriptor walk: one whole tile of each operand per
    iteration. A single iteration is mm, where those steps are unreachable and
    stay zero.
    """
    m_plan = ctx.upper_bound(m)
    k_plan = ctx.upper_bound(k)
    n_plan = ctx.upper_bound(n)
    steps = (m_plan * n_plan, m_plan * k_plan, k_plan * n_plan) if batches > 1 else (0, 0, 0)
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_BATCH_MATMUL,
            inputs=[lhs, rhs, ABSENT, ABSENT, ABSENT],
            outputs=[out],
            params=[FP16_BYTES]
            + _loop_param(
                batches,
                (m_plan, k_plan, n_plan),
                (n_plan * FP16_BYTES, 0, FP16_BYTES),
                (k_plan * FP16_BYTES, FP16_BYTES, 0),
                (0, n_plan * FP16_BYTES, FP16_BYTES),
                batches * m_plan * n_plan,
                batches * m_plan * k_plan,
                batches * k_plan * n_plan,
                steps=steps,
                hmx_prepacked=hmx_prepacked,
                hmx_tile_budget=hmx_tile_budget,
            ),
        )
    )
    if ctx.is_dynamic_dim(m):
        m_scale = _dynamic_scale(ctx, m)
        ctx.add_dynamic_patch(op_index, 2, m_scale, 0)
        ctx.add_dynamic_patch(op_index, 20, batches * n * m_scale, 0)
        ctx.add_dynamic_patch(op_index, 21, batches * k * m_scale, 0)


def _emit_mm(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.mm as BATCH_MATMUL with a single loop iteration."""
    lhs, rhs = node.args[0], node.args[1]
    _require_arena_dtype(node, "mm")
    lhs_val, rhs_val = lhs.meta["val"], rhs.meta["val"]
    if not (lhs_val.is_contiguous() and rhs_val.is_contiguous()):
        raise RuntimeError("hexagon: mm operands must be contiguous; strides come from shape")
    m, k = lhs_val.shape
    contracted, n = rhs_val.shape
    if k != contracted:
        raise RuntimeError(f"hexagon: mm contracts {k} against {contracted}")

    weight, prepacked = _weight_operand(ctx, rhs, ctx.upper_bound(m), ctx.upper_bound(k), ctx.upper_bound(n))
    out = ctx.result_for(node, m * n)
    _matmul_command(ctx, ctx.operand(lhs), weight, out, 1, m, k, n, hmx_prepacked=prepacked,
                    hmx_tile_budget=_hmx_tile_budget())
    return ctx.record(node, out)


def _emit_bmm(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.bmm as one BATCH_MATMUL, one iteration per batch element.

    Both operands are contiguous stacks of tiles, so each step is one whole
    tile of the operand it belongs to.
    """
    lhs, rhs = node.args[0], node.args[1]
    _require_arena_dtype(node, "bmm")
    lhs_val, rhs_val = lhs.meta["val"], rhs.meta["val"]
    if not (lhs_val.is_contiguous() and rhs_val.is_contiguous()):
        raise RuntimeError("hexagon: bmm operands must be contiguous; strides come from shape")
    batches, m, k = lhs_val.shape
    rhs_batches, contracted, n = rhs_val.shape
    if batches != rhs_batches or k != contracted:
        raise RuntimeError(
            f"hexagon: bmm contracts {batches}x{k} against {rhs_batches}x{contracted}"
        )

    weight, prepacked = _weight_operand(ctx, rhs, ctx.upper_bound(m), ctx.upper_bound(k), ctx.upper_bound(n))
    out = ctx.result_for(node, batches * m * n)
    _matmul_command(ctx, ctx.operand(lhs), weight, out, batches, m, k, n, hmx_prepacked=prepacked,
                    hmx_tile_budget=_hmx_tile_budget())
    return ctx.record(node, out)


def _emit_addmm(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.addmm as the matmul plus the bias the graph would add after it.

    The kernel has no bias operand, so the sum is a second command. The bias is
    one row or one column over the tile the matmul writes, and the broadcast
    path the other binary ops already use repeats it with a zero stride on the
    dimensions it is broadcast along.
    """
    bias, lhs, rhs = node.args[0], node.args[1], node.args[2]
    beta = _scalar_arg(node, "beta", 3, 1.0)
    _require_arena_dtype(node, "addmm")
    m, k = lhs.meta["val"].shape
    contracted, n = rhs.meta["val"].shape
    if k != contracted:
        raise RuntimeError(f"hexagon: addmm contracts {k} against {contracted}")
    if beta is None or beta not in (0.0, 1.0):
        raise RuntimeError(f"hexagon: addmm beta {beta} is neither 0 nor 1")

    out = ctx.result_for(node, m * n)
    # beta == 0 folds the bias away and leaves mm, so there is nothing to add.
    target = out if beta == 0.0 else ctx.activation_for_shape((m, n))
    weight, prepacked = _weight_operand(ctx, rhs, ctx.upper_bound(m), ctx.upper_bound(k), ctx.upper_bound(n))
    _matmul_command(ctx, ctx.operand(lhs), weight, target, 1, m, k, n, hmx_prepacked=prepacked,
                    hmx_tile_budget=_hmx_tile_budget())
    if beta == 0.0:
        return ctx.record(node, out)

    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_BINARY_ELEMENTWISE,
            inputs=[target, ctx.operand(bias)],
            outputs=[out],
            params=[
                ctx.upper_bound(m) * ctx.upper_bound(n),
                ctx.upper_bound(m) * ctx.upper_bound(n),
                _numel(bias) if isinstance(bias, torch.fx.Node) else 1,
                BINARY_OP_TYPES["add"],
                FP16_BYTES,
                FP16_BYTES,
                0,  # inputs are not 4-byte floats
                0,  # output is not a 4-byte float
                *_broadcast_tail((m, n), bias, (m, n), ctx),
            ],
        )
    )
    if ctx.is_dynamic_dim(m):
        ctx.add_dynamic_patch(op_index, 0, n, 0)
        ctx.add_dynamic_patch(op_index, 1, n, 0)
    return ctx.record(node, out)


def _emit_mean_dim(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.mean.dim as a single REDUCTION.

    The DSP collapses one contiguous (outside, reduce, inside) span, so the
    reduced dims have to be adjacent; the caller's support check enforces that.
    """
    src = node.args[0]
    _require_arena_dtype(node, "mean")
    dims = node.args[1]
    dims = [dims] if isinstance(dims, int) else list(dims)
    shape = src.meta["val"].shape
    rank = len(shape)
    norm = sorted(d % rank for d in dims)

    def _prod(values) -> int:
        n = 1
        for v in values:
            n *= v
        return n

    out = ctx.result_for(node, _numel(node))
    ctx.builder.add_op(
        Op(
            type=DSP_OP_REDUCTION,
            inputs=[ctx.operand(src)],
            outputs=[out],
            params=[
                _prod(shape[: norm[0]]),
                _prod(shape[norm[0] : norm[-1] + 1]),
                _prod(shape[norm[-1] + 1 :]),
                REDUCTION_MEAN,
                FP16_BYTES,
            ],
        )
    )
    return ctx.record(node, out)


def _emit_softmax(node: torch.fx.Node, ctx) -> TensorRef:
    src = node.args[0]
    _require_arena_dtype(node, "softmax input")
    dim = int(node.args[1])
    shape = list(node.meta["val"].shape)

    if dim < 0:
        dim += len(shape)
    outside = _upper_product(shape[:dim], ctx)
    channel = ctx.upper_bound(shape[dim])
    inside = _upper_product(shape[dim + 1 :], ctx)

    numel = _numel(node)
    out = ctx.result_for(node, numel)
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_SOFTMAX,
            inputs=[ctx.operand(src)],
            outputs=[out],
            # The DSP reduces the middle axis of an [outside][channel][inside]
            # view, so the reduction dim is described by its strides.
            params=[outside, channel, inside, FP16_BYTES],
        )
    )
    _patch_dynamic_product(ctx, op_index, shape[:dim], 0)
    _patch_dynamic_product(ctx, op_index, [shape[dim]], 1)
    _patch_dynamic_product(ctx, op_index, shape[dim + 1 :], 2)
    return ctx.record(node, out)


def _emit_rms_norm(node: torch.fx.Node, ctx) -> TensorRef:
    """One command for the whole norm, which is what the DSP kernel wants.

    The kernel reads fp16 activations and fp32 gamma and accumulates in fp32,
    which is the arithmetic norm.py asks for. beta is ABSENT rather than a
    zero-size tensor: RMSNorm has no bias, and a zero-size operand still maps to
    a live address the kernel would read as data.
    """
    source, weight, eps = node.args
    _require_arena_dtype(node, "rms_norm input")
    shape = tuple(node.meta["val"].shape)
    inner = ctx.upper_bound(shape[-1])
    outer = _upper_product(shape[:-1], ctx)
    out = ctx.result_for(node, _numel(node))
    # The kernel applies gamma itself and reads it as fp32, so the weight is
    # stored at that width rather than in a following elementwise multiply.
    gamma = ctx.constant(weight, torch.float32)
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_LAYER_NORM,
            # beta is ABSENT: RMSNorm has no bias, and a zero-size operand still
            # maps to a live address the kernel would read as data.
            inputs=[ctx.operand(source), gamma, ABSENT],
            outputs=[out],
            params=[outer, inner, _float_bits(float(eps)), 1],
        )
    )
    _patch_dynamic_product(ctx, op_index, shape[:-1], 0)
    return ctx.record(node, out)


def _emit_rope(node: torch.fx.Node, ctx) -> TensorRef:
    """The rotate-half embedding as one command, per tensor.

    The command takes q and k together, but the graph rotates them in separate
    partitions, so the k operand is the same tensor with `kv_num_head = 0`: the
    kernel's k loop runs zero times and never reads or writes it. The table is
    the graph's own `[seq, head_dim]` cos/sin, whose first half is the even
    angles and whose second half is the odd ones, which is the layout the
    dispatcher reconstructs `cos_odd` from.

    The kernel walks tokens as the leading axis and heads inside a token, so the
    geometry is `[seq, num_head, head_dim]`; the graph's one-wide batch axis
    folds into the sequence.
    """
    source, cos, sin = node.args
    _require_arena_dtype(node, "rope input")
    shape = tuple(node.meta["val"].shape)
    head_dim = ctx.upper_bound(shape[-1])
    num_head = ctx.upper_bound(shape[-2])
    batch_seq = _upper_product(shape[:-2], ctx)
    out = ctx.result_for(node, _numel(node))
    # q, k, cos, sin; k is q with kv_num_head = 0, so it is inert.
    inputs = [
        ctx.operand(source),
        ctx.operand(source),
        ctx.operand(cos),
        ctx.operand(sin),
    ]
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_ROPE,
            inputs=inputs,
            outputs=[out, out],
            params=[batch_seq, num_head, 0, head_dim, head_dim, 0],
        )
    )
    _patch_dynamic_product(ctx, op_index, shape[:-2], 0)
    return ctx.record(node, out)


def _add_rms_norm_getitem_user(node: torch.fx.Node, index: int):
    """The getitem reading one of this fused node's two outputs, if present."""
    return next(
        (
            user
            for user in node.users
            if user.target is GETITEM and len(user.args) == 2 and user.args[1] == index
        ),
        None,
    )


def add_rms_norm_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The fused add+norm a getitem reads, for either of its two outputs.

    Like the layer norm, this op hands out several tensors, so a getitem is the
    only reader the DSP can carry. Which output it names decides which of the
    command's two pointers the emitter fills.
    """
    if node.target is not GETITEM or len(node.args) != 2:
        return None
    source, index = node.args
    if not isinstance(source, torch.fx.Node) or index not in (0, 1):
        return None
    return source if source.target is ADD_RMS_NORM else None


def add_rms_norm_is_emittable(node: torch.fx.Node) -> bool:
    """Whether every reader of this fused op takes one of the two outputs it writes.

    The command produces the normalized tensor and the residual sum together; a
    reader that is not a getitem would be left holding a tuple no kernel can be
    handed, so the whole node stays off the DSP.
    """
    return bool(node.users) and all(
        add_rms_norm_getitem(reader) is node for reader in node.users
    )


def _emit_add_rms_norm(node: torch.fx.Node, ctx) -> TensorRef:
    """One command for the residual add and the norm that reads it.

    The kernel adds the two fp16 operands, writes the sum to `add_out`, and
    normalizes it in fp32 with gamma (fp32) -- the RMSNorm flavor, selected by
    the last param and a null beta. Output order is the command's: the
    normalized tensor is mapped_ptrs[inputs], the residual sum the one after,
    so the two getitems name them in that order.
    """
    residual, branch, weight, eps = node.args
    _require_arena_dtype(node, "add_rms_norm input")
    norm_value, add_value = node.meta["val"]
    shape = tuple(norm_value.shape)
    inner = ctx.upper_bound(shape[-1])
    outer = _upper_product(shape[:-1], ctx)
    numel = _upper_product(shape, ctx)

    norm_sink = _add_rms_norm_getitem_user(node, 0)
    add_sink = _add_rms_norm_getitem_user(node, 1)
    normalized = (
        ctx.result_for(norm_sink, numel)
        if norm_sink is not None
        else ctx.activation_for_shape(shape)
    )
    residual_out = (
        ctx.result_for(add_sink, numel)
        if add_sink is not None
        else ctx.activation_for_shape(shape)
    )

    # The kernel applies gamma itself and reads it as fp32.
    gamma = ctx.constant(weight, torch.float32)
    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_ADD_FUSE_LAYERNORM,
            # beta is ABSENT: RMSNorm has no bias and the kernel's RMSNorm path
            # only fires when it is null.
            inputs=[ctx.operand(residual), ctx.operand(branch), gamma, ABSENT],
            outputs=[normalized, residual_out],
            params=[outer, inner, _float_bits(float(eps)), 1],
        )
    )
    _patch_dynamic_product(ctx, op_index, shape[:-1], 0)
    for sink, ref in ((norm_sink, normalized), (add_sink, residual_out)):
        if sink is not None:
            ctx.record(sink, ref)
    return ctx.record(node, normalized)


def layer_norm_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The layer norm a getitem reads, when it reads the first output.

    native_layer_norm returns (out, mean, rstd) and the DSP has a command for
    out alone, so getitem 0 is the only reader a partition can carry.
    """
    if node.target is not GETITEM or len(node.args) != 2:
        return None
    source, index = node.args
    if index != 0 or not isinstance(source, torch.fx.Node):
        return None
    return source if source.target is NATIVE_LAYER_NORM else None


def layer_norm_is_emittable(node: torch.fx.Node) -> bool:
    """Whether every reader of this layer norm takes the output the kernel writes.

    mean and rstd come out of the same node and nothing here produces them, so a
    graph that reads either one keeps the whole node on the portable kernels --
    along with the readers of out, which would otherwise be left holding a tuple
    no kernel can be handed.
    """
    return bool(node.users) and all(
        layer_norm_getitem(reader) is node for reader in node.users
    )


def layer_norm_normalizes_the_trailing_dims(node: torch.fx.Node) -> bool:
    """Whether the kernel's outer-times-inner view of the input is this norm's.

    The command describes the norm as one inner span repeated outer times, which
    only matches a normalized shape covering the trailing dims; any other shape
    would read a span the kernel was never told about.
    """
    shape = list(_value_of(node).shape)
    normalized = list(node.args[1])
    if len(normalized) > len(shape):
        return False
    return [int(size) for size in normalized] == shape[len(shape) - len(normalized):]


def _emit_layer_norm(node: torch.fx.Node, ctx) -> TensorRef:
    src = node.args[0]
    normalized_shape = node.args[1]
    positional = dict(enumerate(node.args))
    weight = node.kwargs.get("weight", positional.get(2))
    bias = node.kwargs.get("bias", positional.get(3))
    eps = _scalar_arg(node, "eps", 4, 1e-5)

    _require_arena_dtype(node, "layer_norm input")
    value = _value_of(node)
    shape = list(value.shape)
    normalized_shape = tuple(normalized_shape)
    inner = _upper_product(normalized_shape, ctx)
    outer_shape = tuple(shape[: len(shape) - len(normalized_shape)])
    outer = _upper_product(outer_shape, ctx)

    # native_layer_norm hands its result on through a getitem, and that getitem
    # is what downstream reads, so its output slot is the one to fill.
    sink = next(
        (reader for reader in node.users if layer_norm_getitem(reader) is node),
        node,
    )
    numel = _upper_product(shape, ctx)
    out = ctx.result_for(sink, numel)

    # The kernel reads gamma and beta as fp32 and the affine step cannot be
    # baked in: a subgraph carries no tensors, its weights arrive as delegate
    # inputs at the width the arena holds. So the norm runs without them and the
    # affine is the same pair of elementwise commands rms_norm uses for its
    # scale. The cost is one rounding to fp16 before the multiply and one after
    # the add; the kernels are fp16 in and out regardless.
    affine = [(arg, kind) for arg, kind in ((weight, "mul"), (bias, "add")) if arg is not None]
    normalized = ctx.activation_for_shape(shape) if affine else out
    norm_op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_LAYER_NORM,
            # Order matters: dst is the DSP's mapped_ptrs[3], so the three
            # inputs must be exactly src, gamma, beta -- null here, which the
            # kernel tests for and skips.
            inputs=[ctx.operand(src), ABSENT, ABSENT],
            outputs=[normalized],
            params=[outer, inner, _float_bits(eps), 0],
        )
    )
    _patch_dynamic_product(ctx, norm_op_index, outer_shape, 0)

    result = normalized
    for index, (arg, kind) in enumerate(affine):
        target = out if index == len(affine) - 1 else ctx.activation_for_shape(shape)
        affine_op_index = ctx.builder.add_op(
            Op(
                type=DSP_OP_BINARY_ELEMENTWISE,
                inputs=[result, ctx.operand(arg)],
                outputs=[target],
                params=[
                    numel,
                    numel,
                    _numel(arg),
                    BINARY_OP_TYPES[kind],
                    FP16_BYTES,
                    FP16_BYTES,
                    0,  # inputs are not 4-byte floats
                    0,  # output is not a 4-byte float
                    *_broadcast_tail((outer, inner), tuple(arg.meta["val"].shape), (outer, inner), ctx),
                ],
            )
        )
        _patch_dynamic_product(ctx, affine_op_index, outer_shape, 0)
        _patch_dynamic_product(ctx, affine_op_index, outer_shape, 1)
        result = target

    return ctx.record(node, out)


def _emit_getitem(node: torch.fx.Node, ctx) -> TensorRef:
    """Re-points the layer norm result, the only getitem this backend takes."""
    source = node.args[0]
    if isinstance(source, torch.fx.Node) and source.target is ADD_RMS_NORM:
        # The fused add+norm records both getitems itself, at the output each
        # index names; re-pointing to the op would collapse them to one.
        return ctx.producer[node]
    return ctx.record(node, ctx.operand(source))


def _scalar_arg(node: torch.fx.Node, name: str, index: int, default: float):
    """A scalar argument from wherever the graph put it, or None if it is not one.

    The graph keeps a keyword argument in kwargs and a positional one in args,
    and the two are the same number to the op; a value that is neither is a
    run-time tensor, which no emitter here can fold. None says so rather than
    reporting the default, so a caller cannot read it as agreement.
    """
    if name in node.kwargs:
        value = node.kwargs[name]
    elif len(node.args) > index:
        value = node.args[index]
    else:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scalar_source(arg):
    """The tensor a scalar argument was extracted from.

    start_pos is a scalar, and a scalar has no arena slot, so the value the
    kernel needs can only be read back from wherever it was extracted.
    """
    # Matched by name rather than identity: the graph holds the edge overloads,
    # which are different objects from the torch.ops ones with the same schema.
    while isinstance(arg, torch.fx.Node) and getattr(arg.target, "__name__", "") in (
        "item.default",
        "_local_scalar_dense.default",
    ):
        arg = arg.args[0]
    return arg


def _attention_workspace_bytes(qo_len, seq_len, n_slots):
    """The per-worker FLASH_ATTN scratch, matching MNN's block scheduler.

    The DSP caps the query block at 64 rows. Allocating the full query length
    here turns a 2048-token prefill into an unnecessarily enormous buffer and
    can make the command fail before the kernel starts.
    """
    qo_len = min(qo_len, 64)
    padded = (seq_len + 31) // 32 * 32
    scores = (qo_len * padded * 4 + 127) // 128 * 128
    probabilities = (qo_len * padded * 2 + 127) // 128 * 128
    return (scores + probabilities) * max(1, n_slots)


def _emit_sdpa(node: torch.fx.Node, ctx) -> TensorRef:
    """llama.sdpa_with_kv_cache as one non-paged FLASH_ATTN.

    The cache is an operand rather than something the kernel keeps: slots four
    and five are the past keys and values, and this graph's own update_cache has
    already written the new rows by the time attention runs, so the kernel's
    push copies cache rows onto themselves and changes nothing. Passing those
    two slots empty, as this did, hands the kernel null pointers it writes
    through before it computes anything. start_pos only exists once the graph
    runs, and the cache is written in place.
    """
    args = node.args
    query, key, value = args[0], args[1], args[2]
    # Unlike the other ops, attention accepts fp32: the runtime narrows those
    # operands to fp16 as they enter the arena, so the DSP still sees fp16.
    _dtype = node.meta["val"].dtype
    if _dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"hexagon: sdpa input must be fp16 or fp32, got {_dtype}")

    mask = args[4] if len(args) > 4 else None
    scale = args[7] if len(args) > 7 else None
    q_shape = query.meta["val"].shape
    kv_shape = key.meta["val"].shape

    # The kernel sizes and indexes this source as [batch, seq, heads, dim]:
    # attention_entry.cc strides it by tokens * heads * headDim, which is the
    # layout the upstream caller builds by transposing a head-major cache
    # (examples/models/llama/source_transformation/sdpa.py), and the layout the
    # vision path hands the same kernel. Slots four and five are the packed cache
    # the kernel writes, not the cache itself -- attn_hmx_k_tile_index lays it
    # out by tile: 256 tokens per block, eight 32-row sequence tiles per block,
    # one 1024-element tile per (32 dim x 32 seq) sub-block. Any cache length
    # pays for whole blocks, the same span attention_entry.cc sizes its own
    # packed K/V to.
    if len(kv_shape) != 4 or kv_shape[3] != q_shape[3]:
        raise RuntimeError(f"hexagon: sdpa cache {tuple(kv_shape)} does not match head_dim {q_shape[3]}")
    # Slots one and two reach the kernel as [batch, seq, heads, dim]: the
    # partitioner hands them the stored cache, and a caller that keeps its cache
    # head-major transposes it into that layout before the op.
    n_kv_heads, max_kv_len = kv_shape[2], kv_shape[1]
    q_len = ctx.upper_bound(q_shape[1])
    if not 0 < n_kv_heads <= q_shape[2] or q_shape[2] % n_kv_heads:
        raise RuntimeError(
            "hexagon: sdpa cache operand is not [batch, seq, heads, dim]"
            f" (heads {n_kv_heads} of {q_shape[2]} from {tuple(kv_shape)})"
        )
    seq_blocks = (max_kv_len + 255) // 256
    # The raw cache tensor can end at the previous complete block when the
    # exporter uses an inclusive symbolic upper bound (for example 2049).
    # Keep packed storage conservatively rounded up, but tell the DSP the
    # actual writable cache capacity.
    cache_capacity = max(256, (max_kv_len // 256) * 256)
    dim_tiles = (q_shape[3] + 31) // 32
    packed_bytes = seq_blocks * n_kv_heads * 8 * dim_tiles * 1024 * FP16_BYTES

    inputs = [
        ctx.operand(args[0]),
        ctx.operand(args[1]),
        ctx.operand(args[2]),
        ABSENT if mask is None else ctx.operand(mask),
        ctx.builder.add_activation(packed_bytes),
        ctx.builder.add_activation(packed_bytes),
    ]
    # start_pos is either a constant the exporter folded in, in which case it is
    # baked into params, or a run-time tensor read, in which case it rides along
    # as an extra input the kernel never reads and the runtime patches it in.
    source = _scalar_source(args[3])
    position = 0
    patch = None
    if isinstance(source, torch.fx.Node):
        inputs.append(ctx.operand(source))
        patch = (1, len(inputs) - 1)
    elif isinstance(args[3], int):
        position = args[3]
    else:
        raise RuntimeError(f"hexagon: sdpa cannot resolve start_pos {args[3]!r}")

    out = ctx.result_for(node, _numel(node))
    # The kernel sizes its scratch from seq_current + seq_add, and seq_current is
    # the run-time position. The cache length bounds it: the position can never
    # pass the rows that exist, so this asks for the longest sequence the cache
    # could hold rather than the one this call happens to use.
    workspace_max_bytes = _attention_workspace_bytes(
        q_len, q_len + max_kv_len, q_shape[2]
    )
    workspace = ctx.builder.add_activation(
        workspace_max_bytes,
        dynamic_layout=ctx.dynamic_bytes_for_function(
            workspace_max_bytes,
            lambda length: _attention_workspace_bytes(
                length, length + max_kv_len, q_shape[2]
            ),
        ),
    )

    # HEXAGON_ATTN_PAGED=1 takes the paged entry (htp_ops_flash_attn_pages) that MNN's
    # HexagonAttention.cpp always uses: one page spanning the whole packed cache, so the
    # buffer layout is unchanged. page_size must be a multiple of 32 and must exceed
    # seq_current, or push_kv_pages skips the insert.
    paged = bool(os.environ.get("HEXAGON_ATTN_PAGED"))
    page_size = 256 if paged else 0

    op_index = ctx.builder.add_op(
        Op(
            type=DSP_OP_FLASH_ATTN,
            inputs=inputs,
            outputs=[out, workspace],
            params=[
                q_len,  # qo_len
                position,  # seq_current
                q_len,  # seq_add
                q_shape[2],  # n_heads
                n_kv_heads,  # n_kv_heads
                q_shape[3],  # head_dim
                _float_bits(
                    scale if isinstance(scale, (int, float)) else q_shape[3] ** -0.5
                ),
                -1,  # mask_stride
                # push_kv only reads this as the capacity its writes must stay
                # inside, so it is the operand's length, not the cache's.
                cache_capacity,  # max_kv_len
                seq_blocks if paged else 0,  # page_count
                page_size,  # page_size
                0,  # value_c4
            ],
            patch=patch,
        )
    )
    if ctx.is_dynamic_dim(q_shape[1]):
        ctx.add_dynamic_patch(op_index, 0, 1, 0)
        ctx.add_dynamic_patch(op_index, 2, 1, 0)
    return ctx.record(node, out)


# torch's dim-order copies, which `to_edge` leaves behind where a memory format
# had to be named. Both reach `_emit_alias` through `dim_order_keeps_the_bytes`,
# which is also what keeps a non-identity order on a portable kernel.
TO_DIM_ORDER_COPY = exir_ops.edge.dim_order_ops._to_dim_order_copy.default
CLONE_DIM_ORDER = exir_ops.edge.dim_order_ops._clone_dim_order.default
DIM_ORDER_TARGETS = frozenset({TO_DIM_ORDER_COPY, CLONE_DIM_ORDER})

EMITTERS = {
    exir_ops.edge.aten.abs.default: _unary("abs"),
    exir_ops.edge.aten.neg.default: _unary("neg"),
    exir_ops.edge.aten.gelu.default: _unary("gelu"),
    exir_ops.edge.aten.sigmoid.default: _unary("sigmoid"),
    exir_ops.edge.aten.exp.default: _unary("exp"),
    exir_ops.edge.aten.log.default: _unary("log"),
    exir_ops.edge.aten.silu.default: _unary("silu"),
    exir_ops.edge.aten.clamp.default: _emit_clamp,
    exir_ops.edge.aten.clamp.out: _emit_clamp,
    ROW_GUARD: _emit_row_guard,
    exir_ops.edge.aten.tanh.default: _unary("tanh"),
    exir_ops.edge.aten.sqrt.default: _unary("sqrt"),
    exir_ops.edge.aten.rsqrt.default: _unary("rsqrt"),
    exir_ops.edge.aten.add.Tensor: _binary("add"),
    exir_ops.edge.aten.sub.Tensor: _binary("sub"),
    exir_ops.edge.aten.mul.Tensor: _binary("mul"),
    exir_ops.edge.aten.div.Tensor: _binary("div"),
    exir_ops.edge.aten.maximum.default: _binary("max"),
    exir_ops.edge.aten.minimum.default: _binary("min"),
    # to_edge rewrites softmax.int into _softmax, which is the name the
    # partitioner then sees.
    exir_ops.edge.aten._softmax.default: _emit_softmax,
    # to_edge grows native_layer_norm out of layer_norm; the functional form is
    # kept for a graph that reaches the backend without that rewrite.
    exir_ops.edge.aten.layer_norm.default: _emit_layer_norm,
    NATIVE_LAYER_NORM: _emit_layer_norm,
    # The getitem that reads a layer norm's first output.
    GETITEM: _emit_getitem,
    exir_ops.edge.aten.mm.default: _emit_mm,
    exir_ops.edge.aten.bmm.default: _emit_bmm,
    exir_ops.edge.aten.addmm.default: _emit_addmm,
    exir_ops.edge.aten.mean.dim: _emit_mean_dim,
    exir_ops.edge.aten.alias_copy.default: _emit_alias,
    exir_ops.edge.aten.unsqueeze_copy.default: _emit_alias,
    exir_ops.edge.aten.squeeze_copy.dims: _emit_alias,
    exir_ops.edge.aten.view_copy.default: _emit_alias,
    exir_ops.edge.aten.expand_copy.default: _emit_alias,
    TO_DIM_ORDER_COPY: _emit_alias,
    CLONE_DIM_ORDER: _emit_alias,
    exir_ops.edge.aten.select_copy.int: _emit_select_copy,
    exir_ops.edge.aten._to_copy.default: _emit_alias,
    exir_ops.edge.aten.to.dtype: _emit_alias,
    exir_ops.edge.aten.slice_copy.Tensor: _emit_slice_copy,
    exir_ops.edge.aten.cat.default: _emit_cat,
    exir_ops.edge.aten.permute_copy.default: _emit_permute_copy,
    exir_ops.edge.aten.mul.Scalar: _emit_mul_scalar,
    UPDATE_CACHE: _emit_update_cache,
    RMS_NORM: _emit_rms_norm,
    ADD_RMS_NORM: _emit_add_rms_norm,
    MUL_SILU: _binary("mul_silu"),
    ROPE: _emit_rope,
}

# Ops whose operands must match the output's shape or be scalar. The support
# check needs this so a broadcast operand keeps the node on a portable kernel
# instead of reaching an emitter that refuses it and failing the whole export.
BINARY_TARGETS = frozenset(
    {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.maximum.default,
        exir_ops.edge.aten.minimum.default,
        MUL_SILU,
    }
)

# Same reasoning as BINARY_TARGETS: mm derives its strides from the operand
# shapes, which is only right for contiguous 2-D tiles.
MM_TARGETS = frozenset({exir_ops.edge.aten.mm.default})

# bmm is the same tile geometry with a batch axis in front of both operands.
BMM_TARGETS = frozenset({exir_ops.edge.aten.bmm.default})

# addmm needs the bias it can broadcast and an alpha it can fold away, so what
# the emitter accepts is narrower than the op's own contract.
ADDMM_TARGETS = frozenset({exir_ops.edge.aten.addmm.default})

# REDUCTION collapses one contiguous span, so the reduced dims must be adjacent.
MEAN_TARGETS = frozenset({exir_ops.edge.aten.mean.dim})

# Views: the operand's bytes read under another shape. Only the forms that
# keep a contiguous layout are listed -- select_copy's other overload reads the
# int64 position tensor down to a scalar, which is not a view of this kind and
# has to stay where the patch mechanism can reach it.
SLICE_TARGETS = frozenset({exir_ops.edge.aten.slice_copy.Tensor})

# Reaches _emit_select_copy, which takes the narrowing form as a blit and leaves
# the same-bytes form to the alias path.
SELECT_TARGETS = frozenset({exir_ops.edge.aten.select_copy.int})

CAT_TARGETS = frozenset({exir_ops.edge.aten.cat.default})

PERMUTE_TARGETS = frozenset({exir_ops.edge.aten.permute_copy.default})

ALIAS_TARGETS = frozenset(
    {
        exir_ops.edge.aten.alias_copy.default,
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.expand_copy.default,
        exir_ops.edge.aten.select_copy.int,
    }
)

# A cast between the two widths the arena already offers emits nothing: every
# kernel reads and writes fp16, the runtime narrows a fp32 operand on the way in
# and widens a fp32 result on the way out, and both directions are exact. So
# delegating one joins the partition on either side of it instead of cutting the
# graph there. Only fp16 and fp32: a cast from int64 is a real conversion.
CAST_TARGETS = frozenset(
    {
        exir_ops.edge.aten._to_copy.default,
        exir_ops.edge.aten.to.dtype,
    }
)

# The DSP's attention entry point is written for `sdpa_with_kv_cache`: key and
# value are the new tokens, which the kernel pushes into a cache it is handed as
# two further operands. The graph carries `custom_sdpa`, whose key and value
# *are* the caches and which has no cache operand at all, so the two contracts
# do not meet. `_emit_sdpa` bridges them by leaving both cache operands ABSENT,
# which the dispatcher maps to a null pointer, and the kernel writes the new rows
# through it: the DSP dies with `execute_command_group failed: 0x8000040d` on
# the first layer. Two of its params are wrong for the same reason -- the key
# operand it ships is the permuted cache, so n_kv_heads and max_kv_len are read
# off the wrong axes.
#
# Delegating attention is therefore off until the emitter is rewritten against
# the kernel (and the kernel validated). Attention stays on the portable
# kernels, which costs speed and not correctness.
SDPA_DELEGATION = True

# The fused attention belongs to the LLM extension, whose schema only appears
# once that extension registers its ops -- which happens after this module is
# imported. Resolving it here would bake in an empty set, so it is resolved on
# first use instead.
# The graph carries edge overloads, so matching torch.ops.llama.* would never
# fire -- the two are different objects even for the same schema.
SDPA_TARGETS: frozenset = frozenset()


def _register_llama_sdpa() -> None:
    global SDPA_TARGETS
    if SDPA_TARGETS:
        return
    if not SDPA_DELEGATION:
        return
    op = getattr(getattr(exir_ops.edge, "llama", None), "custom_sdpa", None)
    if op is None:
        return
    # The edge graph carries the .out overload, but the partitioner reports the
    # op as .default; register both and let whichever the graph holds match.
    SDPA_TARGETS = frozenset(
        target
        for target in (getattr(op, "out", None), getattr(op, "default", None))
        if target is not None
    )
    for target in SDPA_TARGETS:
        EMITTERS[target] = _emit_sdpa


def sdpa_targets() -> frozenset:
    """The attention overloads the DSP can run, resolved once the ops exist."""
    _register_llama_sdpa()
    return SDPA_TARGETS
