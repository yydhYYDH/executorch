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
import struct
from typing import Dict, List, NamedTuple, Optional

import torch

# After rms_norm, which opens the et_hexagon namespace these fragments join.
from executorch.backends.hexagon.add_relu import ADD_RELU
from executorch.backends.hexagon.add_rms_norm import ADD_RMS_NORM
from executorch.backends.hexagon.kv_cache import UPDATE_CACHE
from executorch.backends.hexagon.mul_silu import MUL_SILU
from executorch.backends.hexagon.rms_norm import RMS_NORM
from executorch.backends.hexagon.rope import ROPE
from executorch.backends.hexagon.row_guard import ROW_GUARD
from executorch.backends.hexagon.serialization.blob import ABSENT, Op, TensorRef
from executorch.backends.hexagon.vision_attention import VISION_ATTENTION
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.sym_util import eval_upper_bound

# DSPOpType, from third-party/mnn-htp-ops/include/htp_command.h.
DSP_OP_POOL2D_FP16 = 1
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
# The quantized matmuls, from the DSP's enum (htp_command.h pins both numbers
# with a static_assert). They take an fp16 activation and a packed low-bit
# weight, and each kernel quantizes that activation to symmetric per-token int8
# inside itself: what reaches the multiply is int4/int8, whatever the annotation
# said. The GEMV entries are the M == 1 (decode) kernels -- they read K
# contiguous fp16, which for one row is the row-major layout the graph already
# holds, and write the output linear by output channel, so neither the 64-channel
# activation blocking nor the output repack the prefill entries need is involved.
# The prefill entries (Q4A16_FP16, Q4A16_BLOCK_FP16, W8A16_BLOCK_FP16) are not
# emitted here.
DSP_OP_MATMUL_Q4A16_GEMV_I8 = 41
DSP_OP_MATMUL_W8A16_GEMV_I8 = 45
# The row gather, from the same enum: one command reads selectSize rows out of
# an fp16 table the export step laid out as 32x32 tiles.
DSP_OP_SHARED_GATHER = 23
# The vision tower's attention, from the same enum. Unmasked and non-causal: it
# reads query, key and value as [batch, tokens, heads, headDim] -- heads inside
# a token's row -- and writes the result in that layout, which is why the fused
# op it serves carries the operands under the head transposes rather than the
# head-major tensors a matmul wants.
DSP_OP_VISION_ATTENTION_FP16 = 43

# HtpOpsReductionType, from the DSP's eltwise_ops.cc (the whole enum: there is
# no minimum, so `amin` has no kernel behind it and stays on the host).
REDUCTION_SUM = 1
REDUCTION_MAXIMUM = 2
REDUCTION_MEAN = 3

# The DSP's pool kernel selects on two ints rather than on op types
# (pool_fp16.c:18-19 and :46): poolType picks max or the sum, and countType
# picks the average's divisor -- the kernel window's area, which is torch's
# count_include_pad=True, or the positions that landed inside the input, which
# is count_include_pad=False. padType is read by neither.
POOL_MAX = 0
POOL_AVERAGE = 1
POOL_COUNT_VALID = 0
POOL_COUNT_KERNEL = 1
POOL_PAD_TYPE = 0

# One HVX vector of fp16, which is also the block the DSP's activation layout
# groups channels in: element (n, c, y, x) of an NCHW tensor sits at
# ((c // 64) * batch + n) * height * width * 64 + (y * width + x) * 64 + c % 64,
# where `c` is a channel block index (hvx_pool2d_fp16,
# src/dsp/ops/pool_fp16.c:13 and :22). 64 is __HVX_LENGTH__ / sizeof(__fp16) for
# the -mhvx-length=128b the skel is built with (skel/CMakeLists.txt:38).
POOL_CHANNEL_BLOCK = 64

# The isInt4 slot of htp_ops_shared_gather. 0 is a plain fp16 table, the only
# kind stored here; 2 and 3 are int4 and int8 tables with the extra scale
# params those paths read, and nothing in this backend produces them.
SHARED_GATHER_FP16 = 0

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
        result *= (
            ctx.upper_dim(value)
            if ctx is not None
            else (
                eval_upper_bound(value)
                if isinstance(value, torch.SymInt)
                else int(value)
            )
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


def _patch_dynamic_rows(
    ctx, op_index: int, param_index: int, rows: int, factors
) -> None:
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


def _patch_dynamic_numel(
    ctx, op_index: int, node: torch.fx.Node, param_index: int = 0
) -> None:
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
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(node.args[0])],
            outputs=[out],
            params=[1, FP16_BYTES, 1] + region,
        ),
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
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=inputs,
            outputs=[out],
            # region count, element bytes, source count, then the region.
            params=[1, FP16_BYTES, 1] + region.region,
            patch=patch,
            patch_scale=region.patch_scale,
        ),
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
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(tensor) for tensor in tensors],
            outputs=[out],
            params=params,
        ),
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
    # A folded transpose is stored in the weights section and reached through
    # the folded table, which only a consumer reads. A node the caller reads
    # from a delegate output has no consumer, and an output slot is a region of
    # the arena the runtime copies out, so there the transpose is emitted as the
    # blit it is -- reading the weight, which is where its operand lives either
    # way.
    if not ctx.is_method_output(node):
        folded = _fold_constant_transpose(node, ctx)
        if folded is not None:
            return folded
    out = ctx.result_for(node, _numel(node))
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(node.args[0])],
            outputs=[out],
            params=[1, FP16_BYTES, 1] + permute_region(node),
        ),
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
        op_index = ctx.emit(
            node,
            Op(
                type=DSP_OP_UNARY,
                inputs=[ctx.operand(src)],
                outputs=[out],
                # size is in elements, not bytes.
                params=[
                    _upper_product(tuple(_value_of(node).shape), ctx),
                    UNARY_OP_TYPES[op_name],
                    FP16_BYTES,
                ],
            ),
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
    op_index = ctx.emit(
        node,
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
        ),
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
    op_index = ctx.emit(
        node,
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
        ),
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
    op_index = ctx.emit(
        node,
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
        ),
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
        op_index = ctx.emit(
            node,
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
            ),
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
        # hmxTileBudget: zero asks the unit to fit itself to the VTCM it finds.
        # A runner can raise or lower this slot at load time (the tile_budget
        # option), which is a cap on what the unit keeps resident rather than a
        # change to the weights, so the blob does not have to be rebuilt for it.
        0,
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
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[source],
            outputs=[out],
            params=[1, FP16_BYTES, 1, 0, 0, 0, 1, 1, numel, 0, 0, 1, 0, 0, 1],
        ),
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
    ctx.emit(
        node,
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
        ),
    )
    # dstOffset is the token position times one whole cached position, which is
    # what patch_scale is for. dst is the output, at index 3.
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[value, cache, pos],
            outputs=[out],
            params=[1, FP16_BYTES, 3, 0, 0, 0, 1, rows, run, 0, run, 1, 0, run, 1],
            patch=(5, 2),
            patch_scale=inner,
        ),
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
        ctx.options.hmx_prepack
        and hmx_prefers_general(k, n)
        and hmx_general_eligible(m, k, n)
        and value is not None
        and value.dim() == 2
        and tuple(value.shape) == (k, n)
    ):
        return ctx.packed_weight(rhs, value, k, n), True
    return ctx.operand(rhs), False


# The PT2E weight-only pattern: a matmul's weight reaches it through a
# quantized_decomposed.dequantize_per_channel whose input is the stored low-bit
# weight and whose scales are per output channel. The dequantize is fused into
# the matmul, so it emits no command of its own.
DQ_PER_CHANNEL = exir_ops.edge.quantized_decomposed.dequantize_per_channel.default


class QuantizedWeight(NamedTuple):
    """The parts of a dequantize_per_channel a quantized matmul reads."""

    weight: torch.fx.Node
    scale: torch.fx.Node
    zero_point: torch.fx.Node
    axis: int
    bits: int


def quantized_weight(node) -> Optional[QuantizedWeight]:
    """The quantized weight behind a matmul operand, or None.

    Only symmetric per-channel int4/int8 dequantizes are understood: the GEMV
    kernels carry one fp32 scale per output channel and no zero point, so
    anything else has to stay on a portable kernel rather than be read wrong.
    """
    if not isinstance(node, torch.fx.Node) or node.target is not DQ_PER_CHANNEL:
        return None
    args = node.args
    if len(args) < 7:
        return None
    weight, scale, zero_point, axis = args[0], args[1], args[2], args[3]
    quant_min, quant_max, dtype = args[4], args[5], args[6]
    if not all(isinstance(part, torch.fx.Node) for part in (weight, scale, zero_point)):
        return None
    if dtype is not torch.int8 or not isinstance(axis, int) or isinstance(axis, bool):
        return None
    if (
        isinstance(quant_min, bool)
        or isinstance(quant_max, bool)
        or not isinstance(quant_min, int)
        or not isinstance(quant_max, int)
    ):
        return None
    span = quant_max - quant_min + 1
    if span == 16:
        bits = 4
    elif span in (255, 256):
        bits = 8
    else:
        return None
    return QuantizedWeight(weight, scale, zero_point, axis, bits)


def quantized_matmul_geometry(activation, quantized: QuantizedWeight):
    """(m, k, n) of a matmul over a quantized weight, or None.

    The weight is stored as [k, n] or as its transpose, whichever way the
    quantizer spelled it: the axis the per-channel scale is indexed by is the
    output-feature axis, and the packers put the weight back into the [k, n]
    order the kernels read. A dynamic or non-1 M is refused here because only
    the M == 1 GEMV kernels are wired up.
    """
    if not isinstance(activation, torch.fx.Node):
        return None
    lhs = activation.meta.get("val")
    weight = quantized.weight.meta.get("val")
    if not isinstance(lhs, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return None
    if lhs.dim() != 2 or weight.dim() != 2:
        return None
    try:
        if quantized.axis == 1:
            k, n = int(weight.shape[0]), int(weight.shape[1])
        elif quantized.axis == 0:
            n, k = int(weight.shape[0]), int(weight.shape[1])
        else:
            return None
        inner = lhs.shape[1]
        if isinstance(inner, torch.SymInt):
            return None
        if int(inner) != k:
            return None
        m = lhs.shape[0]
        if isinstance(m, torch.SymInt):
            return None
        return int(m), k, n
    except (TypeError, ValueError):
        return None


def _quantized_gemv_fits(activation, quantized: QuantizedWeight) -> bool:
    """Whether the M == 1 GEMV kernels can run this matmul.

    Both GEMV entries require K to be a multiple of the 64-element block and N a
    multiple of the 32-channel tile; a shape outside that has no kernel that
    reads the packed weight, so it stays on a portable kernel.
    """
    geometry = quantized_matmul_geometry(activation, quantized)
    if geometry is None:
        return False
    m, k, n = geometry
    return m == 1 and k % 64 == 0 and n % 32 == 0


def quantized_matmul_weight(node) -> Optional[torch.fx.Node]:
    """The operand a matmul's weight-only weight arrives through, if any.

    mm and addmm both put the weight last, and it is a `dequantize_per_channel`
    rather than a tensor once PT2E has converted the graph.
    """
    if node.target in MM_TARGETS and len(node.args) > 1:
        return node.args[1]
    if node.target in ADDMM_TARGETS and len(node.args) >= 3:
        return node.args[2]
    return None


def quantized_matmul_is_emittable(node: torch.fx.Node) -> bool:
    """Whether this matmul is one the GEMV emitter can place, in full.

    The support check, the dequantize's fusion test and the emitter all call
    this, so the three cannot disagree: a node the partitioner delegates is one
    the emitter accepts, and a dequantize is only fused away when every matmul
    reading it is one this accepts. The emitter cannot fall back -- it is past
    the partition boundary -- so a condition missing here is an export failure
    or, worse, a command the kernel reads past.
    """
    if node.target in MM_TARGETS:
        if len(node.args) < 2:
            return False
        bias, activation = None, node.args[0]
    elif node.target in ADDMM_TARGETS:
        if len(node.args) < 3:
            return False
        bias, activation = node.args[0], node.args[1]
        # alpha and beta are the same two the plain path checks: alpha has no
        # kernel here, and beta 0 or 1 is what decides whether the kernel's own
        # bias operand is the graph's or absent.
        if _scalar_arg(node, "alpha", 4, 1.0) != 1.0:
            return False
        beta = _scalar_arg(node, "beta", 3, 1.0)
        if beta not in (0.0, 1.0):
            return False
        if beta == 0.0:
            bias = None
    else:
        return False

    quantized = quantized_weight(quantized_matmul_weight(node))
    if quantized is None:
        return False
    value = activation.meta.get("val")
    if not isinstance(value, torch.Tensor) or not value.is_contiguous():
        # The kernel reads K contiguous fp16 with aligned 128-byte loads; a
        # strided activation is one no kernel here walks.
        return False
    geometry = quantized_matmul_geometry(activation, quantized)
    if geometry is None or not _quantized_gemv_fits(activation, quantized):
        return False
    if bias is None:
        return True
    return _bias_is_one_value_per_channel(bias, geometry[2])


def quantized_matmul_is_refused(node: torch.fx.Node) -> bool:
    """Whether this matmul carries a weight-only weight nothing here can run.

    A matmul whose weight is the weight-only pattern goes to a GEMV kernel
    rather than to the flat path, and that kernel has conditions of its own.
    One that fails them has to fall back whole -- the dequantize with it -- so
    this is asked about the matmul and answered for both.
    """
    if quantized_weight(quantized_matmul_weight(node)) is None:
        return False
    return not quantized_matmul_is_emittable(node)


def _bias_is_one_value_per_channel(bias, n: int) -> bool:
    """Whether addmm's bias is the n-element row the GEMV kernel adds.

    The kernel adds one fp16 value per output channel, read as n contiguous
    halfs, so a bias torch would broadcast to anything but a single row of n
    has no command that can carry it: the kernel would read past it rather than
    refuse. A scalar bias is the sharp case -- it is legal in the graph and
    would have the kernel add whatever follows it in the arena.
    """
    if not isinstance(bias, torch.fx.Node):
        return False
    value = bias.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.numel() != n:
        return False
    try:
        return tuple(torch.broadcast_shapes(tuple(value.shape), (1, n))) == (1, n)
    except RuntimeError:
        return False


def _dequantize_is_fused(node: torch.fx.Node) -> bool:
    """Whether every reader of this dequantize is a quantized matmul we can run.

    The dequantize emits nothing; it exists so the matmul's weight arrives as a
    pattern rather than as an int8 constant the support check would refuse. If
    any reader is not a runnable quantized matmul, the whole chain has to stay
    on the portable kernels, so the check is all-or-nothing: a dequantize
    delegated on its own would be a partition whose output nothing ever wrote.
    """
    if not node.users or quantized_weight(node) is None:
        return False
    return all(
        quantized_matmul_weight(user) is node and quantized_matmul_is_emittable(user)
        for user in node.users
    )


def pack_q4a16_gemv_weight(weight, scale, k: int, n: int) -> bytes:
    """A (k, n) int4 weight in the vrmpy tile order the GEMV kernel reads.

    Transcribed from the weight layout contract in
    third-party/mnn-htp-ops/src/dsp/ops/matmul_q4block_gemv_i8.c. The weight
    blob is `icP*ocP` tiles of 512 bytes, tile (ocTile y, kTile x) at
    `(y*icP + x)*512`; inside a tile, group g holds the four k values
    `x*32 + 4g + {0,1,2,3}` for the tile's 32 output channels, and byte
    `g*64 + ocIn*2 + p` packs k = `x*32 + 4g + 2p` in its low nibble and
    `+1` in its high nibble, each stored as `value + 8`. The per-output-channel
    fp32 scales follow the tiles, which is where the kernel's `b_scale` pointer
    lands (`weight + icP*ocP*512`, matmul_q4block_ops.cc).

    What is checked, and what is not: the tile order above is written down
    twice in the vendored tree -- in the kernel header and again in
    include/dsp/vrmpy_to_hmx.h, which the vendored tree says was verified
    byte-exact against MNN's own host reorder -- and the kernel's read path
    agrees with it (unpack_vrmpy_weight_128 nibble-expands byte p lane-wise and
    pairs the group at byte offset 128*l with `a_splat[kt*8 + 2l]`, while
    `a_splat` is qa splatted 4 k values at a time, so the k a lane multiplies
    are the k this packs). No DSP has run it: the upstream host reorder function
    is not in the vendored tree (`src/host/` was dropped at vendoring), so the
    hardware is the only thing left that could disagree, and nothing here has
    compared against it.
    """
    import numpy as np

    w = np.asarray(weight, dtype=np.int32)
    if w.shape != (k, n):
        raise RuntimeError(f"hexagon: q4a16 weight is {w.shape}, expected ({k}, {n})")
    if k % 64 or n % 32:
        raise RuntimeError(
            f"hexagon: q4a16 needs K a multiple of 64 and N of 32, got {k}x{n}"
        )
    kp, np_ = k // 32, n // 32
    padded = np.zeros((np_ * 32, kp * 32), dtype=np.int32)
    padded[:n, :k] = np.clip(w.T, -8, 7) + 8
    # (y, ocIn, x, kk) -> (y, x, ocIn, g, four)
    t = padded.reshape(np_, 32, kp, 32).transpose(0, 2, 1, 3)
    t = t.reshape(np_, kp, 32, 8, 4)
    low = t[..., 0] | (t[..., 1] << 4)
    high = t[..., 2] | (t[..., 3] << 4)
    # (y, x, ocIn, g, p) -> (y, x, g, ocIn, p), which is the tile's byte order.
    packed = np.stack([low, high], axis=-1).transpose(0, 1, 3, 2, 4)
    tiles = packed.astype(np.uint8).reshape(np_ * kp, 512).tobytes()
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scales.size != n:
        raise RuntimeError(f"hexagon: q4a16 has {scales.size} scales, expected {n}")
    return tiles + scales.tobytes()


def pack_w8a16_gemv_weight(weight, k: int, n: int) -> bytes:
    """A (k, n) int8 weight in the HMX tile order the GEMV kernel reads.

    Transcribed from the weight layout contract in
    third-party/mnn-htp-ops/src/dsp/ops/matmul_w8a16_gemv_i8.c. The weight blob
    is `kp*np` tiles of 1024 bytes, tile (oy, kx) at `(oy*kp + kx)*1024`; inside
    a tile, group g covers the four k values `kx*32 + 4g + {0,1,2,3}` for the
    tile's 32 output channels, and byte `g*128 + ocIn*4 + p` holds the weight
    for k = `kx*32 + 4g + perm[p]` with perm = {0, 2, 1, 3}. The per-channel
    fp32 scales are a separate operand the emitter stores on its own.

    What is checked, and what is not: the pair-interleave is checked against the
    kernel's own read path, which is the one place that has to agree with it --
    `splat_group_permuted` swaps bytes 1 and 2 of each 4-k activation word, so
    the activation arrives in the same {0, 2, 1, 3} order and every byte of a
    32-bit vrmpy lane multiplies the weight of the same k. Nothing else here can
    confirm the tile order offline: unlike the int4 layout, this one is written
    down only in the kernel header, and the host reorder that produces it
    (`reorderInt8SymWeightForHmx`) is not in the vendored tree. No DSP has run
    it.
    """
    import numpy as np

    w = np.asarray(weight, dtype=np.int32)
    if w.shape != (k, n):
        raise RuntimeError(f"hexagon: w8a16 weight is {w.shape}, expected ({k}, {n})")
    if k % 64 or n % 32:
        raise RuntimeError(
            f"hexagon: w8a16 needs K a multiple of 64 and N of 32, got {k}x{n}"
        )
    kp, np_ = k // 32, n // 32
    padded = np.zeros((np_ * 32, kp * 32), dtype=np.int32)
    padded[:n, :k] = np.clip(w.T, -128, 127)
    # (oy, ocIn, kx, kk) -> (oy, kx, ocIn, g, four), then permute the four k
    # values within each group to the kernel's byte order.
    t = padded.reshape(np_, 32, kp, 32).transpose(0, 2, 1, 3)
    t = t.reshape(np_, kp, 32, 8, 4)[..., (0, 2, 1, 3)]
    packed = t.transpose(0, 1, 3, 2, 4)
    return packed.astype(np.uint8).reshape(np_ * kp, 1024).tobytes()


def _emit_quantized_matmul(
    node: torch.fx.Node,
    ctx,
    quantized: QuantizedWeight,
    activation: torch.fx.Node,
    bias: Optional[torch.fx.Node],
) -> TensorRef:
    """A quantized matmul as one M == 1 GEMV command.

    The weight is packed at export into the layout the kernel reads and stored
    in the weights section; the activation is the graph's own fp16 row, K
    contiguous halfs, which the kernel then quantizes to int8 per token. The
    low-bit weight never goes through `weight_bytes`: that narrows a fp32 tensor
    to fp16 silently, which is right for a kernel that reads halfs and wrong for
    one that reads packed int4, so this reads the stored tensor and packs it.
    """
    import numpy as np

    geometry = quantized_matmul_geometry(activation, quantized)
    if geometry is None:
        raise RuntimeError("hexagon: quantized matmul has no static geometry")
    m, k, n = geometry
    if not quantized_matmul_is_emittable(node):
        raise RuntimeError(
            f"hexagon: quantized matmul {m}x{k}x{n} is not one the support check "
            "admits; the partitioner should not have delegated it"
        )

    weight = ctx.constant_value(quantized.weight)
    scale = ctx.constant_value(quantized.scale)
    if weight is None or scale is None:
        raise RuntimeError("hexagon: quantized matmul weight is not a constant")
    zero_point = ctx.constant_value(quantized.zero_point)
    if zero_point is not None and bool((zero_point != 0).any()):
        raise RuntimeError("hexagon: quantized matmul weight is not symmetric")

    weight = weight.detach().to(torch.int8).cpu().numpy()
    scale = scale.detach().to(torch.float32).cpu().numpy().reshape(-1)
    if quantized.axis == 0:
        weight = weight.T
    if weight.shape != (k, n):
        raise RuntimeError(
            f"hexagon: quantized matmul weight is {weight.shape}, expected ({k}, {n})"
        )

    out = ctx.result_for(node, n)
    bias_ref = ABSENT if bias is None else ctx.operand(bias)
    # The GEMV dispatch reads K and N out of params[1] and params[2] and the
    # scale block count out of params[8] (execute_command.cc); params[0] is the
    # M the commands that carry one use, and is not read here.
    #
    # params[8] == 1 is the per-channel granularity, not a placeholder: the
    # kernel derives blocksize = K/nblk, so one block spans all of K and
    # compute_oc_tile_i8 walks kp k-tiles reading the entry's 32 fp32 scales,
    # one per output channel of the tile. That is exactly the per_channel
    # scale the quantizer produced. The kernel's own producer contract
    # (nblk = K/blocksize) describes blocksize 64 group quantization, which is
    # a different weight operand this path does not build.
    params = [1, k, n, 0, 0, 0, 0, 0, 1, 0]
    if quantized.bits == 4:
        packed = pack_q4a16_gemv_weight(weight, scale, k, n)
        ctx.emit(
            node,
            Op(
                type=DSP_OP_MATMUL_Q4A16_GEMV_I8,
                inputs=[
                    ctx.operand(activation),
                    ctx.builder.add_weights(packed),
                    bias_ref,
                ],
                outputs=[out],
                params=params,
            ),
        )
    else:
        packed = pack_w8a16_gemv_weight(weight, k, n)
        ctx.emit(
            node,
            Op(
                type=DSP_OP_MATMUL_W8A16_GEMV_I8,
                inputs=[
                    ctx.operand(activation),
                    ctx.builder.add_weights(packed),
                    ctx.builder.add_weights(scale.astype(np.float32).tobytes()),
                    bias_ref,
                ],
                outputs=[out],
                params=params,
            ),
        )
    return ctx.record(node, out)


def _emit_dequantize(node: torch.fx.Node, ctx) -> TensorRef:
    """A dequantize is fused into the matmul that reads it.

    The support check only admits one whose every reader is a quantized matmul
    this can run, so nothing reads the result this returns: the matmul packs the
    stored low-bit weight itself. ABSENT records that, and keeps the low-bit
    weight from being materialized a second time.
    """
    return ctx.record(node, ABSENT)


def _matmul_command(
    node: torch.fx.Node,
    ctx,
    lhs,
    rhs,
    out,
    batches: int,
    m: int,
    k: int,
    n: int,
    hmx_prepacked: bool = False,
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
    steps = (
        (m_plan * n_plan, m_plan * k_plan, k_plan * n_plan)
        if batches > 1
        else (0, 0, 0)
    )
    op_index = ctx.emit(
        node,
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
            ),
        ),
    )
    if ctx.is_dynamic_dim(m):
        m_scale = _dynamic_scale(ctx, m)
        ctx.add_dynamic_patch(op_index, 2, m_scale, 0)
        ctx.add_dynamic_patch(op_index, 20, batches * n * m_scale, 0)
        ctx.add_dynamic_patch(op_index, 21, batches * k * m_scale, 0)


def _emit_mm(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.mm as BATCH_MATMUL, or as one GEMV when its weight is quantized."""
    lhs, rhs = node.args[0], node.args[1]
    quantized = quantized_weight(rhs)
    if quantized is not None:
        return _emit_quantized_matmul(node, ctx, quantized, lhs, None)
    _require_arena_dtype(node, "mm")
    lhs_val, rhs_val = lhs.meta["val"], rhs.meta["val"]
    if not (lhs_val.is_contiguous() and rhs_val.is_contiguous()):
        raise RuntimeError(
            "hexagon: mm operands must be contiguous; strides come from shape"
        )
    m, k = lhs_val.shape
    contracted, n = rhs_val.shape
    if k != contracted:
        raise RuntimeError(f"hexagon: mm contracts {k} against {contracted}")

    weight, prepacked = _weight_operand(
        ctx, rhs, ctx.upper_bound(m), ctx.upper_bound(k), ctx.upper_bound(n)
    )
    out = ctx.result_for(node, m * n)
    _matmul_command(
        node,
        ctx,
        ctx.operand(lhs),
        weight,
        out,
        1,
        m,
        k,
        n,
        hmx_prepacked=prepacked,
    )
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
        raise RuntimeError(
            "hexagon: bmm operands must be contiguous; strides come from shape"
        )
    batches, m, k = lhs_val.shape
    rhs_batches, contracted, n = rhs_val.shape
    if batches != rhs_batches or k != contracted:
        raise RuntimeError(
            f"hexagon: bmm contracts {batches}x{k} against {rhs_batches}x{contracted}"
        )

    weight, prepacked = _weight_operand(
        ctx, rhs, ctx.upper_bound(m), ctx.upper_bound(k), ctx.upper_bound(n)
    )
    out = ctx.result_for(node, batches * m * n)
    _matmul_command(
        node,
        ctx,
        ctx.operand(lhs),
        weight,
        out,
        batches,
        m,
        k,
        n,
        hmx_prepacked=prepacked,
    )
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
    quantized = quantized_weight(rhs)
    if quantized is not None:
        # The quantized kernel has a bias operand of its own, so the graph's bias
        # goes to the kernel rather than to a second command; beta == 0 is the
        # one other spelling the support check lets through, and it drops it.
        return _emit_quantized_matmul(
            node, ctx, quantized, lhs, None if beta == 0.0 else bias
        )
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
    weight, prepacked = _weight_operand(
        ctx, rhs, ctx.upper_bound(m), ctx.upper_bound(k), ctx.upper_bound(n)
    )
    _matmul_command(
        node,
        ctx,
        ctx.operand(lhs),
        weight,
        target,
        1,
        m,
        k,
        n,
        hmx_prepacked=prepacked,
    )
    if beta == 0.0:
        return ctx.record(node, out)

    op_index = ctx.emit(
        node,
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
        ),
    )
    if ctx.is_dynamic_dim(m):
        ctx.add_dynamic_patch(op_index, 0, n, 0)
        ctx.add_dynamic_patch(op_index, 1, n, 0)
    return ctx.record(node, out)


# The pooling family. The kernel is `hvx_pool2d_fp16`, which walks one window
# per output position accumulating whole HVX vectors, so every lane is one
# channel and the window is a strided walk over the same lane of a run of
# spatial positions.
MAX_POOL2D = exir_ops.edge.aten.max_pool2d.default
# to_edge rewrites max_pool2d into the indices form plus a getitem, so the
# values here are what a graph actually carries; the indices have no kernel and
# only a getitem 0 reader can be placed.
MAX_POOL2D_WITH_INDICES = exir_ops.edge.aten.max_pool2d_with_indices.default
AVG_POOL2D = exir_ops.edge.aten.avg_pool2d.default
POOL_TARGETS = frozenset({MAX_POOL2D, MAX_POOL2D_WITH_INDICES, AVG_POOL2D})
MAX_POOL_TARGETS = frozenset({MAX_POOL2D, MAX_POOL2D_WITH_INDICES})


def max_pool_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The max pool a getitem reads, when it reads the values and not the indices.

    max_pool2d_with_indices returns (values, indices); the kernel produces the
    values, so getitem 0 is the only reader a partition can carry.
    """
    if node.target is not GETITEM or len(node.args) != 2:
        return None
    source, index = node.args
    if index != 0 or not isinstance(source, torch.fx.Node):
        return None
    return source if source.target is MAX_POOL2D_WITH_INDICES else None


def max_pool_is_emittable(node: torch.fx.Node) -> bool:
    """Whether every reader of this max pool takes the values.

    The indices come out of the same node and nothing here produces them, so a
    graph that reads them -- or the tuple itself -- keeps the whole node on the
    portable kernels, readers of the values included.
    """
    if node.target is not MAX_POOL2D_WITH_INDICES:
        return True
    return bool(node.users) and all(
        max_pool_getitem(reader) is node for reader in node.users
    )


def operand_dtypes_are_readable(node: torch.fx.Node) -> bool:
    """Whether every tensor operand is a width the arena holds.

    A bool is one byte per element in the `.pte` and two in the arena's
    arithmetic, so a kernel handed one reads the neighbouring slot's bytes as
    the value: no error, just the wrong numbers. The case is reachable from an
    ordinary `x + (a > b)`, whose comparison is portable for a dtype reason of
    its own, so the consumer is refused here rather than left to read it.
    """
    for arg in node.args:
        value = arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None
        if isinstance(value, torch.Tensor) and value.dtype is torch.bool:
            return False
    return True


class PoolSpec(NamedTuple):
    """Everything the pool command carries, once the node is known to fit."""

    batch: int
    ih: int
    iw: int
    oh: int
    ow: int
    kernel_y: int
    kernel_x: int
    stride_y: int
    stride_x: int
    pad_y: int
    pad_x: int
    count_type: int
    pool_type: int


def _pool_arg(node: torch.fx.Node, name: str, index: int, default):
    """A pool argument from wherever the graph put it.

    A keyword argument lives in kwargs and a positional one in args, and the two
    are the same argument to the op; either spelling may carry the kernel size,
    the count_include_pad flag or the missing optional.
    """
    if name in node.kwargs:
        return node.kwargs[name]
    return node.args[index] if len(node.args) > index else default


def _int_pair(value, default) -> Optional[tuple]:
    """A two-entry kernel/stride/padding argument, or None if it is not one.

    The exported graph spells these as ``[n, n]``, and the schema defaults to an
    empty list (`int[2] stride=[]`), which torch reads as "the same as the
    kernel"; a bare int is accepted too because the schema's type is a list of
    ints but a module built with ints exports either way.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and 1 <= len(value) <= 2:
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            return None
        return (value[0], value[-1])
    return None


def _pool_window_intersects(
    size: int, output: int, kernel: int, stride: int, pad: int
) -> bool:
    """Whether every window holds at least one input row/column.

    A window is ``[oy * stride - pad, oy * stride - pad + kernel)``
    (`pool_fp16.c:26`); the kernel skips whatever falls outside and writes zero
    when nothing is left, where torch's max would be -inf. For the paddings
    torch itself accepts (at most half the kernel) this can never happen, so the
    answer is a guard rather than a shape rule, and it is checked on both axes.
    """
    for index in range(output):
        origin = index * stride - pad
        if origin + kernel <= 0 or origin >= size:
            return False
    return True


def pool_spec(node: torch.fx.Node) -> Optional[PoolSpec]:
    """The pool command's parameters, or None when this node is not that shape.

    Both the support check and the emitter call this, so they cannot disagree:
    anything the emitter would refuse has to be refused here instead, or the
    whole export fails rather than the node staying on a portable kernel.
    """
    if node.target not in POOL_TARGETS:
        return None
    args = node.args
    if not args or not isinstance(args[0], torch.fx.Node):
        return None
    value = args[0].meta.get("val")
    result = node.meta.get("val")
    # max_pool2d_with_indices carries (values, indices); the values are the
    # tensor the pool writes.
    if isinstance(result, tuple):
        result = result[0] if result else None
    if not isinstance(value, torch.Tensor) or not isinstance(result, torch.Tensor):
        return None
    if value.dtype not in (torch.float16, torch.float32):
        return None
    shape = list(value.shape)
    if len(shape) not in (3, 4) or len(result.shape) != len(shape):
        return None
    # The pooled buffer is described by three ints per axis, and this emitter
    # derives them from static shapes only: a symbolic extent is refused here
    # rather than emitted with the wrong stride.
    if any(isinstance(dim, torch.SymInt) for dim in shape + list(result.shape)):
        return None
    if len(shape) == 4:
        batch, channels, ih, iw = shape
    else:
        batch, channels, ih, iw = 1, shape[0], shape[1], shape[2]
    # The kernel reads its activation as [ceil(C/64)][batch][h*w][64] blocks,
    # which is not the row-major [batch][C][h*w] the arena holds. At C == 64
    # there is exactly one block and this emitter can hand the kernel that block
    # with one blit either side; any other channel count is a padded or
    # multi-block grid this batch does not build.
    if channels != POOL_CHANNEL_BLOCK:
        return None

    kernel = _int_pair(_pool_arg(node, "kernel_size", 1, None), None)
    if kernel is None or kernel[0] <= 0 or kernel[1] <= 0:
        return None
    stride = _int_pair(_pool_arg(node, "stride", 2, None), kernel)
    if stride is None or stride[0] <= 0 or stride[1] <= 0:
        return None
    padding = _int_pair(_pool_arg(node, "padding", 3, None), (0, 0))
    if padding is None or padding[0] < 0 or padding[1] < 0:
        return None

    if node.target in MAX_POOL_TARGETS:
        dilation = _int_pair(_pool_arg(node, "dilation", 4, None), (1, 1))
        if dilation != (1, 1):
            # The kernel has no dilation: it steps the window by one
            # (pool_fp16.c:34-39).
            return None
        ceil_mode = _pool_arg(node, "ceil_mode", 5, False)
        count_type = POOL_COUNT_KERNEL  # ignored by the max path
        pool_type = POOL_MAX
    else:
        ceil_mode = _pool_arg(node, "ceil_mode", 4, False)
        count_include_pad = _pool_arg(node, "count_include_pad", 5, True)
        divisor_override = _pool_arg(node, "divisor_override", 6, None)
        if divisor_override is not None:
            # The divisor is what the kernel's countType selects; an arbitrary
            # one has no command form.
            return None
        count_type = POOL_COUNT_KERNEL if count_include_pad else POOL_COUNT_VALID
        pool_type = POOL_AVERAGE
    if ceil_mode:
        # ceil_mode adds windows past the input that this kernel's geometry (one
        # window per output position at oy * stride - pad) does not describe.
        return None

    oh, ow = result.shape[-2], result.shape[-1]
    # torch's floor-mode output size, which is also the one the emitter is about
    # to describe with strides.
    if oh != (ih + 2 * padding[0] - kernel[0]) // stride[0] + 1:
        return None
    if ow != (iw + 2 * padding[1] - kernel[1]) // stride[1] + 1:
        return None
    if not _pool_window_intersects(ih, oh, kernel[0], stride[0], padding[0]):
        return None
    if not _pool_window_intersects(iw, ow, kernel[1], stride[1], padding[1]):
        return None

    return PoolSpec(
        batch=batch,
        ih=ih,
        iw=iw,
        oh=oh,
        ow=ow,
        kernel_y=kernel[0],
        kernel_x=kernel[1],
        stride_y=stride[0],
        stride_x=stride[1],
        pad_y=padding[0],
        pad_x=padding[1],
        count_type=count_type,
        pool_type=pool_type,
    )


def _channel_block_region(batch: int, area: int, channels: int, packing: bool) -> list:
    """The blit region that moves one 64-channel block between the two layouts.

    `packing` reads row-major ``[batch][channels][area]`` and writes the
    kernel's ``[batch][area][64]``; the other direction is the same region with
    the two stride triples exchanged. The two are exactly the geometries the
    DSP's own pack paths are written for: with `channels == 64` and this stride
    pair, `htp_ops_try_pack_area_blit` (blit_ops.cc:816-831) takes the case and
    reads/writes element ``(b, c, x)`` as
    ``dst[b * area * 64 + x * 64 + c] = src[b * 64 * area + c * area + x]``,
    which is the same mapping this region describes.
    """
    inner = (
        [channels * area, area, 1],
        [area * POOL_CHANNEL_BLOCK, 1, POOL_CHANNEL_BLOCK],
    )
    if not packing:
        inner = (inner[1], inner[0])
    return [0, 0, 0, batch, POOL_CHANNEL_BLOCK, area] + list(inner[0]) + list(inner[1])


def _emit_pool2d(node: torch.fx.Node, ctx) -> TensorRef:
    """max_pool2d / avg_pool2d as one POOL2D_FP16, blocked either side.

    The kernel reads and writes its activation in the DSP's 64-channel blocked
    layout, so the row-major buffer the arena holds has to be rearranged into it
    before the window walk and back out after. That is two blits around the
    command, and they are not optional: reading the blocked layout's
    ``(y * width + x) * 64`` step over a row-major buffer would return other
    channels' values at every position. A spatial extent of one is the one case
    where the two layouts agree element for element, and the blit is dropped.
    """
    spec = pool_spec(node)
    if spec is None:
        raise RuntimeError(
            "hexagon: this pool2d is not one the DSP kernel can run (see pool_spec)"
        )
    _require_arena_dtype(node, "pool2d")
    source = ctx.operand(node.args[0])
    # max_pool2d_with_indices hands its values on through a getitem, and that
    # getitem is what downstream reads, so its output slot is the one to fill.
    sink = next(
        (reader for reader in node.users if max_pool_getitem(reader) is node),
        node,
    )
    out = ctx.result_for(sink, _numel(node))
    area = spec.ih * spec.iw
    out_area = spec.oh * spec.ow
    channels = POOL_CHANNEL_BLOCK

    packed_in = source
    if area != 1:
        packed_in = ctx.builder.add_activation(
            spec.batch * area * channels * FP16_BYTES
        )
        ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[source],
                outputs=[packed_in],
                params=[1, FP16_BYTES, 1]
                + _channel_block_region(spec.batch, area, channels, True),
            ),
        )

    pooled = (
        out
        if out_area == 1
        else ctx.builder.add_activation(spec.batch * out_area * channels * FP16_BYTES)
    )
    ctx.emit(
        node,
        Op(
            type=DSP_OP_POOL2D_FP16,
            inputs=[packed_in],
            outputs=[pooled],
            params=[
                spec.batch,
                spec.ih,
                spec.iw,
                spec.oh,
                spec.ow,
                # One block: the support check only lets C == 64 through.
                1,
                spec.kernel_y,
                spec.kernel_x,
                spec.stride_y,
                spec.stride_x,
                spec.pad_y,
                spec.pad_x,
                POOL_PAD_TYPE,
                spec.count_type,
                spec.pool_type,
            ],
        ),
    )
    if out_area != 1:
        ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[pooled],
                outputs=[out],
                params=[1, FP16_BYTES, 1]
                + _channel_block_region(spec.batch, out_area, channels, False),
            ),
        )
    return ctx.record(node, out)


# The reductions. The DSP collapses one contiguous (outside, reduce, inside)
# span of the buffer, so the reduced dims have to be adjacent; the caller's
# support checks enforce that before an emitter ever runs.
SUM_DIM = exir_ops.edge.aten.sum.dim_IntList
AMAX = exir_ops.edge.aten.amax.default
REDUCTION_TARGETS = frozenset({SUM_DIM, AMAX})
SUM_TARGETS = frozenset({SUM_DIM})


def reduction_dims(node: torch.fx.Node) -> Optional[list]:
    """The reduced axes as positive indices, or None when they are not a span.

    A missing dim and an empty list both mean every dim -- torch reduces the
    whole tensor for either -- which is the single span the kernel wants.
    """
    src = node.args[0] if node.args else None
    if not isinstance(src, torch.fx.Node):
        return None
    value = src.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dim() == 0:
        return None
    dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    rank = value.dim()
    if dims is None or (isinstance(dims, (list, tuple)) and not dims):
        return list(range(rank))
    dims = [dims] if isinstance(dims, int) else list(dims)
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims):
        return None
    norm = sorted(dim % rank for dim in dims)
    if norm != list(range(norm[0], norm[0] + len(norm))):
        return None
    return norm


def _emit_reduction(node: torch.fx.Node, ctx, kind: int) -> TensorRef:
    """One command for a reduction over one contiguous span.

    The three params are that span as ``[outside][reduce][inside]``, which is
    what `htp_ops_reduction` walks (eltwise_ops.cc:2812-2862). Sum and maximum
    are both accumulated per lane there -- the maximum seeds from the first
    element rather than from zero, so an all-negative window is not clamped.
    """
    src = node.args[0]
    _require_arena_dtype(node, "reduction")
    dims = reduction_dims(node)
    if dims is None:
        raise RuntimeError("hexagon: this reduction's dims are not one span")
    shape = src.meta["val"].shape
    outside = shape[: dims[0]]
    span = shape[dims[0] : dims[-1] + 1]
    inside = shape[dims[-1] + 1 :]
    out = ctx.result_for(node, _numel(node))
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_REDUCTION,
            inputs=[ctx.operand(src)],
            outputs=[out],
            params=[
                _upper_product(outside, ctx),
                _upper_product(span, ctx),
                _upper_product(inside, ctx),
                kind,
                FP16_BYTES,
            ],
        ),
    )
    # Each of the three is the upper bound of a product that may hold the
    # run-time length, so each is patched with the length it was written for --
    # the pairing every other shape-derived param here has. Without it the
    # command describes the allocation rather than the buffer: a sum over the
    # sequence, or a whole-tensor mean, folds in whatever the arena holds past
    # the run-time length and answers with it.
    _patch_dynamic_product(ctx, op_index, outside, 0)
    _patch_dynamic_product(ctx, op_index, span, 1)
    _patch_dynamic_product(ctx, op_index, inside, 2)
    return ctx.record(node, out)


def _emit_sum_dim(node: torch.fx.Node, ctx) -> TensorRef:
    if not sum_dim_is_emittable(node):
        raise RuntimeError("hexagon: this sum reduces into another dtype")
    return _emit_reduction(node, ctx, REDUCTION_SUM)


def _emit_amax(node: torch.fx.Node, ctx) -> TensorRef:
    return _emit_reduction(node, ctx, REDUCTION_MAXIMUM)


def sum_dim_is_emittable(node: torch.fx.Node) -> bool:
    """Whether this sum is the fp16 sum the kernel computes.

    An accumulator of another width is a different op, not a narrower command:
    the kernel sums in fp32 and stores fp16 (`htp_ops_reduce_fp16_scalar_range`,
    eltwise_ops.cc:2507-2525). The schema makes dtype keyword-only, so the graph
    carries it in kwargs; the positional slot is checked as well because a
    hand-built node may put it there.
    """
    dtype = node.kwargs.get("dtype")
    if dtype is None and len(node.args) > 3:
        dtype = node.args[3]
    return dtype is None


def _emit_mean_dim(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.mean.dim as a single REDUCTION."""
    return _emit_reduction(node, ctx, REDUCTION_MEAN)


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
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_SOFTMAX,
            inputs=[ctx.operand(src)],
            outputs=[out],
            # The DSP reduces the middle axis of an [outside][channel][inside]
            # view, so the reduction dim is described by its strides.
            params=[outside, channel, inside, FP16_BYTES],
        ),
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
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_LAYER_NORM,
            # beta is ABSENT: RMSNorm has no bias, and a zero-size operand still
            # maps to a live address the kernel would read as data.
            inputs=[ctx.operand(source), gamma, ABSENT],
            outputs=[out],
            params=[outer, inner, _float_bits(float(eps)), 1],
        ),
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
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_ROPE,
            inputs=inputs,
            outputs=[out, out],
            params=[batch_seq, num_head, 0, head_dim, head_dim, 0],
        ),
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
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_ADD_FUSE_LAYERNORM,
            # beta is ABSENT: RMSNorm has no bias and the kernel's RMSNorm path
            # only fires when it is null.
            inputs=[ctx.operand(residual), ctx.operand(branch), gamma, ABSENT],
            outputs=[normalized, residual_out],
            params=[outer, inner, _float_bits(float(eps)), 1],
        ),
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
    return [int(size) for size in normalized] == shape[len(shape) - len(normalized) :]


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
    affine = [
        (arg, kind) for arg, kind in ((weight, "mul"), (bias, "add")) if arg is not None
    ]
    normalized = ctx.activation_for_shape(shape) if affine else out
    norm_op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_LAYER_NORM,
            # Order matters: dst is the DSP's mapped_ptrs[3], so the three
            # inputs must be exactly src, gamma, beta -- null here, which the
            # kernel tests for and skips.
            inputs=[ctx.operand(src), ABSENT, ABSENT],
            outputs=[normalized],
            params=[outer, inner, _float_bits(eps), 0],
        ),
    )
    _patch_dynamic_product(ctx, norm_op_index, outer_shape, 0)

    result = normalized
    for index, (arg, kind) in enumerate(affine):
        target = out if index == len(affine) - 1 else ctx.activation_for_shape(shape)
        affine_op_index = ctx.emit(
            node,
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
                    *_broadcast_tail(
                        (outer, inner),
                        tuple(arg.meta["val"].shape),
                        (outer, inner),
                        ctx,
                    ),
                ],
            ),
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
    query, key = args[0], args[1]
    if len(args) > 4 and args[4] is not None:
        # The mask slot has no stride this emitter can fill in: the kernel reads
        # the mask as fp16 rows of `mask_stride` columns and copies them into a
        # fp32 region it places after the per-task rows, out of a workspace this
        # backend sizes for the unmasked shape. A command with mask_stride = -1
        # binds a mask the kernel then ignores, which is a wrong answer rather
        # than a failure, so a masked node is refused where it is delegated and
        # again here, for a graph that reaches preprocess without the
        # partitioner (the tests call it that way).
        raise RuntimeError(
            "hexagon: sdpa with an attention mask is not emittable: the mask "
            "stride and the fp32 workspace it is copied into are unsized"
        )
    # Unlike the other ops, attention accepts fp32: the runtime narrows those
    # operands to fp16 as they enter the arena, so the DSP still sees fp16.
    _dtype = node.meta["val"].dtype
    if _dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"hexagon: sdpa input must be fp16 or fp32, got {_dtype}")

    scale = args[7] if len(args) > 7 else None
    q_shape = query.meta["val"].shape
    kv_shape = key.meta["val"].shape

    # The kernel sizes and indexes this source as [batch, seq, heads, dim]:
    # attention_entry.cc strides it by tokens * heads * headDim, which is the
    # layout the upstream caller builds by transposing a head-major cache
    # (examples/models/llama/source_transformation/sdpa.py), and the layout the
    # vision entry point reads its operands in (attention_entry.cc:36,41).
    # Slots four and five are the packed cache
    # the kernel writes, not the cache itself -- attn_hmx_k_tile_index lays it
    # out by tile: 256 tokens per block, eight 32-row sequence tiles per block,
    # one 1024-element tile per (32 dim x 32 seq) sub-block. Any cache length
    # pays for whole blocks, the same span attention_entry.cc sizes its own
    # packed K/V to.
    if len(kv_shape) != 4 or kv_shape[3] != q_shape[3]:
        raise RuntimeError(
            f"hexagon: sdpa cache {tuple(kv_shape)} does not match head_dim {q_shape[3]}"
        )
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
        # Slot three carries no mask: the dispatcher maps an absent ref to a
        # null pointer, which is the "no mask" the command below asks for.
        ABSENT,
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

    # The paged entry (htp_ops_flash_attn_pages) is what MNN's
    # HexagonAttention.cpp always uses: one page spanning the whole packed cache,
    # so the buffer layout is unchanged. page_size must be a multiple of 32 and
    # must exceed seq_current, or push_kv_pages skips the insert. The choice is a
    # compile spec, because it is a page size in the command and so a byte of the
    # blob; init() reads it back out of the command when it checks the specs
    # against what the file actually holds.
    paged = ctx.options.attn_paged
    page_size = 256 if paged else 0

    op_index = ctx.emit(
        node,
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
                -1,  # mask_stride: no mask, which is also the causal mode
                # push_kv only reads this as the capacity its writes must stay
                # inside, so it is the operand's length, not the cache's.
                cache_capacity,  # max_kv_len
                seq_blocks if paged else 0,  # page_count
                page_size,  # page_size
                0,  # value_c4
            ],
            patch=patch,
        ),
    )
    if ctx.is_dynamic_dim(q_shape[1]):
        ctx.add_dynamic_patch(op_index, 0, 1, 0)
        ctx.add_dynamic_patch(op_index, 2, 1, 0)
    return ctx.record(node, out)


# The vision tower's attention. `et_hexagon.vision_attention` is the decomposed
# pattern -- head split, q k^T, scale, softmax, v -- stated once, and its
# operands are the tensors under the head transposes because that is the layout
# the kernel walks.
VISION_ATTENTION_TARGETS = frozenset({VISION_ATTENTION})


def _vision_attention_workspace_bytes(length) -> int:
    """htp_ops_vision_attention_fp16's scratch, as a function of the token count.

    The kernel wants `tokens` fp32 scores and aligns the pointer up by 127 bytes
    itself (`attention_entry.cc:26-31`); it refuses the command outright below
    that, and there is no workspace it will allocate for itself.
    """
    return int(length) * 4 + 128


def vision_attention_is_emittable(node: torch.fx.Node) -> bool:
    """Whether this fused attention is one the kernel can be handed.

    The batch, head count and head width reach the command as params and are
    never patched, so a symbolic one would emit the traced example and compute a
    different function at run time. The token count is the one number the
    run-time sequence length can supply, which is what a dynamic patch is for.
    """
    if node.target not in VISION_ATTENTION_TARGETS or len(node.args) < 4:
        return False
    query, key, value = node.args[0], node.args[1], node.args[2]
    if not all(isinstance(operand, torch.fx.Node) for operand in (query, key, value)):
        return False
    shape = _shape_of(query)
    if len(shape) != 4 or _shape_of(key) != shape or _shape_of(value) != shape:
        return False
    if any(isinstance(dim, torch.SymInt) for dim in (shape[0], shape[2], shape[3])):
        return False
    if any(not int(dim) > 0 for dim in (shape[0], shape[2], shape[3])):
        return False
    # The scale is a param rather than an operand, so it has to be a number the
    # pass could read out of the graph at export time.
    scale = node.args[3]
    return isinstance(scale, (int, float)) and not isinstance(scale, bool)


def _emit_vision_attention(node: torch.fx.Node, ctx) -> TensorRef:
    """et_hexagon.vision_attention as one VISION_ATTENTION_FP16.

    `execute_command.cc:784-789` hands the kernel input 0/1/2 as query, key and
    value, slot three as the mask, the first output as the result and the second
    as the workspace, with `intParams[0..3]` the batch, token count, head count
    and head width, `floatParams[4]` the scale, `intParams[5]` the mask stride
    and `intParams[6]` the workspace size. Only three operands are bound, so the
    dispatcher passes a null mask pointer and the kernel's `maskStride > 0` test
    never reads one (`attention_entry.cc:49`) -- a mask has no operand to hide
    in, which is the mistake the FLASH_ATTN emitter made with `-1`.

    The operands are bound as the graph carries them: `[batch, tokens, heads,
    headDim]`, which `attention_entry.cc:36,41,57` strides by `heads * headDim`
    per token and reads `headDim` at a time inside it.
    """
    if not vision_attention_is_emittable(node):
        raise RuntimeError(f"hexagon: vision attention shape {_shape_of(node.args[0])}")
    query, key, value = node.args[0], node.args[1], node.args[2]
    scale = float(node.args[3])
    batch, tokens, heads, head_dim = _shape_of(query)
    batch, heads, head_dim = int(batch), int(heads), int(head_dim)
    # The token count is a param the run-time length recomputes, so the value
    # that goes in the blob is the longest the export declared.
    token_upper = ctx.upper_bound(tokens)
    workspace_bytes = _vision_attention_workspace_bytes(token_upper)

    out = ctx.result_for(node, _numel(node))
    workspace = ctx.builder.add_activation(
        workspace_bytes,
        dynamic_layout=ctx.dynamic_bytes_for_function(
            workspace_bytes, _vision_attention_workspace_bytes
        ),
    )
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_VISION_ATTENTION_FP16,
            inputs=[ctx.operand(query), ctx.operand(key), ctx.operand(value)],
            outputs=[out, workspace],
            params=[
                batch,  # batch
                token_upper,  # tokens, the same count for query and key
                heads,  # heads
                head_dim,  # headDim
                _float_bits(scale),  # scale, read through a float* cast
                0,  # maskStride, which no operand lets the kernel read
                workspace_bytes,  # providedWorkspaceBytes
            ],
        ),
    )
    _patch_dynamic_product(ctx, op_index, [tokens], 1)
    return ctx.record(node, out)


# The row gathers: one command reads k rows out of a table that lives in the
# weights section. `embedding` is every LLM's token table -- the largest single
# weight in the model, and one the DSP could not read at all before this -- and
# `index_select`/`index.Tensor` over axis 0 are the same read written another
# way.
EMBEDDING = exir_ops.edge.aten.embedding.default
INDEX_SELECT = exir_ops.edge.aten.index_select.default
INDEX_TENSOR = exir_ops.edge.aten.index.Tensor

GATHER_TARGETS = frozenset({EMBEDDING, INDEX_SELECT, INDEX_TENSOR})


class GatherTable(NamedTuple):
    """The operands and geometry of one row gather."""

    #: The constant whose rows are gathered.
    table: torch.fx.Node
    #: The runtime int32 tensor naming those rows.
    indices: torch.fx.Node
    #: Table rows, as htp_ops_shared_gather's `oc`.
    oc: int
    #: Elements per table row, as its `ic`.
    ic: int
    #: Rows the command gathers, as its `selectSize`.
    select_size: int
    indices_shape: tuple


def gather_table(node: torch.fx.Node, is_constant) -> Optional[GatherTable]:
    """The table and indices this node gathers rows with, or None.

    Returns None for every shape whose result is *not* a row gather, because an
    emitter that cannot express a node has no way to say so without failing the
    whole export: the node has to be rejected here instead, which puts it back on
    a portable kernel.

    The two checks that are not geometry are about what the DSP can be handed.
    The table has to be a value this layer can read now -- the kernel reads a
    tiled table, so its bytes are rearranged at export (`pack_shared_gather_table`)
    and a table that only exists at run time has no bytes to rearrange. The
    indices have to be the caller's own int32 tensor: the kernel reads them as
    `const int32_t[]`, and an int64 index tensor is not one, so the whole op stays
    on a portable kernel until the runtime can narrow it (see README).

    `is_constant` is the caller's own test for "a value I can read now". The
    support check passes the graph's, keyed on the names the partitioner tags as
    delegate data, and the emitter passes `ctx.constant_value`, which is the one
    that can actually see the tensor. They agree on every operand: a parameter,
    buffer or lifted constant is both tagged and visible, and a method input is
    neither. Both call this function, so the shape one accepts is the shape the
    other emits.
    """
    table_arg, index_arg = _gather_operands(node)
    if table_arg is None or index_arg is None:
        return None
    table_value = table_arg.meta.get("val")
    indices_value = index_arg.meta.get("val")
    if not isinstance(table_value, torch.Tensor) or not isinstance(
        indices_value, torch.Tensor
    ):
        return None
    if not is_constant(table_arg) or is_constant(index_arg):
        return None
    # The kernel reads the indices as `const int32_t[]`, and a method input keeps
    # the width its own dtype declares, so an int64 index tensor would be read as
    # alternating low and high words. The narrower type is also what makes the
    # operand safe to read at four bytes an element wherever it comes from: an
    # op whose result is int32 is never delegated -- the support check only
    # accepts fp16 and fp32 results -- so an int32 tensor can only reach a
    # command as a method input, which is the width the emitter declares for it.
    if indices_value.dtype is not torch.int32:
        return None
    if indices_value.dim() < 1:
        return None
    indices_shape = tuple(indices_value.shape)
    # One run of rows per call is what the command's selectSize describes, so a
    # second moving axis has no way to reach it. The exported sequence symbol is
    # the one dynamic axis these graphs carry.
    if sum(isinstance(dim, torch.SymInt) for dim in indices_shape) > 1:
        return None
    if table_value.dim() != 2 or not table_value.is_contiguous():
        return None
    # Both widths the arena holds reach the same two-byte table.
    if table_value.dtype not in (torch.float16, torch.float32):
        return None
    oc, ic = table_value.shape
    if not (isinstance(oc, int) and isinstance(ic, int)) or oc <= 0 or ic <= 0:
        return None
    # The command carries oc and ic as int32 params and the kernel derives every
    # element offset from them, so a table past that is refused rather than
    # emitted with arithmetic that would have wrapped. No model is close: this is
    # 4 GiB of fp16 across ceil(oc/32) * ceil(ic/32) whole 32x32 tiles.
    if oc > 2**31 - 1 or ic > 2**31 - 1:
        return None
    if -(-oc // 32) * -(-ic // 32) * 1024 > 2**31 - 1:
        return None
    return GatherTable(
        table_arg,
        index_arg,
        int(oc),
        int(ic),
        _upper_product(indices_shape),
        indices_shape,
    )


def _gather_operands(node: torch.fx.Node):
    """The (table, indices) operands of a row gather, or (None, None).

    `embedding(weight, indices)` reads rows of its weight; `index_select` and
    `index.Tensor` read rows of their first operand, and are the same read only
    where they name axis 0 and nothing else.
    """
    if node.target is EMBEDDING:
        if len(node.args) < 2:
            return None, None
        return node.args[0], node.args[1]
    if node.target is INDEX_SELECT:
        if len(node.args) < 3 or node.args[1] != 0:
            return None, None
        return node.args[0], node.args[2]
    if node.target is INDEX_TENSOR:
        names = node.args[1] if len(node.args) > 1 else None
        # A list of one index reads axis 0; a second entry indexes a second
        # axis, which is a gather over two axes and not this command.
        if not isinstance(names, (list, tuple)) or len(names) != 1:
            return None, None
        return node.args[0], names[0]
    return None, None


def pack_shared_gather_table(weight, oc: int, ic: int) -> bytes:
    """A (oc, ic) fp16 table in the order htp_ops_shared_gather reads it.

    The fp16 path of the kernel is not a row gather over a row-major table: the
    table is a grid of 32x32 tiles in which adjacent column pairs come first, so
    element (row, col) of the tile at (row // 32, col // 32) sits at
    ((col % 32) // 2) * 64 + (row % 32) * 2 + ((col % 32) & 1), which is what
    shared_gather_ops.cc:296-311 computes. Tiles run row-major over the grid, and
    a tile is 1024 elements whether or not the table fills it, so this costs
    ceil(oc/32) * ceil(ic/32) * 2048 bytes: the row-major size itself when both
    sides are multiples of 32, and up to 32x it when the row is one element wide.
    """
    import numpy as np

    w = weight.astype(np.float16, copy=False)
    if w.shape != (oc, ic):
        raise RuntimeError(f"hexagon: table is {w.shape}, expected ({oc}, {ic})")
    rows = -(-oc // 32)
    columns = -(-ic // 32)
    padded = np.zeros((rows * 32, columns * 32), dtype=np.float16)
    padded[:oc, :ic] = w
    # (row tile, row in tile, column tile, column pair, element of pair), read
    # back in the order the kernel walks it.
    tiles = padded.reshape(rows, 32, columns, 16, 2).transpose(0, 2, 3, 1, 4)
    return np.ascontiguousarray(tiles).tobytes()


def _emit_gather(node: torch.fx.Node, ctx) -> TensorRef:
    """One SHARED_GATHER command for the whole row gather.

    The command's operands are the indices and then the table, in that order:
    `htp_ops_shared_gather(mapped_ptrs[inputs->size()], mapped_ptrs[0],
    mapped_ptrs[1], ...)` reads the output past every input and takes the first
    input as the indices and the second as the table (execute_command.cc:806-811).
    Five params are enough for it: the two the quantized paths read past them
    (scaleBlockNum, scaleAsymmetric) have no slot here and the dispatcher supplies
    its own defaults, which the fp16 path never looks at.
    """
    fit = gather_table(node, lambda operand: ctx.constant_value(operand) is not None)
    if fit is None:
        raise RuntimeError(
            f"hexagon: no SHARED_GATHER command for {node.name}; the partitioner "
            "should not have delegated it"
        )
    table = ctx.constant_value(fit.table)
    if table is None:
        raise RuntimeError(f"hexagon: no value for the table of {node.name}")
    out = ctx.result_for(node, _numel(node))
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_SHARED_GATHER,
            inputs=[
                ctx.operand(fit.indices),
                ctx.gather_table(fit.table, table, fit.oc, fit.ic),
            ],
            outputs=[out],
            params=[
                fit.select_size,
                fit.ic,
                fit.oc,
                FP16_BYTES,
                SHARED_GATHER_FP16,
            ],
        ),
    )
    # The rows gathered are the run of tokens this call was handed, so the count
    # is the exported symbol's value rather than the length the graph was traced
    # at.
    _patch_dynamic_product(ctx, op_index, fit.indices_shape, 0)
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
    # The weight-only quantized matmul's weight arrives through this node; the
    # matmul emitter reads it and the node itself emits nothing.
    DQ_PER_CHANNEL: _emit_dequantize,
    exir_ops.edge.aten.mean.dim: _emit_mean_dim,
    SUM_DIM: _emit_sum_dim,
    AMAX: _emit_amax,
    MAX_POOL2D: _emit_pool2d,
    MAX_POOL2D_WITH_INDICES: _emit_pool2d,
    AVG_POOL2D: _emit_pool2d,
    exir_ops.edge.aten.fmod.Tensor: _binary("mod"),
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
    ADD_RELU: _binary("add_relu"),
    ROPE: _emit_rope,
    EMBEDDING: _emit_gather,
    INDEX_SELECT: _emit_gather,
    INDEX_TENSOR: _emit_gather,
    # A vision tower's attention, which `FuseVisionAttention` states once.
    VISION_ATTENTION: _emit_vision_attention,
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
        exir_ops.edge.aten.fmod.Tensor,
        MUL_SILU,
        ADD_RELU,
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
