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

import struct
from typing import Dict, List, NamedTuple, Optional

import torch
from executorch.backends.hexagon.kv_cache import UPDATE_CACHE
from executorch.backends.hexagon.rms_norm import RMS_NORM

# After rms_norm, which opens the et_hexagon namespace this fragment joins.
from executorch.backends.hexagon.mul_silu import MUL_SILU
from executorch.backends.hexagon.serialization.blob import ABSENT, Op, TensorRef
from executorch.exir.dialects._ops import ops as exir_ops

# DSPOpType, from third-party/mnn-htp-ops/include/htp_command.h.
DSP_OP_RASTER_BLIT = 3
DSP_OP_UNARY = 4
DSP_OP_LAYER_NORM = 8
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


def _numel(node: torch.fx.Node) -> int:
    return node.meta["val"].numel()


def _require_fp16(node: torch.fx.Node, what: str) -> None:
    dtype = node.meta["val"].dtype
    if dtype != torch.float16:
        raise RuntimeError(f"hexagon: {what} must be fp16, got {dtype}")


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
    ctx.builder.add_op(
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
    out = ctx.result_for(node, _numel(node))
    ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(tensor) for tensor in tensors],
            outputs=[out],
            params=cat_region(node),
        )
    )
    return ctx.record(node, out)


def permute_region(node: torch.fx.Node):
    """The blit region for a transpose of the last two axes, or None.

    Every `permute_copy` in the graph is `permute(w, [1, 0])` between a weight
    and its `mm`, so it is one matrix transpose per outer index. A region
    describes that exactly -- the inner run reads a contiguous source row and
    writes a strided destination column -- and it is the shape
    `htp_ops_prepare_transpose` recognises and routes to the HVX transpose.

    A permutation that moves any other axis is refused rather than approximated:
    the row index stops being linear in the destination, which a single region
    cannot describe.
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
    if source_value.dtype is not torch.float16:
        return None
    if not source_value.is_contiguous() or not result_value.is_contiguous():
        return None

    rank = len(source_value.shape)
    if rank < 2 or sorted(dims) != list(range(rank)):
        return None
    if list(dims) != list(range(rank - 2)) + [rank - 1, rank - 2]:
        return None

    rows = int(source_value.shape[rank - 2])
    cols = int(source_value.shape[rank - 1])
    outer = 1
    for size in source_value.shape[: rank - 2]:
        outer *= size
    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
    return [0, 0, 0, outer, rows, cols, rows * cols, cols, 1, rows * cols, 1, rows]


def _emit_permute_copy(node: torch.fx.Node, ctx) -> TensorRef:
    """A transpose is a strided read and a strided write, which is one region."""
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


def _unary(op_name: str):
    def emit(node: torch.fx.Node, ctx) -> TensorRef:
        src = node.args[0]
        _require_fp16(node, f"unary {op_name} input")
        numel = _numel(node)
        out = ctx.result_for(node, numel)
        ctx.builder.add_op(
            Op(
                type=DSP_OP_UNARY,
                inputs=[ctx.operand(src)],
                outputs=[out],
                # size is in elements, not bytes.
                params=[numel, UNARY_OP_TYPES[op_name], FP16_BYTES],
            )
        )
        return ctx.record(node, out)

    return emit


def _broadcast_strides(arg, out_shape):
    """Row-major strides of arg over out_shape, zero on broadcast dimensions.

    The DSP walks the output linearly and computes each operand's offset as
    sum(coord[d] * stride[d]), so a dimension of extent 1 contributes nothing.
    """
    rank = len(out_shape)
    shape = tuple(arg.meta["val"].shape) if isinstance(arg, torch.fx.Node) else ()
    padded = (1,) * (rank - len(shape)) + shape
    strides = [0] * rank
    acc = 1
    for d in range(rank - 1, -1, -1):
        strides[d] = 0 if padded[d] == 1 else acc
        acc *= padded[d]
    return strides


def _broadcast_tail(lhs, rhs, out_shape):
    """The 25 params the DSP's broadcast path reads from params[8]."""
    rank = len(out_shape)
    pad = 8 - rank
    return (
        [rank]
        + list(out_shape)
        + [0] * pad
        + _broadcast_strides(lhs, out_shape)
        + [0] * pad
        + _broadcast_strides(rhs, out_shape)
        + [0] * pad
    )


def _binary(op_name: str):
    def emit(node: torch.fx.Node, ctx) -> TensorRef:
        lhs, rhs = node.args[0], node.args[1]
        _require_fp16(node, f"binary {op_name} input")

        lhs_numel = _numel(lhs) if isinstance(lhs, torch.fx.Node) else 1
        rhs_numel = _numel(rhs) if isinstance(rhs, torch.fx.Node) else 1
        out_numel = _numel(node)
        out_shape = tuple(node.meta["val"].shape)

        out = ctx.result_for(node, out_numel)
        ctx.builder.add_op(
            Op(
                type=DSP_OP_BINARY_ELEMENTWISE,
                inputs=[ctx.operand(lhs), ctx.operand(rhs)],
                outputs=[out],
                params=[
                    out_numel,
                    lhs_numel,
                    rhs_numel,
                    BINARY_OP_TYPES[op_name],
                    FP16_BYTES,
                    FP16_BYTES,
                    0,  # inputs are not 4-byte floats
                    0,  # output is not a 4-byte float
                    *_broadcast_tail(lhs, rhs, out_shape),
                ],
            )
        )
        return ctx.record(node, out)

    return emit


# HtpOpsLoopParam, from the DSP's region_ops.h. The struct is packed, so its
# three int64 tails sit at byte offsets 76/84/92 with no padding: "<19i3q"
# reproduces that layout and "<25i" re-reads it as the int32 words the param
# vector carries.
_LOOP_PARAM = struct.Struct("<19i3q")


def _loop_param(
    loop_number: int,
    size_xyz,
    dst_stride,
    src0_stride,
    src1_stride,
    out_elems: int,
    in0_elems: int,
    in1_elems: int,
):
    """The descriptor BATCH_MATMUL reads out of params[1:].

    Sizes are in elements and strides in bytes. The DSP settles that by
    dividing the stride terms by the element size before bounds-checking them.
    """
    packed = _LOOP_PARAM.pack(
        loop_number,
        *size_xyz,
        *dst_stride,
        *src0_stride,
        *src1_stride,
        0,
        0,
        0,  # cmdSteps: unused while the three iter operands are absent
        0,
        0,
        0,  # cmdViewOffset: every operand starts at its own base
        out_elems,
        in0_elems,
        in1_elems,
    )
    return list(struct.unpack("<25i", packed))


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
    ctx.builder.add_op(
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[value, cache, pos],
            outputs=[out],
            params=[1, FP16_BYTES, 3, 0, 0, 0, 1, rows, run, 0, run, 1, 0, run, 1],
            patch=(5, 2),
            patch_scale=inner,
        )
    )
    return ctx.record(node, out)


def _emit_mm(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.mm as BATCH_MATMUL with a single loop iteration.

    The DSP takes dst from mapped_ptrs[inputs->size()] and reads iter0..2 out of
    slots 2..4, so the three unused iterators still have to be present as absent
    operands rather than dropped.
    """
    lhs, rhs = node.args[0], node.args[1]
    _require_fp16(node, "mm")
    lhs_val, rhs_val = lhs.meta["val"], rhs.meta["val"]
    if not (lhs_val.is_contiguous() and rhs_val.is_contiguous()):
        raise RuntimeError("hexagon: mm operands must be contiguous; strides come from shape")
    m, k = lhs_val.shape
    contracted, n = rhs_val.shape
    if k != contracted:
        raise RuntimeError(f"hexagon: mm contracts {k} against {contracted}")

    out = ctx.result_for(node, m * n)
    ctx.builder.add_op(
        Op(
            type=DSP_OP_BATCH_MATMUL,
            inputs=[ctx.operand(lhs), ctx.operand(rhs), ABSENT, ABSENT, ABSENT],
            outputs=[out],
            params=[FP16_BYTES]
            + _loop_param(
                1,
                (m, k, n),
                (n * FP16_BYTES, 0, FP16_BYTES),
                (k * FP16_BYTES, FP16_BYTES, 0),
                (0, n * FP16_BYTES, FP16_BYTES),
                m * n,
                m * k,
                k * n,
            ),
        )
    )
    return ctx.record(node, out)


def _emit_mean_dim(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.mean.dim as a single REDUCTION.

    The DSP collapses one contiguous (outside, reduce, inside) span, so the
    reduced dims have to be adjacent; the caller's support check enforces that.
    """
    src = node.args[0]
    _require_fp16(node, "mean")
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
    _require_fp16(node, "softmax input")
    dim = int(node.args[1])
    shape = list(node.meta["val"].shape)

    if dim < 0:
        dim += len(shape)
    outside = 1
    for size in shape[:dim]:
        outside *= int(size)
    channel = int(shape[dim])
    inside = 1
    for size in shape[dim + 1 :]:
        inside *= int(size)

    numel = _numel(node)
    out = ctx.result_for(node, numel)
    ctx.builder.add_op(
        Op(
            type=DSP_OP_SOFTMAX,
            inputs=[ctx.operand(src)],
            outputs=[out],
            # The DSP reduces the middle axis of an [outside][channel][inside]
            # view, so the reduction dim is described by its strides.
            params=[outside, channel, inside, FP16_BYTES],
        )
    )
    return ctx.record(node, out)


def _emit_rms_norm(node: torch.fx.Node, ctx) -> TensorRef:
    """One command for the whole norm, which is what the DSP kernel wants.

    The kernel reads fp16 activations and fp32 gamma and accumulates in fp32,
    which is the arithmetic norm.py asks for. beta is ABSENT rather than a
    zero-size tensor: RMSNorm has no bias, and a zero-size operand still maps to
    a live address the kernel would read as data.
    """
    source, eps = node.args
    _require_fp16(node, "rms_norm input")
    inner = int(node.meta["val"].shape[-1])
    out = ctx.result_for(node, _numel(node))
    ctx.builder.add_op(
        Op(
            type=DSP_OP_LAYER_NORM,
            # gamma and beta are both ABSENT: the scale is a separate multiply,
            # and the kernel skips the affine step when gamma is null.
            inputs=[ctx.operand(source), ABSENT, ABSENT],
            outputs=[out],
            params=[_numel(node) // inner, inner, _float_bits(float(eps)), 1],
        )
    )
    return ctx.record(node, out)


def _emit_layer_norm(node: torch.fx.Node, ctx) -> TensorRef:
    src = node.args[0]
    normalized_shape = node.args[1]
    weight = node.args[2] if len(node.args) > 2 else None
    bias = node.args[3] if len(node.args) > 3 else None
    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5

    _require_fp16(node, "layer_norm input")
    shape = list(node.meta["val"].shape)
    inner = 1
    for size in normalized_shape:
        inner *= int(size)
    outer = 1
    for size in shape[: len(shape) - len(list(normalized_shape))]:
        outer *= int(size)

    out = ctx.result_for(node, _numel(node))
    ctx.builder.add_op(
        Op(
            type=DSP_OP_LAYER_NORM,
            # Order matters: dst is the DSP's mapped_ptrs[3], so this must be
            # exactly src, gamma, beta.
            inputs=[
                ctx.operand(src),
                ctx.constant(weight, dtype=torch.float32) if weight is not None else ABSENT,
                ctx.constant(bias, dtype=torch.float32) if bias is not None else ABSENT,
            ],
            outputs=[out],
            params=[outer, inner, _float_bits(eps), 0],
        )
    )
    return ctx.record(node, out)


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
    """The scratch FLASH_ATTN needs, for the largest worker count it could pick.

    sync_attention_head_workspace_bytes is a scores buffer and a probability
    buffer, each qo_len rows of the sequence length rounded up to 32 and both
    128-aligned, and the kernel takes one per worker. The worker count comes
    from g_max_num_workers on the DSP, which the host cannot know, so this asks
    for what the widest count could take: one slot per head.
    """
    padded = (seq_len + 31) // 32 * 32
    scores = (qo_len * padded * 4 + 127) // 128 * 128
    probabilities = (qo_len * padded * 2 + 127) // 128 * 128
    return (scores + probabilities) * n_slots


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
    # Unlike the other ops, attention accepts fp32: the runtime narrows those
    # operands to fp16 as they enter the arena, so the DSP still sees fp16.
    _dtype = node.meta["val"].dtype
    if _dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"hexagon: sdpa input must be fp16 or fp32, got {_dtype}")

    mask = args[4] if len(args) > 4 else None
    scale = args[7] if len(args) > 7 else None
    q_shape = query.meta["val"].shape
    kv_shape = key.meta["val"].shape

    inputs = [
        ctx.operand(args[0]),
        ctx.operand(args[1]),
        ctx.operand(args[2]),
        ABSENT if mask is None else ctx.operand(mask),
        ctx.operand(args[1]),
        ctx.operand(args[2]),
    ]
    # The position tensor is not an operand of this op, so it rides along as an
    # extra input the kernel never reads. See STATUS.md.
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
    workspace = ctx.builder.add_activation(
        _attention_workspace_bytes(q_shape[1], q_shape[1] + kv_shape[1], q_shape[2])
    )

    ctx.builder.add_op(
        Op(
            type=DSP_OP_FLASH_ATTN,
            inputs=inputs,
            outputs=[out, workspace],
            params=[
                q_shape[1],  # qo_len
                position,  # seq_current
                q_shape[1],  # seq_add
                q_shape[2],  # n_heads
                kv_shape[2],  # n_kv_heads
                q_shape[3],  # head_dim
                _float_bits(
                    scale if isinstance(scale, (int, float)) else q_shape[3] ** -0.5
                ),
                -1,  # mask_stride
                kv_shape[1],  # max_kv_len
                0,  # page_count: the paged paths need fds, which only exist at load
                0,  # page_size
                0,  # value_c4
            ],
            patch=patch,
        )
    )
    return ctx.record(node, out)


EMITTERS = {
    exir_ops.edge.aten.abs.default: _unary("abs"),
    exir_ops.edge.aten.neg.default: _unary("neg"),
    exir_ops.edge.aten.gelu.default: _unary("gelu"),
    exir_ops.edge.aten.sigmoid.default: _unary("sigmoid"),
    exir_ops.edge.aten.exp.default: _unary("exp"),
    exir_ops.edge.aten.log.default: _unary("log"),
    exir_ops.edge.aten.silu.default: _unary("silu"),
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
    exir_ops.edge.aten.layer_norm.default: _emit_layer_norm,
    exir_ops.edge.aten.mm.default: _emit_mm,
    exir_ops.edge.aten.mean.dim: _emit_mean_dim,
    exir_ops.edge.aten.alias_copy.default: _emit_alias,
    exir_ops.edge.aten.unsqueeze_copy.default: _emit_alias,
    exir_ops.edge.aten.view_copy.default: _emit_alias,
    exir_ops.edge.aten.select_copy.int: _emit_select_copy,
    exir_ops.edge.aten._to_copy.default: _emit_alias,
    exir_ops.edge.aten.to.dtype: _emit_alias,
    exir_ops.edge.aten.slice_copy.Tensor: _emit_slice_copy,
    exir_ops.edge.aten.cat.default: _emit_cat,
    exir_ops.edge.aten.permute_copy.default: _emit_permute_copy,
    UPDATE_CACHE: _emit_update_cache,
    RMS_NORM: _emit_rms_norm,
    MUL_SILU: _binary("mul_silu"),
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
        exir_ops.edge.aten.view_copy.default,
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
