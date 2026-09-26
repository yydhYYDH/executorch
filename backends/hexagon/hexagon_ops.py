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
import re
import math
import struct
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

# After rms_norm, which opens the et_hexagon namespace these fragments join.
from executorch.backends.hexagon.add_relu import ADD_RELU
from executorch.backends.hexagon.add_rms_norm import ADD_RMS_NORM
from executorch.backends.hexagon.cumsum import CUMSUM, cumsum_is_emittable
from executorch.backends.hexagon.kv_cache import UPDATE_CACHE
from executorch.backends.hexagon.mul_silu import MUL_SILU
from executorch.backends.hexagon.prelu import PRELU
from executorch.backends.hexagon.reflect_pad import REFLECT_PAD
from executorch.backends.hexagon.rms_norm import RMS_NORM
from executorch.backends.hexagon.rope import ROPE
from executorch.backends.hexagon.row_guard import ROW_GUARD
from executorch.backends.hexagon.serialization.blob import ABSENT, Op, TensorRef
from executorch.backends.hexagon.vision_attention import VISION_ATTENTION
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.sym_util import eval_upper_bound

# DSPOpType, from third-party/mnn-htp-ops/include/htp_command.h.
DSP_OP_POOL2D_FP16 = 1
DSP_OP_CONV_DEPTHWISE2D_FP16 = 2
DSP_OP_RASTER_BLIT = 3
# The im2col convolution. DSP_OP_CONV1X1_DIRECT_FP16 (17) resolves to the
# same function (htp_ops_conv1x1_direct_fp16 is a second name for
# hmx_im2col_convolution_fp16, im2col_convolution_fp16.cc:1840), so the
# difference between the two commands is not which kernel runs but which of
# the function's own activation fills is eligible, and that is decided by the
# parameters the command carries. conv_1x1_direct_applies() below is the host
# transcription of the fill selection the C makes (fill_im2col_activation_
# tiles, :1132): a 1x1 convolution only takes a direct fill when there is one.
DSP_OP_IM2COL_CONVOLUTION_FP16 = 12
DSP_OP_CONV1X1_DIRECT_FP16 = 17
DSP_OP_UNARY = 4
# A memset over one operand, which is the only way to clear the padding lanes a
# ragged channel count leaves in a blocked activation (blit_ops.cc:1724).
DSP_OP_ZERO = 24
# The one kernel that answers two outputs: htp_ops_topkv2_k1_fp16 writes a
# maximum and its position per row (topk_ops.cc:48).
DSP_OP_TOPKV2_K1_FP16 = 27
# Row-wise argmax/argmin. The command carries a mode in its third parameter:
# zero selects max and one selects min. Its output is one int64 per row, which
# is the width ATen declares for an index result; an int32 slot cannot be
# attached to an int64 result, because the upper four bytes of every element
# would otherwise stay stale, so the kernel writes the 64-bit width directly
# rather than a 32-bit one widened by a second command.
DSP_OP_ARGMAX_FP16 = 47

# One command's parameters live in a fixed-size block of 40 ints
# (serialization/hexagon_schema.h:44). The raster blit spends three of them on its
# header and twelve per region, which is what caps how many 64-channel blocks one
# command can move.
MAX_OP_PARAMS = 40
BLIT_REGION_INTS = 12
BLIT_BLOCKS_PER_COMMAND = (MAX_OP_PARAMS - 3) // BLIT_REGION_INTS
DSP_OP_LAYER_NORM = 8
DSP_OP_ROPE = 14
DSP_OP_ADD_FUSE_LAYERNORM = 16
DSP_OP_BINARY_ELEMENTWISE = 19
DSP_OP_SOFTMAX = 28
DSP_OP_REDUCTION = 29
DSP_OP_BATCH_MATMUL = 38
DSP_OP_RELU = 37
DSP_OP_PRELU = 39
DSP_OP_FLASH_ATTN = 18
# The quantized matmuls, from the DSP's enum (htp_command.h pins both GEMV
# numbers with a static_assert). They take an fp16 activation and a packed
# low-bit weight. The GEMV entries are the M == 1 (decode) kernels -- they read K
# contiguous fp16, which for one row is the row-major layout the graph already
# holds, and write the output linear by output channel, so neither the 64-channel
# activation blocking nor the output repack the prefill entry needs is involved.
# The q4a16 GEMV additionally quantizes the activation to symmetric per-token
# int8 inside the kernel, because its multiply is the integer vrmpy one; the
# prefill entry below multiplies fp16 by the dequantized fp16 weight instead.
#
# DSP_OP_MATMUL_Q4A16_FP16 is dispatch 22 of execute_command.cc. It forwards to
# hmx_matmulq4fp16_mle32 for M <= 32 and to hmx_matmulq4fp16 above that; both read
# the same weight, and both are reached through this one command. It carries no
# static_assert of its own, so `test_the_op_ids_are_the_ones_the_dsp_defines`
# pins the number against htp_command.h.
DSP_OP_MATMUL_Q4A16_FP16 = 22
DSP_OP_MATMUL_Q4A16_GEMV_I8 = 41
DSP_OP_MATMUL_W8A16_BLOCK_FP16 = 42
DSP_OP_MATMUL_W8A16_GEMV_I8 = 45

# The prefill command's own knobs, and the VTCM budget they are spent out of.
# mp and np are the chunk sizes the kernel stages: np counts output-channel
# tiles held in VTCM at once, mp counts 32-row activation bands, and both buy
# traffic at the cost of the bands themselves. The prefill path passes one
# activation band and two output-channel tiles -- the pair the kernel's own
# store walks two at a time -- and PREFILL_VTCM_FIXED covers the small
# allocations around them (output tile buffers, scales, HMX column scales).
PREFILL_OUTPUT_CHANNEL_CHUNK = 2
PREFILL_VTCM_FIXED = 64 * 1024
#: The M the dispatcher switches prefill kernels on (`matmul_ops.cc:28`). At or
#: below it, command 22 selects `hmx_matmulq4fp16_mle32`; the baseline DSP source
#: now heap-allocates that kernel's per-tile descriptors. The host ceiling below
#: remains conservative because a stale skel or a rebuild from another base can
#: still run the former stack allocation; above M=32 only the VTCM bound applies.
PREFILL_M32_MAX_M = 32
#: The conservative host ceiling for the Q4 command-22 branch at M <= 32.
#: The original phone measurement found K=12672 pass and K=12736 abort with
#: `execute_command_group failed: 0x8000040d` and no output, when the DSP used
#: the former stack-backed descriptor array. The baseline source now allocates
#: those descriptors on the heap, and a later OnePlus 13 A/B at K=25216 passed
#: for M=2, 4, and 32. This host refusal nevertheless remains: deployment does
#: not encode source provenance, so a stale skel can still carry the VLA. The
#: constant is not a claim that K=12672 is the largest shape the heap kernel can
#: compute; widening it requires a matched rebuild and a broader measured matrix.
PREFILL_M32_MAX_K = 12672
#: The VTCM the prefill kernels may reserve. The simulator reports 8 MiB
#: (`vtcm_manager_get_vtcm_size`) and the kernels' own guard is 8 MiB less
#: 16 KiB (`matmul_q4fp16.c:946`); this asks for less than either, because what
#: a device hands the skel is not something this tree can measure.
PREFILL_VTCM_BYTES = 7 * 1024 * 1024
# The row gather, from the same enum: one command reads selectSize rows out of
# an fp16 table the export step laid out as 32x32 tiles.
DSP_OP_SHARED_GATHER = 23
# The vision tower's attention, from the same enum. Unmasked and non-causal: it
# reads query, key and value as [batch, tokens, heads, headDim] -- heads inside
# a token's row -- and writes the result in that layout, which is why the fused
# op it serves carries the operands under the head transposes rather than the
# head-major tensors a matmul wants.
DSP_OP_VISION_ATTENTION_FP16 = 43
# The element-wise select, `cond ? a : b`. Its condition operand is the one
# operand in this backend that is not two bytes wide: htp_ops_select takes a
# condBytes param and reads the condition bytewise when it is 1
# (eltwise_ops.cc:2116-2125), which is exactly the width torch.bool has in the
# `.pte`. The runtime copies an input at the size the blob declares for its slot,
# so a bool operand reaches the arena as one byte per element and this command is
# the only one that asks for it that way.
DSP_OP_SELECT = 26

# HtpOpsReductionType, from the DSP's eltwise_ops.cc.
REDUCTION_SUM = 1
REDUCTION_MAXIMUM = 2
REDUCTION_MEAN = 3
REDUCTION_MINIMUM = 4

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

# The two params the int4 and int8 table paths read past isInt4 (execute_command.cc:809).
# The dispatcher supplies 1 and 0 for a command that stops before them, which is
# what a plain fp16 table wants and what this backend writes explicitly so the
# index width below can have a slot of its own.
SHARED_GATHER_SCALE_BLOCK_NUM = 1
SHARED_GATHER_SCALE_ASYMMETRIC = 0

# Params[7] of the same command: the width one index has in the caller's tensor.
# Nothing on the DSP reads it -- the index slot is four bytes an element either
# way, and the kernel reads int32 -- so it is the host's: the runtime and the
# host interpreter narrow a wider tensor into the slot with it.
SHARED_GATHER_INDEX_BYTES = 7

# Params[8], and what the op this command came from does with a negative index.
# `embedding` and `index_select` raise on one -- measured on torch 2.14, whose
# answer is "index out of range in self" for both -- while advanced indexing
# counts back from the last row, so `table[-1]` is the last row and `table[-5]`
# in a table of four rows is a range error. A narrowing that picked one of those
# for the other would answer where torch refuses, so the command carries which
# one it is. The DSP reads no param past scaleAsymmetric.
SHARED_GATHER_NEGATIVE_INDEXES = 8
NEGATIVE_INDEX_REFUSED = 0
NEGATIVE_INDEX_FROM_END = 1

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

# What the topk kernel writes its positions as, whatever the graph says they are.
INT32_BYTES = 4

# A torch.bool is one byte per element in the `.pte`, and the arena holds what
# the blob declares for an input's slot rather than a fixed width. SELECT is the
# one command here that names this width; every other kernel reads two bytes.
BOOL_BYTES = 1

#: The element-wise select. The library's only ATen node for it.
WHERE = exir_ops.edge.aten.where.self

#: A clamp whose bounds are tensors rather than numbers, and the exponential
#: linear unit. Both are compositions of commands the table above already emits,
#: which is why they cost no kernel; the docstrings on the emitters say which.
CLAMP_TENSOR = exir_ops.edge.aten.clamp.Tensor
ELU = exir_ops.edge.aten.elu.default
# The two order comparisons the DSP's own binary op types answer. The Scalar
# overloads are not here: the measured route is two tensors, and a python
# literal would reach the same binary command as an fp16 constant, which is a
# second thing to have measured.
GREATER_THAN = exir_ops.edge.aten.gt.Tensor
LESS_THAN = exir_ops.edge.aten.lt.Tensor


LAYER_NORM = exir_ops.edge.aten.layer_norm.default
NATIVE_LAYER_NORM = exir_ops.edge.aten.native_layer_norm.default
# GroupNorm reaches the backend as native_group_norm, whose normalized span is
# one group's channels times the spatial product: the trailing dims of an
# [N*group][(C/group)*HxW] view of the input, which is the layer norm kernel's
# [outer][inner] view with outer = N*group.
NATIVE_GROUP_NORM = exir_ops.edge.aten.native_group_norm.default
# InstanceNorm reaches it as the batch norm without running statistics, over a
# [1, N*C, *spatial] view. The batch axis the exporter made one is what turns
# per-channel statistics into the per-(n, c) ones the op is named for.
BATCH_NORM_NO_STATS = exir_ops.edge.aten._native_batch_norm_legit.no_stats
LOG_SOFTMAX = exir_ops.edge.aten._log_softmax.default
GETITEM = operator.getitem
SOFTMAX_TARGETS = frozenset({exir_ops.edge.aten._softmax.default})
LOG_SOFTMAX_TARGETS = frozenset({LOG_SOFTMAX})
GROUP_NORM_TARGETS = frozenset({NATIVE_GROUP_NORM})
BATCH_NORM_TARGETS = frozenset({BATCH_NORM_NO_STATS})

# The longest reduced span a log_softmax may carry. The shift makes every
# exponential at most one, so the sum of them is at most the span's length, and
# the kernel stores that sum as fp16; past 65504 it saturates and the log of it
# is an infinity rather than a large negative number.
LOG_SOFTMAX_MAX_SPAN = 65504

# The row width one HVX vector holds, and the width at which the standalone
# softmax kernel stops agreeing with torch. The kernel walks a row in chunks of
# this many fp16 lanes and then finishes it with a tail it copies in and masks;
# the tail is within rounding of torch and the chunked part is not (see
# `_emit_softmax`), so a row shorter than one vector -- which is all tail -- is
# the only width the command is used at.
SOFTMAX_VECTOR_WIDTH = 64


def softmax_reduces_the_last_axis(node: torch.fx.Node) -> bool:
    """Whether softmax already reduces the last axis of its input."""
    if len(node.args) < 2:
        return False
    return int(node.args[1]) in (-1, node.meta["val"].dim() - 1)


def _softmax_permutation(node: torch.fx.Node):
    """Return the axis order and inverse used to put the reduction last."""
    value = _value_of(node)
    shape = list(value.shape)
    dim = int(node.args[1]) % len(shape)
    order = [axis for axis in range(len(shape)) if axis != dim] + [dim]
    inverse = [0] * len(shape)
    for position, axis in enumerate(order):
        inverse[axis] = position
    permuted_shape = [shape[axis] for axis in order]
    return dim, order, inverse, permuted_shape


def softmax_reduces_the_inner_axis(node: torch.fx.Node) -> bool:
    """Whether the DSP can express this softmax after a bounded reshape.

    A last-axis softmax already has the layout the kernel reads. A reduction
    over another axis is safe only when its source is contiguous, the axis can
    be moved last and back with one raster region each way, and the reduced
    span is below the measured standalone-command width. In particular, a
    non-unit-stride view is not silently materialized by this path.
    """
    if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
        return False
    result = _value_of(node)
    if not isinstance(result, torch.Tensor):
        return False
    raw_dim = int(node.args[1])
    shape = list(result.shape)
    if raw_dim < -len(shape) or raw_dim >= len(shape):
        return False
    dim = raw_dim % len(shape)
    if dim == len(shape) - 1:
        return True
    source = node.args[0].meta.get("val")
    if not isinstance(source, torch.Tensor):
        return False
    if source.dtype not in (torch.float16, torch.float32):
        return False
    if result.dtype not in (torch.float16, torch.float32):
        return False
    if len(shape) < 2 or len(source.shape) != len(shape):
        return False
    if source.numel() != result.numel():
        return False
    if not source.is_contiguous() or not result.is_contiguous():
        return False
    if any(not isinstance(extent, int) for extent in shape):
        return False
    if tuple(source.stride()) != tuple(_row_major_strides(shape)):
        return False
    if tuple(result.stride()) != tuple(_row_major_strides(shape)):
        return False
    if shape[dim] >= SOFTMAX_VECTOR_WIDTH:
        return False
    _, order, inverse, permuted_shape = _softmax_permutation(node)
    forward = _permute_region_from_shapes(shape, permuted_shape, order)
    backward = _permute_region_from_shapes(permuted_shape, shape, inverse)
    return forward is not None and backward is not None


def _permute_region_from_shapes(source_shape, result_shape, dims):
    """Build the three-level blit region for a concrete axis permutation."""
    if not isinstance(dims, (list, tuple)):
        return None
    rank = len(source_shape)
    if rank < 2 or len(dims) != rank or sorted(dims) != list(range(rank)):
        return None
    if len(result_shape) != rank:
        return None
    source_strides = _row_major_strides(source_shape)
    result_strides = _row_major_strides(result_shape)
    positions = [0] * rank
    for position, axis in enumerate(dims):
        positions[axis] = position
    destinations = [result_strides[position] for position in positions]

    groups = []
    first = 0
    for axis in range(1, rank):
        if positions[axis] != positions[first] + (axis - first):
            groups.append((first, axis - 1))
            first = axis
    groups.append((first, rank - 1))

    levels = []
    for start, last in groups:
        run = 1
        for axis in range(start, last + 1):
            run *= int(source_shape[axis])
        if run > 1:
            levels.append((run, source_strides[last], destinations[last]))
    if len(levels) > 3:
        return None
    while len(levels) < 3:
        levels.append((1, 0, 0))
    for index, level in enumerate(levels):
        if level[1] == 1 and level[2] == 1 and index != 2:
            levels[index] = levels[2]
            levels[2] = level
            break
    size = [level[0] for level in levels]
    src = [level[1] for level in levels]
    dst = [level[2] for level in levels]
    return [0, 0, 0] + size + src + dst


def log_softmax_shifts_within_the_arena(node: torch.fx.Node) -> bool:
    """Whether this log_softmax's shifted sum is one the kernel can store.

    The composition below is the log-sum-exp the softmax kernel computes
    internally, written out so the answer is not a second rounding of a
    probability: with the span's maximum subtracted, every exponential is at
    most one and their sum is at most the span's length. The kernel stores that
    sum as fp16, so a span longer than 65504 is a saturated sum, a log of
    infinity and a row of infinities where torch has finite values. The span is
    the last axis (see softmax_reduces_the_inner_axis), whose extent the export
    knows whether or not it is static.
    """
    if not softmax_reduces_the_last_axis(node):
        return False
    shape = list(node.meta["val"].shape)
    return _upper_product([shape[int(node.args[1]) % len(shape)]]) <= (
        LOG_SOFTMAX_MAX_SPAN
    )


def _value_of(node: torch.fx.Node) -> torch.Tensor:
    """A node's value, or its first when the node hands out several.

    native_layer_norm returns (out, mean, rstd), so the node's own value is the
    tuple; the tensor the kernel reads is its first element. A split hands its
    pieces out in a list rather than a tuple, and is the same case.
    """
    value = node.meta["val"]
    return value[0] if isinstance(value, (list, tuple)) else value


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


#: A bare dynamic symbol, as `torch.SymInt` prints one: `s53`. A dim derived from
#: that symbol prints as an expression (`(((s53 - 1)//2)) + 1`), which is the tell
#: that the extent would have to be divided rather than scaled.
_RUNTIME_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
#: A dim the export has already resolved to one number. The graph a backend is
#: handed for a delegate has had every symbol folded to the traced example, so
#: this is what a moved dim looks like from inside an emitter.
_SPECIALIZED = re.compile(r"\d+\Z")

#: The three things a shape dim can be here, and what each one means.
STATIC_DIM = "static"
RUNTIME_DIM = "runtime"
SPECIALIZED_DIM = "specialized"
DERIVED_DIM = "derived"


def extent_kind(value) -> str:
    """Which of the four a shape dim is, read off how it prints.

    A static dim is an int. The run-time length is the bare symbol, and reads as
    one wherever the graph still carries the symbol. A delegate's subgraph
    instead carries the symbol already resolved to the traced example, which is
    a number that is still a `torch.SymInt` -- the only way to tell a dim that
    moves, from an emitter, once the expression is gone. Anything left is an
    expression over the symbol, whose extent no affine patch can rebuild.
    """
    if not isinstance(value, torch.SymInt):
        return STATIC_DIM
    text = str(value)
    if _RUNTIME_SYMBOL.match(text):
        return RUNTIME_DIM
    if _SPECIALIZED.match(text):
        return SPECIALIZED_DIM
    return DERIVED_DIM


def _is_runtime_symbol(value) -> bool:
    """Whether a shape dim is the bare symbol the runtime hands a length for."""
    return extent_kind(value) == RUNTIME_DIM


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


def result_dtype_is_emittable(dtype: torch.dtype, target) -> bool:
    """Whether a result of this width and this op is one the arena holds.

    fp16 and fp32 both reach the same fp16 command -- the runtime narrows a fp32
    operand and widens a fp32 result at the boundary -- and a torch.bool is a
    third width, held by one command: SELECT's one-byte arm copies a byte at a
    time, and a comparison is the only node here that produces one. The exemption
    is keyed on the target rather than on the width, so a bool arriving anywhere
    else is still refused.

    Both gates read this one definition. The partitioner asks it before it
    delegates and the emitters raise on the same answer, so a node the support
    check admitted cannot fail the export for want of a width.
    """
    if dtype is torch.bool:
        return target in COMPARISON_TARGETS
    return dtype in (torch.float16, torch.float32)


def _require_arena_dtype(node: torch.fx.Node, what: str) -> None:
    """The widths the arena holds, both of which reach the same fp16 command.

    Every kernel reads and writes two bytes per element, and the runtime narrows
    a fp32 operand and widens a fp32 result at the boundary, so a node the graph
    declares fp32 is emitted exactly as its fp16 twin. Any other width would have
    the kernels reading those bits as half floats.
    """
    dtype = _value_of(node).dtype
    if not result_dtype_is_emittable(dtype, node.target):
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


def expand_region(node: torch.fx.Node) -> list[int] | None:
    """The blit region for an `expand_copy` that repeats, or None.

    An expand that grows nothing is a view, and then the alias path re-points
    the operand because the result describes the operand's own bytes. An
    expand that grows an axis is a broadcast: the result holds more elements
    than the operand, so the bytes have to be written out, and the only
    question left is whether a three-level region can address them.

    It can exactly when the shape's per-axis plain/broadcast pattern has at
    most three runs of equal values. A run of axes is one level: a plain run
    advances the source by the pitch of the axis that ends it, and a
    broadcast run holds the source still and advances the result instead,
    which is the per-side stride doing the work the alias path cannot. An
    expand only ever turns a one into a larger extent, so the result is a
    dense buffer and its own strides are the ones the region walks; the
    innermost level is therefore always destination stride one.

    A fourth run is the refusal, and it is a descriptor with three levels
    rather than a kernel that is missing: an axis that is broadcast, then one
    that is not, then one that is, needs four levels of walk.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return None
    shape = list(source_value.shape)
    result = list(result_value.shape)
    if len(shape) != len(result) or not source_value.is_contiguous():
        return None
    if any(grown < plain or grown % plain for plain, grown in zip(shape, result)):
        return None
    if source_value.numel() == result_value.numel():
        return None

    broadcasts = [grown > plain for plain, grown in zip(shape, result)]
    boundaries = [0]
    for axis in range(1, len(broadcasts)):
        if broadcasts[axis] != broadcasts[axis - 1]:
            boundaries.append(axis)
    boundaries.append(len(broadcasts))
    if len(boundaries) - 1 > 3:
        return None

    sizes, source_strides, result_strides = [], [], []
    for start, stop in zip(boundaries, boundaries[1:]):
        count = 1
        for extent in result[start:stop]:
            count *= extent
        source_pitch = 1
        for extent in shape[stop:]:
            source_pitch *= extent
        result_pitch = 1
        for extent in result[stop:]:
            result_pitch *= extent
        sizes.append(count)
        source_strides.append(0 if broadcasts[start] else source_pitch)
        result_strides.append(result_pitch)
    # The last run is the innermost level, because the region's third level
    # walks the result's own innermost axis, so a run short of three leaves
    # its unused levels on the outside where a neutral one cannot shorten the
    # walk.
    while len(sizes) < 3:
        sizes.insert(0, 1)
        source_strides.insert(0, 0)
        result_strides.insert(0, 0)

    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
    return [0, 0, 0] + sizes + source_strides + result_strides


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


def _emit_expand_copy(node: torch.fx.Node, ctx) -> TensorRef:
    """A broadcast expand writes a buffer; an expand that grows nothing does not.

    The same two forms as the narrowing select, decided here rather than by
    splitting the target in the op table.
    """
    region = expand_region(node)
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


# torch's split, the one multi-output op here whose every result a command
# produces. It reaches this backend in the form `to_edge` leaves both
# `aten.split.Tensor` and `aten.chunk` in: `split_with_sizes_copy`, whose results
# the graph reads one getitem at a time. Each piece is the narrowing slice the
# blit already runs, so the command list is a run per piece and the only thing
# missing was a producer for the getitems -- the reason `aten.sort` still has
# none is that no command here writes a sorted run.
#
# `split_copy.Tensor` is the same read with the piece size rather than the piece
# list, where torch is free to cut the last piece short; a graph reaches it by
# naming the functional op itself, since `torch.split` decomposes to the other
# one. Both are read through the same spec, so the two spellings cannot disagree
# about which slice a piece covers.
SPLIT_WITH_SIZES_COPY = exir_ops.edge.aten.split_with_sizes_copy.default
SPLIT_COPY = exir_ops.edge.aten.split_copy.Tensor

SPLIT_TARGETS = frozenset({SPLIT_WITH_SIZES_COPY, SPLIT_COPY})


class SplitSpec(NamedTuple):
    """Everything a split's blits need: one run per piece, and its readers.

    `start` and `extent` are counted in elements of the split axis, and the two
    numbers a region needs are those times `inner`. `axis_extent` is the whole
    axis, which is the row stride of the source. A piece the graph never reads
    keeps its place in the list with no reader, because the offsets of the pieces
    after it are what its extent is still needed for.
    """

    source: torch.fx.Node
    dim: int
    rows: int
    inner: int
    axis_extent: int
    #: (start, extent, readers) per piece, in the order the outputs come out.
    pieces: Tuple[Tuple[int, int, Tuple[torch.fx.Node, ...]], ...]


def split_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The split a getitem reads a piece of, whatever the piece is."""
    if node.target is not GETITEM or len(node.args) != 2:
        return None
    source = node.args[0]
    if not isinstance(source, torch.fx.Node):
        return None
    return source if source.target in SPLIT_TARGETS else None


def split_spec(node: torch.fx.Node) -> Optional[SplitSpec]:
    """The pieces this split's results cover, or None when it is not one.

    Every gate here is a number a command has no argument for. The piece extents
    and each piece's offset are baked into the region list when the command is
    built, so a symbolic extent would describe the shape the graph was exported
    at rather than the run, and the rows count strides between them, so a
    symbolic one is the same problem. A piece list that does not add up to the
    axis, an axis the source does not have, a source that is not the contiguous
    layout these strides describe, a third width the arena does not hold, and a
    result shape that disagrees with the piece it is supposed to hold all leave
    the node on a portable kernel rather than reach a blit that reads it wrong.

    The reader rule is the same one topk is under: every reader has to be a
    getitem of a piece this spec names. A split whose tuple is handed on whole
    has no command form at all, so it keeps the whole node portable rather than
    leave a reader holding a tuple no kernel can be handed. A piece nothing reads
    is not that case: its extent is still what the pieces after it are offset by,
    and only its own blit is missing.
    """
    if node.target not in SPLIT_TARGETS:
        return None
    dim = _node_arg(node, "dim", 2, 0)
    if isinstance(dim, bool) or not isinstance(dim, int):
        return None
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    value = source.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dim() == 0:
        return None
    if not value.is_contiguous():
        return None
    if value.dtype not in (torch.float16, torch.float32):
        return None
    shape = list(value.shape)
    if not all(isinstance(extent, int) and extent > 0 for extent in shape):
        return None
    if dim < 0:
        dim += len(shape)
    if not 0 <= dim < len(shape):
        return None

    sizes = _split_sizes(node, shape[dim])
    if sizes is None:
        return None
    declared = node.meta.get("val")
    if not isinstance(declared, (list, tuple)) or len(declared) != len(sizes):
        return None

    inner = 1
    for extent in shape[dim + 1 :]:
        inner *= extent
    rows = 1
    for extent in shape[:dim]:
        rows *= extent

    pieces = []
    start = 0
    for index, size in enumerate(sizes):
        piece = declared[index]
        if not isinstance(piece, torch.Tensor):
            return None
        expected = list(shape)
        expected[dim] = size
        if tuple(piece.shape) != tuple(expected):
            return None
        pieces.append((start, size, _split_piece_readers(node, index)))
        start += size
    if not any(readers for _, _, readers in pieces):
        return None
    return SplitSpec(source, dim, rows, inner, shape[dim], tuple(pieces))


def _split_sizes(node: torch.fx.Node, axis_extent: int) -> Optional[Tuple[int, ...]]:
    """The extent of each result, from whichever argument states it.

    `split_with_sizes_copy` carries the list itself and only has to add up; the
    other spelling carries one size and lets torch cut the last piece short,
    which is the rule written out here.
    """
    if node.target is SPLIT_WITH_SIZES_COPY:
        sizes = node.args[1] if len(node.args) > 1 else None
        if not isinstance(sizes, (list, tuple)) or not sizes:
            return None
        if not all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in sizes
        ):
            return None
        if sum(sizes) != axis_extent:
            return None
        return tuple(sizes)
    size = _node_arg(node, "split_size", 1, None)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return None
    count = -(-axis_extent // size)
    return (size,) * (count - 1) + (axis_extent - size * (count - 1),)


def _split_piece_readers(node: torch.fx.Node, index: int) -> Tuple[torch.fx.Node, ...]:
    """Every getitem that reads this piece, which is usually none or one."""
    return tuple(
        user
        for user in node.users
        if user.target is GETITEM
        and len(user.args) == 2
        and user.args[0] is node
        and not isinstance(user.args[1], bool)
        and user.args[1] == index
    )


def split_is_emittable(node: torch.fx.Node) -> bool:
    """Whether this split is one the blit can write piece by piece.

    Every user of the node is checked here rather than only the pieces with a
    reader, because a user that is not a getitem of a named piece is one the
    emitter has no buffer for: `split_with_sizes_copy` hands out a tuple, and a
    consumer of the tuple itself is not its first element the way a layer norm's
    reader is.
    """
    spec = split_spec(node)
    if spec is None:
        return False
    return all(
        user.target is GETITEM
        and len(user.args) == 2
        and user.args[0] is node
        and isinstance(user.args[1], int)
        and not isinstance(user.args[1], bool)
        and 0 <= user.args[1] < len(spec.pieces)
        for user in node.users
    )


def _emit_split(node: torch.fx.Node, ctx) -> TensorRef:
    """One RASTER_BLIT per piece, each writing the run that piece covers.

    The region is the one a narrowing `slice_copy` of the same piece builds --
    same source offset, same row count, same run -- so a split of a tensor is
    exactly the set of slices it stands for, and the blit that writes a piece is
    byte for byte the blit `x[:, :n]` would have emitted for it.
    """
    spec = split_spec(node)
    if spec is None:
        raise RuntimeError("hexagon: this split is not one the blit can write")
    source = ctx.operand(spec.source)
    first = None
    for start, extent, readers in spec.pieces:
        run = extent * spec.inner
        for reader in readers:
            out = ctx.result_for(reader, run)
            ctx.emit(
                reader,
                Op(
                    type=DSP_OP_RASTER_BLIT,
                    inputs=[source],
                    outputs=[out],
                    # region count, element bytes, source count, then the region.
                    params=[
                        1,
                        FP16_BYTES,
                        1,
                        0,
                        start * spec.inner,
                        0,
                        1,
                        spec.rows,
                        run,
                        0,
                        spec.axis_extent * spec.inner,
                        1,
                        0,
                        run,
                        1,
                    ],
                ),
            )
            ctx.record(reader, out)
            if first is None:
                first = out
    return ctx.record(node, first)


# The region's source offset, counted from the start of the params vector: the
# blit header is three ints, then srcIndex, then srcOffset.
_SLICE_OFFSET_PARAM = 4

#: One blit command holds at most three regions, because its header is three
#: ints and each region twelve out of a forty-int parameter block. A longer
#: concatenation is several commands writing into disjoint slices of one
#: result rather than a refusal: a per-time-step sequence stitch is T pieces,
#: and T is whatever length the model was exported with.
MAX_CAT_INPUTS = BLIT_BLOCKS_PER_COMMAND


def cat_plan(node: torch.fx.Node):
    """One entry per blit command a concatenation needs, or None.

    Each input becomes one region writing into its own slice of the result. A
    RoPE rejoin needs two 64-element halves becoming one 128-element row; a
    per-time-step sequence stitch needs T per-step outputs becoming one
    [B, T, H] tensor. A command holds BLIT_BLOCKS_PER_COMMAND regions, so a
    T-way stitch is ceil(T / BLIT_BLOCKS_PER_COMMAND) commands whose outputs
    are disjoint slices of a single allocation. The region list is fixed when
    a command is built, so the split has to be known here: a concatenation
    whose lengths are only known at the call cannot be described by it.

    Every entry is (params, input_indices, byte_offset, byte_size): the
    parameter block for that command, which of the node's inputs it reads,
    and which slice of the result it writes.
    """
    if not node.args or len(node.args) > 2:
        return None
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else 0
    if not isinstance(tensors, (list, tuple)):
        return None
    if not isinstance(dim, int) or isinstance(dim, bool):
        return None
    if not tensors:
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
    if result.dtype not in (torch.float16, torch.float32) or any(
        value.dtype is not result.dtype for value in values
    ):
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

    # The prefix of the result each input covers, in elements along the
    # concatenated axis. A command's own dstOffsets restart at zero, because
    # its output ref already carries the byte offset of its own slice.
    starts = []
    walked = 0
    for value in values:
        starts.append(walked)
        walked += value.shape[dim]

    plan = []
    for base in range(0, len(values), BLIT_BLOCKS_PER_COMMAND):
        indices = list(range(base, min(base + BLIT_BLOCKS_PER_COMMAND, len(values))))
        params = [len(indices), FP16_BYTES, len(indices)]
        for position, index in enumerate(indices):
            run = values[index].shape[dim] * inner
            # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
            # srcIndex is this command's own input: the kernel indexes it into the
            # src_ptrs array it was handed, which holds this command's inputs and
            # not the node's whole list.
            params += [
                position,
                0,
                (starts[index] - starts[base]) * inner,
                1,
                rows,
                run,
                0,
                run,
                1,
                0,
                combined,
                1,
            ]
        plan.append(
            (
                params,
                indices,
                starts[base] * inner * FP16_BYTES,
                sum(values[index].shape[dim] for index in indices) * inner * FP16_BYTES,
            )
        )
    return plan


def cat_region(node: torch.fx.Node):
    """The parameters of a concatenation that fits in one command, or None.

    The single-command spelling of cat_plan, for the callers that want one
    parameter block: the RoPE rejoin and the split-piece cases.
    """
    plan = cat_plan(node)
    if plan is None or len(plan) != 1:
        return None
    return plan[0][0]


def _emit_cat(node: torch.fx.Node, ctx) -> TensorRef:
    """Each operand is copied into its own slice of a fresh buffer.

    A concatenation longer than one command's region budget is several
    commands over one allocation, each writing the slice of the result it
    owns. A ref is a section, an offset and a size, so a shifted ref is a
    shifted write: the runtime resolves every ref as section base plus
    ref.offset (hexagon_backend.cpp:849) and nothing downstream knows the
    result was filled in pieces.
    """
    tensors = node.args[0]
    plan = cat_plan(node)
    out = ctx.result_for(node, _numel(node))
    result_shape = _value_of(node).shape
    dim = node.args[1] if len(node.args) > 1 else 0
    if dim < 0:
        dim += len(result_shape)
    for params, indices, byte_offset, byte_size in plan:
        op_index = ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[ctx.operand(tensors[index]) for index in indices],
                outputs=[
                    out
                    if len(plan) == 1
                    else TensorRef(out.space, out.offset + byte_offset, byte_size, out.index)
                ],
                params=params,
            ),
        )
        # Every region repeats the row count at its own size1 slot, so each one
        # has to move with the length; the regions that mirror it are patched
        # with it.
        for region_index in range(params[0]):
            rows_param = 3 + region_index * BLIT_REGION_INTS + 4
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
    describable exactly when the axes split into at most three groups that
    advance, each a run of consecutive axes that keeps its order in both layouts:
    such a run is one loop whose size is the product of the run and whose strides
    are its last axis's. `permute(w, [1, 0])` between a weight and its `mm` is the
    two-group case, and it is the shape `htp_ops_prepare_transpose` recognises and
    routes to the HVX transpose -- the inner run reads a contiguous source row and
    writes a strided destination column. `permute(1, 0, 2, 3)` over a fused qkv
    is the same split with the head run carried whole, and on a batch of one the
    leading extent is a group that iterates once and spends no loop.

    A permutation that reverses the axes inside a group, or that needs a fourth
    loop, is refused rather than approximated: no single region describes it, and
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

    return _permute_region_from_shapes(
        source_value.shape, result_value.shape, dims
    )


def constant_pad_region(node: torch.fx.Node):
    """The blit region that places a padded operand inside its result, or None.

    A constant pad is two things this backend already emits for other ops: a
    memset over the result, which fills the border, and one region that copies
    the operand into the interior. The region is three nested loops with a stride
    per side, so what it describes is a copy whose strides are constant per level
    -- and a pad is exactly that when only the last two axes are padded. Those
    are the axes whose output pitch differs from the input's; every axis before
    them has the same extent on both sides, so one step of the leading loop
    advances by a fixed `out_rows * out_inner` on the destination and the whole
    leading product is a single level.

    A pad on a third axis from the end needs a fourth level, and a region that
    dropped it would read the wrong elements rather than fail, so it is refused
    here and the node stays on a portable kernel.

    The value is the other half of the gate, and its reason is the kernel rather
    than the region: `htp_ops_zero` is the only command in the library that fills
    a buffer without reading one, and it writes zero and nothing else. So `value`
    must be absent or zero. A nonzero fill would need a second region source
    holding the value, which the border decomposes into three regions or fewer
    only when the pad is on the last axis alone, or a full-size constant operand,
    which is a weight as large as the result. Neither is worth its commands for a
    case torch's own default already covers.

    A negative pad crops, which is a slice rather than a pad and has a stride
    form of its own; `slice_copy` is the op for it. The memset this emitter
    depends on is only correct when every pad is non-negative, so the negative
    form is refused here rather than read as a zero-size region.

    Both the support check and the emitter call this, so they cannot disagree
    about which pads the DSP can run -- a node the emitter would refuse has to be
    refused here instead, or the whole export fails rather than falling back.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node) or len(node.args) < 2:
        return None
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return None
    if source_value.dtype not in (torch.float16, torch.float32):
        return None

    pads = node.args[1]
    # torch gives the pads in reverse axis order, two per padded axis, and
    # refuses an odd count or one that names more axes than the operand has.
    if not isinstance(pads, (list, tuple)) or not pads or len(pads) % 2:
        return None
    shape = list(source_value.shape)
    rank = len(shape)
    if not rank or len(pads) > 2 * rank:
        return None
    if any(isinstance(size, torch.SymInt) for size in shape + list(result_value.shape)):
        # The offsets and the inner pitch are params, so a symbolic extent would
        # be the traced example at run time: a wrong answer rather than a
        # failure. The leading product is a plain integer for the same reason.
        return None
    if not all(isinstance(size, int) and not isinstance(size, bool) for size in pads):
        return None
    if any(size < 0 for size in pads):
        return None

    value = node.args[2] if len(node.args) > 2 else None
    if isinstance(value, bool) or value not in (None, 0, 0.0):
        return None

    if any(pads[index] for index in range(4, len(pads))):
        return None

    inner = shape[-1]
    rows = shape[-2] if rank >= 2 else 1
    outer = 1
    for size in shape[:-2]:
        outer *= size
    before_inner, after_inner = pads[0], pads[1]
    before_rows, after_rows = (pads[2], pads[3]) if len(pads) > 2 else (0, 0)
    out_inner = inner + before_inner + after_inner
    out_rows = rows + before_rows + after_rows
    if out_inner <= 0 or out_rows <= 0:
        return None
    if not any(pads):
        # Nothing is padded, so the region would be the operand's own bytes at
        # its own strides, which the kernel drops as a self-write -- leaving the
        # result unwritten. A no-op pad is left to the portable kernels.
        return None
    if result_value.numel() != outer * out_rows * out_inner:
        # Guards the arithmetic above against the shape it is meant to describe.
        # They are the same number or the region reads elsewhere.
        return None

    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
    return [
        0,
        0,
        before_rows * out_inner + before_inner,
        outer,
        rows,
        inner,
        rows * inner,
        inner,
        1,
        out_rows * out_inner,
        out_inner,
        1,
    ]


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


def reflect_pad_regions(node: torch.fx.Node):
    source = node.args[0]
    source_value = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
    result_value = node.meta.get("val")
    if source_value is None or result_value is None or not source_value.is_contiguous():
        return None
    shape = tuple(source_value.shape)
    pads = tuple(node.args[1])
    if len(pads) not in (2, 4) or not result_value.is_contiguous():
        return None
    if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
        return None
    if any(isinstance(pad, bool) or not isinstance(pad, int) for pad in pads):
        return None
    if any(pad < 0 for pad in pads) or not any(pads):
        return None
    inner = shape[-1]
    before_inner, after_inner = pads[0], pads[1]
    if max(before_inner, after_inner) >= inner:
        return None
    before_rows = after_rows = 0
    rows = 1
    outer = 1
    if len(pads) == 2:
        outer = math.prod(shape[:-1]) if shape[:-1] else 1
    if len(pads) == 4:
        rows = shape[-2]
        outer = math.prod(shape[:-2]) if shape[:-2] else 1
        before_rows, after_rows = pads[2], pads[3]
        if max(before_rows, after_rows) >= rows:
            return None
    out_inner = inner + before_inner + after_inner
    out_rows = rows + before_rows + after_rows
    if result_value.numel() != outer * out_rows * out_inner:
        return None
    row_pieces = []
    if before_rows:
        row_pieces.append((before_rows, -inner, 0, before_rows))
    row_pieces.append((0, inner, before_rows, rows))
    if after_rows:
        row_pieces.append((rows - 2, -inner, before_rows + rows, after_rows))
    col_pieces = []
    if before_inner:
        col_pieces.append((before_inner, -1, 0, before_inner))
    col_pieces.append((0, 1, before_inner, inner))
    if after_inner:
        col_pieces.append((inner - 2, -1, before_inner + inner, after_inner))
    regions = []
    for src_row, src_row_step, dst_row, row_count in row_pieces:
        for src_col, src_col_step, dst_col, col_count in col_pieces:
            regions.append([
                0,
                src_row * inner + src_col,
                dst_row * out_inner + dst_col,
                outer,
                row_count,
                col_count,
                rows * inner,
                src_row_step,
                src_col_step,
                out_rows * out_inner,
                out_inner,
                1,
            ])
    return regions


def _emit_clone_copy(node: torch.fx.Node, ctx) -> TensorRef:
    source = ctx.operand(node.args[0])
    out = ctx.result_for(node, _numel(node))
    numel = _numel(node)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[source],
            outputs=[out],
            params=[1, FP16_BYTES, 1, 0, 0, 0, 1, 1, numel, 1, 1, 1, 1, 1, 1],
        ),
    )
    return ctx.record(node, out)


def _emit_reflect_pad(node: torch.fx.Node, ctx) -> TensorRef:
    regions = reflect_pad_regions(node)
    if regions is None:
        raise RuntimeError("hexagon: reflect pad has no region plan")
    out = ctx.result_for(node, _numel(node))
    operand = ctx.operand(node.args[0])
    for start in range(0, len(regions), BLIT_BLOCKS_PER_COMMAND):
        ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[operand],
                outputs=[out],
                params=[min(BLIT_BLOCKS_PER_COMMAND, len(regions) - start), FP16_BYTES, 1]
                + [value for region in regions[start : start + BLIT_BLOCKS_PER_COMMAND] for value in region],
            ),
        )
    return ctx.record(node, out)


def _repeat_flip_extents(node: torch.fx.Node):
    """The contiguous source and result a repeat or flip reads and writes.

    Both ops are region walks over the operand's own bytes, so the operand has
    to be contiguous and the result has to be a plain reshape of it: a
    non-contiguous operand would be read at strides this emitter does not
    compute, and a result whose declared layout is not the one the region
    writes would be read by the next node at strides the graph believes.
    `to_edge` records a contiguous stride for a flip's result whatever the
    axes are (the copy is the one that makes it contiguous), so the flip gate
    takes the axes from the argument and the result's shape.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node) or len(node.args) < 2:
        return None
    source_value = source.meta.get("val")
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return None
    if source_value.dtype not in (torch.float16, torch.float32):
        return None
    if not source_value.is_contiguous() or not result_value.is_contiguous():
        return None
    if source_value.numel() <= 0 or result_value.numel() <= 0:
        return None
    return source_value, result_value


def repeat_region(node: torch.fx.Node) -> Optional[List[int]]:
    """The blit region for `aten.repeat.default`, or None when there is none.

    A repeat of one axis is `cat([x] * factor, dim=axis)`: the whole
    trailing block is copied `factor` times, once per leading index. That is
    two loops, so it fits one region whichever axis carries the factor, and
    that region is the answer for a repeat along the last axis as much as for
    one along a middle or leading axis. One command, any factor -- the factor
    rides on a level's extent rather than on a region per phase, so a factor
    of 64 costs the same single region a factor of two does.

    A factor list of all ones is the identity and returns an empty region, so
    the alias path re-points the operand's TensorRef. It is the only view: a
    factor above one adds elements, so the bytes are read more than once and
    an alias would answer a different question than the host's. A factor on
    an axis the list adds in front of the operand's rank is not a reshape
    either -- `x.repeat(2, 1)` on a three-element `x` tiles the whole tensor
    -- so those axes are padded with ones to line the two up rather than
    dismissed.

    A shape this function cannot describe returns None, and the support check
    turns that into a portable kernel: a repeat of two or more axes (four
    loops, and only three levels), a symbolic extent or factor (the offsets
    and sizes are params, and a run-time length would leave them at the traced
    example), a non-contiguous operand, or an extent of zero.
    """
    values = _repeat_flip_extents(node)
    if values is None:
        return None
    source_value, result_value = values
    shape = list(source_value.shape)
    result = list(result_value.shape)
    rank = len(shape)
    if rank == 0:
        return None
    factors = node.args[1]
    if not isinstance(factors, (list, tuple)) or len(factors) < rank:
        return None
    if not all(
        isinstance(factor, int) and not isinstance(factor, bool) and factor >= 1
        for factor in factors
    ):
        return None
    if any(isinstance(size, torch.SymInt) for size in shape + result):
        return None
    # The identity comes first, and before the shape check, because a list of
    # ones may be longer than the operand's rank: the result then has a leading
    # axis of one that the check below would read as a mismatch. It is the only
    # view a repeat has: any factor above one adds elements, so the bytes are
    # read more than once and an alias would answer a different question than
    # the host's. A factor on an axis the list adds in front of the operand's
    # rank is not a reshape either -- `x.repeat(2, 1)` on a three-element `x` tiles the
    # whole tensor -- so the leading axes are padded with ones to line the two
    # up rather than dismissed.
    if all(factor == 1 for factor in factors):
        return []
    # The result's leading axes are the ones the factor list added, so the
    # operand's own axes start at the same offset the factors' do. Reading the
    # result at the operand's index instead is how a prepended axis of two came
    # to be checked against a shape of one.
    offset = len(factors) - rank
    if any(
        result[offset + axis] != shape[axis] * factors[offset + axis]
        for axis in range(rank)
    ):
        return None
    padded = (1,) * (len(factors) - rank) + tuple(shape)
    repeated = [axis for axis in range(len(factors)) if factors[axis] > 1]
    if len(repeated) > 1:
        return None
    axis = repeated[0]

    # A repeat of one axis is `cat([x] * factor, dim=axis)`: the trailing block
    # `shape[axis:]` is copied `factor` times, once per leading index. That is
    # two loops, so it fits a region's three levels whichever axis carries the
    # factor, and the axis need not be the last one -- which is what makes the
    # form uniform.
    #
    # The loops are the leading product and the factor, and the innermost level
    # is the block itself copied contiguously, with the factor as a broadcast
    # of it: source stride zero, destination stride one block. The factor rides
    # on a level's extent rather than on a region per phase, so one command
    # covers any factor, and nothing here needs a region count past the one a
    # command already has room for.
    #
    # A repeat of two or more axes is `cat` along each in turn, which is four
    # loops and does not fit three levels. The host composes it exactly from
    # successive single-axis repeats, but this emitter emits one command, so
    # the shape is refused rather than half applied.
    lead = 1
    for size in padded[:axis]:
        lead *= size
    run = 1
    for size in padded[axis:]:
        run *= size

    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
    return [
        0,
        0,
        0,
        lead,
        factors[axis],
        run,
        run,
        0,
        1,
        run * factors[axis],
        run,
        1,
    ]


def flip_region(node: torch.fx.Node) -> Optional[List[int]]:
    """The blit region for `aten.flip.default`, or None when there is none.

    A reversal of a contiguous operand is a strided read and a contiguous
    write. `HtpOpsRasterRegion`'s strides are `int32_t` and the kernel's
    generic walk multiplies them as signed values, so a flipped axis is a
    negative source stride read from its last element: the innermost axis
    takes -1, the axis before it takes its inner pitch, and an axis further
    out takes the pitch of everything behind it. The source offset is where
    the read starts, which is each reversed axis's last element.

    The walk has three levels -- the last axis, the one before it, and
    everything else -- so the second-to-last and last axes are exact, while a
    reversal in the outer group needs that group's other axes to be one. Two
    reversals inside a group of two is a reversal per axis and is four loops,
    which no region holds; that shape is refused, and on a rank of three or
    less there is no outer group, so every subset reverses.

    A flip of axes whose extents are all one is the identity and returns an
    empty region, as does a flip of no axes, so the alias path re-points the
    operand. The result's stride as the edge graph records it is not consulted:
    a flip's result is a view with negative strides, and `to_edge` reports
    that view as contiguous, so the region's geometry is the only place the
    reversal can live.
    """
    values = _repeat_flip_extents(node)
    if values is None:
        return None
    source_value, result_value = values
    shape = list(source_value.shape)
    rank = len(shape)
    dims = node.args[1]
    if not isinstance(dims, (list, tuple)):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims):
        return None
    # A symbolic extent is a refusal: the source offset is the reversed axis's
    # last element and its stride is that axis's pitch, so both are params and
    # a run-time length would leave them at the traced example.
    if any(isinstance(size, torch.SymInt) for size in shape):
        return None
    flipped = sorted((dim + rank if dim < 0 else dim) for dim in dims)
    if any(not 0 <= dim < rank for dim in flipped) or len(set(flipped)) != len(
        flipped
    ):
        return None
    if list(result_value.shape) != shape:
        return None
    if all(shape[dim] == 1 for dim in flipped):
        return []

    # The walk is three levels: level 2 is the last axis, level 1 the second to
    # last, level 0 everything before them. The result is contiguous whatever
    # the source is, so a level's destination stride is the pitch of the axis
    # it walks in the output, and its source stride is the same pitch read from
    # the source, negated where the axis reverses. The two vectors differ only
    # in their signs.
    pitch = []
    for dim in range(rank):
        inner = 1
        for size in shape[dim + 1 :]:
            inner *= size
        pitch.append(inner)
    offset = sum((shape[dim] - 1) * pitch[dim] for dim in flipped)

    size1 = shape[rank - 2] if rank >= 2 else 1
    size2 = shape[-1]
    if rank >= 3:
        group = list(range(rank - 2))
        size0 = 1
        for size in shape[: rank - 2]:
            size0 *= size
        stride0 = size1 * size2
        stride1 = size2
    else:
        group = []
        size0 = 1
        stride0 = 0
        stride1 = size2 if rank >= 2 else 0

    # Levels 1 and 2 each hold one axis, so negating their stride is exact.
    # Level 0 holds a group, and the walk over it reads the source at an offset
    # that is affine in the level's index only when the group's other axes are
    # one -- two reversed axes in a group of two are a reversal per axis, which
    # is four loops and does not fit.
    reversed_group = [dim for dim in flipped if dim in group]
    if reversed_group and any(
        shape[other] != 1 for other in group if other != reversed_group[0]
    ):
        return None
    source0 = -pitch[reversed_group[0]] if reversed_group else stride0
    source1 = -stride1 if rank >= 2 and (rank - 2) in flipped else stride1
    source2 = -1 if (rank - 1) in flipped else 1

    # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz], dstStride[xyz]
    return [
        0,
        offset,
        0,
        size0,
        size1,
        size2,
        source0,
        source1,
        source2,
        stride0,
        stride1,
        1,
    ]


def _emit_repeat_flip(node: torch.fx.Node, ctx) -> TensorRef:
    """A repeat or a flip is one blit, or no command when it is a view.

    Both reach the same emitter because both are the same decision: the
    region function is the gate, a None answer is the alias path, and a
    region is one RASTER_BLIT command carrying it.
    """
    region = repeat_region(node) if node.target in REPEAT_TARGETS else flip_region(node)
    if region is None:
        raise RuntimeError(f"hexagon: {node.target} is not an emittable repeat or flip")
    if not region:
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
def _emit_constant_pad(node: torch.fx.Node, ctx) -> TensorRef:
    """A zero-filling pad as the memset the pad value is and one region.

    The memset covers the whole result and is the only writer of the border; the
    blit then overwrites the interior with the operand. Two commands write one
    buffer, in that order, which is the shape the pool's blocked activation is
    already built from -- the command group runs in order, so the second sees
    what the first left.
    """
    out = ctx.result_for(node, _numel(node))
    _emit_zero(ctx, node, out)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(node.args[0])],
            outputs=[out],
            params=[1, FP16_BYTES, 1] + constant_pad_region(node),
        ),
    )
    return ctx.record(node, out)


def _emit_leaky_relu(node: torch.fx.Node, ctx) -> TensorRef:
    slope = _scalar_arg(node, "negative_slope", 1, 0.01)
    if slope is None:
        raise RuntimeError("hexagon: leaky_relu needs a compile-time negative_slope")
    out = ctx.result_for(node, _numel(node))
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_RELU,
            inputs=[ctx.operand(node.args[0])],
            outputs=[out],
            params=[_upper_product(tuple(_value_of(node).shape), ctx), FP16_BYTES, _float_bits(slope)],
        ),
    )
    _patch_dynamic_numel(ctx, op_index, node)
    return ctx.record(node, out)


def _emit_prelu(node: torch.fx.Node, ctx) -> TensorRef:
    source, slope = node.args[:2]
    shape = tuple(_value_of(node).shape)
    channel = shape[1]
    plane = math.prod(shape[2:]) if shape[2:] else 1
    batch = shape[0] if shape else 1
    out = ctx.result_for(node, _numel(node))
    ctx.emit(
        node,
        Op(
            type=DSP_OP_PRELU,
            inputs=[ctx.operand(source), ctx.operand(slope)],
            outputs=[out],
            params=[_numel(node), FP16_BYTES, plane, channel, _numel(slope), 1, batch],
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


def _unary_command(
    node: torch.fx.Node, ctx, src: TensorRef, out: TensorRef, numel: int, op_name: str
) -> None:
    """DSP_OP_UNARY over a source and destination of ours.

    The source and the destination are arguments rather than the node's own
    because the elu composition runs `exp` over an intermediate it made; the
    kernel does not care where the value came from, only how wide it is.
    """
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_UNARY,
            inputs=[src],
            outputs=[out],
            # size is in elements, not bytes.
            params=[numel, UNARY_OP_TYPES[op_name], FP16_BYTES],
        ),
    )
    _patch_dynamic_numel(ctx, op_index, node)


def _unary(op_name: str):
    def emit(node: torch.fx.Node, ctx) -> TensorRef:
        src = node.args[0]
        _require_arena_dtype(node, f"unary {op_name} input")
        numel = _numel(node)
        out = ctx.result_for(node, numel)
        _unary_command(
            node,
            ctx,
            ctx.operand(src),
            out,
            _upper_product(tuple(_value_of(node).shape), ctx),
            op_name,
        )
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


def _clamp_command(
    node: torch.fx.Node, ctx, src: TensorRef, out: TensorRef, numel: int, lower, upper
) -> None:
    """DSP_OP_UNARY's clamp entry point over a source and destination of ours.

    The source and the destination are arguments rather than the node's own
    because the elu composition clamps two intermediates of its own: the kernel
    does not care where the values came from, only how wide they are.
    """
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_UNARY,
            inputs=[src],
            outputs=[out],
            # size is in elements, not bytes, as in _unary.
            params=[
                numel,
                UNARY_OP_TYPES["clamp"],
                FP16_BYTES,
                _clamp_bound_bits(lower, float("-inf")),
                _clamp_bound_bits(upper, float("inf")),
            ],
        ),
    )
    _patch_dynamic_numel(ctx, op_index, node)


def _emit_clamp_bounds(node: torch.fx.Node, ctx, lower, upper) -> TensorRef:
    """DSP_OP_UNARY's clamp entry point, with the two bounds given outright.

    Three edge ops are this one command: clamp, hardtanh (whose schema carries
    min_val and max_val in the positions clamp's min and max occupy, and which
    torch itself computes as clamp), and relu, which is clamp between zero and
    infinity. The kernel compares unordered but restores the input where it is a
    NaN (`htp_ops_clamp_fp16_chunk`, unary_ops.cc:505-509), so all three answer
    NaN with NaN as torch does.
    """
    src = node.args[0]
    _require_arena_dtype(node, "clamp input")
    numel = _numel(node)
    out = ctx.result_for(node, numel)
    _clamp_command(
        node,
        ctx,
        ctx.operand(src),
        out,
        _upper_product(tuple(_value_of(node).shape), ctx),
        lower,
        upper,
    )
    return ctx.record(node, out)


def _emit_clamp(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.clamp and aten.hardtanh, whose bounds arrive in the same slots.

    The bounds are positional and not always both there: a one-bound clamp is
    min= that bound, and a clamp with no bounds at all is the identity, which
    the infinities in _clamp_bound_bits express on their own.
    """
    lower = node.args[1] if len(node.args) > 1 else None
    upper = node.args[2] if len(node.args) > 2 else None
    return _emit_clamp_bounds(node, ctx, lower, upper)


def _emit_relu(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.relu as clamp(x, 0, inf).

    torch's relu is max(x, 0), and the clamp kernel's upper bound of infinity is
    a compare that never fires, so the command is that max. The DSP's unary
    table has no relu of its own (HtpOpsUnaryOpType, unary_ops.cc:15-31), which
    is why the op had been left to the portable kernels; the clamp entry point
    is the form that does exist.
    """
    return _emit_clamp_bounds(node, ctx, 0.0, None)


def clamp_tensor_bounds(node: torch.fx.Node) -> Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node]]:
    """`aten.clamp.Tensor`'s min and max operands, or None for an absent one."""

    return (
        node.args[1] if len(node.args) > 1 else None,
        node.args[2] if len(node.args) > 2 else None,
    )


def clamp_tensor_fits(node: torch.fx.Node) -> bool:
    """Whether both bounds are operands the binary stride table can read.

    The two commands are the ordinary binary max and min, so each bound has to
    satisfy what `_broadcast_fits_dsp_limits` asks of a binary's own operands:
    either a single element or the whole output, at rank 8 or below. A clamp with
    neither bound is the identity, which `_emit_clamp` already answers with the
    infinities, and a bound that is a literal belongs to that overload rather than
    to this one.
    """
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return False
    out_shape = tuple(_value_of(node).shape)
    if len(out_shape) > MAX_BROADCAST_RANK:
        return False
    lower, upper = clamp_tensor_bounds(node)
    if lower is None and upper is None:
        return False
    for bound in (lower, upper):
        if bound is None:
            continue
        if not isinstance(bound, torch.fx.Node):
            return False
        value = _value_of(bound)
        if value.dtype not in (torch.float16, torch.float32):
            return False
        if len(value.shape) > MAX_BROADCAST_RANK:
            return False
        try:
            _broadcast_strides(tuple(value.shape), out_shape, _UPPER_SHAPE)
        except ValueError:
            return False
    return True


def _emit_clamp_tensor(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.clamp.Tensor as the two binary commands a clamp already is.

    `min(max(x, lo), hi)`, one `BINARY_ELEMENTWISE` per bound, and therefore
    wire-exact: a max and a min pick one of their two operands and compute
    nothing, so every element of the answer is a pair of bytes the graph already
    had. A one-sided clamp is the one command, not a pair.

    The bound goes in as the FIRST operand on purpose. The kernel's max is
    `a > b ? a : b` and its min `a < b ? a : b` (eltwise_ops.cc:152-155), so
    what survives an unordered pair is the second one, and the activation is the
    second operand here -- so a NaN activation comes back as a NaN, as torch's
    clamp has it. A NaN *bound* is the one case this drops rather than
    propagates; so does the two-select form, because the comparison in front of
    it is what drops it, and so does the portable kernel's own answer differ.
    """
    src = node.args[0]
    _require_arena_dtype(node, "clamp.Tensor input")
    out_shape = tuple(_value_of(node).shape)
    if any(isinstance(dim, torch.SymInt) for dim in out_shape):
        raise RuntimeError("hexagon: clamp.Tensor requires static dimensions")
    out_numel = _upper_product(out_shape, ctx)
    out = ctx.result_for(node, out_numel)
    lower, upper = clamp_tensor_bounds(node)
    value = ctx.operand(src)
    if lower is not None:
        bound_shape = tuple(_value_of(lower).shape)
        held = out if upper is None else ctx.activation_for_shape(out_shape)
        _emit_binary_refs(
            node,
            ctx,
            ctx.operand(lower),
            value,
            _upper_product(bound_shape, ctx),
            out_numel,
            bound_shape,
            out_shape,
            out_shape=out_shape,
            op_name="max",
            out=held,
        )
        value = held
    if upper is not None:
        bound_shape = tuple(_value_of(upper).shape)
        _emit_binary_refs(
            node,
            ctx,
            ctx.operand(upper),
            value,
            _upper_product(bound_shape, ctx),
            out_numel,
            bound_shape,
            out_shape,
            out_shape=out_shape,
            op_name="min",
            out=out,
        )
    return ctx.record(node, out)

#: The three coefficients `aten.elu` carries, read off the node in the positions
#: its schema puts them in: `alpha`, `scale` and `input_scale`.
def elu_coefficients(node: torch.fx.Node):
    return (
        _scalar_arg(node, "alpha", 1, 1.0),
        _scalar_arg(node, "scale", 2, 1.0),
        _scalar_arg(node, "input_scale", 3, 1.0),
    )


def elu_fits(node: torch.fx.Node) -> bool:
    """Whether this elu is the split the DSP's commands can compute.

    torch's kernel is `a < 0 ? expm1(a * input_scale) * (alpha * scale) : a *
    scale` (ATen/native/cpu/Elu.h:25-31), and the composition here is that read
    as `max(a, 0) * scale + min(exp(a) - 1) * (alpha * scale), 0)`. The rewrite is
    the identity only when the second term is non-positive wherever the first is
    not, so both coefficients have to be non-negative; and `input_scale` has to
    be one, because the DSP has no `expm1` to fold an argument scale into. The
    unary table has no elu of its own (1..17, unary_ops.cc:14-31) and the binary
    table no slope form (1..12): the reason this op is cheap is the composition,
    which is what makes the three refusals here the whole of the gate.
    """
    alpha, scale, input_scale = elu_coefficients(node)
    if None in (alpha, scale, input_scale):
        return False
    if input_scale != 1.0:
        return False
    return all(math.isfinite(value) and value >= 0.0 for value in (alpha, scale))


def _emit_elu(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.elu as six commands, or seven when it carries a `scale`.

    `max(x, 0) * scale + min((exp(x) - 1) * alpha * scale, 0)`: the clamp entry
    point is the relu and the min, the unary table's `exp` is the exponential,
    and the two binaries are the `exp(x) - 1` and its scale. No command here is
    new to this backend and no kernel is new to the DSP.

    The clamp kernel restores a NaN input by a bit test (unary_ops.cc:498-541),
    so both halves hand a NaN back as a NaN and the add of two NaNs is the NaN
    torch's kernel has already made by the time it narrows. The cost is the
    fp16 step at every stage: measured against `F.elu` on `randn * 2` the
    largest difference is 9.766e-04, which is one fp16 step at the top of that
    range, and over a grid from -30 to 8 it is the same figure.
    """
    src = node.args[0]
    _require_arena_dtype(node, "elu input")
    out_shape = tuple(_value_of(node).shape)
    if any(isinstance(dim, torch.SymInt) for dim in out_shape):
        raise RuntimeError("hexagon: elu requires static dimensions")
    alpha, scale, _ = elu_coefficients(node)
    numel = _upper_product(out_shape, ctx)
    out = ctx.result_for(node, numel)
    value = ctx.operand(src)

    exponential = ctx.activation_for_shape(out_shape)
    _unary_command(node, ctx, value, exponential, numel, "exp")
    shifted = ctx.activation_for_shape(out_shape)
    _emit_binary_refs(
        node, ctx, exponential, ctx.scalar(1.0), numel, 1, out_shape, (), out_shape, "sub", shifted
    )
    scaled = ctx.activation_for_shape(out_shape)
    _emit_binary_refs(
        node,
        ctx,
        shifted,
        ctx.scalar(alpha * scale),
        numel,
        1,
        out_shape,
        (),
        out_shape,
        "mul",
        scaled,
    )
    negative = ctx.activation_for_shape(out_shape)
    _clamp_command(node, ctx, scaled, negative, numel, None, 0.0)
    positive = ctx.activation_for_shape(out_shape)
    _clamp_command(node, ctx, value, positive, numel, 0.0, None)
    if scale != 1.0:
        # torch's positive branch is `a * scale` computed in fp32 and rounded
        # once, which is what the mul_scalar entry point is for; a fp16 multiply
        # here would round twice.
        widened = ctx.activation_for_shape(out_shape)
        _mul_scalar_command(node, ctx, positive, widened, numel, scale)
        positive = widened
    _emit_binary_refs(
        node, ctx, positive, negative, numel, numel, out_shape, out_shape, out_shape, "add", out
    )
    return ctx.record(node, out)


def _mul_scalar_command(
    node: torch.fx.Node, ctx, src: TensorRef, out: TensorRef, numel: int, scale: float
) -> None:
    """DSP_OP_UNARY's mul_scalar entry point over a ref pair of ours.

    params[3] is the scale as an fp32 bit pattern, because widening to fp32 is
    the whole point of the entry point: torch multiplies a half tensor by a
    python float in fp32 and rounds the product back.
    """
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_UNARY,
            inputs=[src],
            outputs=[out],
            params=[
                numel,
                UNARY_OP_TYPES["mul_scalar"],
                FP16_BYTES,
                struct.unpack("<i", struct.pack("<f", float(scale)))[0],
                0,
            ],
        ),
    )
    _patch_dynamic_numel(ctx, op_index, node)


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
    _mul_scalar_command(
        node,
        ctx,
        ctx.operand(values),
        out,
        _upper_product(tuple(_value_of(node).shape), ctx),
        scale,
    )
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


#: The widest broadcast the binary descriptor can name: the tail after params[8] is
#: a rank, the output extents, and one stride list per operand, all inside the
#: 40-int command budget (9 for the head, 25 for the tail, and a rank 9 needs 33).
MAX_BROADCAST_RANK = 8


class _UpperShape:
    """The upper bound of a shape, for the stride check outside a lowering.

    `_broadcast_strides` asks a context for the upper bound of every extent it
    walks, because a run-time length is a number the runtime refills. A support
    predicate has no context, and the export's own bound is the right answer
    there: it is the largest extent the node can be given.
    """

    @staticmethod
    def upper_shape(shape):
        return tuple(
            eval_upper_bound(extent) if isinstance(extent, torch.SymInt)
            else int(extent)
            for extent in shape
        )


_UPPER_SHAPE = _UpperShape()


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
    """Row-major strides for a singleton-broadcast operand.

    The DSP computes sum(coord[d] * stride[d]) without an operand extent or a
    modulo, so only extent-one axes can repeat. Every other axis has to cover
    the output; a non-singleton smaller extent is a tile the descriptor cannot
    address and is refused instead of walking past the operand.
    """
    shape = ctx.upper_shape(shape)
    out_shape = ctx.upper_shape(out_shape)
    rank = len(out_shape)
    if len(shape) > rank:
        raise ValueError("hexagon: broadcast operand has more axes than its output")
    padded = (1,) * (rank - len(shape)) + tuple(shape)
    mismatches = sum(
        extent != 1 and extent != out_extent
        for extent, out_extent in zip(padded, out_shape)
    )
    if mismatches:
        raise ValueError(
            "hexagon: broadcast operand is neither singleton nor output-sized"
        )
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


def _emit_binary_refs(
    node: torch.fx.Node,
    ctx,
    lhs: TensorRef,
    rhs: TensorRef,
    lhs_numel: int,
    rhs_numel: int,
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    op_name: str,
    out: Optional[TensorRef] = None,
) -> TensorRef:
    if out is None:
        out = ctx.result_for(node, _upper_product(out_shape, ctx))
    ctx.emit(
        node,
        Op(
            type=DSP_OP_BINARY_ELEMENTWISE,
            inputs=[lhs, rhs],
            outputs=[out],
            params=[
                _upper_product(out_shape, ctx),
                lhs_numel,
                rhs_numel,
                BINARY_OP_TYPES[op_name],
                FP16_BYTES,
                FP16_BYTES,
                0,
                0,
                *_broadcast_tail(lhs_shape, rhs_shape, out_shape, ctx),
            ],
        ),
    )
    return out


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


def where_is_emittable(node: torch.fx.Node) -> bool:
    """Whether this `where` is the shape htp_ops_select can walk.

    The command takes one condition, two values and a size for each, and its own
    guard admits a condition that is either the whole output or a single element
    (eltwise_ops.cc:2403-2407). The values are admitted at the whole output, at
    one element, or at a per-channel run, and the per-channel form is what the
    prelu path uses -- it needs a channel count and an inner size this emitter
    does not compute. So the form taken here is the strict one: all three
    operands either match the output's element count or are a single element,
    and the per-channel form stays on the portable kernels.

    The condition's dtype is checked as well as its width, because a
    torch.bool *is* one byte and any other one-byte tensor is not a flag the
    kernel's `!= 0` test would read the way torch would.
    """
    result = _value_of(node)
    if result.dtype not in (torch.float16, torch.float32):
        return False
    out_numel = result.numel()
    cond = node.args[0]
    if not isinstance(cond, torch.fx.Node):
        return False
    cond_value = _value_of(cond)
    if cond_value.dtype is not torch.bool:
        return False
    for operand in node.args[:3]:
        if not isinstance(operand, torch.fx.Node):
            return False
        value = _value_of(operand)
        if value.numel() not in (out_numel, 1):
            return False
    return True


def _emit_where(node: torch.fx.Node, ctx) -> TensorRef:
    """`cond ? a : b` as one SELECT command.

    The condition is the one operand whose width is not two bytes: the runtime
    gives an input the size the blob declares for its slot, and the partitioner
    tags a bool input at its own width, so one byte per element lands in the
    arena and condBytes says so. Reading it as fp16 would take each element's
    neighbour for its high half.
    """
    cond, lhs, rhs = node.args[0], node.args[1], node.args[2]
    out_numel = _numel(node)
    out = ctx.result_for(node, out_numel)
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_SELECT,
            # The dispatcher's order: the condition first, then the value the
            # test selects and the one it does not (execute_command.cc:646-651).
            inputs=[ctx.operand(cond), ctx.operand(lhs), ctx.operand(rhs)],
            outputs=[out],
            params=[
                out_numel,
                _upper_product(tuple(_value_of(cond).shape), ctx),
                _upper_product(tuple(_value_of(lhs).shape), ctx),
                _upper_product(tuple(_value_of(rhs).shape), ctx),
                FP16_BYTES,
                # A torch.bool is one byte per element, and the kernel's `!= 0`
                # test is the test torch makes.
                BOOL_BYTES,
                # The channel and inner sizes are read only by the per-channel
                # input mode, which this emitter never asks for.
                0,
                0,
            ],
        ),
    )
    _patch_dynamic_numel(ctx, op_index, node)
    for index, operand in enumerate((cond, lhs, rhs), start=1):
        _patch_dynamic_numel(ctx, op_index, operand, index)
    return ctx.record(node, out)


def _emit_compare(op_name: str):
    """A comparison as the DSP's own GREATER or LESS, then a one-byte select.

    The compare arms of htp_ops_binary_elementwise write fp16 1.0 and 0.0, and
    at bytes 2 it takes that arm whatever outputIsFloat says
    (eltwise_ops.cc:1166-1173), so the first command leaves a flag at two bytes
    and does not need an op type of its own. The second is SELECT at one byte,
    whose one-byte arm copies a byte at a time (eltwise_ops.cc:2157-2168) and has
    been in the library the whole time; the constants it copies between are what
    make the result a torch.bool rather than two more flags.

    Both halves matter and they are not the same half. GREATER and LESS are IEEE
    -- a NaN compares false and -0.0 is not greater than 0.0 -- which is what a
    subtract cannot be at any length; the select is what only packs.
    """

    def emit(node: torch.fx.Node, ctx) -> TensorRef:
        lhs, rhs = node.args[0], node.args[1]
        _require_arena_dtype(node, f"comparison {op_name}")
        for operand in (lhs, rhs):
            if isinstance(operand, torch.fx.Node):
                _require_arena_dtype(operand, f"comparison {op_name} operand")

        out_shape = tuple(_value_of(node).shape)
        out_numel = _upper_product(out_shape, ctx)
        flags = ctx.activation_for_shape(out_shape, torch.float16)
        op_index = ctx.emit(
            node,
            Op(
                type=DSP_OP_BINARY_ELEMENTWISE,
                inputs=[ctx.operand(lhs), ctx.operand(rhs)],
                outputs=[flags],
                params=[
                    out_numel,
                    _upper_product(_shape_of(lhs), ctx),
                    _upper_product(_shape_of(rhs), ctx),
                    BINARY_OP_TYPES[op_name],
                    FP16_BYTES,
                    FP16_BYTES,
                    0,  # inputs are not 4-byte floats
                    0,  # the compare arms write 1.0 and 0.0 at bytes 2 anyway
                    *_broadcast_tail(lhs, rhs, out_shape, ctx),
                ],
            ),
        )
        _patch_dynamic_numel(ctx, op_index, node)
        for index, operand in enumerate((lhs, rhs), start=1):
            if isinstance(operand, torch.fx.Node):
                _patch_dynamic_numel(ctx, op_index, operand, index)

        out = ctx.result_for(node, out_numel, torch.bool)
        select_index = ctx.emit(
            node,
            Op(
                type=DSP_OP_SELECT,
                # The dispatcher's order: the condition first, then the value the
                # test selects and the one it does not.
                inputs=[flags, ctx.scalar(1, torch.bool), ctx.scalar(0, torch.bool)],
                outputs=[out],
                params=[
                    out_numel,
                    out_numel,  # one flag an element, so the command's own guard takes it whole
                    1,  # each source is a single element
                    1,
                    BOOL_BYTES,  # the sources and the result are one byte wide
                    FP16_BYTES,  # the flags the first command left are two
                    0,  # the per-channel source mode, which this emitter never asks for
                    0,
                ],
            ),
        )
        _patch_dynamic_numel(ctx, select_index, node)
        return ctx.record(node, out)

    return emit


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


def weight_only_matmul_fits(m: int, k: int, n: int, bits: int) -> bool:
    """Whether one of the two quantized matmul emitters can run this geometry.

    The shape half of `quantized_matmul_is_emittable`, asked of the four numbers
    rather than of a graph. The quantizer needs this answer *before* the
    dequantize it would create exists, so it cannot read the pattern those
    predicates read; it calls this instead, which is also what the two `fits`
    predicates below call, so a shape one caller admits is a shape the other
    admits.

    The guards are the kernels' own: `K % 64` is what the activation pack's
    one-region-per-block form needs and is stricter than the kernel's own
    `K % 32` -- the kernel floors the division, so a K that is not a multiple of
    32 would silently drop a tile's tail -- and `N % 32` is the output-channel
    tile. `N % 64` is not required: the output repack carries the ragged last
    pack as a second region. The prefill entry adds the `M <= 32` K ceiling and
    its VTCM budget. `M == 1` belongs to the GEMV entries whatever the width of
    the stored weight, so the width only matters above one row, where it has to
    be a width a prefill emitter reads: 4 reaches command 22 and 8 reaches
    command 42. A width outside that set has no prefill kernel at all.

    The two predicates that ask this question have to agree -- the quantizer
    calls this one before the dequantize exists, and `_quantized_prefill_fits`
    calls it for the node the partitioner is about to hand it -- so the width set
    lives here and nowhere else.
    """
    if k % 64 or n % 32:
        return False
    if m == 1:
        return True
    if bits not in (4, 8):
        return False
    if m <= PREFILL_M32_MAX_M and k > PREFILL_M32_MAX_K:
        return False
    return _prefill_vtcm_bytes(k) <= PREFILL_VTCM_BYTES


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
    return m == 1 and weight_only_matmul_fits(m, k, n, quantized.bits)


def _quantized_prefill_fits(activation, quantized: QuantizedWeight) -> bool:
    """Whether the M > 1 prefill kernel can run this matmul.

    The guards themselves are `weight_only_matmul_fits`, which the quantizer also
    calls before it annotates anything: this predicate owns the `M > 1` half and
    that function owns the shape and budget half, so neither spelling of the
    question can move the guards without the other seeing it.

    M > 1 is the whole point -- M == 1 belongs to the GEMV entries, which read a
    different weight layout and are already wired. The M <= 32 dispatcher branch
    previously put a K-byte descriptor array on its stack. The corrected phone skel
    clears the measured boundary, but the host keeps the conservative safety refusal
    until a broader device matrix justifies widening it. The M > 32 branch
    heap-allocated the same descriptors and has no equivalent K ceiling.

    The int4 and int8 prefill entries share this geometry. Their weights are
    different packed layouts, but each has a host packer for the layout its
    kernel reads; the scale representation is the other difference.
    """
    # The two widths with a prefill emitter and nothing else: 4 reaches
    # _emit_quantized_prefill for command 22 and 8 reaches _emit_w8a16_prefill
    # for command 42. A width outside this set has no emitter, so it is refused
    # here rather than reaching a packer that was written for one of the two.
    if quantized.bits not in (4, 8):
        return False
    geometry = quantized_matmul_geometry(activation, quantized)
    if geometry is None:
        return False
    m, k, n = geometry
    return m > 1 and weight_only_matmul_fits(m, k, n, quantized.bits)


def _prefill_vtcm_bytes(k: int) -> int:
    """The VTCM the prefill kernel reserves for a weight K wide.

    Both dispatch targets hold the same three bands: the output channels of one
    chunk twice over, as fp16 tiles and as their int4 copies, plus the
    activation bands the walk stages through (`matmul_q4fp16.c:936-964`,
    `matmul_q4fp16_mle32.c:725-735`). The chunk sizes are the ones the emitter
    passes, and the activation count is the larger of the two kernels' -- the
    general one double-buffers M for the async store -- so this bound holds for
    either.
    """
    chunk = PREFILL_OUTPUT_CHANNEL_CHUNK
    per_output_tile = 32 * k * 2 + 32 * k // 2
    activation = 2 * 32 * k * 2
    return chunk * per_output_tile + activation + PREFILL_VTCM_FIXED


def quantized_matmul_weight(node) -> Optional[torch.fx.Node]:
    """The operand a matmul's weight-only weight arrives through, if any.

    mm and addmm both put the weight last, and it is a `dequantize_per_channel`
    rather than a tensor once PT2E has converted the graph. `aten.matmul` is not
    a third spelling to look for: it is not Core ATen, so `to_edge` rewrites the
    2-D form into the `mm` this reads before the partitioner ever sees the node.
    """
    if node.target in MM_TARGETS and len(node.args) > 1:
        return node.args[1]
    if node.target in ADDMM_TARGETS and len(node.args) >= 3:
        return node.args[2]
    return None


def quantized_matmul_is_emittable(node: torch.fx.Node, is_constant) -> bool:
    """Whether this matmul is one a quantized matmul emitter can place, in full.

    The support check, the dequantize's fusion test and the emitters all call
    this, so the three cannot disagree: a node the partitioner delegates is one
    an emitter accepts, and a dequantize is only fused away when every matmul
    reading it is one this accepts. The emitter cannot fall back -- it is past
    the partition boundary -- so a condition missing here is an export failure
    or, worse, a command the kernel reads past.

    `is_constant` is the caller's own test for "a value whose bytes I can read
    now", the same parameter the convolution and gather predicates take. The
    emitter has to pack the stored low-bit tensor and read the scale beside it,
    so a matmul whose weight operand is a run-time tensor instead of a
    parameter, buffer or lifted constant is one the emitter can only fail on:
    it is refused here, which puts the whole chain -- the dequantize with it --
    back on the portable kernels. Which operands are constants is not visible
    from the graph alone, since a parameter and a run-time tensor are both
    placeholders carrying a fake tensor, so the answer has to come from the
    caller that knows the program's signature.

    Two emitters answer to it: the M == 1 GEMV one and the M > 1 prefill one.
    Which of them a node belongs to is the M it carries, and the two do not
    overlap -- the prefill entry refuses M <= 1 and the GEMV entries refuse
    anything else. Both read the same two constants, so the test above covers
    the two of them at once.
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
    if not is_constant(quantized.weight) or not is_constant(quantized.scale):
        # Both are values that have to be in hand at export: the emitter packs
        # the one and rescales by the other. A weight the graph computes at
        # run time -- a scores-like `a @ b` over two live tensors -- has no
        # command here, and refusing it keeps the graph exportable.
        return False
    value = activation.meta.get("val")
    if not isinstance(value, torch.Tensor) or not value.is_contiguous():
        # The kernel reads K contiguous fp16 with aligned 128-byte loads; a
        # strided activation is one no kernel here walks.
        return False
    geometry = quantized_matmul_geometry(activation, quantized)
    if geometry is None:
        return False
    if not (
        _quantized_gemv_fits(activation, quantized)
        or _quantized_prefill_fits(activation, quantized)
    ):
        return False
    if bias is None:
        return True
    return _bias_is_one_value_per_channel(bias, geometry[2])


def quantized_matmul_is_refused(node: torch.fx.Node, is_constant) -> bool:
    """Whether this matmul carries a weight-only weight nothing here can run.

    A matmul whose weight is the weight-only pattern goes to a GEMV kernel
    rather than to the flat path, and that kernel has conditions of its own.
    One that fails them has to fall back whole -- the dequantize with it -- so
    this is asked about the matmul and answered for both. A weight the export
    cannot read is one of those conditions, which is why `is_constant` is
    forwarded rather than a local test standing in for the emitter's.
    """
    if quantized_weight(quantized_matmul_weight(node)) is None:
        return False
    return not quantized_matmul_is_emittable(node, is_constant)


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


def _dequantize_is_fused(node: torch.fx.Node, is_constant) -> bool:
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
        quantized_matmul_weight(user) is node
        and quantized_matmul_is_emittable(user, is_constant)
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
    if (w < -8).any() or (w > 7).any():
        raise RuntimeError("hexagon: q4a16 weight contains values outside [-8, 7]")
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
    32-bit vrmpy lane multiplies the weight of the same k. The tile order is
    checked against the host reorder itself: `reorderInt8SymWeightForHmx` is not
    in the vendored tree, but it is readable in the upstream MNN checkout
    (`source/backend/hexagon/execution/HexagonConvolution.cpp:464-548`), and
    `test_w8a16_weight_layout.py` compares this function's bytes against a
    transcription of it and against the contract above, with the fp16 tile order
    as the control that must not match. No DSP has run it.

    One blob serves both int8 entries rather than one each: the prefill kernel
    reads these same tiles and takes its scales from `weight + np*kp*1024`, the
    byte they end on (`conv1x1_w8a16_sym_per_channel.cc:520-521`), which is where
    the host reorder puts the one fp16 scale per output channel
    (HexagonConvolution.cpp:1152-1170, and the same buffer reaches both entries at
    :892-908). What an int8 prefill blob needs beyond this function is that tail,
    not another tile order -- this one stops at the tiles because the GEMV entry
    reads its scales as a separate fp32 operand.
    """
    import numpy as np

    w = np.asarray(weight, dtype=np.int32)
    if w.shape != (k, n):
        raise RuntimeError(f"hexagon: w8a16 weight is {w.shape}, expected ({k}, {n})")
    if k % 64 or n % 32:
        raise RuntimeError(
            f"hexagon: w8a16 needs K a multiple of 64 and N of 32, got {k}x{n}"
        )
    if (w < -128).any() or (w > 127).any():
        raise RuntimeError("hexagon: w8a16 weight contains values outside [-128, 127]")
    kp, np_ = k // 32, n // 32
    padded = np.zeros((np_ * 32, kp * 32), dtype=np.int32)
    padded[:n, :k] = np.clip(w.T, -128, 127)
    # (oy, ocIn, kx, kk) -> (oy, kx, ocIn, g, four), then permute the four k
    # values within each group to the kernel's byte order.
    t = padded.reshape(np_, 32, kp, 32).transpose(0, 2, 1, 3)
    t = t.reshape(np_, kp, 32, 8, 4)[..., (0, 2, 1, 3)]
    packed = t.transpose(0, 1, 3, 2, 4)
    return packed.astype(np.uint8).reshape(np_ * kp, 1024).tobytes()


def pack_w8a16_prefill_weight(weight, scale, k: int, n: int) -> bytes:
    """A (k, n) int8 weight and its fp16 per-channel scales for command 42."""
    import numpy as np

    packed = pack_w8a16_gemv_weight(weight, k, n)
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scales.size != n:
        raise RuntimeError(f"hexagon: w8a16 has {scales.size} scales, expected {n}")
    return packed + scales.astype(np.float16).tobytes()


def pack_q4a16_prefill_weight(weight, scale, k: int, n: int, scale_block_num: int = 1) -> bytes:
    """A (k, n) int4 weight in the tile order the prefill kernel reads.

    The GEMV packer above and this one are different layouts for the same
    weight: the vrmpy one feeds the integer GEMV kernel, this one feeds the HMX
    prefill kernel, which dequantizes a 32x32 tile to fp16 and multiplies it by
    an fp16 activation. The two are not interchangeable, and neither is the
    other's scales -- the prefill kernel reads one fp16 scale per output channel
    where the GEMV kernel reads fp32.

    The tile order is not transcribed from a comment. It is a port of
    `htp_ops_weight_reorder_int4` (`matmul_q4fp16.c:93`), the vendored tree's own
    reorder into this layout: the raw plane is `[n][k/2]` with the even k in the
    high nibble and the value offset by 8, each block of 32 output channels by 32
    k values becomes one 512-byte tile at `(y*kp + x)*512` -- y being the output
    channel tile and x the k tile -- and the fp16 scales follow the tiles, which
    is where the kernel's `b_scale` pointer lands (`im2col`-style wrappers set it
    at `weight + icP*ocP*512`). Inside a tile the block is laid out as
    `local[64*xi + 2*yi + p]`: the k byte index xi outermost, the output channel
    yi next and the nibble p innermost; the 1024 bytes are then byte-shuffled in
    eight 128-byte groups and folded to 512 by packing each byte of the upper
    half into the high nibble of the matching byte below it.

    What is checked: `test_prefill_on_sim.py` runs that vendored reorder on the
    simulator and compares its output with this function byte for byte, and the
    two HVX operations the fold is made of are measured there too rather than
    read out of a manual. What is not: nothing has run this on a device.
    """
    import numpy as np

    w = np.asarray(weight, dtype=np.int32)
    if w.shape != (k, n):
        raise RuntimeError(f"hexagon: q4a16 weight is {w.shape}, expected ({k}, {n})")
    if k % 64 or n % 32:
        raise RuntimeError(
            f"hexagon: q4a16 prefill needs K a multiple of 64 and N of 32, got {k}x{n}"
        )
    if (w < -8).any() or (w > 7).any():
        raise RuntimeError("hexagon: q4a16 weight contains values outside [-8, 7]")
    kp, np_ = k // 32, n // 32
    if scale_block_num < 1 or (k // 32) % scale_block_num:
        raise RuntimeError(
            f"hexagon: q4a16 needs K/32 divisible by the block count, got {k} and {scale_block_num}"
        )
    scales = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scales.size != n * scale_block_num:
        raise RuntimeError(
            f"hexagon: q4a16 has {scales.size} scales, expected {n * scale_block_num}"
        )

    # The raw plane the vendored reorder starts from, one byte per two k values.
    nibbles = w.T + 8
    raw = np.zeros((np_, 32, kp, 16), dtype=np.uint8)
    raw[:, :, :, :] = (nibbles[:, 0::2] * 16 + nibbles[:, 1::2]).reshape(
        np_, 32, kp, 16
    )
    # local[64*xi + 2*yi + p]: xi is the k byte, yi the output channel, p the
    # nibble (0 = the even k, in the high nibble of the byte).
    local = np.empty((np_, kp, 16, 32, 2), dtype=np.uint8)
    local[..., 0] = raw.transpose(0, 2, 3, 1) >> 4
    local[..., 1] = raw.transpose(0, 2, 3, 1) & 0x0F
    local = local.reshape(np_ * kp, 8, 128)
    # Q6_Vb_vshuff_Vb: within 128 bytes, out[2i] = in[i] and out[2i+1] = in[64+i].
    shuffled = np.empty_like(local)
    shuffled[:, :, 0::2] = local[:, :, :64]
    shuffled[:, :, 1::2] = local[:, :, 64:]
    # Q6_Vh_vasl_VhR(high, 4) | low: the shift is a halfword one, so the byte
    # that ends up in a byte's high nibble is the low nibble of the byte before
    # it, which is why this cannot be an element-wise shift.
    pairs = np.ascontiguousarray(shuffled.reshape(np_ * kp, 4, 2, 128)).view("<u2")
    low = np.ascontiguousarray(pairs[:, :, 0, :])
    high = np.ascontiguousarray(pairs[:, :, 1, :])
    shifted = ((high.astype(np.uint32) << 4) & 0xFFFF).astype("<u2")
    tiles = (
        (low.view(np.uint8) | shifted.view(np.uint8)).reshape(np_ * kp, 512).tobytes()
    )
    if scale_block_num == 1:
        scale_tail = scales.astype(np.float16)
    else:
        # The non-vrmpy kernel addresses each output channel's record first,
        # then its K blocks; every record is one aligned 64-fp16 vector.
        records = np.zeros((n // 32, scale_block_num, 64), dtype=np.float16)
        values = scales.reshape(n // 32, 32, scale_block_num).transpose(0, 2, 1)
        records[:, :, 0::2] = values.astype(np.float16)
        records[:, :, 1::2] = records[:, :, 0::2]
        scale_tail = records.reshape(-1)
    return tiles + scale_tail.tobytes()


def _quantized_weight_values(ctx, quantized: QuantizedWeight, k: int, n: int):
    """The stored low-bit weight and its per-channel scales, as [k][n] and [n].

    Both emitters read the same two tensors and neither may let them through
    `weight_bytes`: that narrows a fp32 tensor to fp16 silently, which is right
    for a kernel that reads halfs and wrong for one that reads packed int4.
    """
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
    return weight, scale


def _prefill_activation_ref(node, ctx, activation, m: int, k: int) -> TensorRef:
    """The activation in the layout the prefill kernel's DMA reads.

    The kernel loads a band of M rows per 32-element k tile out of
    `[k/64][m][64]`: one descriptor per k tile takes 64 bytes from row m at
    `(k/2)*m*64 + m*64` (`matmul_q4fp16_mle32.c:528-546`,
    `matmul_q4fp16.c:182-196`). That layout is one blit region -- the k tile is
    the outer axis, whose source stride is the 64 elements the tile covers --
    and for K == 64 it is the row-major tensor itself, so no copy is emitted.
    """
    source = ctx.operand(activation)
    if k == 64:
        return source
    packed = ctx.builder.add_activation(m * k * FP16_BYTES)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[source],
            outputs=[packed],
            params=[1, FP16_BYTES, 1]
            + [0, 0, 0, k // 64, m, 64, 64, k, 1, m * 64, 64, 1],
        ),
    )
    return packed


def _prefill_output_blit(node, ctx, blocked: TensorRef, out: TensorRef, m: int, n: int):
    """Move the kernel's `[n/64][m][64]` result back to a `[m][n]` row.

    The kernel writes its output in 64-channel packs, each of which holds M rows
    of 64 channels (`matmul_q4fp16.c:634`, `matmul_q4fp16_mle32.c:69`), and each
    row of a pack is what this moves. A ragged last pack -- N % 64 == 32 -- is a
    second region, and both fit in one command's parameter block.
    """
    packs, tail = n // 64, n % 64
    regions: List[int] = []
    if packs:
        regions += [0, 0, 0, packs, m, 64, m * 64, 64, 1, 64, n, 1]
    if tail:
        regions += [0, packs * m * 64, packs * 64, 1, m, tail, 1, 64, 1, 1, n, 1]
    if not regions:
        return
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[blocked],
            outputs=[out],
            params=[len(regions) // BLIT_REGION_INTS, FP16_BYTES, 1] + regions,
        ),
    )


def _emit_w8a16_prefill(
    node: torch.fx.Node,
    ctx,
    activation: torch.fx.Node,
    weight,
    scale,
    bias_ref: TensorRef,
    out: TensorRef,
    m: int,
    k: int,
    n: int,
) -> TensorRef:
    """A W8A16 matmul with M > 1 as command 42 plus its layout blits."""
    packed_weight = pack_w8a16_prefill_weight(weight, scale, k, n)
    packed_activation = _prefill_activation_ref(node, ctx, activation, m, k)
    packs = (n + 63) // 64
    blocked = ctx.builder.add_activation(packs * m * 64 * FP16_BYTES)
    output_bytes = packs * m * 64 * FP16_BYTES
    params = [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        k // 4,
        k // 32,
        m,
        m,
        m,
        1,
        m * 64,
        64,
        64,
        m * packs * 64,
        k,
        k,
        n,
        1,
        1,
        0,
        0,
        1,
        output_bytes,
        1,
        0,
    ]
    ctx.emit(
        node,
        Op(
            type=DSP_OP_MATMUL_W8A16_BLOCK_FP16,
            inputs=[
                packed_activation,
                ctx.builder.add_weights(packed_weight),
                bias_ref,
            ],
            outputs=[blocked],
            params=params,
        ),
    )
    _prefill_output_blit(node, ctx, blocked, out, m, n)
    return ctx.record(node, out)


def _emit_quantized_prefill(
    node: torch.fx.Node,
    ctx,
    activation: torch.fx.Node,
    weight,
    scale,
    bias_ref: TensorRef,
    out: TensorRef,
    m: int,
    k: int,
    n: int,
) -> TensorRef:
    """A quantized matmul with M > 1 as one DSP_OP_MATMUL_Q4A16_FP16 command.

    Three commands: the pack blit that puts the activation in the blocked layout
    (skipped when K is one block, where the layouts coincide), the matmul, and
    the repack blit that moves the kernel's 64-channel packs back to rows.

    What the kernel is given, and why: the weight is the tile order
    `pack_q4a16_prefill_weight` builds, which is what the vendored reorder for
    this kernel writes, and it carries fp16 per-channel scales where the GEMV
    weight carries fp32 -- the same quantizer output, two different kernel
    contracts. params carry M, K and N, the two chunk sizes, K/32 as the tile
    count the kernel validates, and 1 scale block per output channel, which is
    the granularity the quantizer produced.

    The output-channel chunk is two even where N is one tile of 32 channels,
    and that is a measurement rather than a leftover: `test_prefill_on_sim.py`
    runs this entry at `np_chunk == 1` and finds the kernel answers NaNs for
    every width at K == 64, while `np_chunk == 2` on the same shapes, walk and
    allocation order is exact. The vendored entry corrects an odd chunk count
    above one (`matmul_q4fp16_mle32.c:723`) and leaves one alone. Nothing about
    this emitter wants a single chunk -- the pool of chunks is what the kernel
    stages through, and one tile is one chunk of work either way -- so it asks
    for the pair in every case rather than for the shape the allocator would
    have picked.
    """
    import numpy as np

    if n <= 0 or scale.size % n:
        raise RuntimeError("hexagon: q4a16 prefill scales do not divide by output channels")
    scale_block_num = scale.size // n
    if scale_block_num < 1 or (k // 32) % scale_block_num:
        raise RuntimeError("hexagon: q4a16 prefill scale blocks do not divide K/32")
    packed_weight = pack_q4a16_prefill_weight(weight, scale, k, n, scale_block_num)
    packed_activation = _prefill_activation_ref(node, ctx, activation, m, k)
    packs = (n + 63) // 64
    blocked = ctx.builder.add_activation(packs * m * 64 * FP16_BYTES)
    np_chunk = PREFILL_OUTPUT_CHANNEL_CHUNK
    ctx.emit(
        node,
        Op(
            type=DSP_OP_MATMUL_Q4A16_FP16,
            inputs=[
                packed_activation,
                ctx.builder.add_weights(packed_weight),
                bias_ref,
            ],
            outputs=[blocked],
            params=[m, k, n, 0, 1, 1, np_chunk, k // 32, scale_block_num, 0],
        ),
    )
    _prefill_output_blit(node, ctx, blocked, out, m, n)
    return ctx.record(node, out)


def _emit_quantized_matmul(
    node: torch.fx.Node,
    ctx,
    quantized: QuantizedWeight,
    activation: torch.fx.Node,
    bias: Optional[torch.fx.Node],
) -> TensorRef:
    """A quantized matmul as one GEMV command, or as the three a prefill takes.

    The weight is packed at export into the layout the kernel reads and stored
    in the weights section. The M == 1 entry reads the graph's own fp16 row, K
    contiguous halfs, and quantizes it to int8 per token itself; the M > 1 entry
    reads fp16 activation bands in the blocked layout, which is what the extra
    blits are for.

    The M == 1 branch is unchanged and must stay that way: its commands are what
    the decode path has been running, and a byte of difference there is a
    regression even where the numbers agree.
    """
    import numpy as np

    geometry = quantized_matmul_geometry(activation, quantized)
    if geometry is None:
        raise RuntimeError("hexagon: quantized matmul has no static geometry")
    m, k, n = geometry
    if not quantized_matmul_is_emittable(
        node, lambda operand: ctx.constant_value(operand) is not None
    ):
        raise RuntimeError(
            f"hexagon: quantized matmul {m}x{k}x{n} is not one the support check "
            "admits; the partitioner should not have delegated it"
        )

    weight, scale = _quantized_weight_values(ctx, quantized, k, n)
    scale_block_num = 1 if quantized.bits == 8 or m == 1 else scale.size // n
    if scale.size not in (n, n * scale_block_num):
        raise RuntimeError("hexagon: quantized matmul has an unsupported scale shape")
    if m == 1 and scale_block_num != 1:
        raise RuntimeError("hexagon: q4a16 GEMV does not consume block-wise scales")
    if quantized.bits == 8 and scale_block_num != 1:
        raise RuntimeError("hexagon: w8a16 prefill does not consume block-wise scales")
    out = ctx.result_for(node, n)
    bias_ref = ABSENT if bias is None else ctx.operand(bias)
    if m > 1:
        if quantized.bits == 8:
            return _emit_w8a16_prefill(
                node, ctx, activation, weight, scale, bias_ref, out, m, k, n
            )
        return _emit_quantized_prefill(
            node, ctx, activation, weight, scale, bias_ref, out, m, k, n
        )

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


def _emit_cumsum(node: torch.fx.Node, ctx) -> TensorRef:
    """The fused scan as a batched matmul, plus the add that carries a chunk.

    The mask is the upper-triangular ones: column j of the weight selects
    inputs 0..j, so out[r, j] is the prefix ending at j. The lower triangle
    would select j..L-1 and sum the suffix instead, which is the scan read
    backwards.

    The mask is a host constant here rather than a graph node, so it goes
    through the same packing rule a matmul's constant weight does: the tile
    order when the DSP's general kernel will take the region, row-major bytes
    otherwise. The packed form is only readable by that kernel, so packing
    unconditionally would hand the fallbacks a matrix in the wrong order.
    """
    x, carry = node.args[0], node.args[1] if len(node.args) > 1 else None
    _require_arena_dtype(node, "cumsum")
    x_val = x.meta["val"]
    if not x_val.is_contiguous():
        raise RuntimeError(
            "hexagon: cumsum input must be contiguous; strides come from shape"
        )
    length = x_val.shape[-1]
    rows = _upper_product(tuple(x_val.shape[:-1]), ctx)
    mask = torch.triu(torch.ones((length, length), dtype=torch.float16))
    mask_ref, prepacked = _mask_operand(ctx, node, mask, rows, length, length)

    out = ctx.result_for(node, rows * length)
    out_shape = tuple(node.meta["val"].shape)
    target = out if carry is None else ctx.activation_for_shape(out_shape)
    _matmul_command(
        node,
        ctx,
        ctx.operand(x),
        mask_ref,
        target,
        1,
        rows,
        length,
        length,
        hmx_prepacked=prepacked,
    )
    if carry is None:
        return ctx.record(node, out)
    return _emit_carry_add(node, ctx, target, carry, out, out_shape)


def _mask_operand(ctx, node, mask, m: int, k: int, n: int):
    """The (k, n) scan mask in the layout this region will be read in."""
    if (
        ctx.options.hmx_prepack
        and hmx_prefers_general(k, n)
        and hmx_general_eligible(m, k, n)
    ):
        return ctx.packed_weight(node, mask, k, n), True
    return ctx.folded_weight(node, mask), False


def _emit_carry_add(node, ctx, scanned, carry, out, shape):
    """The chunk boundary: the previous chunk's total, broadcast over the rows.

    This is the binary emitter's descriptor with the scan's result already in
    hand, so the add is one BINARY_ELEMENTWISE on a value the graph never
    named. The carry broadcasts the way the graph's own add would, through
    the same tail, so a carry of [..., 1] reaches every column and a carry of
    the full shape adds elementwise.
    """
    carry_numel = _upper_product(tuple(carry.meta["val"].shape), ctx)
    shape = ctx.upper_shape(shape)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_BINARY_ELEMENTWISE,
            inputs=[scanned, ctx.operand(carry)],
            outputs=[out],
            params=[
                _upper_product(shape, ctx),
                _upper_product(shape, ctx),
                carry_numel,
                BINARY_OP_TYPES["add"],
                FP16_BYTES,
                FP16_BYTES,
                0,  # inputs are not 4-byte floats
                0,  # output is not a 4-byte float
                *_broadcast_tail(shape, carry, shape, ctx),
            ],
        ),
    )
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
ADAPTIVE_AVG_POOL2D = exir_ops.edge.aten._adaptive_avg_pool2d.default
POOL_TARGETS = frozenset(
    {MAX_POOL2D, MAX_POOL2D_WITH_INDICES, AVG_POOL2D, ADAPTIVE_AVG_POOL2D}
)
MAX_POOL_TARGETS = frozenset({MAX_POOL2D, MAX_POOL2D_WITH_INDICES})


def max_pool_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The max pool a getitem reads, when it reads the values and not the indices.

    max_pool2d_with_indices returns (values, indices); the kernel produces the
    values, so getitem 0 is the only reader a partition can carry.
    """
    return _values_getitem(node, MAX_POOL2D_WITH_INDICES)


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


def max_dim_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The max-over-a-dim a getitem reads, when it reads the values."""
    return _values_getitem(node, MAX_DIM)


def min_dim_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    return _values_getitem(node, MIN_DIM)


def min_dim_is_emittable(node: torch.fx.Node) -> bool:
    if node.target is not MIN_DIM:
        return True
    return bool(node.users) and all(min_dim_getitem(reader) is node for reader in node.users)


def max_dim_is_emittable(node: torch.fx.Node) -> bool:
    """Whether every reader of this max takes the values.

    Same shape of node and the same rule as the pool: indices are positions, and
    the reduction kernel here computes values, so a graph that reads them keeps
    the whole node -- values reader included -- on the portable kernels.
    """
    if node.target is not MAX_DIM:
        return True
    return bool(node.users) and all(
        max_dim_getitem(reader) is node for reader in node.users
    )


def _values_getitem(node: torch.fx.Node, source_target) -> Optional[torch.fx.Node]:
    """The two-output op a ``getitem 0`` reads, if that is what this node is."""
    if node.target is not GETITEM or len(node.args) != 2:
        return None
    source, index = node.args
    if index != 0 or not isinstance(source, torch.fx.Node):
        return None
    return source if source.target is source_target else None


def _values_sink(node: torch.fx.Node) -> torch.fx.Node:
    """The node whose output slot a two-output op's values end up in.

    max_pool2d_with_indices, max.dim and topk all hand their values on through a
    getitem, and that getitem is what downstream reads -- so it is the sink the
    emitter has to fill, not the tuple the kernel cannot describe.
    """
    return next(
        (
            reader
            for reader in node.users
            if max_pool_getitem(reader) is node
            or max_dim_getitem(reader) is node
            or min_dim_getitem(reader) is node
            or topk_getitem(reader) is node
        ),
        node,
    )


def operand_dtypes_are_readable(node: torch.fx.Node) -> bool:
    """Whether every tensor operand is a width the arena holds.

    A bool is one byte per element in the `.pte` and two in the arena's
    arithmetic, so a kernel handed one reads the neighbouring slot's bytes as
    the value: no error, just the wrong numbers. The case is reachable from an
    ordinary `x + (a > b)`, whose comparison is portable for a dtype reason of
    its own, so the consumer is refused here rather than left to read it.

    SELECT is the exception, and it is an exception about the *slot* rather than
    about the value: an input's slot is the size the blob declares for it, so a
    bool operand of a `where` is copied in at one byte per element and read back
    at that width by the one kernel that asks for it. Every other target here
    reads two bytes, so for those the rule stands unchanged.
    """
    if node.target in WHERE_TARGETS:
        return True
    for arg in node.args:
        value = arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None
        if isinstance(value, torch.Tensor) and value.dtype is torch.bool:
            return False
    return True


class PoolSpec(NamedTuple):
    """Everything the pool command carries, once the node is known to fit."""

    batch: int
    channels: int
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


def _pool_output_extent(
    size: int, kernel: int, stride: int, pad: int, ceil_mode: bool
) -> int:
    """torch's output size along one axis, in torch's own arithmetic.

    Floor mode is `(size + 2 * pad - kernel) // stride + 1`. Ceil mode adds
    `stride - 1` to the numerator and then drops the last output position once
    when it would start at or past `size + pad`
    (`pooling_output_shape_pad_lr`), so a ceil window may hang off the far edge
    but the last one always starts inside the padded input. That decrement is
    the whole reason a ceil window is safe for this kernel: it is what stops the
    last position being a window over nothing, which `_pool_window_intersects`
    exists to rule out.
    """
    output = (size + 2 * pad - kernel + (stride - 1 if ceil_mode else 0)) // stride + 1
    if ceil_mode and (output - 1) * stride >= size + pad:
        output -= 1
    return output


def _pool_kernel_divisor_holds(
    ih: int,
    iw: int,
    oh: int,
    ow: int,
    kernel_y: int,
    kernel_x: int,
    stride_y: int,
    stride_x: int,
    pad_y: int,
    pad_x: int,
) -> bool:
    """Whether every window still holds a whole kernel of the padded input.

    `count_include_pad` divides by the window clipped to the **padded** input,
    so the divisor is `kY * kX` for every window except one that hangs off the
    padded edge, where it is the part that is still inside. The kernel's
    countType 1 divides by `kY * kX` unconditionally
    (`pool_fp16.c:74-79`), so it is that window's divisor only when nothing
    hangs off. A window's clipped length is `min(k, size + 2 * pad - o * stride)`,
    which falls as `o` grows, so the last output position is the tightest on
    each axis and the test is two comparisons rather than a walk.

    Floor mode satisfies this by construction -- `(oh - 1) * stride <= ih + 2 *
    pad - kernel` is the floor formula -- which is why the divisor only becomes
    a question once ceil mode can push a window off the padded edge.
    """
    return (
        ih + 2 * pad_y - (oh - 1) * stride_y >= kernel_y
        and iw + 2 * pad_x - (ow - 1) * stride_x >= kernel_x
    )


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
    # The kernel reads its activation as [ceil(C/64)][batch][h*w][64] blocks
    # (pool_fp16.c:22), which is not the row-major [batch][C][h*w] the arena
    # holds, and it writes the same blocked layout. The channel axis is a
    # packing granularity rather than a shape the kernel needs: `c4` is the
    # number of 64-lane blocks and the walk already loops over all of them
    # (pool_fp16.c:21), so any channel count is a number of blocks and the
    # emitter's job is to describe the rearranged grid, not to reshape the
    # command. A ragged last block costs a narrower blit region and a zeroed
    # tail, because the pack writes only the channels the tensor has while the
    # kernel reads whole 64-lane vectors.
    if channels <= 0:
        return None

    if node.target is ADAPTIVE_AVG_POOL2D:
        output = _int_pair(_pool_arg(node, "output_size", 1, None), None)
        if output is None or any(extent <= 0 for extent in output):
            return None
        oh, ow = result.shape[-2], result.shape[-1]
        if output != (oh, ow) or ih % oh or iw % ow:
            # An adaptive window is fixed exactly when both input extents are
            # integer multiples of the output extents. Otherwise its window
            # changes, or clamps, from output position to output position and
            # one pool command cannot describe it.
            return None
        kernel = (ih // oh, iw // ow)
        return PoolSpec(
            batch=batch,
            channels=channels,
            ih=ih,
            iw=iw,
            oh=oh,
            ow=ow,
            kernel_y=kernel[0],
            kernel_x=kernel[1],
            stride_y=kernel[0],
            stride_x=kernel[1],
            pad_y=0,
            pad_x=0,
            count_type=POOL_COUNT_KERNEL,
            pool_type=POOL_AVERAGE,
        )

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

    oh, ow = result.shape[-2], result.shape[-1]
    # torch's output size on each axis, which is also the one the emitter is
    # about to describe with strides. ceil mode is this command's own geometry,
    # not a second one: oh and ow are parameters (pool_fp16.c:9) and the window
    # at each position is [oy * strideY - padY, +kernelY) with whatever falls
    # outside skipped (pool_fp16.c:26-43), which is the clip torch does.
    if oh != _pool_output_extent(ih, kernel[0], stride[0], padding[0], ceil_mode):
        return None
    if ow != _pool_output_extent(iw, kernel[1], stride[1], padding[1], ceil_mode):
        return None
    if not _pool_window_intersects(ih, oh, kernel[0], stride[0], padding[0]):
        return None
    if not _pool_window_intersects(iw, ow, kernel[1], stride[1], padding[1]):
        return None
    if (
        pool_type == POOL_AVERAGE
        and count_type == POOL_COUNT_KERNEL
        and not _pool_kernel_divisor_holds(
            ih,
            iw,
            oh,
            ow,
            kernel[0],
            kernel[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
        )
    ):
        # An average over a window that hangs off the padded input divides by
        # what is left of it, and the command has no divisor that is not
        # kY * kX. count_include_pad=False divides by the window clipped to the
        # raw input, which is the kernel's validCount, so that one takes ceil
        # mode at any geometry.
        return None

    return PoolSpec(
        batch=batch,
        channels=channels,
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


def _emit_pool2d(node: torch.fx.Node, ctx) -> TensorRef:
    """Fixed-window max, average, and adaptive average as one POOL2D_FP16.

    The kernel reads and writes its activation in the DSP's 64-channel blocked
    layout, so the row-major buffer the arena holds has to be rearranged into it
    before the window walk and back out after. That is two blits around the
    command, and they are not optional: reading the blocked layout's
    ``(y * width + x) * 64`` step over a row-major buffer would return other
    channels' values at every position. A spatial extent of one over whole blocks
    is the one case where the two layouts agree element for element, and the
    blit is dropped.

    The channel axis is a count of blocks and not a shape the kernel needs: the
    command carries `ceil(C / 64)` in its `c4` slot and `hvx_pool2d_fp16` walks
    every one of them, so a wide pool is this same command with a larger `c4`
    and one more blit region per block. Nothing here widens the parameter
    budget: `c4` was already one of the fifteen ints the pool command carries,
    and a blit region is twelve of the forty `kMaxOpParams` leaves after the
    three-int header, so three blocks move per command at any width.
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
    sink = _values_sink(node)
    out = ctx.result_for(sink, _numel(node))
    area = spec.ih * spec.iw
    out_area = spec.oh * spec.ow
    channels = spec.channels
    blocks = _channel_blocks(channels)

    packed_in = source
    if not _conv_layouts_agree(spec.batch, area, channels):
        packed_in = ctx.builder.add_activation(
            spec.batch * area * blocks * POOL_CHANNEL_BLOCK * FP16_BYTES
        )
        if channels % POOL_CHANNEL_BLOCK:
            # The kernel loads whole 64-lane vectors and the pack blit writes
            # only the channels the tensor has, so the tail lanes of the last
            # block are whatever the arena last held. Each lane pools
            # independently of every other, so those lanes cannot reach a real
            # channel and the unpack never reads them -- but an arena block is
            # memset once, on the first allocation of its size, and reused after
            # that (runtime/hexagon_driver.cpp, SharedArenaPool::Acquire), so
            # what is there is the previous run rather than a known value.
            _emit_zero(ctx, node, packed_in)
        _emit_channel_block_blit(
            ctx, node, source, packed_in, spec.batch, area, channels, True
        )

    pooled = (
        out
        if _conv_layouts_agree(spec.batch, out_area, channels)
        else ctx.builder.add_activation(
            spec.batch * out_area * blocks * POOL_CHANNEL_BLOCK * FP16_BYTES
        )
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
                blocks,
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
    if not _conv_layouts_agree(spec.batch, out_area, channels):
        _emit_channel_block_blit(
            ctx, node, pooled, out, spec.batch, out_area, channels, False
        )
    return ctx.record(node, out)


# The convolution family. One ATen op carries all of it -- conv2d is the form
# torch.export writes for nn.Conv2d, and convolution is the one its own rewrites
# produce, with transposed/output_padding/benchmark arguments it adds -- and the
# two kernels behind it split on groups: groups == in_channels == out_channels
# is the per-channel walk MobileNet's depthwise layers are, and any other grouped
# convolution is partitioned below into one dense walk per group.
#
# Both read and write their activation in the same 64-channel blocking pooling
# uses, so both are wrapped in the same pair of blits, and the general path
# wants its weight in the HMX unit's 32x32 tiles, which is an export-time
# rearrange of the same kind pack_hmx_weight does for a matmul.
CONV1D = exir_ops.edge.aten.conv1d.default
CONV2D = exir_ops.edge.aten.conv2d.default
CONV3D = exir_ops.edge.aten.conv3d.default
CONVOLUTION = exir_ops.edge.aten.convolution.default
CONV_TARGETS = frozenset({CONV1D, CONV2D, CONV3D, CONVOLUTION})

# What the im2col kernel asks VTCM for before it runs anything
# (im2col_convolution_fp16.cc:1783-1786): one weight staging buffer per channel
# tile, one activation staging buffer per position tile, an accumulator block and
# a scale block. The emitter pins mp = 1 and np = 2, so the staging request is
# three tiles' worth of kp 1024-element planes, and the fixed part is the
# accumulator plus the scales. vtcm_seq_alloc rounds each of the four up to 128
# bytes, which is what the slack covers.
CONV_VTCM_BYTES = 8192 * 1024
CONV_VTCM_STAGING_TILES = 3
CONV_VTCM_FIXED_BYTES = 4096 + 256 + 4 * 128


class ConvSpec(NamedTuple):
    """Everything a convolution command carries, once the node is known to fit."""

    batch: int
    in_channels: int
    in_h: int
    in_w: int
    out_channels: int
    out_h: int
    out_w: int
    kernel_y: int
    kernel_x: int
    stride_y: int
    stride_x: int
    pad_y: int
    pad_x: int
    dilate_y: int
    dilate_x: int
    depthwise: bool
    # A transposed convolution reaches the same two commands through the
    # identity
    #
    #   conv_transpose(x, w, s, p, op) ==
    #       conv2d(zero_insert(x, s, op), flip(w).transpose(ic, oc),
    #               d * (K - 1) - p, dilation=d)
    #
    # so everything above is the *convolution's* geometry: stride is 1,
    # pad_y/pad_x are the remapped padding, and in_h/in_w are the upsampled
    # extents the kernel actually walks. Dilation stays on the DSP because its
    # im2col kernel reads dilateX and dilateY when it gathers each tap. The
    # `transposed` flag says the weight needs the flip; `upsample_*` and
    # `tail_*` describe the zero-insert blit. All four are the identity
    # (1, 1, 0, 0) for a plain convolution, which needs no blit.
    transposed: bool = False
    upsample_y: int = 1
    upsample_x: int = 1
    tail_y: int = 0
    tail_x: int = 0
    # A grouped convolution is lowered as one dense walk per group. The DSP
    # command has no group field; the host partitions the channels and gives
    # each command the group-local in/out counts. One is the identity.
    groups: int = 1
    # The operand's height held the bare run-time symbol, so every extent below
    # is the traced example's and the commands carry a patch the runtime
    # resolves from the length it was handed. See `_patch_conv_extents`.
    dynamic_h: bool = False
    # The graph spelled the window over one axis, and this is the same command
    # read from the 4-D spelling `(B, C, T, 1)` with a `(k, 1)` window: the
    # graph's own batch stays the batch, the time axis is the height, the width
    # is one, and the weight is the `(O, I, k)` one with a unit column
    # appended. The buffer is the same bytes and the kernel the same walk;
    # `conv_spec` admits the 3-D spelling and `pack_conv_weight` reads the
    # weight for it.
    conv1d: bool = False


def conv_1x1_direct_applies(spec: ConvSpec) -> bool:
    """Whether a convolution should be emitted as CONV1X1_DIRECT_FP16(17).

    `htp_ops_conv1x1_direct_fp16` is a second name for the im2col function
    (im2col_convolution_fp16.cc:1840), so the command type does not choose a
    kernel; it records that the function's own activation fill has a 1x1 path
    this geometry takes. The fill is chosen from the parameters the command
    carries, in the C (fill_im2col_activation_tiles, :1593): packCUnit 64,
    kernelX == kernelY == 1, unit dilation, and kp == ceil(ic / 32), which the
    emitter's own kp (kernel_y * kernel_x * conv_k_units(ic)) only equals when
    the in-channel count is a whole number of 32-channel blocks. Inside that
    fill (fill_im2col_activation_1x1_pack64_tiles, :723) a direct copy of the
    plane is taken when the output plane is the input plane (batch 1, no
    padding, unit stride: out == in), and a strided direct gather when the
    stride is not 1 with no padding and batch 1. Everything else -- a batch
    above one, padding, a stride of 1 with an output plane that is not the
    input plane -- falls through to the general per-position window walk,
    which for a 1x1 window is the im2col materialisation the fast path exists
    to avoid, and that is what CONV1X1_DIRECT_FP16 would then be claiming in
    the stream. This returns False there, so the command stays
    IM2COL_CONVOLUTION_FP16(12) and the blob does not name a fast path that
    will not run.

    A transposed convolution never gets here (it reaches the emitter with the
    1x1 window spelled after the zero-insert, and the fast path assumes the
    packed plane the blit produced), so it is refused explicitly rather than
    by accident.
    """
    if spec.transposed or spec.depthwise or spec.upsample_y != 1 or spec.upsample_x != 1:
        return False
    if (spec.kernel_x, spec.kernel_y) != (1, 1) or spec.in_channels % 32 != 0:
        return False
    if (spec.dilate_x, spec.dilate_y) != (1, 1) or (spec.pad_x, spec.pad_y) != (0, 0):
        return False
    if (spec.stride_x, spec.stride_y) == (1, 1):
        return (
            spec.batch == 1
            and spec.out_h == spec.in_h
            and spec.out_w == spec.in_w
        )
    return spec.batch == 1


def _zero_insert_regions(source_shape, spec: ConvSpec) -> List[int]:
    """The raster region that scatters an input plane into an interleaved one.

    Zero-interleaving is one strided box: the source reads at stride 1 along a
    row and the destination writes at the interleave factor, with the row and
    plane strides of each layout. `source_shape` is the tensor's own
    ``(N, C, H, W)``, which is what the region's extents describe even though the
    destination is the taller plane.

    `htp_ops_raster_blit` tries its fast paths before its own region walk, and
    `htp_ops_try_interleave_c64_single_blit` (src/dsp/blit_ops.cc:1551) claims
    any region with ``size[2] == 16 && srcStride[2] == 1 && dstStride[2] == 4``,
    which a width of 16 interleaved by 4 is. That is not a hazard here: the
    guard fixes exactly those three numbers and the body writes ``dst[4 * i] =
    src[i]`` over the same row and plane grid the walk uses, so on every region
    it accepts it writes that region's own mapping -- the hardcoded offsets are
    the strides the guard matched on. test_zero_insert_sim.py measures that on
    the simulator, and measures a mutant region that comes out wrong.
    """
    source_h, source_w = source_shape[2], source_shape[3]
    return [
        0,
        0,
        0,  # source index and both offsets
        spec.batch * spec.in_channels,
        source_h,
        source_w,
        source_h * source_w,
        source_w,
        1,  # source strides: plane, row, element
        spec.in_h * spec.in_w,
        spec.upsample_y * spec.in_w,
        spec.upsample_x,  # destination: plane, row, interleave
    ]


def _conv_spatial_pair(value, rank: int, default: tuple) -> Optional[tuple]:
    if value is None:
        return default
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        return None
    if rank == 3:
        return (1, values[0]) if len(values) == 1 else None
    if rank == 4:
        return (values[0], values[1]) if len(values) == 2 else None
    if rank == 5 and len(values) == 3:
        return values
    return None

def _is_one_axis_geometry(args, weight) -> bool:
    """Whether this node's geometry is the one-axis form, read from the weight.

    A one-axis graph carries a 3-D weight -- (O, I, k) or (C, 1, k) for the
    depthwise form -- and a 2-D graph a 4-D one, whatever the input's rank
    turns out to be. Reading the weight here, rather than the input, keeps the
    stride and padding below, which are one entry long on this form, from being
    read as a 2-D pair first.
    """
    if len(args) < 6:
        return False
    value = weight.meta.get("val") if isinstance(weight, torch.fx.Node) else None
    return isinstance(value, torch.Tensor) and value.dim() == 3


def _shape_only(val: torch.Tensor, shape) -> torch.Tensor:
    """A val carrying `shape` and the dtype it came with, and nothing else.

    Only the shape and dtype are ever read back, so a meta tensor of the kind
    `new_empty` produces is enough. It is written this way rather than as
    `unsqueeze` because a fake tensor's unsqueeze adds a dim of its own
    (torch 2.14) where a unit dim is meant, and a symbolic batch would not be
    preserved in any case; a meta tensor keeps both.
    """
    return val.new_empty(shape)


def conv_spec(node: torch.fx.Node, is_constant) -> Optional[ConvSpec]:
    """The convolution's geometry, or None when neither kernel here can run it.

    The support check and the emitter both call this, so they cannot disagree
    about which convolutions are delegated; `is_constant` is the caller's own
    test for "a value whose bytes I can read now", since both kernels take a
    weight this layer has to rearrange before the DSP ever sees it.

    A height that is the run-time symbol is admitted, and everything below is
    then the traced example's extent: the commands carry the example's numbers
    plus a patch record per affected param, which is what
    `_patch_conv_extents` writes. Three things keep that honest -- the height has
    to be the symbol itself rather than an expression over it, the geometry has
    to be affine in the height (one row per output row, and padding that leaves
    the height alone), and the result's height has to be that same dim -- so the
    extent the patch recomputes is the one torch's own shape rule gives.

    Two graphs call this and they do not look the same. The partitioner sees the
    edge graph, where the symbol is still a symbol; the emitter sees the
    delegate's subgraph, where EXIR has already folded every symbol to the traced
    example, so a dim that moves prints as its number and `extent_kind` can only
    say that it is still a `torch.SymInt`. The expression test therefore holds
    only on the partitioner's side, which is the side that decides: a node whose
    height is a real expression is refused there and never reaches an emitter.
    """
    if node.target not in CONV_TARGETS:
        return None
    args = node.args
    # conv2d is the form torch.export produces for nn.Conv2d and convolution is
    # the one every rewrite of it lands on; the two differ only in the arguments
    # the emitter refuses anyway.
    convolve = node.target is CONVOLUTION
    if len(args) < (9 if convolve else 7):
        return None
    source, weight = args[0], args[1]
    if not (isinstance(source, torch.fx.Node) and isinstance(weight, torch.fx.Node)):
        return None
    if not is_constant(weight):
        return None
    bias = args[2] if len(args) > 2 else None
    if bias is not None and not (isinstance(bias, torch.fx.Node) and is_constant(bias)):
        return None
    value = source.meta.get("val")
    kernel = weight.meta.get("val")
    result = node.meta.get("val")
    if not all(isinstance(item, torch.Tensor) for item in (value, kernel, result)):
        return None
    rank = value.dim()
    if rank != kernel.dim() or rank != result.dim():
        return None
    group = args[8] if convolve else args[6]
    stride_3d = _conv_spatial_pair(args[3], rank, (1, 1, 1))
    padding_3d = _conv_spatial_pair(args[4], rank, (0, 0, 0))
    dilation_3d = _conv_spatial_pair(args[5], rank, (1, 1, 1))
    # A one-axis convolution carries one number where the 2-D form carries a
    # pair, and the pair's absent axis is the one the reading supplies: the
    # stride, padding and dilation along the time axis repeat onto the unit
    # width, and the output padding along it is zero. Every 2-D graph spells
    # these as a pair (or a bare int), so this is the 3-D spelling alone.
    rank1d = _is_one_axis_geometry(args, weight)
    stride = _int_pair(args[3], None)
    padding = _int_pair(args[4], None)
    dilation = _int_pair(args[5], (1, 1))
    transposed = False
    tail = (0, 0)
    if convolve:
        if not isinstance(args[6], bool):
            return None
        transposed = args[6]
        tail = (
            _int_pair(args[7], (0, 0))
            if rank in (3, 4)
            else (0, 0, 0)
            if rank == 5 and args[7] == [0, 0, 0]
            else None
        )
        if tail is None:
            return None
    if rank1d:
        if None in (stride, padding) or dilation is None or tail is None:
            return None
        # The axis the one-axis form has and the 2-D form does not is the
        # width, and the reading makes it one element wide with no movement
        # across it: a stride, a dilation and an output padding of 1, 1 and 0.
        # The lengths are read off the raw argument because _int_pair has
        # already collapsed a one-entry list to a pair by the time it is
        # called, so a two-entry list here is the 2-D spelling of a 3-D node
        # and is refused rather than silently collapsed.
        for raw in (args[3], args[4], args[5]):
            if isinstance(raw, (list, tuple)) and len(raw) != 1:
                return None
        if convolve and isinstance(args[7], (list, tuple)) and len(args[7]) > 1:
            return None
        stride = (stride[0], 1)
        padding = (padding[0], 0)
        dilation = (dilation[0], 1)
        tail = (tail[0], 0)
    if isinstance(group, bool) or not isinstance(group, int) or group <= 0:
        return None
    if stride_3d is None or padding_3d is None or dilation_3d is None:
        return None
    if any(step <= 0 for step in stride_3d) or any(step <= 0 for step in dilation_3d):
        return None
    if any(pad < 0 for pad in padding_3d):
        return None
    if any(extra < 0 for extra in tail):
        return None

    if value.dtype not in (torch.float16, torch.float32):
        return None
    conv1d = False
    if value.dim() == 3 and kernel.dim() == 3 and result.dim() == 3:
        conv1d = True
        # A one-axis convolution, read from the 4-D spelling this kernel walks:
        # a (B, C, T) run is a (B, C, T, 1) plane, so the graph's own batch is
        # the command's batch, the time axis is the height, and the width is
        # one. The weight (O, I, k) is the same bytes as an (O, I, k, 1) one.
        # Every gate below runs on that reading, so the stride, padding,
        # dilation and group rules are the 2-D ones unchanged, and the emitter
        # sees the same `conv1d` flag and packs the weight with the column
        # appended.
        # The extents are written out by axis rather than by unsqueezing the
        # val, because a fake tensor's unsqueeze is not a unit dim here (it
        # inserts one of its own under torch 2.14) and a view cannot add one
        # either. This is the reading torch's own conv2d over x.unsqueeze(-1)
        # computes, and _emit_convolution reads the spec rather than the
        # graph, so the two cannot disagree about the shape.
        value = _shape_only(value, tuple(value.shape) + (1,))
        kernel = _shape_only(kernel, tuple(kernel.shape) + (1,))
        result = _shape_only(result, tuple(result.shape) + (1,))
        # From here the node is a 4-D one, so the rank-keyed pairs are the
        # one-axis ones the block above fixed: the single entry repeats onto
        # the unit width, not onto a unit height.
        rank = 4
        stride_3d, padding_3d, dilation_3d = stride, padding, dilation
    elif value.dim() == 5 and kernel.dim() == 5 and result.dim() == 5:
        # The 5-D volume form, which the conv3d work reads directly below.
        pass
    elif value.dim() != 4 or kernel.dim() != 4 or result.dim() != 4:
        return None
    if conv1d and transposed:
        # The identity a transposed convolution relies on has never been
        # measured with a one-axis window, and the transposed weight is indexed
        # the other way round to begin with, so the 3-D spelling stays refused
        # rather than read as a degenerate 2-D one.
        return None
    if rank == 4:
        stride, padding, dilation = stride_3d, padding_3d, dilation_3d
        kernel_y, kernel_x = (int(dim) for dim in kernel.shape[2:])
    # Only the operand's height may hold the run-time length. A batch, a channel
    # count or the window's width that moved would each need its own patch over
    # the arena's own geometry, and the weight is a constant whose extents are
    # the export's. A height that is an expression over the symbol is the
    # stride-2 case, which no affine patch can rebuild.
    if any(isinstance(dim, torch.SymInt) for dim in kernel.shape):
        return None
    value_kinds = [extent_kind(dim) for dim in value.shape]
    result_kinds = [extent_kind(dim) for dim in result.shape]
    if any(kind == DERIVED_DIM for kind in value_kinds + result_kinds):
        return None
    if rank in (3, 5):
        if any(kind != STATIC_DIM for kind in value_kinds + result_kinds):
            return None
        if transposed:
            return None
        dynamic_h = False
    else:
        if any(kind != STATIC_DIM for kind in value_kinds[0:2] + value_kinds[3:]):
            return None
        if any(kind != STATIC_DIM for kind in result_kinds[0:2] + result_kinds[3:]):
            return None
        dynamic_h = value_kinds[2] in (RUNTIME_DIM, SPECIALIZED_DIM)
    if value_kinds[2] == SPECIALIZED_DIM and result_kinds[2] != SPECIALIZED_DIM:
        return None
    if dynamic_h and transposed:
        # A transposed convolution's height is the interleaved one, and the patch
        # this path writes describes the affine relation a plain convolution's
        # height has to the run length. Neither of the two was measured against
        # the other, so the combination stays on the portable kernels.
        return None
    if rank == 3:
        batch, in_channels, in_w = (int(dim) for dim in value.shape)
        in_h = 1
        kernel_y, kernel_x = 1, int(kernel.shape[2])
        stride, padding, dilation = stride_3d, padding_3d, dilation_3d
        stride, padding, dilation = (1, stride[0]), (0, padding[0]), (1, dilation[0])
    elif rank == 5:
        spatial_in = [int(dim) for dim in value.shape[2:]]
        spatial_result = [int(dim) for dim in result.shape[2:]]
        spatial_kernel = [int(dim) for dim in kernel.shape[2:]]
        if spatial_kernel[0] == spatial_kernel[1] == 1:
            if (
                spatial_in[0] != spatial_result[0]
                or spatial_in[1] != spatial_result[1]
                or spatial_in[0] != 1
                or spatial_in[1] != 1
                or stride_3d[0] != 1
                or stride_3d[1] != 1
                or padding_3d[0] != 0
                or padding_3d[1] != 0
                or dilation_3d[0] != 1
                or dilation_3d[1] != 1
            ):
                return None
            in_h, in_w = 1, spatial_in[2]
            kernel_y, kernel_x = 1, spatial_kernel[2]
            stride, padding, dilation = stride_3d[2], padding_3d[2], dilation_3d[2]
        else:
            singleton_axes = [
                axis for axis, size in enumerate(spatial_kernel) if size == 1
            ]
            dropped_axis = next(
                (
                    axis
                    for axis in singleton_axes
                    if spatial_in[axis] == spatial_result[axis] == 1
                    and stride_3d[axis] == 1
                    and padding_3d[axis] == 0
                    and dilation_3d[axis] == 1
                ),
                None,
            )
            if dropped_axis is None:
                return None
            keep_axes = [axis for axis in range(3) if axis != dropped_axis]
            in_h, in_w = (spatial_in[axis] for axis in keep_axes)
            kernel_y, kernel_x = (spatial_kernel[axis] for axis in keep_axes)
            stride = tuple(stride_3d[axis] for axis in keep_axes)
            padding = tuple(padding_3d[axis] for axis in keep_axes)
            dilation = tuple(dilation_3d[axis] for axis in keep_axes)
        stride, padding, dilation = (1, stride), (0, padding), (1, dilation)
        batch, in_channels = (int(dim) for dim in value.shape[:2])
    else:
        batch, in_channels, in_h, in_w = (int(dim) for dim in value.shape)
    if transposed:
        # A transposed convolution's weight is indexed the other way round: the
        # leading axis is the input's channels and the second the output's share
        # of the group.
        weight_in, per_group, kernel_y, kernel_x = (int(dim) for dim in kernel.shape)
        if weight_in != in_channels:
            return None
        out_channels = per_group * group
    else:
        out_channels, per_group = (int(dim) for dim in kernel.shape[:2])
        if per_group * group != in_channels:
            return None
    extents = (batch, in_channels, in_h, in_w, out_channels, kernel_y, kernel_x)
    if any(extent <= 0 for extent in extents):
        return None

    # A plain convolution walks its own stride and has nothing to interleave;
    # only a transposed one turns its stride into a zero-insert.
    upsample = stride if transposed else (1, 1)
    if transposed:
        # The kernel's dilated im2col walk is the same window walk as the
        # undilated one. Only the effective kernel extent changes, so remap
        # padding without expanding the kernel.
        conv_pad = (
            dilation[0] * (kernel_y - 1) - padding[0],
            dilation[1] * (kernel_x - 1) - padding[1],
        )
        if conv_pad[0] < 0 or conv_pad[1] < 0:
            return None
        # The interleaved plane reaches as far as the transposed window does,
        # plus whatever output_padding asks for at the far edge. The kernel then
        # walks it the way it walks any other input.
        in_h = (in_h - 1) * stride[0] + 1 + tail[0]
        in_w = (in_w - 1) * stride[1] + 1 + tail[1]
        stride = (1, 1)
        padding = conv_pad

    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_y - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_x - 1) - 1) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        return None
    expected_result = (
        [batch, out_channels, out_w]
        if rank == 3
        else [batch, out_channels, 1, out_h, out_w]
        if rank == 5
        else [batch, out_channels, out_h, out_w]
    )
    if [int(dim) for dim in result.shape] != expected_result:
        return None
    if dynamic_h and (stride[0] != 1 or 2 * padding[0] != dilation[0] * (kernel_y - 1)):
        # One output row per input row, and a window that starts on its own row:
        # then the output height is the input height, which is the one relation
        # an affine patch over the run length reproduces. A stride divides and a
        # lopsided padding shifts; both need the extent itself, not a multiple
        # of the length.
        return None
    if dynamic_h and value_kinds[2] == RUNTIME_DIM:
        if str(result.shape[2]) != str(value.shape[2]):
            # The height-preserving rule above says the result's height is the
            # operand's, so torch's own shape rule has to spell it as that same
            # symbol. Anything else is a relation this emitter has not seen.
            return None
    # The region stride of a channel block is the plane's element count, and the
    # DSP holds it in an int32. The interleaved plane is the one the kernel and
    # the blit both address, so it is the one this bounds.
    if in_channels * in_h * in_w >= 1 << 31:
        return None

    # A transposed convolution is never the depthwise walk: its weight is a
    # dense (in, out) pair even when both channel counts are one.
    depthwise = (
        not transposed and group == in_channels == out_channels and per_group == 1
    )
    if not depthwise and group != 1 and dynamic_h:
        # A grouped walk is lowered as one dense command per group, and each
        # command's plane extents and the group slice's own offsets are affine
        # in a run-time height only when every group sees the same share of the
        # plane. The per-group patch this would need is a different record per
        # command, so a grouped height that moves stays portable rather than
        # being emitted with the export's example extent baked in.
        return None
    if (
        not depthwise
        and conv_vtcm_bytes(
            kernel_y, kernel_x, in_channels // group if group > 1 else in_channels
        )
        > CONV_VTCM_BYTES
    ):
        # The staging buffers are sized from kp, so a wide enough window has
        # nowhere to put them. The budget is the one the simulator reports and
        # the arithmetic is the kernel's own; a device that has less VTCM than
        # its manager hands out is not something this can see from here.
        return None
    return ConvSpec(
        batch=batch,
        in_channels=in_channels,
        in_h=in_h,
        in_w=in_w,
        out_channels=out_channels,
        out_h=out_h,
        out_w=out_w,
        kernel_y=kernel_y,
        kernel_x=kernel_x,
        stride_y=stride[0],
        stride_x=stride[1],
        pad_y=padding[0],
        pad_x=padding[1],
        dilate_y=dilation[0],
        dilate_x=dilation[1],
        depthwise=depthwise,
        transposed=transposed,
        upsample_y=upsample[0],
        upsample_x=upsample[1],
        tail_y=tail[0],
        tail_x=tail[1],
        groups=group,
        dynamic_h=dynamic_h,
        conv1d=conv1d,
    )


def deconv_weight_as_conv(weight):
    """The convolution weight a transposed convolution's own weight stands for.

    A transposed convolution is the gradient of a convolution: its weight is
    indexed ``(in_channels, out_channels / groups, ky, kx)`` and its window runs
    backwards, so the convolution that reproduces it carries the two channel
    axes swapped and both spatial axes flipped. With that weight and the padding
    `conv_spec` remaps, the two are the same arithmetic -- the same products
    summed in the same order over the same taps, which is what makes this a
    rereading of the node rather than an approximation of it.
    """
    import numpy as np

    return np.ascontiguousarray(weight.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])


def conv_k_units(in_channels: int) -> int:
    """The kernel's k-block count over the reduction: 32-element slices."""
    return -(-in_channels // 32)


def conv_vtcm_bytes(kernel_y: int, kernel_x: int, in_channels: int) -> int:
    """The VTCM the im2col kernel requests for a window this wide.

    The kernel takes `kp` from the command when it is positive and computes it
    the same way otherwise (`im2col_convolution_fp16.cc:1767`), so this is the
    value the emitter puts on the wire, and the staging buffers are sized from
    it.
    """
    kp = kernel_y * kernel_x * conv_k_units(in_channels)
    return CONV_VTCM_STAGING_TILES * kp * 2048 + CONV_VTCM_FIXED_BYTES


def _channel_blocks(channels: int) -> int:
    return -(-channels // POOL_CHANNEL_BLOCK)


def pack_depthwise_weight(weight, channels: int, kernel_y: int, kernel_x: int) -> bytes:
    """A depthwise ``(channels, 1, ky, kx)`` weight in the order the kernel reads it.

    The walk starts each block's weights at ``weight + cb * ky * kx * 64``
    (depthwise_conv_fp16.c:27) and each tap at ``w + (ky * kernelX + kx) * 64``
    (:49), so the flat index is ``((cb * kernelY + ky) * kernelX + kx) * 64 +
    lane``: one 64-lane vector per tap, with the channel block outermost. Lanes
    past the channel count are zero -- the kernel multiplies them like any
    other lane, and only the lanes below the count are read back.
    """
    import numpy as np

    values = weight.astype(np.float16, copy=False)
    if values.ndim == 3 and values.shape == (channels, 1, kernel_y):
        # The one-axis depthwise weight torch writes is (C, 1, k): the channel,
        # the per-group input count (one) and the taps. The 4-D form is the
        # same bytes with the unit column the reading gives it. This is
        # matched against kernel_y because a one-axis window is k x 1: the
        # taps are the height and the width is the unit the reading supplies.
        values = values.reshape(channels, 1, kernel_y, kernel_x)
    if values.shape != (channels, 1, kernel_y, kernel_x):
        raise RuntimeError(f"hexagon: depthwise weight has shape {values.shape}")
    padded = np.zeros(
        (_channel_blocks(channels), POOL_CHANNEL_BLOCK, kernel_y, kernel_x),
        dtype=np.float16,
    )
    for index in range(_channel_blocks(channels)):
        first = index * POOL_CHANNEL_BLOCK
        width = min(POOL_CHANNEL_BLOCK, channels - first)
        padded[index, :width] = values[first : first + width, 0]
    return np.ascontiguousarray(padded.transpose(0, 2, 3, 1)).tobytes()


def pack_conv_weight(weight, spec: ConvSpec) -> bytes:
    """A general ``(oc, ic, ky, kx)`` weight as the HMX unit's 32x32 tiles.

    ``fill_weight_tiles_fp16`` copies each tile out of the blob verbatim
    (im2col_convolution_fp16.cc:1673-1690), so the tile the unit reads *is* the
    blob's bytes and the column order has to be built here: element (k, c) sits
    at ``(k // 2) * 64 + c * 2 + (k & 1)``, the same rule pack_hmx_weight
    writes for a matmul. The tile index decomposes as ``(ky * kernelX + kx) *
    ic_blocks + ic_block`` with ``ic_blocks = ceil(ic / 32)``
    (fill_im2col_activation_kk_range, :248), and the k inside a tile is the
    channel inside that 32-channel group. Channels past ``ic`` inside the last
    group are zero: the fill reads them from the padding lanes of the blocked
    activation, so a non-zero weight there would add whatever those lanes hold.
    """
    import numpy as np

    values = weight.astype(np.float16, copy=False)
    if spec.conv1d and values.ndim == 3:
        # The one-axis weight (O, I, K) is the (O, I, K, 1) one: same bytes, one
        # column of taps that is a single tap wide.
        values = values[..., None]
    expected = (spec.out_channels, spec.in_channels, spec.kernel_y, spec.kernel_x)
    if values.shape != expected:
        raise RuntimeError(f"hexagon: convolution weight has shape {values.shape}")
    k_blocks = -(-spec.in_channels // 32)
    tiles = -(-spec.out_channels // 32) * spec.kernel_y * spec.kernel_x * k_blocks
    out = np.zeros(tiles * 1024, dtype=np.float16)
    for tile_out in range(-(-spec.out_channels // 32)):
        for ky in range(spec.kernel_y):
            for kx in range(spec.kernel_x):
                for k_block in range(k_blocks):
                    index = (
                        tile_out * spec.kernel_y * spec.kernel_x
                        + ky * spec.kernel_x
                        + kx
                    ) * k_blocks + k_block
                    base = index * 1024
                    for kin in range(32):
                        channel = k_block * 32 + kin
                        if channel >= spec.in_channels:
                            break
                        for c in range(32):
                            out_channel = tile_out * 32 + c
                            if out_channel >= spec.out_channels:
                                break
                            out[base + (kin // 2) * 64 + c * 2 + kin % 2] = values[
                                out_channel, channel, ky, kx
                            ]
    return out.tobytes()


def pack_conv_bias(bias, channels: int, lanes: int) -> bytes:
    """A bias as the flat fp16 array both kernels read a vector at a time from.

    The depthwise walk reads 64 lanes from ``bias + cb * 64`` and the im2col
    walk 64 lanes from ``bias + tile * 32`` (output_conv_fp16.cc:66), so the
    buffer has to be as long as the widest read rather than as long as the
    channel count -- and it can never be absent: the depthwise kernel
    dereferences it unconditionally, so a convolution with no bias needs a zero
    buffer here rather than a null operand.
    """
    import numpy as np

    values = bias.astype(np.float16, copy=False).reshape(-1)
    out = np.zeros(lanes, dtype=np.float16)
    out[: min(channels, lanes)] = values[: min(channels, lanes)]
    return out.tobytes()


def _conv_layouts_agree(batch: int, area: int, channels: int) -> bool:
    """Whether a row-major ``[batch][channels][area]`` buffer is the blocked one.

    The blocked index ``((c // 64) * batch + n) * area * 64 + (m * 64) + c % 64``
    collapses to the row-major ``(n * channels + c) * area + m`` only when a plane
    is a single element and either the tensor is exactly one full block or the
    batch is one. The batch clause is the one that is easy to leave out: the
    blocked layout puts the channel block outside the batch and the row-major one
    puts it inside, so from the second block on the two orders part company at
    any batch above one even when every block is full. With a single block there
    is no order to disagree about and the batch does not enter. A ragged channel
    count widens the blocked form past the tensor it would be read from, so the
    kernel would walk past the buffer rather than inside it.
    """
    return area == 1 and (
        channels == POOL_CHANNEL_BLOCK
        or (batch == 1 and channels % POOL_CHANNEL_BLOCK == 0)
    )


def _channel_block_regions(
    batch: int, area: int, channels: int, packing: bool
) -> List[int]:
    """The blits that move every 64-channel block between the two layouts.

    The geometry is one block's, repeated once per block, with the offsets
    that move block ``i`` from ``[batch][channels][area]`` (packing) or back to
    it (unpacking). The blocked layout puts the channel block outside the batch
    (``[c4][batch][area][64]``), which is why the source offset is the block
    start times the plane and the destination offset counts whole blocks.
    """
    regions: List[int] = []
    for index in range(_channel_blocks(channels)):
        first = index * POOL_CHANNEL_BLOCK
        width = min(POOL_CHANNEL_BLOCK, channels - first)
        blocked_offset = index * batch * area * POOL_CHANNEL_BLOCK
        row_major = [channels * area, area, 1]
        blocked = [area * POOL_CHANNEL_BLOCK, 1, POOL_CHANNEL_BLOCK]
        if packing:
            source, dest = row_major, blocked
            source_offset, dest_offset = first * area, blocked_offset
        else:
            source, dest = blocked, row_major
            source_offset, dest_offset = blocked_offset, first * area
        regions += [0, source_offset, dest_offset, batch, width, area]
        regions += source + dest
    return regions


def _emit_channel_block_blit(
    ctx,
    node: torch.fx.Node,
    source: TensorRef,
    dest: TensorRef,
    batch: int,
    area: int,
    channels: int,
    packing: bool,
    dynamic_area: int = 0,
) -> None:
    """Move every 64-channel block, in as many commands as the parameter block allows.

    One command carries its regions in a fixed 40-int parameter block
    (serialization/hexagon_schema.h:44) and each region takes twelve of them after
    the three-int header, so a command moves at most three blocks and a wide
    enough tensor needs several commands. The count each command carries is its
    own chunk's, so the kernel still sees well-formed commands.

    `dynamic_area` is what one row of the plane is worth in the run length, and
    zero means the plane is static. A plane that moves turns each region's
    offsets and pitches into multiples of the length, which is the one thing the
    runtime can rebuild without the graph.
    """
    regions = _channel_block_regions(batch, area, channels, packing)
    for start in range(0, _channel_blocks(channels), BLIT_BLOCKS_PER_COMMAND):
        chunk = _channel_blocks(channels) - start
        if chunk > BLIT_BLOCKS_PER_COMMAND:
            chunk = BLIT_BLOCKS_PER_COMMAND
        op_index = ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[source],
                outputs=[dest],
                params=[chunk, FP16_BYTES, 1]
                + regions[
                    start * BLIT_REGION_INTS : (start + chunk) * BLIT_REGION_INTS
                ],
            ),
        )
        if not dynamic_area:
            continue
        # Every region int that holds the plane extent is affine in the run-time
        # height once the height is the length itself, and the block offsets are
        # that extent times a static block index, which stays affine. The scales
        # are read off `_channel_block_regions`' own expressions: the offset a
        # packed block starts at counts whole blocks, and the pitches are each
        # layout's own. The two layouts' non-plane pitches -- the row-major
        # channel step is the plane and the blocked one's is a lane -- are the
        # entries that stay static, and the two packings put them in different
        # places.
        for offset in range(chunk):
            blocked_offset = (
                (start + offset) * batch * dynamic_area * POOL_CHANNEL_BLOCK
            )
            row_major_offset = (start + offset) * POOL_CHANNEL_BLOCK * dynamic_area
            if packing:
                # source row-major, dest blocked.
                scales = (
                    0,
                    row_major_offset,
                    blocked_offset,
                    0,
                    0,
                    dynamic_area,
                    channels * dynamic_area,
                    dynamic_area,
                    0,
                    dynamic_area * POOL_CHANNEL_BLOCK,
                    0,
                    0,
                )
            else:
                # source blocked, dest row-major.
                scales = (
                    0,
                    blocked_offset,
                    row_major_offset,
                    0,
                    0,
                    dynamic_area,
                    dynamic_area * POOL_CHANNEL_BLOCK,
                    0,
                    0,
                    channels * dynamic_area,
                    dynamic_area,
                    0,
                )
            base = 3 + offset * BLIT_REGION_INTS
            for position, scale in enumerate(scales):
                if scale > 0:
                    ctx.add_dynamic_patch(op_index, base + position, scale, 0)


def _emit_zero(
    ctx, node: torch.fx.Node, dest: TensorRef, dynamic_frame_bytes: int = 0
) -> None:
    """Clears a whole activation buffer with the DSP's own memset.

    `htp_ops_zero` takes one operand and a byte count (`blit_ops.cc:1724`), and
    the command carries no inputs: the output is the buffer it clears. The count
    is a count of that buffer rather than of the tensor the graph reads, and the
    runtime resizes the buffer for the run it is given
    (`ResizeDynamicDelegate`, runtime/hexagon_backend.cpp), so a count left at
    the export's longest length clears past the end of a shorter arena -- the one
    number in a convolution's command group that is a buffer extent rather than a
    tensor one. `dynamic_frame_bytes` is what one row of the buffer is worth, so
    the runtime rebuilds the count the way it rebuilds the extents. It stays zero
    everywhere the size does not move, which is every static graph and the pad,
    the one other caller: a pad whose height is an expression is what
    `constant_pad_region` refuses to emit at all.
    """
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_ZERO,
            inputs=[],
            outputs=[dest],
            params=[dest.size],
        ),
    )
    if dynamic_frame_bytes:
        ctx.add_dynamic_patch(op_index, 0, dynamic_frame_bytes, 0)


def _blocked_activation(ctx, batch, height, width, channels) -> TensorRef:
    """The 64-channel blocked buffer one convolution reads or writes.

    A static height is the allocation the emitter has always made. The run-time
    symbol instead sizes the buffer for the longest run the export allows and
    records how many bytes the run it actually gets needs, which is the same
    pairing the matmul and reduction paths use: the arena holds the bound, the
    blob's layout says what of it this call owns, and the command's patched
    extents say which part of that the kernel may read.
    """
    blocks = _channel_blocks(channels)
    if not isinstance(height, torch.SymInt):
        return ctx.builder.add_activation(
            batch * int(height) * width * blocks * POOL_CHANNEL_BLOCK * FP16_BYTES
        )
    return ctx.activation_for_shape((blocks, batch, height, width, POOL_CHANNEL_BLOCK))


def _patch_conv_extents(ctx, op_index: int, spec: ConvSpec, im2col: bool) -> None:
    """Recompute the extents a convolution command baked in at export time.

    The height, the output height and both plane strides reach the kernel as
    param ints, and every one of them is the run-time length times a static
    factor once the geometry is height-preserving and one row per output row.
    `scale` is that factor -- the length itself for the two heights, a row of the
    operand or of the result for the strides -- so the value the runtime writes
    is the one the emitter would have written for a run of that length.
    """
    if not spec.dynamic_h:
        return
    if not im2col:
        # The depthwise walk takes the two heights only; its plane strides are
        # derived inside the kernel from in_h and in_w.
        ctx.add_dynamic_patch(op_index, 1, 1, 0)
        ctx.add_dynamic_patch(op_index, 3, 1, 0)
        return
    for param_index, scale in (
        (11, 1),
        (13, 1),
        (14, spec.batch * spec.in_w * POOL_CHANNEL_BLOCK),
        (17, spec.in_w * POOL_CHANNEL_BLOCK),
    ):
        ctx.add_dynamic_patch(op_index, param_index, scale, 0)


def _emit_zero_insert(ctx, node: torch.fx.Node, source: TensorRef, spec: ConvSpec):
    """The zero-interleaved copy a transposed convolution's input needs.

    A transposed convolution scatters each input element across the window it
    covers, which is not a walk any kernel here makes. Interleaving the input
    with zeros turns it back into one: every input element is written at a
    multiple of the stride, and the gaps the kernel then walks over are the
    zeros that stand for the elements the transposed window never visits.

    Two commands, because the gaps have to be zeros rather than whatever the
    arena last held: `DSP_OP_ZERO` clears the plane and one raster region
    scatters the input into it. `_zero_insert_regions` says what that region is.
    """
    batch, channels = spec.batch, spec.in_channels
    regions = _zero_insert_regions(_value_of(node.args[0]).shape, spec)
    plane = spec.in_h * spec.in_w
    buffer = ctx.builder.add_activation(batch * channels * plane * FP16_BYTES)
    _emit_zero(ctx, node, buffer)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[source],
            outputs=[buffer],
            # Region count, element bytes, source count, then the regions.
            params=[len(regions) // BLIT_REGION_INTS, FP16_BYTES, 1] + regions,
        ),
    )
    return buffer


def _nearest_index(scale: float, index: int, extent: int) -> int:
    """The source index torch's nearest kernel reads, in torch's own arithmetic.

    `nearest_neighbor_compute_source_index` is
    `min((int64_t)floorf(index * (float)scale), extent - 1)` -- a **float**
    multiply and a float floor, not exact division. Reproducing that here is what
    lets the gate check the mapping it is about to emit instead of assuming
    `index // stride` and hoping the two agree.
    """
    import numpy as np

    product = np.float32(index) * np.float32(scale)
    return min(int(np.floor(product)), extent - 1)


def upsample_regions(node: torch.fx.Node):
    """The blit regions that replicate a plane for a nearest upsample, or None.

    A replication is many-to-one and one affine region cannot express it: a
    region's destination index is a constant stride times a source index, so
    every source element reaches exactly one destination cell, and the
    destination cells that need the *same* source element twice are unreachable.
    The fix is not a different primitive but a different count: one on each side
    of that, and a region can express it exactly.

    A *set* of regions does express it, one per phase of the destination index.
    For an exact integer multiple `s`, destination index `k * s + t` reads source
    index `k`, so the phase `t` is the affine map `k -> k` in the source and
    `k -> k * s + t` in the destination. There are `s` phases on each axis and so
    `s * s` regions, all reading the same input with the same strides and
    differing only in their destination offset. Nothing is computed, so the
    result is the input's bytes and is exact rather than merely close.

    **This is the whole of what is supported.** Scale factors that are not exact
    integer multiples are refused: `nearest` then reads
    `in[floor(oh * in / out)]`, whose runs of repeated source rows are no longer
    all the same length, so the phases would need one region per distinct run
    length instead of `s` of them. `upsample_bilinear2d` is a different shape
    again and is refused here -- its taps are parity-dependent (an even output
    row reads the row *below* it, an odd one the row *above*), so it is neither
    one shift-invariant filter nor the transposed convolution that the
    zero-interleaving machinery would need.
    """
    if len(node.args) < 2:
        return None
    source_value = (
        node.args[0].meta.get("val")
        if isinstance(node.args[0], torch.fx.Node)
        else None
    )
    result_value = node.meta.get("val")
    if not isinstance(source_value, torch.Tensor) or not isinstance(
        result_value, torch.Tensor
    ):
        return None
    if source_value.dtype not in (torch.float16, torch.float32):
        return None
    if source_value.dim() != 4 or result_value.dim() != 4:
        return None
    if not source_value.is_contiguous() or not result_value.is_contiguous():
        return None
    batch, channels, in_h, in_w = source_value.shape
    out_batch, out_channels, out_h, out_w = result_value.shape
    if (batch, channels) != (out_batch, out_channels):
        return None
    if min(batch, channels, in_h, in_w) <= 0:
        return None
    if out_h % in_h or out_w % in_w:
        return None
    stride_y, stride_x = out_h // in_h, out_w // in_w
    if stride_y < 1 or stride_x < 1:
        return None

    # torch reads the input at `floor(index * scale)`, where `scale` is the
    # reciprocal of a scale_factor the caller gave, or the ratio of the shapes
    # when it gave a size instead. Check that against the integer division this
    # emitter is about to hard-code, over every output index on both axes.
    scale_factors = node.args[2] if len(node.args) > 2 else None
    for axis, (in_extent, out_extent, stride) in enumerate(
        ((in_h, out_h, stride_y), (in_w, out_w, stride_x))
    ):
        if scale_factors is not None and len(scale_factors) == 2:
            factor = scale_factors[axis]
            scale = 1.0 / float(factor) if float(factor) > 0 else None
        else:
            scale = in_extent / out_extent
        if scale is None:
            return None
        if any(
            _nearest_index(scale, index, in_extent) != index // stride
            for index in range(out_extent)
        ):
            return None

    regions: List[int] = []
    for phase_y in range(stride_y):
        for phase_x in range(stride_x):
            # srcIndex, srcOffset, dstOffset, size[xyz], srcStride[xyz],
            # dstStride[xyz]: the source is read whole once per phase and the
            # destination is written at the interleave factor, offset into the
            # phase.
            regions += [
                0,
                0,
                phase_y * out_w + phase_x,
                batch * channels,
                in_h,
                in_w,
                in_h * in_w,
                in_w,
                1,
                out_h * out_w,
                stride_y * out_w,
                stride_x,
            ]
    return regions


def _emit_upsample(node: torch.fx.Node, ctx) -> TensorRef:
    """A nearest upsample as raster regions, three to a command.

    The params vector holds 40 ints, the blit header takes three and each region
    twelve, so a command carries three regions and a wider factor takes more than
    one command. The interpreter applies them in order and the regions do not
    overlap, so the split is a partitioning of the work rather than a sequence of
    passes over the same bytes.
    """
    regions = upsample_regions(node)
    if regions is None:
        raise RuntimeError(
            f"hexagon: {node.target} is not an integer-multiple nearest upsample"
        )
    _require_arena_dtype(node, "upsample")
    source = ctx.operand(node.args[0])
    out = ctx.result_for(node, _numel(node))
    for start in range(0, len(regions), BLIT_BLOCKS_PER_COMMAND * BLIT_REGION_INTS):
        chunk = regions[start : start + BLIT_BLOCKS_PER_COMMAND * BLIT_REGION_INTS]
        ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[source],
                outputs=[out],
                # Region count, element bytes, source count, then the regions.
                params=[len(chunk) // BLIT_REGION_INTS, FP16_BYTES, 1] + chunk,
            ),
        )
    return ctx.record(node, out)






def _emit_convolution(node: torch.fx.Node, ctx) -> TensorRef:
    """A convolution as the DSP's depthwise walk or its im2col convolution.

    Dense and depthwise forms use the same blocked activation, with the blit
    in, the convolution, and the blit out of its result. A transposed
    convolution is the same three with an interleaving blit ahead of them and a
    flipped weight, and a grouped one is the dense walk repeated per group with
    a raster region on either side of each group's command.
    """
    spec = conv_spec(node, lambda operand: ctx.constant_value(operand) is not None)
    if spec is None:
        raise RuntimeError(
            "hexagon: this convolution is not one the DSP kernels can run (see conv_spec)"
        )
    _require_arena_dtype(node, "convolution")
    source = ctx.operand(node.args[0])
    if spec.upsample_y != 1 or spec.upsample_x != 1 or spec.tail_y or spec.tail_x:
        source = _emit_zero_insert(ctx, node, source, spec)
    weight_node = node.args[1]
    bias_node = node.args[2] if len(node.args) > 2 else None
    out = ctx.result_for(node, _numel(node))
    if not spec.depthwise:
        if spec.groups > 1:
            return _emit_grouped_convolution(node, ctx, spec, source, out)

        def _pack(array):
            # A rank-3 or rank-5 weight reaches the same two-dimensional kernel
            # as the rank-4 one, so the axis conv_spec kept and the axis it
            # dropped have to be taken off here; a transposed weight is indexed
            # the other way round before either.
            if spec.transposed:
                array = deconv_weight_as_conv(array)
            elif array.ndim == 3:
                array = array.reshape(
                    spec.out_channels, spec.in_channels, spec.kernel_y, spec.kernel_x
                )
            elif array.ndim == 5:
                spatial = [int(size) for size in array.shape[2:]]
                if spatial[0] == spatial[1] == 1:
                    array = array[:, :, 0, 0, :]
                else:
                    singleton = next(axis for axis, size in enumerate(spatial) if size == 1)
                    axes = [0, 1] + [axis + 2 for axis in range(3) if axis != singleton]
                    array = array[(slice(None), slice(None)) + tuple(axes)]
                array = array.reshape(
                    spec.out_channels, spec.in_channels, spec.kernel_y, spec.kernel_x
                )
            return pack_conv_weight(array, spec)

        weight = ctx.packed_weights(
            weight_node,
            _pack,
            "transposed im2col" if spec.transposed else "im2col",
        )
        bias = _conv_bias_ref(
            ctx,
            bias_node,
            spec.out_channels,
            -(-spec.out_channels // 32) * 32 + 32,
            "im2col",
        )
        _emit_dense_im2col(node, ctx, spec, source, out, weight, bias)
        return ctx.record(node, out)

    # The packed input holds the plane the kernel walks, and for a transposed
    # convolution that is the interleaved plane the zero-insert writes, so its
    # height is the spec's rather than the operand's. A plain convolution keeps
    # the operand's own dim, which has to stay the symbol when it is the run-time
    # length so the frame is sized from the length handed in.
    # The dynamic length is the graph's own time axis on the one-axis form
    # (axis 2 of a (B, C, T) node) and the command's height on the 4-D one
    # (axis 2 of (N, C, H, W)), so both are read at index 2 and the two forms
    # need no separate case here.
    in_h_dim = (
        _value_of(node.args[0]).shape[2]
        if spec.dynamic_h and not spec.transposed
        else spec.in_h
    )
    out_h_dim = _value_of(node).shape[2] if spec.dynamic_h else spec.out_h
    in_area = spec.in_h * spec.in_w
    out_area = spec.out_h * spec.out_w
    packed_in = source
    if not _conv_layouts_agree(spec.batch, in_area, spec.in_channels):
        packed_in = _blocked_activation(
            ctx, spec.batch, in_h_dim, spec.in_w, spec.in_channels
        )
        if not spec.depthwise and spec.in_channels % POOL_CHANNEL_BLOCK:
            # The im2col fill copies whole 64-lane groups out of the blocked
            # activation, so the lanes past the last channel are read -- and
            # multiplied by the zero weights the tiles carry for them, which a
            # NaN there would turn into another NaN rather than a zero. The pack
            # blit below writes only the channels the tensor has, so those lanes
            # have to be zeroed first.
            _emit_zero(
                ctx,
                node,
                packed_in,
                dynamic_frame_bytes=(
                    spec.batch
                    * spec.in_w
                    * _channel_blocks(spec.in_channels)
                    * POOL_CHANNEL_BLOCK
                    * FP16_BYTES
                    if spec.dynamic_h
                    else 0
                ),
            )
        _emit_channel_block_blit(
            ctx,
            node,
            source,
            packed_in,
            spec.batch,
            in_area,
            spec.in_channels,
            True,
            dynamic_area=spec.in_w if spec.dynamic_h else 0,
        )
    packed_out = out
    if not _conv_layouts_agree(spec.batch, out_area, spec.out_channels):
        packed_out = _blocked_activation(
            ctx, spec.batch, out_h_dim, spec.out_w, spec.out_channels
        )

    weight = ctx.packed_weights(
        weight_node,
        lambda array: pack_depthwise_weight(
            array.reshape(spec.in_channels, 1, spec.kernel_y, spec.kernel_x), spec.in_channels, spec.kernel_y, spec.kernel_x
        ),
        "depthwise",
    )
    bias = _conv_bias_ref(
        ctx,
        bias_node,
        spec.out_channels,
        _channel_blocks(spec.out_channels) * POOL_CHANNEL_BLOCK,
        "depthwise",
    )
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_CONV_DEPTHWISE2D_FP16,
            inputs=[packed_in, weight, bias],
            outputs=[packed_out],
            params=[
                spec.batch,
                spec.in_h,
                spec.in_w,
                spec.out_h,
                spec.out_w,
                _channel_blocks(spec.in_channels),
                spec.kernel_y,
                spec.kernel_x,
                spec.stride_y,
                spec.stride_x,
                spec.pad_y,
                spec.pad_x,
                spec.dilate_y,
                spec.dilate_x,
                # The kernel fuses relu and relu6 after the bias, and to_edge
                # leaves a relu as its own node rather than folding it in, so
                # both are always off here.
                0,
                0,
            ],
        ),
    )
    _patch_conv_extents(ctx, op_index, spec, im2col=False)
    if packed_out is not out:
        _emit_channel_block_blit(
            ctx,
            node,
            packed_out,
            out,
            spec.batch,
            out_area,
            spec.out_channels,
            False,
            dynamic_area=spec.out_w if spec.dynamic_h else 0,
        )
    return ctx.record(node, out)


def _emit_dense_im2col(
    node: torch.fx.Node,
    ctx,
    spec: ConvSpec,
    source: TensorRef,
    out: TensorRef,
    weight: TensorRef,
    bias: TensorRef,
) -> None:
    """Run one dense im2col walk over a row-major activation pair."""
    in_h_dim = (
        _value_of(node.args[0]).shape[2]
        if spec.dynamic_h and not spec.transposed
        else spec.in_h
    )
    out_h_dim = _value_of(node).shape[2] if spec.dynamic_h else spec.out_h
    in_area = spec.in_h * spec.in_w
    out_area = spec.out_h * spec.out_w
    packed_in = source
    if not _conv_layouts_agree(spec.batch, in_area, spec.in_channels):
        packed_in = _blocked_activation(
            ctx, spec.batch, in_h_dim, spec.in_w, spec.in_channels
        )
        if spec.in_channels % POOL_CHANNEL_BLOCK:
            _emit_zero(
                ctx,
                node,
                packed_in,
                dynamic_frame_bytes=(
                    spec.batch
                    * spec.in_w
                    * _channel_blocks(spec.in_channels)
                    * POOL_CHANNEL_BLOCK
                    * FP16_BYTES
                    if spec.dynamic_h
                    else 0
                ),
            )
        _emit_channel_block_blit(
            ctx,
            node,
            source,
            packed_in,
            spec.batch,
            in_area,
            spec.in_channels,
            True,
            dynamic_area=spec.in_w if spec.dynamic_h else 0,
        )
    packed_out = out
    if not _conv_layouts_agree(spec.batch, out_area, spec.out_channels):
        packed_out = _blocked_activation(
            ctx, spec.batch, out_h_dim, spec.out_w, spec.out_channels
        )
    op_index = ctx.emit(
        node,
        Op(
            # 17 names the function's 1x1 activation fill, which this geometry
            # takes; 12 is the same function on its general window walk.
            # conv_1x1_direct_applies is the host transcription of the C's own
            # selection, so the stream says which fill runs.
            type=(
                DSP_OP_CONV1X1_DIRECT_FP16
                if conv_1x1_direct_applies(spec)
                else DSP_OP_IM2COL_CONVOLUTION_FP16
            ),
            inputs=[packed_in, weight, bias],
            outputs=[packed_out],
            params=[
                spec.pad_x,
                spec.pad_y,
                spec.dilate_x,
                spec.dilate_y,
                spec.stride_x,
                spec.stride_y,
                spec.kernel_x,
                spec.kernel_y,
                # icDiv4 and icup4 belong to the int4 entry points; the fp16
                # path reads neither.
                spec.in_channels // 4,
                spec.kernel_y * spec.kernel_x * conv_k_units(spec.in_channels),
                spec.in_w,
                spec.in_h,
                spec.out_w,
                spec.out_h,
                # srcZStep steps between 64-channel blocks, srcYStep between
                # rows of one, and destICStride is what a batch advances by:
                # all three in the blocked layout the blit above produced
                # (input_block_offset_fp16, :213).
                spec.batch * in_area * POOL_CHANNEL_BLOCK,
                spec.in_w * POOL_CHANNEL_BLOCK,
                POOL_CHANNEL_BLOCK,
                in_area * POOL_CHANNEL_BLOCK,
                spec.in_channels,
                (spec.in_channels + 3) // 4 * 4,
                spec.out_channels,
                # One position tile and two channel tiles per pass. The pair
                # is not a tuning choice: the single-tile store rotates an
                # odd tile's accumulator *after* adding its bias, so an odd
                # tile that reaches it is handed the neighbouring tile's bias
                # on the last position of a ragged position tile.
                # test/sim/conv_runner.cpp runs that case as CV_ODD and
                # test_conv_sim.py pins the wrong channels and the size of
                # the error on the simulator.
                1,
                2,
                # relu and relu6 are fused into the store, and to_edge
                # leaves a relu as its own node, so both are off here.
                0,
                0,
                spec.batch,
                # outputBytes turns the store's own bounds check off.
                0,
                0,
                0,
            ],
        ),
    )
    _patch_conv_extents(ctx, op_index, spec, im2col=True)
    if packed_out is not out:
        _emit_channel_block_blit(
            ctx,
            node,
            packed_out,
            out,
            spec.batch,
            out_area,
            spec.out_channels,
            False,
            dynamic_area=spec.out_w if spec.dynamic_h else 0,
        )


def _emit_grouped_convolution(
    node: torch.fx.Node, ctx, spec: ConvSpec, source: TensorRef, out: TensorRef
) -> TensorRef:
    """Run one dense im2col walk for each group of a grouped convolution.

    The DSP command carries no group field, so the host supplies the channel
    mapping: group `g` reads the `C_in / groups` input channels at offset
    `g * C_in / groups * area`, its own weight tile and bias, and writes the
    `C_out / groups` result channels at the same ordinal position in the
    output. A plain convolution's weight is `[C_out, C_in/groups, ky, kx]`,
    so the group's rows are the contiguous slice of that first axis; a
    transposed one's is `[C_in, C_out/groups, ky, kx]`, so the slice is of
    the first axis too, after the flip the identity asks for. torch orders the
    output by group as well, so the ordinal placement is the tensor's own
    order and no permutation is involved.
    """
    in_per_group = spec.in_channels // spec.groups
    out_per_group = spec.out_channels // spec.groups
    in_area = spec.in_h * spec.in_w
    out_area = spec.out_h * spec.out_w
    weight_node = node.args[1]
    bias_node = node.args[2] if len(node.args) > 2 else None
    for index in range(spec.groups):
        group = spec._replace(
            groups=1,
            in_channels=in_per_group,
            out_channels=out_per_group,
        )
        group_in = ctx.builder.add_activation(
            spec.batch * in_per_group * in_area * FP16_BYTES
        )
        ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[source],
                outputs=[group_in],
                params=[
                    1,
                    FP16_BYTES,
                    1,
                    0,
                    index * in_per_group * in_area,
                    0,
                    spec.batch,
                    in_per_group,
                    in_area,
                    spec.in_channels * in_area,
                    in_area,
                    1,
                    in_per_group * in_area,
                    in_area,
                    1,
                ],
            ),
        )
        group_out = ctx.builder.add_activation(
            spec.batch * out_per_group * out_area * FP16_BYTES
        )
        out_first = index * out_per_group
        if spec.transposed:
            in_first = index * in_per_group
            weight = ctx.packed_weights(
                weight_node,
                lambda array: pack_conv_weight(
                    deconv_weight_as_conv(array[in_first : in_first + in_per_group]),
                    group,
                ),
                f"transposed im2col group {index}",
            )
        else:
            weight = ctx.packed_weights(
                weight_node,
                lambda array: pack_conv_weight(
                    array[out_first : out_first + out_per_group], group
                ),
                f"grouped im2col group {index}",
            )
        # The kind carries the group index because packed_weights caches
        # on (node, kind): without it every group would read the first group's
        # bias, which is a plausible wrong tensor rather than a refusal.
        lanes = -(-out_per_group // 32) * 32 + 32
        if bias_node is None:
            bias = _conv_bias_ref(
                ctx, None, out_per_group, lanes, f"im2col group {index}"
            )
        else:
            bias = ctx.packed_weights(
                bias_node,
                lambda array: pack_conv_bias(
                    array[out_first : out_first + out_per_group],
                    out_per_group,
                    lanes,
                ),
                f"im2col group {index}",
            )
        _emit_dense_im2col(node, ctx, group, group_in, group_out, weight, bias)
        ctx.emit(
            node,
            Op(
                type=DSP_OP_RASTER_BLIT,
                inputs=[group_out],
                outputs=[out],
                params=[
                    1,
                    FP16_BYTES,
                    1,
                    0,
                    0,
                    index * out_per_group * out_area,
                    spec.batch,
                    out_per_group,
                    out_area,
                    out_per_group * out_area,
                    out_area,
                    1,
                    spec.out_channels * out_area,
                    out_area,
                    1,
                ],
            ),
        )
    return ctx.record(node, out)


def _conv_bias_ref(ctx, bias_node, channels: int, lanes: int, kind: str) -> TensorRef:
    """The bias operand, padded to the widest vector read either kernel makes.

    A convolution without a bias still needs a buffer here rather than a null
    operand, and the buffer has to be longer than the channel count: every read
    starts at a 32- or 64-channel group's first lane and runs a whole HVX vector
    past it.
    """
    if bias_node is not None:
        return ctx.packed_weights(
            bias_node, lambda array: pack_conv_bias(array, channels, lanes), kind
        )
    return ctx.builder.add_weights(bytes(lanes * FP16_BYTES))


# REDUCTION collapses one contiguous span, so the reduced dims have to be
# adjacent; the callers' support checks enforce that before an emitter runs.
SUM_DIM = exir_ops.edge.aten.sum.dim_IntList
AMAX = exir_ops.edge.aten.amax.default
MAX_DEFAULT = exir_ops.edge.aten.max.default
MIN_DEFAULT = exir_ops.edge.aten.min.default
# max.dim is the reduction max.default already runs, with the positions a second
# output of the same node: the values are emittable exactly when nothing reads
# those, the same rule max_pool2d_with_indices is placed under.
MAX_DIM = exir_ops.edge.aten.max.dim
AMIN = exir_ops.edge.aten.amin.default
MIN_DIM = exir_ops.edge.aten.min.dim
REDUCTION_TARGETS = frozenset(
    {SUM_DIM, AMAX, MAX_DEFAULT, MAX_DIM, AMIN, MIN_DIM, MIN_DEFAULT}
)
SUM_TARGETS = frozenset({SUM_DIM})

# The mean family's two overloads, both of which are this same span rule: .dim
# names the axes and .default has none, which is the whole buffer.
MEAN_DIM = exir_ops.edge.aten.mean.dim
MEAN_DEFAULT = exir_ops.edge.aten.mean.default
MEAN_TARGETS = frozenset({MEAN_DIM, MEAN_DEFAULT})

# The unary kernel's square entry point, reached as x ** 2. to_edge emits no
# aten.square.default at all, so pow.Tensor_Scalar is the form that exists.
POW_TENSOR_SCALAR = exir_ops.edge.aten.pow.Tensor_Scalar
POW_TENSOR_TENSOR = exir_ops.edge.aten.pow.Tensor_Tensor
SQUARE_POW_TARGETS = frozenset({POW_TENSOR_SCALAR})
POW_TENSOR_TENSOR_TARGETS = frozenset({POW_TENSOR_TENSOR})
# The exponent values for which repeated multiply/divide is a complete,
# domain-checked fp16 operation. Larger values are refused before lowering.
POW_TENSOR_TENSOR_EXPONENTS = frozenset({-1, 0, 1, 2, 3, 4})


def _pow_static_value(node, values):
    if not isinstance(node, torch.fx.Node):
        return None
    value = values.get(node) if values is not None else None
    if value is not None:
        return value.detach() if isinstance(value, torch.Tensor) else None
    if node.op == "get_attr":
        value = node.meta.get("val")
        return value.detach() if isinstance(value, torch.Tensor) and type(value).__name__ != "FakeTensor" else None
    if node.op == "call_function" and node.args:
        target_name = getattr(node.target, "__name__", "")
        if any(
            marker in target_name
            for marker in (
                "clone_dim_order",
                "to_dim_order",
                "alias_copy",
                "detach_",
                "lift_fresh_copy",
            )
        ):
            return _pow_static_value(node.args[0], values)
    return None


def pow_tensor_tensor_exponent(node, values=None):
    if node.target is not POW_TENSOR_TENSOR or len(node.args) < 2:
        return None
    exponent = _pow_static_value(node.args[1], values)
    if exponent is None or exponent.numel() == 0:
        return None
    if type(exponent).__name__ == "FakeTensor":
        return None
    if exponent.dtype not in (torch.float16, torch.float32):
        return None
    if not torch.isfinite(exponent).all():
        return None
    if not torch.equal(exponent, exponent.round()):
        return None
    flat = exponent.reshape(-1)
    first = flat[0]
    if not torch.equal(flat, first.expand_as(flat)):
        return None
    value = int(first.item())
    return value if value in POW_TENSOR_TENSOR_EXPONENTS else None


def pow_tensor_tensor_is_emittable(node, values=None):
    if node.target is not POW_TENSOR_TENSOR:
        return False
    exponent = pow_tensor_tensor_exponent(node, values)
    if exponent is None:
        return False
    if len(node.args) < 2 or not all(isinstance(arg, torch.fx.Node) for arg in node.args[:2]):
        return False
    output = node.meta.get("val")
    if not isinstance(output, torch.Tensor) or output.dtype not in (torch.float16, torch.float32):
        return False
    base = _pow_static_value(node.args[0], values)
    exponent_value = _pow_static_value(node.args[1], values)
    if exponent_value is None:
        return False
    if type(exponent_value).__name__ == "FakeTensor":
        return False
    if not isinstance(node.args[0], torch.fx.Node) or not isinstance(node.args[1], torch.fx.Node):
        return False
    base_value = node.args[0].meta.get("val")
    exponent_meta = node.args[1].meta.get("val")
    if not isinstance(base_value, torch.Tensor) or not isinstance(exponent_meta, torch.Tensor):
        return False
    base_shape = tuple(base_value.shape)
    exponent_shape = tuple(exponent_meta.shape)
    if base_value.dtype not in (torch.float16, torch.float32) or exponent_meta.dtype not in (torch.float16, torch.float32):
        return False
    if any(
        isinstance(dim, torch.SymInt)
        for dim in (*base_shape, *exponent_shape, *tuple(output.shape))
    ):
        return False
    try:
        broadcast = torch.broadcast_shapes(base_shape, exponent_shape)
    except RuntimeError:
        return False
    if tuple(output.shape) != broadcast or len(broadcast) > 8:
        return False
    if exponent == 1:
        return True
    if base is None or type(base).__name__ == "FakeTensor":
        return False
    if base.dtype not in (torch.float16, torch.float32):
        return False
    if not torch.isfinite(base).all() or not torch.isfinite(exponent_value).all():
        return False
    if exponent <= 0 and torch.any(base == 0):
        return False
    result = torch.pow(base.detach().to(torch.float64), exponent)
    if not torch.isfinite(result).all() or not torch.isfinite(result.to(torch.float16)).all():
        return False
    if torch.any((result != 0) & (result.to(torch.float16) == 0)):
        return False
    return True


def _emit_tensor_pow(node: torch.fx.Node, ctx) -> TensorRef:
    values = {}
    for arg in node.all_input_nodes:
        current = arg
        while isinstance(current, torch.fx.Node):
            value = ctx.constant_value(current)
            if value is not None:
                values[arg] = value
                break
            if current.op == "call_function" and current.args:
                current = current.args[0]
            else:
                break
    if not pow_tensor_tensor_is_emittable(node, values):
        raise RuntimeError(f"hexagon: unsupported tensor power for {node.name}")
    base, _ = node.args[:2]
    exponent = pow_tensor_tensor_exponent(node, values)
    base_ref = ctx.operand(base)
    base_shape = tuple(node.args[0].meta["val"].shape)
    if any(isinstance(dim, torch.SymInt) for dim in base_shape):
        raise RuntimeError("hexagon: tensor power requires static dimensions")
    out_shape = tuple(node.meta["val"].shape)
    out_numel = _upper_product(out_shape, ctx)
    if exponent == 1:
        out = ctx.result_for(node, out_numel)
        if ctx.is_method_output(node):
            ctx.emit(
                node,
                Op(
                    type=DSP_OP_RASTER_BLIT,
                    inputs=[base_ref],
                    outputs=[out],
                    params=[1, FP16_BYTES, 1, 0, 0, 0, 1, 1, out_numel, 0, 0, 1, 0, 0, 1],
                ),
            )
        else:
            out = base_ref
        return ctx.record(node, out)
    if exponent == 0:
        out = ctx.result_for(node, out_numel)
        one = ctx.scalar(1.0)
        _emit_binary_refs(node, ctx, one, one, 1, 1, (), (), out_shape, "div", out)
        return ctx.record(node, out)
    if exponent < 0:
        one = ctx.scalar(1.0)
        current = one
        current_shape = ()
        for step in range(-exponent):
            out = (
                ctx.result_for(node, out_numel)
                if step == -exponent - 1
                else ctx.activation_for_shape(out_shape)
            )
            _emit_binary_refs(
                node,
                ctx,
                current,
                base_ref,
                _upper_product(current_shape, ctx),
                _upper_product(base_shape, ctx),
                current_shape,
                base_shape,
                out_shape,
                "div",
                out,
            )
            current = out
            current_shape = out_shape
    else:
        current = base_ref
        current_shape = base_shape
        for step in range(1, exponent):
            out = (
                ctx.result_for(node, out_numel)
                if step == exponent - 1
                else ctx.activation_for_shape(out_shape)
            )
            _emit_binary_refs(
                node,
                ctx,
                current,
                base_ref,
                _upper_product(current_shape, ctx),
                _upper_product(base_shape, ctx),
                current_shape,
                base_shape,
                out_shape,
                "mul",
                out,
            )
            current = out
            current_shape = out_shape
    return ctx.record(node, current)


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
    out = ctx.result_for(_values_sink(node), _numel(node))
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


def _emit_amin(node: torch.fx.Node, ctx) -> TensorRef:
    return _emit_reduction(node, ctx, REDUCTION_MINIMUM)


def _emit_max_default(node: torch.fx.Node, ctx) -> TensorRef:
    """torch.max(x): every element, as the single span amax already takes.

    The overload carries no dim, so the whole buffer is one
    ``[1][numel][1]`` span -- the same shape, the same kernel and the same
    seeding that torch.amax reaches through AMAX. Its values are torch.amax's;
    the two differ only in the sign of a zero and in the payload of a NaN,
    which is the byte-level caveat fmod already carries.
    """
    return _emit_reduction(node, ctx, REDUCTION_MAXIMUM)


def _emit_min_default(node: torch.fx.Node, ctx) -> TensorRef:
    if len(node.args) == 2:
        return _binary("min")(node, ctx)
    return _emit_reduction(node, ctx, REDUCTION_MINIMUM)


def _emit_min_dim(node: torch.fx.Node, ctx) -> TensorRef:
    return _emit_reduction(node, ctx, REDUCTION_MINIMUM)


def _emit_max_dim(node: torch.fx.Node, ctx) -> TensorRef:
    """torch.max(x, dim): the values, one reduction command.

    The node hands out (values, indices) and the kernel computes values, so this
    is amax over the same span; the support check has already established that
    nothing reads the indices, and the sink is the getitem that hands the values
    on. `torch.max(x, dim=1)` and `torch.amax(x, dim=1)` are the same value, with
    the same signed-zero and NaN caveat max.default carries.
    """
    return _emit_reduction(node, ctx, REDUCTION_MAXIMUM)


# The one kernel in this backend that answers two outputs, and the reason it
# still lands under the values-only rule the pool and max.dim are under -- for a
# different reason from theirs. `htp_ops_topkv2_k1_fp16` (topk_ops.cc:48) walks
# each row of the last axis, keeps the maximum's bit pattern, and writes that
# value and the first index whose bits equal it. k == 1 is not a subset of the
# kernel: there is no other k, no rank argument and no second pass.
#
# What it writes for the position is not what torch writes. On a tie -- two
# equal maxima, ordinary in fp16 -- torch's CPU kernel returns whichever index
# its own partial sort lands on and this one returns the first. Measured:
# `[1, 1, .5, -2, 1, .5, 0]` comes back as 1 from torch and 0 from the kernel,
# a row of four equal values comes back as 2 from torch and 0 from the kernel,
# and over 200 rows of quantized values torch matches neither the first nor the
# last occurrence on 175 of them. Handing the graph's indices that answer would
# put a position torch never produced into a tensor the model reads, which is
# what a refusal is for; the values, which are the row's maximum either way, are
# the half this emitter takes. The kernel dereferences both pointers, so the
# other half gets a scratch slot nothing reads rather than being left ABSENT,
# which would fail the command.
#
# The values carry the caveat max and amax already carry -- a NaN row and the
# sign of a zero are where the DSP's maximum and torch's differ.
TOPK = exir_ops.edge.aten.topk.default
ARGMAX = exir_ops.edge.aten.argmax.default
ARGMIN = exir_ops.edge.aten.argmin.default
ARG_REDUCTION_TARGETS = frozenset({ARGMAX, ARGMIN})


class TopkSpec(NamedTuple):
    """The two extents htp_ops_topkv2_k1_fp16 walks: rowSize and rows."""

    row_size: int
    rows: int


def _node_arg(node: torch.fx.Node, name: str, index: int, default):
    """An argument from wherever the graph put it, or the schema's default.

    `to_edge` writes an argument it was given positionally -- `dim` lands in
    args[2] -- and leaves one that kept its default out of the node altogether,
    so a missing argument means the schema's value rather than an unknown one.
    Both topk and split are read through this: neither op's arguments survive
    `to_edge` in a single place.
    """
    if name in node.kwargs:
        return node.kwargs[name]
    return node.args[index] if len(node.args) > index else default


class ArgReductionSpec(NamedTuple):
    """The contiguous rows one arg-reduction command can walk."""

    row_size: int
    rows: int
    is_min: bool


def arg_reduction_spec(node: torch.fx.Node) -> Optional[ArgReductionSpec]:
    """The row geometry for a precisely supported argmax or argmin.

    The command walks contiguous fp16 rows and writes int64 positions. A
    flattened reduction is one row; an axis reduction is accepted only on the
    last axis, where the row pitch is the graph's contiguous innermost extent.
    The command's row and column counts are int32 words, so symbolic or
    unrepresentable extents are refused rather than baked at the example shape.
    """
    if node.target not in ARG_REDUCTION_TARGETS:
        return None
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    value = source.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.float16:
        return None
    if value.dim() == 0 or value.numel() <= 0 or not value.is_contiguous():
        return None
    shape = list(value.shape)
    if not all(isinstance(extent, int) and extent > 0 for extent in shape):
        return None
    dim = _node_arg(node, "dim", 1, None)
    keepdim = _node_arg(node, "keepdim", 2, False)
    if not isinstance(keepdim, bool):
        return None
    is_min = node.target is ARGMIN
    if dim is None:
        row_size = value.numel()
    else:
        rank = value.dim()
        if (
            isinstance(dim, bool)
            or not isinstance(dim, int)
            or not (-rank <= dim < rank)
        ):
            return None
        if dim % rank != rank - 1:
            return None
        row_size = shape[-1]
    if row_size <= 0 or row_size > 0x7FFFFFFF:
        return None
    rows = value.numel() // row_size
    if rows <= 0 or rows > 0x7FFFFFFF:
        return None
    return ArgReductionSpec(row_size=row_size, rows=rows, is_min=is_min)


def arg_reduction_is_emittable(node: torch.fx.Node) -> bool:
    return arg_reduction_spec(node) is not None


def _emit_arg_reduction(node: torch.fx.Node, ctx) -> TensorRef:
    spec = arg_reduction_spec(node)
    if spec is None:
        raise RuntimeError("hexagon: this arg reduction is outside the DSP boundary")
    source = node.args[0]
    out = ctx.result_for(node, spec.rows, dtype=torch.int64)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_ARGMAX_FP16,
            inputs=[ctx.operand(source)],
            outputs=[out],
            params=[spec.row_size, spec.rows, int(spec.is_min)],
        ),
    )
    return ctx.record(node, out)


def topk_spec(node: torch.fx.Node) -> Optional[TopkSpec]:
    """The row the kernel walks and how many of them, for a topk it answers.

    Every condition is one the command has no argument for: k is 1 by
    construction, the reduction axis is the last because the kernel advances
    `rowSize` elements between rows, and both extents are integers in the
    command, so a symbolic one would describe the shape it was exported at
    instead of the run. `largest` is the same kind of limit -- there is no
    descending walk to select -- and `sorted` is not a limit at all, since one
    element is in order either way.
    """
    if node.target is not TOPK:
        return None
    k = _node_arg(node, "k", 1, None)
    if isinstance(k, bool) or not isinstance(k, int) or k != 1:
        return None
    if _node_arg(node, "largest", 3, True) is not True:
        return None
    source = node.args[0] if node.args else None
    if not isinstance(source, torch.fx.Node):
        return None
    value = source.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.dim() == 0:
        return None
    if not value.is_contiguous():
        return None
    shape = list(value.shape)
    if not all(isinstance(extent, int) and extent > 0 for extent in shape):
        return None
    dim = _node_arg(node, "dim", 2, -1)
    if isinstance(dim, bool) or not isinstance(dim, int):
        return None
    if dim % value.dim() != value.dim() - 1:
        return None
    row_size = shape[-1]
    return TopkSpec(row_size=row_size, rows=value.numel() // row_size)


def topk_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The topk a getitem reads, when it reads the values and not the positions."""
    return _values_getitem(node, TOPK)


def topk_is_emittable(node: torch.fx.Node) -> bool:
    """Whether this topk is one the kernel answers and nothing reads past it.

    The same all-readers-take-the-values rule the pool and max.dim are under, for
    the reason topk_getitem gives; a graph that reads the positions keeps the
    values reader portable with the node it reads. The rule is not only about what
    the kernel computes: the node declares its positions as int64 and this kernel
    writes one int32 per row (topk_ops.cc:54), so a slot wide enough for the
    declared dtype would have its upper half left as the arena held it. The width
    is the smaller of the two reasons -- the position itself is the wrong number
    -- and it is the one that would still stand if a future kernel broke ties
    torch's way.
    """
    if node.target is not TOPK:
        return True
    if topk_spec(node) is None:
        return False
    return bool(node.users) and all(
        topk_getitem(reader) is node for reader in node.users
    )


def _emit_topk(node: torch.fx.Node, ctx) -> TensorRef:
    """k == 1 over the last axis as one TOPKV2_K1_FP16.

    One input and two outputs, in the order the dispatcher reads them
    (execute_command.cc:653): the values first, the positions second, both per
    row. The graph has a slot for the first only, so the second is scratch -- four
    bytes a row, which is the width the kernel writes, and not the eight the
    node's own dtype would ask for. The kernel returns -1 without writing when
    either pointer is null, so this slot has to exist even though nothing reads
    it.
    """
    spec = topk_spec(node)
    if spec is None:
        raise RuntimeError("hexagon: this topk is not one the DSP kernel answers")
    _require_arena_dtype(node, "topk")
    out = ctx.result_for(_values_sink(node), spec.rows)
    indices = ctx.builder.add_activation(spec.rows * INT32_BYTES)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_TOPKV2_K1_FP16,
            inputs=[ctx.operand(node.args[0])],
            outputs=[out, indices],
            params=[spec.row_size, spec.rows],
        ),
    )
    return ctx.record(node, out)


def pow_is_square(node: torch.fx.Node) -> bool:
    """Whether this pow is x ** 2, the one exponent with a unary kernel.

    to_edge leaves both ``x ** 2`` and ``torch.square(x)`` as pow.Tensor_Scalar
    (aten.square.default is not what the exporter emits), and the DSP's
    HTP_OPS_UNARY_SQUARE computes ``(float)x * (float)x``, which is the same
    value as fp16 multiplication for every exponent of two.
    """
    if node.target is not POW_TENSOR_SCALAR:
        return False
    exponent = node.args[1] if len(node.args) > 1 else node.kwargs.get("exponent")
    if isinstance(exponent, bool):
        return False
    return isinstance(exponent, (int, float)) and exponent == 2


def _emit_square_pow(node: torch.fx.Node, ctx) -> TensorRef:
    if not pow_is_square(node):
        raise RuntimeError("hexagon: only x ** 2 has a unary kernel here")
    return _unary("square")(node, ctx)


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


def mean_result_width_is_emittable(node: torch.fx.Node) -> bool:
    """Whether this mean asks for the width the kernel stores.

    Both mean overloads make dtype keyword-only, and the kernel accumulates in
    fp32 and stores fp16, so a requested dtype is a different op rather than a
    narrower command -- exactly the rule `sum_dim_is_emittable` enforces on
    sum.dim_IntList. mean.dim did not carry that check, so
    `torch.mean(x, dim=1, dtype=torch.float32)` reached a kernel that stored
    fp16 and let the runtime widen it: a fp32 tensor whose values were rounded
    to fp16, which is the same defect the sum gate was added for.
    """
    return node.kwargs.get("dtype") is None


def _emit_mean_dim(node: torch.fx.Node, ctx) -> TensorRef:
    """aten.mean.dim as a single REDUCTION."""
    if not mean_result_width_is_emittable(node):
        raise RuntimeError("hexagon: this mean asks for another width")
    return _emit_reduction(node, ctx, REDUCTION_MEAN)


def _emit_mean_default(node: torch.fx.Node, ctx) -> TensorRef:
    """torch.mean(x): every dim, which is the whole buffer as one span.

    The overload has no dim at all, so it is the widest form of what mean.dim
    already emits -- ``[1][numel][1]`` -- and the kernel's inside==1 path is the
    one the whole-tensor mean takes. Same rule, same command, one target name
    further: without this entry `torch.mean(x)` left the graph whole rather than
    failing, which is why it went unnoticed next to mean.dim.
    """
    if not mean_result_width_is_emittable(node):
        raise RuntimeError("hexagon: this mean asks for another width")
    return _emit_reduction(node, ctx, REDUCTION_MEAN)


def _emit_middle_axis_softmax(node: torch.fx.Node, ctx, src, shape):
    """Copy a contiguous middle-axis reduction into the last-axis kernel.

    The two raster regions are deliberately explicit. The first changes only
    the address order, the softmax command then sees `inside == 1`, and the
    second restores the graph's original row-major shape. The support gate has
    already checked both regions and the reduced span. A direct contiguous input
    needs these three commands; an explicit exported permute plus contiguous
    nodes can carry one additional RASTER_BLIT, which is a graph copy rather
    than a different softmax algorithm.
    """
    _, order, inverse, permuted_shape = _softmax_permutation(node)
    forward = _permute_region_from_shapes(shape, permuted_shape, order)
    backward = _permute_region_from_shapes(permuted_shape, shape, inverse)
    if forward is None or backward is None:
        raise RuntimeError("hexagon: softmax permutation is not representable")

    permuted = ctx.activation_for_shape(permuted_shape)
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[ctx.operand(src)],
            outputs=[permuted],
            params=[1, FP16_BYTES, 1] + forward,
        ),
    )

    reduced = ctx.activation_for_shape(permuted_shape)
    channel = int(permuted_shape[-1])
    outside = _upper_product(permuted_shape[:-1], ctx)
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_SOFTMAX,
            inputs=[permuted],
            outputs=[reduced],
            params=[outside, channel, 1, FP16_BYTES],
        ),
    )
    _patch_dynamic_product(ctx, op_index, permuted_shape[:-1], 0)
    _patch_dynamic_product(ctx, op_index, [permuted_shape[-1]], 1)

    out = ctx.result_for(node, _numel(node))
    ctx.emit(
        node,
        Op(
            type=DSP_OP_RASTER_BLIT,
            inputs=[reduced],
            outputs=[out],
            params=[1, FP16_BYTES, 1] + backward,
        ),
    )
    return ctx.record(node, out)


def _emit_softmax(node: torch.fx.Node, ctx) -> TensorRef:
    """One command for a row the kernel is right for, five for a wider row.

    The vendored kernel is wrong for rows longer than one HVX vector, and it is
    wrong inside its exponential rather than in its arithmetic: its vector loop
    computes exp2 of `(x - max) * log2e` about 1.76x too large once the argument's
    fractional part passes 0.35, while the tail it copies in and masks answers the
    same argument within an fp16 ulp. On the device, a (3, 197, 197) softmax over
    uniform logits came back wrong on 116284 of 116427 elements, mean relative
    error 9.6%, row sums still one, and the last five columns -- the tail -- clean;
    at 63 columns, where the whole row is the tail, the same kernel is within
    rounding, and at 64, where the whole row is the chunked part, it is not.

    So the command is kept for the rows it is measured right for and the wider
    ones are written as the same log-sum-exp the log_softmax emitter uses, with a
    division where that one has a log: the maximum, the shift, the exponential,
    the sum of at most ones, and the division by it. Five commands, no new kernel,
    and the same three kernels -- the reduction, the unary table and the
    element-wise op -- the shifted form is already measured on.

    The composition needs the shift for the same reason the log_softmax one does:
    without it, a logit above 11 overflows the fp16 exponential, and the answer
    is a row of infinities.
    """
    src = node.args[0]
    _require_arena_dtype(node, "softmax input")
    dim = int(node.args[1])
    shape = list(node.meta["val"].shape)
    if dim < 0:
        dim += len(shape)
    if dim != len(shape) - 1:
        if not softmax_reduces_the_inner_axis(node):
            raise RuntimeError("hexagon: softmax input is not emittable")
        return _emit_middle_axis_softmax(node, ctx, src, shape)

    outside = _upper_product(shape[:dim], ctx)
    channel = ctx.upper_bound(shape[dim])
    inside = _upper_product(shape[dim + 1 :], ctx)

    numel = _numel(node)
    if channel < SOFTMAX_VECTOR_WIDTH:
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

    outer_shape = shape[:dim]
    span = shape[dim : dim + 1]
    inside_shape = shape[dim + 1 :]
    reduced_shape = outer_shape + [1] + inside_shape
    span_upper = channel

    def reduce_spans(kind: int, source, out_) -> int:
        index = ctx.emit(
            node,
            Op(
                type=DSP_OP_REDUCTION,
                inputs=[source],
                outputs=[out_],
                params=[outside, span_upper, inside, kind, FP16_BYTES],
            ),
        )
        _patch_dynamic_product(ctx, index, outer_shape, 0)
        _patch_dynamic_product(ctx, index, span, 1)
        _patch_dynamic_product(ctx, index, inside_shape, 2)
        return index

    maximum = ctx.activation_for_shape(reduced_shape)
    reduce_spans(REDUCTION_MAXIMUM, ctx.operand(src), maximum)

    shifted = ctx.activation_for_shape(shape)
    _emit_elementwise(
        node, ctx, ctx.operand(src), maximum, "sub", shape, reduced_shape, shape, shifted
    )

    exponentials = ctx.activation_for_shape(shape)
    exp_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_UNARY,
            inputs=[shifted],
            outputs=[exponentials],
            params=[numel, UNARY_OP_TYPES["exp"], FP16_BYTES],
        ),
    )
    _patch_dynamic_product(ctx, exp_index, shape, 0)

    total = ctx.activation_for_shape(reduced_shape)
    reduce_spans(REDUCTION_SUM, exponentials, total)

    out = ctx.result_for(node, numel)
    _emit_elementwise(
        node,
        ctx,
        exponentials,
        total,
        "div",
        shape,
        reduced_shape,
        shape,
        out,
    )
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


def _first_output_getitem(node: torch.fx.Node, source_target):
    """The node a getitem reads, when it reads the first of its outputs.

    native_layer_norm, native_group_norm and the batch norm all hand out
    (out, statistics...) and the DSP has a command for out alone, so getitem 0 is
    the only reader a partition can carry.
    """
    if node.target is not GETITEM or len(node.args) != 2:
        return None
    source, index = node.args
    if index != 0 or not isinstance(source, torch.fx.Node):
        return None
    return source if source.target is source_target else None


def layer_norm_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The layer norm a getitem reads, when it reads the first output."""
    return _first_output_getitem(node, NATIVE_LAYER_NORM)


def group_norm_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The group norm a getitem reads, when it reads the first output."""
    return _first_output_getitem(node, NATIVE_GROUP_NORM)


def batch_norm_getitem(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The batch norm a getitem reads, when it reads the first output."""
    return _first_output_getitem(node, BATCH_NORM_NO_STATS)


def _only_the_first_output_is_read(node: torch.fx.Node, getitem) -> bool:
    """Whether every reader of this multi-output norm takes its first output.

    The statistics come out of the same node and nothing here produces them, so
    a graph that reads one keeps the whole node on the portable kernels -- along
    with the readers of out, which would otherwise be left holding a tuple no
    kernel can be handed.
    """
    return bool(node.users) and all(getitem(reader) is node for reader in node.users)


def layer_norm_is_emittable(node: torch.fx.Node) -> bool:
    """Whether every reader of this layer norm takes the output the kernel writes."""
    return _only_the_first_output_is_read(node, layer_norm_getitem)


def group_norm_is_emittable(node: torch.fx.Node) -> bool:
    """Whether every reader of this group norm takes the output the kernel writes."""
    return _only_the_first_output_is_read(node, group_norm_getitem)


def batch_norm_is_emittable(node: torch.fx.Node) -> bool:
    """Whether every reader of this batch norm takes the output the kernel writes."""
    return _only_the_first_output_is_read(node, batch_norm_getitem)


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


def _declared_extent(value):
    """The extent one of `native_group_norm`'s own arguments names.

    A static export gives the op plain ints for N, C and HxW; a dynamic one gives
    the `sym_size` node they were read from, and its meta holds the symbol whose
    bound the emitter's params are built from -- so that bound is what this
    compares. None means the argument is neither, which is refused rather than
    guessed at.
    """
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        return eval_upper_bound(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def group_norm_normalizes_one_group_per_row(node: torch.fx.Node) -> bool:
    """Whether this group norm is the [rows][inner] view the norm kernel takes.

    Every argument the command cannot carry is refused here rather than at the
    emitter: the group count has to divide the channels, the declared N, C and
    HxW have to be the input's own extents, the epsilon has to be a number
    rather than a run-time tensor, and both weights one element per channel. The
    input has to be contiguous, because the row per (batch, group) the command
    describes is the input's own layout only then -- a batch dim, the channel
    dim and the group axis are what the rows walk.
    """
    if len(node.args) < 8:
        return False
    src = node.args[0]
    if not isinstance(src, torch.fx.Node):
        return False
    value = src.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.dim() < 3
        or not value.is_contiguous()
        or value.dtype not in (torch.float16, torch.float32)
    ):
        return False
    shape = list(value.shape)
    declared = [
        _declared_extent(node.args[3]),
        _declared_extent(node.args[4]),
        _declared_extent(node.args[5]),
    ]
    if None in declared:
        return False
    if declared != [
        _upper_product([shape[0]]),
        _upper_product([shape[1]]),
        _upper_product(shape[2:]),
    ]:
        return False
    channels = _upper_product([shape[1]])
    group = node.args[6]
    if isinstance(group, bool) or not isinstance(group, int) or group <= 0:
        return False
    if channels % group:
        return False
    if _scalar_arg(node, "eps", 7, 1e-5) is None:
        return False
    for operand in (node.args[1], node.args[2]):
        if operand is None:
            continue
        if not isinstance(operand, torch.fx.Node) or _numel(operand) != channels:
            return False
    return group_norm_is_emittable(node)


def batch_norm_normalizes_one_span(node: torch.fx.Node) -> bool:
    """Whether this batch norm is the [rows][inner] view the norm kernel takes.

    Without running statistics the op's statistics are the batch's, whatever the
    training flag says, and the normalization is over every axis but the channel
    one. That is one contiguous span per channel exactly when the batch axis is
    one, which is the view instance_norm exports: [N, C, *spatial] flattened to
    [1, N*C, *spatial] so that each (n, c) pair becomes a channel of its own. A
    wider batch normalizes each channel over the batch as well, which this
    command -- one row per channel, no second axis to fold in -- cannot describe.
    """
    if len(node.args) < 6:
        return False
    src = node.args[0]
    if not isinstance(src, torch.fx.Node):
        return False
    value = src.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.dim() < 3
        or not value.is_contiguous()
        or value.dtype not in (torch.float16, torch.float32)
    ):
        return False
    shape = list(value.shape)
    if _upper_product([shape[0]]) != 1:
        return False
    # The no-stats op always reduces the batch; this flag only says which
    # statistics a caller believed it was asking for. `training=False` on it
    # returns nothing at all in this torch build (a segfault, measured), so the
    # graph the exporter produces here carries True.
    if node.args[3] is not True:
        return False
    if _scalar_arg(node, "eps", 5, 1e-5) is None:
        return False
    channels = _upper_product([shape[1]])
    for operand in (node.args[1], node.args[2]):
        if operand is None:
            continue
        if not isinstance(operand, torch.fx.Node) or _numel(operand) != channels:
            return False
    return batch_norm_is_emittable(node)


def _channel_broadcast_shape(channels: int, rank: int) -> tuple:
    """The shape a per-channel weight is read under: ``(1, C, 1, ...)``.

    Every norm here normalizes along one axis of the tensor it is handed and its
    weights sit on that axis, which is dim 1 for all of them. The bytes are the
    operand's own -- a (C,) tensor row-major is a (1, C, 1, 1) tensor -- so this
    describes an existing buffer rather than asking for a copy of it.
    """
    return (1, channels) + (1,) * (rank - 2)


def _emit_elementwise(
    node, ctx, lhs, rhs, kind: str, lhs_shape, rhs_shape, out_shape, out
) -> None:
    """One BINARY_ELEMENTWISE command, with every shape given by the caller.

    `_binary` reads its shapes off the node's own operands, which is right where
    the graph's two tensors are the two the DSP reads. Here they are not: a
    weight the graph declares as (C,) is read as (1, C, 1, 1), and one operand
    is an activation this emitter allocated rather than a node's result. The
    shape each operand is walked under is therefore the caller's to state.
    """
    op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_BINARY_ELEMENTWISE,
            inputs=[lhs, rhs],
            outputs=[out],
            params=[
                _upper_product(out_shape),
                _upper_product(lhs_shape),
                _upper_product(rhs_shape),
                BINARY_OP_TYPES[kind],
                FP16_BYTES,
                FP16_BYTES,
                0,  # inputs are not 4-byte floats
                0,  # output is not a 4-byte float
                *_broadcast_tail(
                    ctx.upper_shape(lhs_shape),
                    ctx.upper_shape(rhs_shape),
                    ctx.upper_shape(out_shape),
                    ctx,
                ),
            ],
        ),
    )
    _patch_dynamic_product(ctx, op_index, out_shape, 0)
    _patch_dynamic_product(ctx, op_index, lhs_shape, 1)
    _patch_dynamic_product(ctx, op_index, rhs_shape, 2)
    for axis, size in enumerate(out_shape):
        if ctx.is_dynamic_dim(size):
            ctx.add_dynamic_patch(op_index, 9 + axis, _dynamic_scale(ctx, size), 0)


def _emit_norm_affine(node, ctx, source, shape, affine, out) -> TensorRef:
    """The mul and the add that apply gamma and beta after the norm.

    The kernel normalizes without them, for the reason rms_norm's emitter gives:
    a subgraph carries no tensors, so a weight reaches it as an operand at the
    width the arena holds, and the affine is the same pair of element-wise
    commands layer_norm already spends on its own. The cost is one rounding to
    fp16 before the multiply and one after the add, on kernels that are fp16 in
    and out regardless.
    """
    rank = len(shape)
    result = source
    for index, (weight, kind) in enumerate(affine):
        target = out if index == len(affine) - 1 else ctx.activation_for_shape(shape)
        _emit_elementwise(
            node,
            ctx,
            result,
            ctx.operand(weight),
            kind,
            shape,
            _channel_broadcast_shape(_numel(weight), rank),
            shape,
            target,
        )
        result = target
    return result


def _emit_group_norm(node: torch.fx.Node, ctx) -> TensorRef:
    """GroupNorm as the norm kernel over one row per (batch, group).

    The op's statistics are one mean and one variance per group over that
    group's channels and the whole spatial block, which is exactly one row of an
    [N*group][(C/group)*HxW] view of the input -- the [outer][inner] span
    DSP_OP_LAYER_NORM already reduces. Nothing else has to be emitted for the
    statistics: the kernel accumulates in fp32 and divides, and its epsilon is
    this op's, so the variance it computes is the one native_group_norm defines
    rather than a second moment computed elsewhere.
    """
    src = node.args[0]
    weight, bias = node.args[1], node.args[2]
    group = int(node.args[6])
    eps = _scalar_arg(node, "eps", 7, 1e-5)
    _require_arena_dtype(node, "group_norm input")

    # The input's own extents rather than the op's arguments, which are the same
    # for every node the gate admits -- and are a `sym_size` node rather than a
    # number when the batch axis is dynamic, which has no upper bound to take.
    shape = list(_value_of(node).shape)
    outer_shape = [shape[0], group]
    inner_shape = [shape[1] // group] + shape[2:]
    rows = _upper_product(outer_shape, ctx)
    inner = _upper_product(inner_shape, ctx)

    sink = next(
        (reader for reader in node.users if group_norm_getitem(reader) is node), node
    )
    out = ctx.result_for(sink, rows * inner)
    affine = [
        (arg, kind) for arg, kind in ((weight, "mul"), (bias, "add")) if arg is not None
    ]
    normalized = ctx.activation_for_shape(shape) if affine else out
    norm_op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_LAYER_NORM,
            # Order matters: dst is the DSP's mapped_ptrs[3], so the three inputs
            # must be exactly src, gamma, beta -- null here, which the kernel
            # tests for and skips.
            inputs=[ctx.operand(src), ABSENT, ABSENT],
            outputs=[normalized],
            params=[rows, inner, _float_bits(eps), 0],
        ),
    )
    # Both counts can hold the run-time length -- the outer one through the batch
    # axis and the inner one through a spatial extent -- so both are patched with
    # the length they were written for, as the reduction emitter patches its own
    # two. Without the inner one the command would reduce the allocation rather
    # than the buffer, which is a wrong answer rather than a failure.
    _patch_dynamic_product(ctx, norm_op_index, outer_shape, 0)
    _patch_dynamic_product(ctx, norm_op_index, inner_shape, 1)
    _emit_norm_affine(node, ctx, normalized, shape, affine, out)
    return ctx.record(node, out)


def _emit_batch_norm(node: torch.fx.Node, ctx) -> TensorRef:
    """InstanceNorm as the norm kernel over one row per (batch, channel).

    The exporter flattens [N, C, *spatial] to [1, N*C, *spatial] so the batch
    norm reduces the spatial block of each (n, c) pair on its own, which is one
    row of the [channels][spatial] view the kernel already reduces. The affine
    is per row here rather than per channel of a nested axis: the weights arrive
    already repeated to the flattened channel count, so one stride-zero operand
    covers them.
    """
    src = node.args[0]
    weight, bias = node.args[1], node.args[2]
    eps = _scalar_arg(node, "eps", 5, 1e-5)
    _require_arena_dtype(node, "batch_norm input")

    shape = list(_value_of(node).shape)
    outer_shape = shape[1:2]
    inner_shape = shape[2:]
    rows = _upper_product(outer_shape, ctx)
    inner = _upper_product(inner_shape, ctx)

    sink = next(
        (reader for reader in node.users if batch_norm_getitem(reader) is node), node
    )
    out = ctx.result_for(sink, rows * inner)
    affine = [
        (arg, kind) for arg, kind in ((weight, "mul"), (bias, "add")) if arg is not None
    ]
    normalized = ctx.activation_for_shape(shape) if affine else out
    norm_op_index = ctx.emit(
        node,
        Op(
            type=DSP_OP_LAYER_NORM,
            inputs=[ctx.operand(src), ABSENT, ABSENT],
            outputs=[normalized],
            params=[rows, inner, _float_bits(eps), 0],
        ),
    )
    _patch_dynamic_product(ctx, norm_op_index, outer_shape, 0)
    _patch_dynamic_product(ctx, norm_op_index, inner_shape, 1)
    _emit_norm_affine(node, ctx, normalized, shape, affine, out)
    return ctx.record(node, out)


def _emit_log_softmax(node: torch.fx.Node, ctx) -> TensorRef:
    """log_softmax as the log-sum-exp the softmax kernel computes internally.

    The short composition -- the SOFTMAX command followed by the unary log of
    its result -- is the obvious one and it is measurably wrong at fp16. The
    softmax kernel stores probabilities two bytes wide, and a row of logits with
    a realistic spread drives the small ones under fp16's subnormal floor: over
    a 1000-way log_softmax of unit-variance logits scaled by three, log(softmax)
    came back infinite on 136 of 4000 outputs where torch's smallest is -23.2,
    and over 16-way rows it is already 8x further from torch than this form.
    Because the softmax's output is the whole of the intermediate, no tolerance
    or gate on the input can recover those values.

    Subtracting the row maximum first is what the kernel does internally anyway,
    and it is the step that has to be visible to the output: x - m is a number
    of the log_softmax's own size, so rounding it costs one fp16 ulp of the
    answer rather than the answer. What is left of the log-sum-exp -- the sum of
    at most 1s and its log -- is small, well-conditioned, and lands on a log
    that the unary table already carries. Six commands, no new kernel: the
    maximum, the shift, the exponential, the sum, the log of it, and the
    subtraction that also removes the shift.
    """
    src = node.args[0]
    _require_arena_dtype(node, "log_softmax input")
    if not softmax_reduces_the_inner_axis(node):
        raise RuntimeError("hexagon: this log_softmax does not reduce the inner axis")

    shape = list(node.meta["val"].shape)
    dim = int(node.args[1]) % len(shape)
    outer_shape = shape[:dim]
    span = shape[dim : dim + 1]
    inside_shape = shape[dim + 1 :]
    reduced_shape = outer_shape + [1] + inside_shape
    outer = _upper_product(outer_shape, ctx)
    span_upper = ctx.upper_bound(shape[dim])
    inside = _upper_product(inside_shape, ctx)
    numel = _numel(node)

    def reduce_spans(kind: int, source, out) -> int:
        index = ctx.emit(
            node,
            Op(
                type=DSP_OP_REDUCTION,
                inputs=[source],
                outputs=[out],
                params=[outer, span_upper, inside, kind, FP16_BYTES],
            ),
        )
        _patch_dynamic_product(ctx, index, outer_shape, 0)
        _patch_dynamic_product(ctx, index, span, 1)
        _patch_dynamic_product(ctx, index, inside_shape, 2)
        return index

    def unary(kind: str, source, out, count, count_shape) -> None:
        index = ctx.emit(
            node,
            Op(
                type=DSP_OP_UNARY,
                inputs=[source],
                outputs=[out],
                params=[count, UNARY_OP_TYPES[kind], FP16_BYTES],
            ),
        )
        _patch_dynamic_product(ctx, index, count_shape, 0)

    # The row maximum, which is the shift the exponentials need and the term the
    # last subtraction removes again.
    maximum = ctx.activation_for_shape(reduced_shape)
    reduce_spans(REDUCTION_MAXIMUM, ctx.operand(src), maximum)

    shifted = ctx.activation_for_shape(shape)
    _emit_elementwise(
        node,
        ctx,
        ctx.operand(src),
        maximum,
        "sub",
        shape,
        reduced_shape,
        shape,
        shifted,
    )

    exponentials = ctx.activation_for_shape(shape)
    unary("exp", shifted, exponentials, numel, shape)

    total = ctx.activation_for_shape(reduced_shape)
    reduce_spans(REDUCTION_SUM, exponentials, total)

    correction = ctx.activation_for_shape(reduced_shape)
    unary("log", total, correction, _upper_product(reduced_shape, ctx), reduced_shape)

    out = ctx.result_for(node, numel)
    _emit_elementwise(
        node,
        ctx,
        shifted,
        correction,
        "sub",
        shape,
        reduced_shape,
        shape,
        out,
    )
    return ctx.record(node, out)


def _emit_getitem(node: torch.fx.Node, ctx) -> TensorRef:
    """Re-points a norm's result, the only getitems this backend takes."""
    source = node.args[0]
    if isinstance(source, torch.fx.Node) and (
        source.target is ADD_RMS_NORM or source.target in SPLIT_TARGETS
    ):
        # A multi-output op records each of its own getitems itself, at the
        # output each index names; re-pointing to the op would collapse them to
        # one.
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


def sdpa_mask_fits_dsp_limits(node: torch.fx.Node) -> bool:
    """Whether an attention mask is one the FLASH_ATTN kernel will apply.

    The kernel takes the mask as `[qo_len, mask_stride]` two-byte rows
    (`attention_entry.cc:72-88,237-242`), where the stride is a command word and
    the row count is the query extent, so every one of these clauses is a
    condition on a number the command carries rather than a preference.

    * The stride is `intParams[7]`, and a param is written once at export: a
      symbolic last extent has no value to write, so it is refused rather than
      patched, because the kernel reads the row at that stride for every query
      row (`preprocess_mask_to_fp32` walks `qo_len` rows of it).
    * The row count is the query extent. A mask with a different count is read
      past or short (the same function, silently), and nothing in the command
      says which.
    * The query extent has to be static and at least two rows. With a positive
      stride the kernel's first-token shortcut is still reachable --
      `flash_attn_try_single_token_output` returns `pV` when `qo_len == 1 &&
      seq_current == 0 && seq_add == 1` before any mask code runs
      (`attention_entry.cc:215-218,364-367`) -- so a graph whose run-time query
      length can be one is a graph whose mask can be silently ignored. A static
      extent of one is that graph; a symbolic extent can become it at run time.
    * The query extent also has to stay at or below `ATTN_PREFILL_SEGMENT_Q`
      (64). The kernel segments a long causal prefill into 64-row blocks
      (`attention_sync_setup.cc:263-265`) but does that *only* while the stride
      is negative; with a mask `task_rows` is the whole query length (`:266`),
      and the per-worker scratch the emitter reserves is sized for 64 rows
      (`_attention_workspace_bytes`). Past 64 the reservation would be short.
    * The width the mask is indexed by is the cache's row count or more, which is
      what makes column `k` the mask of absolute key `k`
      (`attention_sync_process.cc:55-58`: a row's columns are placed at the end
      of the keys, so a wider stride leaves the columns past the cache unread and
      a narrower one masks only the last `stride` keys and leaves the ones before
      them attendable). The cache's own row count bounds the sequence a call can
      attend -- the same invariant the unmasked path rests on -- so covering it
      is what keeps a later, longer call from quietly attending unmasked keys.
      A stride below one is the "no mask" encoding and never a mask.
    """
    if len(node.args) < 4:
        return False
    mask = node.args[4] if len(node.args) > 4 else None
    if mask is None:
        return True
    if not isinstance(mask, torch.fx.Node):
        return False
    nodes = (node.args[0], node.args[1])
    if not all(isinstance(arg, torch.fx.Node) for arg in nodes):
        return False
    value, query, key = (
        mask.meta.get("val"),
        nodes[0].meta.get("val"),
        nodes[1].meta.get("val"),
    )
    if value is None or query is None or key is None:
        return False
    if value.dtype not in (torch.float16, torch.float32):
        # The kernel reads the operand as two-byte elements whatever the graph
        # says, and the runtime narrows an fp32 one on the way in; any other
        # width would be read as half floats.
        return False
    if value.dim() < 2 or query.dim() != 4 or key.dim() != 4:
        return False
    rows, stride, query_rows = value.shape[-2], value.shape[-1], query.shape[1]
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        return False
    if not isinstance(query_rows, int) or isinstance(query_rows, bool):
        return False
    if query_rows < 2 or query_rows > MASK_QUERY_ROWS_MAX:
        return False
    if int(rows) != query_rows:
        return False
    # The kernel places a row's mask columns at the *end* of the keys
    # (`sync_attention_mask_start_pos = N - mask_stride`, clamped at zero,
    # `attention_sync_process.cc:55-58`), so a stride wider than N leaves the
    # columns past the cache unread and a stride narrower than N leaves the keys
    # before it with nothing added. The row count the cache operand holds bounds
    # N -- that is the same invariant the unmasked path already rests on -- so
    # requiring the stride to cover the cache is what makes column `k` the mask
    # of absolute key `k` for every key this call can attend. A narrower mask is
    # a sliding window rather than a wrong answer, but it is indistinguishable
    # from a mask that was built against the wrong length, so it stays portable.
    cache_rows = key.shape[1]
    if not isinstance(cache_rows, int) or isinstance(cache_rows, bool):
        return False
    return stride >= cache_rows


def _attention_workspace_bytes(qo_len, seq_len, n_slots):
    """The per-worker FLASH_ATTN scratch, matching MNN's block scheduler.

    The DSP caps the query block at 64 rows, which is `ATTN_PREFILL_SEGMENT_Q`
    and the row count the kernel segments a causal prefill into; it takes that
    cap only while `mask_stride < 0` (`attention_sync_setup.cc:263-266`), so a
    masked call has to keep its query length at or below 64 rather than expect
    the kernel to segment it. Allocating the full query length here turns a
    2048-token prefill into an unnecessarily enormous buffer and can make the
    command fail before the kernel starts.

    The per-slot size is `sync_attention_head_workspace_bytes(qo_len, seq_len)`
    (`attention_sync_setup.cc:187-196`): a fp32 score row and a fp16
    probability row, each rounded up to 128 bytes. This is that expression
    grouped differently, and the two agree because the first rounding the kernel
    does is of an already 128-aligned number.
    """
    qo_len = min(qo_len, 64)
    padded = (seq_len + 31) // 32 * 32
    scores = (qo_len * padded * 4 + 127) // 128 * 128
    probabilities = (qo_len * padded * 2 + 127) // 128 * 128
    return (scores + probabilities) * max(1, n_slots)


#: The query rows the kernel segments a masked call into, and the widest mask
#: row it will build: `ATTN_PREFILL_SEGMENT_Q` is 64 in the vendored headers and
#: the mask region is the whole query block in fp32.
MASK_QUERY_ROWS_MAX = 64


def _attention_mask_bytes(qo_len, mask_stride):
    """The fp32 mask copy `htp_ops_flash_attn` places after its worker rows.

    The kernel writes `preprocess_mask_to_fp32`'s output at
    `worker_slots * worker_workspace_bytes` inside the caller's workspace
    (`attention_entry.cc:237-242`) and never checks the size of that buffer, so
    the region is the emitter's to size. It is `qo_len` rows of `mask_stride`
    fp32 numbers, which is exactly `qo_len * mask_stride * 4` bytes.

    The paged entry with online pages narrows nothing and reads the operand's
    fp16 rows where they lie (`attention_entry.cc:389-399`), so this region is
    then unused; it is still reserved, because which of the two the kernel picks
    depends on the run-time sequence length.
    """
    return qo_len * mask_stride * 4


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
    mask = args[4] if len(args) > 4 else None
    if mask is not None and not sdpa_mask_fits_dsp_limits(node):
        # The mask's row is read as `mask_stride` two-byte elements, so the
        # stride is a command param and the shape is the whole contract; a
        # geometry the gate refuses is refused here too, for a graph that
        # reaches preprocess without the partitioner (the tests call it that
        # way). The two doors have to agree or the second one hides the first.
        raise RuntimeError(
            "hexagon: sdpa with this attention mask is not emittable: the mask "
            "has to be [query rows, stride] over a static query extent of at "
            "most 64 rows, with a static stride"
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
    if q_shape[0] != 1:
        # The command has no batch field: qo_len is the only row count, the
        # kernel walks `query + q * heads * headDim` for q below it
        # (`attention_entry.cc:164-166`) and writes the same rows back, so a
        # batch of two is one batch computed and one batch left as the arena
        # found it. Refused here as well as at the gate.
        raise RuntimeError(
            f"hexagon: sdpa query batch {q_shape[0]} is not emittable: the "
            "command carries one row count and no batch axis"
        )
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
        # Slot three is the mask. An absent ref is fd = -1, which the
        # dispatcher turns into a null pointer (`execute_command.cc:514`) and
        # the kernel reads as "no mask" -- but only because mask_stride is then
        # -1 as well. The two travel together: a positive stride with a null
        # pointer is non-causal attention over the whole cache, and a null
        # pointer's stride alone cannot say which was meant.
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
    #
    # A mask adds one more region to that scratch -- the fp32 copy the kernel
    # makes of it -- and the kernel computes where it goes rather than being
    # told, so the whole reservation has to cover both. The mask's own rows
    # scale with the query extent, which is why the dynamic form below carries
    # the mask term too and not just the worker rows.
    mask_stride = -1
    mask_region_bytes = 0
    if mask is not None:
        mask_stride = int(mask.meta["val"].shape[-1])
        mask_region_bytes = _attention_mask_bytes(q_len, mask_stride)
    workspace_max_bytes = (
        _attention_workspace_bytes(q_len, q_len + max_kv_len, q_shape[2])
        + mask_region_bytes
    )
    workspace = ctx.builder.add_activation(
        workspace_max_bytes,
        dynamic_layout=ctx.dynamic_bytes_for_function(
            workspace_max_bytes,
            lambda length: _attention_workspace_bytes(
                length, length + max_kv_len, q_shape[2]
            )
            + _attention_mask_bytes(length, mask_stride),
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
                # The mask's row stride: its last extent. -1 is not "no stride"
                # but the other mode -- a negative stride makes the kernel
                # generate the causal clamp itself and ignore slot three
                # (`attention_sync_process.cc:55-58,466-496`), which is what an
                # unmasked sdpa asks for. With a mask the kernel attends all N
                # keys and the mask is the only thing that says otherwise, so a
                # positive stride and a bound operand always travel together.
                mask_stride,  # mask_stride
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
    #: The runtime tensor naming those rows, int32 or int64.
    indices: torch.fx.Node
    #: Bytes one index occupies in the caller's tensor, 4 or 8. The command's
    #: index slot is always four bytes an element; a wider caller's tensor is
    #: narrowed into it on the way in, checked against `oc`.
    index_bytes: int
    #: What a negative index means, which the op this came from decides.
    negative_index: int
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
    indices have to be the caller's own int32 or int64 tensor, and the width is
    what `index_bytes` carries: the kernel reads `const int32_t[]` either way, so
    an int64 tensor is narrowed into a four-byte slot on the way in, with every
    value checked against `oc` (see README).

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
    # The kernel reads the indices as `const int32_t[]`. The caller's tensor
    # keeps the width its own dtype declares, so an int64 index tensor is a slot
    # of eight bytes an element that the kernel would read as alternating low and
    # high words. The slot is declared four bytes an element either way, and the
    # runtime narrows a wider tensor into it on the way in, value by value and
    # refused against `oc` rather than truncated (hexagon_backend.cpp, and
    # `index_bytes` below). Both widths are safe to read at four bytes an
    # element wherever they come from: an op whose result is int32 or int64 is
    # never delegated -- the support check only accepts fp16 and fp32 results --
    # so such a tensor can only reach a command as a method input.
    if indices_value.dtype not in (torch.int32, torch.int64):
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
        8 if indices_value.dtype is torch.int64 else 4,
        (
            NEGATIVE_INDEX_FROM_END
            if node.target is INDEX_TENSOR
            else NEGATIVE_INDEX_REFUSED
        ),
        int(oc),
        int(ic),
        _upper_product(indices_shape),
        indices_shape,
    )


def gather_index_placeholders(graph_module, is_constant) -> frozenset:
    """The method inputs a gather command takes its indices from as int64.

    Those are the slots the blob declares four bytes an element: the kernel reads
    `const int32_t[]`, so an int64 caller's tensor is narrowed into the slot on
    the way in, and the slot's own size is what says so.
    """
    return frozenset(
        fit.indices
        for node in graph_module.graph.nodes
        if node.target in GATHER_TARGETS
        for fit in (gather_table(node, is_constant),)
        if fit is not None and fit.index_bytes == 8 and fit.indices.op == "placeholder"
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
    The five the kernel reads are followed by the two the quantized paths would
    read (scaleBlockNum, scaleAsymmetric), which this table kind never looks at
    and which are written as the dispatcher's own defaults (execute_command.cc:809),
    and then by the width the caller's index tensor has in the .pte, which the DSP
    ignores: the index slot is four bytes an element whatever the command says,
    and the host is what narrows a wider tensor into it.
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
                SHARED_GATHER_SCALE_BLOCK_NUM,
                SHARED_GATHER_SCALE_ASYMMETRIC,
                fit.index_bytes,
                fit.negative_index,
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
DIM_ORDER_TARGETS = frozenset({TO_DIM_ORDER_COPY})

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
    # The bounds as tensors, which the scalar form above cannot hold: two binary
    # commands where the scalar form is one unary. See _emit_clamp_tensor.
    CLAMP_TENSOR: _emit_clamp_tensor,
    # No unary subtype is an elu (1..17) and no binary one is a slope (1..12), so
    # this is six or seven commands of types the table already has. See _emit_elu.
    ELU: _emit_elu,
    # torch computes hardtanh as clamp and gives min_val/max_val the slots
    # clamp's min/max occupy, so one emitter covers both (and relu6, which is
    # F.hardtanh(x, 0, 6)).
    exir_ops.edge.aten.hardtanh.default: _emit_clamp,
    # relu has no entry of its own in the unary table; it is the clamp above
    # with these bounds.
    exir_ops.edge.aten.relu.default: _emit_relu,
    exir_ops.edge.aten.leaky_relu.default: _emit_leaky_relu,
    PRELU: _emit_prelu,
    REFLECT_PAD: _emit_reflect_pad,
    # x ** 2 and torch.square both arrive as pow.Tensor_Scalar.
    POW_TENSOR_SCALAR: _emit_square_pow,
    POW_TENSOR_TENSOR: _emit_tensor_pow,
    ROW_GUARD: _emit_row_guard,
    # One ATen op, two kernels: the depthwise walk and the im2col convolution
    # (see conv_spec for which geometry each takes).
    CONV1D: _emit_convolution,
    CONV2D: _emit_convolution,
    CONV3D: _emit_convolution,
    CONVOLUTION: _emit_convolution,
    exir_ops.edge.aten.tanh.default: _unary("tanh"),
    exir_ops.edge.aten.sqrt.default: _unary("sqrt"),
    exir_ops.edge.aten.rsqrt.default: _unary("rsqrt"),
    # sin and cos run the DSP's own fp32 polynomial rather than an HVX walk, so
    # the error measured for them is the measurement in test_unary_sim.py and
    # not a property of a vector approximation. expm1 is deliberately absent:
    # its HVX form subtracts 1 in fp16, which for |x| below about 1e-3 leaves
    # nothing of the value expm1 exists to compute.
    exir_ops.edge.aten.sin.default: _unary("sin"),
    exir_ops.edge.aten.cos.default: _unary("cos"),
    MIN_DEFAULT: _emit_min_default,
    exir_ops.edge.aten.add.Tensor: _binary("add"),
    exir_ops.edge.aten.sub.Tensor: _binary("sub"),
    exir_ops.edge.aten.mul.Tensor: _binary("mul"),
    exir_ops.edge.aten.div.Tensor: _binary("div"),
    exir_ops.edge.aten.maximum.default: _binary("max"),
    exir_ops.edge.aten.minimum.default: _binary("min"),
    # to_edge rewrites softmax.int into _softmax, which is the name the
    # partitioner then sees.
    exir_ops.edge.aten._softmax.default: _emit_softmax,
    # The log-sum-exp the softmax kernel computes internally, written out as
    # commands: the unary table has a log but no op that fuses it with a
    # softmax, and the two-command composition loses the small probabilities.
    LOG_SOFTMAX: _emit_log_softmax,
    # to_edge grows native_layer_norm out of layer_norm; the functional form is
    # kept for a graph that reaches the backend without that rewrite.
    exir_ops.edge.aten.layer_norm.default: _emit_layer_norm,
    NATIVE_LAYER_NORM: _emit_layer_norm,
    # GroupNorm and InstanceNorm are the same kernel over another view of the
    # input: rows of a group, or rows of a (batch, channel) pair.
    NATIVE_GROUP_NORM: _emit_group_norm,
    BATCH_NORM_NO_STATS: _emit_batch_norm,
    # The getitem that reads a norm's first output.
    GETITEM: _emit_getitem,
    exir_ops.edge.aten.mm.default: _emit_mm,
    exir_ops.edge.aten.bmm.default: _emit_bmm,
    exir_ops.edge.aten.addmm.default: _emit_addmm,
    # The weight-only quantized matmul's weight arrives through this node; the
    # matmul emitter reads it and the node itself emits nothing.
    DQ_PER_CHANNEL: _emit_dequantize,
    exir_ops.edge.aten.mean.dim: _emit_mean_dim,
    MEAN_DEFAULT: _emit_mean_default,
    MAX_DEFAULT: _emit_max_default,
    MAX_DIM: _emit_max_dim,
    SUM_DIM: _emit_sum_dim,
    AMAX: _emit_amax,
    AMIN: _emit_amin,
    MIN_DIM: _emit_min_dim,
    MAX_POOL2D: _emit_pool2d,
    MAX_POOL2D_WITH_INDICES: _emit_pool2d,
    AVG_POOL2D: _emit_pool2d,
    ADAPTIVE_AVG_POOL2D: _emit_pool2d,
    # k == 1 over the last axis; the positions the kernel also writes go to
    # scratch, because they are not the positions torch writes (see TOPK).
    TOPK: _emit_topk,
    ARGMAX: _emit_arg_reduction,
    ARGMIN: _emit_arg_reduction,
    exir_ops.edge.aten.fmod.Tensor: _binary("mod"),
    # `cond ? a : b`. The only ATen node the library's select kernel serves, and
    # the only command here whose condition is one byte per element.
    WHERE: _emit_where,
    # The two order comparisons, as the DSP's own binary op types plus the
    # one-byte select that packs their 1.0 and 0.0 into a bool.
    GREATER_THAN: _emit_compare("greater"),
    LESS_THAN: _emit_compare("less"),
    exir_ops.edge.aten.alias_copy.default: _emit_alias,
    exir_ops.edge.aten.unsqueeze_copy.default: _emit_alias,
    exir_ops.edge.aten.squeeze_copy.dims: _emit_alias,
    exir_ops.edge.aten.view_copy.default: _emit_alias,
    exir_ops.edge.aten.expand_copy.default: _emit_expand_copy,
    TO_DIM_ORDER_COPY: _emit_alias,
    CLONE_DIM_ORDER: _emit_clone_copy,
    exir_ops.edge.aten.select_copy.int: _emit_select_copy,
    exir_ops.edge.aten._to_copy.default: _emit_alias,
    exir_ops.edge.aten.to.dtype: _emit_alias,
    exir_ops.edge.aten.slice_copy.Tensor: _emit_slice_copy,
    SPLIT_WITH_SIZES_COPY: _emit_split,
    SPLIT_COPY: _emit_split,
    exir_ops.edge.aten.cat.default: _emit_cat,
    exir_ops.edge.aten.upsample_nearest2d.vec: _emit_upsample,
    exir_ops.edge.aten.permute_copy.default: _emit_permute_copy,
    # A repeat and a flip are region walks over the operand's own bytes, and
    # the forms that are views of it emit nothing: see repeat_region and
    # flip_region for which is which.
    exir_ops.edge.aten.repeat.default: _emit_repeat_flip,
    exir_ops.edge.aten.flip.default: _emit_repeat_flip,
    # A zero-filling constant pad: a memset for the border and one region for the
    # operand. See constant_pad_region for the shape and the value it takes.
    exir_ops.edge.aten.constant_pad_nd.default: _emit_constant_pad,
    exir_ops.edge.aten.mul.Scalar: _emit_mul_scalar,
    UPDATE_CACHE: _emit_update_cache,
    RMS_NORM: _emit_rms_norm,
    ADD_RMS_NORM: _emit_add_rms_norm,
    MUL_SILU: _binary("mul_silu"),
    ADD_RELU: _binary("add_relu"),
    ROPE: _emit_rope,
    CUMSUM: _emit_cumsum,
    EMBEDDING: _emit_gather,
    INDEX_SELECT: _emit_gather,
    INDEX_TENSOR: _emit_gather,
    # A vision tower's attention, which `FuseVisionAttention` states once.
    VISION_ATTENTION: _emit_vision_attention,
}

# Binary targets that use the rank-eight broadcast-table gate. The support
# check keeps rank-9 operands on a portable kernel because the fixed-width tail
# has no representation for them.
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
        # The comparisons take the same descriptor, and the compare arm walks it
        # with the same broadcast offsets, so they are gated by the same rank.
        GREATER_THAN,
        LESS_THAN,
    }
)

# Same reasoning as BINARY_TARGETS: mm derives its strides from the operand
# shapes, which is only right for contiguous 2-D tiles.
MM_TARGETS = frozenset({exir_ops.edge.aten.mm.default})

# A clamp against tensor bounds, and an elu. Both gates live in hexagon_ops
# (`clamp_tensor_fits`, `elu_fits`) because both emitters read the same things
# those decide: the bounds' strides, and the three coefficients.
CLAMP_TENSOR_TARGETS = frozenset({CLAMP_TENSOR})
ELU_TARGETS = frozenset({ELU})

# The one target whose condition operand is legitimately one byte wide, which is
# why operand_dtypes_are_readable has to know its name.
WHERE_TARGETS = frozenset({WHERE})

# The targets whose own result is a torch.bool, which is why the arena's width
# rule is keyed on the target as well as on the dtype.
COMPARISON_TARGETS = frozenset({GREATER_THAN, LESS_THAN})

# bmm is the same tile geometry with a batch axis in front of both operands.
BMM_TARGETS = frozenset({exir_ops.edge.aten.bmm.default})

# addmm needs the bias it can broadcast and an alpha it can fold away, so what
# the emitter accepts is narrower than the op's own contract.
ADDMM_TARGETS = frozenset({exir_ops.edge.aten.addmm.default})

# REDUCTION collapses one contiguous span, so the reduced dims must be adjacent.

# Views: the operand's bytes read under another shape. Only the forms that
# keep a contiguous layout are listed -- select_copy's other overload reads the
# int64 position tensor down to a scalar, which is not a view of this kind and
# has to stay where the patch mechanism can reach it.
SLICE_TARGETS = frozenset({exir_ops.edge.aten.slice_copy.Tensor})

# Reaches _emit_select_copy, which takes the narrowing form as a blit and leaves
# the same-bytes form to the alias path.
SELECT_TARGETS = frozenset({exir_ops.edge.aten.select_copy.int})

# Reaches _emit_expand_copy, which takes the broadcasting form as a blit with a
# zero source stride and leaves the view form to the alias path.
EXPAND_TARGETS = frozenset({exir_ops.edge.aten.expand_copy.default})

CAT_TARGETS = frozenset({exir_ops.edge.aten.cat.default})

# Reaches _emit_upsample. Only the integer-multiple nearest form has a region
# set; upsample_regions says what the others are missing.
UPSAMPLE_TARGETS = frozenset({exir_ops.edge.aten.upsample_nearest2d.vec})

PERMUTE_TARGETS = frozenset({exir_ops.edge.aten.permute_copy.default})

# A repeat and a flip. Both gates live in the region functions: a repeat whose
# factors are all ones, lead the operand, or sit on a unit axis is a view and
# emits nothing, and a flip of unit axes is the same; every other shape either
# has a region or is refused.
REPEAT_TARGETS = frozenset({exir_ops.edge.aten.repeat.default})

FLIP_TARGETS = frozenset({exir_ops.edge.aten.flip.default})

# The zero-filling pad, whose gate lives in constant_pad_region: an op that is
# two commands, so the shape it is refused on is a property of the region rather
# than of an argument list.
PAD_TARGETS = frozenset({exir_ops.edge.aten.constant_pad_nd.default})

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