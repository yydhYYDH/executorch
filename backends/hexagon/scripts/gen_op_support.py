#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate ``backends/hexagon/OP_SUPPORT.md`` from the emitter table.

The set of supported ops is read from ``hexagon_ops.EMITTERS`` (re-exported as
``hexagon_backend.SUPPORTED_TARGETS``), so the table cannot drift from the code:
an emitter added without a row here fails the generator instead of silently
leaving the documentation stale. The DSP op type, dtype and constraints are
curated from the emitters and the support predicates in
``partition/hexagon_partitioner.py``; the predicate functions themselves are
imported so the generator breaks loudly if they are renamed.

Run from the repository root:

    PYTHONPATH=src python backends/hexagon/scripts/gen_op_support.py

Pass ``--check`` to fail instead of writing when the checked-in file is stale.
"""

import argparse
import operator
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from executorch.backends.hexagon import hexagon_ops as ops
from executorch.backends.hexagon.hexagon_backend import SUPPORTED_TARGETS

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / "backends" / "hexagon" / "OP_SUPPORT.md"

# DSP op type constants, named rather than numbered so a change in
# hexagon_ops.py is picked up here.
UNARY = "DSP_OP_UNARY"
BINARY = "DSP_OP_BINARY_ELEMENTWISE"
SOFTMAX = "DSP_OP_SOFTMAX"
REDUCTION = "DSP_OP_REDUCTION"
MATMUL = "DSP_OP_BATCH_MATMUL"
LAYER_NORM = "DSP_OP_LAYER_NORM"
ADD_LAYER_NORM = "DSP_OP_ADD_FUSE_LAYERNORM"
ROPE = "DSP_OP_ROPE"
BLIT = "DSP_OP_RASTER_BLIT"
FLASH_ATTN = "DSP_OP_FLASH_ATTN"
# The weight-only quantized matmul's two GEMV entries. A matmul whose weight
# arrives as a per-channel dequantize is emitted as one of these instead of a
# BATCH_MATMUL, so both matmul rows below name five commands.
Q4A16_GEMV = "DSP_OP_MATMUL_Q4A16_GEMV_I8"
W8A16_GEMV = "DSP_OP_MATMUL_W8A16_GEMV_I8"
# The prefill entries, which both weight widths reach: M > 1 goes to the one
# matching the width instead of a GEMV, with a blit either side for the
# activation and output layouts. The int4 weight is command 22, which takes
# block-wise scales inside the packed weight; the int8 weight is command 42,
# which takes fp16 per-channel scales in a tail appended to the packed tiles.
Q4A16_PREFILL = "DSP_OP_MATMUL_Q4A16_FP16"
W8A16_PREFILL = "DSP_OP_MATMUL_W8A16_BLOCK_FP16"
# The row-wise arg-reduction command. It writes one int64 index per row and
# takes a mode parameter for min versus max.
ARG_REDUCTION = "DSP_OP_ARGMAX_FP16"
# The row gather. One command reads a run of rows out of a fp16 table whose
# bytes the export step rearranged into the 32x32 tiles the kernel walks.
SHARED_GATHER = "DSP_OP_SHARED_GATHER"
# The vision tower's attention. One command computes a whole unmasked
# `softmax(q k^T scale) v` over operands the export laid out token-major, which
# is why the fused op carries the head transposes rather than the head-major
# tensors a matmul wants.
VISION_ATTENTION = "DSP_OP_VISION_ATTENTION_FP16"
# The pool kernel. It carries no op type of its own beyond this one: max and the
# average are selected by a param the emitter fills in.
POOL = "DSP_OP_POOL2D_FP16"
# The convolution family: the depthwise walk, the im2col convolution the HMX unit
# runs, and the memset a channel count that is not a multiple of 64 needs before
# the activation is packed into 64-lane blocks.
DEPTHWISE = "DSP_OP_CONV_DEPTHWISE2D_FP16"
IM2COL = "DSP_OP_IM2COL_CONVOLUTION_FP16"
# The same im2col function under its 1x1 name: 17 is what the stream carries when
# the 1x1 activation fill is the one the kernel's own dispatch will take.
CONV1X1 = "DSP_OP_CONV1X1_DIRECT_FP16"
ZERO = "DSP_OP_ZERO"
# The one kernel in the library that answers two outputs: each row's maximum and
# the first position holding it. The emitter takes the first and puts the second
# in scratch the graph never reads.
TOPKV2_K1 = "DSP_OP_TOPKV2_K1_FP16"
# The element-wise select. It is the one command whose operand is not the arena's
# width: the condition is a torch.bool, and the kernel reads it one byte at a
# time from the slot the blob sizes for it, which is why a comparison's result
# can be handed to it at all.
SELECT = "DSP_OP_SELECT"
# Leaky ReLU and PReLU. Both are the elementwise relu kernel: it multiplies a
# negative input by a slope that rides in params for the leaky form and comes
# from a second operand for the PReLU form.
RELU = "DSP_OP_RELU"
PRELU = "DSP_OP_PRELU"

# The arena holds two bytes per element, so every kernel reads and writes fp16.
# A fp32 operand is narrowed on the way in and a fp32 result widened on the way
# out (hexagon_backend.cpp), which is why both widths reach the same command.
ARENA_FP16 = "fp16 (arena); fp32 narrowed on entry"


@dataclass(frozen=True)
class OpSupport:
    #: Canonical edge op name, e.g. ``aten.abs.default``.
    op: str
    #: DSP op type constant name in hexagon_ops.py, or None when the emitter
    #: produces no command. A row that can emit more than one command joins the
    #: constant names with " / ".
    dsp_op: Optional[str]
    #: What the kernels actually read and write.
    dtype: str
    #: Constraints taken from the emitters and the support predicates.
    constraints: str
    #: Quantization support. "none" unless a real quantized path exists.
    quantization: str = "none"


SUPPORTED: List[OpSupport] = [
    # --- unary family (DSP_OP_UNARY) -------------------------------------
    OpSupport(
        "aten.abs.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.neg.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.gelu.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.sigmoid.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.exp.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.log.default",
        UNARY,
        ARENA_FP16,
        "One input; the DSP's own fast approximation, not the exact logarithm.",
    ),
    OpSupport(
        "aten.silu.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.tanh.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.sin.default",
        UNARY,
        ARENA_FP16,
        "One input. The DSP's own approximation: the angle is reduced in fp32 and "
        "a truncated series evaluated, with no HVX walk, so this is not obviously "
        "faster than the portable kernel. Measured on hexagon-sim against the "
        "correct value: worst 3.9e-4 absolute inside four pi (0.4 fp16 ulp of the "
        "answer) and 2.2e-3 past a thousand radians, where the reduction rather "
        "than the series is what moves.",
    ),
    OpSupport(
        "aten.cos.default",
        UNARY,
        ARENA_FP16,
        "One input. As sin, and the series is one term shorter, so the error at "
        "the ends of the reduced interval is larger: measured worst 8.9e-4 "
        "absolute at `abs(x) == pi/2` (0.9 fp16 ulp) and 1.2e-3 past a "
        "radians. Near cos's own zeros the absolute error is most of the value, "
        "so a model that divides by cos should not rely on this.",
    ),
    OpSupport(
        "aten.sqrt.default", UNARY, ARENA_FP16, "One input; element count preserved."
    ),
    OpSupport(
        "aten.rsqrt.default",
        UNARY,
        ARENA_FP16,
        "One input; the DSP's own fast approximation, not the exact reciprocal sqrt.",
    ),
    OpSupport(
        "aten.clamp.default",
        UNARY,
        ARENA_FP16,
        "Either bound may be omitted; absent bounds are +/-inf. Bounds are narrowed "
        "to the fp16 bit patterns the kernel compares against.",
    ),
    OpSupport(
        "aten.clamp.out",
        UNARY,
        ARENA_FP16,
        "Same kernel as clamp.default; the out= variant is registered separately.",
    ),
    OpSupport(
        "aten.clamp.Tensor",
        BINARY,
        ARENA_FP16,
        "Bounds are tensors rather than the params the unary clamp carries, so "
        "this is `min(max(x, lo), hi)`: one BINARY_ELEMENTWISE per bound, and a "
        "max or a min picks one of its operands without computing, so the answer "
        "is wire-exact. One bound is one command. Each bound is either a single "
        "element or the whole output, at rank 8 or below, and the bound goes in as "
        "the first operand so that an unordered pair -- a NaN activation -- "
        "leaves the activation in place, as torch's clamp does. A NaN *bound* is "
        "dropped rather than propagated, which is what the comparison in front of "
        "a select-based form does too.",
    ),
    OpSupport(
        "aten.mul.Scalar",
        UNARY,
        ARENA_FP16,
        "The scalar must be a python number, not a tensor. It is widened to fp32 "
        "for the multiply and rounded back, matching torch's promotion.",
    ),
    OpSupport(
        "et_hexagon.row_guard.default",
        UNARY,
        ARENA_FP16,
        "Walks whole rows of the last dimension: the mask selects rows that are "
        "entirely the masked value. Row length and pad value ride in params.",
    ),
    OpSupport(
        "aten.leaky_relu.default",
        RELU,
        ARENA_FP16,
        "The negative slope must be a compile-time scalar: it rides in a param, "
        "so a run-time slope has nowhere to live and the node stays portable. "
        "One input; element count preserved.",
    ),
    OpSupport(
        "et_hexagon.prelu.default",
        PRELU,
        ARENA_FP16,
        "The channel axis is the second: source rank >= 2 and contiguous, slope "
        "fp16, contiguous and rank 1 with one entry or one per channel. The "
        "kernel reads the plane, channel and batch extents from params, so all "
        "three are static. Produced from `aten::prelu` by prelu.py's "
        "PreservePRelu pass before decomposition.",
    ),
    OpSupport(
        "et_hexagon.reflect_pad.default",
        BLIT,
        ARENA_FP16,
        "Contiguous source and result, a pad list of two (last axis) or four "
        "(last two axes) non-negative compile-time integers with at least one "
        "above zero, and each pad strictly below the extent it reflects: a pad "
        "of an extent or more reads before the operand. The regions are the "
        "corner, row and column pieces read with a negative stride, packed "
        "several to a command. A symbolic extent is refused because the "
        "offsets and sizes are params.",
    ),
    OpSupport(
        "aten.pow.Tensor_Tensor",
        BINARY,
        ARENA_FP16,
        "A finite integral uniform exponent in {-1, 0, 1, 2, 3, 4}; the base "
        "must be static and the result must fit fp16. There is no pow kernel: "
        "the supported exponents lower to the existing unary or binary "
        "elementwise commands, and any other exponent, a non-integral or "
        "non-uniform one, and a non-finite base stay portable.",
    ),
    # --- binary family (DSP_OP_BINARY_ELEMENTWISE) -----------------------
    OpSupport(
        "aten.add.Tensor",
        BINARY,
        ARENA_FP16,
        "Broadcasting works through rank 8: the 25-entry tail plus the 9-int "
        "command head uses 33 or 34 of the 40-int budget. Rank 9 has no "
        "representation.",
    ),
    OpSupport(
        "aten.sub.Tensor",
        BINARY,
        ARENA_FP16,
        "Broadcasting works through rank 8: the 25-entry tail plus the 9-int "
        "command head uses 33 or 34 of the 40-int budget. Rank 9 has no "
        "representation.",
    ),
    OpSupport(
        "aten.mul.Tensor",
        BINARY,
        ARENA_FP16,
        "Broadcasting works through rank 8: the 25-entry tail plus the 9-int "
        "command head uses 33 or 34 of the 40-int budget. Rank 9 has no "
        "representation.",
    ),
    OpSupport(
        "aten.div.Tensor",
        BINARY,
        ARENA_FP16,
        "Broadcasting works through rank 8: the 25-entry tail plus the 9-int "
        "command head uses 33 or 34 of the 40-int budget. Rank 9 has no "
        "representation.",
    ),
    OpSupport(
        "aten.maximum.default",
        BINARY,
        ARENA_FP16,
        "Broadcasting works through rank 8: the 25-entry tail plus the 9-int "
        "command head uses 33 or 34 of the 40-int budget. Rank 9 has no "
        "representation.",
    ),
    OpSupport(
        "aten.minimum.default",
        BINARY,
        ARENA_FP16,
        "Broadcasting works through rank 8: the 25-entry tail plus the 9-int "
        "command head uses 33 or 34 of the 40-int budget. Rank 9 has no "
        "representation.",
    ),
    OpSupport(
        "et_hexagon.mul_silu.default",
        BINARY,
        ARENA_FP16,
        "Fused gated activation `a * silu(b)`; produced by mul_silu.py from "
        "`mul(sigmoid(x), x)`. The fusion pass matches equal-shaped fp16 "
        "operands; the command itself supports broadcast through rank 8.",
    ),
    # --- the element-wise select (DSP_OP_SELECT) -------------------------
    OpSupport(
        "aten.where.self",
        SELECT,
        "fp16 values, one byte per condition",
        "The condition must be a torch.bool: it is the one operand the blob sizes "
        "at its own width and the kernel reads one byte at a time, so any other "
        "dtype here would be read at the wrong stride. All three operands must be "
        "the output's element count or a single element, and both modes are "
        "reachable: `masked_fill` reaches it with a one-element value. The kernel's "
        "per-channel value mode needs a channel count and an inner size this emitter "
        "does not compute, so it stays portable. The comparison that produces a "
        "condition is not delegated -- it is in neither EMITTERS nor "
        "SUPPORTED_TARGETS -- so the condition arrives as a delegate input or as a "
        "bool constant weight, and a broadcast condition is refused by the extent "
        "rule above whether it arrives from a comparison or from a graph input. "
        "The SDPA-through-CPU-flash guard is the shape that shows this up in a "
        "census: eq -> logical_not -> any(-1) -> logical_not -> where, whose "
        "condition is one flag per query row, so the where is refused on the extent "
        "rule and the whole five-node island stays portable.",
    ),
    # --- matmul family (DSP_OP_BATCH_MATMUL) -----------------------------
    OpSupport(
        "aten.mm.default",
        f"{MATMUL} / {Q4A16_GEMV} / {W8A16_GEMV} / {Q4A16_PREFILL} / {W8A16_PREFILL}",
        ARENA_FP16,
        "Contiguous 2-D operands only; contraction dims must match; all sizes and "
        "steps must fit int32. A visible constant weight with m*k*n >= 32768 is "
        "pre-packed in the HMX tile order at export. A weight that arrives as a "
        "per-channel dequantize is emitted instead: one GEMV command at M == 1, "
        "or the prefill entry for that weight's width with a pack blit and a "
        "repack blit above it. All three entries need K % 64 == 0 and N % 32 == 0.",
        "weight-only int4 or int8, per-channel symmetric; the GEMV entries quantize "
        "the fp16 activation to int8 per token, where the prefill entry multiplies "
        "fp16 activations against the weight it dequantizes itself",
    ),
    OpSupport(
        "aten.bmm.default",
        MATMUL,
        ARENA_FP16,
        "Contiguous 3-D operands with equal batch and matching contraction; one "
        "iteration per batch element. No broadcast batch.",
    ),
    OpSupport(
        "aten.addmm.default",
        f"{MATMUL} / {Q4A16_GEMV} / {W8A16_GEMV} / {Q4A16_PREFILL} / {W8A16_PREFILL}",
        ARENA_FP16,
        "alpha must be 1 and beta 0 or 1; 2-D contiguous matmuls; the bias is read "
        "right-aligned against the 2-D result, so at most 2-D and broadcastable. "
        "Emitted as the matmul plus one broadcast add, or as one quantized command "
        "with the bias as its last operand when the weight is quantized (K % 64 == 0, "
        "N % 32 == 0, and the bias exactly n values; M == 1 reaches the GEMV entries "
        "and M > 1 the prefill entry for the weight's width).",
        "weight-only int4 or int8, per-channel symmetric; the GEMV entries quantize "
        "the fp16 activation to int8 per token, where the prefill entry multiplies "
        "fp16 activations against the weight it dequantizes itself",
    ),
    OpSupport(
        "quantized_decomposed.dequantize_per_channel.default",
        None,
        "the stored int4/int8 weight is read by the quantized matmul entries; this node "
        "emits no "
        "command of its own",
        "The scaling a weight-only quantized matmul carries. Delegated only when "
        "every reader is a quantized matmul one of the quantized entries admits; the "
        "matmul packs the stored low-bit weight itself, so this node records an "
        "ABSENT operand rather than materializing anything.",
        "weight-only int4 or int8, per-channel symmetric",
    ),
    # --- reductions / softmax --------------------------------------------
    OpSupport(
        "aten.mean.dim",
        REDUCTION,
        ARENA_FP16,
        "Reduced dims must be one contiguous span (the kernel collapses a single "
        "[outside][reduce][inside] view); rank >= 1. A missing dim and an empty "
        "dim list both mean every dim -- torch reduces the whole tensor for "
        "`mean(x, dim=())` exactly as for `mean(x)` -- which is the one span "
        "[1][numel][1]. The dtype argument must be absent, as on sum.dim_IntList: "
        "the kernel accumulates in fp32 and stores fp16, so a mean that names "
        "another width is a different op rather than a narrower command.",
    ),
    OpSupport(
        "aten.mean.default",
        REDUCTION,
        ARENA_FP16,
        "Every dim, as the single span [1][numel][1]. The overload carries no dim "
        "argument at all, so there is nothing that could be non-adjacent. This is "
        "the target `torch.mean(x)` produces where `torch.mean(x, dim=1)` produces "
        "mean.dim, and it emits the same command as `torch.mean(x, dim=None)`. The "
        "dtype rule is mean.dim's.",
    ),
    OpSupport(
        "aten.sum.dim_IntList",
        REDUCTION,
        ARENA_FP16,
        "Same single-span rule as mean.dim, with a missing or empty dim read as "
        "every dim. The dtype argument must be absent: the kernel's accumulator is "
        "fp32 and its result fp16, so summing into any other width is a different "
        "op rather than a narrower command.",
    ),
    OpSupport(
        "aten.amax.default",
        REDUCTION,
        ARENA_FP16,
        "Same single-span rule as mean.dim. The minimum is reduction op 4 in "
        "HtpOpsReductionOpType, over the same [1][numel][1] span, so amin reduces "
        "through the same command amax uses with op 4 instead of op 2.",
    ),
    OpSupport(
        "aten.max.default",
        REDUCTION,
        ARENA_FP16,
        "Every element, as the same [1][numel][1] span amax takes and through the "
        "same kernel. `torch.max(x)` and `torch.amax(x)` are the same value; they "
        "differ only in which zero a signed-zero input returns and in the payload "
        "of a NaN, which is the byte-level caveat fmod already carries. This is "
        "the target the reduce-all overload produces; `torch.max(x, dim)` is "
        "max.dim below.",
    ),
    OpSupport(
        "aten.min.default",
        REDUCTION,
        ARENA_FP16,
        "Every element, the minimum of them: the same single-span reduction amax "
        "performs, with op 4 rather than op 2 in the command's reduction type. "
        "`torch.min(x)` and `torch.amin(x)` are the same value and reach the same "
        "command; they differ only in the signed-zero and NaN payload caveats that "
        "amax already carries.",
    ),
    OpSupport(
        "aten.amin.default",
        REDUCTION,
        ARENA_FP16,
        "`torch.amin(x)`, the keepdim-free spelling of the same reduce-all minimum; "
        "the span rule is the one amax and mean.dim already use.",
    ),
    OpSupport(
        "aten.min.dim",
        REDUCTION,
        ARENA_FP16,
        "The values of `torch.min(x, dim)`, which is the minimum over the same "
        "single span amax reduces: dim is a single int, so the span rule always "
        "holds. The node's second output is indices, positions the reduction "
        "kernel does not compute, so the node is placed only when every reader "
        "takes getitem 0 -- min_dim_is_emittable, exactly the rule max.dim is "
        "under. A reader of the indices keeps the values portable too.",
    ),
    OpSupport(
        "aten.max.dim",
        REDUCTION,
        ARENA_FP16,
        "The values of `torch.max(x, dim)`, which are what amax computes over the "
        "same span: the overload's dim is a single int, so the span rule always "
        "holds. The node's second output is indices, which are positions the "
        "reduction kernel does not compute, so it is placed only when every reader "
        "takes getitem 0 -- max_dim_is_emittable, the rule max_pool2d is under. A "
        "reader of the indices keeps the whole node portable, values reader "
        "included. That getitem 0 is where the values are handed on, and it is the "
        "sink the command fills.",
    ),
    OpSupport(
        "aten.topk.default",
        TOPKV2_K1,
        ARENA_FP16,
        "k == 1 over the last axis: the kernel holds one element per row and has "
        "no argument for k, no rank to walk and no descending form, so any other "
        "k, any other dim, `largest=False` and a symbolic extent are all refused "
        "by topk_spec and fall back together. That is also why the two extents "
        "ride in the command as integers: a symbolic row would be a stride no "
        "command can carry. `sorted` is not part of the rule, since one element "
        "is in order either way. The node answers (values, indices) and the "
        "kernel writes both, but the position it writes is the *first* occurrence "
        "of the maximum while torch's own kernel returns whichever index its "
        "partial sort lands on -- neither the first nor the last of a tie in "
        "general -- so it is placed only when every reader takes getitem 0 "
        "(topk_is_emittable, the rule max_pool2d and max.dim are under). The "
        "other half goes to a scratch activation the kernel dereferences and "
        "nothing reads. The values carry the signed-zero and NaN caveat "
        "max.default has.",
    ),
    OpSupport(
        "aten.argmax.default",
        ARG_REDUCTION,
        "fp16 input; int64 output",
        "Contiguous static fp16 input with positive extents. A no-dim form flattens "
        "to one row; a dim form must reduce the last axis. The command writes "
        "one int64 index per row, resolves ordinary ties and signed-zero ties to "
        "the first occurrence, and chooses the first NaN as CPU torch does. Other "
        "axes, non-contiguous inputs, symbolic or scalar inputs, and non-fp16 "
        "inputs stay portable.",
    ),
    OpSupport(
        "aten.argmin.default",
        ARG_REDUCTION,
        "fp16 input; int64 output",
        "The same contiguous static fp16 geometry as argmax.default, with the "
        "command's min mode. It writes int64 positions and preserves the same "
        "first-tie, signed-zero, and first-NaN semantics.",
    ),
    OpSupport(
        "aten.relu.default",
        UNARY,
        ARENA_FP16,
        "clamp between zero and infinity: torch's relu is max(x, 0), and the "
        "clamp entry point is the only form in the DSP's unary table that "
        "expresses it (there is no relu op type). The kernel restores a NaN input "
        "by a bit test, so relu(NaN) is NaN as in torch.",
    ),
    OpSupport(
        "aten.hardtanh.default",
        UNARY,
        ARENA_FP16,
        "min_val and max_val occupy the slots clamp's min and max occupy, and "
        "torch computes hardtanh as clamp, so one emitter covers both. `relu6` is "
        "this op with (0, 6) -- `F.relu6`, `nn.ReLU6` and `nn.Hardtanh` all "
        "arrive here.",
    ),
    OpSupport(
        "aten.elu.default",
        UNARY + " / " + BINARY,
        ARENA_FP16,
        "No unary subtype is an elu (1..17) and no binary one a slope (1..12), "
        "so this is a composition: `max(x, 0) * scale + min((exp(x) - 1) * "
        "alpha * scale, 0)`, which is six commands -- the unary exp, a binary "
        "subtract of one, a binary multiply by `alpha * scale`, the clamp entry "
        "point twice for the relu and the min, and a binary add. Seven when `scale` "
        "is not 1, which is how `nn.SELU` arrives; the extra one is the "
        "`mul_scalar` entry point, because torch's positive branch is `a * scale` "
        "in fp32 rounded once. The rewrite is only the identity when the negative "
        "term is non-positive wherever the positive one is not, so `alpha` and "
        "`scale` must both be non-negative, and `input_scale` must be 1 because "
        "the DSP has no expm1 to fold an argument scale into. Static extents only.",
    ),
    OpSupport(
        "aten.pow.Tensor_Scalar",
        UNARY,
        ARENA_FP16,
        "Exponent 2 only, which is HTP_OPS_UNARY_SQUARE (`(float)x * (float)x`). "
        "`x ** 2` and `torch.square(x)` both arrive as this target -- `to_edge` "
        "emits no aten.square.default -- and they agree with the kernel for every "
        "exponent of two. Another exponent is a different function, not a "
        "narrower command, and stays portable.",
    ),
    OpSupport(
        "aten.max_pool2d.default",
        POOL,
        "fp16 (arena); fp32 narrowed on entry",
        "any channel count: the DSP activation blocking is a count of 64-lane "
        "blocks rather than a shape, c4 carries ceil(C/64) and the kernel walks "
        "every one of them, so a wider pool is the same command with a larger c4. "
        "Also 3-D or 4-D operand; static shapes; dilation 1, and every window "
        "holding at least one input element. ceil_mode is the same command at a "
        "different size: oh and ow are two of the fifteen params and the walk "
        "skips whatever falls outside the input, which is torch's own clip. torch "
        "drops the last output position when it would start at or past the padded "
        "input, so a ceil window runs off the edge without ever being a window "
        "over nothing. Three commands per three blocks: one blit carrying a "
        "region per block into the blocked layout, the pool, one blit back out (a "
        "1x1 spatial extent over whole blocks and a batch of one needs neither, "
        "the layouts already agree), plus a ZERO before the pack when C is not a "
        "multiple of 64, because the kernel loads whole vectors out of lanes the "
        "pack does not write.",
    ),
    OpSupport(
        "aten.max_pool2d_with_indices.default",
        POOL,
        "fp16 (arena); fp32 narrowed on entry",
        "What max_pool2d.default becomes: to_edge decomposes the value-only op "
        "into this one plus a getitem, so this is the form a graph carries. Same "
        "constraints as max_pool2d.default, and every reader must be getitem 0 -- "
        "the kernel produces the values, and a graph that reads the indices keeps "
        "the pool portable.",
    ),
    OpSupport(
        "aten.avg_pool2d.default",
        POOL,
        "fp16 (arena); fp32 narrowed on entry",
        "As max_pool2d.default. count_include_pad selects the divisor the kernel "
        "takes (the window's area, or the positions that landed inside); "
        "divisor_override has no command form and keeps the node portable. Under "
        "ceil_mode a window that hangs off the padded edge divides by the part "
        "still inside it, which is a third divisor the command has no param for: "
        "an average that counts the padding is portable exactly when ceil_mode "
        "changes the shape, and at every geometry where ceil_mode is a no-op it "
        "runs. count_include_pad=False divides by the window clipped to the raw "
        "input, which is the kernel's own count, so that one takes ceil mode at "
        "any geometry.",
    ),
    OpSupport(
        "aten._adaptive_avg_pool2d.default",
        POOL,
        "fp16 (arena); fp32 narrowed on entry",
        "As avg_pool2d.default, but only when each input spatial extent is an "
        "exact positive integer multiple of its output extent. Each axis then "
        "has a constant window and a constant stride, both the input/output "
        "quotient, so this is one ordinary pool command. Identity and a "
        "single-position axis are included; a remainder, an output larger than "
        "its input, or a window whose start would clamp is refused because the "
        "adaptive windows are not fixed. A normal (1, 1) request exports as "
        "aten.mean.dim, not as this target, and reaches the reduction command.",
    ),
    OpSupport(
        "aten.conv2d.default",
        f"{DEPTHWISE} / {IM2COL}",
        "fp16 (arena); fp32 narrowed on entry",
        "The same kernels and the same geometry rules as "
        "`aten.convolution.default` with transposed=False; this is the spelled "
        "two-dimensional target `to_edge` produces for an `nn.Conv2d`.",
    ),
    OpSupport(
        "aten.conv1d.default",
        IM2COL,
        "fp16 (arena); fp32 narrowed on entry",
        "A rank-3 input is a rank-4 convolution whose height is 1 and whose "
        "kernel is 1xK, so it runs the im2col walk with no zero insert and no "
        "height padding. Static extents throughout and a constant weight and "
        "bias; transposed is not a form this target has. Left portable: a "
        "grouped or depthwise form, a symbolic extent, and a stride, padding or "
        "dilation that does not reproduce the declared output.",
    ),
    OpSupport(
        "aten.conv3d.default",
        IM2COL,
        "fp16 (arena); fp32 narrowed on entry",
        "A rank-5 input reaches the two-dimensional kernel only when exactly one "
        "spatial kernel axis is 1 and that axis is also 1 in the input and the "
        "output with stride 1, padding 0 and dilation 1; conv_spec drops that "
        "axis and the rest has to satisfy the rank-4 rules. The two leading "
        "kernel axes both 1 is the same reduction written another way. Static "
        "extents throughout and a constant weight and bias. Left portable: a "
        "transposed form, a genuinely three-dimensional window, a grouped or "
        "depthwise form, a symbolic extent, and any stride, padding or dilation "
        "on the dropped axis that is not the identity.",
    ),
    OpSupport(
        "aten.convolution.default",
        f"{DEPTHWISE} / {CONV1X1} / {IM2COL}",
        "fp16 (arena); fp32 narrowed on entry",
        "A 4-D batched input, static extents, and a weight and bias that are "
        "constants at export. Two forms, chosen by the group count. A group per "
        "channel (weight `[C, 1, ky, kx]`, `groups == C_in == C_out`) runs the "
        "depthwise walk: blit in, the walk, blit out. Every other supported "
        "convolution runs the im2col convolution, one dense command per group, "
        "where the pack and the unpack convert between the row-major tensor and the "
        f"64-channel blocked layout; a channel count that is not a multiple of 64 "
        f"also emits {ZERO} ahead of the pack, because the fill reads whole 64-lane "
        "vectors and the lanes past the last channel would otherwise hold whatever "
        "the arena last held. A 1x1 spatial extent needs neither blit, since the two "
        "layouts are then the same bytes. A group count other than 1 is a per-group "
        "partition of the host's: one raster region copies each group's input "
        "channels and weight rows into a dense walk that sees the group-local "
        "counts, and a second region puts the result back at the group's ordinal "
        "output channels. The command count is linear in the group count and no "
        "single command uses more parameters than the ungrouped form. A transposed "
        "convolution is the same two kernels reached through the identity "
        "`conv_transpose(x, w, s, p, op) == conv2d(zero_insert(x, s, op), "
        "flip(w).transpose(ic, oc), d * (k - 1) - p, dilation=d)`: the weight is "
        "transposed and flipped at export, the transposed padding is remapped onto "
        "the convolution's, and a stride above 1 makes the input interleaved with "
        f"zeros by {ZERO} and one raster region ahead of the usual pair. The DSP "
        "im2col walk consumes the dilation fields directly. A stride of 1 needs no "
        "interleave. Left portable: a grouped convolution whose height is a "
        "run-time length, a fractional or negative padding, a non-4-D or unbatched "
        "operand, symbolic extents, a weight or bias that is not a constant, and "
        "any stride, padding or dilation that does not reproduce the output extent "
        "the graph declares.",
    ),
    OpSupport(
        "et_hexagon.add_relu.default",
        BINARY,
        ARENA_FP16,
        "The fused `max(a + b, 0)`, which no ATen op produces: it is reached by "
        "putting `FuseAddReluPass` (add_relu.py) in the caller's "
        "`transform_passes`, which rewrites `relu(x + y)` into this node. Both "
        "operands must be fp16 tensors, and the operands broadcast the way the "
        "other binary ops do.",
    ),
    OpSupport(
        "aten.fmod.Tensor",
        BINARY,
        ARENA_FP16,
        "Broadcasting works through rank 8: the 25-entry tail plus the 9-int "
        "command head uses 33 or 34 of the 40-int budget. Rank 9 has no "
        "representation. The kernel computes the truncated remainder "
        "(a - trunc(a/b)*b), which is "
        "fmod and not torch's floored remainder; a zero divisor answers 0 where "
        "torch gives NaN.",
    ),
    OpSupport(
        "aten._softmax.default",
        f"{SOFTMAX} / {REDUCTION} / {BINARY} / {UNARY}",
        ARENA_FP16,
        "Last-axis softmaxes, plus contiguous non-last-axis reductions whose "
        "channel is below 64 when the axis can be moved last and back with three-level "
        "raster regions. A non-contiguous or non-permuteable source stays portable, as "
        "does a non-last-axis channel at or above 64. A row shorter than one HVX vector "
        "-- 64 fp16 lanes -- is the SOFTMAX command; a longer one is the shifted sum of "
        "exponentials instead: the maximum over the row, the shift, the exponential, "
        "the sum of at most ones and the division by it. The standalone command's vector "
        "loop answers its exponential up to 1.76x the correctly rounded value, as a "
        "function of the argument's fractional part, where its tail path, which is all "
        "of a row shorter than a vector, is within rounding.",
    ),
    # --- norms -----------------------------------------------------------
    OpSupport(
        "aten.native_group_norm.default",
        f"{LAYER_NORM} / {BINARY}",
        ARENA_FP16,
        "GroupNorm is the norm kernel over one row per (batch, group) of the "
        "input's [N*G][(C/G)*H*W] view: the group count must divide the channels, "
        "the node's N, C and HxW must be the input's own extents, and eps a "
        "compile-time scalar. The kernel takes no weight, so the affine is two "
        "elementwise commands against a per-channel operand, one element per "
        "channel. The input must be contiguous, and every reader of the node has "
        "to be getitem 0 -- the mean and the rstd the op also returns have no "
        "command behind them and keep the whole node portable.",
    ),
    OpSupport(
        "aten._native_batch_norm_legit.no_stats",
        f"{LAYER_NORM} / {BINARY}",
        ARENA_FP16,
        "The batch-of-one view InstanceNorm exports: the exporter flattens "
        "[N, C, *spatial] to [1, N*C, *spatial] so per-channel statistics become "
        "per-(n, c) ones, which is one row per (batch, channel) of an [N*C][H*W] "
        "view. A batch wider than one normalizes each channel over the batch as "
        "well, which one contiguous span per channel cannot describe, so it stays "
        "portable; so does the training flag being false, and any operand that is "
        "not contiguous or any weight that is not one element per row. The affine "
        "is two elementwise commands, as for the group norm.",
    ),
    OpSupport(
        "aten._log_softmax.default",
        f"{REDUCTION} / {BINARY} / {UNARY}",
        ARENA_FP16,
        "Last axis only, and the shifted log-sum-exp: the row's maximum, the "
        "shift, the exponentials, their sum, the log of it, and the subtraction "
        "that removes the shift. The shift is not decoration -- a log of the "
        "softmax command stores its small probabilities as fp16 zeroes and the "
        "log of one is the kernel's -65504 -- and it also bounds what the sum "
        "holds, so a reduced span longer than 65504 elements is refused rather "
        "than saturated. The `half_to_float` argument is ignored, as it is on the "
        "softmax path: the arena is fp16 and the runtime narrows or widens at the "
        "boundary.",
    ),
    OpSupport(
        "aten.layer_norm.default",
        LAYER_NORM,
        "src fp16; gamma/beta fp32",
        "Normalized shape must be the trailing dims; eps must be a compile-time "
        "scalar. The kernel reads gamma and beta as fp32, which no delegate input "
        "holds, so the affine is emitted as separate elementwise mul/add.",
    ),
    OpSupport(
        "aten.native_layer_norm.default",
        LAYER_NORM,
        "src fp16; gamma/beta fp32",
        "Same as layer_norm.default. Every reader must be getitem 0; a graph that "
        "reads mean or rstd keeps the whole node portable.",
    ),
    OpSupport(
        "et_hexagon.rms_norm.default",
        LAYER_NORM,
        "src fp16; gamma fp32",
        "RMSNorm flavour (beta ABSENT, rms flag set). eps must be a compile-time "
        "scalar; gamma is applied by the kernel as fp32.",
    ),
    OpSupport(
        "et_hexagon.add_rms_norm.default",
        ADD_LAYER_NORM,
        "src fp16; gamma fp32",
        "Fused residual add + RMSNorm. Every reader must be getitem 0 or 1; the "
        "command writes the normalized tensor and the residual sum together.",
    ),
    # --- rope ------------------------------------------------------------
    OpSupport(
        "et_hexagon.rope.default",
        ROPE,
        ARENA_FP16,
        "Input is [..., seq, num_head, head_dim] (a one-wide batch folds into seq). "
        "cos/sin are [seq, head_dim] with the even angles in the first half. The k "
        "operand is the same tensor with kv_num_head = 0, so it is inert.",
    ),
    # --- raster blits ----------------------------------------------------
    OpSupport(
        "aten.split_with_sizes_copy.default",
        BLIT,
        ARENA_FP16,
        "One blit per piece, each the narrowing slice that piece stands for. "
        "Contiguous fp16/fp32 source, static shape and piece extents, pieces that "
        "add up to the axis, and every reader a getitem of a piece. A piece nothing "
        "reads keeps its place but emits no command. This is the form `torch.split` "
        "and `torch.chunk` lower to, and the blit a piece emits is byte for byte the "
        "one the equivalent narrowing slice emits.",
    ),
    OpSupport(
        "aten.split_copy.Tensor",
        BLIT,
        ARENA_FP16,
        "The same read with the piece size rather than the piece list, where torch "
        "cuts a short last piece; one more row because the two are separate targets "
        "with one shared spec. Unreachable from `torch.split`, which lowers to the "
        "other spelling, so only a graph naming the functional op reaches it.",
    ),
    OpSupport(
        "aten.slice_copy.Tensor",
        BLIT,
        ARENA_FP16,
        "Narrowing slices only; step must be None or 1; rank preserved and only the "
        "sliced dim may change. A run-time start is patched through the source "
        "offset; the extent comes from the result shape.",
    ),
    OpSupport(
        "aten.cat.default",
        BLIT,
        ARENA_FP16,
        "Any number of contiguous inputs, homogeneously fp16 or fp32: a region is "
        "12 ints and 3 fit in a command, so a longer list is split over as many "
        "blits as it needs, every input still writing its own disjoint slice (four "
        "inputs is two commands, seven is three). Only the concatenated axis "
        "differs and the lengths add up. Every length must be known when the "
        "commands are built.",
    ),
    OpSupport(
        "aten.permute_copy.default",
        BLIT,
        ARENA_FP16,
        "A permutation of all axes whose axes split into at most three ordered "
        "consecutive groups; reversing axes inside a group is refused. A constant "
        "2-D [1, 0] weight transpose is folded at export instead.",
    ),
    OpSupport(
        "aten.select_copy.int",
        BLIT,
        ARENA_FP16,
        "The narrowing form is a one-entry slice emitted as a blit; the form that "
        "keeps the operand's bytes re-points it and emits nothing. int64 position "
        "reads stay on the host.",
    ),
    OpSupport(
        "aten.constant_pad_nd.default",
        f"{ZERO} / {BLIT}",
        ARENA_FP16,
        "Zero fill only: the border is a DSP_OP_ZERO memset over the whole result, "
        "so `value` must be absent or 0. Pads on the last two axes only -- a third "
        "axis from the end would need a fourth level, and a region that dropped it "
        "reads elsewhere -- each non-negative, all extents static. A pad whose "
        "entries are all zero is refused rather than emitted: its region is the "
        "operand's own bytes at its own strides, which the kernel drops as a "
        "self-write, leaving the result unwritten. `mode` is not an argument of this "
        "op: torch lowers reflect and replicate to arange/abs/clamp/index programs "
        "and circular to slice/copy/scatter, so no single pad node reaches the "
        "partitioner for those.",
    ),
    OpSupport(
        "aten.repeat.default",
        BLIT,
        ARENA_FP16,
        "At most one of the operand's own axes above one: that is "
        "cat([x] * factor, dim=axis), two loops, so one region with the factor on "
        "a level's extent rather than on a region per phase -- a factor of 64 costs "
        "the same single region a factor of two does. A factor list of all ones "
        "is the identity and is the only view: a factor on an axis the list adds "
        "in front of the operand's rank tiles the whole tensor rather than "
        "reshaping it, so it copies like any other. Two repeated axes are four "
        "loops against three levels and are refused, as are a non-contiguous "
        "operand, an empty one, and a symbolic extent: the offsets and sizes are "
        "params, and a run-time length would leave them at the traced example.",
    ),
    OpSupport(
        "aten.flip.default",
        BLIT,
        ARENA_FP16,
        "Any subset of the axes, static and contiguous: a reversal is a negative "
        "source stride, which the region's int32 stride and the kernel's signed "
        "walk already carry, and the axes reversed together are one level each. "
        "A subset whose extents are all one re-points the operand and emits no "
        "command. An axis named twice or out of range is refused.",
    ),
    OpSupport(
        "et_hexagon.cumsum.default",
        f"{MATMUL} / {BINARY}",
        ARENA_FP16,
        "Inserted by FuseCumsumPass, which is opt-in (transform_passes). One "
        "BATCH_MATMUL against a host constant (L, L) upper-triangular ones "
        "mask -- row j selects inputs 0..j, so out[r, j] is the prefix ending "
        "at j; the lower triangle is the same plan and the reverse scan -- plus "
        "the BINARY_ELEMENTWISE the graph already wrote for a streaming carry. "
        "fp16 and contiguous, rank >= 2, and the last axis a static multiple "
        "of 64, because the phone skel stages a K-wide row at ceil(K/64)*64 "
        "and reads a shorter one from the wrong place. The accumulator is the "
        "matmul's fp32, narrowed once at the store, and torch.cumsum on an fp16 "
        "CPU tensor accumulates in fp32 the same way, so the two agree bit for "
        "bit at L = 64..1024 and both sit within one fp16 ulp of the output scale "
        "of an exact scan. A scan that rounded to fp16 at every step would drift "
        "by one ulp per frame instead: 0.0078 at L=64, 0.1875 at L=1024.",
    ),
    OpSupport(
        "et_hexagon.update_cache.default",
        BLIT,
        ARENA_FP16,
        "cache and value fp16 and contiguous, rank >= 3, value.shape[2:] == "
        "cache.shape[2:]. Emitted as two blits; the destination row is patched from "
        "the position tensor scaled by one cached position's element count.",
    ),
    # --- attention -------------------------------------------------------
    OpSupport(
        "llama.custom_sdpa.default",
        FLASH_ATTN,
        "fp16; fp32 narrowed on entry",
        "Registered lazily, once the LLM extension defines llama.custom_sdpa, and "
        "only while SDPA_DELEGATION is true. Four-dimensional q/k/v, matching "
        "head_dim, n_kv_heads dividing the query heads, and a batch of one: the "
        "command carries a row count and no batch axis, so a second batch would "
        "have its first batch computed and the rest left alone. start_pos must be "
        "a constant or a run-time tensor read. An attention mask is handed over "
        "where the kernel applies it: `[query rows, stride]` in fp16 or fp32, a "
        "static stride that covers the cache, and a static query extent of at "
        "least two rows and at most 64 (the kernel segments a longer query only "
        "while it generates the causal clamp itself, which a mask replaces). The "
        "stride is a command param and the mask's fp32 copy is a region reserved "
        "past the worker rows; any other geometry stays portable, including a "
        "query extent of one, where the kernel's first-token shortcut returns V "
        "before it reads a mask. One non-paged FLASH_ATTN.",
    ),
    OpSupport(
        "llama.custom_sdpa.out",
        FLASH_ATTN,
        "fp16; fp32 narrowed on entry",
        "Same emitter as llama.custom_sdpa.default; both overloads are registered.",
    ),
    OpSupport(
        "et_hexagon.vision_attention.default",
        VISION_ATTENTION,
        ARENA_FP16,
        "Inserted by FuseVisionAttention, which is opt-in (transform_passes). Three "
        "fp16 [batch, tokens, heads, headDim] operands, a constant scale, and a "
        "square attention: query and key runs have to be equal, which is what a "
        "vision tower's bidirectional attention is and what a causal one is not. No "
        "mask and no causal bias, by construction: the command binds no mask "
        "operand, so the kernel's stride test never reads one. Batch, head count and "
        "head width are params and must be static; the token count is patched from "
        "the run-time length. Emitted as one VISION_ATTENTION_FP16 plus an fp32 "
        "workspace activation.",
    ),
    # --- views / casts / dim-order (no command) --------------------------
    OpSupport(
        "aten.alias_copy.default",
        None,
        ARENA_FP16,
        "Re-points the operand's TensorRef; requires both sides contiguous with "
        "equal element count. A view that would be a partition boundary, or that is "
        "a graph output, is copied instead.",
    ),
    OpSupport(
        "aten.upsample_nearest2d.vec",
        BLIT,
        ARENA_FP16,
        "A 4-D contiguous input whose output extent is an exact integer multiple "
        "of its input extent on both axes, and whose ratio torch's own index "
        "arithmetic agrees with. A replication is many-to-one, so one region "
        "cannot describe it and `s * s` of them do: destination `k * s + t` reads "
        "source `k`, so each phase `t` is an affine map, and every region reads "
        "the whole input at the same strides and differs only in its destination "
        f"offset. Nothing is computed, so the result is the input's bytes. {BLIT} "
        "carries three regions per command, so a factor of `s` takes "
        "`ceil(s * s / 3)` of them -- two commands for the 2x an SD up-block "
        "uses -- and the split is a partition of the work rather than a sequence "
        "of passes. Left portable: a ratio that is not an exact integer multiple, "
        "including a shrinking one, because the runs of repeated source elements "
        "are then of unequal length and the phases stop being a constant stride "
        "apart. `aten.upsample_bilinear2d.vec` and `aten.upsample_bicubic2d.vec` "
        "have no region form at all: their taps carry weights that vary with the "
        "output position, which is arithmetic rather than an index map.",
    ),
    OpSupport(
        "aten.unsqueeze_copy.default",
        None,
        ARENA_FP16,
        "View: re-points the operand; contiguous, equal element count.",
    ),
    OpSupport(
        "aten.squeeze_copy.dims",
        None,
        ARENA_FP16,
        "View: re-points the operand; contiguous, equal element count.",
    ),
    OpSupport(
        "aten.view_copy.default",
        None,
        ARENA_FP16,
        "View: re-points the operand; contiguous, equal element count.",
    ),
    OpSupport(
        "aten.expand_copy.default",
        None,
        ARENA_FP16,
        "View: re-points the operand; contiguous, equal element count (a real "
        "broadcast is not described by the operand's TensorRef).",
    ),
    OpSupport(
        "aten._to_copy.default",
        None,
        ARENA_FP16,
        "fp16 <-> fp32 only: the runtime already converts at the arena boundary, so "
        "no command is emitted. A cast from/to int64 stays portable.",
    ),
    OpSupport(
        "aten.to.dtype",
        None,
        ARENA_FP16,
        "Same as _to_copy.default: fp16 <-> fp32 only, no command.",
    ),
    OpSupport(
        "dim_order_ops._to_dim_order_copy.default",
        None,
        ARENA_FP16,
        "Identity dim_order (or none) only: the arena is row-major two-byte. Any "
        "other order stays on a portable kernel.",
    ),
    OpSupport(
        "dim_order_ops._clone_dim_order.default",
        None,
        ARENA_FP16,
        "Identity dim_order (or none) only; otherwise portable.",
    ),
    OpSupport(
        "operator.getitem",
        None,
        ARENA_FP16,
        "Re-points a producer's result. Only the getitem reading a layer norm's "
        "output 0, a max pool's values, a max(x, dim)'s values, or one of the "
        "fused add+norm's outputs 0/1, is placed.",
    ),
    # --- row gathers (DSP_OP_SHARED_GATHER) -------------------------------
    OpSupport(
        "aten.embedding.default",
        SHARED_GATHER,
        "fp16 table",
        "The table must be a parameter, buffer or lifted constant whose bytes are "
        "tiled at export, and the indices an int32 or int64 tensor: the kernel "
        "reads `const int32_t[]` into a 32x32 tiled fp16 table, so a wider tensor "
        "is narrowed into that slot on the way in and refused rather than "
        "truncated when a value is outside the table. On the int32 path an index "
        "outside the table clears its row rather than raising, which torch does "
        "not.",
    ),
    OpSupport(
        "aten.index_select.default",
        SHARED_GATHER,
        "fp16 table",
        "As embedding.default, and dim must be 0: another dim is a strided read "
        "across rows the command cannot make.",
    ),
    OpSupport(
        "aten.index.Tensor",
        SHARED_GATHER,
        "fp16 table",
        "As embedding.default, and exactly one index, on axis 0. A second entry "
        "indexes a second axis, which is not this command.",
    ),
]

# Exclusions worth naming: each is something a reader might expect to work.
NOT_SUPPORTED = [
    (
        "aten.convolution.default with transposed=False and a group count "
        "other than 1, except the genuine depthwise form",
        "The dense im2col and depthwise kernels do not carry a channel mapping "
        "for an intermediate group count. A grouped transposed convolution is "
        "different: the host partitions its input, weight, bias and output and "
        "emits one dense im2col walk per group. A dilated transposed window is "
        "also supported because the generic im2col kernel reads its dilation "
        "fields directly.",
    ),
    (
        "aten.conv3d.default, aten.conv_transpose3d.input",
        "No 3-D kernel. `Im2ColParameter` carries kernelX, kernelY, iw, ih, ow and "
        "oh and no depth axis at all, so nothing here walks a volume; the two "
        "dimensions above cannot be stretched to three by another parameter.",
    ),
    (
        "aten.gather.default",
        "An element-wise index along one axis: out[i][j] = input[index[i][j]][j] "
        "coincides with a whole-row read only for particular index shapes, so an "
        "approximation would be a wrong answer rather than a slow one.",
    ),
    (
        "aten.embedding.default with a table that is not a constant, or with a "
        "vocabulary past an int32",
        "The table is tiled at export, so a tensor that only exists at run time "
        "has no bytes to rearrange. The vocabulary is a command param and every "
        "element offset in the kernel is derived from it, so a table with more "
        "rows than an int32 can name is refused rather than emitted with "
        "arithmetic that would have wrapped.",
    ),
    (
        "aten.index_select.default with dim != 0, and aten.index.Tensor with more "
        "than one index",
        "The command takes one run of whole rows; a second axis is a strided read "
        "it cannot describe.",
    ),
    (
        "aten.copy_.default (KV writeback)",
        "No emitter. Delegating the auto_functionalized writeback would move the "
        "same bytes back through in_place with nothing gained.",
    ),
    (
        "w8a8 with a static activation scale",
        "No kernel accepts an int8 activation tensor: both GEMV entries quantize the "
        "fp16 row inside the kernel, per token, with an uncalibrated absmax scale.",
    ),
    (
        "non-permuteable or over-wide non-last-axis softmax",
        "Only contiguous reductions below channel 64 take the two-blit path; the old "
        "strided reduction is not used.",
    ),
    (
        "binary broadcasting at rank 9",
        "The broadcast table has no rank-9 representation. Its 25-entry tail plus "
        "the 9-int command head uses 33 or 34 of the 40-int budget, so ranks "
        "through 8 fit but rank 9 does not.",
    ),
    (
        "aten.addmm.default with alpha != 1 or beta not in {0, 1}",
        "alpha has no kernel and beta is the bias's own scale; anything else would "
        "need a multiply the emitter does not produce.",
    ),
    (
        "aten.layer_norm / native_layer_norm with a non-trailing normalized shape "
        "or a run-time eps",
        "The kernel describes the norm as one inner span repeated; a run-time eps is "
        "not a number the command can carry.",
    ),
    (
        "aten.mean.dim with non-adjacent reduced dims",
        "REDUCTION collapses one contiguous span only.",
    ),
    (
        "aten.min.dim reading .indices",
        "The values are the minimum amax-shaped reduction computes, but a reader of "
        "the indices puts the node out of reach on the values side too: indices "
        "are positions and the reduction kernel computes values. The two-output "
        "shape is the same one max_pool2d and max.dim have, and so is the "
        "all-readers-are-getitem-0 rule -- min_dim_is_emittable. "
        "`torch.min(x, dim).values` reaches the covered target instead.",
    ),
    (
        "aten.max.dim reading .indices",
        "The values are what amax computes, but a reader of the indices puts the "
        "node out of reach on the values side too: indices are positions and the "
        "reduction kernel computes values. The two-output shape is the same one "
        "max_pool2d has, and so is the all-readers-are-getitem-0 rule. "
        "`torch.amax(x, dim)`, `torch.max(x)` and `torch.max(x, dim).values` "
        "reach the covered targets instead.",
    ),
    (
        "aten.argmax.default / aten.argmin.default outside the supported "
        "contiguous-last-axis fp16 geometry",
        "The arg-reduction command has one row walk and no general-axis or layout "
        "conversion; unsupported forms stay on portable kernels.",
    ),
    (
        "aten.sort.default",
        "A multi-output op with no producer for the extra outputs. Only the "
        "getitems reading a layer norm's result, a max pool's values, a max(x, "
        "dim)'s values, a topk's values, a piece of a split or the fused add+norm's "
        "outputs are placed, so a sort stays portable together with its getitems: "
        "no command here writes a sorted run.",
    ),
    (
        "aten.topk.default reading .indices, or asking for another k, dim or order",
        "The one kernel that writes positions writes the *first* occurrence of the "
        "row maximum, and torch's own kernel returns whatever index its partial "
        "sort lands on -- for a row of equal values torch answers 2 where this "
        "answers 0, and over 200 rows of quantized values it matches neither the "
        "first nor the last occurrence on 175 of them. A graph that reads the "
        "positions would therefore be reading a number torch never produced, which "
        "is what the all-readers-are-getitem-0 rule refuses. k != 1, another dim "
        "and largest=False are refused by the same gate, because the kernel has "
        "no argument for any of them. `torch.topk(x, 1).values`, `torch.amax(x, "
        "dim)` and `torch.max(x, dim).values` reach the covered targets instead.",
    ),
    (
        "aten.eq / ne / gt / lt / ge / le",
        "No target here is in EMITTERS, so SUPPORTED_TARGETS refuses each of them "
        "at the first gate in the predicate and neither the dtype rule nor the "
        "one-byte question is ever reached. Behind that, the DSP's own comparison "
        "is two op types and not six: HtpOpsBinaryOpType carries GREATER(9) and "
        "LESS(10) and htp_ops_binary_is_compare admits nothing else, so eq, ne, ge "
        "and le have no kernel at any width. The output width is a real limit and "
        "not the first one: htp_ops_binary_elementwise returns -1 for a bytes that "
        "is not 2 or 4, and the compare arms write fp16 1.0/0.0, fp32 1.0/0.0 or "
        "int32 1/0. A bool *operand* is a separate rule and the one "
        "operand_dtypes_are_readable states, about a misread rather than an "
        "overrun, with SELECT the exception because it declares the width it "
        "reads. A comparison that reached a delegate would hand its bool to a "
        "where inside the same arena, so the one-byte mode is worth having "
        "eventually; it is not what is holding these nodes.",
    ),
    (
        "aten.sin / cos / expm1 defaults, and aten.erf.default",
        "erf has no entry in HtpOpsUnaryOpType at all. expm1 has one and its kernel "
        "was measured and rejected: its HVX walk computes exp2 and subtracts 1 in "
        "fp16, so for small arguments the result carries no correct digits -- the "
        "measured relative error is 1.0 -- while a 21-element array of the same "
        "values, which falls through to the kernel's fp32 `expf(x) - 1`, is exact. "
        "Which of the two an array takes depends only on its length. What it does "
        "past fp16's range is not the reason: it answers a small finite number "
        "where the correct value overflows, and aten.exp is already wired with "
        "exactly that behaviour. sin and cos are supported rows above, with the "
        "error they were accepted at.",
    ),
    (
        "aten.prelu.default",
        "No emitter and no fusion pass: the node does not survive export, and the "
        "partitioner sees `gt`, `mul` and `where` in its place. The `mul` and the "
        "`where` delegate today as one multiply and one select, and the comparison "
        "stays on the portable kernels because it is in neither EMITTERS nor SUPPORTED_TARGETS. `DSP_OP_PRELU` "
        "(39) exists in the library and nothing reaches it: fusing the three nodes "
        "back into one command is a pass of the `mul_silu.py` kind, not another "
        "emitter.",
    ),
    (
        "aten.full / full_like / arange / scalar_tensor",
        "Not kernels: nothing emits a tensor that was not read from memory. A "
        "lifted constant is carried as a delegate weight and needs no command, "
        "which is why `x * torch.full(...)` still delegates its multiply.",
    ),
    (
        "aten.repeat.default with two or more axes above one",
        "One repeated axis is cat([x] * factor, dim=axis), which is two loops and "
        "so fits one region with the factor on a level's extent. Two repeated "
        "axes are four loops and do not fit a region's three levels. It is "
        "expressible as a sequence of single-axis regions, but the emitter emits "
        "one command, so the shape is refused rather than half applied.",
    ),
    (
        "aten.prod.default",
        "No kernel. The reduction table has sum, maximum, mean and minimum; a "
        "running product is a different walk. The log/sum/exp substitute that "
        "stands in for it is inexact as well, and a sixteen-element product of "
        "values in [2, 3] already reaches 2**16 and overflows fp16 to infinity.",
    ),
    (
        "aten.var.correction",
        "No kernel, and no composition of the existing commands is a substitute "
        "for it. The two-pass form -- mean, subtract, square, mean, scale -- is "
        "four or five commands of types the table already has and reproduces the "
        "fp16 two-pass bit for bit, but it is not `torch.var`: measured against "
        "torch on fp16 the two differ by 6.2e-04 relative on centred data and "
        "2.5e-03 on data with a mean of 200, and the square is materialised in "
        "fp16, so a deviation past 256 overflows to infinity where torch's "
        "accumulator, which is fp32 throughout, does not. The export cannot bound "
        "the deviation, so the node keeps a portable kernel.",
    ),
    (
        "aten.pow.Tensor_Tensor outside {-1, 0, 1, 2, 3, 4}",
        "The supported exponents are the ones the existing unary and binary "
        "elementwise commands compute exactly, or within the elementwise "
        "tolerance. Any other finite integral exponent, a non-integral or "
        "non-uniform one, and a non-finite base have no command and stay "
        "portable; see the supported row for the forms that do.",
    ),
    (
        "aten.permute_copy.default reversing axes inside a group, or needing more "
        "than three groups",
        "No single blit region describes it; the emitter refuses rather than "
        "reading the wrong elements.",
    ),
    (
        "aten.slice_copy.Tensor with step != 1",
        "A region describes one run per row, so a step is out.",
    ),
    (
        "casts from/to int64, and int64 select_copy",
        "The kernels read two-byte elements; int64 values (such as start_pos) stay "
        "where the patch mechanism can reach them.",
    ),
    (
        "aten.split / getitem of a split over a symbolic axis or with symbolic "
        "piece sizes",
        "A piece's offset and extent are baked into the region list when the "
        "command is built, so both have to be numbers at export. A static split "
        "reaches the covered target instead.",
    ),
    (
        "aten.bmm.default with a broadcast batch",
        "One tile geometry and one step are derived from the shapes; a broadcast "
        "batch is not described by them.",
    ),
    (
        "llama.custom_sdpa shapes other than the one FLASH_ATTN form",
        "Non-4-D operands, a head_dim mismatch, n_kv_heads that does not divide "
        "the query heads, and a batch above one stay portable.",
    ),
    (
        "an attention mask outside the geometry the kernel applies one in",
        "The kernel reads the mask as `[qo_len, mask_stride]` rows at a stride "
        "the command carries, and its first-token shortcut returns V before it "
        "reads any mask, so a run-time query extent, a query extent of one, more "
        "than 64 query rows, a run-time mask stride, a stride narrower than the "
        "cache, a row count that is not the query's, and any dtype that is not "
        "two bytes wide all stay on the portable kernels.",
    ),
    (
        "a vision attention the fusion pass did not state as one op",
        "The fused op is the only thing the emitter knows: the decomposed pattern, a "
        "mask, a causal bias, an unequal query and key run, a scale that is not a "
        "constant, or operands that are not the head transposes of `[batch, tokens, "
        "heads, headDim]` tensors all stay on the portable kernels.",
    ),
]

# Support predicates the constraints above are taken from. Imported so a rename
# fails here rather than leaving the documentation claiming a check that no
# longer exists.
PREDICATES = [
    "softmax_reduces_the_inner_axis",
    "slice_region",
    "select_region",
    "cat_region",
    "permute_region",
    "constant_pad_region",
    "dim_order_keeps_the_bytes",
    "update_cache_layout",
    "sdpa_targets",
    "layer_norm_normalizes_the_trailing_dims",
    "layer_norm_is_emittable",
    "group_norm_normalizes_one_group_per_row",
    "batch_norm_normalizes_one_span",
    "log_softmax_shifts_within_the_arena",
    "add_rms_norm_is_emittable",
    "cumsum_is_emittable",
    "quantized_matmul_is_emittable",
    "conv_spec",
    "vision_attention_is_emittable",
    "pool_spec",
    "reduction_dims",
    "sum_dim_is_emittable",
    "mean_result_width_is_emittable",
    "pow_is_square",
    "max_pool_is_emittable",
    "max_dim_is_emittable",
    "topk_is_emittable",
    "arg_reduction_is_emittable",
    "where_is_emittable",
]


def _canonical(target) -> str:
    """The edge op's canonical name, matching the keys used above.

    An ``EdgeOpOverload``'s ``__name__`` is its qualified overload name
    (``aten.abs.default``); ``str`` renders the whole schema and is not a key.
    """
    if target is operator.getitem:
        return "operator.getitem"
    return target.__name__


def _verify_predicates() -> None:
    missing = [name for name in PREDICATES if not hasattr(ops, name)]
    if missing:
        raise SystemExit(
            "gen_op_support: hexagon_ops no longer defines " + ", ".join(missing)
        )


def _dsp_cell(support: OpSupport) -> str:
    if support.dsp_op is None:
        return "none (no command)"
    return " / ".join(
        f"`{name}` ({getattr(ops, name)})" for name in support.dsp_op.split(" / ")
    )


def _rows() -> List[OpSupport]:
    by_op = {support.op: support for support in SUPPORTED}
    _verify_predicates()

    # Every emitter must have a row: this is the drift guard.
    missing = sorted(
        _canonical(target)
        for target in SUPPORTED_TARGETS
        if _canonical(target) not in by_op
    )
    if missing:
        raise SystemExit(
            "gen_op_support: no OP_SUPPORT row for emitter(s): " + ", ".join(missing)
        )

    # The two tables are one document, so an op named as an exclusion and as a
    # supported row at once is a contradiction -- what an emitter landing on an
    # op the exclusion list still leaves on the host looks like. Only an
    # exclusion that names an op and nothing else counts: the rest are
    # conditional ("with non-adjacent reduced dims", "over a non-last axis") and
    # have to say which form they mean.
    both = sorted(op for op, _ in NOT_SUPPORTED if op in by_op)
    if both:
        raise SystemExit(
            "gen_op_support: row(s) in both tables: "
            + ", ".join(both)
            + " -- an emitter now exists, so drop the exclusion or name the form"
        )

    # The attention overloads resolve lazily; when the extension is loaded, make
    # sure the targets it registers are covered too.
    for target in ops.sdpa_targets():
        name = _canonical(target)
        if name not in by_op:
            raise SystemExit(f"gen_op_support: no OP_SUPPORT row for {name}")

    # The table is the emitter set plus the lazily-registered attention rows,
    # sorted so the output is stable across runs and environments.
    listed = {_canonical(target) for target in SUPPORTED_TARGETS}
    if ops.SDPA_DELEGATION:
        listed.update({"llama.custom_sdpa.default", "llama.custom_sdpa.out"})
    return sorted((by_op[name] for name in listed), key=lambda support: support.op)


def render() -> str:
    lines: List[str] = []
    lines.append("<!-- Generated by backends/hexagon/scripts/gen_op_support.py. -->")
    lines.append("<!-- Do not edit by hand; regenerate with the command below. -->")
    lines.append("")
    lines.append("# Hexagon op support")
    lines.append("")
    lines.append(
        "The DSP kernels read and write two bytes per element, so every delegated op "
        "runs fp16 in the arena. A node the graph declares fp32 is emitted exactly as "
        "its fp16 twin: the runtime narrows a fp32 operand on the way in and widens a "
        "fp32 result on the way out. An op is delegated exactly when "
        "`hexagon_ops.EMITTERS` has an entry for it and the support predicate in "
        "`partition/hexagon_partitioner.py` accepts the node."
    )
    lines.append("")
    lines.append(
        "Regenerate with `PYTHONPATH=src python "
        "backends/hexagon/scripts/gen_op_support.py`."
    )
    lines.append("")
    lines.append("## Supported ops")
    lines.append("")
    lines.append(
        "| PyTorch edge op | DSP op type | Compute dtype | Quantization | Constraints |"
    )
    lines.append("|---|---|---|---|---|")
    for support in _rows():
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                support.op,
                _dsp_cell(support),
                support.dtype,
                support.quantization,
                support.constraints,
            )
        )
    lines.append("")
    lines.append("## Not supported")
    lines.append("")
    lines.append(
        "Each of these is a deliberate exclusion: the node stays on the portable "
        "kernels rather than reaching an emitter that would read it wrong."
    )
    lines.append("")
    lines.append("| PyTorch edge op | Reason |")
    lines.append("|---|---|")
    for op, reason in NOT_SUPPORTED:
        lines.append(f"| `{op}` | {reason} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if OP_SUPPORT.md differs from the generated content.",
    )
    args = parser.parse_args(argv)

    content = render()
    if args.check:
        existing = OUTPUT.read_text() if OUTPUT.exists() else ""
        if existing != content:
            print(f"{OUTPUT} is stale; run the generator", file=sys.stderr)
            return 1
        return 0

    OUTPUT.write_text(content)
    print(f"wrote {OUTPUT} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
