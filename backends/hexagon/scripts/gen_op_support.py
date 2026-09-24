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
# BATCH_MATMUL, so both matmul rows below name three commands.
Q4A16_GEMV = "DSP_OP_MATMUL_Q4A16_GEMV_I8"
W8A16_GEMV = "DSP_OP_MATMUL_W8A16_GEMV_I8"
# The prefill entry, which the int4 weight also reaches: M > 1 goes to it instead
# of a GEMV, with a blit either side for the activation and output layouts. The
# int8 weight has no such entry, so a w8a16 matmul stays portable above one row.
Q4A16_PREFILL = "DSP_OP_MATMUL_Q4A16_FP16"
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
    # --- binary family (DSP_OP_BINARY_ELEMENTWISE) -----------------------
    OpSupport(
        "aten.add.Tensor",
        BINARY,
        ARENA_FP16,
        "Operands must be the result's shape or a scalar; the DSP's broadcast path "
        "is unreachable (it needs 25 more params than a command carries). Rank <= 8.",
    ),
    OpSupport(
        "aten.sub.Tensor",
        BINARY,
        ARENA_FP16,
        "Operands must be the result's shape or a scalar; broadcast unreachable. Rank <= 8.",
    ),
    OpSupport(
        "aten.mul.Tensor",
        BINARY,
        ARENA_FP16,
        "Operands must be the result's shape or a scalar; broadcast unreachable. Rank <= 8.",
    ),
    OpSupport(
        "aten.div.Tensor",
        BINARY,
        ARENA_FP16,
        "Operands must be the result's shape or a scalar; broadcast unreachable. Rank <= 8.",
    ),
    OpSupport(
        "aten.maximum.default",
        BINARY,
        ARENA_FP16,
        "Operands must be the result's shape or a scalar; broadcast unreachable. Rank <= 8.",
    ),
    OpSupport(
        "aten.minimum.default",
        BINARY,
        ARENA_FP16,
        "Operands must be the result's shape or a scalar; broadcast unreachable. Rank <= 8.",
    ),
    OpSupport(
        "et_hexagon.mul_silu.default",
        BINARY,
        ARENA_FP16,
        "Fused gated activation `a * silu(b)`; produced by mul_silu.py from "
        "`mul(sigmoid(x), x)`. Same shape/scalar operand rule as the binary family.",
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
        "condition is not delegated -- its result is bool and no kernel here writes "
        "one -- so the condition arrives as a delegate input or as a bool constant "
        "weight.",
    ),
    # --- matmul family (DSP_OP_BATCH_MATMUL) -----------------------------
    OpSupport(
        "aten.mm.default",
        f"{MATMUL} / {Q4A16_GEMV} / {W8A16_GEMV} / {Q4A16_PREFILL}",
        ARENA_FP16,
        "Contiguous 2-D operands only; contraction dims must match; all sizes and "
        "steps must fit int32. A visible constant weight with m*k*n >= 32768 is "
        "pre-packed in the HMX tile order at export. A weight that arrives as a "
        "per-channel dequantize is emitted instead: one GEMV command at M == 1, or "
        "the prefill entry with a pack blit and a repack blit above it. Both need "
        "K % 64 == 0 and N % 32 == 0, and only the int4 weight has a prefill entry.",
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
        f"{MATMUL} / {Q4A16_GEMV} / {W8A16_GEMV} / {Q4A16_PREFILL}",
        ARENA_FP16,
        "alpha must be 1 and beta 0 or 1; 2-D contiguous matmuls; the bias is read "
        "right-aligned against the 2-D result, so at most 2-D and broadcastable. "
        "Emitted as the matmul plus one broadcast add, or as one quantized command "
        "with the bias as its last operand when the weight is quantized (K % 64 == 0, "
        "N % 32 == 0, and the bias exactly n values; M == 1 reaches the GEMV entries "
        "and M > 1 the int4 prefill entry).",
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
        "Same single-span rule as mean.dim. There is no minimum in "
        "HtpOpsReductionOpType (sum, maximum, mean), so amin stays portable.",
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
        "C == 64 only, which is the one channel count whose DSP activation "
        "blocking coincides with a single 64-channel block; 3-D or 4-D operand; "
        "static shapes; dilation 1, ceil_mode false, and every window holding at "
        "least one input element. Three commands: one blit into the blocked "
        "layout, the pool, one blit back out (a 1x1 spatial extent needs neither "
        "blit, the layouts already agree).",
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
        "divisor_override has no command form and keeps the node portable.",
    ),
    OpSupport(
        "aten.convolution.default",
        f"{DEPTHWISE} / {IM2COL}",
        "fp16 (arena); fp32 narrowed on entry",
        "A 4-D batched input, static extents, and a weight and bias that are "
        "constants at export. Two forms, chosen by the group count. A group per "
        "channel (weight `[C, 1, ky, kx]`, `groups == C_in == C_out`) runs the "
        "depthwise walk: blit in, the walk, blit out. Every other supported "
        "convolution has `groups == 1` and runs the im2col convolution the same way, "
        "where the pack and the unpack convert between the row-major tensor and the "
        f"64-channel blocked layout; a channel count that is not a multiple of 64 "
        f"also emits {ZERO} ahead of the pack, because the fill reads whole 64-lane "
        "vectors and the lanes past the last channel would otherwise hold whatever "
        "the arena last held. A 1x1 spatial extent needs neither blit, since the two "
        "layouts are then the same bytes. Left portable: transposed convolution, "
        "non-zero output_padding, a group count between 1 and C_in, a fractional or "
        "negative padding, a non-4-D or unbatched operand, symbolic extents, a weight "
        "or bias that is not a constant, and any stride, padding or dilation that "
        "does not reproduce the output extent the graph declares.",
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
        "Operands must be the result's shape or a scalar, as the other binary ops. "
        "The kernel computes the truncated remainder (a - trunc(a/b)*b), which is "
        "fmod and not torch's floored remainder; a zero divisor answers 0 where "
        "torch gives NaN.",
    ),
    OpSupport(
        "aten._softmax.default",
        SOFTMAX,
        ARENA_FP16,
        "Last axis only. The kernel's strided reduction over any other axis "
        "disagrees with torch on hardware, so softmax_reduces_the_inner_axis keeps "
        "those nodes portable.",
    ),
    # --- norms -----------------------------------------------------------
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
        "One to three inputs (a region is 12 ints and only 3 fit in a command); all "
        "contiguous fp16; only the concatenated axis differs and the lengths add "
        "up. Every length must be known when the command is built.",
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
        "head_dim, n_kv_heads dividing the query heads; start_pos must be a "
        "constant or a run-time tensor read; no attention mask (the emitter "
        "has no stride to hand the kernel, and the fp32 workspace the mask would "
        "be copied into is sized for the unmasked shape, so a masked node stays "
        "portable). One non-paged FLASH_ATTN.",
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
        "tiled at export, and the indices an int32 tensor: the kernel reads "
        "`const int32_t[]` into a 32x32 tiled fp16 table. An index outside the "
        "table clears its row rather than raising, which torch does not.",
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
        "aten.convolution.default with transposed=True, or output_padding != 0",
        "No transposed-convolution kernel. The im2col kernel walks its window "
        "forward over the input, which is a scatter read for a transposed "
        "convolution: a different kernel rather than another parameter set.",
    ),
    (
        "aten.gather.default",
        "An element-wise index along one axis: out[i][j] = input[index[i][j]][j] "
        "coincides with a whole-row read only for particular index shapes, so an "
        "approximation would be a wrong answer rather than a slow one.",
    ),
    (
        "aten.embedding.default with int64 indices, or a table that is not a "
        "constant",
        "The kernel reads `const int32_t[]` and its table is tiled at export, so "
        "neither a width-8 index nor a tensor that only exists at run time has a "
        "command form. A `tokens.to(torch.int32)` in the model is enough to reach "
        "the DSP: the cast itself stays portable.",
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
        "prefill (M > 1) with a quantized int8 weight",
        "The GEMV entries read one activation row linearly and the prefill entry "
        "reads the int4 weight's tile order; no kernel reads the int8 weight with "
        "M > 1 in a layout this backend packs, so those nodes stay on the portable "
        "kernels. The int4 weight does prefill: the prefill entry takes it, with one "
        "blit to put the activation in the blocked layout and one to move the "
        "kernel's 64-channel output packs back to rows.",
    ),
    (
        "w8a8 with a static activation scale",
        "No kernel accepts an int8 activation tensor: both GEMV entries quantize the "
        "fp16 row inside the kernel, per token, with an uncalibrated absmax scale.",
    ),
    (
        "softmax over a non-last axis",
        "The kernel's strided reduction path disagrees with torch on hardware.",
    ),
    (
        "broadcasting binary ops",
        "The DSP's broadcast path needs 25 more params than a command carries, so "
        "only same-shape and scalar operands work.",
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
        "aten.amin.default, aten.min.default and aten.min.dim",
        "HtpOpsReductionOpType is sum, maximum and mean (eltwise_ops.cc:2441-2445) "
        "and the dispatcher rejects anything else, so there is no minimum to select "
        "-- this needs a kernel, not an emitter. min.dim is a two-output node whose "
        "second output is indices, which no reduction kernel here produces.",
    ),
    (
        "aten.max.dim reading .indices, and aten.min.dim in either form",
        "The values are what amax computes, but a reader of the indices puts the "
        "node out of reach on the values side too: indices are positions and the "
        "reduction kernel computes values. The two-output shape is the same one "
        "max_pool2d has, and so is the all-readers-are-getitem-0 rule. min.dim is "
        "additionally a minimum, which HtpOpsReductionOpType does not have. "
        "`torch.amax(x, dim)`, `torch.max(x)` and `torch.max(x, dim).values` "
        "reach the covered targets instead.",
    ),
    (
        "aten.argmax.default / aten.argmin.default",
        "A reduction kernel that returns values and no positions.",
    ),
    (
        "aten.split_with_sizes_copy.default and aten.sort.default",
        "Multi-output ops with no producer for the extra outputs. Only the getitems "
        "reading a layer norm's result, a max pool's values, a max(x, dim)'s values, "
        "a topk's values or the fused add+norm's outputs are placed, so these stay "
        "portable together with their getitems.",
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
        "The DSP's comparison writes int32 1/0 or fp16 1.0/0.0 and has no one-byte "
        "mode, so a node declaring torch.bool cannot be handed one without an "
        "out-of-bounds write. A bool operand is refused at the gate for the same "
        "reason (operand_dtypes_are_readable), with SELECT the one exception, "
        "because it is the one command that declares the width it reads. The "
        "comparison stays on the portable kernels and its result reaches a `where` "
        "as an ordinary input, which is why `aten.where.self` is a supported row "
        "and these are not.",
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
        "stays on the portable kernels because its result is bool. `DSP_OP_PRELU` "
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
        "aten.repeat.default, aten.flip.default, aten.constant_pad_nd.default",
        "No command describes them: a tile, an axis reversal and a pad are each a "
        "different region walk from the blits the backend has.",
    ),
    (
        "aten._adaptive_avg_pool2d.default",
        "The pool command takes one fixed window and stride; an adaptive output "
        "sizes the window per output position.",
    ),
    (
        "aten.leaky_relu.default, aten.elu.default, aten._log_softmax.default",
        "No kernel: leaky_relu needs a slope the binary table has no form for, elu "
        "an exponential the unary table does not carry, and log_softmax composes a "
        "log with a softmax in a way no single command describes.",
    ),
    (
        "aten.prod.default, aten.var.correction, aten.cumsum.default",
        "No kernel. The reduction table has sum, maximum and mean; a running "
        "product, a second moment and a prefix scan are each a different walk.",
    ),
    (
        "aten.clamp.Tensor and aten.pow.Tensor_Tensor",
        "The operand is the parameter: clamp's entry point carries its bounds as "
        "two fp16 params and cannot hold a tensor, and there is no pow kernel at "
        "all. Both stay portable rather than being read as a scalar operand.",
    ),
    (
        "aten.cat.default with more than three operands",
        "A command holds at most three 12-int regions.",
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
        "aten.split / getitem of a split",
        "No producer for the extra outputs; only the getitems that read a layer "
        "norm's result, a max pool's values, a max(x, dim)'s values or the fused "
        "add+norm's outputs are placed.",
    ),
    (
        "aten.bmm.default with a broadcast batch",
        "One tile geometry and one step are derived from the shapes; a broadcast "
        "batch is not described by them.",
    ),
    (
        "llama.custom_sdpa shapes other than the one FLASH_ATTN form",
        "Non-4-D operands, a head_dim mismatch, or n_kv_heads that does not divide "
        "the query heads stay portable.",
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
    "dim_order_keeps_the_bytes",
    "update_cache_layout",
    "sdpa_targets",
    "layer_norm_normalizes_the_trailing_dims",
    "layer_norm_is_emittable",
    "add_rms_norm_is_emittable",
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
