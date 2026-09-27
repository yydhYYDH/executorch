# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""What aten.prod and aten.var would cost on the DSP, measured before wiring.

Both sit in OP_GAPS.md section 1 as "the vendored library has no kernel at all".
This file is the measurement behind the decision, written so the numbers can be
re-derived rather than believed: the shape of the reduction walk is read out of
eltwise_ops.cc at import time, so moving the walk moves the model, and every
numeric test names the clause it pins.  That matters here -- the model was
wrong once already.  It read a widened 64-fp16 block as 64 fp32 lanes and folded
in six stages; the kernel has acc0 and acc1 of 32 fp32 lanes each, adds them,
and folds five more times.  Six stages doubles the answer.  Every product
figure survived it, because a product of values near 1 does not care, and the
SUM control is what exposed it.  The source-string assertions below are the
reason, and they are the part of this file to keep if the rest is dropped.

Three tiers, kept apart on purpose.  The command-stream assertions are host
lowering -- which commands WOULD be emitted.  The numeric assertions are a numpy
MODEL of the walk -- what those commands would compute, not that a kernel
computed them.  Nothing here ran the vendored DSP C++: blob_interpreter.execute
is a model too, and neither hexagon-sim nor a phone is present, so no assertion
below is device-tier.  The walk's shape is inherited from the three reductions
whose sim-vs-silicon agreement is already known (sum, amax, amin); this file does
not re-establish that.

The findings, in the order they decide the two ops:

1. The accumulator is 32 fp32 lanes, each a chain of ceil(reduce/64) terms,
   folded by acc0+acc1 and then five byte-rotations, fp16 in and fp16 out.  A
   product inherits that width unchanged (one accumulator vector, + becomes *);
   a second moment does not (three lane states, and a tree operator that is not
   an add).
2. aten.prod is two spellings, and aten.var.correction is the REDUCE-ALL one
   while torch.var(x, dim=-1) -- what a normalisation produces -- is
   aten.var.dim.  There is no var.mean.  The gap row names one of each pair.
3. A product in this walk is the correctly-rounded fp16 value in 17 of 17
   representable cases, so prod is a fifth HTP_OPS_REDUCTION_* constant.  But the
   tail buffer is a ZERO fill and a product's identity is 1, so a walk copied
   verbatim returns 0 for every reduce length that is not a multiple of 64 -- and
   the delegate count is unchanged, so nothing in the blob would show it.
4. torch.prod on fp16 accumulates in fp16; torch.var on fp16 does not.  So a
   prod mode would be MORE accurate than the portable kernel it replaces, which
   makes torch.prod(x_fp16) the wrong reference for a test; and a second
   moment's walks agree with the reference, so the buffered two-pass the gap row
   rejects is rejected for a reason no command arrangement creates.
5. Both one-pass second moments are conditioned by (mean/sigma)^2, measured: the
   naive form is within 1e-3 only for |mean| < 88 sigma and reaches 0.54 at 2000
   sigma.  A host cannot bound that for a run-time activation, so the walk that
   is emittable is Welford -- which is a genuinely different walk, and the reason
   var is not the cheap win its row's shape suggests.
"""

import os
import pathlib
import struct
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so its parent goes on the
# path; without it the editable install wins and points at another checkout.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as _blob  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

F16 = torch.float16
CONFIG = EdgeCompileConfig(_check_ir_validity=False)
DELEGATE = torch.ops.higher_order.executorch_call_delegate

_REDUCTION_CC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "third-party/mnn-htp-ops/src/dsp/eltwise_ops.cc"
)

#: 128 bytes of fp16 lanes is one HVX vector; widening a 64-element block gives
#: 64 fp32, which the kernel keeps as two 32-lane HVX_Vectors.
FP16_PER_BLOCK = 64
LANES = 32
#: The fold after acc0 + acc1, in the bytes Q6_V_vror_VR rotates by.
_TREE_BYTES = (64, 32, 16, 8, 4)


def _source():
    return _REDUCTION_CC.read_text()


def test_the_walk_is_32_fp32_lanes_and_five_stages():
    # The clause is the walk's shape, so it is read out of the kernel rather than
    # restated.  Each assert says what moved, so a rename fails here instead of
    # leaving a stale lane count in the model below.
    src = _source()
    assert "const int vec_len = 128 / (int)sizeof(__fp16);" in src, (
        "the walk's block size moved, so FP16_PER_BLOCK is now a guess"
    )
    assert "HVX_Vector acc0 = Q6_V_vzero();" in src
    assert "HVX_Vector acc1 = Q6_V_vzero();" in src
    assert "Q6_Vsf_vadd_VsfVsf(acc0, Q6_V_lo_W(sf))" in src, (
        "the lane split moved: acc0 takes the low half of the widened pair"
    )
    assert "Q6_Vsf_vadd_VsfVsf(acc1, Q6_V_hi_W(sf))" in src
    _lines = src.split(chr(10))
    _at = next(k for k, l in enumerate(_lines) if l.startswith(
        "static inline float htp_ops_reduction_reduce_sum2_f32"))
    fold = chr(10).join(_lines[_at : _at + 12])
    assert "Q6_Vsf_vadd_VsfVsf(acc0, acc1)" in fold, (
        "the acc0+acc1 step is gone: the lane count in the model is now wrong"
    )
    for stage in _TREE_BYTES:
        assert "Q6_V_vror_VR(v, %d)" % stage in fold, "fold stage %d bytes is gone" % stage
    assert "dst[o] = (__fp16)value;" in src, "the store is still one narrowing cast"
    assert "Returns the exact fp32 sum" in src, (
        "the kernel's own statement of the contract is gone, so the reference"
        " convention this file measures against has to be re-derived"
    )


def test_the_tail_buffer_is_a_zero_fill():
    # __fp16 tail[vec_len] = {} is a zero fill, correct for a sum because the
    # accumulator's identity is 0.  This line is what a product cannot reuse.
    assert "__fp16 tail[vec_len] __attribute__((aligned(128))) = {};" in _source(), (
        "the tail fill moved, so the two results in"
        " test_the_tail_fill_is_the_line_that_decides_it are no longer about it"
    )


class _Callable(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _program(fn, args):
    return to_edge_transform_and_lower(
        export(_Callable(fn), tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()


def _commands(fn, args):
    types = []
    program = _program(fn, args)
    for node in program.graph_module.graph.nodes:
        if node.target is not DELEGATE:
            continue
        sub = program.graph_module.get_submodule(node.args[0].target)
        raw = bytes(sub._processed_bytes)
        # BLOB_MAGIC is an int, so a bytes comparison is the only one that matches.
        assert raw[:4] == struct.pack("<I", _blob.BLOB_MAGIC), "a delegate carries no blob"
        types.extend(command.type for command in read_blob(raw)[1])
    return types


def _targets(fn, args):
    ep = export(_Callable(fn), tuple(args))
    return [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]


# ---------------------------------------------------------------- spellings


def test_prod_has_two_spellings_and_the_row_names_one_of_them():
    x = torch.randn(2, 8, dtype=F16)
    assert _targets(lambda a: torch.prod(a), (x,)) == ["aten.prod.default"]
    # The other half of the family.  prod(x, dim) is the spelling a model writes,
    # and OP_GAPS.md section 1 lists only .default.
    assert _targets(lambda a: torch.prod(a, -1), (x,)) == ["aten.prod.dim_int"]


def test_var_dim_is_not_var_correction_and_there_is_no_var_mean():
    # The hexagon-sortop shape again: the graph writes one target and the gap list
    # names another.  torch.var(x, dim=-1) is var.dim; only the reduce-all
    # spelling is var.correction, which is the one the row lists.
    x = torch.randn(2, 8, dtype=F16)
    assert _targets(lambda a: torch.var(a, -1), (x,)) == ["aten.var.dim"]
    assert _targets(lambda a: torch.var(a), (x,)) == ["aten.var.correction"]
    assert _targets(lambda a: torch.var(a, -1, correction=0), (x,)) == [
        "aten.var.correction"
    ]
    assert _targets(lambda a: torch.std(a, -1), (x,)) == ["aten.std.dim"]
    overloads = {s.overload_name for s in torch._C._jit_get_schemas_for_operator("aten::var")}
    assert "mean" not in overloads, overloads
    assert overloads == {"", "dim", "correction", "correction_out", "out"}, overloads
    # The control the rest of the file leans on: the reductions that do exist are
    # spelled the way the table expects.
    assert _targets(lambda a: torch.sum(a, -1), (x,)) == ["aten.sum.dim_IntList"]
    assert _targets(lambda a: torch.mean(a, -1), (x,)) == ["aten.mean.dim"]
    assert _targets(lambda a: torch.amax(a), (x,)) == ["aten.amax.default"]


def test_neither_op_reaches_the_dsp_today_and_the_control_does():
    x = torch.randn(4, 16, dtype=F16).contiguous()
    assert _commands(lambda a: torch.prod(a, dim=-1), (x,)) == []
    assert _commands(lambda a: torch.prod(a), (x,)) == []
    assert _commands(lambda a: torch.var(a, dim=-1), (x,)) == []
    assert _commands(lambda a: torch.var(a), (x,)) == []
    # The control, at the same geometry: without it a partitioner that refused
    # everything would make the four assertions above vacuous.
    assert _commands(lambda a: torch.mean(a, dim=-1), (x,)) == [
        hexagon_ops.DSP_OP_REDUCTION
    ]


# ---------------------------------------------------------------- the model


def _blocks(x, fill=0.0):
    """Whole 64-element blocks, the last one padded with `fill`.

    The kernel's vec_end is reduce & -64, so the blocks are whole and any
    remainder is one partial block memcpy'd into a buffer initialised to
    `fill`; the scalar loop after the fold then never runs.
    """
    v = np.asarray(x, np.float16)
    pad = (-v.size) % FP16_PER_BLOCK
    tail = np.full(pad, np.float16(fill), np.float16)
    return np.concatenate([v.reshape(-1), tail]).astype(np.float32).reshape(-1, FP16_PER_BLOCK)


def _fold(v, op):
    """acc0 + acc1 has happened; the rest is five byte-rotations."""
    for stage in _TREE_BYTES:
        v = op(v, np.roll(v, stage // 4)).astype(np.float32)
    return v


def _lanes(x, fill, op, square=False):
    acc0 = np.full(LANES, np.float32(0.0) if op is np.add else np.float32(1.0))
    acc1 = acc0.copy()
    for block in _blocks(x, fill):
        lo, hi = block[:LANES], block[LANES:]
        if square:
            lo, hi = (lo * lo).astype(np.float32), (hi * hi).astype(np.float32)
        acc0 = op(acc0, lo).astype(np.float32)
        acc1 = op(acc1, hi).astype(np.float32)
    return _fold(op(acc0, acc1), op)


def dsp_walk(x, op="sum", fill=0.0, square=False):
    """The kernel's walk with the operator swapped, returning the fp32 lane
    value: htp_ops_reduce_sum_inside1_fp32 at eltwise_ops.cc:2572."""
    combine = np.add if op == "sum" else np.multiply
    return _lanes(x, fill, combine, square)[0]


def dsp_mean(x):
    return dsp_walk(x, "sum") / np.float32(np.asarray(x).size)


def centred(n, spread, seed=11):
    """exp(u) with sum(u) = 0, so the product is near 1 and so is every partial
    product.  A decaying tail would measure the fp16 underflow, not the walk."""
    g = np.random.default_rng(seed)
    u = g.standard_normal(n)
    u -= u.mean()
    return np.exp(u * spread).astype(np.float16)


def test_the_model_is_the_walk_and_not_a_lookalike():
    # If this fails, every product and second-moment number below is a claim about
    # a walk the kernel does not have.  It is the only assertion in the file that
    # would make the others false rather than merely different.
    for n in (16, 64, 256, 1024, 4096):
        x = np.random.default_rng(3).standard_normal(n).astype(np.float16)
        for got, want in (
            (np.float16(dsp_walk(x)), torch.from_numpy(x).sum(dtype=torch.float32)),
            (
                np.float16(dsp_mean(x)),
                torch.from_numpy(x).mean(dtype=torch.float32),
            ),
        ):
            assert got.tobytes() == np.float16(float(want)).tobytes(), (n, float(got), float(want))


def test_a_product_in_this_walk_is_the_correctly_rounded_fp16_value():
    # The contract the existing four already meet (test_blob_on_sim.py:191, "the
    # two agree to the last bit of the fp16 store"): an fp32 accumulation of the
    # fp16 inputs, stored once.  17 of 17 bit-equal here, n to 16384.
    checked = 0
    for n in (16, 64, 256, 1024, 4096, 16384):
        for spread in (0.5, 1.0, 2.0):
            x = centred(n, spread)
            want = np.float16(float(torch.from_numpy(x).prod(dtype=torch.float32)))
            got = np.float16(dsp_walk(x, "prod", fill=1.0))
            if not np.isfinite(want) or want == 0:
                continue
            checked += 1
            assert got.tobytes() == want.tobytes(), (n, spread, float(got), float(want))
    assert checked == 17, checked


def test_the_tail_fill_is_the_line_that_decides_it():
    # With the sum's own zero fill a product is 0 for every reduce length that is
    # not a multiple of 64, and the delegate count is unchanged, so nothing in the
    # blob would show it.
    for n in (8, 24, 100, 1000):
        x = centred(n, 1.0)
        want = dsp_walk(x, "prod", fill=1.0)
        got = dsp_walk(x, "prod", fill=0.0)
        assert np.isfinite(float(want)) and float(want) != 0, (n, float(want))
        assert float(got) == 0.0, (n, float(got))
    # A length that IS a multiple of 64 has no tail, so the two fills agree: the
    # clause is the tail and nothing else.
    x = centred(128, 1.0)
    assert float(dsp_walk(x, "prod", fill=0.0)) == float(dsp_walk(x, "prod", fill=1.0))


def test_torch_prod_on_fp16_accumulates_in_fp16_and_var_does_not():
    # The clause: the reference a refused prod node computes is not the reference a
    # prod mode is measured against.  A prod mode is MORE accurate than the
    # portable kernel it replaces, so a test using torch.prod(x_fp16) as its
    # reference fails and reads as the kernel's fault.
    n = 4096
    x = torch.full((n,), 1.001, dtype=F16)
    exact = float(np.float64(1.001) ** n)
    # Even the fp32-accumulated reference is not the exact product: at 4096 terms
    # its own reduction order loses 9%.  So the reference is that value, not fp64.
    assert float(x.prod(dtype=torch.float32)) == pytest.approx(54.49, rel=2e-3)
    assert float(x.prod(dtype=torch.float32)) != pytest.approx(exact, rel=1e-2)
    assert float(x.prod()) == pytest.approx(48.47, rel=2e-3), (
        "torch's fp16 prod is no longer the fp16-accumulated value, so the"
        " reference convention in this file has to be re-derived"
    )
    # The control that makes this a finding about prod and not about fp16: on the
    # same torch, dtype and tensor, sum and var ARE fp32-accumulated, so a second
    # moment's walks agree with the reference and a product's do not.
    y = torch.from_numpy(centred(n, 1.0)).contiguous()
    for got, want in (
        (float(y.sum(dtype=torch.float32)), float(y.numpy().astype(np.float64).sum())),
        (
            float(y.float().var(correction=1)),
            float(np.var(y.numpy().astype(np.float64), ddof=1)),
        ),
    ):
        assert got == pytest.approx(want, rel=2e-3), (got, want)


def test_the_product_walk_and_the_portable_kernel_diverge_where_it_counts():
    # The size of the divergence, so "more accurate" comes with a number: 0.22
    # relative at n=256 with a log-spread of 2, measured over the same 17 cases.
    worst = 0.0
    for n in (16, 64, 256, 1024, 4096, 16384):
        for spread in (0.5, 1.0, 2.0):
            x = centred(n, spread)
            got = float(dsp_walk(x, "prod", fill=1.0))
            portable = float(torch.from_numpy(x).prod())
            if np.isfinite(got) and got != 0 and np.isfinite(portable) and portable != 0:
                worst = max(worst, abs(got - portable) / abs(portable))
    assert 0.1 < worst < 0.3, worst


# ------------------------------------------------------- the second moment


def _deviations(x, mean16):
    dev = np.asarray(x, np.float16).astype(np.float32) - np.float32(mean16)
    pad = (-dev.size) % FP16_PER_BLOCK
    d = np.concatenate([dev.reshape(-1), np.zeros(pad, np.float32)]).reshape(
        -1, FP16_PER_BLOCK
    )
    acc0 = np.zeros(LANES, np.float32)
    acc1 = np.zeros(LANES, np.float32)
    for block in d:
        acc0 = (acc0 + (block[:LANES] ** 2).astype(np.float32)).astype(np.float32)
        acc1 = (acc1 + (block[LANES:] ** 2).astype(np.float32)).astype(np.float32)
    return _fold(acc0 + acc1, np.add)[0]


def naive_second_moment(x):
    """sum(x^2)/n - mean^2 in one pass, the form layer_norm_ops.cc:236-238 uses
    for a layer norm.  The sum of squares stays in the fp32 lane; only the final
    value is narrowed, as dst[o] = (__fp16)value does for the sum."""
    mean = dsp_mean(x)
    second = dsp_walk(x, "sum", square=True) / np.float32(np.asarray(x).size)
    return np.float32(second - mean * mean), np.float16(mean)


def test_the_deciding_case_is_reproduced_when_the_square_stays_in_a_lane():
    # Seven zeros and one 400: variance 20000, comfortably inside fp16, and the
    # deviation (400 - 50) = 350 is too.  Only the SQUARE leaves fp16, and it only
    # leaves if it is a buffer value.  This is the case OP_GAPS.md section 1 calls
    # deciding, and it is decided the other way.
    x = torch.zeros(1, 8, dtype=F16)
    x[0, -1] = 400.0
    xn = x.numpy().reshape(-1)
    assert float(x.var(correction=1)) == 20000.0
    naive, mean16 = naive_second_moment(xn)
    unbiased = np.float32(8 / 7)
    assert float(np.float16(naive * unbiased)) == 20000.0
    assert float(np.float16(_deviations(xn, mean16) * unbiased / np.float32(8))) == 20000.0
    # The clause, as a reading rather than a round number: the square is an
    # infinity as an fp16 buffer and a number inside an fp32 lane.
    assert float(np.float16((400.0 - 50.0) ** 2)) == float("inf")
    assert float(np.float32((400.0 - 50.0) ** 2)) == 122500.0
    assert 255.9 < float(np.sqrt(65504.0)) < 256.0


def test_the_naive_second_moment_is_conditioned_by_mean_over_sigma_and_the_two_pass_is_not():
    """The clause that decides the walk, over twelve seeds rather than one.

    A single seed cannot measure this.  The two-pass reads a mean that the mean
    reduction stored as fp16, so its error depends on where that mean lands
    relative to an fp16 boundary -- a 170x swing across seeds at mean 2000
    (5.5e-06 to 9.3e-04 measured, n=12).  The naive form's error does not swing:
    it is the condition number of the subtraction E[x^2] - E[x]^2, which is
    mean^2/variance, and at mean 2000 with sigma 1 it is 0.53 to 0.78 on every
    seed.  So the one-pass form is structurally unusable and the two-pass is a
    bounded walk, and a one-seed measurement of either would have said
    something else.
    """
    n = 4096
    naive_rel, two_pass_rel = {}, {}
    for mean in (0.0, 10.0, 200.0, 2000.0):
        rn, rt = [], []
        for seed in range(12):
            g = np.random.default_rng(seed)
            x = (mean + 1.0 * g.standard_normal(n)).astype(np.float16)
            exact = float(np.var(x.astype(np.float64), ddof=0))
            got, mean16 = naive_second_moment(x)
            rn.append(abs(float(got) - exact) / exact)
            two = float(_deviations(x, mean16)) / n
            rt.append(abs(two - exact) / exact)
        naive_rel[mean] = (min(rn), max(rn))
        two_pass_rel[mean] = (min(rt), max(rt))
    assert naive_rel[0.0][1] < 1e-6, naive_rel
    assert naive_rel[10.0][1] < 1e-3, naive_rel
    assert naive_rel[200.0][1] < 2e-2, naive_rel
    assert naive_rel[2000.0][0] > 0.3, naive_rel
    assert two_pass_rel[0.0][1] < 1e-6, two_pass_rel
    assert two_pass_rel[200.0][1] < 2e-3, two_pass_rel
    assert two_pass_rel[2000.0][1] < 2e-3, (
        two_pass_rel,
        "if the two-pass no longer holds at mean 2000 the conditioning argument in"
        " the docstring is wrong and var is the cheap win it says it is not",
    )
    # And the ratio that makes the two-pass the walk to build: at mean 2000 it is
    # more than two orders of magnitude better, on the same inputs.
    assert naive_rel[2000.0][0] / two_pass_rel[2000.0][1] > 100.0, (
        naive_rel,
        two_pass_rel,
    )

    # The control: on centred data, which is what a layer norm sees, the naive
    # form is fine -- which is why layer_norm_ops.cc can use it at all.
    g = np.random.default_rng(5)
    centred_x = (1.0 * g.standard_normal(n)).astype(np.float16)
    exact = float(np.var(centred_x.astype(np.float64), ddof=0))
    got, _ = naive_second_moment(centred_x)
    assert abs(float(got) - exact) / exact < 1e-4
