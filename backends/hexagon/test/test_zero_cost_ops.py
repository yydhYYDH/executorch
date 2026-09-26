# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The op families whose DSP cost was already paid, and what each one is.

`aten.clamp.Tensor` and `aten.elu.default` are compositions of command types this
backend already writes, so the cost of delegating them is zero new C++ and the
only question is what the composition computes. This file answers that by
lowering each op, reading the command stream out of the delegate's blob, and
running that blob: a delegate count would score `torch.cumsum` as covered when
the one command in it is a `view_copy`, and it would score these two the same
way whatever the commands were.

`aten.var.correction` is the third family the triage named, and the measurements
here are the reason it is not wired. The two-pass form is four commands the
table already has and reproduces the fp16 two-pass exactly, which is not the
same claim as reproducing `torch.var`: the two differ by 7.5e-04 relative on
centred fp16 data and 4.0e-03 on data with a mean of 200, and because the
square is materialised in fp16 a deviation past 255.9 becomes an infinity where
torch's fp32 accumulator still has a number. The test that pins the overflow is
the one that decides the family, and the positive control beside it is a mean
over the same span, so a partitioner that refused everything could not make this
file green.

Nothing here ran a kernel. `blob_interpreter.execute` is a numpy model of the
command stream, so a pass means the stream is shaped correctly and the modelled
arithmetic agrees; hexagon-sim compiles part of the DSP source and a device run
is the only thing that executes the vendored C++, and neither is this file.
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
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
    HexagonOperatorSupport,
)
from executorch.backends.hexagon.serialization import blob as _blob  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

F16 = torch.float16
CONFIG = EdgeCompileConfig(_check_ir_validity=False)
DELEGATE = torch.ops.higher_order.executorch_call_delegate
BINARY = hexagon_ops.DSP_OP_BINARY_ELEMENTWISE
UNARY = hexagon_ops.DSP_OP_UNARY
#: The op names the two families arrive as, read off the emitter table so that a
#: rename on the ATen side shows up here as a failure rather than as a silence.
CLAMP_TENSOR = hexagon_ops.CLAMP_TENSOR
ELU = hexagon_ops.ELU

#: The fp16 step at 8, which is where every elu measurement below tops out:
#: the largest representable gap below 8 is 2**-10.
ONE_STEP_AT_EIGHT = 2.0**-10


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


def _blobs(program):
    """Every delegate's blob, read off the program rather than counted."""
    out = []
    for node in program.graph_module.graph.nodes:
        if node.target is not DELEGATE:
            continue
        raw = program.graph_module.get_submodule(node.args[0].target)._processed_bytes
        # BLOB_MAGIC is an int, so a bytes comparison is the only one that
        # matches anything at all.
        assert isinstance(raw, (bytes, bytearray)) and bytes(raw[:4]) == struct.pack(
            "<I", _blob.BLOB_MAGIC
        ), "a delegate carries no blob; the reader would be reading nothing"
        out.append(bytes(raw))
    return out


def _commands(fn, args):
    """The command types in the delegates, in order."""
    types = []
    for raw in _blobs(_program(fn, args)):
        _header, found = read_blob(raw)
        types.extend(command.type for command in found)
    return types


def _run(fn, args, feed):
    """The delegates' outputs, through the model, as one flat fp16 array."""
    pieces = []
    for raw in _blobs(_program(fn, args)):
        pieces.append(np.frombuffer(execute(raw, feed)[0], dtype=np.float16))
    assert pieces, "nothing delegated, so there is nothing to run"
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def _bits(values):
    return np.frombuffer(np.ascontiguousarray(values).tobytes(), dtype=np.uint16)

def _verdict(support, program, index=0):
    """`is_node_supported` for the index'th call_function of a lowered program."""
    nodes = [n for n in program.graph.nodes if n.op == "call_function"]
    return support.is_node_supported(program, nodes[index])

def _x(seed=0, shape=(4, 16), scale=1.0):
    return (
        torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=F16)
        * scale
    )


def test_a_refused_everything_partitioner_could_not_pass_this_file():
    # The control: relu is the op every other test here is measured against.
    # Without it, a change that broke the partitioner outright would make every
    # assertion in this file about an empty command stream pass.
    assert _commands(torch.relu, (_x(),)) == [UNARY]


def test_clamp_tensor_is_one_binary_command_per_bound():
    # The stream is the claim, so it is read from the blob and spelled out. Two
    # bounds are min(max(x, lo), hi) and so two commands; one bound is one. The
    # scalar overload is the comparison, so that two is read as a difference
    # between the two spellings of one function rather than as a constant.
    x, low, high = _x(), (_x(1) - 0.5).contiguous(), (_x(2) + 0.5).contiguous()
    assert _commands(lambda a, b, c: torch.clamp(a, min=b, max=c), (x, low, high)) == [
        BINARY,
        BINARY,
    ]
    assert _commands(lambda a, b: torch.clamp(a, min=b), (x, low)) == [BINARY]
    assert _commands(lambda a, b: torch.clamp(a, max=b), (x, high)) == [BINARY]
    assert _commands(lambda a: torch.clamp(a, min=-1.0, max=1.0), (x,)) == [UNARY]

@pytest.mark.parametrize(
    "bound_shape",
    [(4, 16), (16,), (1,)],
    ids=["same-shape", "per-row", "one-element"],
)
def test_clamp_tensor_is_bit_exact(bound_shape):
    # A max and a min pick one of their operands and compute nothing, so every
    # element of the answer is a pair of bytes the graph already had. The per-row
    # bound is the shape a quantized model uses -- a scale or zero-point vector
    # against a (batch, features) activation -- and it is the case that walks the
    # stride table instead of the flat path, so it is in the list rather than left
    # out as the easy one.
    differing = 0
    total = 0
    for seed in range(4):
        x = _x(seed)
        bound = _x(seed + 100, bound_shape)
        low, high = (bound - 0.5).contiguous(), (bound + 0.5).contiguous()
        want = torch.clamp(x, min=low, max=high)
        got = _run(
            lambda a, b, c: torch.clamp(a, min=b, max=c),
            (x, low, high),
            [x.numpy(), low.numpy(), high.numpy()],
        )
        flat_got = got.reshape(-1)
        flat_want = want.numpy().reshape(-1)
        differing += int((_bits(flat_got) != _bits(flat_want)).sum())
        total += flat_want.size
    assert (differing, total) == (0, 256), (differing, total)

def test_clamp_tensor_keeps_a_nan_activation():
    # The bound is the first operand on purpose, and this is the reason. The
    # kernel's max is `a > b ? a : b` (eltwise_ops.cc:152), so an unordered pair
    # hands back the SECOND operand, and putting the activation second is what
    # makes a NaN survive as the NaN torch's clamp returns. The portable kernel
    # and a two-select form both need a comparison in front to agree with it.
    # Compared as bit patterns: NaN != NaN is a true statement about the
    # comparison operator and not about the answer.
    x = torch.tensor([[float("nan"), 1.0, -1.0, 0.0, 2.0]], dtype=F16)
    low = torch.tensor([[0.0, 0.0, -9.0, 0.0, 0.0]], dtype=F16)
    high = torch.tensor([[9.0, 9.0, 0.0, 0.0, 0.0]], dtype=F16)
    want = torch.clamp(x, min=low, max=high)
    got = _run(
        lambda a, b, c: torch.clamp(a, min=b, max=c),
        (x, low, high),
        [x.numpy(), low.numpy(), high.numpy()],
    )
    assert np.array_equal(_bits(got.reshape(-1)), _bits(want.numpy().reshape(-1))), (
        got,
        want,
    )


def test_clamp_tensor_drops_a_nan_bound_and_says_so():
    # The one input on which this is not torch's clamp, pinned rather than left
    # to be discovered. A NaN bound answers the activation unchanged where the
    # portable kernel answers NaN. The two-select form the triage measured drops
    # it too -- the comparison in front of the select is what drops it, and so
    # does torch's own where -- so the two routes agree with each other and
    # differ from torch.clamp in the same one place. This test is here so that
    # the limitation cannot change silently in either direction.
    x = torch.tensor([[1.0]], dtype=F16)
    nan_bound = torch.tensor([[float("nan")]], dtype=F16)
    high = torch.tensor([[9.0]], dtype=F16)
    assert np.isnan(torch.clamp(x, min=nan_bound, max=high).numpy()).all()
    got = _run(
        lambda a, b, c: torch.clamp(a, min=b, max=c),
        (x, nan_bound, high),
        [x.numpy(), nan_bound.numpy(), high.numpy()],
    )
    assert got.tolist() == [1.0], got

def test_clamp_tensor_refuses_a_bound_the_arena_cannot_hold():
    # An int64 bound is the reachable case, not a hypothetical one: a quantized
    # model clamps against a zero-point buffer, and torch's promotion leaves that
    # buffer an int64 node next to an fp16 activation. The arena holds fp16, so
    # the node is refused rather than narrowed on the way in -- narrowing a
    # zero-point that does not fit fp16 is a wrong answer rather than a slow one.
    # The control is the same graph with the bound in fp16, at the same geometry,
    # so a partitioner that refuses every clamp.Tensor cannot pass this.
    x = _x(0, (4, 16)).contiguous()
    assert _commands(
        lambda a, b: torch.clamp(a, min=b),
        (x, torch.zeros(4, 16, dtype=torch.int64)),
    ) == []
    assert _commands(
        lambda a, b: torch.clamp(a, min=b),
        (x, torch.zeros(4, 16, dtype=F16)),
    ) == [BINARY]
    # fp32 bounds are kept: the partitioner narrows them on the way into the
    # arena the same way it narrows an fp32 activation, so the gate is the dtype
    # and not the width.
    assert _commands(
        lambda a, b: torch.clamp(a, min=b),
        (x, torch.zeros(4, 16, dtype=torch.float32)),
    ) == [BINARY]


def test_clamp_tensor_predicate_is_asked_not_just_the_table():
    # A literal bound belongs to the scalar overload, which the unary clamp
    # already answers in one command, and a clamp with no bound at all is the
    # identity. Both stay where they were, and the (1, 16) self-bound beside
    # them is the control that has to delegate.
    x = _x()
    assert _commands(lambda a: torch.clamp(a, min=0.0), (x,)) == [UNARY]
    assert _commands(lambda a: torch.clamp(a, min=-1.0, max=1.0), (x,)) == [UNARY]
    assert _commands(lambda a: torch.clamp(a, min=x), (x,)) == [BINARY]

def _stream(fn, args):
    """(command type, the param that names its variant) for every command.

    The count alone is too weak: three unary and three binary commands is what
    any composition of a relu, a min, an exp and two arithmetic steps looks like,
    and it is also what a stream that computed the wrong thing entirely looks
    like. params[1] names a unary variant and params[3] a binary one, so the
    order and the variant of all six are readable off the blob.
    """
    variant_param = {UNARY: 1, BINARY: 3}
    found_stream = []
    for raw in _blobs(_program(fn, args)):
        _header, commands = read_blob(raw)
        for command in commands:
            params = list(command.params)
            index = variant_param.get(command.type)
            found_stream.append(
                (command.type, params[index] if index is not None else None)
            )
    return found_stream


def test_elu_is_six_commands_and_seven_with_a_scale():
    # max(x, 0) * scale + min((exp(x) - 1) * alpha * scale, 0). The subtypes are
    # HtpOpsUnaryOpType and HtpOpsBinaryOpType: 5 is exp, 15 is the clamp entry
    # point, 17 is mul_scalar, and the binaries are 2 sub, 3 mul, 1 add. nn.SELU
    # arrives as an elu carrying torch's scale of 1.0507, so it is the same
    # composition with one more command.
    x = _x(0, (1, 64, 1, 48), 2.0).contiguous()
    assert _stream(lambda a: torch.nn.functional.elu(a), (x,)) == [
        (UNARY, 5),
        (BINARY, 2),
        (BINARY, 3),
        (UNARY, 15),
        (UNARY, 15),
        (BINARY, 1),
    ]
    assert _stream(lambda a: torch.nn.SELU()(a), (x,)) == [
        (UNARY, 5),
        (BINARY, 2),
        (BINARY, 3),
        (UNARY, 15),
        (UNARY, 15),
        (UNARY, 17),
        (BINARY, 1),
    ]
    # A negative coefficient is the same seven commands' worth of work, and the
    # graph that produces it stays put: a control for the gate below, which
    # cannot be satisfied by a partitioner that refuses every elu.
    assert _stream(
        lambda a: torch.nn.functional.elu(a, alpha=1.0), (x,)
    ) == _stream(lambda a: torch.nn.functional.elu(a), (x,))

@pytest.mark.parametrize(
    "alpha, scale",
    [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.6732632423543772, 1.0507009873554805)],
    ids=["default", "alpha-0.5", "alpha-2", "selu"],
)
def test_elu_is_within_two_fp16_steps(alpha, scale):
    # The composition rounds once per stage, where torch's kernel holds fp32 and
    # narrows once. Measured worst over four seeds of randn * 2 and over a grid
    # from -30 to 8: 4.883e-04 at alpha 1, 2.441e-04 at alpha 0.5 and 9.766e-04
    # at alpha 2 and for nn.SELU, so two steps is the bound with room, and one
    # step is 4.883e-04 at the top of the range. The bound is twice the measured
    # worst rather than the measured worst, because a test that asserts the
    # number it measured is a test that cannot see the next seed.
    worst = 0.0
    for seed in range(4):
        x = _x(seed, (1, 64, 1, 48), 2.0).contiguous()
        want = torch.ops.aten.elu.default(x, alpha, scale, 1.0)
        got = _run(
            lambda a: torch.ops.aten.elu.default(a, alpha, scale, 1.0), (x,), [x.numpy()]
        )
        worst = max(worst, float(np.abs(got.astype(np.float32) - want.numpy().reshape(-1).astype(np.float32)).max()))
    grid = torch.arange(-30, 8, 0.01, dtype=torch.float32).to(F16).reshape(1, -1).contiguous()
    want = torch.ops.aten.elu.default(grid, alpha, scale, 1.0)
    got = _run(
        lambda a: torch.ops.aten.elu.default(a, alpha, scale, 1.0), (grid,), [grid.numpy()]
    )
    worst = max(worst, float(np.abs(got.astype(np.float32) - want.numpy().reshape(-1).astype(np.float32)).max()))
    assert worst <= 2.0 * ONE_STEP_AT_EIGHT, worst
    assert worst > 0.0, "the composition cannot be bit-exact; the bound is hiding that"


def test_elu_keeps_a_nan_activation():
    # Both halves of the split are the clamp entry point, which restores a NaN
    # input by a magnitude test (unary_ops.cc:498-541), and the add of two NaNs
    # is a NaN. The NaN is compared as a bit pattern, because NaN != NaN is a true
    # statement about the comparison operator and not about the answer. The
    # finite elements are held to the same two-step bound as the numerics test
    # above and no tighter: this is a test about NaN, and -1.0 comes back one fp16
    # step away because exp(-1) - 1 in fp16 is not expm1(-1), which is the
    # tolerance the composition is allowed and not a defect of the NaN path.
    x = torch.tensor([[float("nan"), 1.0, -1.0, 0.0, 2.0]], dtype=F16)
    want = torch.nn.functional.elu(x).numpy().reshape(-1)
    got = _run(lambda a: torch.nn.functional.elu(a), (x,), [x.numpy()])
    assert np.array_equal(_bits(got[:1]), _bits(want[:1])), (got, want)
    assert np.isnan(got[0]) and np.isnan(want[0])
    finite = ~np.isnan(want)
    assert np.abs(got[finite].astype(np.float32) - want[finite].astype(np.float32)).max() <= (
        2.0 * ONE_STEP_AT_EIGHT
    ), (got, want)


def test_the_elu_bound_is_absolute_and_its_relative_error_is_not_small():
    # The bound above is an absolute one, and an absolute bound is blind to the
    # one place the composition is worst in relative terms. exp(x) - 1 in fp16
    # cancels near zero exactly the way `aten.expm1` does -- which is why §3 of
    # OP_GAPS keeps expm1 unwired -- and a relative error of 1.0 is what a
    # cancellation buys. Here it is contained: the answer stays within two fp16
    # steps, and the negative branch saturates at -alpha * scale, so the error is
    # 0.2% of the branch's range for F.elu and 0.1% for nn.SELU. Both halves are
    # asserted because either one alone is a claim a reader could not check.
    alpha, scale = 1.0, 1.0
    xs = torch.tensor([[-0.01, -0.1, -1.0, -3.3, -30.0]], dtype=F16)
    want = torch.ops.aten.elu.default(xs, alpha, scale, 1.0).float()
    composed = torch.clamp(xs, 0, None) * scale + torch.clamp(
        (torch.exp(xs) - 1.0) * alpha * scale, None, 0
    )
    absolute = (composed.float() - want).abs()
    assert absolute.max().item() <= 2.0 * ONE_STEP_AT_EIGHT, absolute
    assert (2.0**-9) / (alpha * scale) < 2.0e-3, "the bound is no longer small"
    # The cancellation, read off rather than asserted in prose: at x = -0.01 the
    # answer is about -9.95e-3 and the composition is 2.4e-4 away from it, which
    # is a few per cent relative. A model that consumed this band as a ratio
    # would see it, and that is the reason this test is here at all.
    relative = absolute / want.abs()
    assert 0.01 < relative[0, 0].item() < 0.1, relative[0, 0].item()
    # And below about 3e-4 the answer is smaller than one fp16 step of exp(0), so
    # the relative error is 1.0 -- the same statement OP_GAPS makes for expm1,
    # with the difference that here the absolute error is bounded by a branch
    # that saturates one order of magnitude above it.
    tiny = torch.tensor([[-3.0e-4]], dtype=F16)
    tiny_want = torch.ops.aten.elu.default(tiny, alpha, scale, 1.0).float()
    tiny_got = torch.clamp(tiny, 0, None) + torch.clamp((torch.exp(tiny) - 1.0) * alpha, None, 0)
    assert (tiny_got.float() - tiny_want).abs().item() <= 2.0 * ONE_STEP_AT_EIGHT
    assert abs(tiny_want.item()) < 2.0**-9, "the band is empty now"


@pytest.mark.parametrize(
    "alpha, scale, input_scale",
    [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 2.0)],
    ids=["negative-alpha", "negative-scale", "input-scale-2"],
)
def test_elu_refuses_a_coefficient_the_split_cannot_represent(
    alpha, scale, input_scale
):
    # The relu-plus-min rewrite is the identity only when the negative term is
    # non-positive wherever the positive one is not, so a negative alpha or a
    # negative scale makes it a different function; and the DSP has no expm1 to
    # fold an argument scale into, so an input_scale has nowhere to go. The
    # control is the same op at alpha 1, which must reach the delegate.
    x = _x(0, (1, 64, 1, 48), 2.0).contiguous()
    assert _commands(
        lambda a: torch.ops.aten.elu.default(a, alpha, scale, input_scale), (x,)
    ) == []
    assert len(_stream(lambda a: torch.nn.functional.elu(a), (x,))) == 6


def test_the_two_predicates_are_the_ones_doing_the_refusing():
    # The emitter table alone would delegate all of these, so the refusals above
    # are the predicates' doing. Read through the operator support rather than by
    # counting delegates -- a node the partitioner never offered is a refusal it
    # would report for the wrong reason -- and with the accepted case at the same
    # geometry beside each refusal, so that a support object which refuses
    # everything cannot pass.
    support = HexagonOperatorSupport()
    x = _x(0, (1, 8), 2.0).contiguous()
    elu = to_edge_transform_and_lower(
        export(_Callable(torch.nn.functional.elu), (x,))
    ).exported_program()
    assert _verdict(support, elu), "the elu the other tests delegate is not supported"

    for bad in (
        lambda a: torch.ops.aten.elu.default(a, -1.0, 1.0, 1.0),
        lambda a: torch.ops.aten.elu.default(a, 1.0, 1.0, 2.0),
    ):
        lowered = to_edge_transform_and_lower(export(_Callable(bad), (x,))).exported_program()
        assert not _verdict(support, lowered), bad

    for bound in (torch.zeros(4, 16, dtype=torch.int64),):
        lowered = to_edge_transform_and_lower(
            export(_Callable(lambda a, b: torch.clamp(a, min=b)), (_x(0, (4, 16)), bound))
        ).exported_program()
        assert not _verdict(support, lowered), bound.dtype
    accepted = to_edge_transform_and_lower(
        export(_Callable(lambda a, b: torch.clamp(a, min=b)), (_x(0, (4, 16)), _x(1, (4, 16))))
    ).exported_program()
    assert _verdict(support, accepted), (
        "the same clamp with an fp16 bound is refused, so this support object "
        "refuses everything and the refusals above mean nothing"
    )


def test_the_predicates_are_the_functions_the_table_names():
    # HexagonOperatorSupport takes its answers from these two predicates and the
    # emitter table, so a test that asserts a refusal should be able to name the
    # thing that made it. Read the predicates off the operators module rather
    # than off a private import, which is the same object the partitioner calls.
    assert CLAMP_TENSOR in hexagon_ops.EMITTERS
    assert ELU in hexagon_ops.EMITTERS
    assert hexagon_ops.CLAMP_TENSOR_TARGETS == frozenset({CLAMP_TENSOR})
    assert hexagon_ops.ELU_TARGETS == frozenset({ELU})
    # And the composition needs nothing from the vendored C++ that is not
    # already there: every command it writes is one of these, which is what
    # "zero new C++" has to mean for it to be a claim rather than a figure of
    # speech.
    assert (UNARY, BINARY) == (4, 19)


def _two_pass(x, correction=1):
    """The variance as a mean of squared deviations, in fp16, as a graph would."""
    n = x.shape[-1]
    return ((x - x.mean(-1, keepdim=True)) ** 2).mean(-1) * (n / (n - correction))


def test_var_stays_on_the_portable_kernel():
    # The measurement that decides the family. The two-pass form is four
    # commands of types this table already has -- REDUCTION, BINARY, UNARY,
    # REDUCTION -- and reproduces the fp16 two-pass bit for bit, which is a
    # statement about the composition and not about torch.var. This is the
    # composition's own command stream, read off a blob.
    x = _x(0, (4, 16)).contiguous()
    assert _commands(_two_pass, (x,)) == [
        hexagon_ops.DSP_OP_REDUCTION,
        BINARY,
        UNARY,
        hexagon_ops.DSP_OP_REDUCTION,
        BINARY,
    ]
    # And the op itself, which is what the row in OP_SUPPORT.md says: no
    # commands, because no composition of them is the op. The mean over the
    # same span is the control -- it is the first command of the two-pass, and
    # if it did not delegate there would be nothing here to compose.
    assert _commands(lambda a: torch.var(a, dim=-1), (x,)) == []
    assert _commands(lambda a: a.mean(-1, keepdim=True), (x,)) == [
        hexagon_ops.DSP_OP_REDUCTION
    ]


def test_the_fp16_two_pass_is_not_torch_var():
    # Two figures, both over four seeds. The first is the rounding: the
    # composition holds fp16 across the square and the mean while torch's
    # accumulator is fp32, so the two drift apart by steps of the answer's own
    # scale. The second is the offset: a mean of 200 makes the deviations large
    # and the relative gap four times the centred one.
    for label, make, bound in (
        ("centred", lambda g: torch.randn(4, 16, generator=g, dtype=F16), 1.5e-3),
        ("mean 200", lambda g: torch.randn(8, 256, generator=g, dtype=F16) + 200, 8.0e-3),
    ):
        worst = 0.0
        for seed in range(4):
            x = make(torch.Generator().manual_seed(seed))
            want = torch.var(x, dim=-1, correction=1).float()
            got = _two_pass(x).float()
            worst = max(worst, float(((got - want).abs() / want.abs().clamp_min(1e-6)).max()))
        assert 0.0 < worst < bound, (label, worst)


def test_the_fp16_two_pass_overflows_where_torch_var_does_not():
    # The deciding measurement. Seven zeros and one 400: torch's variance is
    # 20000, comfortably inside fp16, and the two-pass has to materialise
    # (400 - 50) ** 2 = 122500 as an fp16 intermediate, which is an infinity.
    # No command arrangement avoids it -- the square is a buffer either way, and
    # the DSP's reduction accumulates in fp32 but reads fp16 -- and the export
    # cannot bound the deviation of a run-time activation, so the family stays
    # closed. This is the test that would fail first if that ever changed.
    x = torch.zeros(1, 8, dtype=F16)
    x[0, -1] = 400.0
    assert torch.var(x, dim=-1, correction=1).item() == 20000.0
    assert _two_pass(x).item() == float("inf")
    # The boundary itself, so "past 255.9" is a reading and not a round number:
    # the largest fp16 is 65504 and its square root is where the square stops.
    assert float(torch.finfo(torch.float16).max) == 65504.0
    assert 255.9 < float(np.sqrt(65504.0)) < 256.0

def test_the_model_transcribes_the_kernels_max_and_min():
    # The instrument, and the reason the NaN tests above are worth reading. The
    # model used to answer binary max and min with numpy's, which propagate a
    # NaN; htp_ops_binary_apply_fp16 is `a > b ? a : b` and `a < b ? a : b`
    # (eltwise_ops.cc:152-155), which return the second operand on an unordered
    # pair. Had that stayed, a clamp emitter with its operands in the wrong
    # order would have measured as correct. Asserted here against the DSP's rule
    # rather than against the model, so the pin survives a change to either.
    from blob_interpreter import _BINARY

    nan = np.float16(np.nan)
    one = np.float16(1.0)
    assert np.isnan(nan)
    assert _BINARY[hexagon_ops.BINARY_OP_TYPES["max"]](np.array([nan]), np.array([one]))[0] == 1.0, (
        "a NaN in the first operand is dropped, which is what the kernel does"
    )
    assert np.isnan(_BINARY[hexagon_ops.BINARY_OP_TYPES["max"]](np.array([one]), np.array([nan]))[0]), (
        "a NaN in the second operand survives, which is the order the clamp uses"
    )
    assert np.isnan(_BINARY[hexagon_ops.BINARY_OP_TYPES["min"]](np.array([one]), np.array([nan]))[0])
    # And the control: numpy would have answered both of those differently, so
    # this is a difference and not a restatement.
    assert np.isnan(np.maximum(np.array([nan]), np.array([one]))[0])


def test_the_model_runs_a_blob_carrying_the_unary_scale():
    # The model's unary scale entry was absent, which left every blob carrying
    # `aten.mul.Scalar` unrunnable here -- and the elu composition's SELU form
    # needs it. htp_ops_scale_fp32 widens, multiplies in fp32 and narrows once
    # (unary_ops.cc:711-742). The control is the same blob with a different
    # scale: an entry that ignored params[3] would answer both alike.
    x = (_x(0, (4, 16), 2.0)).contiguous()
    got = _run(lambda a: a * 1.5, (x,), [x.numpy()])
    want = (x.float() * 1.5).to(F16).numpy().reshape(-1)
    assert np.array_equal(_bits(got), _bits(want)), (got, want)
    other = _run(lambda a: a * 2.5, (x,), [x.numpy()])
    assert not np.array_equal(_bits(got), _bits(other)), (
        "the scale in params[3] is not being read"
    )
