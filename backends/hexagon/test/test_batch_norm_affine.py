# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A batch norm with no convolution in front of it, as two broadcast commands.

The claim under test is that this needs no new kernel: a `[C, 1, 1]` constant
against an `[N, C, ...]` value is the stride-0 broadcast the binary elementwise
descriptor already carries, so the node is a `MUL` and an `ADD` against two
constants. Every case here therefore pins three things at once: that the graph
rewrite produces those two commands and no others, that the broadcast strides in
the tail are the ones that place a per-channel value, and that the numbers the
blob produces are the ones torch produces.

The strides are the interesting half, because a per-channel constant read at the
wrong stride is a *plausible* wrong answer: it is the right shape, made of the
right handful of numbers, differing by a few ulp rather than by a factor. The
cases that check the placement use powers of two and a constant input, so any
misplaced element differs by eight times and fp16 arithmetic is exact; a tolerance
here would accept the wrong answer, which is the one thing it must not.
"""

import operator
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.batch_norm import (  # noqa: E402
    BATCH_NORM_TARGETS,
    RewriteBatchNormToAffine,
    batch_norm_is_rewritable,
    trains_on_batch_statistics,
)
from executorch.backends.hexagon.fold_batch_norm import (  # noqa: E402
    FoldBatchNormIntoConv,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

#: Where an op's params start inside a command in the blob: the header, the op
#: record, then the fixed-length input and output ref arrays.
_PARAMS_AT = 16

_BINARY = hexagon_ops.DSP_OP_BINARY_ELEMENTWISE
_MUL = hexagon_ops.BINARY_OP_TYPES["mul"]
_ADD = hexagon_ops.BINARY_OP_TYPES["add"]
_CONV = hexagon_ops.DSP_OP_CONV_DEPTHWISE2D_FP16
_CONV = hexagon_ops.DSP_OP_CONV_DEPTHWISE2D_FP16


class _OnlyNorm(torch.nn.Module):
    """A graph whose only batch norm is the node under test."""

    def __init__(self, channels, affine=True, eps=1e-5, value_dim=4, training=False):
        super().__init__()
        batch_norm = (
            torch.nn.BatchNorm1d if value_dim == 3 else torch.nn.BatchNorm2d
        )
        self.norm = batch_norm(channels, eps=eps, affine=affine)
        if training:
            self.norm.train()

    def forward(self, x):
        return self.norm(x)


def _populated(channels, eps=1e-5, affine=True, value_dim=4, seed=0):
    """A batch norm whose statistics are not the identity and not all positive.

    A fresh `BatchNorm2d` scales by `1/sqrt(1+eps)` and leaves the mean at zero,
    so folding it and not folding it differ by one part in a hundred thousand --
    a difference a tolerance that is merely tight still calls equal. The mean is
    given a spread here so an affine that drops it, and one scale is made negative
    and one small, because a monotone assumption is a common and wrong one.
    """
    torch.manual_seed(seed)
    model = _OnlyNorm(channels, affine=affine, eps=eps, value_dim=value_dim)
    with torch.no_grad():
        if affine:
            model.norm.weight.uniform_(0.5, 2.0)
            model.norm.bias.uniform_(-1.0, 1.0)
        model.norm.running_mean.uniform_(-3.0, 3.0)
        model.norm.running_var.uniform_(0.01, 4.0)
    if affine:
        with torch.no_grad():
            model.norm.weight[0] = -1.5
            if channels > 1:
                model.norm.weight[1] = 1e-3
    return model.eval()


def _edge(model, x):
    return to_edge(export(model, (x,))).exported_program()


def _bn_nodes(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.target in BATCH_NORM_TARGETS
    ]


def _blob(program):
    from executorch.backends.hexagon.hexagon_backend import HexagonBackend

    return HexagonBackend.preprocess(program, []).processed_bytes


def _run(model, x):
    """The numbers the DSP would produce, through the emitted blob."""
    program = _edge(model, x.half())
    RewriteBatchNormToAffine()(program)
    assert not _bn_nodes(program), "the batch norm is still in the graph"
    blob = _blob(program)
    _, commands = read_blob(blob)
    got = np.frombuffer(execute(blob, [x.half().numpy()])[0], dtype=np.float16)
    return blob, commands, got.reshape(tuple(x.shape))


def _step(values: np.ndarray) -> np.ndarray:
    """One fp16 step at each value's own magnitude, as a float32 array."""
    half = values.astype(np.float16)
    return np.abs(
        np.nextafter(half, np.float16(np.inf)).astype(np.float32)
        - half.astype(np.float32)
    )


def _coefficients(model):
    """The per-channel `s` and `b` of this batch norm, in fp64.

    Read from the module rather than from the graph, so a test can state the
    budget the two commands are entitled to without trusting the code under test
    to tell it what it computed.
    """
    norm = model.norm
    mean = norm.running_mean.double()
    var = norm.running_var.double()
    scale = torch.rsqrt(var + norm.eps)
    if norm.affine:
        scale = scale * norm.weight.double()
        shift = norm.bias.double() - mean * scale
    else:
        shift = -mean * scale
    return scale, shift


def _budget(x: torch.Tensor, scale, shift) -> np.ndarray:
    """The fp16 rounding the two commands are entitled to, per element.

    `x * s` rounds once at the magnitude of the product and `+ b` rounds once at
    the magnitude of the shift, so the sum of the two is what the answer may move
    by. Measuring the move against the step at the *result*'s own magnitude
    instead is the wrong bound: `x*s + b` cancels, and an element whose answer is
    near zero out of terms of size ten is exact in absolute terms and many steps
    out in relative ones. That is a property of the arithmetic, not a defect, and
    a tolerance shaped the other way would call every such element a failure.
    """
    with torch.no_grad():
        product = (x.double() * _channel_view(scale, x)).abs().numpy()
        shift_magnitude = _channel_view(shift.abs(), x).numpy()
    return _step(product) + _step(np.broadcast_to(shift_magnitude, product.shape))


def _channel_view(per_channel, x: torch.Tensor) -> torch.Tensor:
    """`per_channel` shaped so it multiplies axis 1 of `x` and nothing else.

    The arena is row-major, so the channel is axis one of a `[N, C, ...]` value
    and a `reshape(-1, C)` would line it up with the *innermost* axis instead --
    which reads the same weights onto the wrong elements and still produces a
    tensor of exactly the right shape.
    """
    return per_channel.reshape((1, -1) + (1,) * (x.dim() - 2))


def _within_budget(got, x, scale, shift) -> float:
    """The measured disagreement in units of that budget, largest over the tensor.

    One is the exact bound; the tests assert two, so a single extra rounding on
    either term is still inside and a wrong element is three orders out.
    """
    budget = _budget(x, scale, shift)
    error = np.abs(got.astype(np.float32) - expected_fp64(x, scale, shift))
    return float(np.max(error / np.where(budget > 0, budget, np.finfo(np.float16).tiny)))


def expected_fp64(x: torch.Tensor, scale, shift) -> np.ndarray:
    """What the affine is, computed in the width it is defined in.

    The comparison the tests make is against this and not against
    `model(x.half())`: torch's own expression rounds the normalized value, the
    scale and the bias into one fp16 answer, and a difference from that is a
    difference of expression rather than of kernel.
    """
    with torch.no_grad():
        return (x.double() * _channel_view(scale, x) + _channel_view(shift, x)).numpy()


def _reference(model, x):
    """torch's own answer, in the width the arena holds."""
    with torch.no_grad():
        return model(x.half()).numpy()


def test_the_broadcast_places_a_per_channel_value_on_the_channel_axis():
    """A `[C, 1, 1]` constant is a stride-0 spatial repeat, not a dense read.

    The input is all ones and the scale is 1, 8 and 64, so the answer is the
    constant itself and every element is a power of two. Reading it at the wrong
    stride still yields a number of the right shape out of the same three
    values, so this compares bit patterns rather than magnitudes: a difference
    here is eight times, not a rounding step.
    """
    channels, height, width = 3, 5, 7
    x = torch.ones(1, channels, height, width, dtype=torch.float16)
    scale = torch.tensor([1.0, 8.0, 64.0], dtype=torch.float16).reshape(
        channels, 1, 1
    )

    class _Scaled(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.register_buffer("s", scale)

        def forward(self, x):
            return x * self.s

    _, commands, got = _run(_Scaled(scale), x)
    assert [command.type for command in commands] == [_BINARY]
    # The broadcast tail: the rank, the output's extents, then one stride list
    # per operand. The activation is row-major, so its own strides are the
    # products of the extents to its right; the constant is [3, 1, 1] against a
    # [1, 3, 5, 7] output, so its strides are zero on every axis but the
    # channel, where one step is one element of the constant.
    assert list(commands[0].params[17:21]) == [0, 35, 7, 1]
    assert list(commands[0].params[25:29]) == [0, 1, 0, 0]
    assert list(got[0, :, 0, 0]) == [1.0, 8.0, 64.0]
    assert np.array_equal(got, (x * scale).numpy()), "the broadcast read elsewhere"


def test_a_wrong_broadcast_stride_is_caught():
    """The control the first case rests on: zeroing a stride changes the answer.

    Without this, the first case could be green because the kernel ignores the
    tail and reads the constant densely, which would make every other stride
    assertion in this file vacuous.
    """
    import struct

    from executorch.backends.hexagon.serialization import blob as B

    channels, height, width = 3, 5, 7
    x = torch.ones(1, channels, height, width, dtype=torch.float16)
    scale = torch.tensor([1.0, 8.0, 64.0], dtype=torch.float16).reshape(
        channels, 1, 1
    )

    class _Scaled(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.register_buffer("s", scale)

        def forward(self, x):
            return x * self.s

    blob, _, good = _run(_Scaled(scale), x)
    data = bytearray(blob)
    _, commands = read_blob(blob)
    # params[26] is the constant's channel-axis stride. Zeroing it makes every
    # element read the same value -- the right shape, the wrong product.
    struct.pack_into("<i", data, B.HEADER_SIZE + _PARAMS_AT + 4 * 26, 0)
    bad = np.frombuffer(
        execute(bytes(data), [x.numpy()])[0], dtype=np.float16
    ).reshape(good.shape)
    assert not np.array_equal(bad, good), "the stride is not what places the value"

@pytest.mark.parametrize("channels", [1, 2, 3, 5, 16])
@pytest.mark.parametrize("height,width", [(4, 4), (3, 5), (1, 8)])
def test_the_rewrite_is_torch_s_values(channels, height, width):
    """The two commands are the batch norm, on statistics that are not the identity.

    The tolerance is two fp16 steps, which is what a multiply and an add in the
    kernel's own width can differ by from torch's single fused expression; the
    case above is the one that says the placement is right, and this one says
    the arithmetic is.
    """
    x = torch.randn(1, channels, height, width, dtype=torch.float16) * 3.0
    model = _populated(channels, seed=channels * 7 + width)
    _, _, got = _run(model, x)
    scale, shift = _coefficients(model)
    assert _within_budget(got, x, scale, shift) <= 2.0, (
        f"C={channels} {height}x{width} is "
        f"{_within_budget(got, x, scale, shift):.2f} budgets out"
    )


def test_a_three_dimensional_value_takes_a_two_dimensional_constant():
    """The constant is right-aligned, so its rank follows the value's.

    A `[C, 1, 1]` against an `[N, C, L]` value would put the channel on the wrong
    axis and widen the result from `[N, C, L]` to `[N, C, L, C]` -- a shape error
    rather than a wrong number, which is the good kind of failure to have.
    """
    channels, length = 7, 9
    x = torch.randn(1, channels, length, dtype=torch.float16) * 3.0
    model = _populated(channels, value_dim=3, seed=3)
    _, commands, got = _run(model, x)
    assert got.shape == (1, channels, length)
    assert commands[0].params[8] == 3, "the output is three-dimensional"
    assert list(commands[0].params[25:28]) == [0, 1, 0], commands[0].params[25:33]
    scale, shift = _coefficients(model)
    assert _within_budget(got, x, scale, shift) <= 2.0


@pytest.mark.parametrize("extent", [62, 63, 64, 65, 66, 127, 128, 129])
def test_the_sixty_four_boundary_is_walked_on_every_side(extent):
    """The innermost extent is the vector run, and 64 is where HVX splits.

    `htp_ops_binary_try_broadcast` takes a flat path when both operands are
    contiguous, a row path when the innermost extent is a multiple of eight, and
    otherwise the per-element offset loop, which is a different program with the
    same descriptor. 63 has to be all scalar, 64 all vector and 65 is 64 plus
    one, so the three are swept rather than reasoned about.
    """
    channels = 3
    x = torch.randn(1, channels, extent, dtype=torch.float16) * 2.0
    model = _populated(channels, value_dim=3, seed=extent)
    _, _, got = _run(model, x)
    scale, shift = _coefficients(model)
    assert _within_budget(got, x, scale, shift) <= 2.0, (
        f"extent {extent} is {_within_budget(got, x, scale, shift):.2f} budgets out"
    )


@pytest.mark.parametrize("channels", [63, 64, 65])
def test_a_channel_count_at_the_boundary_is_the_same_case(channels):
    """Sixty-four channels is a structural boundary on this DSP, so it is swept.

    The binary broadcast path has no channel rule of its own, which is the claim:
    if there were one it would refuse or diverge at one of these three.
    """
    x = torch.randn(1, channels, 4, 4, dtype=torch.float16) * 2.0
    model = _populated(channels, seed=channels)
    _, _, got = _run(model, x)
    scale, shift = _coefficients(model)
    assert _within_budget(got, x, scale, shift) <= 2.0


def test_one_channel_is_the_per_element_case_and_does_not_divide_by_zero():
    """One channel is where the per-channel view and the per-element view meet.

    The statistics are normalized over the batch, the height and the width, so
    with a single channel there is no axis the map is constant along: every
    element gets its own scale and shift, and both are one number. It is the
    adversarial case for an implementation that divides by a count rather than by
    the variance, and the variance here is a millionth, so a scale built without
    eps would be a thousand times the one that is right.
    """
    x = torch.randn(1, 1, 5, 7, dtype=torch.float16) * 4.0
    model = _populated(1, seed=11)
    with torch.no_grad():
        model.norm.running_var.fill_(1e-6)
    _, commands, got = _run(model, x)
    assert len(commands) == 2
    assert np.isfinite(got).all(), f"a non-finite value came back: {got}"
    # Both constants are single elements, so each is a scalar broadcast: the
    # stride the descriptor carries is zero on every axis, the channel included.
    for command in commands:
        assert command.params[2] == 1, f"the constant has {command.params[2]} elements"
        assert list(command.params[25:29]) == [0, 0, 0, 0], command.params[25:33]
    # One channel means one affine for the whole tensor, not one per element: the
    # best straight line through (input, output) has no residual left over. A
    # per-element reading of the same node would leave the spread of the input
    # behind as scatter, which at these magnitudes is three orders over the
    # rounding of a line.
    xs = x.numpy().reshape(-1).astype(np.float64)
    ys = got.reshape(-1).astype(np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    residual = np.abs(ys - (slope * xs + intercept))
    assert residual.max() <= 4.0 * _step(np.abs(ys)).max(), (
        f"one channel left a residual of {residual.max()}"
    )
    scale, shift = _coefficients(model)
    assert _within_budget(got, x, scale, shift) <= 2.0


def test_the_scale_is_not_assumed_positive_or_large():
    """A negative scale and a near-zero one, on values at, near and far from mu.

    `_populated` makes channel 0's scale negative and channel 1's a thousandth
    of the variance's root, and the input is placed at the mean, three sigma from
    it and below zero. A monotone reading of the map, or one that clamps, fails
    on the first two.
    """
    channels = 4
    model = _populated(channels, seed=5)
    mean = model.norm.running_mean.float()
    with torch.no_grad():
        model.norm.weight[1] = 1e-3
        model.norm.weight[2] = -2.0
    sigma = (model.norm.running_var.float() + 1e-5).sqrt()
    rows = torch.stack(
        [mean, mean + 3.0 * sigma, mean - 3.0 * sigma, -mean.abs() - 4.0 * sigma]
    )
    x = rows.to(torch.float16).reshape(4, channels, 1, 1)
    _, _, got = _run(model, x)
    scale, shift = _coefficients(model)
    assert _within_budget(got, x, scale, shift) <= 2.0, (
        f"a signed scale is {_within_budget(got, x, scale, shift):.2f} budgets out"
    )
    assert (got[1] * got[2] < 0).any(), "a negative scale came out positive"


def test_epsilon_is_carried_and_a_dropped_one_is_caught():
    """`eps` is the difference between `sqrt(var)` and `sqrt(var + eps)`.

    The two graphs below are the same network with two epsilons, and the rewrite
    is what differs. If the affine were built without reading eps the two would
    agree, which is the failure this pins: an epsilon that is silently dropped
    is a wrong number and no error anywhere.
    """
    x = torch.randn(1, 3, 4, 4, dtype=torch.float16) * 2.0
    model_zero = _populated(3, eps=0.0, seed=2)
    with torch.no_grad():
        model_zero.norm.running_var.fill_(1e-6)
    _, _, without = _run(model_zero, x)
    model_one = _populated(3, eps=1.0, seed=2)
    with torch.no_grad():
        model_one.norm.running_var.fill_(1e-6)
    _, _, with_one = _run(model_one, x)
    assert not np.allclose(
        without.astype(np.float32), with_one.astype(np.float32)
    ), "eps=0 and eps=1 gave the same numbers, so eps is being dropped"
    # eps=1 with a variance of 1e-6 is a scale of about one, and eps=0 with the
    # same variance is a scale of about a thousand, so the two outputs differ by
    # two and a half orders of magnitude. The variance cannot be smaller than
    # that here: fp16 tops out near 65504, so a scale of a million would overflow
    # to inf and the test would be measuring the width rather than eps.
    assert np.isfinite(without).all() and np.isfinite(with_one).all()
    assert np.abs(without).max() > 100.0 * np.abs(with_one).max()


def test_affine_false_is_the_identity_scale():
    """A batch norm with no weight and no bias is `x/sqrt(var+eps) - mu/sqrt(var+eps)`.

    The operands are absent rather than empty, which is a shape the upstream
    fold refuses; the affine is still defined, so it is written.
    """
    channels = 5
    x = torch.randn(1, channels, 3, 3, dtype=torch.float16) * 2.0
    model = _populated(channels, affine=False, seed=9)
    _, commands, got = _run(model, x)
    assert len(commands) == 2, f"{len(commands)} commands for a non-affine batch norm"
    scale, shift = _coefficients(model)
    assert _within_budget(got, x, scale, shift) <= 2.0, (
        f"a non-affine batch norm is "
        f"{_within_budget(got, x, scale, shift):.2f} budgets out"
    )

# ---------------------------------------------------------------------------
# What this refuses, and the evidence that a refusal emitted nothing.
# ---------------------------------------------------------------------------


def _command_types(blob):
    _, commands = read_blob(blob)
    return [command.type for command in commands]


def _hand_built_module(channels, value_shape=(1, 4, 4), training=True):
    """An edge program carrying `aten.native_batch_norm` and a chosen shape.

    Built by hand because neither node this file has to refuse can be produced by
    `torch.export` at all: a module in eval mode never carries the training flag,
    and `BatchNorm1d` rejects a one-dimensional value before the trace begins.
    The overload used is the edge one, because that is the graph a pass sees.
    """
    from executorch.exir.dialects._ops import ops as exir_ops

    graph = torch.fx.Graph()
    inp = graph.placeholder("x")
    weight = graph.get_attr("weight")
    bias = graph.get_attr("bias")
    mean = graph.get_attr("mean")
    var = graph.get_attr("var")
    out = graph.call_function(
        exir_ops.edge.aten.native_batch_norm.default,
        (inp, weight, bias, mean, var, training, 0.1, 1e-5),
    )
    read = graph.call_function(operator.getitem, (out, 0))
    graph.output(read)
    module = torch.fx.GraphModule(
        {
            "weight": torch.ones(channels, dtype=torch.float16),
            "bias": torch.zeros(channels, dtype=torch.float16),
            "mean": torch.zeros(channels, dtype=torch.float16),
            "var": torch.ones(channels, dtype=torch.float16),
        },
        graph,
    )
    value = torch.empty(value_shape, dtype=torch.float16)
    inp.meta["val"] = value
    read.meta["val"] = value
    from types import SimpleNamespace

    return SimpleNamespace(
        graph_module=module,
        graph_signature=SimpleNamespace(
            inputs_to_buffers={},
            buffers_to_mutate={},
            inputs_to_parameters={},
            inputs_to_lifted_tensor_constants={},
        ),
        range_constraints={},
    )




def _emits_nothing(program):
    """No command is written for a node the pass refused.

    The backend raises rather than producing a blob, because the refused node is
    still in the graph and the emitter table has no entry for it. That is the
    strongest form of the claim available: there is not a blob with a wrong
    command in it, there is no blob.
    """
    with pytest.raises(RuntimeError, match="no DSP emitter"):
        _blob(program)


def test_training_mode_is_refused_and_the_reason_names_the_flag():
    """A flagged node means the batch statistics, so the running ones are wrong.

    The refusal is not a courtesy: the affine this pass writes is built from the
    running statistics, and applying it to a training-mode node would compute a
    different function with no error anywhere. The name `_no_training` is not
    consulted -- the overload that carries the flag is matched, read, and refused.
    """
    program = _hand_built_module(4)
    node = next(
        n
        for n in program.graph_module.graph.nodes
        if n.op == "call_function" and "native_batch_norm" in str(n.target)
    )
    assert trains_on_batch_statistics(node), "the flag was not read"
    assert batch_norm_is_rewritable(node, program) is not None

    rewrite = RewriteBatchNormToAffine()
    result = rewrite(program)
    assert not result.modified, "a training-mode batch norm was rewritten"
    assert any("training mode" in line for line in rewrite.refused), rewrite.refused
    # Nothing was emitted for it: the node is still there and the blob is the
    # one the graph without a rewrite would have produced.
    assert _bn_nodes(program) == [node]
    _emits_nothing(program)


def test_the_same_overload_unset_is_rewritten():
    """The predicate reads the flag rather than the overload's name.

    Without this the refusal above could be satisfied by refusing the overload
    wholesale, which would also refuse every eval node and the win with it.
    """
    program = _hand_built_module(4)
    node = next(
        n
        for n in program.graph_module.graph.nodes
        if n.op == "call_function" and "native_batch_norm" in str(n.target)
    )
    node.args = (node.args[0], node.args[1], node.args[2], node.args[3], node.args[4],
                 False, 0.1, 1e-5)
    assert not trains_on_batch_statistics(node)
    assert batch_norm_is_rewritable(node, program) is None, batch_norm_is_rewritable(
        node, program
    )


def test_a_reader_of_the_saved_statistics_is_refused():
    """`save_invstd` is a real value this pass cannot compute.

    The op returns three elements and only the first is the normalized value; a
    graph that holds either of the other two is left whole rather than given a
    stand-in that means something else.
    """
    x = torch.randn(1, 3, 4, 4, dtype=torch.float16)
    model = _populated(3, seed=4)
    program = _edge(model, x)
    node = _bn_nodes(program)[0]
    reads = [read for read in node.users if read.op == "call_function"]
    graph = program.graph_module.graph
    with graph.inserting_after(reads[0]):
        graph.call_function(operator.getitem, (node, 2))
    reason = batch_norm_is_rewritable(node, program)
    assert reason is not None and "save_mean" in reason, reason
    rewrite = RewriteBatchNormToAffine()
    assert not rewrite(program).modified
    _emits_nothing(program)


def test_statistics_the_export_cannot_see_are_refused():
    """`s` and `b` are built from the running statistics, so they must be there.

    A graph whose `running_mean` is a method input has no constant affine to
    write. The node stays, and nothing is emitted for it.
    """
    channels = 3
    x = torch.randn(1, channels, 4, 4, dtype=torch.float16)
    model = _populated(channels, seed=6)
    program = _edge(model, x)
    node = _bn_nodes(program)[0]
    # A user input carries no value the export can read: its meta is a fake
    # tensor, which is exactly what a runtime-supplied statistic looks like.
    fake = torch.empty(channels, dtype=torch.float32)
    node.args = (node.args[0], None, None, fake, fake, 1e-5)
    reason = batch_norm_is_rewritable(node, program)
    assert reason is not None and "running_mean" in reason, reason
    assert not RewriteBatchNormToAffine()(program).modified


def test_a_value_with_no_channel_axis_is_refused():
    """A one-dimensional value's statistics are over axis zero, not a channel.

    The `[C, 1, 1]` constant is the right shape for a `[N, C, ...]` value and the
    wrong one here, so this is refused on shape rather than written and caught by
    a broadcast error later.
    """
    # A one-dimensional value: torch reduces over axis 0 and calls it the channel,
    # so a `[C, 1, 1]` constant would be the wrong rank and the pass refuses on
    # shape rather than letting a broadcast error find it later.
    program = _hand_built_module(6, value_shape=(6,), training=False)
    node = _bn_nodes(program)[0]
    reason = batch_norm_is_rewritable(node, program)
    assert reason is not None and "channel axis" in reason, reason
    assert not RewriteBatchNormToAffine()(program).modified
    _emits_nothing(program)

# ---------------------------------------------------------------------------
# The two passes together, and what the fold is worth.
# ---------------------------------------------------------------------------


class _ConvNormCatNorm(torch.nn.Module):
    """The shape a real network has: one norm behind a convolution, one behind a cat.

    A convolution, its batch norm, a branch whose output is concatenated back,
    and a second batch norm over the concatenation -- the second one is the case
    `FoldBatchNormIntoConv` cannot reach, because a `cat` is not a convolution.
    """

    def __init__(self, channels=64, width=4):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.branch = torch.nn.Conv2d(channels, channels, 1, bias=False)
        self.after_conv = torch.nn.BatchNorm2d(channels)
        self.after_cat = torch.nn.BatchNorm2d(2 * channels)

    def forward(self, x):
        first = self.after_conv(self.conv(x))
        return self.after_cat(torch.cat([first, self.branch(first)], dim=1))


def _count(program):
    return len(_bn_nodes(program))


def test_the_fold_takes_the_convolution_and_the_affine_takes_the_rest():
    """The node count before, between and after the two passes.

    This is the win stated in nodes rather than in commands: the fold's claim and
    the affine's claim are about different nodes, and a graph with only the first
    shape would make the second one vacuous.
    """
    torch.manual_seed(0)
    model = _ConvNormCatNorm().eval().half()
    with torch.no_grad():
        model.after_conv.running_mean.uniform_(-2, 2)
        model.after_conv.running_var.uniform_(0.1, 3.0)
        model.after_conv.weight.uniform_(0.5, 2.0)
        model.after_conv.bias.uniform_(-1, 1)
        model.after_cat.running_mean.uniform_(-2, 2)
        model.after_cat.running_var.uniform_(0.1, 3.0)
        model.after_cat.weight.uniform_(0.5, 2.0)
        model.after_cat.bias.uniform_(-1, 1)
    x = torch.randn(1, 64, 4, 4, dtype=torch.float16)
    program = _edge(model, x)
    assert _count(program) == 2, f"{_count(program)} batch norms in the graph"

    assert FoldBatchNormIntoConv()(program).modified
    assert _count(program) == 1, "the fold did not take the convolution's norm"

    assert RewriteBatchNormToAffine()(program).modified
    assert _count(program) == 0, "the affine did not take the rest"
    _, commands = read_blob(_blob(program))
    kinds = [command.type for command in commands]
    # Two convolutions, the channel-block conversions around them, the cat, and
    # then the rewritten norm: one multiply and one add. No batch norm reached
    # the emitter, which is the whole claim of the two passes together.
    assert kinds.count(_BINARY) == 2, f"the rewrite emitted {kinds}"
    assert 12 in kinds, f"the fold did not leave a convolution: {kinds}"


def test_the_fold_is_exact_and_the_measurement_is_in_ulps():
    """A fold is a claim of exactness, so this measures it rather than asserting it.

    The comparison is in fp64, between the exported graph before the fold and the
    same graph after it, so what is left is the fold's own arithmetic and not the
    width. It is not bit-identical: `w/sqrt(var+eps)` and the placement of the
    bias are a reassociation, and the number below is how far it moves. Running
    the same model in fp32 as well says the drift scales with the width, which is
    what makes it round-off rather than a mistake.
    """
    torch.manual_seed(0)
    model = _ConvNormCatNorm().double().eval()
    with torch.no_grad():
        for norm in (model.after_conv, model.after_cat):
            norm.running_mean.uniform_(-2, 2)
            norm.running_var.uniform_(0.1, 3.0)
            norm.weight.uniform_(0.5, 2.0)
            norm.bias.uniform_(-1, 1)
    x = torch.randn(1, 64, 4, 4, dtype=torch.float64)
    program = _edge(model, x)
    before = program.module()(x).detach()
    assert FoldBatchNormIntoConv()(program).modified
    after = program.module()(x)
    scale = float(before.abs().max())
    drift = float((after - before).abs().max())
    assert not torch.equal(after, before), (
        "the fold is bit-identical here, so the measurement below is measuring "
        "nothing; the statistics would have to be chosen to make it round"
    )
    assert drift <= 1e-12 * scale, f"the fold moved the output by {drift} at {scale}"
    # The same reassociation in fp32 on the same statistics, as a fraction of
    # the same output: about six orders of magnitude further out. Drift that
    # shrinks with the width is round-off; drift that does not is a mistake,
    # and the two are told apart by the ratio rather than by the bare number.
    narrow_model = _ConvNormCatNorm().float().eval()
    with torch.no_grad():
        for norm in (narrow_model.after_conv, narrow_model.after_cat):
            norm.running_mean.copy_(model.after_conv.running_mean
                                    if norm is narrow_model.after_conv
                                    else model.after_cat.running_mean)
            norm.running_var.copy_(model.after_conv.running_var
                                   if norm is narrow_model.after_conv
                                   else model.after_cat.running_var)
            norm.weight.copy_(model.after_conv.weight
                              if norm is narrow_model.after_conv
                              else model.after_cat.weight)
            norm.bias.copy_(model.after_conv.bias
                            if norm is narrow_model.after_conv
                            else model.after_cat.bias)
    narrow_program = _edge(narrow_model, x.float())
    narrow_before = narrow_program.module()(x.float())
    FoldBatchNormIntoConv()(narrow_program)
    narrow_drift = float(
        (narrow_program.module()(x.float()) - narrow_before).abs().max()
        / narrow_before.abs().max()
    )
    # Both are relative to the same output, so the quotient is a ratio of
    # round-off: fp32's epsilon is 2**-24 and fp64's is 2**-53, and a
    # reassociation measured at five to eight orders of magnitude above fp64's
    # is rounding rather than a second value being computed.
    quotient = narrow_drift / (drift / scale)
    assert 1e5 < quotient < 1e10, (
        f"fp32 drift is {narrow_drift} relative against {drift / scale} in fp64, "
        f"a quotient of {quotient}"
    )


def test_the_two_passes_together_are_one_delegate():
    """A batch norm with no convolution behind it is what split the delegate chain.

    Without a DSP command the node falls back to a portable kernel, and every
    portable node is a boundary: the same stack of blocks that lowers to one
    delegate lowers to one delegate per block. With the two passes the whole
    stack is one call, and the command list is the evidence for what is in it.
    """
    torch.manual_seed(0)
    model = _ConvNormCatNorm().eval().half()
    x = torch.randn(1, 64, 4, 4, dtype=torch.float16)
    program = to_edge_transform_and_lower(
        export(model, (x,)),
        transform_passes=[FoldBatchNormIntoConv(), RewriteBatchNormToAffine()],
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the stack lowered to {len(calls)} delegates"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    _, commands = read_blob(bytes(lowered._processed_bytes))
    assert _BINARY in [command.type for command in commands], (
        f"the delegate carries {[c.type for c in commands]}"
    )