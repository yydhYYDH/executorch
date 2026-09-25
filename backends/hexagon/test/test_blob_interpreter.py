# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that a blob the emitters wrote produces the right numbers.

The blob comes from the real `HexagonBackend.preprocess` on a real edge program
rather than being assembled by hand, so what is under test is the emitters and
the packing. `blob_interpreter` then runs the command stream the DSP would run,
against an arena laid out the way the runtime lays it out.
"""

import ast
import os
import pathlib
import struct
import sys
from types import SimpleNamespace

import numpy as np
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import blob_interpreter  # noqa: E402
from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.kv_cache import (  # noqa: E402
    _update_cache,
    UPDATE_CACHE,
)
from executorch.backends.hexagon.rms_norm import RMS_NORM  # noqa: E402
from executorch.backends.hexagon.rope import _rope, ROPE  # noqa: E402
from executorch.backends.hexagon.serialization import blob as B  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402


# The outputs are fp16, so the granularity of the comparison is the step of that type at the
# largest value the answer holds -- and a fixed absolute tolerance cannot be both tight for a
# small answer and possible for a large one: at `|v| >= 16` one step is `1.56e-2`. Over 20 000
# free draws of `addmm(8, 64, 128)` the largest disagreement with torch was two steps; four
# leaves a factor of two on that measurement and is far below what a wrong stride moves.


def _fp16_step(expected):
    return float(np.spacing(np.float16(np.max(np.abs(expected.astype(np.float32))))))


_FP16_STEPS = 4


def _program(graph_module):
    """Wrap a hand-built graph module the way preprocess expects a program.

    These graphs own no parameters or buffers, but the backend reads the
    signature to tell an owned constant from a caller-passed input, so the
    wrapper carries an empty one.
    """
    signature = SimpleNamespace(
        inputs_to_buffers={},
        buffers_to_mutate={},
        inputs_to_parameters={},
        inputs_to_lifted_tensor_constants={},
    )
    return SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        range_constraints={},
    )


#: DSP_OP_RASTER_BLIT, the only op every shape here lowers to.
_RASTER_BLIT = 3

#: DSP_OP_LAYER_NORM, which is what the fused norm lowers to.
_LAYER_NORM = 8

#: DSP_OP_BATCH_MATMUL, which is what both products lower to.
_BATCH_MATMUL = 38

#: DSP_OP_BINARY_ELEMENTWISE, which is where add, mul and the comparisons go.
_BINARY_ELEMENTWISE = 19

#: type, n_inputs, n_outputs, n_params, then the params.
_PARAMS_AT = 16

#: patch_param, patch_input, then patch_scale. The scale is the only field of a
#: command that makes a param dynamic, so it is worth naming.
_PATCH_SCALE_AT = 16 + 4 * B.MAX_OP_PARAMS + 8


class _Shapes(torch.nn.Module):
    """Concatenate, slice and transpose -- three blits back to back.

    A wrong region, offset, stride or byte unit shows up as wrong numbers rather
    than as a shape error, which is the point.
    """

    def __init__(self, cut: int) -> None:
        super().__init__()
        self.cut = cut

    def forward(self, a, b):
        joined = torch.cat([a, b], dim=-1)
        return joined[:, :, :, : self.cut].permute(0, 1, 3, 2)


def _blob_and_reference(shape, cut):
    a = torch.randn(*shape, dtype=torch.float16)
    b = torch.randn(*shape, dtype=torch.float16)
    model = _Shapes(cut)
    program = to_edge(export(model, (a, b))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    return blob, (a, b), model(a, b)


def _run(shape, cut):
    blob, (a, b), expected = _blob_and_reference(shape, cut)
    outputs = execute(blob, [a.numpy(), b.numpy()])
    assert len(outputs) == 1, f"expected one output, got {len(outputs)}"
    return np.frombuffer(outputs[0], dtype=np.float16).reshape(expected.shape), expected


def _swap_size_axes(blob: bytes) -> bytes:
    """Moves the row count from size[1] into size[0], the way it was once wrong.

    size[0] is the outermost axis and dstStride[0] is unused when it is 1;
    filling it from size[1] while leaving the strides alone means the outer loop
    re-copies one run instead of walking the rows.
    """
    _, commands = read_blob(blob)
    data = bytearray(blob)
    at = B.HEADER_SIZE
    for command in commands:
        if command.type == _RASTER_BLIT:
            base = at + _PARAMS_AT
            size0, size1 = 3, 4
            struct.pack_into("<i", data, base + 4 * size0, command.params[size1])
            struct.pack_into("<i", data, base + 4 * size1, 1)
            return bytes(data)
        at += B.OP_SIZE
    raise AssertionError("the blob has no blit to perturb")


def _cache_program(cache_shape, value_shape):
    """A graph holding just the fused cache advance.

    The emitters only need a graph module, so the program is a namespace around
    a hand-built one rather than a full export: there is no checkpointed Llama
    small enough to reach this op through the real pipeline.
    """
    graph = torch.fx.Graph()
    cache = graph.placeholder("cache")
    cache.meta["val"] = torch.empty(cache_shape, dtype=torch.float16)
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty(value_shape, dtype=torch.float16)
    position = graph.placeholder("position")
    position.meta["val"] = torch.empty(1, dtype=torch.int64)

    fused = graph.call_function(UPDATE_CACHE, args=(cache, value, position))
    fused.meta["val"] = torch.empty(cache_shape, dtype=torch.float16)
    graph.output(fused)
    return _program(torch.fx.GraphModule(torch.nn.Module(), graph))


def _run_cache(cache_shape, value_shape, position):
    cache = torch.arange(int(np.prod(cache_shape)), dtype=torch.float16).reshape(
        cache_shape
    )
    value = torch.randn(*value_shape, dtype=torch.float16)
    blob = HexagonBackend.preprocess(_cache_program(cache_shape, value_shape), [])
    blob = blob.processed_bytes
    got = np.frombuffer(
        execute(
            blob, [cache.numpy(), value.numpy(), np.array([position], dtype=np.int64)]
        )[0],
        dtype=np.float16,
    ).reshape(cache_shape)
    return blob, got, _update_cache(cache, value, torch.tensor([position])).numpy()


def test_cache_advance_matches_torch():
    for cache_shape, value_shape, position in (
        ((1, 16, 2, 8), (1, 2, 2, 8), 0),
        ((1, 16, 2, 8), (1, 2, 2, 8), 3),
        ((1, 16, 2, 8), (1, 4, 2, 8), 12),
    ):
        _, got, expected = _run_cache(cache_shape, value_shape, position)
        assert np.array_equal(got, expected), f"cache differs at position {position}"


def test_a_wrong_patch_scale_is_caught():
    """The position reaches the region only through the patch slot.

    `dstOffset` has to be `position * inner`, and the patch copies the position
    through unscaled, so `patch_scale` is the whole mechanism. Setting it to one
    has to move the written rows; if it did not, these tests would pass without
    the scale doing anything.
    """
    cache_shape, value_shape, position = (1, 16, 2, 8), (1, 2, 2, 8), 5
    blob, got, expected = _run_cache(cache_shape, value_shape, position)
    assert np.array_equal(got, expected)

    cache = torch.arange(int(np.prod(cache_shape)), dtype=torch.float16).reshape(
        cache_shape
    )
    value = torch.randn(*value_shape, dtype=torch.float16)
    data = bytearray(blob)
    header, commands = read_blob(blob)
    at = B.HEADER_SIZE
    patched = 0
    for command in commands:
        if command.patch_param != B.NO_PATCH:
            assert command.patch_scale != 1, "the scale is already one"
            struct.pack_into("<I", data, at + _PATCH_SCALE_AT, 1)
            patched += 1
        at += B.OP_SIZE
    assert patched == 1, f"expected one patched op, found {patched}"

    bad = np.frombuffer(
        execute(
            bytes(data),
            [cache.numpy(), value.numpy(), np.array([position], dtype=np.int64)],
        )[0],
        dtype=np.float16,
    ).reshape(cache_shape)
    assert not np.array_equal(
        bad, expected
    ), "unscaling the patch changed nothing, so the scale is not being checked"


def _norm_blob(shape, eps):
    """A graph holding just the fused norm, with the weight as a constant.

    The node is built rather than reached by fusing an export: FuseRmsNormPass
    anchors on the weight multiply that RMSNorm.forward ends with, and a graph
    written to match it here would be testing the pattern rather than the
    emitter. The fusion itself is what the Qwen3 run measures.
    """
    weight = ((torch.arange(shape[-1]) % 5) * 0.25 + 0.5).half()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, dtype=torch.float16)
    gamma = graph.get_attr("weight")
    gamma.meta["val"] = torch.empty(shape[-1], dtype=torch.float16)
    fused = graph.call_function(RMS_NORM, args=(x, gamma, eps))
    fused.meta["val"] = torch.empty(shape, dtype=torch.float16)
    graph.output(fused)
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(weight, requires_grad=False)
    graph_module = torch.fx.GraphModule(root, graph)
    blob = HexagonBackend.preprocess(_program(graph_module), []).processed_bytes
    return blob, weight


def _run_norm(shape, eps):
    x = torch.randn(*shape, dtype=torch.float16)
    blob, weight = _norm_blob(shape, eps)
    got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16).reshape(shape)
    expected = torch.nn.functional.rms_norm(x, (shape[-1],), weight, eps).numpy()
    return blob, x, got, expected


def test_fused_norm_matches_torch():
    # fp16 carries about three decimal digits, and the kernel reduces in a
    # different order than numpy, so this is a tolerance rather than equality.
    for shape, eps in (((1, 8), 1e-5), ((4, 64), 1e-6), ((2, 3, 128), 1e-5)):
        _, _, got, expected = _run_norm(shape, eps)
        worst = float(
            np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
        )
        assert worst < 2e-2, f"norm differs by {worst} at {shape}"


def test_a_layernorm_flag_is_caught():
    """RMS mode is a flag in the last param, not a separate kernel.

    Clearing it makes the kernel subtract the mean as well, so the output has to
    move. If it did not, nothing here would be checking that the emitter asks
    for an RMSNorm rather than a LayerNorm.
    """
    shape, eps = (4, 64), 1e-5
    blob, x, got, expected = _run_norm(shape, eps)
    worst = float(np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32))))
    assert worst < 2e-2

    data = bytearray(blob)
    _, commands = read_blob(blob)
    at = B.HEADER_SIZE
    flipped = 0
    for command in commands:
        if command.type == _LAYER_NORM:
            assert command.params[3] == 1, "the emitter did not ask for RMS mode"
            struct.pack_into("<i", data, at + _PARAMS_AT + 4 * 3, 0)
            flipped += 1
        at += B.OP_SIZE
    assert flipped == 1, f"expected one norm op, found {flipped}"

    bad = np.frombuffer(execute(bytes(data), [x.numpy()])[0], dtype=np.float16).reshape(
        shape
    )
    assert not np.allclose(
        bad, expected, atol=2e-2
    ), "clearing the RMS flag changed nothing, so the flag is not being checked"


def _rope_table_shape(shape):
    """The table's shape: one row per token, broadcast over the head axis.

    The kernel only reads the row, so the head axis is what makes the tables
    broadcast against the activations in torch's own expression of the rotation.
    """
    if len(shape) == 3:
        return (shape[0], 1, shape[-1])
    return (shape[0], shape[1], 1, shape[-1])


def _rope_blob(shape):
    """A graph holding just the fused rope, on a hand-built node.

    Like the fused norm, the op is reached by hand rather than by matching the
    decomposition: FuseRopePass anchors on the add that ends the rotate-half
    chain, and a graph written to match it would be testing the pattern.
    """
    table_shape = _rope_table_shape(shape)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, dtype=torch.float16)
    cos = graph.placeholder("cos")
    cos.meta["val"] = torch.empty(table_shape, dtype=torch.float16)
    sin = graph.placeholder("sin")
    sin.meta["val"] = torch.empty(table_shape, dtype=torch.float16)
    fused = graph.call_function(ROPE, args=(x, cos, sin))
    fused.meta["val"] = torch.empty(shape, dtype=torch.float16)
    graph.output(fused)
    return HexagonBackend.preprocess(
        _program(torch.fx.GraphModule(torch.nn.Module(), graph)), []
    ).processed_bytes


def test_rope_matches_torch():
    # A one-wide batch axis folds into the sequence, and a three-dim shape has
    # none at all, so both spellings the exporter produces are covered. The
    # kernel multiplies in fp16, so this is a tolerance rather than equality.
    for shape in ((2, 4, 8), (1, 3, 2, 16)):
        x = torch.randn(*shape, dtype=torch.float16)
        table_shape = _rope_table_shape(shape)
        cos = torch.randn(*table_shape, dtype=torch.float16)
        sin = torch.randn(*table_shape, dtype=torch.float16)
        blob = _rope_blob(shape)
        got = np.frombuffer(
            execute(blob, [x.numpy(), cos.numpy(), sin.numpy()])[0], dtype=np.float16
        ).reshape(shape)
        expected = _rope(x, cos, sin).numpy()
        worst = float(
            np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
        )
        assert worst < 2e-2, f"rope differs by {worst} at {shape}"


class _Mm(torch.nn.Module):
    """A plain matrix product, so the subgraph is one BATCH_MATMUL."""

    def forward(self, a, b):
        return torch.mm(a, b)


def _mm_blob(m, k, n):
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    program = to_edge(export(_Mm(), (a, b))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    return blob, a, b


def _run_mm(m, k, n):
    blob, a, b = _mm_blob(m, k, n)
    got = np.frombuffer(execute(blob, [a.numpy(), b.numpy()])[0], dtype=np.float16)
    # Comparing against a fp32 product is the delegate's own contract -- fp16
    # operands accumulated in fp32 -- which is not what torch's fp16 mm does.
    expected = (a.float() @ b.float()).half().numpy().reshape(-1)
    return blob, a, b, got, expected


def test_matmul_matches_torch():
    for m, k, n in ((1, 1, 1), (4, 8, 3), (16, 32, 16)):
        _, _, _, got, expected = _run_mm(m, k, n)
        assert (
            got.shape == expected.shape
        ), f"{got.shape} != {expected.shape} at {m}x{k}x{n}"
        worst = float(
            np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
        )
        assert worst <= _FP16_STEPS * _fp16_step(
            expected
        ), f"matmul differs by {worst} at {m}x{k}x{n}"


def test_a_broken_contraction_is_caught():
    """The contraction stride is one int in a 26-param descriptor.

    Zeroing it makes every term read the first row of the right operand, which
    is a wrong product that still has the right shape. If the result did not
    move, these tests would not be checking the descriptor at all.
    """
    m, k, n = 4, 8, 3
    blob, a, b, got, expected = _run_mm(m, k, n)
    worst = float(np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32))))
    assert worst <= _FP16_STEPS * _fp16_step(expected)

    data = bytearray(blob)
    _, commands = read_blob(blob)
    assert len(commands) == 1 and commands[0].type == _BATCH_MATMUL
    # params[1] starts the descriptor; src1StrideXYZ[1] is its int 11.
    struct.pack_into("<i", data, B.HEADER_SIZE + _PARAMS_AT + 4 * 12, 0)

    bad = np.frombuffer(
        execute(bytes(data), [a.numpy(), b.numpy()])[0], dtype=np.float16
    )
    assert not np.array_equal(bad, got), (
        "zeroing the contraction stride changed nothing, so the descriptor is "
        "not being checked"
    )


class _Bmm(torch.nn.Module):
    """A batch of tiles contracted against a stack, so the loop count is the batch."""

    def forward(self, a, b):
        return torch.bmm(a, b)


def _bmm_blob(batches, m, k, n):
    a = torch.randn(batches, m, k, dtype=torch.float16)
    b = torch.randn(batches, k, n, dtype=torch.float16)
    program = to_edge(export(_Bmm(), (a, b))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    return blob, a, b


def _run_bmm(batches, m, k, n):
    blob, a, b = _bmm_blob(batches, m, k, n)
    got = np.frombuffer(execute(blob, [a.numpy(), b.numpy()])[0], dtype=np.float16)
    expected = torch.bmm(a.float(), b.float()).half().numpy().reshape(-1)
    return blob, a, b, got, expected


def test_batched_matmul_matches_torch():
    """One iteration per batch element, which is what the descriptor steps say.

    The shapes cover the degenerate batch, a non-square tile and a contraction
    that is neither a power of two nor a multiple of the DSP's tile width.
    """
    for batches, m, k, n in ((1, 16, 24, 32), (4, 16, 24, 32), (3, 8, 48, 20)):
        _, _, _, got, expected = _run_bmm(batches, m, k, n)
        assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
        worst = float(
            np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
        )
        assert worst <= _FP16_STEPS * _fp16_step(
            expected
        ), f"bmm differs by {worst} at {batches}x{m}x{k}x{n}"


def test_a_batch_step_of_zero_is_caught():
    """Zeroing the output step makes every iteration write the first tile.

    That is a wrong result of the right shape: if the numbers did not move, the
    steps in the descriptor are not what carries the batch.
    """
    batches, m, k, n = 4, 16, 24, 32
    blob, a, b, got, expected = _run_bmm(batches, m, k, n)
    worst = float(np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32))))
    assert worst <= _FP16_STEPS * _fp16_step(expected)

    data = bytearray(blob)
    _, commands = read_blob(blob)
    assert len(commands) == 1 and commands[0].type == _BATCH_MATMUL
    assert commands[0].params[1] == batches, commands[0].params[:5]
    # cmdSteps[0], the first int of the step triple.
    struct.pack_into("<i", data, B.HEADER_SIZE + _PARAMS_AT + 4 * 14, 0)

    bad = np.frombuffer(
        execute(bytes(data), [a.numpy(), b.numpy()])[0], dtype=np.float16
    )
    assert not np.array_equal(
        bad, got
    ), "zeroing the output step changed nothing, so the batch is not being walked"


class _Addmm(torch.nn.Module):
    """A weight and a bias the emitter has to reach as constants. bias is one input."""

    def __init__(self, k, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n, dtype=torch.float16) * 0.5)
        self.bias = torch.nn.Parameter(torch.randn(n, dtype=torch.float16) * 0.5)

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight)


def _run_addmm(m, k, n):
    x = torch.randn(m, k, dtype=torch.float16)
    model = _Addmm(k, n)
    program = to_edge(export(model, (x,))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    # The backend owns the lifted parameters, so they are stored in the blob's
    # weight section and x is the only method input.
    operands = [x.numpy()]
    got = np.frombuffer(execute(blob, operands)[0], dtype=np.float16)
    # The delegate rounds the product to fp16 in an activation before the bias is
    # added, so the reference does the same rather than adding in fp32.
    expected = (
        (
            (x.float() @ model.weight.detach().float()).half().float()
            + model.bias.detach().float()
        )
        .half()
        .numpy()
        .reshape(-1)
    )
    return blob, operands, got, expected


def test_addmm_matches_torch():
    """The product then the broadcast bias, as the two commands the emitter writes."""
    for m, k, n in ((8, 64, 128), (4, 8, 3)):
        blob, _, got, expected = _run_addmm(m, k, n)
        _, commands = read_blob(blob)
        assert [c.type for c in commands] == [
            _BATCH_MATMUL,
            _BINARY_ELEMENTWISE,
        ], f"addmm emitted {[c.type for c in commands]}"
        # params[2] of the binary op is the bias length, which the broadcast walks.
        assert commands[1].params[2] == n, commands[1].params[:8]
        worst = float(
            np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
        )
        assert worst <= _FP16_STEPS * _fp16_step(
            expected
        ), f"addmm differs by {worst} at {m}x{k}x{n}"


def test_the_bias_broadcast_strides_are_the_ones_read():
    """The bias is one row repeated down the tile, so its outer stride is zero.

    Collapsing its inner stride makes every column read the same bias entry,
    which is a wrong sum of the right shape. If the numbers did not move, the
    strides in the broadcast tail are not what places the bias.
    """
    m, k, n = 8, 64, 128
    blob, operands, got, expected = _run_addmm(m, k, n)
    # The same shape is asserted by `test_addmm_matches_torch` above, at the step of
    # the fp16 it holds: an accumulation's order is not the reference's, so the two
    # disagree on the elements a rounding step from a boundary. Over 20 000 free
    # draws the largest disagreement was 1.56e-2, two steps at `|v| >= 16`, which the
    # fixed `1e-2` this file used to carry would have called a failure.
    assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
    worst = float(np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32))))
    assert worst <= _FP16_STEPS * _fp16_step(
        expected
    ), f"addmm differs by {worst} at {m}x{k}x{n}"

    data = bytearray(blob)
    _, commands = read_blob(blob)
    assert commands[1].type == _BINARY_ELEMENTWISE
    # The broadcast tail starts at params[8]: rank, outDims[8], in0Strides[8],
    # in1Strides[8]. in1Strides is the bias, one row over the tile: its outer
    # stride is zero and its inner one is the contiguous unit.
    assert commands[1].params[25] == 0, commands[1].params[25:33]
    assert commands[1].params[26] == 1, commands[1].params[25:33]
    # The add is the second command of the blob, so its params sit one op later.
    struct.pack_into("<i", data, B.HEADER_SIZE + B.OP_SIZE + _PARAMS_AT + 4 * 26, 0)

    bad = np.frombuffer(execute(bytes(data), operands)[0], dtype=np.float16).reshape(
        got.shape
    )
    assert not np.array_equal(
        bad, got
    ), "changing the bias stride changed nothing, so it is not being read"


class _Eltwise(torch.nn.Module):
    """A product then a negation: one BINARY_ELEMENTWISE and one UNARY."""

    def forward(self, a, b):
        return torch.neg(torch.mul(a, b))


class _Scaled(torch.nn.Module):
    """A product against a shorter operand, which is the broadcast path."""

    def forward(self, a, b):
        return a * b


def _run_eltwise(model, args, expected):
    program = to_edge(export(model, args)).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    got = np.frombuffer(execute(blob, [t.numpy() for t in args])[0], dtype=np.float16)
    return blob, got.reshape(expected.shape), expected.numpy()


def test_elementwise_matches_torch():
    a = torch.randn(2, 3, 4, dtype=torch.float16)
    b = torch.randn(2, 3, 4, dtype=torch.float16)
    _, got, expected = _run_eltwise(_Eltwise(), (a, b), torch.neg(torch.mul(a, b)))
    assert np.array_equal(got, expected), "mul/neg differs from torch"


def test_broadcast_matches_torch():
    a = torch.randn(2, 3, 4, dtype=torch.float16)
    b = torch.randn(4, dtype=torch.float16)
    _, got, expected = _run_eltwise(_Scaled(), (a, b), a * b)
    assert np.array_equal(got, expected), "the broadcast product differs from torch"


def test_a_wrong_broadcast_stride_is_caught():
    """A broadcast dimension is a stride of zero, and the wrong stride still adds.

    Setting the innermost stride of the short operand to zero makes every
    element read the same value -- the right shape, the wrong product.
    """
    a = torch.randn(2, 3, 4, dtype=torch.float16)
    b = torch.randn(4, dtype=torch.float16)
    blob, got, expected = _run_eltwise(_Scaled(), (a, b), a * b)
    assert np.array_equal(got, expected)

    data = bytearray(blob)
    _, commands = read_blob(blob)
    assert len(commands) == 1 and commands[0].type == _BINARY_ELEMENTWISE
    assert commands[0].params[27] == 1, commands[0].params[25:33]
    # in1StrideXYZ[2], the innermost extent of the short operand.
    struct.pack_into("<i", data, B.HEADER_SIZE + _PARAMS_AT + 4 * 27, 0)

    bad = np.frombuffer(
        execute(bytes(data), [a.numpy(), b.numpy()])[0], dtype=np.float16
    ).reshape(got.shape)
    assert not np.array_equal(bad, got), (
        "zeroing the innermost broadcast stride changed nothing, so the strides "
        "are not being checked"
    )


def test_blob_matches_torch_on_the_host():
    got, expected = _run((1, 2, 3, 4), 4)
    assert np.array_equal(got, expected.numpy()), "blob output differs from torch"


def test_blob_matches_torch_across_shapes_and_cuts():
    # A run of one, a partial row, and a whole row each exercise a different
    # path through the region.
    for shape, cut in (((1, 1, 1, 1), 1), ((2, 3, 5, 7), 3), ((1, 4, 8, 16), 16)):
        got, expected = _run(shape, cut)
        assert np.array_equal(got, expected), f"differs from torch at {shape}/{cut}"


def test_a_perturbed_region_changes_the_result():
    """A check whose numbers do not move is not a check.

    The region is nudged into the shape of a real mistake and the output has to
    disagree with torch afterwards; if it still agreed, this test would be
    passing for the wrong reason.
    """
    blob, (a, b), expected = _blob_and_reference((1, 2, 3, 4), 4)
    good = np.frombuffer(
        execute(blob, [a.numpy(), b.numpy()])[0], dtype=np.float16
    ).reshape(expected.shape)
    assert np.array_equal(good, expected.numpy())

    bad = np.frombuffer(
        execute(_swap_size_axes(blob), [a.numpy(), b.numpy()])[0], dtype=np.float16
    ).reshape(expected.shape)
    assert not np.array_equal(bad, expected.numpy()), (
        "swapping the size axes changed nothing, so these tests cannot detect a "
        "wrong region"
    )


class _LeadingAxes(torch.nn.Module):
    """Permutations that move the leading axes, with a contiguous run kept whole.

    A region is three nested loops, so a permutation is describable exactly when
    the axes group into runs that keep their order in both layouts. These are the
    shapes the vision tower's attention needs, where the axes that move are the
    outer ones and the head run travels as one.
    """

    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def _run_permute(shape, dims):
    x = torch.randn(*shape, dtype=torch.float16)
    model = _LeadingAxes(dims)
    program = to_edge(export(model, (x,))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    expected = model(x)
    got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16).reshape(
        expected.shape
    )
    return blob, got, expected.numpy()


def test_leading_axes_permutes_match_torch():
    """The permutations the region has to describe beyond the last two axes.

    The fused qkv of the vision tower swaps the two outer axes of a 4-D tile and
    carries the head run whole; the attention output swaps three of the four axes
    of an unsqueezed tile. Both are one region each, and a wrong grouping shows
    up as wrong numbers rather than as a shape error.
    """
    for shape, dims in (
        ((64, 3, 16, 64), (1, 0, 2, 3)),
        ((64, 16, 64), (1, 0, 2)),
        ((1, 16, 64, 64), (2, 0, 1, 3)),
        ((16, 1, 64, 64), (1, 2, 0, 3)),
        ((1, 16, 64, 64), (0, 1, 3, 2)),
        ((2, 3, 5, 7), (1, 0, 2, 3)),
    ):
        _, got, expected = _run_permute(shape, dims)
        assert np.array_equal(got, expected), f"differs from torch at {shape}/{dims}"


def _permute_copy(shape, dims):
    """A permute_copy node, holding the shapes the region is computed from."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=torch.float16)
    node = graph.call_function(
        exir_ops.edge.aten.permute_copy.default, args=(source, list(dims))
    )
    node.meta["val"] = torch.empty([shape[d] for d in dims], dtype=torch.float16)
    return node


def test_a_permutation_needing_a_fourth_run_is_refused():
    """Four groups is one more loop than a region has, so it has to be refused.

    Swapping the two outer axes *and* the two inner ones pairs every axis with a
    different one, so every group is a single axis and no single region describes
    it. The emitter would write the wrong elements rather than fail, which is why
    the refusal lives in the predicate the support check reads.
    """
    assert hexagon_ops.permute_region(_permute_copy((2, 3, 5, 7), (1, 0, 3, 2))) is None
    assert (
        hexagon_ops.permute_region(_permute_copy((2, 3, 5, 7), (1, 0, 2, 3)))
        is not None
    )
    assert (
        hexagon_ops.permute_region(_permute_copy((2, 3, 4, 5, 6), (4, 3, 2, 1, 0)))
        is None
    )


def test_a_head_split_on_a_batch_of_one_is_three_loops():
    """The batch axis of a head split is an extent of one, so it is not a loop.

    `[1, tokens, heads, dim]` to `[1, heads, tokens, dim]` is how a batch-one
    export spells every head split -- an SD cross-attention's query, key, value
    and its output merge, and the same shape in every vision tower. The axes
    split into four groups, but the group holding the batch axis multiplies to
    one: it iterates once and neither stride advances, so the permutation is
    three levels and counting that group only refuses it. The identical split at
    a batch of two needs four loops that do advance, and stays refused -- which
    is what makes the acceptance a rule about extents rather than a loosening.
    """
    for shape in ((1, 16, 8, 40), (1, 77, 8, 40), (1, 4096, 8, 40), (1, 16, 5, 64)):
        assert (
            hexagon_ops.permute_region(_permute_copy(shape, (0, 2, 1, 3))) is not None
        ), f"refused {shape}"
    for shape in ((1, 16, 8, 40), (1, 16, 5, 64)):
        _, got, expected = _run_permute(shape, (0, 2, 1, 3))
        assert np.array_equal(got, expected), f"differs from torch at {shape}"

    assert (
        hexagon_ops.permute_region(_permute_copy((2, 16, 8, 40), (0, 2, 1, 3))) is None
    )


#: Nothing is excused. FLASH_ATTN was, on the grounds that modelling it needed
#: the kernel's approximated exponential; it turned out to need only the causal
#: rule and the head grouping, so the interpreter covers it and the set below is
#: the whole command stream an emitter can produce.


def _emitted_op_types():
    """The DSP op types the emitters can produce, read out of their source.

    Parsed rather than exercised: every `type=` argument in hexagon_ops.py names
    a DSP_OP_ constant, and that is the whole set of commands that can reach a
    blob.
    """
    tree = ast.parse(pathlib.Path(hexagon_ops.__file__).read_text())
    names = {
        node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "type"
        and isinstance(node.value, ast.Name)
    }
    return {getattr(hexagon_ops, name) for name in names}


def test_every_op_an_emitter_can_emit_is_modelled():
    """The interpreter has to cover the command stream, not a sample of it."""
    emitted = _emitted_op_types()
    assert len(emitted) >= 8, f"only found {sorted(emitted)} -- the parse failed"
    missing = emitted - set(blob_interpreter._EXECUTORS)
    assert not missing, f"an emitter produces unmodelled ops: {sorted(missing)}"


def test_the_ops_actually_emitted_are_the_ones_we_think():
    """Pins the set, so a new emitter is a failure here rather than a silent
    hole in the interpreter's coverage."""
    # 1 pool2d, 2 depthwise convolution, 3 blit, 4 unary, 8 layer norm, 12 im2col
    # convolution, 14 rope, 16 add+fused norm, 18 flash attention, 19 element-wise,
    # 22 q4a16 prefill, 23 shared gather, 24 zero, 26 select, 27 topk, 28 softmax,
    # 29 reduction, 38 batch matmul, 41 q4a16 GEMV, 43 vision attention, 45 w8a16
    # GEMV, 47 argmax/argmin.
    # Tensor convert (7) is in the DSP's enum but no emitter here produces it.
    assert _emitted_op_types() == {
        1,
        2,
        3,
        4,
        8,
        12,
        14,
        16,
        18,
        19,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        38,
        41,
        43,
        45,
        47,
    }
