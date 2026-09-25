# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization from the PT2E annotation to the DSP command.

The AOT half goes through the real quantizer, `prepare_pt2e`/`convert_pt2e`
and `HexagonBackend.preprocess`, so what is under test is the whole chain: the
weight-only annotation, the pattern the emitters match, the host-side weight
packing and the command's parameters. `blob_interpreter` then runs the bytes the
DSP would run, against the same arena the runtime builds.
"""

import os
import pathlib
import re
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import blob_interpreter  # noqa: E402
from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon import quantizer as hexagon_quantizer  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import (  # noqa: E402
    HexagonBackend,
    owned_weight,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    _data_placeholders,
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.backends.hexagon.quantizer import (  # noqa: E402
    get_hexagon_quantization_config,
    get_hexagon_quantizer,
    get_q4a16_config,
    get_w8a16_config,
    HexagonQuantizer,
    SUPPORTED_SCHEMES,
)
from executorch.exir import to_edge, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from executorch.exir.lowered_backend_module import LoweredBackendModule  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: E402
    convert_pt2e,
    prepare_pt2e,
)

_Q4A16 = blob_interpreter.MATMUL_Q4A16_GEMV_I8
_W8A16 = blob_interpreter.MATMUL_W8A16_GEMV_I8
_PREFILL = blob_interpreter.MATMUL_Q4A16_FP16
_BLIT = blob_interpreter.RASTER_BLIT


class _Live(torch.nn.Module):
    """`a @ b` over two run-time tensors: neither operand is a weight."""

    def __init__(self, spelling):
        super().__init__()
        self.spelling = spelling

    def forward(self, a, b):
        return a @ b if self.spelling == "at" else torch.mm(a, b)


class _Mm(torch.nn.Module):
    """A weight-only matmul, which is the pattern the quantizer annotates."""

    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n) * 0.3)

    def forward(self, x):
        return torch.mm(x, self.weight)


class _At(torch.nn.Module):
    """The same multiply written `@`, which exports as a different op.

    `torch.mm` is `aten.mm`; `x @ w` is `aten.matmul`, and for two 2-D operands
    `to_edge` rewrites that one back into `mm`. The annotation stage sees only
    the matmul, so the weight has to be annotated there for either spelling to
    reach a GEMV.
    """

    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n) * 0.3)

    def forward(self, x):
        return x @ self.weight


class _Bias(torch.nn.Module):
    """The same matmul with a bias of a caller-chosen shape."""

    def __init__(self, k: int, n: int, bias_shape) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n) * 0.3)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape) * 0.1)

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight)


class _Linear(torch.nn.Module):
    """`nn.Linear`, the spelling every projection in a transformer is written in.

    Its weight is stored `[out, in]` rather than `[k, n]`, so `to_edge` lowers it
    to an `addmm` over a `permute_copy` of the weight -- one node further from
    the emitter's pattern than `_Mm`, and the reason the quantizer rewrites an
    admissible one into the `addmm` spelling before the observers go in.
    """

    def __init__(self, k: int, n: int, bias: bool = False) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(k, n, bias=bias)

    def forward(self, x):
        return self.projection(x)


class _BiasedLinear(_Linear):
    """`nn.Linear` with the bias it carries by default, which is the common one."""

    def __init__(self, k: int, n: int) -> None:
        super().__init__(k, n, bias=True)


def _carries_a_dequantize(converted) -> bool:
    """Whether the converted graph holds the weight-only dequantize."""
    return any(
        node.op == "call_function"
        and node.target
        is torch.ops.quantized_decomposed.dequantize_per_channel.default
        for node in converted.graph.nodes
    )


def _edge_program(converted, *inputs):
    """The edge program for an already-converted graph module."""
    return to_edge(torch.export.export(converted, inputs)).exported_program()


def _command_types(converted, *inputs):
    """The command types the backend emits for an already-converted graph."""
    program = _edge_program(converted, *inputs)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    return [command.type for command in read_blob(blob)[1]]


def _serializes(converted, *inputs) -> int:
    """Serialize the lowered program, in bytes, or fail the way a caller would.

    `to_executorch` is the tier that says a program exports, and it is stricter
    than the lowering that most of the shape tests here stop at: every
    functional op in the portable part of the graph is converted to its out
    variant, and `quantized_decomposed` has none -- neither the dequantize a
    weight-only annotation leaves behind nor the quantize a run-time weight would.
    A program with one stranded in it cannot be serialized at all, whichever
    kernels a runner links, so a graph that stays portable only counts as
    portable if this returns.
    """
    lowered = to_edge_transform_and_lower(
        torch.export.export(converted, inputs),
        partitioner=[HexagonPartitioner()],
    )
    return len(bytes(lowered.to_executorch().buffer))


def _quantized_program(scheme, m, k, n):
    """Export, quantize and lower one mm, the way a real pipeline would.

    Returns the edge program the backend preprocesses, the converted graph
    module whose forward is the dequantize-plus-mm reference, and the input.
    """
    converted, x = _converted(scheme, m, k, n)
    program = to_edge(torch.export.export(converted, (x,))).exported_program()
    return program, converted, x


def _at_program(scheme, x_shape, k, n):
    """The same program with the multiply written `@`.

    `x_shape` is the activation's whole shape, since its rank is what decides
    whether this `@` is the 2-D `mm` spelling or one with a batch axis in front
    of it.
    """
    x = torch.randn(*x_shape, dtype=torch.float32)
    converted, _ = _quantize(scheme, _At(k, n).eval(), x)
    program = to_edge(torch.export.export(converted, (x,))).exported_program()
    return program, converted, x


def _converted(scheme, m, k, n, module=_Mm):
    """The PT2E-converted module and its input, before any lowering."""
    x = torch.randn(m, k, dtype=torch.float32)
    converted, _ = _quantize(scheme, module(k, n).eval(), x)
    return converted, x


def _linear_program(scheme, m, k, n, module=_Linear):
    """`_quantized_program` again, over the spelling a model is written in."""
    converted, x = _converted(scheme, m, k, n, module=module)
    return _edge_program(converted, x), converted, x


def _quantize(scheme, model, *inputs):
    """One PT2E trip over `model`: annotate, calibrate, convert."""
    exported = torch.export.export(model, inputs)
    prepared = prepare_pt2e(exported.module(), get_hexagon_quantizer(scheme))
    with torch.no_grad():
        prepared(*inputs)
    return convert_pt2e(prepared), inputs


def _annotated(model, x, scheme="q4a16"):
    """The targets `annotate` gave a weight annotation to.

    `annotate` is called on its own: the question here is what it decided about
    the node, and a lowering afterwards would answer a different one.
    """
    exported = torch.export.export(model, (x,))
    annotated = get_hexagon_quantizer(scheme).annotate(exported.module())
    return {
        node.target
        for node in annotated.graph.nodes
        if node.meta.get("quantization_annotation") is not None
        and node.meta["quantization_annotation"].input_qspec_map
    }


def _quantized_addmm_program(scheme, k, n, bias_shape):
    """The same, for addmm with a bias of the given shape."""
    model = _Bias(k, n, bias_shape).eval()
    x = torch.randn(1, k, dtype=torch.float32)
    exported = torch.export.export(model, (x,))
    prepared = prepare_pt2e(exported.module(), get_hexagon_quantizer(scheme))
    with torch.no_grad():
        prepared(x)
    converted = convert_pt2e(prepared)
    program = to_edge(torch.export.export(converted, (x,))).exported_program()
    return program, converted, x


def _support(program):
    """The partitioner's own view of which of a program's inputs it owns.

    A bare `HexagonOperatorSupport()` sees the graph and nothing else, and
    whether a weight is a constant is a fact about the program's signature
    rather than about the node -- a parameter and a run-time tensor are both
    placeholders carrying a fake tensor. The partitioner passes the names of the
    values the program owns, so a test that asks the support check a question
    about a weight has to pass them too.
    """
    return HexagonOperatorSupport(_data_placeholders(program))


def _node_of(program, target):
    return next(
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.target is target
    )


def _mm_node(program):
    return _node_of(program, exir_ops.edge.aten.mm.default)


def _relative_error(got, expected):
    return float(np.max(np.abs(got - expected))) / (
        float(np.max(np.abs(expected))) + 1e-6
    )


def _run(scheme, k, n):
    program, converted, x = _quantized_program(scheme, 1, k, n)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    got = np.frombuffer(execute(blob, [x.half().numpy()])[0], dtype=np.float16).reshape(
        n
    )
    with torch.no_grad():
        expected = converted(x).detach().float().numpy().reshape(n)
    return blob, got.astype(np.float32), expected


def test_quantizer_exposes_the_kernels_schemes():
    assert set(SUPPORTED_SCHEMES) == {"q4a16", "w8a16"}
    for scheme in SUPPORTED_SCHEMES:
        config = get_hexagon_quantization_config(scheme)
        # Weight-only: no activation annotation, so the graph stays fp16.
        assert config.input_activation is None
        assert config.output_activation is None
        assert config.weight.qscheme == torch.per_channel_symmetric
        quantizer = get_hexagon_quantizer(scheme)
        assert isinstance(quantizer, HexagonQuantizer)
    assert get_q4a16_config().weight.quant_min == -8
    assert get_q4a16_config().weight.quant_max == 7
    assert get_w8a16_config().weight.quant_min == -128
    assert get_w8a16_config().weight.quant_max == 127


def test_q4a16_gemv_matches_the_dequantized_reference():
    for k, n in ((64, 32), (128, 96)):
        blob, got, expected = _run("q4a16", k, n)
        _, commands = read_blob(blob)
        assert [c.type for c in commands] == [_Q4A16], [c.type for c in commands]
        assert commands[0].params[1] == k and commands[0].params[2] == n
        # int4 weight plus per-token int8 activation: about a percent of the
        # output's own magnitude, so the comparison is a tolerance.
        worst = _relative_error(got, expected)
        assert worst < 0.06, f"q4a16 differs by {worst} at {k}x{n}"


def test_w8a16_gemv_matches_the_dequantized_reference():
    for k, n in ((64, 32), (128, 96)):
        blob, got, expected = _run("w8a16", k, n)
        _, commands = read_blob(blob)
        assert [c.type for c in commands] == [_W8A16], [c.type for c in commands]
        # The weight is a separate operand from the fp32 scales.
        assert len(commands[0].inputs) == 4, commands[0].inputs
        worst = _relative_error(got, expected)
        assert worst < 0.03, f"w8a16 differs by {worst} at {k}x{n}"


def test_the_at_spelling_reaches_the_same_gemv():
    """`x @ w` is `aten.matmul`, a target the annotation table has to name.

    The two spellings are different ops in the exported graph even though the
    multiply is the same one, and neither is the other's target: the annotation
    has to be placed on the matmul, because the dequantize it produces is what
    `to_edge`'s rewrite to `mm` carries into the emitter's pattern. Annotating
    only `mm` leaves the weight a plain fp16 tensor and the command the fp16
    `BATCH_MATMUL` -- a quantizer that silently did nothing.
    """
    for scheme, op, tolerance in (("q4a16", _Q4A16, 0.06), ("w8a16", _W8A16, 0.03)):
        for k, n in ((64, 32), (128, 96)):
            program, converted, x = _at_program(scheme, (1, k), k, n)
            # Both halves of the path are pinned: the converted graph is the
            # matmul with its dequantize, and the edge graph it lowers to is the
            # `mm` the emitter reads. Either one moving changes what this means.
            targets = {
                node.target
                for node in converted.graph.nodes
                if node.op == "call_function"
            }
            assert torch.ops.aten.matmul.default in targets, (scheme, k, n)
            assert targets == {
                torch.ops.aten.matmul.default,
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
            }
            mm = _node_of(program, exir_ops.edge.aten.mm.default)
            assert mm.args[1].target is hexagon_ops.DQ_PER_CHANNEL, (scheme, k, n)

            blob = HexagonBackend.preprocess(program, []).processed_bytes
            _, commands = read_blob(blob)
            assert [c.type for c in commands] == [op], (scheme, k, n)
            assert commands[0].params[1:3] == [k, n] and commands[0].params[8] == 1
            got = np.frombuffer(
                execute(blob, [x.half().numpy()])[0], dtype=np.float16
            ).astype(np.float32)
            with torch.no_grad():
                expected = converted(x).detach().float().numpy().reshape(-1)
            worst = _relative_error(got, expected)
            assert worst < tolerance, f"{scheme} differs by {worst} at {k}x{n}"


def test_a_batched_at_matmul_is_not_annotated():
    """The 2-D rule, from the annotator itself.

    `aten.matmul` is the batched spelling too, and the GEMV kernels have no
    batch axis: `to_edge` folds a batch into M, which is the dimension the
    emitter refuses. A batch of one is the sharp row -- flattened it is the
    M == 1 tile the GEMV path takes, so the rule has to be about the operands'
    ranks and not about the M they fold to. The 2-D row comes last, so the
    negative rows cannot pass by annotating nothing at all.
    """
    model = _At(64, 32).eval()
    for label, shape in (
        ("one batch axis", (2, 1, 64)),
        ("two batch axes", (2, 3, 64)),
        ("a batch of one", (1, 1, 64)),
        ("a 1-D activation", (64,)),
    ):
        assert _annotated(model, torch.randn(*shape)) == set(), label
    assert _annotated(model, torch.randn(1, 64)) == {torch.ops.aten.matmul.default}


def test_a_batched_at_matmul_keeps_the_fp16_batch_matmul():
    """The same rule where a reader sees it: the command stream.

    `_at_program` runs the real quantizer over the batched spelling, so this is
    the whole chain and not the predicate alone -- nothing annotated, no
    dequantize, and the fp16 BATCH_MATMUL the graph used before.
    """
    for scheme, op in (("q4a16", _Q4A16), ("w8a16", _W8A16)):
        for shape in ((2, 1, 64), (1, 1, 64)):
            program, _, _ = _at_program(scheme, shape, 64, 32)
            assert not [
                node
                for node in program.graph_module.graph.nodes
                if node.target is hexagon_ops.DQ_PER_CHANNEL
            ], (scheme, shape)
            blob = HexagonBackend.preprocess(program, []).processed_bytes
            _, commands = read_blob(blob)
            types = [c.type for c in commands]
            assert blob_interpreter.BATCH_MATMUL in types, (scheme, shape, types)
            assert op not in types, (scheme, shape, types)


def test_the_nn_linear_spelling_reaches_the_same_gemv():
    """`nn.Linear` is `aten.linear`, a target the annotation table does not hold.

    Its weight is stored `[out, in]`, so `to_edge` lowers it to an `addmm` over a
    `permute_copy` of the weight and the emitters' pattern -- a dequantize whose
    only reader is the matmul -- is one node further away than it is for `mm`.
    The quantizer therefore rewrites an admissible `linear` into the `addmm`
    spelling before the observers go in, over a `[k, n]` constant, and the entry
    it reaches is the one the `mm` path already reaches: one command, this
    geometry, one scale block per output channel, and the same answer. Without
    the rewrite the graph exports the fp16 model byte for byte and nothing at all
    is quantized -- a quantizer that silently does nothing, which is what this
    spelling used to be.
    """
    for module, multiply, edge_target, weight_arg in (
        (_Linear, torch.ops.aten.mm.default, exir_ops.edge.aten.mm.default, 1),
        (
            _BiasedLinear,
            torch.ops.aten.addmm.default,
            exir_ops.edge.aten.addmm.default,
            2,
        ),
    ):
        for scheme, op, tolerance in (
            ("q4a16", _Q4A16, 0.06),
            ("w8a16", _W8A16, 0.03),
        ):
            for k, n in ((64, 32), (128, 96)):
                program, converted, x = _linear_program(scheme, 1, k, n, module)
                # Both halves of the path are pinned: the converted graph is the
                # matmul with its dequantize and no permute left, and the edge
                # graph has the dequantize on the matmul's weight operand, which
                # is the pattern the emitter matches. A biased `nn.Linear` leaves
                # the bias in the kernel's own bias operand, so the result is one
                # command rather than a matmul and an add.
                targets = {
                    node.target
                    for node in converted.graph.nodes
                    if node.op == "call_function"
                }
                assert multiply in targets, (scheme, k, n, targets)
                assert targets == {
                    multiply,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default,
                }, (scheme, k, n, targets)
                matmul = _node_of(program, edge_target)
                assert (
                    matmul.args[weight_arg].target is hexagon_ops.DQ_PER_CHANNEL
                ), (scheme, k, n)

                blob = HexagonBackend.preprocess(program, []).processed_bytes
                _, commands = read_blob(blob)
                assert [c.type for c in commands] == [op], (scheme, k, n)
                assert commands[0].params[1:3] == [k, n]
                assert commands[0].params[8] == 1
                got = np.frombuffer(
                    execute(blob, [x.half().numpy()])[0], dtype=np.float16
                ).astype(np.float32)
                with torch.no_grad():
                    expected = converted(x).detach().float().numpy().reshape(-1)
                worst = _relative_error(got, expected)
                assert worst < tolerance, f"{scheme} differs by {worst} at {k}x{n}"


def test_an_nn_linear_the_kernels_would_refuse_stays_portable():
    """The same gate, reached from the spelling a model is written in.

    A classifier's head is `nn.Linear(k, 1000)`, and 1000 is not a multiple of
    the 32-channel tile: there is no kernel that reads it, so the `nn.Linear`
    keeps the permuted weight and the fp16 path it already had rather than be
    rewritten into a matmul the emitters would refuse. The rewrite is gated with
    the annotation and not separately from it, which is what the converted
    target says: it is still `linear`.
    """
    for k in (64, 128):
        converted, x = _converted("q4a16", 1, k, 8, module=_Linear)
        assert torch.ops.aten.linear.default in {
            node.target for node in converted.graph.nodes
        }, k
        assert not _carries_a_dequantize(converted), k
        assert _Q4A16 not in _command_types(converted, x), k
        assert _serializes(converted, x) > 0, k
    # The same projection at a shape the tile divides, so this is not what a
    # quantizer that ignores `nn.Linear` entirely says.
    converted, x = _converted("q4a16", 1, 64, 32, module=_Linear)
    assert _carries_a_dequantize(converted)
    assert _Q4A16 in _command_types(converted, x)



def _live_operand_program(scheme, spelling, k, n):
    """The whole chain for a matmul whose second operand is a run-time tensor.

    Both operands are method inputs, so there is no weight to recognise: this is
    what the annotator used to read as a weight when the caller handed the value
    in.
    """
    model = _Live(spelling).eval()
    x = torch.randn(1, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    converted, inputs = _quantize(scheme, model, x, b)
    lowered = to_edge_transform_and_lower(
        torch.export.export(converted, inputs),
        partitioner=[HexagonPartitioner()],
    )
    return lowered, converted, x, b


def test_a_matmul_over_two_run_time_tensors_stays_portable():
    """A scores-like matmul is left alone, and it still has to export.

    `a @ b` and `mm(a, b)` over two live tensors put a run-time value where a
    weight would be, and there is nothing there to quantize: the export cannot
    read its bytes, and PT2E quantizes what it cannot freeze *at run time*
    instead -- which leaves a `quantize_per_channel` beside the
    `dequantize_per_channel` rather than a stored low-bit tensor. Neither has an
    out variant, so a graph carrying them cannot be serialized at all. This test
    used to assert the annotation and stop at the lowered program, where that is
    invisible; `_serializes` is the same graph asked for a `.pte`.

    So the refusal moved to the annotation: the matmul keeps the fp16 path it
    had, the graph holds no quantized op, and the program serializes. The same
    shape with a weight in that position does reach the kernel, so this cannot
    pass by way of an annotator that refuses everything.
    """
    k, n = 64, 32
    for spelling, multiply in (
        ("at", torch.ops.aten.matmul.default),
        ("mm", torch.ops.aten.mm.default),
    ):
        lowered, converted, x, b = _live_operand_program("q4a16", spelling, k, n)
        # The matmul is still there in the spelling it was written in, so what
        # this says below is about an operand the export does not own and not
        # about a node that was rewritten into something else.
        converted_targets = {node.target for node in converted.graph.nodes}
        assert multiply in converted_targets, (spelling, converted_targets)
        assert not _carries_a_dequantize(converted), spelling
        assert torch.ops.quantized_decomposed.quantize_per_channel.default not in (
            converted_targets
        ), (spelling, converted_targets)
        # What comes out is the fp16 kernel the graph already had: the live
        # operand reaches the DSP as an ordinary tensor, and neither quantized
        # entry appears.
        types = [
            command.type
            for command in read_blob(
                HexagonBackend.preprocess(
                    _edge_program(converted, x, b), []
                ).processed_bytes
            )[1]
        ]
        assert blob_interpreter.BATCH_MATMUL in types, (spelling, types)
        for op in (_Q4A16, _W8A16, _PREFILL):
            assert op not in types, (spelling, types)
        assert len(bytes(lowered.to_executorch().buffer)) > 0, spelling
    # The same shape with a weight in that position does reach the kernel.
    program, _, _ = _quantized_program("q4a16", 1, k, n)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, commands = read_blob(blob)
    assert [c.type for c in commands] == [_Q4A16]


def test_q4a16_weight_bytes_are_the_kernel_layout():
    """Pins the tile byte order without going through the unpacker.

    Tile (y, x) holds the 32 output channels against 32 contraction values;
    group g covers k = x*32 + 4g + {0,1,2,3}, and byte g*64 + ocIn*2 + p packs
    k = x*32 + 4g + 2p in its low nibble and +1 in its high nibble, offset by 8.
    """
    k, n = 64, 32
    weight = (torch.arange(k * n).reshape(k, n) % 16 - 8).numpy()
    packed = hexagon_ops.pack_q4a16_gemv_weight(weight, np.ones(n), k, n)
    assert len(packed) == (k // 32) * (n // 32) * 512 + n * 4

    def nibble(value):
        return (int(value) + 8) & 0x0F

    # Tile (0, 0), group 0, output channel 0.
    assert packed[0] == nibble(weight[0, 0]) | (nibble(weight[1, 0]) << 4)
    assert packed[1] == nibble(weight[2, 0]) | (nibble(weight[3, 0]) << 4)
    # Tile (0, 0), group 1, output channel 3 sits at 1*64 + 3*2.
    at = 64 + 3 * 2
    assert packed[at] == nibble(weight[4, 3]) | (nibble(weight[5, 3]) << 4)
    # The scales follow the single tile.
    scales = np.frombuffer(packed[(k // 32) * (n // 32) * 512 :], dtype=np.float32)
    assert np.array_equal(scales, np.ones(n, dtype=np.float32))

    unpacked = blob_interpreter._unpack_vrmpy_int4(packed, k, n)
    assert np.array_equal(unpacked, weight.T.astype(np.int32))


def test_w8a16_weight_bytes_are_the_kernel_layout():
    """Pins the HMX int8 tile order, whose four k values are pair-interleaved."""
    k, n = 64, 32
    weight = ((torch.arange(k * n).reshape(k, n) % 255) - 127).numpy()
    packed = hexagon_ops.pack_w8a16_gemv_weight(weight, k, n)
    assert len(packed) == (k // 32) * (n // 32) * 1024

    perm = (0, 2, 1, 3)
    # Tile (0, 0), group 0, output channel 0.
    for p in range(4):
        assert packed[p] == (int(weight[perm[p], 0]) & 0xFF)
    # Tile (0, 0), group 1, output channel 5 sits at 1*128 + 5*4.
    at = 128 + 5 * 4
    for p in range(4):
        assert packed[at + p] == (int(weight[4 + perm[p], 5]) & 0xFF)

    unpacked = blob_interpreter._unpack_hmx_int8(packed, k, n)
    assert np.array_equal(unpacked, weight.T.astype(np.int32))


def test_the_quantized_matmul_is_delegated():
    program, _, _ = _quantized_program("q4a16", 1, 64, 128)
    support = _support(program)
    mm = _mm_node(program)
    assert support.is_node_supported(None, mm)
    assert support.is_node_supported(None, mm.args[1])


def test_a_prefill_quantized_matmul_is_delegated():
    """M > 1 is the prefill entry, and the dequantize feeding it is fused.

    The two M ranges are two different kernels with two different weight
    layouts, and which one a matmul goes to is the M it carries. This is the
    support half; `test_the_prefill_command_carries_its_shape_and_chunks` below
    reads the command that comes out.
    """
    program, _, _ = _quantized_program("q4a16", 8, 64, 128)
    support = _support(program)
    mm = _mm_node(program)
    assert support.is_node_supported(None, mm)
    assert support.is_node_supported(None, mm.args[1])


def test_a_prefill_shape_the_kernels_would_refuse_stays_portable():
    """The same two guards, asked of the M > 1 entry, and asked before the fact.

    K % 64 and N % 32 gate it for the prefill kernel's own reasons -- a K that
    is not a multiple of 64 packs into half a 64-element activation block, and N
    lands in 32-channel tiles -- and neither is the GEMV path's rule restated:
    the prefill entry would otherwise take shapes the GEMV entries refuse and
    the other way round. The quantizer asks the same question of the operands
    before it annotates anything, because a matmul it annotated and the emitter
    then refused leaves its dequantize in the graph, and `to_executorch` cannot
    serialize a graph carrying one: the refusal has to happen before the
    dequantize exists or the caller gets no program at all.
    """
    for k, n in ((96, 64), (64, 48), (128, 16)):
        converted, x = _converted("q4a16", 4, k, n)
        assert not _carries_a_dequantize(converted), (k, n)
        types = _command_types(converted, x)
        assert _PREFILL not in types and _Q4A16 not in types, (k, n, types)
        assert _serializes(converted, x) > 0, (k, n)
    # A row group that does divide, at the same M: without this the loop above
    # is also what a quantizer that refuses everything says.
    converted, x = _converted("q4a16", 4, 128, 64)
    assert _carries_a_dequantize(converted)
    assert _PREFILL in _command_types(converted, x)


def test_a_prefill_past_the_m_le_32_k_ceiling_stays_portable():
    """A K the M <= 32 prefill kernel cannot carry has to stay portable.

    The dispatcher sends `M <= 32` down a kernel that holds one 32-byte
    activation descriptor per K/32 tile in its own stack frame, and that frame
    stops fitting somewhere around the teens of KB -- see PREFILL_M32_MAX_K for
    the measurement, and for how much of that attribution is measured as against
    inferred. Emitting past it is worse than refusing: the failure is
    `execute_command_group failed: 0x8000040d` with no output at all, not a wrong
    number a caller could notice.

    The geometries are the measured ones, and the two assertions after the loop
    are what keep this from being a test of "some wide shape is refused": the
    same K is admitted one M over the dispatch, and the same M is admitted at the
    last K measured to work.
    """
    for m, k in ((4, 12736), (4, 12800), (2, 12800), (32, 12800)):
        converted, x = _converted("q4a16", m, k, 64)
        assert not _carries_a_dequantize(converted), (m, k)
        assert _PREFILL not in _command_types(converted, x), (m, k)
        assert _serializes(converted, x) > 0, (m, k)
    # One M over the dispatch, same K: that kernel heap-allocates the
    # descriptors, so nothing about this K is a problem for it.
    converted, x = _converted("q4a16", hexagon_ops.PREFILL_M32_MAX_M + 1, 12800, 64)
    assert _carries_a_dequantize(converted)
    assert _PREFILL in _command_types(converted, x)
    # And the widest K that does fit the same branch.
    converted, x = _converted("q4a16", 4, hexagon_ops.PREFILL_M32_MAX_K, 64)
    assert _carries_a_dequantize(converted)


def test_the_m_le_32_k_ceiling_is_the_thing_that_refuses(monkeypatch):
    """Move the ceiling and the annotation follows it, at two different ceilings.

    The control for the test above. A K one tile past the ceiling is refused
    whatever the constant says only while the guard reads it, so taking the guard
    out turns this red rather than leaving it green -- which is the difference
    between testing the shape and testing the rule. The quantizer reads that
    constant through the same helper the emitters do, so what moves here is
    whether the annotation is made at all. The second ceiling is not 12672, so a
    constant that agreed with one hard-coded number by accident would not pass
    here.
    """
    for ceiling in (hexagon_ops.PREFILL_M32_MAX_K, 13056):
        monkeypatch.setattr(hexagon_ops, "PREFILL_M32_MAX_K", ceiling)
        converted, x = _converted("q4a16", 4, ceiling, 64)
        assert _carries_a_dequantize(converted), ceiling
        converted, x = _converted("q4a16", 4, ceiling + 64, 64)
        assert not _carries_a_dequantize(converted), ceiling


def test_the_m_over_32_prefill_still_reaches_its_kernel_at_the_vtcm_ceiling():
    """The K bound belongs to one branch, so the other keeps its widest shape.

    The top of the VTCM budget is where the M > 32 entry used to stop, and it has
    to keep being emitted with that geometry in it rather than be closed along
    with the M <= 32 bound. Asserting the command's parameters rather than the
    support verdict is what says the prefill entry is still the one that comes
    out, carrying this M, K and N.
    """
    m, k, n = 40, 25216, 64
    program, _, _ = _quantized_program("q4a16", m, k, n)
    assert _support(program).is_node_supported(None, _mm_node(program))
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, commands = read_blob(blob)
    assert [c.type for c in commands] == [_BLIT, _PREFILL, _BLIT]
    assert list(commands[1].params[:10]) == [m, k, n, 0, 1, 1, 2, k // 32, 1, 0]


def test_a_w8a16_prefill_matmul_stays_portable():
    """Only the q4a16 prefill entry is wired, and the difference is the weight.

    `DSP_OP_MATMUL_W8A16_BLOCK_FP16` reads int8 in an order nothing here packs
    and takes its geometry in an im2col struct, so a w8a16 matmul with M > 1
    stays on the portable kernels: an int8 weight in the int4 tile order would
    be read, multiplied and answered, and only the answer would be wrong. The
    quantizer leaves it unannotated for the same reason an int4 one is
    annotated: the alternative is a dequantize the emitters refuse, which is not
    a fallback but a program that cannot be serialized.
    """
    converted, x = _converted("w8a16", 8, 64, 128)
    assert not _carries_a_dequantize(converted)
    types = _command_types(converted, x)
    assert _W8A16 not in types and _PREFILL not in types, types
    assert _serializes(converted, x) > 0
    # The same shape one weight type over does reach the kernel, so what this
    # turns on is the int8 weight and not the M it carries.
    converted, x = _converted("q4a16", 8, 64, 128)
    assert _carries_a_dequantize(converted)
    assert _PREFILL in _command_types(converted, x)


def test_a_prefill_over_two_run_time_tensors_is_left_alone():
    """The run-time-weight rule above M == 1, where a different kernel decides.

    An `a @ b` over two live tensors at eight rows used to be annotated and
    refused downstream like the M == 1 spelling, and it failed the same way one
    step later: the prefill emitter packs the weight into its tile order, so a
    tensor the export cannot read is a refusal at the emitter, and the stranded
    dequantize made the program unserializable. The refusal is now the
    annotation's, and it has to be drawn for `M > 1` too: the entry that answers
    above one row is a different kernel from the two the M == 1 rule was written
    for, and nothing about sharing the shape predicate with those makes it the
    same question.
    """
    k, n = 64, 128
    x = torch.randn(8, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    converted, _ = _quantize("q4a16", _Live("at").eval(), x, b)
    assert not _carries_a_dequantize(converted)
    types = [
        command.type
        for command in read_blob(
            HexagonBackend.preprocess(_edge_program(converted, x, b), []).processed_bytes
        )[1]
    ]
    assert blob_interpreter.BATCH_MATMUL in types, types
    assert _PREFILL not in types, types
    assert _serializes(converted, x, b) > 0
    # The same shape with a weight in that position does reach the prefill entry,
    # so what refuses this is the run-time weight and not the eight rows.
    program, _, _ = _quantized_program("q4a16", 8, k, n)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, commands = read_blob(blob)
    # K == 64 needs no activation pack, so this is the kernel and the repack of
    # its 64-channel output packs.
    assert [c.type for c in commands] == [_PREFILL, _BLIT]


def test_an_annotated_shape_is_a_shape_that_serializes_as_quantized():
    """The two symptoms in one table: no shape fails to export, none lies.

    Each row is the whole chain twice, over `mm` and over the `nn.Linear` the
    quantizer rewrites into it, and the claim is a biconditional: a shape the
    quantizer annotated is a shape whose command stream carries a quantized
    entry, and a shape it left alone is a shape whose command stream does not.
    Either half failing alone is a different bug -- an annotation with no command
    is the silent no-op, a command with no annotation is not something the
    emitter can produce -- and the rows are the shapes the kernels' guards
    divide, so `weight_only_matmul_fits` above them is the same answer asked a
    second way. Every row has to serialize: that is the symptom the gate exists
    for, and `to_executorch` cannot serialize a graph carrying a dequantize
    nothing runs.
    """
    rows = (
        ("q4a16", 1, 64, 32),
        ("w8a16", 1, 64, 32),
        ("q4a16", 2, 64, 32),
        ("w8a16", 2, 64, 32),
        ("q4a16", 1, 32, 32),
        ("q4a16", 1, 64, 16),
        ("q4a16", 4, 128, 64),
        ("q4a16", 2, 64, 128),
    )
    for module in (_Mm, _BiasedLinear):
        for scheme, m, k, n in rows:
            converted, x = _converted(scheme, m, k, n, module=module)
            annotated = _carries_a_dequantize(converted)
            assert annotated == hexagon_ops.weight_only_matmul_fits(
                m, k, n, SUPPORTED_SCHEMES[scheme]
            ), (module.__name__, scheme, m, k, n)
            types = _command_types(converted, x)
            quantized = [
                command for command in types if command in (_Q4A16, _W8A16, _PREFILL)
            ]
            assert len(quantized) == (1 if annotated else 0), (
                module.__name__,
                scheme,
                m,
                k,
                n,
                types,
            )
            assert _serializes(converted, x) > 0, (module.__name__, scheme, m, k, n)


def test_a_batched_quantized_matmul_keeps_the_batch_matmul():
    """A batch axis in front of M stays the fp16 BATCH_MATMUL the graph had.

    `to_edge` folds the batch into rows, so a batch of four of eight rows is a
    matmul with 32 rows of activation and one weight -- which is exactly the
    shape the prefill entry reads as prefill rows, and the one a predicate that
    looked at M alone would take. The command stream is where that shows, so
    this asks it there rather than of the predicate: the batch matmul the graph
    already had, and neither quantized entry.
    """
    for scheme in ("q4a16", "w8a16"):
        for shape in ((2, 8, 64), (1, 8, 64)):
            program, _, _ = _at_program(scheme, shape, 64, 32)
            blob = HexagonBackend.preprocess(program, []).processed_bytes
            _, commands = read_blob(blob)
            types = [c.type for c in commands]
            assert blob_interpreter.BATCH_MATMUL in types, (scheme, shape, types)
            for op in (_PREFILL, _Q4A16, _W8A16):
                assert op not in types, (scheme, shape, types)


def test_a_shape_the_kernels_would_refuse_stays_portable():
    """K % 64 and N % 32 are the kernels' own guards, so they gate the annotation.

    Both entries check those multiples and return an error rather than compute
    something wrong, but a delegated node cannot fall back, so the check has to
    happen before the node is annotated: 96 is below no power of two, it just is
    not a multiple of 64, and 48 is not a multiple of the 32-channel tile. What
    the caller gets is the fp16 matmul the graph already had, and a program that
    serializes -- which a graph with a refused annotation's dequantize in it does
    not, no matter which kernels a runner links.
    """
    for k, n in ((96, 64), (64, 48)):
        converted, x = _converted("q4a16", 1, k, n)
        assert not _carries_a_dequantize(converted), (k, n)
        types = _command_types(converted, x)
        for op in (_Q4A16, _W8A16, _PREFILL):
            assert op not in types, (k, n, types)
        assert _serializes(converted, x) > 0, (k, n)
    # The same matmul one multiple over, so this is not what a quantizer that
    # annotates nothing at all says.
    converted, x = _converted("q4a16", 1, 64, 32)
    assert _carries_a_dequantize(converted)
    assert _Q4A16 in _command_types(converted, x)


def test_the_gate_is_what_keeps_a_refused_shape_serializable(monkeypatch):
    """Open the gate and the same shape fails the way every refused shape did.

    `Missing out variants` is `to_executorch`'s and it is raised for any
    `quantized_decomposed` node left in the portable part of the graph -- there
    is no out variant to convert one to, whichever kernels a runner links. That
    is the whole reason the quantizer asks the emitters' question before it
    annotates, and this is the control for every test above that says a refused
    shape "stays portable": with the gate forced open the same shape annotates,
    its dequantize is refused downstream and stranded, and the program cannot be
    serialized at all. `N % 32` is the shape a classifier's `nn.Linear(k, 1000)`
    head has, which is where this was first seen.
    """
    monkeypatch.setattr(
        hexagon_quantizer, "_admits_the_emitters", lambda *args, **kwargs: True
    )
    converted, x = _converted("q4a16", 1, 64, 8)
    assert _carries_a_dequantize(converted)
    with pytest.raises(RuntimeError, match="Missing out variants"):
        _serializes(converted, x)


def test_the_op_ids_are_the_ones_the_dsp_defines():
    """The two numbers have to be the DSP's, and the DSP pins them itself.

    htp_command.h carries a static_assert per op, so this reads that file and
    compares against it: the pair being transposed would otherwise only show up
    as a kernel running with the other one's operand order, on a device.
    """
    header = (
        pathlib.Path(hexagon_ops.__file__).resolve().parent
        / "third-party"
        / "mnn-htp-ops"
        / "include"
        / "htp_command.h"
    ).read_text()
    for name in ("DSP_OP_MATMUL_Q4A16_GEMV_I8", "DSP_OP_MATMUL_W8A16_GEMV_I8"):
        value = getattr(hexagon_ops, name)
        assert re.search(rf"\b{name}\s*=\s*{value}\b", header), name


def _stored_floats(blob: bytes, ref, count: int) -> np.ndarray:
    """The last `count` fp32 of a weight operand, as the arena holds them."""
    header, _ = read_blob(blob)
    arena = blob_interpreter.Arena(header, blob, "blob")
    return np.frombuffer(bytes(arena.view(ref)), dtype=np.float32)[-count:]


def test_the_command_carries_one_scale_block_per_output_channel():
    """Pins params[1..2] and params[8], which the dispatch reads.

    params[8] is the scale block count. 1 means one block spanning all of K,
    which is what per-channel quantization is -- the kernel derives
    blocksize = K/nblk and walks every k-tile of the output channel under that
    one entry -- so the scale operand is exactly the observer's per-channel
    list, in output-channel order. The same operand check is what says the
    granularity is the quantizer's and not something else the kernel could
    have been handed.
    """
    for scheme, op in (("q4a16", _Q4A16), ("w8a16", _W8A16)):
        k, n = 128, 96
        program, _, _ = _quantized_program(scheme, 1, k, n)
        blob = HexagonBackend.preprocess(program, []).processed_bytes
        header, commands = read_blob(blob)
        assert [c.type for c in commands] == [op], scheme
        params = commands[0].params
        assert params[1] == k and params[2] == n
        assert params[8] == 1
        assert params[9] == 0, "the scale operand carries no qbias"

        # q4a16 appends the scales to the tiles (the kernel's b_scale pointer is
        # weight + icP*ocP*512); w8a16 takes them as an operand of their own.
        scale_ref = commands[0].inputs[2 if scheme == "w8a16" else 1]
        stored = _stored_floats(blob, scale_ref, n)
        assert scale_ref.size >= n * 4

        # The value the observer produced, read the way the backend reads it:
        # the placeholder carries a fake in the edge program's metadata, and the
        # real tensor is the buffer the graph module holds.
        dequantize = _node_of(program, hexagon_ops.DQ_PER_CHANNEL)
        expected = owned_weight(program, dequantize.args[1])
        assert expected is not None and expected.numel() == n
        expected = expected.detach().to(torch.float32).reshape(-1).numpy()
        assert np.array_equal(stored, expected), f"{scheme} scales differ"


def test_the_prefill_command_carries_its_shape_and_chunks():
    """The M > 1 entry: three commands, and the numbers the kernel walks.

    params carry M, K and N, one activation band and two output-channel tiles,
    K/32 as the tile count the kernel validates, and 1 scale block per output
    channel. The weight operand is the tiles plus one fp16 scale per channel,
    which is the length the kernel's own `b_scale` pointer assumes -- an fp32
    list here would make the kernel read the scales as weight tiles.
    """
    m, k, n = 8, 128, 96
    program, converted, x = _quantized_program("q4a16", m, k, n)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    header, commands = read_blob(blob)
    assert [c.type for c in commands] == [_BLIT, _PREFILL, _BLIT]
    params = commands[1].params
    assert list(params[:10]) == [m, k, n, 0, 1, 1, 2, k // 32, 1, 0]
    assert commands[1].inputs[1].size == (k // 32) * (n // 32) * 512 + n * 2

    # The activation pack: one region, the 64-k block outermost, whose source
    # stride is the 64 elements that block covers and whose destination stride
    # is one row of the blocked layout.
    pack = commands[0].params
    assert pack[0] == 1 and pack[1] == 2 and pack[2] == 1
    assert pack[3:15] == [0, 0, 0, k // 64, m, 64, 64, k, 1, m * 64, 64, 1]
    # The repack: the kernel's 64-channel packs back to rows, with the ragged
    # last pack as a second region.
    unpack = commands[2].params
    assert unpack[0] == 2
    assert unpack[3:15] == [0, 0, 0, n // 64, m, 64, m * 64, 64, 1, 64, n, 1]
    assert unpack[15:27] == [0, m * 64, 64, 1, m, n % 64, 1, 64, 1, 1, n, 1]

    got = np.frombuffer(execute(blob, [x.half().numpy()])[0], dtype=np.float16).reshape(
        m, n
    )
    with torch.no_grad():
        expected = converted(x).detach().float().numpy().reshape(m, n)
    worst = _relative_error(got.astype(np.float32), expected)
    assert worst < 0.02, f"the prefill command differs by {worst} at {m}x{k}x{n}"


def test_a_one_block_activation_needs_no_pack_blit():
    """K == 64: the blocked layout and the row-major one are the same memory.

    The pack blit is skipped rather than emitted with an identity region, and
    the kernel reads the graph's own tensor: a copy that moved nothing would
    still cost a command, and the reason it can be skipped is a property of the
    layout rather than of the emitter.
    """
    program, _, _ = _quantized_program("q4a16", 8, 64, 64)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, commands = read_blob(blob)
    assert [c.type for c in commands] == [_PREFILL, _BLIT]
    assert commands[0].inputs[0].space == commands[0].inputs[0].space
    assert commands[0].inputs[0].size == 8 * 64 * 2, "the activation is not the input"


def test_the_two_branches_are_two_different_commands():
    """M decides which kernel, and the two do not overlap.

    The control for everything above: if the emitter sent M > 1 to the GEMV
    entry, or M == 1 to the prefill one, the weight layout and the parameters
    would be the other kernel's and the shapes below would still look plausible.
    """
    for m, expected in ((1, [_Q4A16]), (2, [_PREFILL, _BLIT]), (64, [_PREFILL, _BLIT])):
        program, _, _ = _quantized_program("q4a16", m, 64, 64)
        blob = HexagonBackend.preprocess(program, []).processed_bytes
        _, commands = read_blob(blob)
        assert [c.type for c in commands] == expected, (m, [c.type for c in commands])


def test_the_dequantize_never_materializes_the_weight_twice():
    """The blob holds the packed weight and the scales, and nothing else.

    The stored low-bit weight, its fp32 scales and its int64 zero point all
    reach `preprocess` as program state, but only what a command reads belongs
    in the file: the packing replaces the weight, and the zero point is a
    constant the kernel has no operand for.
    """
    k, n = 64, 32
    for scheme, tile_bytes in (("q4a16", 512), ("w8a16", 1024)):
        program, _, _ = _quantized_program(scheme, 1, k, n)
        blob = HexagonBackend.preprocess(program, []).processed_bytes
        header, _ = read_blob(blob)
        assert header.weights_bytes == (k // 32) * (n // 32) * tile_bytes + n * 4


def test_the_lowered_program_calls_one_delegate_holding_the_gemv_command():
    """The positive assertion: a delegate call node, whose payload is the command.

    A green partitioner test is not the same as a delegated op -- a graph can be
    tagged and never reach a delegate -- so this lowers the program the way a
    caller does and reads the command out of the delegate's own bytes.
    """
    k, n = 64, 32
    converted, x = _converted("q4a16", 1, k, n)
    lowered = to_edge_transform_and_lower(
        torch.export.export(converted, (x,)),
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    graph = lowered.graph_module.graph
    calls = [
        node
        for node in graph.nodes
        if str(node.target).endswith("executorch_call_delegate")
    ]
    assert len(calls) == 1, [str(node.target) for node in graph.nodes]

    delegates = [
        module
        for module in lowered.graph_module.modules()
        if isinstance(module, LoweredBackendModule)
    ]
    assert len(delegates) == 1
    assert delegates[0].backend_id == "HexagonBackend"

    header, commands = read_blob(delegates[0].processed_bytes)
    assert [c.type for c in commands] == [_Q4A16]
    assert commands[0].params[1:3] == [k, n] and commands[0].params[8] == 1
    assert header.n_inputs == 1 and header.n_outputs == 1

    got = np.frombuffer(
        execute(delegates[0].processed_bytes, [x.half().numpy()])[0], dtype=np.float16
    ).astype(np.float32)
    with torch.no_grad():
        expected = converted(x).detach().float().numpy().reshape(-1)
    assert _relative_error(got, expected) < 0.03


def test_the_quantized_graph_reaches_the_partitioner():
    """The whole path, partitioner included: one delegate, one command.

    `preprocess` is not what decides whether a node is delegated, so a test that
    only calls it would pass with a partitioner that refuses the graph. This runs
    the real partitioner first and then the real preprocess on what it tags.
    """
    program, _, _ = _quantized_program("q4a16", 1, 64, 128)
    result = HexagonPartitioner()(program)
    assert list(result.partition_tags) == ["hexagon_1"], result.partition_tags
    delegated = [
        node
        for node in result.tagged_exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert [node.meta.get("delegation_tag") for node in delegated] == [
        "hexagon_1"
    ] * len(delegated)
    assert len(delegated) == 2, [node.name for node in delegated]
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, commands = read_blob(blob)
    assert [c.type for c in commands] == [_Q4A16]


def test_a_bias_that_is_not_one_value_per_channel_stays_portable():
    """The kernel adds n contiguous halfs, so only that bias has a command.

    A one-element bias is legal in the graph -- torch broadcasts it against the
    result -- and would have the kernel read n values out of a two-byte operand.
    Neither the shape check nor the emitter's broadcast machinery is what
    decides this one; it is the kernel's own reading of its bias operand, which
    is also why the quantizer asks it before annotating: the addmm it would
    otherwise have annotated is refused downstream, and its dequantize then
    strands a program that cannot be serialized.
    """
    for bias_shape in ((1, 1), (1,)):
        program, converted, x = _quantized_addmm_program("q4a16", 64, 32, bias_shape)
        assert not _carries_a_dequantize(converted), bias_shape
        types = [
            command.type
            for command in read_blob(
                HexagonBackend.preprocess(program, []).processed_bytes
            )[1]
        ]
        assert _Q4A16 not in types, (bias_shape, types)
        assert _serializes(converted, x) > 0, bias_shape
    for bias_shape in ((32,), (1, 32)):
        program, converted, x = _quantized_addmm_program("q4a16", 64, 32, bias_shape)
        assert _carries_a_dequantize(converted), bias_shape
        assert _Q4A16 in _command_types(converted, x), bias_shape


def test_a_quantized_addmm_puts_the_bias_in_the_kernel():
    """One command, whose bias operand is the graph's own bias.

    The plain addmm needs a second command for the sum; the quantized kernel
    takes the bias itself, so the result is written once.
    """
    k, n = 64, 32
    program, converted, x = _quantized_addmm_program("q4a16", k, n, (n,))
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, commands = read_blob(blob)
    assert [c.type for c in commands] == [_Q4A16]
    assert commands[0].inputs[2].space == blob_interpreter.B.TensorSpace.WEIGHTS
    got = np.frombuffer(execute(blob, [x.half().numpy()])[0], dtype=np.float16).astype(
        np.float32
    )
    with torch.no_grad():
        expected = converted(x).detach().float().numpy().reshape(-1)
    assert _relative_error(got, expected) < 0.06
