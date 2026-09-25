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


def _live_operand_program(scheme, spelling, k, n):
    """The whole chain for a matmul whose second operand is a run-time tensor.

    Both operands are method inputs, so there is no weight to recognise: this is
    what the annotator reads as a weight when the caller hands the value in.
    """
    model = _Live(spelling).eval()
    x = torch.randn(1, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    converted, inputs = _quantize(scheme, model, x, b)
    lowered = to_edge_transform_and_lower(
        torch.export.export(converted, inputs),
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    return lowered, converted, x, b


def test_a_matmul_over_two_run_time_tensors_stays_portable():
    """A scores-like matmul is annotated, and it still has to export.

    `annotate` reads the operand's position rather than its provenance, so
    `a @ b` and `mm(a, b)` over two live tensors get a per-channel observer for
    something that is not a weight at all. The emitter needs those bytes in hand
    and cannot have them, and which side says so is the whole point: the
    partitioner refuses the chain, so the caller is left with portable kernels
    rather than an export that fails part way through. The dequantize goes back
    with it -- one delegated on its own would be a partition whose output
    nothing ever wrote.
    """
    k, n = 64, 32
    for spelling, multiply in (
        ("at", torch.ops.aten.matmul.default),
        ("mm", torch.ops.aten.mm.default),
    ):
        program, converted, _, _ = _live_operand_program("q4a16", spelling, k, n)
        # The annotator did its part, so what refuses this is the partitioner
        # and not an annotation that never happened.
        converted_targets = {node.target for node in converted.graph.nodes}
        assert multiply in converted_targets, (spelling, converted_targets)
        assert (
            torch.ops.quantized_decomposed.dequantize_per_channel.default
            in converted_targets
        ), (spelling, converted_targets)
        graph = program.graph_module.graph
        assert not [
            node
            for node in graph.nodes
            if str(node.target).endswith("executorch_call_delegate")
        ], (spelling, [node.name for node in graph.nodes])
        assert exir_ops.edge.aten.mm.default in {
            node.target for node in graph.nodes if node.op == "call_function"
        }, spelling
    # The same shape with a weight in that position does reach the kernel, so
    # this cannot pass by way of a partitioner that refuses everything.
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


def test_w8a16_prefill_weight_appends_fp16_scales():
    k, n = 64, 32
    weight = ((torch.arange(k * n).reshape(k, n) % 17) - 8).numpy()
    scale = np.linspace(0.125, 2.0, n, dtype=np.float32)
    packed = hexagon_ops.pack_w8a16_prefill_weight(weight, scale, k, n)
    tiles = hexagon_ops.pack_w8a16_gemv_weight(weight, k, n)
    assert packed[: len(tiles)] == tiles
    tail = np.frombuffer(packed[len(tiles) :], dtype=np.float16)
    assert np.array_equal(tail, scale.astype(np.float16))


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
    """The same two guards, asked of the M > 1 entry.

    K % 64 and N % 32 gate it for the prefill kernel's own reasons -- a K that
    is not a multiple of 64 packs into half a 64-element activation block, and N
    lands in 32-channel tiles -- and neither is the GEMV path's rule restated:
    the prefill entry would otherwise take shapes the GEMV entries refuse and
    the other way round.
    """
    for k, n in ((96, 64), (64, 48), (128, 16)):
        program, _, _ = _quantized_program("q4a16", 4, k, n)
        support = _support(program)
        mm = _mm_node(program)
        assert not support.is_node_supported(None, mm), (k, n)
        assert not support.is_node_supported(None, mm.args[1]), (k, n)
    # A row group that does divide, at the same M: without this the loop above
    # is also what a support object that refuses every quantized matmul says.
    program, _, _ = _quantized_program("q4a16", 4, 128, 64)
    support = _support(program)
    mm = _mm_node(program)
    assert support.is_node_supported(None, mm)
    assert support.is_node_supported(None, mm.args[1])


def test_a_w8a16_prefill_matmul_lowers_to_command_42():
    """A real partitioned int8 graph contains command 42 in its delegate blob."""
    converted, x = _converted("w8a16", 2, 64, 128)
    lowered = to_edge_transform_and_lower(
        torch.export.export(converted, (x,)),
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    delegates = [
        module
        for module in lowered.graph_module.modules()
        if isinstance(module, LoweredBackendModule)
    ]
    assert len(delegates) == 1
    blob = delegates[0].processed_bytes
    header, commands = read_blob(blob)
    assert 42 in [command.type for command in commands]
    command = next(command for command in commands if command.type == 42)
    assert len(command.params) == 29
    tile_bytes = (64 // 32) * (128 // 32) * 1024
    assert command.inputs[1].size == tile_bytes + 128 * 2
    packed_weight = bytes(
        blob_interpreter.Arena(header, blob, "test").view(command.inputs[1])
    )
    expected_scale = converted.state_dict()["_scale_0"].detach().float().numpy()
    actual_scale = np.frombuffer(packed_weight[tile_bytes:], dtype=np.float16)
    assert np.array_equal(
        actual_scale.astype(np.float32), expected_scale.astype(np.float16).astype(np.float32)
    )

    # The same geometry with int4 still uses command 22, so the assertion above
    # is specifically about the int8 prefill path.
    program, _, _ = _quantized_program("q4a16", 2, 64, 128)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, q4_commands = read_blob(blob)
    assert 22 in [command.type for command in q4_commands]


def test_w8a16_prefill_bracket_m32_m33_k64_k128_and_refuse_k_tail():
    """The admitted W8 geometry is explicit at the M and K boundaries."""
    for m in (32, 33):
        for k in (64, 128):
            program, _, _ = _quantized_program("w8a16", m, k, 64)
            blob = HexagonBackend.preprocess(program, []).processed_bytes
            _, commands = read_blob(blob)
            assert 42 in [command.type for command in commands], (m, k)
            command = next(command for command in commands if command.type == 42)
            assert command.params[12] == m
            assert command.params[18] == k

    program, _, _ = _quantized_program("w8a16", 32, 64, 64)
    support = _support(program)
    mm = _mm_node(program)
    assert support.is_node_supported(None, mm)
    assert support.is_node_supported(None, mm.args[1])

    program, _, _ = _quantized_program("w8a16", 33, 96, 64)
    support = _support(program)
    mm = _mm_node(program)
    assert not support.is_node_supported(None, mm)
    assert not support.is_node_supported(None, mm.args[1])


def test_a_prefill_over_two_run_time_tensors_stays_portable():
    """The run-time-weight rule above M == 1, where a different kernel decides.

    `annotate` reads the operand's position rather than its provenance, so an
    `a @ b` over two live tensors gets a per-channel observer for something that
    is not a weight. The prefill emitter packs that tensor into the tile order
    and reads the fp16 scale beside it, so a weight the export cannot read is a
    failure at the emitter rather than a fallback -- which is why the refusal has
    to happen here, and has to happen for `M > 1` too: the entry that answers
    above one row is a different kernel from the two the M == 1 rule was written
    for, and nothing about sharing the predicate for `M == 1` makes it the same
    question.
    """
    k, n = 64, 128
    x = torch.randn(8, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    converted, inputs = _quantize("q4a16", _Live("at").eval(), x, b)
    program = to_edge_transform_and_lower(
        torch.export.export(converted, inputs),
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    graph = program.graph_module.graph
    assert not [
        node
        for node in graph.nodes
        if str(node.target).endswith("executorch_call_delegate")
    ], [node.name for node in graph.nodes]
    # The same shape with a weight in that position does reach the prefill entry,
    # so what refuses this is the run-time weight and not the eight rows.
    program, _, _ = _quantized_program("q4a16", 8, k, n)
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    _, commands = read_blob(blob)
    # K == 64 needs no activation pack, so this is the kernel and the repack of
    # its 64-channel output packs.
    assert [c.type for c in commands] == [_PREFILL, _BLIT]


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
    """K % 64 and N % 32 are the kernels' own guards, so they gate the partition.

    Both entries check those multiples and return an error rather than compute
    something wrong, but a delegated node cannot fall back, so the check has to
    happen here: 96 is below no power of two, it just is not a multiple of 64,
    and 48 is not a multiple of the 32-channel tile.
    """
    for k, n in ((96, 64), (64, 48)):
        program, _, _ = _quantized_program("q4a16", 1, k, n)
        support = _support(program)
        mm = _mm_node(program)
        assert not support.is_node_supported(None, mm), (k, n)
        assert not support.is_node_supported(None, mm.args[1]), (k, n)


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
    for name in ("DSP_OP_MATMUL_Q4A16_GEMV_I8", "DSP_OP_MATMUL_W8A16_BLOCK_FP16", "DSP_OP_MATMUL_W8A16_GEMV_I8"):
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
    decides this one; it is the kernel's own reading of its bias operand.
    """
    for bias_shape in ((1, 1), (1,)):
        program, _, _ = _quantized_addmm_program("q4a16", 64, 32, bias_shape)
        addmm = _node_of(program, exir_ops.edge.aten.addmm.default)
        support = _support(program)
        assert not support.is_node_supported(None, addmm), bias_shape
        assert not support.is_node_supported(None, addmm.args[2]), bias_shape
    for bias_shape in ((32,), (1, 32)):
        program, _, _ = _quantized_addmm_program("q4a16", 64, 32, bias_shape)
        addmm = _node_of(program, exir_ops.edge.aten.addmm.default)
        support = _support(program)
        assert support.is_node_supported(None, addmm), bias_shape


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
