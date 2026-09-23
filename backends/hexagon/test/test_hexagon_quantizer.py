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


class _Mm(torch.nn.Module):
    """A weight-only matmul, which is the pattern the quantizer annotates."""

    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n) * 0.3)

    def forward(self, x):
        return torch.mm(x, self.weight)


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


def _converted(scheme, m, k, n):
    """The PT2E-converted module and its input, before any lowering."""
    model = _Mm(k, n).eval()
    x = torch.randn(m, k, dtype=torch.float32)
    exported = torch.export.export(model, (x,))
    prepared = prepare_pt2e(exported.module(), get_hexagon_quantizer(scheme))
    with torch.no_grad():
        prepared(x)
    return convert_pt2e(prepared), x


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
    support = HexagonOperatorSupport()
    mm = _mm_node(program)
    assert support.is_node_supported(None, mm)
    assert support.is_node_supported(None, mm.args[1])


def test_a_prefill_quantized_matmul_stays_portable():
    """M > 1 needs the pack64 activation and output repack, so it is refused.

    The GEMV kernels are the M == 1 entries; a prefill matmul has no wired-up
    kernel, so both the matmul and the dequantize that feeds it have to stay on
    the portable kernels rather than reach an emitter that cannot place them.
    """
    program, _, _ = _quantized_program("q4a16", 8, 64, 128)
    support = HexagonOperatorSupport()
    mm = _mm_node(program)
    assert not support.is_node_supported(None, mm)
    assert not support.is_node_supported(None, mm.args[1])


def test_a_shape_the_kernels_would_refuse_stays_portable():
    """K % 64 and N % 32 are the kernels' own guards, so they gate the partition.

    Both entries check those multiples and return an error rather than compute
    something wrong, but a delegated node cannot fall back, so the check has to
    happen here: 96 is below no power of two, it just is not a multiple of 64,
    and 48 is not a multiple of the 32-channel tile.
    """
    support = HexagonOperatorSupport()
    for k, n in ((96, 64), (64, 48)):
        program, _, _ = _quantized_program("q4a16", 1, k, n)
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
    support = HexagonOperatorSupport()
    for bias_shape in ((1, 1), (1,)):
        program, _, _ = _quantized_addmm_program("q4a16", 64, 32, bias_shape)
        addmm = _node_of(program, exir_ops.edge.aten.addmm.default)
        assert not support.is_node_supported(None, addmm), bias_shape
        assert not support.is_node_supported(None, addmm.args[2]), bias_shape
    for bias_shape in ((32,), (1, 32)):
        program, _, _ = _quantized_addmm_program("q4a16", 64, 32, bias_shape)
        addmm = _node_of(program, exir_ops.edge.aten.addmm.default)
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
