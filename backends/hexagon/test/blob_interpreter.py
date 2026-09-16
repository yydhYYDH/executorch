# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs an emitted blob on the host, so a delegated subgraph can be checked.

Every other check in this backend compares a hand transcription of the vendored
C against torch, which says whether the semantics were understood and says
nothing about what the emitters actually wrote. This reads the bytes the
emitters produced -- the same header, ops and sections the runtime parses --
builds the same arena, and executes the command stream.

What it does not model is the HVX kernels: an op whose arithmetic is absent here
raises rather than guessing, so a subgraph that reaches one is reported as
unverified instead of silently passing. The layer it does cover is the one where
the mistakes have been: a region that walks the wrong axis, a patch that lands in
the wrong param, a stride in the wrong slot. All of those are byte-level and
visible here.

The structs come from `serialization/blob.py` rather than being restated, so a
layout change cannot make this agree with a writer it no longer matches.
"""

import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from executorch.backends.hexagon.serialization import blob as B

#: DSPOpType values this can execute.
RASTER_BLIT = 3
LAYER_NORM = 8
UNARY = 4
BINARY_ELEMENTWISE = 19
BATCH_MATMUL = 38

#: fp16, the only element size any op this backend emits carries.
FP16_BYTES = 2

SOFTMAX = 28
REDUCTION = 29

#: HtpOpsReductionOpType.
REDUCTION_SUM = 1
REDUCTION_MAXIMUM = 2
REDUCTION_MEAN = 3

#: Operands the DSP reads as a null pointer.
ABSENT = B.TensorSpace.ABSENT


class UnsupportedOp(Exception):
    """A command whose arithmetic is not modelled here."""


@dataclass(frozen=True)
class Header:
    magic: int
    version: int
    n_ops: int
    n_inputs: int
    n_outputs: int
    weights_bytes: int
    inputs_bytes: int
    activations_bytes: int
    outputs_bytes: int


@dataclass(frozen=True)
class Command:
    type: int
    inputs: List[B.TensorRef]
    outputs: List[B.TensorRef]
    params: List[int]
    patch_param: int
    patch_input: int
    patch_scale: int
    in_place: int


def read_blob(data: bytes) -> Tuple[Header, List[Command]]:
    """Decodes a blob exactly as the runtime does."""
    if len(data) < B.HEADER_SIZE:
        raise ValueError("hexagon: truncated blob header")
    header = Header(*B._HEADER.unpack_from(data, 0))
    if header.magic != B.BLOB_MAGIC:
        raise ValueError(f"hexagon: bad blob magic 0x{header.magic:08x}")

    commands: List[Command] = []
    at = B.HEADER_SIZE
    for _ in range(header.n_ops):
        if at + B.OP_SIZE > len(data):
            raise ValueError("hexagon: truncated blob ops")
        fields = B._OP_PREFIX.unpack_from(data, at)
        n_params = fields[3]
        tail = 4 + B.MAX_OP_PARAMS
        refs_at = at + B._OP_PREFIX.size
        # TensorRef packs as (space, index, offset, size) and constructs as
        # (space, offset, size, index); the two orders are not the same.
        refs = []
        for i in range(B.MAX_OP_INPUTS + B.MAX_OP_OUTPUTS):
            space, index, offset, size = B._TENSOR_REF.unpack_from(
                data, refs_at + i * B._TENSOR_REF.size
            )
            refs.append(B.TensorRef(B.TensorSpace(space), offset, size, index))
        commands.append(
            Command(
                type=fields[0],
                inputs=refs[: B.MAX_OP_INPUTS][: fields[1]],
                outputs=refs[B.MAX_OP_INPUTS :][: fields[2]],
                params=list(fields[4 : 4 + n_params]),
                patch_param=fields[tail],
                patch_input=fields[tail + 1],
                patch_scale=fields[tail + 2],
                in_place=fields[tail + 3],
            )
        )
        at += B.OP_SIZE
    return header, commands


class Arena:
    """The four sections the runtime lays out, in the order it lays them out."""

    def __init__(self, header: Header, data: bytes, name: str) -> None:
        self.header = header
        self.name = name
        self.base: Dict[int, int] = {}
        cursor = 0
        for space, size in (
            (B.TensorSpace.WEIGHTS, header.weights_bytes),
            (B.TensorSpace.INPUT, header.inputs_bytes),
            (B.TensorSpace.ACTIVATION, header.activations_bytes),
            (B.TensorSpace.OUTPUT, header.outputs_bytes),
        ):
            self.base[int(space)] = cursor
            cursor = _align(cursor + size)
        self.bytes = bytearray(cursor)

        # The blob carries the two sections the host owns; the other two are
        # filled from the caller's tensors.
        host_at = B.HEADER_SIZE + header.n_ops * B.OP_SIZE
        self.bytes[
            self.base[int(B.TensorSpace.WEIGHTS)] : self.base[int(B.TensorSpace.WEIGHTS)]
            + header.weights_bytes
        ] = data[host_at : host_at + header.weights_bytes]
        act = self.base[int(B.TensorSpace.ACTIVATION)]
        self.bytes[act : act + header.activations_bytes] = data[
            host_at + header.weights_bytes : host_at
            + header.weights_bytes
            + header.activations_bytes
        ]

    def address(self, ref: B.TensorRef) -> int:
        if ref.space == ABSENT:
            raise UnsupportedOp(f"{self.name}: operand is absent")
        return self.base[int(ref.space)] + ref.offset

    def view(self, ref: B.TensorRef) -> memoryview:
        at = self.address(ref)
        return memoryview(self.bytes)[at : at + ref.size]


def _align(value: int, alignment: int = B.ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


class RegionOutOfBounds(Exception):
    """A region reaches outside the arena.

    Worth its own error: a slice assignment in Python silently resizes the
    bytearray when the two sides differ in length, so a region that runs off the
    end would otherwise truncate the arena and corrupt every later read instead
    of failing here.
    """


def _store(arena: Arena, at: int, raw: bytes) -> None:
    if at < 0 or at + len(raw) > len(arena.bytes):
        raise RegionOutOfBounds(
            f"{arena.name}: [{at}, {at + len(raw)}) is outside an arena of "
            f"{len(arena.bytes)} bytes"
        )
    arena.bytes[at : at + len(raw)] = raw


def _copy(arena: Arena, dst_at: int, src_at: int, size: int) -> None:
    if src_at < 0 or src_at + size > len(arena.bytes):
        raise RegionOutOfBounds(
            f"{arena.name}: [{src_at}, {src_at + size}) is outside an arena of "
            f"{len(arena.bytes)} bytes"
        )
    # The right-hand side is materialized before assignment, so an element that
    # overlaps its own destination still reads the old value.
    _store(arena, dst_at, arena.bytes[src_at : src_at + size])


def _patched(command: Command, arena: Arena) -> List[int]:
    """Applies the patch slot, the way the runtime does before the flush.

    A raw four-byte read scaled by patch_scale, truncated back to int32; no
    operand's dtype is consulted, which is exactly why a scaled position has to
    be requested explicitly.
    """
    params = list(command.params)
    if command.patch_param == B.NO_PATCH:
        return params
    raw = bytes(arena.view(command.inputs[command.patch_input])[:4])
    value = int.from_bytes(raw, "little", signed=True)
    value = int(value * command.patch_scale)
    value &= 0xFFFFFFFF
    params[command.patch_param] = value - (1 << 32) if value >= (1 << 31) else value
    return params


def _run_raster_blit(command: Command, params: List[int], arena: Arena) -> None:
    """The general fallback of the vendored blit_ops.cc.

    size[0] is the outermost axis and size[1] the middle one. The innermost axis
    is copied element by element with its own stride whenever either inner
    stride is not one; the vendored code takes that shortcut only when both are
    one, and the specialised paths above it -- row copy, pack, transpose --
    compute the same thing faster. Modelling the fallback rather than those is
    deliberate: it is the definition the fast paths are supposed to agree with,
    so a disagreement is a kernel bug rather than something to reproduce here.

    Every offset and stride counts elements of `unit` bytes.
    """
    regions = params[0]
    unit = params[1]
    # The destination is the op's own output rather than a position among the
    # refs: params[2] counts the sources, and an op can carry inputs past them
    # whose refs sit between the sources and the output. A slice whose start row
    # is patched sends the row along that way.
    sources = command.inputs
    dst = command.outputs[0]
    for index in range(regions):
        at = 3 + 12 * index
        (
            src_index,
            src_offset,
            dst_offset,
            s0,
            s1,
            s2,
            ss0,
            ss1,
            ss2,
            ds0,
            ds1,
            ds2,
        ) = params[at : at + 12]
        src = sources[src_index]
        src_base = arena.address(src) + src_offset * unit
        dst_base = arena.address(dst) + dst_offset * unit
        # A region onto itself with matching strides is dropped, before any of
        # the specialised paths are tried.
        if src_base == dst_base and (ss0, ss1, ss2) == (ds0, ds1, ds2):
            continue
        element_wise = ss2 != 1 or ds2 != 1
        for z in range(s0):
            for y in range(s1):
                a = src_base + (z * ss0 + y * ss1) * unit
                b = dst_base + (z * ds0 + y * ds1) * unit
                if not element_wise:
                    _copy(arena, b, a, s2 * unit)
                    continue
                for x in range(s2):
                    _copy(arena, b + x * ds2 * unit, a + x * ss2 * unit, unit)


def _run_layer_norm(command: Command, params: List[int], arena: Arena) -> None:
    """The scalar path of htp_ops_layer_norm.

    gamma and beta are null here, so the affine step is skipped -- the emitter
    leaves both absent because the scale is a separate multiply in the graph.
    The last param selects RMS mode, and in RMS mode the mean is not merely
    dropped but never accumulated, which is the whole difference between this op
    doing an RMSNorm and doing a LayerNorm. Accumulation is fp32 throughout and
    the result is written back as fp16.

    The vendored kernel reduces in a different order than numpy, so a result
    here is only expected to agree to about fp16's own precision.
    """
    refs = list(command.inputs) + list(command.outputs)
    rows, inner, eps_bits, rms = params[0], params[1], params[2], params[3]
    eps = _as_float(eps_bits)
    dst = refs[3]
    gamma = refs[1]
    beta = refs[2]
    if gamma.space != ABSENT or beta.space != ABSENT:
        raise UnsupportedOp("blob: the affine step of layer_norm is not modelled")

    source = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16).reshape(
        rows, inner
    )
    x = source.astype(np.float32)
    sqsum = np.sum(x * x, axis=1, dtype=np.float32)
    variance = sqsum / inner
    if not rms:
        mean = np.sum(x, axis=1, dtype=np.float32) / inner
        variance = variance - mean * mean
    else:
        mean = np.zeros_like(variance)
    inv_std = (1.0 / np.sqrt(variance + eps)).astype(np.float32)
    out = ((x - mean[:, None]) * inv_std[:, None]).astype(np.float16)
    _store(arena, arena.address(dst), out.tobytes())


def _as_float(bits: int) -> float:
    """A param that carries a float in its low 32 bits."""
    return struct.unpack("<f", struct.pack("<i", bits))[0]


def _run_batch_matmul(command: Command, params: List[int], arena: Arena) -> None:
    """The general path of htp_ops_loop_matmul_region.

    The descriptor names three axes -- rows, contraction, columns -- and gives
    the strides in bytes, which the vendored code adds to a byte pointer
    directly. Accumulation is fp32 in the axis order the kernel uses and the
    store is fp16, so a sequential sum here reproduces it rather than merely
    approximating it.

    The iterator operands are absent in every op this backend emits, which is
    what makes the loop count one; anything else is left unmodelled.
    """
    # params[0] is the element size, then the descriptor.
    unit = params[0]
    rows, inner, cols = params[2], params[3], params[4]
    dst_stride = params[5:8]
    src0_stride = params[8:11]
    src1_stride = params[11:14]
    if params[1] != 1:
        raise UnsupportedOp(f"blob: a matmul loop of {params[1]} is not modelled")

    refs = list(command.inputs) + list(command.outputs)
    dst = refs[len(command.inputs)]
    for index in range(2, len(command.inputs)):
        if command.inputs[index].space != ABSENT:
            raise UnsupportedOp("blob: a matmul with an iterator is not modelled")

    src0 = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16)
    src1 = np.frombuffer(bytes(arena.view(refs[1])), dtype=np.float16)
    out = np.zeros(rows * cols, dtype=np.float16)
    for row in range(rows):
        for col in range(cols):
            total = np.float32(0.0)
            for k in range(inner):
                a = (row * src0_stride[0] + k * src0_stride[1]) // unit
                b = (k * src1_stride[1] + col * src1_stride[2]) // unit
                total = np.float32(total) + np.float32(src0[a]) * np.float32(src1[b])
            at = (row * dst_stride[0] + col * dst_stride[2]) // unit
            out[at] = total
    _store(arena, arena.address(dst), out.tobytes())


def _f32(value):
    return np.float32(value)


#: The unary ops computed exactly. The rest -- log, rsqrt, expm1, cos and sin --
#: go through the DSP's own fast approximations, so a host model would be
#: guessing at them rather than reproducing them.
_UNARY = {
    1: lambda x: np.where(x < 0, -x, x),  # abs
    2: lambda x: -x,  # neg
    3: lambda x: (  # gelu, the tanh form
        0.5 * x * (1.0 + np.tanh(_f32(0.79788456) * (x + _f32(0.044715) * x * x * x)))
    ),
    4: lambda x: 1.0 / (1.0 + np.exp(-x)),  # sigmoid
    5: lambda x: np.exp(x),  # exp
    7: lambda x: np.where(x >= 8, x, np.where(x <= -8, 0.0, x / (1.0 + np.exp(-x)))),
    8: lambda x: np.tanh(x),  # tanh
    9: lambda x: x * x,  # square
    10: lambda x: np.sqrt(x),  # sqrt
}


def _run_unary(command: Command, params: List[int], arena: Arena) -> None:
    """Element-wise, one element at a time, from htp_ops_unary_apply_fp16."""
    numel, op_type = params[0], params[1]
    if op_type not in _UNARY:
        raise UnsupportedOp(
            f"blob: unary op {op_type} uses a DSP approximation, not modelled"
        )
    refs = list(command.inputs) + list(command.outputs)
    source = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16)[:numel]
    out = _UNARY[op_type](source.astype(np.float32)).astype(np.float16)
    _store(arena, arena.address(refs[1]), out.tobytes())


#: The binary ops, evaluated the way htp_ops_binary_apply_fp16 does. The ones
#: spelled with a cast promote both operands first; the rest are single fp16
#: operations, which round the same either way.
_BINARY = {
    1: lambda a, b: a + b,
    2: lambda a, b: a - b,
    3: lambda a, b: a * b,
    4: lambda a, b: (a.astype(np.float32) / b.astype(np.float32)).astype(np.float16),
    5: lambda a, b: np.maximum(a, b),
    6: lambda a, b: np.minimum(a, b),
    7: lambda a, b: (  # mul_silu
        a.astype(np.float32)
        * b.astype(np.float32)
        / (1.0 + np.exp(-b.astype(np.float32)))
    ).astype(np.float16),
    8: lambda a, b: np.maximum(a + b, np.float16(0)),  # add_relu
    11: lambda a, b: ((a.astype(np.float32) - b.astype(np.float32)) ** 2).astype(
        np.float16
    ),
}


def _run_binary(command: Command, params: List[int], arena: Arena) -> None:
    """The broadcast path of htp_ops_binary_try_broadcast.

    The tail after params[8] is a rank, the output extents, then one stride list
    per operand; a broadcast dimension carries a stride of zero, which is the
    whole mechanism. Strides count elements.
    """
    out_size, op_type = params[0], params[3]
    if op_type not in _BINARY:
        raise UnsupportedOp(f"blob: binary op {op_type} is not modelled")
    dims = params[8]
    out_dims = params[9:17][:dims]
    in0_strides = params[17:25][:dims]
    in1_strides = params[25:33][:dims]
    # The kernel refuses the descriptor unless the extents multiply out to the
    # declared output size, so a mismatch here is a mismatch there.
    if int(np.prod(out_dims)) != out_size:
        raise UnsupportedOp(
            f"blob: extents {out_dims} do not multiply to {out_size}"
        )

    refs = list(command.inputs) + list(command.outputs)
    in0 = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16)
    in1 = np.frombuffer(bytes(arena.view(refs[1])), dtype=np.float16)

    off0 = np.zeros(out_size, dtype=np.int64)
    off1 = np.zeros(out_size, dtype=np.int64)
    index = np.arange(out_size, dtype=np.int64)
    for d in range(dims - 1, -1, -1):
        coord = index % out_dims[d]
        index = index // out_dims[d]
        off0 += coord * in0_strides[d]
        off1 += coord * in1_strides[d]

    out = _BINARY[op_type](in0[off0], in1[off1])
    _store(arena, arena.address(refs[2]), out.tobytes())


def _run_softmax(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_softmax, over the middle axis of an [outside][channel][inside] view.

    The kernel subtracts the row maximum, exponentiates, sums in fp32 and
    divides. Its exponential is an HVX approximation (hvx_my_exp2_vhf) rather
    than libm, so this models the shape of the computation and not its last
    bits, and a comparison against it has to be a tolerance.
    """
    outside, channel, inside, unit = params[0], params[1], params[2], params[3]
    if unit != FP16_BYTES:
        raise UnsupportedOp(f"blob: a softmax over {unit}-byte values is not modelled")
    src = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)
    if src.size != outside * channel * inside:
        raise UnsupportedOp(
            f"blob: {src.size} values do not fill [{outside}][{channel}][{inside}]"
        )
    rows = src.reshape(outside, channel, inside).astype(np.float32)
    shifted = np.exp(rows - rows.max(axis=1, keepdims=True))
    out = (shifted / shifted.sum(axis=1, keepdims=True)).astype(np.float16)
    _store(arena, arena.address(command.outputs[0]), out.tobytes())


def _run_reduction(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_reduction over the middle axis, summing in fp32.

    The mean is taken after the sum rather than incrementally, which is what the
    scalar and inside1 paths both do.
    """
    outside, reduce, inside = params[0], params[1], params[2]
    op_type, unit = params[3], params[4]
    if unit != FP16_BYTES:
        raise UnsupportedOp(f"blob: a reduction over {unit}-byte values is not modelled")
    src = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)
    if src.size != outside * reduce * inside:
        raise UnsupportedOp(
            f"blob: {src.size} values do not fill [{outside}][{reduce}][{inside}]"
        )
    rows = src.reshape(outside, reduce, inside).astype(np.float32)
    if op_type == REDUCTION_SUM:
        out = rows.sum(axis=1)
    elif op_type == REDUCTION_MAXIMUM:
        out = rows.max(axis=1)
    elif op_type == REDUCTION_MEAN:
        out = rows.sum(axis=1) / reduce
    else:
        raise UnsupportedOp(f"blob: reduction kind {op_type} is not modelled")
    _store(
        arena,
        arena.address(command.outputs[0]),
        out.astype(np.float16).tobytes(),
    )


_EXECUTORS = {
    RASTER_BLIT: _run_raster_blit,
    SOFTMAX: _run_softmax,
    REDUCTION: _run_reduction,
    UNARY: _run_unary,
    BINARY_ELEMENTWISE: _run_binary,
    LAYER_NORM: _run_layer_norm,
    BATCH_MATMUL: _run_batch_matmul,
}


def execute(data: bytes, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Runs a blob over the host arena and returns its outputs by index.

    `inputs` is indexed by method input index, which is the order the emitters
    declared their placeholders in.
    """
    header, commands = read_blob(data)
    if len(inputs) != header.n_inputs:
        raise ValueError(f"blob wants {header.n_inputs} inputs, got {len(inputs)}")

    arena = Arena(header, data, "blob")
    for index, tensor in enumerate(inputs):
        ref = _slot(commands, B.TensorSpace.INPUT, index)
        raw = tensor.tobytes()
        if len(raw) != ref.size:
            raise ValueError(
                f"input {index} is {len(raw)} bytes, the blob expects {ref.size}"
            )
        _copy(arena, arena.address(ref), arena.address(ref), 0)
        arena.bytes[arena.address(ref) : arena.address(ref) + ref.size] = raw

    for command in commands:
        executor = _EXECUTORS.get(command.type)
        if executor is None:
            raise UnsupportedOp(
                f"blob: op type {command.type} is not modelled on the host"
            )
        executor(command, _patched(command, arena), arena)

    outputs = []
    for index in range(header.n_outputs):
        ref = _slot(commands, B.TensorSpace.OUTPUT, index)
        outputs.append(np.frombuffer(bytes(arena.view(ref)), dtype=np.uint8))
    return outputs


def _slot(commands: List[Command], space: B.TensorSpace, index: int) -> B.TensorRef:
    """The ref the emitters used for one method input or output."""
    for command in commands:
        for ref in list(command.inputs) + list(command.outputs):
            if ref.space == space and ref.index == index:
                return ref
    raise ValueError(f"blob: no slot for {space.name.lower()} {index}")
