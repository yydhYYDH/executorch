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

The arithmetic it does model is a model, not the kernels' instruction sequence:
the quantized GEMV ops reproduce what the HVX code computes -- the per-token
activation quantization and the scales -- in numpy, float by float, and numpy
does not round the way a vector unit with two roundings in a row does in every
last case. Read those ops as "the numbers this should produce", not as a
bit-exact emulator.

The structs come from `serialization/blob.py` rather than being restated, so a
layout change cannot make this agree with a writer it no longer matches.
"""

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from executorch.backends.hexagon.serialization import blob as B

#: DSPOpType values this can execute.
POOL2D_FP16 = 1
CONV_DEPTHWISE2D_FP16 = 2
IM2COL_CONVOLUTION_FP16 = 12
ZERO = 24
RASTER_BLIT = 3
LAYER_NORM = 8
ADD_FUSE_LAYERNORM = 16
UNARY = 4
BINARY_ELEMENTWISE = 19
BATCH_MATMUL = 38
FLASH_ATTN = 18
ROPE = 14
MATMUL_Q4A16_GEMV_I8 = 41
MATMUL_W8A16_GEMV_I8 = 45
SHARED_GATHER = 23
VISION_ATTENTION_FP16 = 43
SELECT = 26

#: The isInt4 slot of htp_ops_shared_gather that names an fp16 table, which is
#: the only kind this backend stores; 2 and 3 are int4 and int8 tables.
SHARED_GATHER_FP16 = 0

#: fp16, the only element size any op this backend emits carries.
FP16_BYTES = 2

SOFTMAX = 28
REDUCTION = 29
TOPKV2_K1_FP16 = 27

#: HtpOpsReductionOpType.
REDUCTION_SUM = 1
REDUCTION_MAXIMUM = 2
REDUCTION_MEAN = 3

#: HTP_OPS_UNARY_CLAMP, the one HtpOpsUnaryOpType whose params carry operands
#: of their own: params[3] and params[4] are its two fp16 bounds.
_UNARY_CLAMP = 15

#: The DSP pool kernel's two selectors and its channel block
#: (`hvx_pool2d_fp16`).
POOL_MAX = 0
POOL_COUNT_VALID = 0
POOL_COUNT_KERNEL = 1
POOL_PACK = 64

#: The HMX unit's tile geometry: a 32x32 tile of fp16, and the 64 lanes the
#: blocked activation carries.
HMX_TILE = 32
HMX_TILE_ELMS = HMX_TILE * HMX_TILE

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


@dataclass(frozen=True)
class ExternalWeights:
    """Where the weights the .pte does not carry have to be filled in.

    `weights_bytes` is the arena's weights section -- the file's own bytes
    followed by every entry -- which is what the runtime sizes that section
    from. The entries' offsets are measured from the start of it, so each one
    lands at `weights_base + entry.offset` and the file's half is a prefix.
    """

    weights_bytes: int
    entries: List[B.ExternalWeight]
    #: Where the section after the weights starts in the file.
    trailer_at: int


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


def read_external_weights(data: bytes) -> ExternalWeights:
    """Reads the trailer listing the weights the file does not carry.

    Mirrors the runtime: a blob that is not v3 has none, and the bytes between
    the end of the weights and the dynamic trailer belong to the file's own
    weights section and to nothing else.
    """
    header, _ = read_blob(data)
    weights_at = B.HEADER_SIZE + header.n_ops * B.OP_SIZE
    weights_end = weights_at + header.weights_bytes
    if weights_end > len(data):
        raise ValueError("hexagon: truncated weights section")
    if header.version != B.BLOB_VERSION_EXTERNAL_WEIGHTS:
        return ExternalWeights(header.weights_bytes, [], weights_end)

    if len(data) - weights_end < B._EXTERNAL_HEADER.size:
        raise ValueError("hexagon: truncated external weights trailer")
    magic, version, n_ext, _reserved, weights_bytes = B._EXTERNAL_HEADER.unpack_from(
        data, weights_end
    )
    if magic != B.EXTERNAL_WEIGHTS_MAGIC:
        raise ValueError(f"hexagon: bad external weights magic 0x{magic:08x}")
    if version != B.EXTERNAL_WEIGHTS_VERSION:
        raise ValueError(f"hexagon: unknown external weights version {version}")
    if weights_bytes < header.weights_bytes:
        raise ValueError("hexagon: the trailer's weights section is too small")

    entries_at = weights_end + B._EXTERNAL_HEADER.size
    if n_ext == 0 or entries_at + n_ext * B._EXTERNAL_WEIGHT.size > len(data):
        raise ValueError("hexagon: external weight records out of bounds")

    entries: List[B.ExternalWeight] = []
    end = header.weights_bytes
    for i in range(n_ext):
        offset, size, raw_key = B._EXTERNAL_WEIGHT.unpack_from(
            data, entries_at + i * B._EXTERNAL_WEIGHT.size
        )
        key = raw_key.split(b"\x00", 1)[0]
        if not key:
            raise ValueError("hexagon: an external weight has no name")
        if size == 0 or offset < end or offset + size > weights_bytes:
            raise ValueError(
                f"hexagon: external weight {key!r} is out of bounds at {offset}"
            )
        end = offset + size
        entries.append(B.ExternalWeight(offset, size, key.decode("ascii")))
    return ExternalWeights(
        weights_bytes, entries, entries_at + n_ext * B._EXTERNAL_WEIGHT.size
    )


class Arena:
    """The four sections the runtime lays out, in the order it lays them out."""

    def __init__(
        self,
        header: Header,
        data: bytes,
        name: str,
        named_data: Optional[Dict[str, bytes]] = None,
    ) -> None:
        self.header = header
        self.name = name
        self.base: Dict[int, int] = {}
        external = read_external_weights(data)
        # The weights section is the file's bytes followed by the weights the
        # file left out, so it is the trailer's size that bounds it, not the
        # header's: the header's is a file offset, this is an arena size.
        cursor = 0
        for space, size in (
            (B.TensorSpace.WEIGHTS, external.weights_bytes),
            (B.TensorSpace.INPUT, header.inputs_bytes),
            (B.TensorSpace.ACTIVATION, header.activations_bytes),
            (B.TensorSpace.OUTPUT, header.outputs_bytes),
        ):
            self.base[int(space)] = cursor
            cursor = _align(cursor + size)
        self.bytes = bytearray(cursor)

        # The blob carries the weights and nothing else: the inputs and outputs
        # come from the caller's tensors, and the activations are scratch the
        # runtime reserves from the header size and never reads back, so they
        # are zero here exactly as they would be if the file carried them.
        host_at = B.HEADER_SIZE + header.n_ops * B.OP_SIZE
        weights_base = self.base[int(B.TensorSpace.WEIGHTS)]
        self.bytes[weights_base : weights_base + header.weights_bytes] = data[
            host_at : host_at + header.weights_bytes
        ]

        # The rest of the section comes from the caller, key by key, the way the
        # runtime copies it out of the named data map.
        store = named_data or {}
        for entry in external.entries:
            supply = store.get(entry.key)
            if supply is None:
                raise ValueError(
                    f"{self.name}: the blob needs external weight {entry.key!r} "
                    f"and no data map has it"
                )
            if len(supply) != entry.size:
                raise ValueError(
                    f"{self.name}: external weight {entry.key!r} is "
                    f"{len(supply)} bytes, the blob expects {entry.size}"
                )
            _store(self, weights_base + entry.offset, supply)

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


@dataclass(frozen=True)
class DynamicTrailer:
    """The dynamic metadata a blob carries after its weights.

    `input_index` and `axis` name the operand the run-time length is read off;
    `max_length` is the longest the export declared. Each patch names one
    command's parameter slot and the scale a length reaches it by.
    """

    input_index: int
    axis: int
    max_length: int
    example_length: int
    patches: List[Tuple[int, int, int, int]]


def read_dynamic_trailer(data: bytes) -> Optional[DynamicTrailer]:
    """The trailer, or None for a blob whose lengths are all static.

    Mirrors the runtime: it starts where the weights section ends, which is
    `read_external_weights`' own answer for where the next section begins.
    """
    at = read_external_weights(data).trailer_at
    if at + B._DYNAMIC_HEADER_V3.size > len(data):
        return None
    magic, _version, input_index, axis, max_length, n_patches, example = (
        B._DYNAMIC_HEADER_V3.unpack_from(data, at)
    )
    if magic != B.DYNAMIC_TRAILER_MAGIC:
        return None
    at += B._DYNAMIC_HEADER_V3.size
    patches = [
        B._DYNAMIC_PATCH.unpack_from(data, at + 16 * i) for i in range(n_patches)
    ]
    return DynamicTrailer(
        input_index=input_index,
        axis=axis,
        max_length=max_length,
        example_length=example,
        patches=patches,
    )


def _length_patches(
    data: bytes, inputs: Sequence[np.ndarray], length: Optional[int] = None
) -> Dict[int, Dict[int, int]]:
    """The params the run-time length rewrites, keyed by command then slot.

    The runtime derives the length from the shape of the input the trailer names
    and writes `length * scale + add` into the named slot of the named command
    before it issues anything (hexagon_backend.cpp:2147-2156). Reading it back
    off the operand the caller actually passed is the whole point: a command
    whose span holds the exported bound instead of the run length answers a
    question about the arena rather than about the caller's tensor.

    `length` is that number stated outright, for a caller whose operand is the
    whole bound-sized arena slot rather than the tensor the runtime would be
    handed -- a fixture has to write the slot to say what is past the live rows.
    """
    trailer = read_dynamic_trailer(data)
    if trailer is None:
        return {}
    if length is None:
        if trailer.input_index >= len(inputs):
            raise ValueError(
                f"hexagon: the trailer names input {trailer.input_index}, "
                f"and {len(inputs)} were passed"
            )
        shape = inputs[trailer.input_index].shape
        if trailer.axis >= len(shape):
            raise ValueError(
                f"hexagon: the trailer's axis {trailer.axis} is past a rank "
                f"{len(shape)} operand"
            )
        length = int(shape[trailer.axis])
    rewritten: Dict[int, Dict[int, int]] = {}
    for op_index, param_index, scale, add in trailer.patches:
        value = length * scale + add
        if not -(1 << 31) <= value < (1 << 31):
            raise ValueError(f"hexagon: dynamic patch {value} does not fit int32")
        rewritten.setdefault(op_index, {})[param_index] = value
    return rewritten


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

    beta is null here, so any bias is skipped; gamma carries the norm's weight
    when the emitter folds one in, and is read as fp32 so the affine step runs in
    fp32 like the kernel's. The last param selects RMS mode, and in RMS mode the
    mean is not merely dropped but never accumulated, which is the whole
    difference between this op doing an RMSNorm and doing a LayerNorm.
    Accumulation is fp32 throughout and the result is written back as fp16.

    The vendored kernel reduces in a different order than numpy, so a result
    here is only expected to agree to about fp16's own precision.
    """
    refs = list(command.inputs) + list(command.outputs)
    rows, inner, eps_bits, rms = params[0], params[1], params[2], params[3]
    eps = _as_float(eps_bits)
    dst = refs[3]
    gamma = refs[1]
    beta = refs[2]
    if beta.space != ABSENT:
        raise UnsupportedOp("blob: the bias of layer_norm is not modelled")

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
    out = (x - mean[:, None]) * inv_std[:, None]
    if gamma.space != ABSENT:
        weight = np.frombuffer(bytes(arena.view(gamma)), dtype=np.float32).reshape(
            inner
        )
        out = out * weight[None, :]
    _store(arena, arena.address(dst), out.astype(np.float16).tobytes())


def _as_float(bits: int) -> float:
    """A param that carries a float in its low 32 bits."""
    return struct.unpack("<f", struct.pack("<i", bits))[0]


def _run_add_fuse_layernorm(command: Command, params: List[int], arena: Arena) -> None:
    """The RMSNorm path of htp_ops_add_fuse_layernorm.

    The kernel adds the two fp16 operands, writes the sum to its second output,
    and normalizes that sum in fp32 with the fp32 gamma -- beta is null, which
    is what selects the RMSNorm flavor. Like `_run_layer_norm` this is the
    scalar path; the vendored kernel reduces in another order and uses an
    approximate rsqrt, so agreement is to about fp16 precision.
    """
    rows, inner, eps_bits, rms = params[0], params[1], params[2], params[3]
    eps = _as_float(eps_bits)
    if not rms:
        raise UnsupportedOp(
            "blob: the layer-norm mode of add_fuse_layernorm is not modelled"
        )
    refs = list(command.inputs) + list(command.outputs)
    if refs[3].space != ABSENT:
        raise UnsupportedOp("blob: the bias of add_fuse_layernorm is not modelled")

    src0 = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16).reshape(
        rows, inner
    )
    src1 = np.frombuffer(bytes(arena.view(refs[1])), dtype=np.float16).reshape(
        rows, inner
    )
    added = (src0 + src1).astype(np.float16)
    _store(arena, arena.address(refs[len(command.inputs) + 1]), added.tobytes())

    x = added.astype(np.float32)
    sqsum = np.sum(x * x, axis=1, dtype=np.float32)
    inv_std = (1.0 / np.sqrt(sqsum / inner + eps)).astype(np.float32)
    out = x * inv_std[:, None]
    if refs[2].space != ABSENT:
        gamma = np.frombuffer(bytes(arena.view(refs[2])), dtype=np.float32).reshape(
            inner
        )
        out = out * gamma[None, :]
    _store(
        arena,
        arena.address(refs[len(command.inputs)]),
        out.astype(np.float16).tobytes(),
    )


def _run_rope(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_rope's scalar path for one (q) tensor.

    The command names q and k, but every emitter here hands it q as both and
    passes kv_num_head = 0, so the k loop runs zero times. Tokens are the leading
    axis and heads sit inside a token, which is the geometry the emitter builds
    by folding a one-wide batch axis into the sequence. The table is the graph's
    own [seq, head_dim] row: the first half is the even angles and the second the
    odd ones, so element d of a head pairs with element d + head_dim/2:

        out[d]          = in[d] * cos[d] - in[d + half] * sin[d]
        out[d + half]   = in[d + half] * cos[d + half] + in[d] * sin[d + half]

    The kernel multiplies and subtracts in fp16, so the products are rounded to
    fp16 rather than accumulated in fp32.
    """
    batch_seq, num_head, kv_num_head, head_dim, rope_dim, input_c4 = params[:6]
    if input_c4:
        raise UnsupportedOp("blob: a c4-packed rope input is not modelled")
    if rope_dim != head_dim:
        raise UnsupportedOp("blob: a rope that rotates only part of a head")
    del kv_num_head

    half = head_dim // 2
    refs = list(command.inputs) + list(command.outputs)
    source = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16)
    cos = np.frombuffer(bytes(arena.view(refs[2])), dtype=np.float16)
    sin = np.frombuffer(bytes(arena.view(refs[3])), dtype=np.float16)
    out = np.array(source, dtype=np.float16)
    token_elems = num_head * head_dim
    for token in range(batch_seq):
        at = token * token_elems
        table = token * head_dim
        x = source[at : at + token_elems].reshape(num_head, head_dim).astype(np.float32)
        c = cos[table : table + head_dim].astype(np.float32)
        s = sin[table : table + head_dim].astype(np.float32)
        lo = (x[:, :half] * c[:half] - x[:, half:] * s[:half]).astype(np.float16)
        hi = (x[:, half:] * c[half:] + x[:, :half] * s[half:]).astype(np.float16)
        out[at : at + token_elems] = np.concatenate([lo, hi], axis=1).reshape(-1)
    _store(arena, arena.address(refs[4]), out.tobytes())


def _untile_hmx(flat: np.ndarray, k: int, n: int) -> np.ndarray:
    """The inverse of `pack_hmx_weight`, back to a row-major (k, n) weight.

    The command carries the packed bytes for the HMX route, which is the route
    the emitter takes by default, so the interpreter has to undo the tile order
    before it can walk the descriptor's row-major strides.
    """
    kp, nt = -(-k // 32), -(-n // 32)
    tiles = flat.reshape(nt, kp, 16, 32, 2)
    units = tiles.transpose(0, 1, 2, 4, 3).reshape(nt, kp, 32, 32)
    padded = units.transpose(1, 2, 0, 3).reshape(kp * 32, nt * 32)
    return np.ascontiguousarray(padded[:k, :n]).reshape(-1)


def _run_batch_matmul(command: Command, params: List[int], arena: Arena) -> None:
    """The general path of htp_ops_loop_matmul_region.

    The descriptor names three axes -- rows, contraction, columns -- and gives
    the strides in bytes, which the vendored code adds to a byte pointer
    directly. Accumulation is fp32 in the axis order the kernel uses and the
    store is fp16, so a sequential sum here reproduces it rather than merely
    approximating it.

    The iterator operands are absent in every op this backend emits, so the DSP
    numbers each iteration itself and reaches that iteration's operands through
    the descriptor's steps, which are elements: with one iteration the steps are
    unreachable and mm is what comes out, and with one step per tile bmm is.

    The plan word marks a weight the host stored in the unit's tile order; that
    operand is un-tiled here, because every stride below is row-major.
    """
    # params[0] is the element size, then the descriptor: the loop count, the
    # three axes, three stride triples in bytes, three steps and three view
    # offsets in elements, then the three operand sizes as int64.
    unit = params[0]
    loops = params[1]
    rows, inner, cols = params[2], params[3], params[4]
    dst_stride = params[5:8]
    src0_stride = params[8:11]
    src1_stride = params[11:14]
    steps = params[14:17]
    views = params[17:20]
    out_elems, in0_elems, in1_elems = params[20], params[22], params[24]
    hmx_prepacked = bool(params[26] & 1)
    if params[21] or params[23] or params[25]:
        raise UnsupportedOp("blob: a matmul operand of 2**31 elements or more")

    refs = list(command.inputs) + list(command.outputs)
    dst = refs[len(command.inputs)]
    for index in range(2, len(command.inputs)):
        if command.inputs[index].space != ABSENT:
            raise UnsupportedOp("blob: a matmul with an iterator is not modelled")

    src0 = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16)
    src1_raw = np.frombuffer(bytes(arena.view(refs[1])), dtype=np.float16)
    src1 = _untile_hmx(src1_raw, inner, cols) if hmx_prepacked else src1_raw
    out = np.zeros(len(arena.view(dst)) // unit, dtype=np.float16)
    for loop in range(loops):
        # An iteration whose three bases do not all land inside their operand is
        # skipped rather than clamped, which is what the DSP does with them.
        out_at = loop * steps[0] + views[0]
        in0_at = loop * steps[1] + views[1]
        in1_at = loop * steps[2] + views[2]
        if not 0 <= out_at < out_elems:
            continue
        if not 0 <= in0_at < in0_elems or not 0 <= in1_at < in1_elems:
            continue
        for row in range(rows):
            for col in range(cols):
                total = np.float32(0.0)
                for k in range(inner):
                    a = in0_at + (row * src0_stride[0] + k * src0_stride[1]) // unit
                    b = in1_at + (k * src1_stride[1] + col * src1_stride[2]) // unit
                    total = np.float32(total) + np.float32(src0[a]) * np.float32(
                        src1[b]
                    )
                at = out_at + (row * dst_stride[0] + col * dst_stride[2]) // unit
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
    refs = list(command.inputs) + list(command.outputs)
    source = np.frombuffer(bytes(arena.view(refs[0])), dtype=np.float16)[:numel]
    if op_type == _UNARY_CLAMP:
        # clamp is the one unary type whose entry point is not
        # htp_ops_unary_compute_fp16_chunk: params[3] and params[4] are the fp16
        # bit patterns of its bounds, and htp_ops_clamp_fp16_chunk compares
        # against those (unary_ops.cc:498-541). The compares are unordered, so a
        # NaN would otherwise come back as the upper bound; the kernel restores
        # the input where |x| > 0x7c00 -- an all-ones exponent with a non-zero
        # mantissa -- which is what torch's clamp does as well.
        lo = np.array([params[3]], dtype=np.uint16).view(np.float16)[0]
        hi = np.array([params[4]], dtype=np.uint16).view(np.float16)[0]
        values = source.astype(np.float32)
        clamped = np.clip(values, np.float32(lo), np.float32(hi))
        isnan = (np.frombuffer(source.tobytes(), dtype=np.uint16) & 0x7FFF) > 0x7C00
        out = np.where(isnan, values, clamped).astype(np.float16)
        _store(arena, arena.address(refs[1]), out.tobytes())
        return
    if op_type not in _UNARY:
        raise UnsupportedOp(
            f"blob: unary op {op_type} uses a DSP approximation, not modelled"
        )
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
    12: lambda a, b: _binary_fmod(a, b),  # mod, the truncated remainder
}


def _binary_fmod(a, b):
    """HTP_OPS_BINARY_MOD's fp16 scalar path, element by element.

    `a - trunc(a/b) * b` in fp32 with the kernel's two guards: a zero divisor and
    a quotient outside int32 both answer zero rather than the NaN fmodf would
    give (eltwise_ops.cc:148-166). This is torch's `fmod`, not its `remainder`,
    which is the floored one -- the two differ in sign whenever the operands do.
    """
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = a32 / b32
    safe = (b32 != 0.0) & (quotient > -2147483648.0) & (quotient < 2147483648.0)
    truncated = np.trunc(np.where(safe, quotient, 0.0)).astype(np.int32)
    result = np.where(safe, a32 - truncated.astype(np.float32) * b32, 0.0)
    return result.astype(np.float16)


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
        raise UnsupportedOp(f"blob: extents {out_dims} do not multiply to {out_size}")

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


def _run_zero(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_zero (blit_ops.cc:1724): a memset over the output operand."""
    at = arena.address(command.outputs[0])
    _store(arena, at, bytes(params[0]))


#: The element size htp_ops_select_cond_at reads the condition at, keyed by the
#: condBytes param. The kernel tests one whole element against zero, so a
#: one-byte condition is a byte per flag and nothing wider (eltwise_ops.cc:2116).
_COND_DTYPES = {1: np.uint8, 2: np.uint16, 4: np.uint32}


def _run_select(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_select (eltwise_ops.cc:2380): `cond ? in1 : in2`, element by element.

    The kernel picks its walk from three sizes: an operand of one element is
    broadcast, the output's own size is read along, and anything else is the
    per-channel mode this backend never emits. The condition is the one operand
    whose width is a parameter rather than two bytes, which is what makes a
    one-byte torch.bool readable.

    The operands are read out of the arena from their own addresses rather than
    out of the reference's own slice, because that is what the kernel does: it is
    handed pointers and sized by params, so a descriptor that named more elements
    than the slot holds reads whatever follows it. A model that stopped at the
    slot's edge would answer a mutated descriptor correctly and the run would say
    the encoding does not matter.
    """
    out_size, cond_size, in1_size, in2_size, bytes_, cond_bytes = params[:6]
    if bytes_ != FP16_BYTES or cond_bytes not in _COND_DTYPES:
        raise UnsupportedOp(
            f"blob: a select over {bytes_}-byte values with {cond_bytes}-byte "
            "conditions is not modelled"
        )
    if in1_size not in (1, out_size) or in2_size not in (1, out_size):
        # A value of any other size is the kernel's per-channel mode, which reads
        # params[6] and params[7]; this emitter never produces one.
        raise UnsupportedOp(
            "blob: a select with a per-channel value operand is not modelled"
        )

    cond_ref = command.inputs[0]
    in1_ref, in2_ref = command.inputs[1], command.inputs[2]
    out_ref = command.outputs[0]

    def read(ref, dtype, count: int) -> np.ndarray:
        at = arena.address(ref)
        width = np.dtype(dtype).itemsize
        if at + count * width > len(arena.bytes):
            raise UnsupportedOp("blob: a select reads an operand past the arena")
        return np.frombuffer(bytes(arena.bytes[at : at + count * width]), dtype=dtype)

    cond_step = 0 if cond_size == 1 else 1
    cond = read(cond_ref, _COND_DTYPES[cond_bytes], cond_step * (out_size - 1) + 1)
    in1 = read(in1_ref, np.uint16, 1 if in1_size == 1 else out_size)
    in2 = read(in2_ref, np.uint16, 1 if in2_size == 1 else out_size)

    at = np.arange(out_size) * cond_step
    on = np.zeros(out_size, dtype=np.intp) if in1_size == 1 else np.arange(out_size)
    off = np.zeros(out_size, dtype=np.intp) if in2_size == 1 else np.arange(out_size)
    _store(arena, arena.address(out_ref), np.where(cond[at] != 0, in1[on], in2[off]).tobytes())


def _run_conv_depthwise2d(command: Command, params: List[int], arena: Arena) -> None:
    """hvx_conv_depthwise2d_fp16 (depthwise_conv_fp16.c:9), one lane at a time.

    Both sides are the DSP's 64-channel blocked activation, so a lane of the
    accumulator is one channel and the whole walk is that channel's own filter.
    The weight is one HMX vector per tap with the channel block outside the taps
    (``wBase = weight + cb * kernelY * kernelX * pack``, :27, and
    ``(ky * kernelX + kx) * pack``, :49). The bias is read as a whole vector per
    block and is never checked for absence, it lands on the narrowed accumulator,
    and relu/relu6 follow it (:64-71).
    """
    (
        batch,
        ih,
        iw,
        oh,
        ow,
        c4,
        kernel_y,
        kernel_x,
        stride_y,
        stride_x,
        pad_y,
        pad_x,
        dilate_y,
        dilate_x,
        relu,
        relu6,
    ) = params[:16]
    source = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)
    weight = np.frombuffer(bytes(arena.view(command.inputs[1])), dtype=np.float16)
    bias = np.frombuffer(bytes(arena.view(command.inputs[2])), dtype=np.float16)
    if source.size != c4 * batch * ih * iw * POOL_PACK:
        raise UnsupportedOp(
            f"blob: {source.size} activation values do not fill {c4} blocks of "
            f"{batch}x{ih}x{iw}"
        )
    if weight.size != c4 * kernel_y * kernel_x * POOL_PACK:
        raise UnsupportedOp(
            f"blob: {weight.size} weight values are not {c4}x{kernel_y}x{kernel_x} taps"
        )
    if bias.size < c4 * POOL_PACK:
        raise UnsupportedOp(
            f"blob: {bias.size} bias values are under one vector per block"
        )

    src = source.reshape(c4, batch, ih * iw, POOL_PACK)
    wgt = weight.reshape(c4, kernel_y, kernel_x, POOL_PACK)
    out = np.zeros((c4, batch, oh * ow, POOL_PACK), dtype=np.float16)
    for n in range(batch):
        for cb in range(c4):
            for oy in range(oh):
                for ox in range(ow):
                    acc = bias[cb * POOL_PACK : (cb + 1) * POOL_PACK].astype(np.float32)
                    for ky in range(kernel_y):
                        iy = oy * stride_y - pad_y + ky * dilate_y
                        if not 0 <= iy < ih:
                            continue
                        for kx in range(kernel_x):
                            ix = ox * stride_x - pad_x + kx * dilate_x
                            if not 0 <= ix < iw:
                                continue
                            # The kernel accumulates with an fp16
                            # multiply-accumulate, so every tap narrows.
                            product = src[cb, n, iy * iw + ix].astype(np.float32) * wgt[
                                cb, ky, kx
                            ].astype(np.float32)
                            acc = (acc + product).astype(np.float16).astype(np.float32)
                    value = acc.astype(np.float16)
                    if relu or relu6:
                        value = np.maximum(value, np.float16(0))
                        if relu6:
                            value = np.minimum(value, np.float16(6))
                    out[cb, n, oy * ow + ox] = value
    _store(arena, arena.address(command.outputs[0]), out.tobytes())


#: Where element (k, c) of a 32x32 weight tile sits in the tile's bytes, which
#: is how the HMX unit reads a column of it (`pack_hmx_weight` writes the same
#: order, and `hmx_load_tiles_fp16` reads it back).
_HMX_TILE_INDEX = (
    (np.arange(HMX_TILE)[:, None] // 2) * 64
    + np.arange(HMX_TILE)[None, :] * 2
    + (np.arange(HMX_TILE)[:, None] % 2)
)


def _run_im2col_convolution(command: Command, params: List[int], arena: Arena) -> None:
    """hmx_im2col_convolution_fp16 (im2col_convolution_fp16.cc:1761), as a product.

    The command is a matrix product over an im2col patch: tile ``i`` of the
    weight is ``(ky * kernelX + kx) * ic_blocks + ic_block`` with
    ``ic_blocks = ceil(ic / 32)`` (fill_im2col_activation_kk_range, :248), the k
    inside a tile is the channel inside that 32-channel group, and the tile's
    bytes are the blob's verbatim (fill_weight_tiles_fp16, :1673). The unit
    accumulates in fp32 and narrows once; the store then adds the bias to that
    narrowed value in fp16 (store_output_tile_pair_fp16, :163-181). This model
    sums the same products in a different order, which only shows on values that
    round differently.
    """
    (
        pad_x,
        pad_y,
        dilate_x,
        dilate_y,
        stride_x,
        stride_y,
        kernel_x,
        kernel_y,
        _ic_div4,
        kernel_units,
        iw,
        ih,
        ow,
        oh,
        _src_z_step,
        _src_y_step,
        _pack_c_unit,
        _dest_ic_stride,
        ic,
        _icup4,
    ) = params[:20]
    (
        oc,
        _mp,
        _np,
        relu,
        relu6,
        batch,
        _output_bytes,
        _scale_block_num,
        _scale_asymmetric,
    ) = params[20:29]
    if relu or relu6:
        # The emitters leave the fused activations off and let to_edge's own
        # relu node reach the unary kernel, so a nonzero one here is a command
        # this model has never been asked to reproduce.
        raise UnsupportedOp("blob: the im2col convolution's fused relu is not modelled")

    source = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)
    weight = np.frombuffer(bytes(arena.view(command.inputs[1])), dtype=np.float16)
    bias = np.frombuffer(bytes(arena.view(command.inputs[2])), dtype=np.float16)
    k_units = -(-ic // 32)
    if kernel_units != kernel_y * kernel_x * k_units:
        raise UnsupportedOp(
            f"blob: {kernel_units} kernel units are not {kernel_y}x{kernel_x} of "
            f"{k_units} channel groups"
        )
    in_blocks = -(-ic // 64)
    if source.size != in_blocks * batch * ih * iw * POOL_PACK:
        raise UnsupportedOp(
            f"blob: {source.size} activation values do not fill the input blocks"
        )
    if weight.size != -(-oc // 32) * kernel_units * HMX_TILE_ELMS:
        raise UnsupportedOp(
            f"blob: {weight.size} weight values are not the {oc}-channel tiling"
        )
    if bias.size < oc:
        raise UnsupportedOp(f"blob: {bias.size} bias values are under {oc} channels")

    src = source.reshape(in_blocks, batch, ih * iw, POOL_PACK)
    area = oh * ow
    positions = np.arange(batch * area)
    n = positions // area
    oy = (positions % area) // ow
    ox = positions % ow
    patches = np.zeros((batch * area, kernel_units * HMX_TILE), dtype=np.float16)
    for kernel_index in range(kernel_y * kernel_x):
        ky, kx = divmod(kernel_index, kernel_x)
        iy = oy * stride_y - pad_y + ky * dilate_y
        ix = ox * stride_x - pad_x + kx * dilate_x
        inside = (iy >= 0) & (iy < ih) & (ix >= 0) & (ix < iw)
        flat = (np.clip(iy, 0, ih - 1) * iw + np.clip(ix, 0, iw - 1))[inside]
        for ic_block in range(k_units):
            first = ic_block * HMX_TILE
            width = min(HMX_TILE, ic - first)
            column = (kernel_index * k_units + ic_block) * HMX_TILE
            lanes = np.arange(first, first + width) % POOL_PACK
            patches[inside, column : column + width] = src[first // 64][
                n[inside], flat
            ][:, lanes]

    tiles = weight.reshape(-1, HMX_TILE_ELMS)
    tap = np.zeros((kernel_units * HMX_TILE, oc), dtype=np.float16)
    for channel_block in range(-(-oc // 32)):
        width = min(HMX_TILE, oc - channel_block * HMX_TILE)
        for unit in range(kernel_units):
            tile = tiles[channel_block * kernel_units + unit]
            tap[unit * HMX_TILE : (unit + 1) * HMX_TILE, :][
                :, channel_block * HMX_TILE : channel_block * HMX_TILE + width
            ] = tile[_HMX_TILE_INDEX][:, :width]

    accumulated = patches.astype(np.float32) @ tap.astype(np.float32)
    narrowed = accumulated.astype(np.float16).astype(np.float32) + bias[:oc].astype(
        np.float32
    )
    values = narrowed.astype(np.float16)

    out_blocks = -(-oc // 64)
    blocked = np.zeros((out_blocks, batch, area, POOL_PACK), dtype=np.float16)
    for channel_block in range(out_blocks):
        width = min(POOL_PACK, oc - channel_block * POOL_PACK)
        blocked[channel_block, :, :, :width] = values[
            :, channel_block * POOL_PACK : channel_block * POOL_PACK + width
        ].reshape(batch, area, width)
    _store(arena, arena.address(command.outputs[0]), blocked.tobytes())


def _run_pool2d(command: Command, params: List[int], arena: Arena) -> None:
    """hvx_pool2d_fp16, one vector lane at a time.

    The kernel walks one lane of a whole HVX vector per spatial position, which
    is a channel of the 64-channel block its activation is stored in, so the
    element addressing below is `((c//64) * batch + n) * ih * iw * 64 +
    (y * iw + x) * 64 + c % 64` (pool_fp16.c:22 and :44) written out lane by
    lane. It skips a window position that is outside the input and, when the
    whole window is, stores zero rather than a maximum over nothing.
    """
    (
        batch,
        ih,
        iw,
        oh,
        ow,
        c4,
        kernel_y,
        kernel_x,
        stride_y,
        stride_x,
        pad_y,
        pad_x,
        _pad_type,
        count_type,
        pool_type,
    ) = params[:15]
    if pool_type not in (POOL_MAX, 1):
        raise UnsupportedOp(f"blob: pool type {pool_type} is not modelled")
    source = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)
    if source.size != batch * c4 * ih * iw * POOL_PACK:
        raise UnsupportedOp(
            f"blob: {source.size} values do not fill the {batch}x{c4} blocks of "
            f"{ih}x{iw} the pool was told about"
        )
    out = np.zeros((batch, c4, oh, ow, POOL_PACK), dtype=np.float16)
    blocks = source.reshape(batch, c4, ih, iw, POOL_PACK)
    for n in range(batch):
        for cb in range(c4):
            for oy in range(oh):
                for ox in range(ow):
                    window = []
                    for ky in range(kernel_y):
                        iy = oy * stride_y - pad_y + ky
                        if not 0 <= iy < ih:
                            continue
                        for kx in range(kernel_x):
                            ix = ox * stride_x - pad_x + kx
                            if not 0 <= ix < iw:
                                continue
                            window.append(blocks[n, cb, iy, ix])
                    if not window:
                        continue
                    # The kernel seeds the accumulator from the first position
                    # that landed inside and rounds every step to fp16, so the
                    # walk order is part of the arithmetic.
                    accumulated = window[0]
                    for value in window[1:]:
                        if pool_type == POOL_MAX:
                            accumulated = np.maximum(accumulated, value)
                        else:
                            accumulated = (accumulated + value).astype(np.float16)
                    if pool_type == POOL_MAX:
                        out[n, cb, oy, ox] = accumulated
                        continue
                    # The divisor is one over the count, narrowed to fp16 before
                    # the multiply (pool_fp16.c:74-86).
                    divisor = (
                        kernel_y * kernel_x
                        if count_type == POOL_COUNT_KERNEL
                        else len(window)
                    )
                    inverse = np.float16(1.0 / max(1, divisor))
                    out[n, cb, oy, ox] = (accumulated * inverse).astype(np.float16)
    _store(arena, arena.address(command.outputs[0]), out.tobytes())


def _run_reduction(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_reduction over the middle axis, summing in fp32.

    The mean is taken after the sum rather than incrementally, which is what the
    scalar and inside1 paths both do.
    """
    outside, reduce, inside = params[0], params[1], params[2]
    op_type, unit = params[3], params[4]
    if unit != FP16_BYTES:
        raise UnsupportedOp(
            f"blob: a reduction over {unit}-byte values is not modelled"
        )
    src = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)
    # Fewer elements than the slot holds is the run-time case, not a mistake: the
    # operand's ref is the exported bound, while a span the trailer patches
    # describes the part of it this call filled. More than the slot holds is the
    # direction that reads another tensor's bytes, and stays refused.
    if src.size < outside * reduce * inside:
        raise UnsupportedOp(
            f"blob: {src.size} values do not fill [{outside}][{reduce}][{inside}]"
        )
    rows = (
        src[: outside * reduce * inside]
        .reshape(outside, reduce, inside)
        .astype(np.float32)
    )
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


def _run_topk(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_topkv2_k1_fp16: each row's maximum, and the first index holding it.

    The kernel keeps the maximum's *bit pattern* over a row and then walks the
    row again for the first position whose bits equal it, so an index it writes
    is the first occurrence of the maximum -- which is not the position torch's
    own kernel produces (see `hexagon_ops.TOPK`). The values are fp16 per row and
    the indices int32 per row, whatever the graph declares the latter to be.
    """
    row_size, rows = params[0], params[1]
    src = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)
    if src.size < rows * row_size:
        raise UnsupportedOp(
            f"blob: {src.size} values do not fill [{rows}][{row_size}]"
        )
    row_major = np.ascontiguousarray(src[: rows * row_size].reshape(rows, row_size))
    # The vector walk is a vmax over the row and the scalar tail only replaces a
    # running maximum with something strictly greater, so a NaN never wins.
    ranked = row_major.astype(np.float32)
    ranked[np.isnan(row_major)] = -np.inf
    best = ranked.max(axis=1).astype(np.float16)
    first = (row_major.view(np.uint16) == best.view(np.uint16)[:, None]).argmax(axis=1)
    _store(arena, arena.address(command.outputs[0]), best.tobytes())
    _store(arena, arena.address(command.outputs[1]), first.astype(np.int32).tobytes())


def _run_flash_attn(command: Command, params: List[int], arena: Arena) -> None:
    """Sync attention, over the cache the op was handed rather than one it keeps.

    Query row q reads every cached position up to seq_current + q. The kernel
    applies that causally by clamping the row length, not by filling a mask with
    -inf, which is the same thing up to the softmax. Query head h reads key and
    value head h // gqa_factor. The cache arrives twice, as the new keys and as
    the past ones, because update_cache has already written the new rows by the
    time attention runs, so the kernel's push copies each row onto itself and
    this models the result rather than the copy.
    """
    qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim = params[:6]
    scale = struct.unpack("<f", struct.pack("<i", params[6]))[0]
    positions = seq_current + seq_add
    if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise UnsupportedOp(f"blob: {n_heads} heads over {n_kv_heads} kv heads")

    def rows(ref, heads):
        values = np.frombuffer(bytes(arena.view(ref)), dtype=np.float16)
        return values.reshape(-1, heads, head_dim).astype(np.float32)

    query = rows(command.inputs[0], n_heads)
    key = rows(command.inputs[4], n_kv_heads)
    value = rows(command.inputs[5], n_kv_heads)
    if query.shape[0] != qo_len or key.shape[0] < positions:
        raise UnsupportedOp(
            f"blob: {query.shape[0]} queries and a cache of {key.shape[0]} rows "
            f"do not cover {qo_len} over {positions}"
        )

    group = n_heads // n_kv_heads
    out = np.zeros((qo_len, n_heads, head_dim), dtype=np.float32)
    for row in range(qo_len):
        valid = min(seq_current + row + 1, positions)
        for head in range(n_heads):
            kv_head = head // group
            scores = (query[row, head] @ key[:valid, kv_head].T) * scale
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            out[row, head] = weights @ value[:valid, kv_head]
    _store(arena, arena.address(command.outputs[0]), out.astype(np.float16).tobytes())


def vision_attention_kernel_offset(
    batch_index: int,
    token: int,
    head: int,
    dimension: int,
    tokens: int,
    heads: int,
    head_dim: int,
) -> int:
    """The element `htp_ops_vision_attention_fp16` reads one row's value at.

    A transcription of the kernel rather than the inverse of a formula, so the
    two can disagree: `attention_entry.cc:36` makes a token's stride
    `heads * headDim`, `:41` and `:43` index the query and the key rows as
    `((b * tokens + q) * tokenStride) + h * headDim`, the value rows the same
    way at `:61`, and the output rows at `:57`. The dimension runs contiguously
    inside that. That is a token-major layout -- heads inside a token's row --
    and not the head-major one a batched matmul would read.
    """
    token_stride = heads * head_dim
    return ((batch_index * tokens + token) * token_stride) + head * head_dim + dimension


def vision_attention_row_major_offset(
    batch_index: int,
    token: int,
    head: int,
    dimension: int,
    tokens: int,
    heads: int,
    head_dim: int,
) -> int:
    """The same element, derived from the shape instead of from the source.

    A `[batch, tokens, heads, headDim]` tensor in row-major order holds its
    element `(b, t, h, d)` at the products of the extents to its right. Written
    that way it shares no expression with the transcription above.
    """
    return ((batch_index * tokens + token) * heads + head) * head_dim + dimension


def _run_vision_attention(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_vision_attention_fp16 (`attention_entry.cc:18-70`).

    Unmasked, non-causal attention over three tensors that are already
    token-major: every query row reads every key, which is what a vision tower
    computes and what a language model's causal attention does not. The mask
    operand is absent here -- bound to nothing, which is the pointer the
    dispatcher passes for a slot the command does not carry -- and the second
    output is the fp32 score row the kernel uses as scratch.

    The arithmetic is modelled rather than emulated: the kernel exponentiates
    with an HVX approximation (`hvx_my_exp2_vsf`) and accumulates a row in
    fp32, so a comparison against this has to be a tolerance, and the shape of
    the computation is what is being checked here.
    """
    batch, tokens, heads, head_dim = params[:4]
    scale = struct.unpack("<f", struct.pack("<i", params[4]))[0]
    mask_stride, workspace_bytes = params[5], params[6]
    if mask_stride > 0:
        raise UnsupportedOp("blob: a masked vision attention is not modelled")
    # The kernel refuses the command below this (`attention_entry.cc:26-29`),
    # because it aligns the buffer up by 127 bytes and then writes `tokens`
    # fp32 scores into it.
    if workspace_bytes < tokens * 4 + 127:
        raise UnsupportedOp(
            f"blob: a workspace of {workspace_bytes} bytes cannot hold "
            f"{tokens} fp32 scores"
        )
    count = batch * tokens * heads * head_dim
    refs = list(command.inputs) + list(command.outputs)
    if any(arena.view(ref).nbytes < count * FP16_BYTES for ref in refs[:4]):
        raise UnsupportedOp(
            f"blob: a vision attention operand is short of {count} values"
        )

    def rows(ref):
        values = np.frombuffer(bytes(arena.view(ref)), dtype=np.float16)
        out = np.zeros(count, dtype=np.float32)
        for b in range(batch):
            for token in range(tokens):
                for head in range(heads):
                    for dimension in range(head_dim):
                        at = vision_attention_kernel_offset(
                            b, token, head, dimension, tokens, heads, head_dim
                        )
                        out[
                            (b * tokens + token) * heads * head_dim
                            + head * head_dim
                            + dimension
                        ] = values[at]
        return out

    query = rows(refs[0]).reshape(batch * tokens, heads, head_dim)
    key = rows(refs[1]).reshape(batch * tokens, heads, head_dim)
    value = rows(refs[2]).reshape(batch * tokens, heads, head_dim)

    result = np.zeros((batch * tokens, heads, head_dim), dtype=np.float32)
    for row in range(batch * tokens):
        # The kernel's outer loop is over the batch (`attention_entry.cc:37`),
        # so a query row reads the keys of its own batch and no others.
        batch_index = row // tokens
        window = slice(batch_index * tokens, (batch_index + 1) * tokens)
        for head in range(heads):
            scores = (query[row, head] @ key[window, head].T) * scale
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            result[row, head] = weights @ value[window, head]

    flat = np.zeros(count, dtype=np.float16)
    for b in range(batch):
        for token in range(tokens):
            for head in range(heads):
                for dimension in range(head_dim):
                    flat[
                        vision_attention_kernel_offset(
                            b, token, head, dimension, tokens, heads, head_dim
                        )
                    ] = result[b * tokens + token, head, dimension]
    _store(arena, arena.address(command.outputs[0]), flat.tobytes())


def _unpack_vrmpy_int4(raw: bytes, k: int, n: int) -> np.ndarray:
    """The inverse of `pack_q4a16_gemv_weight`, back to a signed (n, k) int4.

    Tile (y, x) at `(y*icP + x)*512`; group g holds k = `x*32 + 4g + {0,1,2,3}`
    for the tile's 32 output channels, and byte `g*64 + ocIn*2 + p` packs
    k = `x*32 + 4g + 2p` low and `+1` high, each offset by 8.
    """
    kp, np_ = k // 32, n // 32
    tiles = np.frombuffer(raw[: kp * np_ * 512], dtype=np.uint8).reshape(
        np_, kp, 8, 32, 2
    )
    w = np.zeros((n, k), dtype=np.int32)
    for y in range(np_):
        for x in range(kp):
            for g in range(8):
                base = x * 32 + 4 * g
                for oc_in in range(32):
                    low = int(tiles[y, x, g, oc_in, 0])
                    high = int(tiles[y, x, g, oc_in, 1])
                    oc = y * 32 + oc_in
                    w[oc, base + 0] = (low & 0x0F) - 8
                    w[oc, base + 1] = (low >> 4) - 8
                    w[oc, base + 2] = (high & 0x0F) - 8
                    w[oc, base + 3] = (high >> 4) - 8
    return w


def _unpack_hmx_int8(raw: bytes, k: int, n: int) -> np.ndarray:
    """The inverse of `pack_w8a16_gemv_weight`, back to a signed (n, k) int8.

    Tile (oy, kx) at `(oy*kp + kx)*1024`; group g covers k = `kx*32 + 4g +
    {0,1,2,3}` for the tile's 32 output channels, and byte `g*128 + ocIn*4 + p`
    holds k = `kx*32 + 4g + perm[p]` with perm = {0, 2, 1, 3}.
    """
    kp, np_ = k // 32, n // 32
    tiles = np.frombuffer(raw[: kp * np_ * 1024], dtype=np.int8).reshape(
        np_, kp, 8, 32, 4
    )
    perm = (0, 2, 1, 3)
    w = np.zeros((n, k), dtype=np.int32)
    for oy in range(np_):
        for kx in range(kp):
            for g in range(8):
                base = kx * 32 + 4 * g
                for oc_in in range(32):
                    oc = oy * 32 + oc_in
                    for p in range(4):
                        w[oc, base + perm[p]] = int(tiles[oy, kx, g, oc_in, p])
    return w


def _quantize_activation_row(a: np.ndarray):
    """Per-token symmetric int8 quantization, as the GEMV kernels do it.

    `quantize_activation_row` in both kernels takes the row's absmax, narrows
    127/absmax to fp16, multiplies, narrows the product back to fp16, converts
    that to int16 round-to-nearest, and finally clamps to [-127, 127] because
    the int8 saturation would leave -128 in place. The two roundings are both
    modelled: fp16(a*inv) is exact in fp32, so the narrowing here is the one the
    kernel's Q6_Vhf_equals_Vqf16 does. This is an arithmetic model, not the
    kernel's instruction sequence -- the HVX maximum, its horizontal reduction
    and the saturation order are not reproduced, only what they compute.
    """
    x = a.astype(np.float32)
    absmax = float(np.max(np.abs(x))) if x.size else 0.0
    if absmax <= 0.0:
        absmax = 1.0
    inv = np.float16(127.0 / absmax)
    scaled = (x * np.float32(inv)).astype(np.float16).astype(np.float32)
    qa = np.rint(scaled)
    return np.clip(qa, -127, 127).astype(np.int32), np.float32(absmax / 127.0)


def _gemv_bias(command: Command, arena: Arena, index: int, n: int):
    if len(command.inputs) <= index:
        return None
    ref = command.inputs[index]
    if ref.space == ABSENT:
        return None
    return np.frombuffer(bytes(arena.view(ref)), dtype=np.float16)[:n].astype(
        np.float32
    )


def _run_matmul_q4a16_gemv(command: Command, params: List[int], arena: Arena) -> None:
    """htp_ops_matmul_q4a16_gemv_i8, one scale block per output channel.

    The weight is int4 in the vrmpy tile order with its fp32 scales appended,
    the activation is fp16 and quantized to int8 per token, and the int dot
    product is scaled once by the activation scale and the channel scale. The
    kernel's N-block machinery is only exercised at one block here, which is
    what the emitter writes.
    """
    k, n = params[1], params[2]
    nblk = params[8] if len(params) > 8 else 1
    if nblk != 1:
        raise UnsupportedOp(f"blob: a q4a16 gemv over {nblk} scale blocks")
    if k % 64 or n % 32:
        raise UnsupportedOp(f"blob: a q4a16 gemv of {k}x{n}")

    refs = list(command.inputs) + list(command.outputs)
    dst = refs[len(command.inputs)]
    raw = bytes(arena.view(command.inputs[1]))
    w = _unpack_vrmpy_int4(raw, k, n)
    tiles_bytes = (k // 32) * (n // 32) * 512
    scales = np.frombuffer(raw[tiles_bytes:], dtype=np.float32)[:n]

    a = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)[:k]
    qa, sa = _quantize_activation_row(a)
    acc = w @ qa
    out = (sa * scales * acc.astype(np.float32)).astype(np.float16)
    bias = _gemv_bias(command, arena, 2, n)
    if bias is not None:
        out = (out.astype(np.float32) + bias).astype(np.float16)
    _store(arena, arena.address(dst), out.tobytes())


def _run_matmul_w8a16_gemv(command: Command, params: List[int], arena: Arena) -> None:
    """hmx_matmulw8a16block_gemv_i8, one scale block per output channel.

    Same arithmetic as the q4a16 GEMV, but the weight is int8 in the HMX tile
    order and its fp32 scales are a separate operand.
    """
    k, n = params[1], params[2]
    nblk = params[8] if len(params) > 8 else 1
    if nblk != 1:
        raise UnsupportedOp(f"blob: a w8a16 gemv over {nblk} scale blocks")
    if k % 64 or n % 32:
        raise UnsupportedOp(f"blob: a w8a16 gemv of {k}x{n}")

    refs = list(command.inputs) + list(command.outputs)
    dst = refs[len(command.inputs)]
    w = _unpack_hmx_int8(bytes(arena.view(command.inputs[1])), k, n)
    scales = np.frombuffer(bytes(arena.view(command.inputs[2])), dtype=np.float32)[:n]

    a = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.float16)[:k]
    qa, sa = _quantize_activation_row(a)
    acc = w @ qa
    out = (sa * scales * acc.astype(np.float32)).astype(np.float16)
    bias = _gemv_bias(command, arena, 3, n)
    if bias is not None:
        out = (out.astype(np.float32) + bias).astype(np.float16)
    _store(arena, arena.address(dst), out.tobytes())


def untile_shared_gather(flat: np.ndarray, oc: int, ic: int) -> np.ndarray:
    """The fp16 table htp_ops_shared_gather reads, back to a row-major (oc, ic).

    A transcription of the kernel's own read rather than the inverse of a
    formula, so the two can disagree: the kernel walks the table as a grid of
    32x32 tiles with the column pairs inside a tile first, reading element
    (index, c) of tile (index // 32, c // 32) at
    ((c % 32) // 2) * 64 + (index % 32) * 2 + ((c % 32) & 1)
    (`shared_gather_ops.cc:296-311`). The last column tile is short whenever ic is
    not a multiple of 32, which is what the odd tail below stands for.
    """
    rows = np.zeros((oc, ic), dtype=np.float16)
    tile_columns = -(-ic // 32)
    for index in range(oc):
        tile_row, yi = divmod(index, 32)
        for x in range(tile_columns):
            base = (tile_row * tile_columns + x) * 32 * 32
            channels = min(ic - x * 32, 32)
            pairs = channels // 2
            for pair in range(pairs):
                source = base + pair * 64 + yi * 2
                rows[index, x * 32 + 2 * pair] = flat[source]
                rows[index, x * 32 + 2 * pair + 1] = flat[source + 1]
            if channels & 1:
                rows[index, x * 32 + channels - 1] = flat[base + pairs * 64 + yi * 2]
    return rows


def _run_shared_gather(command: Command, params: List[int], arena: Arena) -> None:
    """The fp16 path of htp_ops_shared_gather (`shared_gather_ops.cc:287-315`).

    selectSize rows are copied out of the table into as many output rows, and an
    index outside [0, oc) clears its row instead of failing -- the kernel's own
    answer, and not torch's: `embedding` raises there. The count and the table
    come from the graph while the index comes from the caller, so the only thing
    standing between the two is the vocabulary, which is why the tests pin this
    boundary down rather than assume it.
    """
    select_size, ic, oc, width, kind = params[:5]
    if kind != SHARED_GATHER_FP16:
        raise UnsupportedOp(f"blob: shared_gather table kind {kind} is not modelled")
    if width != FP16_BYTES:
        raise UnsupportedOp(f"blob: shared_gather writes {width}-byte elements")
    indices = np.frombuffer(bytes(arena.view(command.inputs[0])), dtype=np.int32)
    table = untile_shared_gather(
        np.frombuffer(bytes(arena.view(command.inputs[1])), dtype=np.float16), oc, ic
    )
    out = np.zeros((select_size, ic), dtype=np.float16)
    for row in range(select_size):
        index = int(indices[row])
        if 0 <= index < oc:
            out[row] = table[index]
    _store(arena, arena.address(command.outputs[0]), out.tobytes())


_EXECUTORS = {
    POOL2D_FP16: _run_pool2d,
    CONV_DEPTHWISE2D_FP16: _run_conv_depthwise2d,
    IM2COL_CONVOLUTION_FP16: _run_im2col_convolution,
    ZERO: _run_zero,
    SELECT: _run_select,
    RASTER_BLIT: _run_raster_blit,
    SOFTMAX: _run_softmax,
    REDUCTION: _run_reduction,
    TOPKV2_K1_FP16: _run_topk,
    UNARY: _run_unary,
    BINARY_ELEMENTWISE: _run_binary,
    LAYER_NORM: _run_layer_norm,
    ADD_FUSE_LAYERNORM: _run_add_fuse_layernorm,
    ROPE: _run_rope,
    BATCH_MATMUL: _run_batch_matmul,
    FLASH_ATTN: _run_flash_attn,
    MATMUL_Q4A16_GEMV_I8: _run_matmul_q4a16_gemv,
    MATMUL_W8A16_GEMV_I8: _run_matmul_w8a16_gemv,
    SHARED_GATHER: _run_shared_gather,
    VISION_ATTENTION_FP16: _run_vision_attention,
}


def execute(
    data: bytes,
    inputs: Sequence[np.ndarray],
    named_data: Optional[Dict[str, bytes]] = None,
    length: Optional[int] = None,
    arena: Optional[Arena] = None,
) -> List[np.ndarray]:
    """Runs a blob over the host arena and returns its outputs by index.

    `inputs` is indexed by method input index, which is the order the emitters
    declared their placeholders in. `named_data` is the named data map a blob
    with external weights needs, keyed the way the trailer names its entries.

    `length` is the run-time sequence length a dynamic blob's trailer patches are
    computed from. Left out, it is read off the shape of the operand the trailer
    names, which is what the runtime does; it is stated only when the operand
    handed over is the whole bound-sized slot an arena holds rather than the
    tensor a caller would pass.

    `arena` is the arena to run in, for a caller that wants to read something
    back out of it afterwards -- an activation an emitter filled and no output
    names, which is the only way to see what a kernel wrote into a slot the graph
    does not ask for. It has to be built over the same bytes and the same named
    data, since those are what the caller would otherwise have supplied here.
    """
    header, commands = read_blob(data)
    if len(inputs) != header.n_inputs:
        raise ValueError(f"blob wants {header.n_inputs} inputs, got {len(inputs)}")

    if arena is None:
        arena = Arena(header, data, "blob", named_data)
    for index, tensor in enumerate(inputs):
        ref = _slot(commands, B.TensorSpace.INPUT, index)
        raw = tensor.tobytes()
        if len(raw) != ref.size:
            raise ValueError(
                f"input {index} is {len(raw)} bytes, the blob expects {ref.size}"
            )
        _copy(arena, arena.address(ref), arena.address(ref), 0)
        arena.bytes[arena.address(ref) : arena.address(ref) + ref.size] = raw

    rewritten = _length_patches(data, inputs, length)
    for index, command in enumerate(commands):
        executor = _EXECUTORS.get(command.type)
        if executor is None:
            raise UnsupportedOp(
                f"blob: op type {command.type} is not modelled on the host"
            )
        params = _patched(command, arena)
        for param_index, value in rewritten.get(index, {}).items():
            if param_index >= len(params):
                raise ValueError(
                    f"hexagon: command {index} carries {len(params)} params and "
                    f"the trailer patches slot {param_index}"
                )
            params[param_index] = value
        executor(command, params, arena)

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
