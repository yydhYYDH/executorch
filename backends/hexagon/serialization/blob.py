# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serializer for the Hexagon delegate blob.

Mirrors backends/hexagon/serialization/hexagon_schema.h field for field. The
runtime reads this once, in HexagonBackend::init, and turns each op into a
FlatBuffers command descriptor, so the two layouts have to agree exactly.
"""

import hashlib
import struct
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
from executorch.exir.sym_util import eval_upper_bound

BLOB_MAGIC = 0x4E584748  # 'HXGN'
# Bumped with the HexagonOp layout, which patch_scale changed from 476 bytes to
# 480. A v1 blob read as v2 would take an in_place bit for a scale.
BLOB_VERSION = 2
# A blob that leaves weights behind when it is copied out of the .pte. The two
# share a layout up to the end of the weights section; what follows it says
# where the rest of the weights live, so an all-inline blob stays v2 and stays
# byte-for-byte what it was before this existed.
BLOB_VERSION_EXTERNAL_WEIGHTS = 3
DYNAMIC_TRAILER_MAGIC = 0x44594E48  # 'HYND'
EXTERNAL_WEIGHTS_MAGIC = 0x57455848  # 'HXEW'
EXTERNAL_WEIGHTS_VERSION = 1
ALIGNMENT = 128
MAX_OP_INPUTS = 8
MAX_OP_OUTPUTS = 4
# BATCH_MATMUL carries a packed HtpOpsLoopParam in its params, which needs 26.
MAX_OP_PARAMS = 40
# The trailer names each external weight by a NUL-padded fixed-width key, so a
# key the runtime can look up is at most 31 bytes.
EXTERNAL_KEY_BYTES = 32
EXTERNAL_KEY_MAX = EXTERNAL_KEY_BYTES - 1

_DYNAMIC_HEADER = struct.Struct("<6I")
_DYNAMIC_HEADER_V3 = struct.Struct("<7I")
_DYNAMIC_PATCH = struct.Struct("<4i")
_DYNAMIC_LAYOUT = struct.Struct("<IIqqq")
# magic, version, n_ext, reserved, weights_bytes
_EXTERNAL_HEADER = struct.Struct("<4IQ")
_EXTERNAL_WEIGHT = struct.Struct("<2Q" + f"{EXTERNAL_KEY_BYTES}s")


class TensorSpace(IntEnum):
    """Must match executorch::backends::hexagon::HexagonTensorSpace."""

    WEIGHTS = 0
    INPUT = 1
    OUTPUT = 2
    ACTIVATION = 3
    # An operand that is not there. The runtime sends it as fd = -1, which the
    # DSP maps to a null pointer; a zero-size tensor is still a live address.
    ABSENT = 0xFFFFFFFF


_HEADER = struct.Struct("<9I")
_TENSOR_REF = struct.Struct("<IIQQ")
_OP_PREFIX = struct.Struct("<IIII" + "i" * MAX_OP_PARAMS + "IIII")

#: patch slot meaning the emitter already knows every param.
NO_PATCH = 0xFFFFFFFF
# Cross-checked against the C++ sizeof(HexagonOp) by
# test/test_blob_roundtrip.py, which is also what the static_asserts in
# hexagon_schema.h pin down from the other side.
OP_SIZE = _OP_PREFIX.size + (MAX_OP_INPUTS + MAX_OP_OUTPUTS) * _TENSOR_REF.size
HEADER_SIZE = _HEADER.size


@dataclass(frozen=True)
class TensorRef:
    """Where a tensor lives: a section, an offset within it, and its size."""

    space: TensorSpace
    offset: int
    size: int
    index: int = 0

    def pack(self) -> bytes:
        return _TENSOR_REF.pack(int(self.space), self.index, self.offset, self.size)


#: Padding for unused input/output slots. Never read, since the DSP walks only
#: the first n_inputs/n_outputs entries.
NULL_REF = TensorRef(TensorSpace.ABSENT, 0, 0)

#: Pass where an op takes an optional operand, such as a bias or a gamma.
ABSENT = TensorRef(TensorSpace.ABSENT, 0, 0)


@dataclass
class Op:
    """One DSP command.

    type is a DSPOpType from the vendored headers; params is that op's
    parameter vector, whose meaning is defined per op in execute_command.cc.
    """

    type: int
    inputs: List[TensorRef]
    outputs: List[TensorRef]
    params: List[int] = field(default_factory=list)
    # (param index, input index): before issuing the command the runtime
    # overwrites that param with the first element of that input. Some DSP
    # values, such as a KV cache position, only exist once the graph runs.
    patch: Optional[Tuple[int, int]] = None
    # Multiplies the patched value on its way into the param, so a token
    # position can index a row of a multi-headed cache.
    patch_scale: int = 1
    # Bit j marks inputs[j] as mutated in place, so the runtime copies it back
    # to the caller once the command group has run.
    in_place: int = 0

    def pack(self) -> bytes:
        if len(self.inputs) > MAX_OP_INPUTS:
            raise ValueError(
                f"op {self.type}: {len(self.inputs)} inputs, max {MAX_OP_INPUTS}"
            )
        if len(self.outputs) > MAX_OP_OUTPUTS:
            raise ValueError(
                f"op {self.type}: {len(self.outputs)} outputs, max {MAX_OP_OUTPUTS}"
            )
        if len(self.params) > MAX_OP_PARAMS:
            raise ValueError(
                f"op {self.type}: {len(self.params)} params, max {MAX_OP_PARAMS}"
            )

        params = list(self.params) + [0] * (MAX_OP_PARAMS - len(self.params))
        refs = list(self.inputs) + [NULL_REF] * (MAX_OP_INPUTS - len(self.inputs))
        refs += list(self.outputs) + [NULL_REF] * (MAX_OP_OUTPUTS - len(self.outputs))

        patch_param, patch_input = self.patch if self.patch else (NO_PATCH, 0)

        return _OP_PREFIX.pack(
            self.type,
            len(self.inputs),
            len(self.outputs),
            len(self.params),
            *params,
            patch_param,
            patch_input,
            self.patch_scale,
            self.in_place,
        ) + b"".join(ref.pack() for ref in refs)


@dataclass(frozen=True)
class DynamicPatch:
    """A command parameter derived from the runtime sequence length."""

    op_index: int
    param_index: int
    scale: int = 1
    add: int = 0

    def pack(self) -> bytes:
        return _DYNAMIC_PATCH.pack(
            self.op_index, self.param_index, self.scale, self.add
        )


@dataclass(frozen=True)
class DynamicLayout:
    """Bytes for one tensor as an affine/quadratic function of sequence L."""

    space: TensorSpace
    index: int
    c0: int
    c1: int = 0
    c2: int = 0

    def pack(self) -> bytes:
        return _DYNAMIC_LAYOUT.pack(
            int(self.space), self.index, self.c0, self.c1, self.c2
        )


def external_weight_key(data: bytes) -> str:
    """The key a weight stored outside the .pte is looked up under.

    Derived from the bytes, so the same weight exported twice lands on one key
    and the store keeps one copy of it. Thirty-one characters is what the
    trailer's fixed-width key field holds; twenty-nine hex digits of SHA-256
    distinguishes far more weights than any model has, and a collision is a hard
    error in the store rather than a silent mix-up.
    """
    digest = hashlib.sha256(data).hexdigest()[: EXTERNAL_KEY_MAX - 2]
    return "hx" + digest


@dataclass(frozen=True)
class ExternalWeight:
    """One weight the .pte does not carry.

    `offset` is measured from the start of the arena's weights section, which
    the runtime builds as the bytes from the file followed by these entries, so
    it is always past `HexagonBlobHeader::weights_bytes`.
    """

    offset: int
    size: int
    key: str
    #: Index into the builder's weight list, so the store can be handed the
    #: bytes without the trailer having to carry them. Not part of the format,
    #: the way TensorRef::index is not.
    index: int = 0

    def pack(self) -> bytes:
        key = self.key.encode("ascii")
        if not key or len(key) > EXTERNAL_KEY_MAX:
            raise ValueError(
                f"external weight key {self.key!r} needs 1 to {EXTERNAL_KEY_MAX} bytes"
            )
        return _EXTERNAL_WEIGHT.pack(self.offset, self.size, key)


@dataclass(frozen=True)
class ExternalWeights:
    """The trailer listing them, written only when there are any."""

    #: Bytes the arena reserves for the weights section: the file's own inline
    #: bytes plus every entry below. Never a length of the file.
    weights_bytes: int
    entries: List[ExternalWeight]

    def pack(self) -> bytes:
        trailer = _EXTERNAL_HEADER.pack(
            EXTERNAL_WEIGHTS_MAGIC,
            EXTERNAL_WEIGHTS_VERSION,
            len(self.entries),
            0,
            self.weights_bytes,
        )
        return trailer + b"".join(entry.pack() for entry in self.entries)


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


class BlobBuilder:
    """Collects ops and the four tensor sections into one blob.

    Sections are accumulated independently and packed back to back at build
    time, so a caller never reasons about arena offsets. Only the weights are
    written out: the activation section is a size the runtime reserves for
    itself, and the input and output sections are sizes it copies to and from
    the caller, so neither is a region on disk.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        external_weights_max_bytes: Optional[int] = None,
    ) -> None:
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        # A weight larger than this is stored outside the .pte. None keeps every
        # weight inline, which is the format as it was.
        self._external_weights_max_bytes = external_weights_max_bytes
        self._ops: List[Op] = []
        self._weight_data: List[bytes] = []
        self._activations_bytes = 0
        self._activation_sizes: Dict[int, int] = {}
        self._activation_offsets: Dict[int, int] = {}
        self._activation_virtual: Dict[int, int] = {}
        self._next_activation_id = 0
        self._virtual_activation_bytes = 0
        self._input_slots: Dict[int, TensorRef] = {}
        self._output_slots: Dict[int, TensorRef] = {}
        self._dynamic_input: Optional[Tuple[int, int, int]] = None
        self._dynamic_example: Optional[int] = None
        self._dynamic_patches: List[DynamicPatch] = []
        self._dynamic_layouts: Dict[Tuple[TensorSpace, int], DynamicLayout] = {}
        # Filled in by _pack_weights, which is where the sizes are final.
        self._weights_disk_bytes = 0
        self._weights_arena_bytes = 0
        self._external_weights: List[ExternalWeight] = []

    def set_dynamic_sequence(
        self, input_index: int, axis: int, max_length: int, example_length: int = 0
    ) -> None:
        if input_index < 0 or input_index >= self._n_inputs:
            raise ValueError(f"dynamic input index {input_index} out of range")
        if axis < 0 or max_length <= 0:
            raise ValueError("dynamic sequence axis and max length must be positive")
        self._dynamic_input = (input_index, axis, max_length)
        self._dynamic_example = int(example_length)

    def add_dynamic_patch(self, patch: DynamicPatch) -> None:
        if patch.op_index < 0 or patch.param_index < 0:
            raise ValueError(f"invalid dynamic patch {patch}")
        self._dynamic_patches.append(patch)

    def add_dynamic_layout(self, layout: DynamicLayout) -> None:
        if layout.c0 < 0 or layout.c1 < 0 or layout.c2 < 0:
            raise ValueError(
                f"dynamic layout coefficients must be non-negative: {layout}"
            )
        key = (layout.space, layout.index)
        previous = self._dynamic_layouts.get(key)
        if previous is not None and previous != layout:
            raise ValueError(f"conflicting dynamic layouts for {key}")
        self._dynamic_layouts[key] = layout

    @staticmethod
    def _screen_slot(slots: Dict[int, TensorRef], ref: TensorRef, what: str) -> None:
        existing = slots.get(ref.index)
        if existing is not None and existing.size != ref.size:
            raise ValueError(
                f"{what} {ref.index} is used as both {existing.size} and {ref.size} bytes"
            )
        slots.setdefault(ref.index, ref)

    def add_op(self, op: Op) -> int:
        for ref in op.inputs:
            if ref.space == TensorSpace.INPUT:
                if ref.index >= self._n_inputs:
                    raise ValueError(f"input index {ref.index} out of range")
                self._screen_slot(self._input_slots, ref, "input")
            elif ref.space == TensorSpace.OUTPUT:
                if ref.index >= self._n_outputs:
                    raise ValueError(f"output index {ref.index} out of range")
                self._screen_slot(self._output_slots, ref, "output")
        self._ops.append(op)
        return len(self._ops) - 1

    def add_weights(self, data: bytes, alignment: int = ALIGNMENT) -> TensorRef:
        """Keeps constant data and returns the reference the DSP will follow.

        The offset is assigned at build time, not here: a weight an emitter
        computed and then did not use -- the plain copy of a weight that a
        second pass stored in another order, say -- costs the file nothing.
        """
        index = len(self._weight_data)
        self._weight_data.append(data)
        return TensorRef(TensorSpace.WEIGHTS, 0, len(data), index)

    def add_activation(
        self,
        nbytes: int,
        alignment: int = ALIGNMENT,
        dynamic_layout: Optional[Tuple[int, int, int]] = None,
    ) -> TensorRef:
        """Declare scratch space for an intermediate tensor.

        Physical offsets are assigned in ``build`` after op lifetimes are known,
        so tensors whose uses do not overlap can share one activation block.
        """
        if isinstance(nbytes, torch.SymInt):
            nbytes = eval_upper_bound(nbytes)
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("activation size must be non-negative")
        index = self._next_activation_id
        self._next_activation_id += 1
        self._activation_sizes[index] = nbytes
        if dynamic_layout is not None:
            self.add_dynamic_layout(
                DynamicLayout(TensorSpace.ACTIVATION, index, *dynamic_layout)
            )
        virtual_offset = _align_up(self._virtual_activation_bytes, alignment)
        self._virtual_activation_bytes = virtual_offset + nbytes
        self._activation_virtual[index] = virtual_offset
        return TensorRef(TensorSpace.ACTIVATION, virtual_offset, nbytes, index)

    def _pack_activations(self) -> None:
        """Assign offsets using non-overlapping op live ranges."""
        lifetimes: Dict[int, List[int]] = {}
        for op_index, op in enumerate(self._ops):
            for ref in op.inputs + op.outputs:
                if ref.space != TensorSpace.ACTIVATION:
                    continue
                live = lifetimes.setdefault(ref.index, [op_index, op_index])
                live[0] = min(live[0], op_index)
                live[1] = max(live[1], op_index)

        # An allocation is reusable once its last consumer is before the next
        # allocation's producer. Reuse the smallest suitable free block to keep
        # fragmentation bounded without introducing a second packing pass.
        active: List[Tuple[int, int, int]] = []  # (last use, offset, size)
        free: List[Tuple[int, int]] = []  # (offset, size)
        arena_end = 0
        for index, (first, last) in sorted(
            lifetimes.items(), key=lambda item: (item[1][0], item[0])
        ):
            still_active: List[Tuple[int, int, int]] = []
            for active_last, offset, size in active:
                if active_last < first:
                    free.append((offset, size))
                else:
                    still_active.append((active_last, offset, size))
            active = still_active

            requested = self._activation_sizes[index]
            candidate = min(
                (block for block in free if block[1] >= requested),
                key=lambda block: block[1],
                default=None,
            )
            if candidate is None:
                tail = next(
                    (block for block in free if block[0] + block[1] == arena_end),
                    None,
                )
                if tail is not None:
                    offset, _ = tail
                    free.remove(tail)
                    arena_end = offset + requested
                else:
                    offset = _align_up(arena_end)
                    arena_end = offset + requested
            else:
                offset, _ = candidate
                free.remove(candidate)
            self._activation_offsets[index] = offset
            active.append((last, offset, requested))

        self._activations_bytes = arena_end

    def method_input(
        self,
        index: int,
        nbytes: int,
        dynamic_layout: Optional[Tuple[int, int, int]] = None,
    ) -> TensorRef:
        """Declares the slot for method input index."""
        ref = TensorRef(TensorSpace.INPUT, 0, nbytes, index)
        self._screen_slot(self._input_slots, ref, "input")
        self._input_slots[index] = ref
        if dynamic_layout is not None:
            self.add_dynamic_layout(
                DynamicLayout(TensorSpace.INPUT, index, *dynamic_layout)
            )
        return ref

    def method_output(
        self,
        index: int,
        nbytes: int,
        dynamic_layout: Optional[Tuple[int, int, int]] = None,
    ) -> TensorRef:
        """Declares the slot for method output index."""
        ref = TensorRef(TensorSpace.OUTPUT, 0, nbytes, index)
        self._screen_slot(self._output_slots, ref, "output")
        self._output_slots[index] = ref
        if dynamic_layout is not None:
            self.add_dynamic_layout(
                DynamicLayout(TensorSpace.OUTPUT, index, *dynamic_layout)
            )
        return ref

    def _pack_section(self, slots: Dict[int, TensorRef], count: int, what: str) -> int:
        """Assigns each slot a packed offset and returns the section size."""
        offset = 0
        for index in range(count):
            ref = slots.get(index)
            if ref is None:
                raise ValueError(f"method {what} {index} has no slot")
            slots[index] = TensorRef(ref.space, offset, ref.size, index)
            offset = _align_up(offset + ref.size)
        return offset

    def _pack_weights(self) -> Tuple[bytes, Dict[int, int]]:
        """Lays out the weights a command actually reads and returns the section.

        Order follows first use, which is what keeps a delegate's weights in the
        order its ops were emitted in.

        A weight too large to belong in the .pte is not written to the section;
        it is appended past it instead, in the arena the runtime builds, and the
        trailer records where. That leaves the inline entries at exactly the
        offsets they would have had, so the section in the file is a prefix of
        the section in the arena and the runtime can copy both halves without a
        map between them.
        """
        used: List[int] = []
        seen: set = set()
        for op in self._ops:
            for ref in op.inputs + op.outputs:
                if ref.space == TensorSpace.WEIGHTS and ref.index not in seen:
                    seen.add(ref.index)
                    used.append(ref.index)

        inline: List[int] = []
        external: List[int] = []
        for index in used:
            data = self._weight_data[index]
            if (
                self._external_weights_max_bytes is not None
                and len(data) > self._external_weights_max_bytes
            ):
                external.append(index)
            else:
                inline.append(index)

        offsets: Dict[int, int] = {}
        data = bytearray()
        for index in inline:
            offset = _align_up(len(data))
            data.extend(b"\x00" * (offset - len(data)))
            data.extend(self._weight_data[index])
            offsets[index] = offset
        self._weights_disk_bytes = len(data)

        self._external_weights = []
        cursor = _align_up(self._weights_disk_bytes)
        for index in external:
            blob = self._weight_data[index]
            offset = _align_up(cursor)
            offsets[index] = offset
            cursor = offset + len(blob)
            self._external_weights.append(
                ExternalWeight(
                    offset, len(blob), external_weight_key(blob), index=index
                )
            )
        self._weights_arena_bytes = _align_up(cursor)
        return bytes(data), offsets

    def external_weight_data(self) -> List[Tuple[str, bytes]]:
        """The externalized weights as (key, bytes), for the .ptd store.

        One entry per weight rather than one per key: two weights that are equal
        bytes share a key, and the store turns those into one buffer.
        """
        return [
            (entry.key, self._weight_data[entry.index])
            for entry in self._external_weights
        ]

    def _remap(self, ref: TensorRef) -> TensorRef:
        if ref.space == TensorSpace.INPUT:
            return self._input_slots[ref.index]
        if ref.space == TensorSpace.OUTPUT:
            return self._output_slots[ref.index]
        if ref.space == TensorSpace.WEIGHTS:
            return TensorRef(
                TensorSpace.WEIGHTS,
                self._weight_offsets[ref.index],
                ref.size,
                ref.index,
            )
        if ref.space == TensorSpace.ACTIVATION:
            # A ref that points into part of its allocation -- one slice of a
            # result an emitter fills in several commands -- keeps the offset
            # it was handed relative to the block the packer assigns. Replacing
            # the offset outright would fold every slice onto the block start.
            virtual = self._activation_virtual.get(ref.index)
            within = 0 if virtual is None else ref.offset - virtual
            return TensorRef(
                TensorSpace.ACTIVATION,
                self._activation_offsets[ref.index] + within,
                ref.size,
                ref.index,
            )
        return ref

    def _record_default_dynamic_layouts(self) -> None:
        """Record max-size constants for tensors without a dynamic formula."""
        for index, size in self._activation_sizes.items():
            self._dynamic_layouts.setdefault(
                (TensorSpace.ACTIVATION, index),
                DynamicLayout(TensorSpace.ACTIVATION, index, size),
            )
        for index, ref in self._input_slots.items():
            self._dynamic_layouts.setdefault(
                (TensorSpace.INPUT, index),
                DynamicLayout(TensorSpace.INPUT, index, ref.size),
            )
        for index, ref in self._output_slots.items():
            self._dynamic_layouts.setdefault(
                (TensorSpace.OUTPUT, index),
                DynamicLayout(TensorSpace.OUTPUT, index, ref.size),
            )

    def build(self) -> bytes:
        self._pack_activations()
        inputs_bytes = self._pack_section(self._input_slots, self._n_inputs, "input")
        outputs_bytes = self._pack_section(
            self._output_slots, self._n_outputs, "output"
        )
        weights, self._weight_offsets = self._pack_weights()
        external = ExternalWeights(self._weights_arena_bytes, self._external_weights)

        # rebuild via replace() so every other field survives automatically:
        # listing them by hand silently dropped patch and in_place.
        ops = b"".join(
            replace(
                op,
                inputs=[self._remap(ref) for ref in op.inputs],
                outputs=[self._remap(ref) for ref in op.outputs],
            ).pack()
            for op in self._ops
        )

        header = _HEADER.pack(
            BLOB_MAGIC,
            BLOB_VERSION_EXTERNAL_WEIGHTS if external.entries else BLOB_VERSION,
            len(self._ops),
            self._n_inputs,
            self._n_outputs,
            self._weights_disk_bytes,
            inputs_bytes,
            self._activations_bytes,
            outputs_bytes,
        )
        body = header + ops + weights
        if external.entries:
            body += external.pack()
        if self._dynamic_input is None:
            return body
        self._record_default_dynamic_layouts()
        input_index, axis, max_length = self._dynamic_input
        trailer = _DYNAMIC_HEADER_V3.pack(
            DYNAMIC_TRAILER_MAGIC,
            3,
            input_index,
            axis,
            max_length,
            len(self._dynamic_patches),
            self._dynamic_example or 0,
        ) + b"".join(patch.pack() for patch in self._dynamic_patches)
        trailer += struct.pack("<I", len(self._dynamic_layouts))
        trailer += b"".join(layout.pack() for layout in self._dynamic_layouts.values())
        return body + trailer
