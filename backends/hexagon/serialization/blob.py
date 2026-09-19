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

import struct
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

BLOB_MAGIC = 0x4E584748  # 'HXGN'
# Bumped with the HexagonOp layout, which patch_scale changed from 476 bytes to
# 480. A v1 blob read as v2 would take an in_place bit for a scale.
BLOB_VERSION = 2
ALIGNMENT = 128
MAX_OP_INPUTS = 8
MAX_OP_OUTPUTS = 4
# BATCH_MATMUL carries a packed HtpOpsLoopParam in its params, which needs 26.
MAX_OP_PARAMS = 40


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

        return (
            _OP_PREFIX.pack(
                self.type,
                len(self.inputs),
                len(self.outputs),
                len(self.params),
                *params,
                patch_param,
                patch_input,
                self.patch_scale,
                self.in_place,
            )
            + b"".join(ref.pack() for ref in refs)
        )


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

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._ops: List[Op] = []
        self._weights = bytearray()
        self._activations_bytes = 0
        self._input_slots: Dict[int, TensorRef] = {}
        self._output_slots: Dict[int, TensorRef] = {}

    @staticmethod
    def _screen_slot(slots: Dict[int, TensorRef], ref: TensorRef, what: str) -> None:
        existing = slots.get(ref.index)
        if existing is not None and existing.size != ref.size:
            raise ValueError(
                f"{what} {ref.index} is used as both {existing.size} and {ref.size} bytes"
            )
        slots.setdefault(ref.index, ref)

    def add_op(self, op: Op) -> None:
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

    def add_weights(self, data: bytes, alignment: int = ALIGNMENT) -> TensorRef:
        """Appends constant data and returns where the DSP will find it."""
        offset = _align_up(len(self._weights), alignment)
        self._weights.extend(b"\x00" * (offset - len(self._weights)))
        self._weights.extend(data)
        return TensorRef(TensorSpace.WEIGHTS, offset, len(data))

    def add_activation(self, nbytes: int, alignment: int = ALIGNMENT) -> TensorRef:
        """Bump-allocates scratch space for an intermediate tensor.

        The bytes are not kept. They are all zero by construction, the runtime
        sizes the section from the header without reading it, and a padding
        region materialised here is duplicated into the file by every delegate
        that shares the tensor layout -- 810 MB of zeros for the vision tower,
        none of which anything reads.
        """
        offset = _align_up(self._activations_bytes, alignment)
        self._activations_bytes = offset + nbytes
        return TensorRef(TensorSpace.ACTIVATION, offset, nbytes)

    def method_input(self, index: int, nbytes: int) -> TensorRef:
        """Declares the slot for method input index."""
        ref = TensorRef(TensorSpace.INPUT, 0, nbytes, index)
        self._screen_slot(self._input_slots, ref, "input")
        self._input_slots[index] = ref
        return ref

    def method_output(self, index: int, nbytes: int) -> TensorRef:
        """Declares the slot for method output index."""
        ref = TensorRef(TensorSpace.OUTPUT, 0, nbytes, index)
        self._screen_slot(self._output_slots, ref, "output")
        self._output_slots[index] = ref
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

    def _remap(self, ref: TensorRef) -> TensorRef:
        if ref.space == TensorSpace.INPUT:
            return self._input_slots[ref.index]
        if ref.space == TensorSpace.OUTPUT:
            return self._output_slots[ref.index]
        return ref

    def build(self) -> bytes:
        inputs_bytes = self._pack_section(self._input_slots, self._n_inputs, "input")
        outputs_bytes = self._pack_section(self._output_slots, self._n_outputs, "output")

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
            BLOB_VERSION,
            len(self._ops),
            self._n_inputs,
            self._n_outputs,
            len(self._weights),
            inputs_bytes,
            self._activations_bytes,
            outputs_bytes,
        )
        return header + ops + bytes(self._weights)
