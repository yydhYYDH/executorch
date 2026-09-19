/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

// On-disk format of a Hexagon delegated subgraph, as emitted by the AOT
// preprocess step and consumed by init().
//
// The DSP never reads this: init() translates it into the FlatBuffers command
// format the skel expects. Keeping the AOT format separate means the Python
// side needs no flatbuffers codegen, and the wire format the skel sees stays
// stable even if the AOT format changes.
//
// Little-endian, packed. Section offsets inside a HexagonTensorRef are relative
// to the start of the section named by its space, and init() places the
// sections in the arena in the order they appear in HexagonBlobHeader.

namespace executorch::backends::hexagon {

constexpr uint32_t kHexagonBlobMagic = 0x4E584748; // 'HXGN'
// Bumped with the HexagonOp layout, which patch_scale changed from 476 bytes
// to 480. A v1 blob read as v2 would take an in_place bit for a scale.
constexpr uint32_t kHexagonBlobVersion = 2;

constexpr uint32_t kMaxOpInputs = 8;
constexpr uint32_t kMaxOpOutputs = 4;
// BATCH_MATMUL carries a packed HtpOpsLoopParam in its params, which needs 26.
constexpr uint32_t kMaxOpParams = 40;
constexpr size_t kHexagonAlignment = 128;

// Sentinel patch_param: every param is already known at export time.
constexpr uint32_t kNoOpPatch = 0xFFFFFFFF;

// Which arena section a tensor lives in. Each section is one contiguous run,
// which is what keeps a tensor's address computable from (space, offset) alone.
enum class HexagonTensorSpace : uint32_t {
  // Packed weights, resident for the whole session.
  kWeights = 0,
  // A method input, copied in at the start of execute().
  kInput = 1,
  // A method output, copied out at the end of execute().
  kOutput = 2,
  // Scratch produced and consumed inside the subgraph.
  kActivation = 3,
  // The operand is absent, which the DSP is told with fd = -1 and reads as a
  // null pointer. This is the only way to say "no bias", "no gamma": a
  // zero-size tensor still maps to a live address and gets read as data.
  kAbsent = 0xFFFFFFFF,
};

#pragma pack(push, 1)

struct HexagonTensorRef {
  uint32_t space; // HexagonTensorSpace
  uint32_t index; // which method input/output; unused for weights/activations
  uint64_t offset; // byte offset from the start of that space's section
  uint64_t size; // byte size
};

struct HexagonOp {
  uint32_t type; // DSPOpType
  uint32_t n_inputs;
  uint32_t n_outputs;
  uint32_t n_params;
  int32_t params[kMaxOpParams];
  // Before issuing the command the runtime overwrites params[patch_param] with
  // the first element of inputs[patch_input] times patch_scale, or leaves
  // params alone when patch_param is kNoOpPatch.
  uint32_t patch_param;
  uint32_t patch_input;
  // A token position indexes a row of a multi-headed cache, and a row is a
  // whole number of operands, so the position reaches the param scaled. One
  // leaves the value alone.
  uint32_t patch_scale;
  // Bit j is set when inputs[j] is mutated in place and has to be copied back
  // to the caller after the command group runs.
  uint32_t in_place;
  HexagonTensorRef inputs[kMaxOpInputs];
  HexagonTensorRef outputs[kMaxOpOutputs];
};

struct HexagonBlobHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t n_ops;
  uint32_t n_inputs; // method inputs, in signature order
  uint32_t n_outputs; // method outputs, in signature order
  uint32_t weights_bytes; // on disk, right after the ops
  uint32_t inputs_bytes; // arena sizes, not regions of the blob
  uint32_t activations_bytes;
  uint32_t outputs_bytes;
};

#pragma pack(pop)

// The Python writer in serialization/blob.py derives the same numbers from its
// own struct formats, and test/test_blob_roundtrip.py compares them field by
// field. Asserting them here turns a layout change into a build failure instead
// of a blob the runtime silently misreads.
static_assert(sizeof(HexagonBlobHeader) == 36);
static_assert(sizeof(HexagonTensorRef) == 24);
static_assert(sizeof(HexagonOp) == 480);
static_assert(offsetof(HexagonOp, params) == 16);
static_assert(offsetof(HexagonOp, patch_param) == 176);
static_assert(offsetof(HexagonOp, patch_scale) == 184);
static_assert(offsetof(HexagonOp, in_place) == 188);
static_assert(offsetof(HexagonOp, inputs) == 192);

// Layout: header, then n_ops of HexagonOp, then the weights, 128-byte aligned.
// The other three sections are sizes the header carries and the runtime uses to
// lay out its own arenas; only the weights are on disk, because they are the
// only section it copies out of the blob.

} // namespace executorch::backends::hexagon
