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
#include <cstdlib>
#include <cstring>

#include <executorch/backends/hexagon/serialization/hexagon_schema.h>

// What the compile specs claim about a blob, checked against what the blob
// actually holds.
//
// Some hexagon tunables change the bytes of the blob -- whether matmul weights
// are stored in the HMX unit's tile order, whether attention goes through the
// paged entry, whether weights live outside the .pte -- so they are compile
// specs rather than runtime options, and they ride in the .pte with the blob.
// Being written down is not the same as being true, though: the file is the
// only thing the DSP reads, and a spec that disagrees with the bytes it
// describes is a program that runs a layout it was never compiled for. So
// init() re-derives each of these facts from the blob and compares.
//
// Deliberately free of ExecuTorch, FlatBuffers and rpcmem: the check is a
// function of the blob's bytes and a list of key/value pairs, and keeping it
// that way is what lets test/test_compile_specs.py compile this file with the
// host compiler and cover the failure paths offline, which is the only way they
// get covered at all on a machine that cannot build the backend.
//
// The one ExecuTorch thing this file does not model is logging: a verdict
// carries a sentence for the caller to ET_LOG.
namespace executorch::backends::hexagon {

// The compile spec keys this backend defines. The same strings are the keys in
// hexagon_backend.py's HexagonCompileOptions.
constexpr char kHmxPrepackSpecKey[] = "hexagon_hmx_prepack";
constexpr char kAttnPagedSpecKey[] = "hexagon_attn_paged";
constexpr char kExternalWeightsMaxBytesSpecKey[] =
    "hexagon_external_weights_max_bytes";

// DSPOpType values the checks below read. BATCH_MATMUL's params carry the loop
// descriptor, whose hmxFlags word says whether the weight is tiled; FLASH_ATTN's
// page_size is nonzero only on the paged entry point.
constexpr uint32_t kBatchMatMulType = 38;
constexpr uint32_t kFlashAttnType = 18;
// Param index of hmxFlags in a BATCH_MATMUL's params, and of page_size in a
// FLASH_ATTN's. Both are positional, as every param in this format is. The
// tile budget sits in the word after hmxFlags, which is where a runtime that
// caps the unit's resident tiles overwrites it.
constexpr uint32_t kMatmulHmxFlagsParam = 26;
constexpr uint32_t kMatmulTileBudgetParam = kMatmulHmxFlagsParam + 1;
constexpr uint32_t kAttnPageSizeParam = 10;
// hmxFlags' upper bits mark the plan fields as present, so a command that
// predates them reads as unplanned rather than as a tile budget. Bit 0 is the
// flag itself -- the mask has to clear it, or a packed weight never matches the
// magic it is spelled with and this check quietly passes everything.
constexpr uint32_t kHmxPlanMagic = 0x484D58;
constexpr uint32_t kHmxPlanMagicMask = 0xFFFFFE;
constexpr uint32_t kHmxPlanPrepackedWeights = 1;

constexpr size_t kMaxCompileSpecKeyBytes = 64;

// One compile spec, in whatever shape the caller has it.
struct HexagonSpec {
  const char* key;
  const uint8_t* value;
  size_t size;
};

enum class HexagonSpecStatus {
  kOk,
  // A key this backend does not define. Refused rather than ignored: a caller
  // that asked for something is told, instead of getting the default.
  kUnknownKey,
  // A value whose width or contents cannot mean what its key says.
  kBadPayload,
  // The spec and the blob describe different programs.
  kMismatch,
};

struct HexagonSpecVerdict {
  HexagonSpecStatus status;
  const char* reason;
};

// What the blob says about itself, read the same way the DSP's descriptors are
// built so that both agree about where a field is.
struct HexagonBlobFacts {
  // Any matmul command whose weight is stored in the unit's tile order.
  bool hmx_prepacked;
  // Any attention command at all, and whether it is the paged entry point. A
  // program with no attention is consistent with either setting.
  bool attention_seen;
  bool attention_paged;
  // The blob leaves weights behind for the named data map to supply.
  bool external_weights;
};

inline const HexagonBlobHeader* HexagonHeader(const uint8_t* blob) {
  return reinterpret_cast<const HexagonBlobHeader*>(blob);
}

inline const HexagonOp* HexagonOps(const uint8_t* blob) {
  return reinterpret_cast<const HexagonOp*>(blob + sizeof(HexagonBlobHeader));
}

// Reads the facts above out of `n_ops` commands starting at the blob's op array.
inline HexagonBlobFacts ScanBlobFacts(
    const HexagonOp* ops,
    uint32_t n_ops,
    uint32_t version) {
  HexagonBlobFacts facts{false, false, false, false};
  for (uint32_t i = 0; i < n_ops; i++) {
    const HexagonOp& op = ops[i];
    if (op.type == kBatchMatMulType &&
        op.n_params > kMatmulHmxFlagsParam) {
      const uint32_t flags = static_cast<uint32_t>(op.params[kMatmulHmxFlagsParam]);
      if ((flags & kHmxPlanMagicMask) == kHmxPlanMagic &&
          (flags & kHmxPlanPrepackedWeights) != 0) {
        facts.hmx_prepacked = true;
      }
    } else if (op.type == kFlashAttnType && op.n_params > kAttnPageSizeParam) {
      facts.attention_seen = true;
      if (op.params[kAttnPageSizeParam] != 0) {
        facts.attention_paged = true;
      }
    }
  }
  facts.external_weights = version == kHexagonBlobVersionExternalWeights;
  return facts;
}

namespace detail {

inline bool KeyEquals(const char* left, const char* right) {
  return std::strncmp(left, right, kMaxCompileSpecKeyBytes) == 0;
}

inline HexagonSpecVerdict Verdict(
    HexagonSpecStatus status,
    const char* reason) {
  return HexagonSpecVerdict{status, reason};
}

inline bool ReadBoolSpec(const HexagonSpec& spec, bool* out) {
  if (spec.size != 1) {
    return false;
  }
  if (spec.value[0] > 1) {
    return false;
  }
  *out = spec.value[0] != 0;
  return true;
}

inline bool ReadUint64Spec(const HexagonSpec& spec, uint64_t* out) {
  if (spec.size != 8) {
    return false;
  }
  uint64_t value = 0;
  for (size_t i = 0; i < 8; i++) {
    value |= static_cast<uint64_t>(spec.value[i]) << (8 * i);
  }
  if (value == 0) {
    return false;
  }
  *out = value;
  return true;
}

} // namespace detail

// Checks a delegate's compile specs against the blob they came with. Reads the
// header, the command array and nothing else: the trailer parse below is
// separate because init() needs its result whether or not any spec was given.
inline HexagonSpecVerdict CheckCompileSpecs(
    const HexagonSpec* specs,
    size_t n_specs,
    const uint8_t* blob,
    size_t size) {
  if (size < sizeof(HexagonBlobHeader)) {
    return detail::Verdict(
        HexagonSpecStatus::kMismatch, "hexagon: blob too small to check");
  }
  const HexagonBlobHeader* header = HexagonHeader(blob);
  const size_t ops_bytes = sizeof(HexagonOp) * header->n_ops;
  if (size < sizeof(HexagonBlobHeader) + ops_bytes) {
    return detail::Verdict(
        HexagonSpecStatus::kMismatch, "hexagon: blob is truncated");
  }
  const HexagonBlobFacts facts =
      ScanBlobFacts(HexagonOps(blob), header->n_ops, header->version);
  bool declared_external = false;

  for (size_t i = 0; i < n_specs; i++) {
    const HexagonSpec& spec = specs[i];
    if (spec.key == nullptr) {
      return detail::Verdict(
          HexagonSpecStatus::kBadPayload, "hexagon: compile spec with no key");
    }
    if (detail::KeyEquals(spec.key, kHmxPrepackSpecKey)) {
      bool declared = false;
      if (!detail::ReadBoolSpec(spec, &declared)) {
        return detail::Verdict(
            HexagonSpecStatus::kBadPayload,
            "hexagon: hexagon_hmx_prepack must be a one-byte 0 or 1");
      }
      // Only one direction can be contradicted: a packed weight in the blob
      // proves the exporter packed, while a program with no matmul, or none
      // large enough to be worth packing, packs nothing either way.
      if (!declared && facts.hmx_prepacked) {
        return detail::Verdict(
            HexagonSpecStatus::kMismatch,
            "hexagon: hexagon_hmx_prepack is 0 but the blob's matmul weights "
            "are stored in the HMX tile order");
      }
    } else if (detail::KeyEquals(spec.key, kAttnPagedSpecKey)) {
      bool declared = false;
      if (!detail::ReadBoolSpec(spec, &declared)) {
        return detail::Verdict(
            HexagonSpecStatus::kBadPayload,
            "hexagon: hexagon_attn_paged must be a one-byte 0 or 1");
      }
      // This one is checked both ways: the choice is global, so every attention
      // command in the blob went through the same entry point.
      if (facts.attention_seen && declared != facts.attention_paged) {
        return detail::Verdict(
            HexagonSpecStatus::kMismatch,
            declared
                ? "hexagon: hexagon_attn_paged is 1 but the blob's attention "
                  "command is the non-paged entry point"
                : "hexagon: hexagon_attn_paged is 0 but the blob's attention "
                  "command is the paged entry point");
      }
    } else if (detail::KeyEquals(spec.key, kExternalWeightsMaxBytesSpecKey)) {
      uint64_t declared = 0;
      if (!detail::ReadUint64Spec(spec, &declared)) {
        return detail::Verdict(
            HexagonSpecStatus::kBadPayload,
            "hexagon: hexagon_external_weights_max_bytes must be a positive "
            "uint64");
      }
      // The threshold itself cannot be recovered from the blob -- a weight is
      // either in the file or it is not -- so what is checked is the fact that
      // some of them are not.
      declared_external = true;
    } else {
      return detail::Verdict(
          HexagonSpecStatus::kUnknownKey,
          "hexagon: unknown compile spec for this backend");
    }
  }

  // A blob carrying weights outside the .pte was compiled by an exporter that
  // was told where to draw the line, so the spec has to be there. Absent it,
  // the runtime would be asked to fill in a section nothing can fill.
  if (facts.external_weights && !declared_external) {
    return detail::Verdict(
        HexagonSpecStatus::kMismatch,
        "hexagon: the blob stores weights outside the .pte but no "
        "hexagon_external_weights_max_bytes spec came with it");
  }

  return detail::Verdict(HexagonSpecStatus::kOk, "");
}

// Where a weight the .pte does not carry has to land, and the key to fetch it
// under. `key` points into the blob and is NUL-terminated.
struct HexagonExternalWeightChunk {
  uint64_t offset;
  uint64_t size;
  const char* key;
};

struct HexagonExternalWeights {
  uint32_t count;
  const HexagonExternalWeight* entries;
  // The arena's weights section: the bytes the file carries, then these
  // entries. Every chunk's offset is measured from the start of it.
  uint64_t weights_bytes;
  // Where the section after the weights starts in the file: the dynamic
  // trailer, if there is one. For a blob with no external weights this is
  // exactly where that trailer always was.
  size_t dynamic_trailer_at;
};

struct HexagonWeightVerdict {
  bool ok;
  const char* reason;
};

// Reads the external weights trailer, or reports that there is none. Called
// before the arena is sized, because how much room the weights need is what
// this says.
inline HexagonWeightVerdict ParseExternalWeights(
    const uint8_t* blob,
    size_t size,
    HexagonExternalWeights* out) {
  out->count = 0;
  out->entries = nullptr;
  out->weights_bytes = 0;
  out->dynamic_trailer_at = 0;

  if (size < sizeof(HexagonBlobHeader)) {
    return HexagonWeightVerdict{false, "hexagon: blob too small"};
  }
  const HexagonBlobHeader* header = HexagonHeader(blob);
  const size_t ops_bytes = sizeof(HexagonOp) * header->n_ops;
  const size_t weights_at = sizeof(HexagonBlobHeader) + ops_bytes;
  if (weights_at > size || header->weights_bytes > size - weights_at) {
    return HexagonWeightVerdict{false, "hexagon: truncated weights section"};
  }
  const size_t weights_end = weights_at + header->weights_bytes;
  out->weights_bytes = header->weights_bytes;
  out->dynamic_trailer_at = weights_end;

  if (header->version != kHexagonBlobVersionExternalWeights) {
    return HexagonWeightVerdict{true, ""};
  }
  if (size - weights_end < sizeof(HexagonExternalWeightsTrailer)) {
    return HexagonWeightVerdict{
        false, "hexagon: truncated external weights trailer"};
  }
  const auto* trailer =
      reinterpret_cast<const HexagonExternalWeightsTrailer*>(blob + weights_end);
  if (trailer->magic != kHexagonExternalWeightsMagic) {
    return HexagonWeightVerdict{
        false, "hexagon: bad external weights trailer magic"};
  }
  if (trailer->version != kHexagonExternalWeightsVersion) {
    return HexagonWeightVerdict{
        false, "hexagon: unknown external weights trailer version"};
  }
  const size_t entries_at = weights_end + sizeof(HexagonExternalWeightsTrailer);
  const size_t entries_bytes =
      static_cast<size_t>(trailer->n_ext) * sizeof(HexagonExternalWeight);
  if (trailer->n_ext == 0 || entries_bytes > size - entries_at) {
    return HexagonWeightVerdict{
        false, "hexagon: external weight records out of bounds"};
  }
  if (trailer->weights_bytes < header->weights_bytes) {
    return HexagonWeightVerdict{
        false,
        "hexagon: the trailer's weights section is smaller than the file's"};
  }

  const auto* entries =
      reinterpret_cast<const HexagonExternalWeight*>(blob + entries_at);
  uint64_t end = header->weights_bytes;
  for (uint32_t i = 0; i < trailer->n_ext; i++) {
    const HexagonExternalWeight& entry = entries[i];
    if (entry.size == 0) {
      return HexagonWeightVerdict{false, "hexagon: empty external weight"};
    }
    // Past the bytes the file carries, and inside what the trailer promised.
    // Stated as two comparisons because the subtraction below would wrap if the
    // entry claimed to start past the end of the section.
    if (entry.offset < end || entry.offset > trailer->weights_bytes ||
        entry.size > trailer->weights_bytes - entry.offset) {
      return HexagonWeightVerdict{
          false, "hexagon: external weight out of bounds"};
    }
    end = entry.offset + entry.size;
    const void* terminator = std::memchr(
        entry.key, '\0', kHexagonExternalKeyBytes);
    if (terminator == nullptr || terminator == entry.key) {
      return HexagonWeightVerdict{
          false, "hexagon: external weight key is not a NUL-terminated name"};
    }
  }

  out->count = trailer->n_ext;
  out->entries = entries;
  out->weights_bytes = trailer->weights_bytes;
  out->dynamic_trailer_at = entries_at + entries_bytes;
  return HexagonWeightVerdict{true, ""};
}

} // namespace executorch::backends::hexagon
