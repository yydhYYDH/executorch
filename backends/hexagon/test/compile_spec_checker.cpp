/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Reads a blob the way the runtime's init() reads it, and reports what it
// found. The verdicts are hexagon_compat.h's, so what this prints is what a
// delegate would decide about the same bytes; the test drives it with blobs the
// Python writer produced, which is the only way to check the two halves of the
// format against each other without a device.
//
//   compile_spec_checker <blob> [key=value ...]
//
// A value is decimal for the two keys whose payload has a width -- one byte for
// a flag, eight for the byte count -- or `0x` and hex digits for any payload at
// all, which is how a test hands over the wrong width on purpose.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hexagon_compat.h"

namespace {

using executorch::backends::hexagon::kAttnPagedSpecKey;
using executorch::backends::hexagon::kExternalWeightsMaxBytesSpecKey;
using executorch::backends::hexagon::kHexagonBlobMagic;
using executorch::backends::hexagon::kHexagonBlobVersion;
using executorch::backends::hexagon::kHexagonBlobVersionExternalWeights;
using executorch::backends::hexagon::kHmxPrepackSpecKey;
using executorch::backends::hexagon::CheckCompileSpecs;
using executorch::backends::hexagon::HexagonBlobHeader;
using executorch::backends::hexagon::HexagonExternalWeight;
using executorch::backends::hexagon::HexagonExternalWeights;
using executorch::backends::hexagon::HexagonExternalWeightsTrailer;
using executorch::backends::hexagon::HexagonOp;
using executorch::backends::hexagon::HexagonSpec;
using executorch::backends::hexagon::HexagonSpecStatus;
using executorch::backends::hexagon::HexagonSpecVerdict;
using executorch::backends::hexagon::ParseExternalWeights;
using executorch::backends::hexagon::ScanBlobFacts;

const char* StatusName(HexagonSpecStatus status) {
  switch (status) {
    case HexagonSpecStatus::kOk:
      return "ok";
    case HexagonSpecStatus::kUnknownKey:
      return "unknown_key";
    case HexagonSpecStatus::kBadPayload:
      return "bad_payload";
    case HexagonSpecStatus::kMismatch:
      return "mismatch";
  }
  return "?";
}

bool IsBoolKey(const char* key) {
  return std::strcmp(key, kHmxPrepackSpecKey) == 0 ||
      std::strcmp(key, kAttnPagedSpecKey) == 0;
}

bool ReadFile(const char* path, std::vector<uint8_t>* out) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr) {
    return false;
  }
  uint8_t chunk[4096];
  size_t read = 0;
  while ((read = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
    out->insert(out->end(), chunk, chunk + read);
  }
  std::fclose(file);
  return true;
}

bool HexDigit(char c, uint8_t* out) {
  if (c >= '0' && c <= '9') {
    *out = static_cast<uint8_t>(c - '0');
  } else if (c >= 'a' && c <= 'f') {
    *out = static_cast<uint8_t>(c - 'a' + 10);
  } else if (c >= 'A' && c <= 'F') {
    *out = static_cast<uint8_t>(c - 'A' + 10);
  } else {
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <blob> [key=value ...]\n", argv[0]);
    return 2;
  }

  // The layout, from the C++ side, for the test to compare against the formats
  // it writes with: two derivations of one format is the point of the exercise.
  std::printf(
      "size trailer %zu\n", sizeof(HexagonExternalWeightsTrailer));
  std::printf("size entry %zu\n", sizeof(HexagonExternalWeight));
  std::printf(
      "offsetof entry key %zu\n",
      offsetof(HexagonExternalWeight, key));
  std::printf(
      "offsetof trailer weights_bytes %zu\n",
      offsetof(HexagonExternalWeightsTrailer, weights_bytes));

  std::vector<uint8_t> blob;
  if (!ReadFile(argv[1], &blob)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 2;
  }

  // The keys and payloads outlive the HexagonSpec views, so they are owned in
  // vectors that are filled before the views are taken.
  std::vector<std::string> keys;
  std::vector<std::vector<uint8_t>> payloads;
  for (int i = 2; i < argc; i++) {
    std::string argument(argv[i]);
    const size_t at = argument.find('=');
    if (at == std::string::npos) {
      std::fprintf(stderr, "not a key=value spec: %s\n", argv[i]);
      return 2;
    }
    const std::string key = argument.substr(0, at);
    const std::string value = argument.substr(at + 1);
    std::vector<uint8_t> payload;
    if (value.size() > 2 && value[0] == '0' && value[1] == 'x') {
      for (size_t j = 2; j + 1 < value.size(); j += 2) {
        uint8_t high = 0;
        uint8_t low = 0;
        if (!HexDigit(value[j], &high) || !HexDigit(value[j + 1], &low)) {
          std::fprintf(stderr, "bad hex payload: %s\n", value.c_str());
          return 2;
        }
        payload.push_back(static_cast<uint8_t>((high << 4) | low));
      }
    } else {
      const unsigned long long number =
          std::strtoull(value.c_str(), nullptr, 10);
      if (IsBoolKey(key.c_str())) {
        payload.push_back(static_cast<uint8_t>(number & 0xFF));
      } else if (
          std::strcmp(key.c_str(), kExternalWeightsMaxBytesSpecKey) == 0) {
        for (int byte = 0; byte < 8; byte++) {
          payload.push_back(
              static_cast<uint8_t>((number >> (8 * byte)) & 0xFF));
        }
      } else {
        std::fprintf(stderr, "unknown spec key: %s\n", key.c_str());
        return 2;
      }
    }
    keys.push_back(key);
    payloads.push_back(std::move(payload));
  }
  std::vector<HexagonSpec> specs;
  for (size_t i = 0; i < keys.size(); i++) {
    specs.push_back(
        HexagonSpec{keys[i].c_str(), payloads[i].data(), payloads[i].size()});
  }

  if (blob.size() < sizeof(HexagonBlobHeader)) {
    std::printf("header truncated\n");
    return 0;
  }
  const auto* header = reinterpret_cast<const HexagonBlobHeader*>(blob.data());
  std::printf(
      "header %u %u %u\n", header->magic, header->version, header->n_ops);
  if (header->magic != kHexagonBlobMagic ||
      (header->version != kHexagonBlobVersion &&
       header->version != kHexagonBlobVersionExternalWeights)) {
    std::printf("header bad\n");
    return 0;
  }

  const HexagonSpecVerdict verdict = CheckCompileSpecs(
      specs.data(), specs.size(), blob.data(), blob.size());
  std::printf("verdict %s\n", StatusName(verdict.status));
  std::printf("reason %s\n", verdict.reason);

  const auto* ops = reinterpret_cast<const HexagonOp*>(
      blob.data() + sizeof(HexagonBlobHeader));
  const auto facts = ScanBlobFacts(ops, header->n_ops, header->version);
  std::printf("facts hmx_prepacked %d\n", facts.hmx_prepacked ? 1 : 0);
  std::printf(
      "facts attention %d %d\n",
      facts.attention_seen ? 1 : 0,
      facts.attention_paged ? 1 : 0);
  std::printf("facts external %d\n", facts.external_weights ? 1 : 0);

  HexagonExternalWeights weights{};
  const auto& parsed = ParseExternalWeights(blob.data(), blob.size(), &weights);
  if (!parsed.ok) {
    std::printf("inspect failed %s\n", parsed.reason);
    return 0;
  }
  std::printf("inspect ok\n");
  std::printf(
      "weights %llu %zu\n",
      static_cast<unsigned long long>(weights.weights_bytes),
      weights.dynamic_trailer_at);
  std::printf("external %u\n", weights.count);
  for (uint32_t i = 0; i < weights.count; i++) {
    std::printf(
        "entry %u %llu %llu %s\n",
        i,
        static_cast<unsigned long long>(weights.entries[i].offset),
        static_cast<unsigned long long>(weights.entries[i].size),
        weights.entries[i].key);
  }
  return 0;
}
