/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// FSMN-Monophone VAD runner. Reads a 16 kHz mono 16-bit WAV or raw PCM file,
// computes fbank + LFR + CMVN on the device, runs the streaming FSMN encoder
// .pte over fixed 20-frame chunks with a four-layer memory cache, and writes
// the per-frame 248-way softmax scores. The score stream is decoded to
// speech/silence segments by the host reference so the device is compared on
// the same post-processing path as FunASR.

#include <gflags/gflags.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "fsmn_vad_frontend.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {

DEFINE_string(model_path, "", "FSMN encoder .pte.");
DEFINE_string(audio, "", "16 kHz mono 16-bit WAV or raw PCM input.");
DEFINE_string(cmvn, "", "Float32 CMVN asset (2 x 400 values).");
DEFINE_string(scores_out, "", "Write per-frame float32 scores [frames, 248].");
DEFINE_string(psil_out, "", "Write per-frame float32 silence probability (class 0).");
DEFINE_int32(chunk, 20, "Encoder frames per invocation.");
DEFINE_int32(limit_chunks, 0, "Stop after this many chunks (0 = all).");

constexpr int kFeatDim = 400;
constexpr int kProjDim = 128;
constexpr int kCacheFrames = 19;
constexpr int kClasses = 248;

uint16_t le16(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t le32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    ET_LOG(Error, "cannot open %s", path.c_str());
    return false;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize(n > 0 ? n : 0);
  size_t got = out.empty() ? 0 : fread(out.data(), 1, out.size(), f);
  fclose(f);
  return got == out.size();
}

// Incremental 16 kHz mono 16-bit WAV or raw PCM reader.
class AudioReader {
 public:
  explicit AudioReader(const std::string& path) : file_(std::fopen(path.c_str(), "rb")) {
    if (!file_) {
      return;
    }
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".pcm") {
      std::fseek(file_, 0, SEEK_END);
      remaining_ = std::ftell(file_);
      std::rewind(file_);
      valid_ = remaining_ % 2 == 0;
      total_bytes_ = remaining_;
      return;
    }
    uint8_t header[12];
    if (std::fread(header, 1, sizeof(header), file_) != sizeof(header) ||
        std::memcmp(header, "RIFF", 4) || std::memcmp(header + 8, "WAVE", 4)) {
      return;
    }
    bool format_ok = false;
    uint8_t chunk[8];
    while (std::fread(chunk, 1, sizeof(chunk), file_) == sizeof(chunk)) {
      const uint32_t size = le32(chunk + 4);
      if (!std::memcmp(chunk, "fmt ", 4)) {
        uint8_t format[16];
        if (size < sizeof(format) ||
            std::fread(format, 1, sizeof(format), file_) != sizeof(format)) {
          return;
        }
        format_ok = le16(format) == 1 && le16(format + 2) == 1 &&
            le32(format + 4) == 16000 && le16(format + 14) == 16;
        std::fseek(file_, size - sizeof(format) + (size & 1), SEEK_CUR);
      } else if (!std::memcmp(chunk, "data", 4)) {
        remaining_ = size;
        valid_ = format_ok && remaining_ % 2 == 0;
        total_bytes_ = remaining_;
        return;
      } else {
        std::fseek(file_, size + (size & 1), SEEK_CUR);
      }
    }
  }

  ~AudioReader() {
    if (file_) {
      std::fclose(file_);
    }
  }

  bool valid() const { return valid_; }
  size_t bytes() const { return total_bytes_; }

  bool next(std::vector<float>& samples, bool& final) {
    if (!remaining_) {
      return false;
    }
    const size_t count = std::min<size_t>(9600, remaining_ / 2);
    std::array<int16_t, 9600> pcm;
    if (std::fread(pcm.data(), sizeof(int16_t), count, file_) != count) {
      return false;
    }
    remaining_ -= count * 2;
    samples.resize(count);
    for (size_t i = 0; i < count; ++i) {
      samples[i] = pcm[i] / 32768.0f;
    }
    final = remaining_ == 0;
    return true;
  }

 private:
  FILE* file_ = nullptr;
  size_t remaining_ = 0;
  size_t total_bytes_ = 0;
  bool valid_ = false;
};

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty() || FLAGS_audio.empty() || FLAGS_cmvn.empty()) {
    ET_LOG(Error, "--model_path, --audio and --cmvn are required");
    return 2;
  }
  if (FLAGS_chunk <= 0) {
    ET_LOG(Error, "--chunk must be positive");
    return 2;
  }
  std::vector<uint8_t> cmvn_bytes;
  if (!read_file(FLAGS_cmvn, cmvn_bytes) || cmvn_bytes.size() != 800 * sizeof(float)) {
    ET_LOG(Error, "CMVN must contain 800 float32 values");
    return 3;
  }
  std::vector<float> cmvn(800);
  std::memcpy(cmvn.data(), cmvn_bytes.data(), cmvn_bytes.size());

  AudioReader audio(FLAGS_audio);
  if (!audio.valid() || audio.bytes() == 0) {
    ET_LOG(Error, "audio must be nonempty 16 kHz mono 16-bit WAV or raw PCM");
    return 4;
  }

  Module module(FLAGS_model_path, Module::LoadMode::Mmap);
  if (module.load() != Error::Ok || module.load_forward() != Error::Ok) {
    ET_LOG(Error, "cannot load %s", FLAGS_model_path.c_str());
    return 5;
  }
  auto meta = module.method_meta("forward");
  if (!meta.ok() || meta.get().num_inputs() != 5) {
    ET_LOG(Error, "expected five inputs in %s", FLAGS_model_path.c_str());
    return 6;
  }
  for (size_t i = 0; i < meta.get().num_inputs(); ++i) {
    auto info = meta.get().input_tensor_meta(i);
    if (info.ok()) {
      std::string shape = "[";
      for (auto dim : info.get().sizes()) {
        shape += std::to_string(dim) + ",";
      }
      shape += "]";
      std::printf("IN %zu dtype=%d shape=%s\n", i,
                  static_cast<int>(info.get().scalar_type()), shape.c_str());
    }
  }

  fsmn_vad::VadFrontend frontend(cmvn);
  std::vector<float> caches(4 * kProjDim * kCacheFrames, 0);
  std::vector<std::array<float, kFeatDim>> pending;
  std::vector<float> scores;
  std::vector<float> psil;
  size_t chunk_index = 0;
  bool stopped = false;
  const auto start_time = std::chrono::steady_clock::now();

  auto run_chunk = [&](size_t valid) -> int {
    const int T = FLAGS_chunk;
    std::vector<std::array<float, kFeatDim>> chunk(
        pending.begin(), pending.begin() + valid);
    pending.erase(pending.begin(), pending.begin() + valid);
    while (chunk.size() < static_cast<size_t>(T)) {
      chunk.push_back(chunk.back());
    }
    std::vector<float> input(static_cast<size_t>(T) * kFeatDim);
    for (int row = 0; row < T; ++row) {
      for (int col = 0; col < kFeatDim; ++col) {
        input[row * kFeatDim + col] = chunk[row][col];
      }
    }
    std::vector<executorch::aten::SizesType> x_shape = {1, T, kFeatDim};
    std::vector<executorch::aten::SizesType> c_shape = {1, kProjDim, kCacheFrames, 1};
    // EValue stores a non-owning Tensor, so the TensorPtr owners must outlive
    // the call to execute.
    std::vector<executorch::extension::TensorPtr> held;
    std::vector<EValue> args;
    held.push_back(from_blob(input.data(), x_shape, ScalarType::Float));
    args.emplace_back(*held.back());
    for (int layer = 0; layer < 4; ++layer) {
      held.push_back(from_blob(
          caches.data() + static_cast<size_t>(layer) * kProjDim * kCacheFrames,
          c_shape,
          ScalarType::Float));
      args.emplace_back(*held.back());
    }
    auto result = module.execute("forward", args);
    if (!result.ok()) {
      std::fprintf(stderr, "encoder failed at chunk %zu: 0x%08x\n", chunk_index,
                   static_cast<uint32_t>(result.error()));
      return 1;
    }
    int probs_index = -1;
    std::vector<int> cache_indices;
    std::vector<std::vector<float>> outputs;
    size_t out_index = 0;
    for (const auto& value : result.get()) {
      if (!value.isTensor()) {
        std::fprintf(stderr, "non-tensor output\n");
        return 1;
      }
      const auto& tensor = value.toTensor();
      const size_t numel = tensor.numel();
      const auto* data = static_cast<const float*>(tensor.data_ptr());
      if (numel == static_cast<size_t>(T) * kClasses) {
        probs_index = static_cast<int>(out_index);
      } else if (numel == static_cast<size_t>(kProjDim) * kCacheFrames) {
        cache_indices.push_back(static_cast<int>(out_index));
      }
      outputs.emplace_back(data, data + numel);
      ++out_index;
    }
    if (probs_index < 0 || cache_indices.size() != 4) {
      std::fprintf(stderr, "unexpected encoder outputs at chunk %zu\n", chunk_index);
      return 1;
    }
    for (size_t layer = 0; layer < 4; ++layer) {
      std::memcpy(
          caches.data() + layer * kProjDim * kCacheFrames,
          outputs[cache_indices[layer]].data(),
          kProjDim * kCacheFrames * sizeof(float));
    }
    const std::vector<float>& p = outputs[probs_index];
    scores.insert(scores.end(), p.begin(), p.begin() + valid * kClasses);
    for (size_t row = 0; row < valid; ++row) {
      psil.push_back(p[row * kClasses]);
    }
    ++chunk_index;
    if (FLAGS_limit_chunks && chunk_index >= static_cast<size_t>(FLAGS_limit_chunks)) {
      stopped = true;
    }
    return 0;
  };

  std::vector<float> samples;
  bool final = false;
  int rc = 0;
  while (!stopped && audio.next(samples, final)) {
    auto features = frontend.Accept(samples.data(), samples.size(), final);
    pending.insert(pending.end(), features.begin(), features.end());
    while (pending.size() >= static_cast<size_t>(FLAGS_chunk)) {
      if ((rc = run_chunk(FLAGS_chunk)) != 0) {
        break;
      }
      if (stopped) {
        break;
      }
    }
    if (rc != 0 || stopped) {
      break;
    }
  }
  if (rc == 0 && !stopped && !pending.empty()) {
    rc = run_chunk(pending.size());
  }
  if (rc != 0) {
    return 7;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  const size_t frames = scores.size() / kClasses;
  std::printf("FRAMES %zu CHUNKS %zu\n", frames, chunk_index);
  std::printf("TIMING audio=%.3fs elapsed=%.3fs rtf=%.3f\n",
              audio.bytes() / 32000.0, seconds, seconds / (audio.bytes() / 32000.0));
  if (!FLAGS_scores_out.empty()) {
    FILE* f = fopen(FLAGS_scores_out.c_str(), "wb");
    if (!f) {
      ET_LOG(Error, "cannot write %s", FLAGS_scores_out.c_str());
      return 8;
    }
    fwrite(scores.data(), sizeof(float), scores.size(), f);
    fclose(f);
    std::printf("SCORES %s %zu\n", FLAGS_scores_out.c_str(), scores.size());
  }
  if (!FLAGS_psil_out.empty()) {
    FILE* f = fopen(FLAGS_psil_out.c_str(), "wb");
    if (!f) {
      ET_LOG(Error, "cannot write %s", FLAGS_psil_out.c_str());
      return 8;
    }
    fwrite(psil.data(), sizeof(float), psil.size(), f);
    fclose(f);
    std::printf("PSIL %s %zu\n", FLAGS_psil_out.c_str(), psil.size());
  }
  return 0;
}
