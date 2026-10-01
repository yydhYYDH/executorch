/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Multi-stage ASR runner: loads a .pte stage, prints its graph signature, and
// drives it from input .bin files.  One binary serves both the offline chain
// (each stage executed once) and the streaming chain (stage 0 repeated over
// chunks, its state threaded into the later stages).
//
// Stage I/O comes from the graph signature, so a stage is described by its .pte
// and its input .bin files rather than by a hard-coded layout.

#include <gflags/gflags.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "audio_frontend.h"
#include "fsmn_vad_decoder.h"
#include "fsmn_vad_frontend.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/log.h>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;

namespace {

DEFINE_string(model_path, "", "Path to a .pte stage.");
DEFINE_string(inputs, "", "Comma-separated input .bin files, bound positionally.");
DEFINE_string(output_file, "", "Prefix for output .bin files.");
DEFINE_bool(sig, false, "Print the graph signature and exit.");
DEFINE_int32(num_executions, 1, "Number of times to execute the stage.");
DEFINE_bool(serve, false, "Execute tab-separated model, input, output requests on stdin.");
DEFINE_bool(serve_audio, false, "Transcribe audio paths from stdin in one process.");
DEFINE_int32(tcp_port, 0, "Listen on 127.0.0.1 at this TCP port; 0 disables TCP mode.");
DEFINE_string(tcp_tmp_dir, ".", "Directory for temporary uploaded audio in TCP mode.");
DEFINE_string(audio, "", "16 kHz mono 16-bit WAV or raw PCM input.");
DEFINE_string(cmvn, "", "Float32 CMVN asset for direct audio input.");
DEFINE_string(tokens, "", "JSON token list for direct audio input.");
DEFINE_string(predictor, "", "Predictor .pte for direct audio input.");
DEFINE_string(decoder_first, "", "First decoder .pte for direct audio input.");
DEFINE_string(decoder_next, "", "Subsequent decoder .pte for direct audio input.");
DEFINE_string(vad_model, "", "Optional FSMN-VAD encoder .pte; enables VAD endpointing.");
DEFINE_string(cmvn_vad, "", "Float32 CMVN asset for the VAD (2 x 400 values).");
DEFINE_string(spk_model, "", "Optional CAM++ speaker .pte; tags each segment with a speaker.");
DEFINE_double(spk_threshold, 0.35, "Cosine threshold for merging speaker embeddings.");
DEFINE_int32(spk_min_ms, 1500, "Segments shorter than this get no own speaker decision.");
DEFINE_int32(vad_chunk, 20, "VAD encoder frames per invocation.");
DEFINE_int32(
    vad_delay_chunks,
    0,
    "Delay the Paraformer stream behind VAD by this many 600 ms chunks. 0 keeps "
    "text aligned to the segment start; larger values trade that for boundary "
    "alignment on the end side.");
DEFINE_string(
    vad_mode,
    "continuous",
    "continuous: never reset Paraformer and split tokens into segments by chunk "
    "time. reset: flush accumulated text at each VAD endpoint and reset.");
DEFINE_bool(
    vad_realtime,
    true,
    "In continuous mode, print each segment as soon as VAD closes it instead of "
    "assigning all chunks after the run.");

const char* dtype_name(ScalarType t) {
  switch (t) {
    case ScalarType::Float:
      return "fp32";
    case ScalarType::Half:
      return "fp16";
    case ScalarType::BFloat16:
      return "bf16";
    case ScalarType::Int:
      return "int32";
    case ScalarType::Long:
      return "int64";
    case ScalarType::Byte:
      return "uint8";
    case ScalarType::Char:
      return "int8";
    case ScalarType::Bool:
      return "bool";
    default:
      return "other";
  }
}

size_t elem_size(ScalarType t) {
  switch (t) {
    case ScalarType::Float:
    case ScalarType::Int:
      return 4;
    case ScalarType::Half:
    case ScalarType::BFloat16:
      return 2;
    case ScalarType::Long:
      return 8;
    default:
      return 1;
  }
}

std::string shape_str(const executorch::runtime::TensorInfo& ti) {
  std::string s = "[";
  auto sizes = ti.sizes();
  for (size_t i = 0; i < sizes.size(); i++) {
    if (i) {
      s += ",";
    }
    s += std::to_string(sizes[i]);
  }
  return s + "]";
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

void split_csv(const std::string& s, std::vector<std::string>& out) {
  size_t start = 0;
  while (start <= s.size()) {
    size_t c = s.find(',', start);
    std::string tok = s.substr(start, c == std::string::npos ? c : c - start);
    if (!tok.empty()) {
      out.push_back(tok);
    }
    if (c == std::string::npos) {
      break;
    }
    start = c + 1;
  }
}

int execute_stage(
    Module& module,
    const MethodMeta& meta,
    const std::string& inputs,
    const std::string& output_file,
    int num_executions) {
  std::vector<std::string> input_paths;
  split_csv(inputs, input_paths);
  if (input_paths.size() != meta.num_inputs()) {
    ET_LOG(
        Error,
        "graph wants %zu inputs, got %zu bin files",
        meta.num_inputs(),
        input_paths.size());
    return 6;
  }

  std::vector<EValue> evals;
  std::vector<std::vector<uint8_t>> backing(input_paths.size());
  for (size_t i = 0; i < input_paths.size(); i++) {
    if (!read_file(input_paths[i], backing[i])) {
      return 7;
    }
    auto ti = meta.input_tensor_meta(i);
    ScalarType dt = ti.ok() ? ti.get().scalar_type() : ScalarType::Float;
    size_t esz = elem_size(dt);
    if (esz && backing[i].size() % esz != 0) {
      ET_LOG(
          Error,
          "input %zu size %zu is not a multiple of %zu",
          i,
          backing[i].size(),
          esz);
      return 8;
    }
    std::vector<executorch::aten::SizesType> isizes;
    if (ti.ok()) {
      auto s = ti.get().sizes();
      for (size_t d = 0; d < s.size(); d++) {
        isizes.push_back(s[d]);
      }
    }
    if (isizes.empty() && !ti.ok()) {
      isizes.push_back(static_cast<executorch::aten::SizesType>(backing[i].size() / esz));
    }
    evals.emplace_back(from_blob(backing[i].data(), isizes, dt));
    std::printf(
        "BOUND %zu %s %s %zu bytes\n",
        i,
        input_paths[i].c_str(),
        shape_str(ti.get()).c_str(),
        backing[i].size());
  }
  std::fflush(stdout);

  for (int it = 0; it < num_executions; it++) {
    auto outs = module.execute("forward", evals);
    if (!outs.ok()) {
      std::printf("EXECUTE_FAILED 0x%08x\n", static_cast<uint32_t>(outs.error()));
      std::fflush(stdout);
      return 9;
    }
    const auto& vals = outs.get();
    std::printf("EXEC %d outputs %zu\n", it, vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
      if (!vals[i].isTensor()) {
        continue;
      }
      const auto& t = vals[i].toTensor();
      std::string path = output_file + "-" + std::to_string(i) + ".bin";
      FILE* f = fopen(path.c_str(), "wb");
      if (!f) {
        ET_LOG(Error, "cannot write %s", path.c_str());
        return 10;
      }
      fwrite(t.data_ptr(), 1, t.numel() * elem_size(t.scalar_type()), f);
      fclose(f);
      std::printf(
          "OUT %d %s numel %zu %s\n",
          it,
          path.c_str(),
          t.numel(),
          dtype_name(t.scalar_type()));
    }
    std::fflush(stdout);
  }

  return 0;
}

uint16_t half_bits(float value) {
  return executorch::aten::Half(value).x;
}

float half_value(uint16_t bits) {
  return static_cast<float>(executorch::aten::Half(
      bits, executorch::aten::Half::from_bits()));
}

// Branch-based fp16 -> fp32, used in the per-chunk argmax where the tensor
// holds millions of values and the Half constructor dominates.
static inline float fp16_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x3ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while (!(mantissa & 0x400u)) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t read_half(const std::vector<uint8_t>& data, size_t index) {
  uint16_t value;
  std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
  return value;
}

float read_float(const std::vector<uint8_t>& data, size_t index) {
  float value;
  std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
  return value;
}

struct StageTiming {
  double load_seconds = 0;
  double run_seconds = 0;
  size_t calls = 0;
};

bool run_half_stage(
    std::unordered_map<std::string, std::unique_ptr<Module>>& modules,
    const std::string& path,
    std::vector<uint16_t>& input,
    std::vector<std::vector<uint8_t>>& output,
    const int64_t* position = nullptr,
    StageTiming* timing = nullptr) {
  auto stage_start = std::chrono::steady_clock::now();
  auto& module = modules[path];
  if (!module) {
    module = std::make_unique<Module>(path, Module::LoadMode::Mmap);
    const Error load_error = module->load();
    if (load_error != Error::Ok) {
      std::fprintf(stderr, "cannot load %s: %u\n", path.c_str(),
                   static_cast<unsigned>(load_error));
      module.reset();
      return false;
    }
    const Error forward_error = module->load_forward();
    if (forward_error != Error::Ok) {
      std::fprintf(stderr, "cannot load forward in %s: %u\n", path.c_str(),
                   static_cast<unsigned>(forward_error));
      module.reset();
      return false;
    }
  }
  const auto loaded_at = std::chrono::steady_clock::now();
  if (timing) {
    timing->load_seconds += std::chrono::duration<double>(loaded_at - stage_start).count();
  }
  auto meta = module->method_meta("forward");
  if (!meta.ok() || (meta.get().num_inputs() != 1 && meta.get().num_inputs() != 2)) {
    std::fprintf(stderr, "unsupported tensor inputs in %s\n", path.c_str());
    return false;
  }
  auto info = meta.get().input_tensor_meta(0);
  if (!info.ok() || info.get().scalar_type() != ScalarType::Half) {
    std::fprintf(stderr, "expected fp16 input in %s\n", path.c_str());
    return false;
  }
  std::vector<executorch::aten::SizesType> sizes;
  size_t expected = 1;
  for (auto dim : info.get().sizes()) {
    sizes.push_back(dim);
    expected *= dim;
  }
  constexpr size_t encoder_values = 3 + 10 * 560 + 2 * 50 * 4 * 50 * 128 + 5 * 560;
  if (position && meta.get().num_inputs() == 1 &&
      input.size() == encoder_values + 4 && expected == encoder_values + 10 * 560 + 65) {
    const int nvalid = static_cast<int>(half_value(input[1]));
    const int ncache = static_cast<int>(half_value(input[2]));
    input.resize(expected);
    const float increment = std::log(10000.0f) / 279.0f;
    for (size_t row = 0; row < 10; ++row) {
      const float pos = static_cast<float>(*position + row + 1);
      for (size_t col = 0; col < 280; ++col) {
        const float phase = pos * std::exp(static_cast<float>(col) * -increment);
        input[encoder_values + row * 560 + col] = half_bits(std::sin(phase));
        input[encoder_values + row * 560 + 280 + col] = half_bits(std::cos(phase));
      }
    }
    for (int i = 0; i < 65; ++i) {
      const bool valid = i < 50 ? i >= 50 - ncache : i < 55 || i < 55 + nvalid;
      input[encoder_values + 10 * 560 + i] = half_bits(valid ? 1.0f : 0.0f);
    }
  }
  if (position && meta.get().num_inputs() == 1 && expected + 4 == input.size()) {
    if (*position > 2048) {
      std::fprintf(stderr, "%s stores position in fp16; use the packed-position PTE\n",
                   path.c_str());
      return false;
    }
    input.resize(expected);
  }
  if (expected != input.size()) {
    std::fprintf(stderr, "%s wants %zu values, got %zu\n", path.c_str(), expected, input.size());
    return false;
  }
  std::vector<EValue> args;
  args.emplace_back(from_blob(input.data(), sizes, ScalarType::Half));
  int64_t position_value = position ? *position : 0;
  if (meta.get().num_inputs() == 2) {
    auto position_info = meta.get().input_tensor_meta(1);
    if (!position || !position_info.ok() ||
        position_info.get().scalar_type() != ScalarType::Long ||
        position_info.get().sizes().size() != 1 ||
        position_info.get().sizes()[0] != 1) {
      std::fprintf(stderr, "expected one int64 position in %s\n", path.c_str());
      return false;
    }
    args.emplace_back(from_blob(&position_value, {1}, ScalarType::Long));
  }
  auto result = module->execute("forward", args);
  if (!result.ok()) {
    std::fprintf(stderr, "execution failed in %s: %u\n", path.c_str(),
                 static_cast<unsigned>(result.error()));
    return false;
  }
  output.clear();
  for (const auto& value : result.get()) {
    if (!value.isTensor()) {
      std::fprintf(stderr, "non-tensor output from %s\n", path.c_str());
      return false;
    }
    const auto& tensor = value.toTensor();
    const size_t bytes = tensor.numel() * elem_size(tensor.scalar_type());
    output.emplace_back(bytes);
    std::memcpy(output.back().data(), tensor.data_ptr(), bytes);
  }
  if (timing) {
    timing->run_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - loaded_at).count();
    ++timing->calls;
  }
  return true;
}

uint16_t le16(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t le32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
      (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) |
      (static_cast<uint32_t>(bytes[3]) << 24);
}

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
  double duration_seconds() const { return total_bytes_ / 32000.0; }

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

// CAM++ embedding for one span of audio: 80-mel fbank frames with their time
// mean removed, padded or cropped to the PTE's fixed length, through the
// network. This is FunASR's extract_feature followed by the CAM++ forward.
std::vector<float> campplus_embedding(
    Module& spk_module,
    const float* samples,
    size_t count,
    int T) {
  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = 16000;
  opts.frame_opts.frame_length_ms = 25;
  opts.frame_opts.frame_shift_ms = 10;
  opts.frame_opts.window_type = "hamming";
  opts.frame_opts.dither = 0;
  // A whole segment is handed to the fbank in one call, so the bounded queue
  // has to hold every frame we will read (the T we keep, plus edge frames).
  opts.frame_opts.max_feature_vectors = T + 8;
  opts.mel_opts.num_bins = 80;
  opts.energy_floor = 0;
  knf::OnlineFbank fbank(opts);
  // Only the first T frames are kept, so there is nothing to gain from more
  // than the samples that produce them.
  const size_t usable = std::min<size_t>(count, static_cast<size_t>(T) * 160 + 400);
  std::vector<float> scaled(usable);
  for (size_t i = 0; i < usable; ++i) {
    scaled[i] = samples[i] * 32768.0f;
  }
  fbank.AcceptWaveform(16000, scaled.data(), static_cast<int32_t>(usable));
  fbank.InputFinished();
  const int32_t ready = fbank.NumFramesReady();
  if (ready <= 0) {
    return {};
  }
  const int rows = std::min<int32_t>(ready, T);
  if (std::getenv("SPK_DEBUG") != nullptr) {
    std::fprintf(stderr, "campplus: usable=%zu ready=%d rows=%d T=%d max=%d\n",
                 usable, ready, rows, T, opts.frame_opts.max_feature_vectors);
  }
  std::vector<std::array<float, 80>> frames(rows);
  std::array<double, 80> mean{};
  try {
    for (int i = 0; i < rows; ++i) {
      std::copy_n(fbank.GetFrame(i), 80, frames[i].begin());
      for (int c = 0; c < 80; ++c) {
        mean[c] += frames[i][c];
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "campplus fbank read failed: %s\n", e.what());
    return {};
  }
  for (int c = 0; c < 80; ++c) {
    mean[c] /= rows;
  }
  std::vector<float> input(static_cast<size_t>(T) * 80);
  for (int r = 0; r < T; ++r) {
    const auto& f = frames[std::min(r, rows - 1)];
    for (int c = 0; c < 80; ++c) {
      input[r * 80 + c] = static_cast<float>(f[c] - mean[c]);
    }
  }
  std::vector<executorch::aten::SizesType> shape = {1, T, 80};
  auto tensor = from_blob(input.data(), shape, ScalarType::Float);
  std::vector<EValue> args;
  args.emplace_back(*tensor);
  auto result = spk_module.execute("forward", args);
  if (!result.ok()) {
    return {};
  }
  for (const auto& value : result.get()) {
    if (value.isTensor()) {
      const auto& t = value.toTensor();
      const auto* data = static_cast<const float*>(t.data_ptr());
      return std::vector<float>(data, data + t.numel());
    }
  }
  return {};
}

void normalize_vector(std::vector<float>& v) {
  double norm = 0;
  for (float x : v) {
    norm += static_cast<double>(x) * x;
  }
  norm = std::sqrt(norm) + 1e-9;
  for (float& x : v) {
    x = static_cast<float>(x / norm);
  }
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9));
}

// Offline agglomerative clustering with average linkage and a cosine
// threshold, the operating point measured on AISHELL-4.
std::vector<int> cluster_offline(
    const std::vector<std::vector<float>>& embs,
    float threshold) {
  const size_t n = embs.size();
  std::vector<int> labels(n, 0);
  if (n == 0) {
    return labels;
  }
  std::vector<std::vector<float>> sim(n, std::vector<float>(n, 0.0f));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      sim[i][j] = sim[j][i] = cosine_similarity(embs[i], embs[j]);
    }
  }
  std::vector<std::vector<int>> clusters(n);
  for (size_t i = 0; i < n; ++i) {
    clusters[i] = {static_cast<int>(i)};
  }
  while (clusters.size() > 1) {
    float best = -2.0f;
    int bi = -1, bj = -1;
    for (size_t i = 0; i < clusters.size(); ++i) {
      for (size_t j = i + 1; j < clusters.size(); ++j) {
        double sum = 0;
        for (int a : clusters[i]) {
          for (int b : clusters[j]) {
            sum += sim[a][b];
          }
        }
        const float s = static_cast<float>(
            sum / (clusters[i].size() * clusters[j].size()));
        if (s > best) {
          best = s;
          bi = static_cast<int>(i);
          bj = static_cast<int>(j);
        }
      }
    }
    if (best < threshold) {
      break;
    }
    clusters[bi].insert(clusters[bi].end(), clusters[bj].begin(), clusters[bj].end());
    clusters.erase(clusters.begin() + bj);
  }
  for (size_t k = 0; k < clusters.size(); ++k) {
    for (int i : clusters[k]) {
      labels[i] = static_cast<int>(k);
    }
  }
  return labels;
}

// Fixed part of the encoder input before the appended position encoding and
// mask: three scalars, ten feature rows, the encoder state, and the overlap.
constexpr size_t kEncoderBaseValues =
    3 + 10 * 560 + 2 * 50 * 4 * 50 * 128 + 5 * 560 + 4;

int transcribe_audio(
    const std::string& audio_path,
    std::unordered_map<std::string, std::unique_ptr<Module>>& modules,
    std::string* result_text = nullptr,
    uint32_t* result_elapsed_ms = nullptr,
    uint32_t* result_audio_ms = nullptr) {
  const auto start_time = std::chrono::steady_clock::now();
  if (FLAGS_model_path.empty() || FLAGS_predictor.empty() ||
      FLAGS_decoder_first.empty() || FLAGS_decoder_next.empty() ||
      FLAGS_cmvn.empty() || FLAGS_tokens.empty()) {
    ET_LOG(Error, "audio mode requires all four PTEs, CMVN, and tokens");
    return 2;
  }
  AudioReader audio(audio_path);
  if (!audio.valid() || audio.duration_seconds() == 0) {
    ET_LOG(Error, "audio must be nonempty 16 kHz mono 16-bit WAV or raw PCM");
    return 3;
  }
  std::vector<uint8_t> cmvn_bytes;
  if (!read_file(FLAGS_cmvn, cmvn_bytes) || cmvn_bytes.size() != 1120 * sizeof(float)) {
    ET_LOG(Error, "CMVN must contain 1120 float32 values");
    return 4;
  }
  std::vector<float> cmvn(1120);
  std::memcpy(cmvn.data(), cmvn_bytes.data(), cmvn_bytes.size());
  std::ifstream token_file(FLAGS_tokens);
  if (!token_file) {
    ET_LOG(Error, "cannot open %s", FLAGS_tokens.c_str());
    return 5;
  }
  const auto tokens = nlohmann::json::parse(token_file).get<std::vector<std::string>>();
  paraformer::AudioFrontend frontend(cmvn);
  std::vector<uint8_t> enc_state(2 * 50 * 4 * 50 * 128 * 2);
  std::vector<uint8_t> overlap(5 * 560 * 2);
  std::vector<uint8_t> fsmn(16 * 512 * 26 * 2);
  std::vector<uint8_t> opt(32 * 5120 * 2);
  std::array<float, 513> predictor_state{};
  bool decoder_started = false;
  StageTiming encoder_timing;
  StageTiming predictor_timing;
  StageTiming decoder_first_timing;
  StageTiming decoder_next_timing;
  size_t chunk_index = 0;
  std::string text;
  std::vector<uint16_t> enc_input;
  enc_input.reserve(kEncoderBaseValues + 10 * 560 + 65);
  std::vector<float> samples;
  bool final = false;
  while (audio.next(samples, final)) {
    auto features = frontend.Accept(samples.data(), samples.size(), final);
    if (features.empty()) {
      continue;
    }
    if (features.size() > 10) {
      ET_LOG(Error, "audio chunk exceeds current encoder PTE contract");
      return 6;
    }
    enc_input.resize(kEncoderBaseValues);
    const int64_t position = static_cast<int64_t>(chunk_index * 10);
    if (position >= (int64_t{1} << 40)) {
      ET_LOG(Error, "audio position exceeds packed encoder range");
      return 6;
    }
    enc_input[0] = half_bits(static_cast<float>(std::min<int64_t>(position, 2048)));
    enc_input[1] = half_bits(static_cast<float>(features.size()));
    enc_input[2] = half_bits(static_cast<float>(std::min<size_t>(chunk_index * 10, 40)));
    for (size_t row = 0; row < features.size(); ++row) {
      for (size_t col = 0; col < 560; ++col) {
        enc_input[3 + row * 560 + col] = half_bits(features[row][col]);
      }
    }
    std::memcpy(enc_input.data() + 3 + 10 * 560, enc_state.data(), enc_state.size());
    std::memcpy(enc_input.data() + 3 + 10 * 560 + enc_state.size() / 2,
                overlap.data(), overlap.size());
    for (size_t digit = 0; digit < 4; ++digit) {
      enc_input[enc_input.size() - 4 + digit] =
          half_bits(static_cast<float>((position >> (digit * 10)) & 1023));
    }
    std::vector<std::vector<uint8_t>> enc;
    if (!run_half_stage(modules, FLAGS_model_path, enc_input, enc, &position, &encoder_timing) || enc.size() != 3 ||
        enc[0].size() != 15 * 512 * 2 || enc[1].size() != enc_state.size() ||
        enc[2].size() != overlap.size()) {
      return 7;
    }
    enc_state = std::move(enc[1]);
    overlap = std::move(enc[2]);
    for (size_t i = 0; i < enc[0].size() / 2; ++i) {
      if ((read_half(enc[0], i) & 0x7c00) == 0x7c00) {
        // A saturating chunk on loud far-field audio is tolerated by the
        // streaming session; the file paths used to abort on it, which stops a
        // whole meeting over one chunk. Warn and carry on, as the session does.
        std::fprintf(
            stderr,
            "encoder chunk %zu produced non-finite values; continuing\n",
            chunk_index);
        std::fflush(stderr);
        break;
      }
    }
    std::vector<uint16_t> pred_input(8195);
    std::memcpy(pred_input.data(), enc[0].data(), enc[0].size());
    pred_input[7680] = half_bits(predictor_state[0]);
    for (size_t i = 0; i < 512; ++i) {
      pred_input[7681 + i] = half_bits(predictor_state[i + 1]);
    }
    pred_input[8194] = half_bits(final ? 1.0f : 0.0f);
    std::vector<std::vector<uint8_t>> pred;
    if (!run_half_stage(modules, FLAGS_predictor, pred_input, pred, nullptr, &predictor_timing)) {
      return 9;
    }
    if (pred.size() != 4 || pred[0].size() < 16 * 512 * sizeof(float) ||
        pred[1].size() % 2 || pred[2].size() < sizeof(float) ||
        pred[3].size() < 512 * sizeof(float)) {
      std::fprintf(stderr, "predictor output sizes:");
      for (const auto& value : pred) {
        std::fprintf(stderr, " %zu", value.size());
      }
      std::fprintf(stderr, "\n");
      return 9;
    }
    size_t nvalid = 0;
    for (size_t i = 0; i + 1 < pred[1].size(); i += 2) {
      nvalid += read_half(pred[1], i / 2) != 0;
    }
    if (nvalid > 16) {
      ET_LOG(Error, "predictor chunk %zu exceeds 16 decoder slots", chunk_index);
      return 10;
    }
    predictor_state[0] = read_float(pred[2], 0);
    for (size_t i = 0; i < 512; ++i) {
      predictor_state[i + 1] = read_float(pred[3], i);
    }
    if (nvalid) {
      std::vector<uint16_t> dec_input(15 * 512 + 16 * 512 + 16 * 512 * 26 + 32 * 5120 + 1);
      std::memcpy(dec_input.data(), enc[0].data(), enc[0].size());
      for (size_t i = 0; i < nvalid * 512; ++i) {
        dec_input[7680 + i] = half_bits(read_float(pred[0], i));
      }
      std::memcpy(dec_input.data() + 7680 + 16 * 512, fsmn.data(), fsmn.size());
      std::memcpy(dec_input.data() + 7680 + 16 * 512 + fsmn.size() / 2,
                  opt.data(), opt.size());
      dec_input.back() = half_bits(static_cast<float>(nvalid));
      std::vector<std::vector<uint8_t>> dec;
      const std::string& decoder = decoder_started ? FLAGS_decoder_next : FLAGS_decoder_first;
      StageTiming& decoder_timing = decoder_started ? decoder_next_timing : decoder_first_timing;
      if (!run_half_stage(modules, decoder, dec_input, dec, nullptr, &decoder_timing) || dec.size() != 49 ||
          dec[0].size() != 16 * 8404 * 2) {
        return 11;
      }
      decoder_started = true;
      for (size_t row = 0; row < nvalid; ++row) {
        const uint16_t* logits = reinterpret_cast<const uint16_t*>(
            dec[0].data() + row * 8404 * sizeof(uint16_t));
        size_t best = 0;
        float score = -INFINITY;
        for (size_t col = 0; col < 8404; ++col) {
          const float value = fp16_to_float(logits[col]);
          if (value > score) {
            score = value;
            best = col;
          }
        }
        if (best >= tokens.size()) {
          ET_LOG(Error, "decoder produced out-of-range token %zu", best);
          return 12;
        }
        if (best != 0 && best != 1 && best != 2) {
          text += tokens[best] == "<space>" ? " " : tokens[best];
        }
      }
      fsmn.clear();
      opt.clear();
      for (size_t i = 1; i < 17; ++i) {
        fsmn.insert(fsmn.end(), dec[i].begin(), dec[i].end());
      }
      for (size_t i = 17; i < 49; ++i) {
        opt.insert(opt.end(), dec[i].begin(), dec[i].end());
      }
      if (fsmn.size() != 16 * 512 * 26 * 2 || opt.size() != 32 * 5120 * 2) {
        ET_LOG(Error, "decoder cache size mismatch");
        return 13;
      }
    }
    std::printf("chunk %zu: %zu tokens\n", chunk_index, nvalid);
    std::fflush(stdout);
    ++chunk_index;
  }
  std::printf("TEXT %s\n", text.c_str());
  if (result_text) {
    *result_text = text;
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start_time).count();
  if (result_elapsed_ms) {
    *result_elapsed_ms = static_cast<uint32_t>(seconds * 1000.0 + 0.5);
  }
  if (result_audio_ms) {
    *result_audio_ms = static_cast<uint32_t>(audio.duration_seconds() * 1000.0 + 0.5);
  }
  std::printf("TIMING audio=%.3fs elapsed=%.3fs rtf=%.3f\n",
              audio.duration_seconds(), seconds, seconds / audio.duration_seconds());
  std::printf("TIMING_STAGE encoder=%zu/%.3f/%.3f predictor=%zu/%.3f/%.3f decoder_first=%zu/%.3f/%.3f decoder_next=%zu/%.3f/%.3f (calls/load_s/run_s)\n",
              encoder_timing.calls, encoder_timing.load_seconds, encoder_timing.run_seconds,
              predictor_timing.calls, predictor_timing.load_seconds, predictor_timing.run_seconds,
              decoder_first_timing.calls, decoder_first_timing.load_seconds, decoder_first_timing.run_seconds,
              decoder_next_timing.calls, decoder_next_timing.load_seconds, decoder_next_timing.run_seconds);
  return 0;
}

constexpr int kVadFeat = 400;
constexpr int kVadProj = 128;
constexpr int kVadCacheFrames = 19;
constexpr int kVadClasses = 248;

struct PfChunk {
  int64_t center_ms;
  std::string text;
};

struct DiarSegment {
  fsmn_vad::VadSegment span;
  std::string text;
  int speaker = -1;
};

float decibel_at(const std::vector<float>& samples, int64_t frame) {
  const int64_t begin = frame * 160;
  const int64_t end = std::min<int64_t>(begin + 400, samples.size());
  double acc = 0.0;
  for (int64_t j = begin; j < end; ++j) {
    acc += static_cast<double>(samples[j]) * samples[j];
  }
  return static_cast<float>(10.0 * std::log10(acc + 1e-6));
}

bool run_vad_encoder(
    Module& module,
    std::vector<float>& caches,
    const std::vector<std::array<float, kVadFeat>>& frames,
    size_t valid,
    std::vector<float>& psil_out) {
  const int T = FLAGS_vad_chunk;
  std::vector<std::array<float, kVadFeat>> chunk(frames.begin(), frames.begin() + valid);
  while (chunk.size() < static_cast<size_t>(T)) {
    chunk.push_back(chunk.back());
  }
  std::vector<float> input(static_cast<size_t>(T) * kVadFeat);
  for (int row = 0; row < T; ++row) {
    for (int col = 0; col < kVadFeat; ++col) {
      input[row * kVadFeat + col] = chunk[row][col];
    }
  }
  std::vector<executorch::aten::SizesType> x_shape = {1, T, kVadFeat};
  std::vector<executorch::aten::SizesType> c_shape = {1, kVadProj, kVadCacheFrames, 1};
  std::vector<executorch::extension::TensorPtr> held;
  std::vector<EValue> args;
  held.push_back(from_blob(input.data(), x_shape, ScalarType::Float));
  args.emplace_back(*held.back());
  for (int layer = 0; layer < 4; ++layer) {
    held.push_back(from_blob(
        caches.data() + static_cast<size_t>(layer) * kVadProj * kVadCacheFrames,
        c_shape,
        ScalarType::Float));
    args.emplace_back(*held.back());
  }
  auto result = module.execute("forward", args);
  if (!result.ok()) {
    std::fprintf(stderr, "VAD encoder failed: %u\n",
                 static_cast<unsigned>(result.error()));
    return false;
  }
  int probs_index = -1;
  std::vector<int> cache_indices;
  std::vector<std::vector<float>> outputs;
  size_t out_index = 0;
  for (const auto& value : result.get()) {
    if (!value.isTensor()) {
      return false;
    }
    const auto& tensor = value.toTensor();
    const size_t numel = tensor.numel();
    const auto* data = static_cast<const float*>(tensor.data_ptr());
    if (numel == static_cast<size_t>(T) * kVadClasses) {
      probs_index = static_cast<int>(out_index);
    } else if (numel == static_cast<size_t>(kVadProj) * kVadCacheFrames) {
      cache_indices.push_back(static_cast<int>(out_index));
    }
    outputs.emplace_back(data, data + numel);
    ++out_index;
  }
  if (probs_index < 0 || cache_indices.size() != 4) {
    return false;
  }
  for (size_t layer = 0; layer < 4; ++layer) {
    std::memcpy(
        caches.data() + layer * kVadProj * kVadCacheFrames,
        outputs[cache_indices[layer]].data(),
        kVadProj * kVadCacheFrames * sizeof(float));
  }
  psil_out.resize(valid);
  for (size_t row = 0; row < valid; ++row) {
    psil_out[row] = outputs[probs_index][row * kVadClasses];
  }
  return true;
}

// B-a: run the streaming Paraformer continuously and let VAD endpoints close an
// utterance. The accumulated text is flushed as the segment's transcript and the
// Paraformer state is reset at the boundary.
int transcribe_audio_vad(
    const std::string& audio_path,
    std::unordered_map<std::string, std::unique_ptr<Module>>& modules) {
  const auto start_time = std::chrono::steady_clock::now();
  if (FLAGS_model_path.empty() || FLAGS_predictor.empty() ||
      FLAGS_decoder_first.empty() || FLAGS_decoder_next.empty() ||
      FLAGS_cmvn.empty() || FLAGS_tokens.empty() || FLAGS_vad_model.empty() ||
      FLAGS_cmvn_vad.empty()) {
    ET_LOG(Error, "VAD mode requires the four PTEs, CMVN, tokens, VAD PTE and VAD CMVN");
    return 2;
  }
  AudioReader audio(audio_path);
  if (!audio.valid() || audio.duration_seconds() == 0) {
    ET_LOG(Error, "audio must be nonempty 16 kHz mono 16-bit WAV or raw PCM");
    return 3;
  }
  std::vector<uint8_t> cmvn_bytes;
  if (!read_file(FLAGS_cmvn, cmvn_bytes) || cmvn_bytes.size() != 1120 * sizeof(float)) {
    ET_LOG(Error, "CMVN must contain 1120 float32 values");
    return 4;
  }
  std::vector<float> cmvn(1120);
  std::memcpy(cmvn.data(), cmvn_bytes.data(), cmvn_bytes.size());
  std::vector<uint8_t> vad_cmvn_bytes;
  if (!read_file(FLAGS_cmvn_vad, vad_cmvn_bytes) ||
      vad_cmvn_bytes.size() != 800 * sizeof(float)) {
    ET_LOG(Error, "VAD CMVN must contain 800 float32 values");
    return 4;
  }
  std::vector<float> vad_cmvn(800);
  std::memcpy(vad_cmvn.data(), vad_cmvn_bytes.data(), vad_cmvn_bytes.size());
  std::ifstream token_file(FLAGS_tokens);
  if (!token_file) {
    ET_LOG(Error, "cannot open %s", FLAGS_tokens.c_str());
    return 5;
  }
  const auto tokens = nlohmann::json::parse(token_file).get<std::vector<std::string>>();

  Module vad_module(FLAGS_vad_model, Module::LoadMode::Mmap);
  if (vad_module.load() != Error::Ok || vad_module.load_forward() != Error::Ok) {
    ET_LOG(Error, "cannot load %s", FLAGS_vad_model.c_str());
    return 5;
  }

  std::unique_ptr<Module> spk_module;
  int spk_T = 300;
  if (!FLAGS_spk_model.empty()) {
    spk_module = std::make_unique<Module>(FLAGS_spk_model, Module::LoadMode::Mmap);
    if (spk_module->load() != Error::Ok || spk_module->load_forward() != Error::Ok) {
      ET_LOG(Error, "cannot load %s", FLAGS_spk_model.c_str());
      return 5;
    }
    auto smeta = spk_module->method_meta("forward");
    if (smeta.ok() && smeta.get().num_inputs() >= 1) {
      auto ti = smeta.get().input_tensor_meta(0);
      if (ti.ok() && ti.get().sizes().size() == 3) {
        spk_T = ti.get().sizes()[1];
      }
    }
  }

  paraformer::AudioFrontend frontend(cmvn);
  fsmn_vad::VadFrontend vad_frontend(vad_cmvn);
  fsmn_vad::VadDecoder vad_decoder;

  std::vector<uint8_t> enc_state(2 * 50 * 4 * 50 * 128 * 2);
  std::vector<uint8_t> overlap(5 * 560 * 2);
  std::vector<uint8_t> fsmn(16 * 512 * 26 * 2);
  std::vector<uint8_t> opt(32 * 5120 * 2);
  std::array<float, 513> predictor_state{};
  bool decoder_started = false;
  size_t chunk_index = 0;

  std::vector<float> vad_caches(4 * kVadProj * kVadCacheFrames, 0);
  std::vector<std::array<float, kVadFeat>> vad_pending;
  std::deque<std::pair<int64_t, std::vector<std::array<float, 560>>>> pf_queue;
  std::vector<PfChunk> pf_chunks;
  // Only a bounded tail of the waveform is kept: enough for the decibel of the
  // frames in flight and for the longest segment's speaker embedding. Holding
  // the whole meeting would cost 150 MB over 39 minutes for nothing.
  std::vector<float> sample_buf;
  int64_t sample_base = 0;
  int64_t vad_frames = 0;
  int64_t audio_chunks = 0;

  const bool b_reset = FLAGS_vad_mode == "reset";
  const bool b_realtime = !b_reset && FLAGS_vad_realtime;
  const bool b_diarize = spk_module != nullptr && !b_reset;
  std::string cur_text;
  std::vector<fsmn_vad::VadSegment> segments;
  std::vector<std::string> segment_texts;
  std::vector<std::vector<float>> diar_embs;
  std::vector<size_t> diar_index;
  size_t pf_flush = 0;

  auto reset_paraformer = [&]() {
    std::fill(enc_state.begin(), enc_state.end(), 0);
    std::fill(overlap.begin(), overlap.end(), 0);
    std::fill(fsmn.begin(), fsmn.end(), 0);
    std::fill(opt.begin(), opt.end(), 0);
    predictor_state.fill(0.0f);
    decoder_started = false;
    chunk_index = 0;
  };

  std::vector<uint16_t> enc_input;
  enc_input.reserve(kEncoderBaseValues + 10 * 560 + 65);
  auto process_paraformer_chunk =
      [&](const std::vector<std::array<float, 560>>& features,
          bool final,
          std::string* out) -> int {
    out->clear();
    if (features.empty()) {
      return 0;
    }
    if (features.size() > 10) {
      ET_LOG(Error, "audio chunk exceeds current encoder PTE contract");
      return 6;
    }
    enc_input.resize(kEncoderBaseValues);
    const int64_t position = static_cast<int64_t>(chunk_index * 10);
    if (position >= (int64_t{1} << 40)) {
      ET_LOG(Error, "audio position exceeds packed encoder range");
      return 6;
    }
    enc_input[0] = half_bits(static_cast<float>(std::min<int64_t>(position, 2048)));
    enc_input[1] = half_bits(static_cast<float>(features.size()));
    enc_input[2] = half_bits(static_cast<float>(std::min<size_t>(chunk_index * 10, 40)));
    for (size_t row = 0; row < features.size(); ++row) {
      for (size_t col = 0; col < 560; ++col) {
        enc_input[3 + row * 560 + col] = half_bits(features[row][col]);
      }
    }
    std::memcpy(enc_input.data() + 3 + 10 * 560, enc_state.data(), enc_state.size());
    std::memcpy(enc_input.data() + 3 + 10 * 560 + enc_state.size() / 2,
                overlap.data(), overlap.size());
    for (size_t digit = 0; digit < 4; ++digit) {
      enc_input[enc_input.size() - 4 + digit] =
          half_bits(static_cast<float>((position >> (digit * 10)) & 1023));
    }
    std::vector<std::vector<uint8_t>> enc;
    if (!run_half_stage(modules, FLAGS_model_path, enc_input, enc, &position, nullptr) ||
        enc.size() != 3 || enc[0].size() != 15 * 512 * 2 ||
        enc[1].size() != enc_state.size() || enc[2].size() != overlap.size()) {
      return 7;
    }
    enc_state = std::move(enc[1]);
    overlap = std::move(enc[2]);
    for (size_t i = 0; i < enc[0].size() / 2; ++i) {
      if ((read_half(enc[0], i) & 0x7c00) == 0x7c00) {
        // A saturating chunk on loud far-field audio is tolerated by the
        // streaming session; the file paths used to abort on it, which stops a
        // whole meeting over one chunk. Warn and carry on, as the session does.
        std::fprintf(
            stderr,
            "encoder chunk %zu produced non-finite values; continuing\n",
            chunk_index);
        std::fflush(stderr);
        break;
      }
    }
    std::vector<uint16_t> pred_input(8195);
    std::memcpy(pred_input.data(), enc[0].data(), enc[0].size());
    pred_input[7680] = half_bits(predictor_state[0]);
    for (size_t i = 0; i < 512; ++i) {
      pred_input[7681 + i] = half_bits(predictor_state[i + 1]);
    }
    pred_input[8194] = half_bits(final ? 1.0f : 0.0f);
    std::vector<std::vector<uint8_t>> pred;
    if (!run_half_stage(modules, FLAGS_predictor, pred_input, pred, nullptr, nullptr)) {
      return 9;
    }
    if (pred.size() != 4 || pred[0].size() < 16 * 512 * sizeof(float) ||
        pred[1].size() % 2 || pred[2].size() < sizeof(float) ||
        pred[3].size() < 512 * sizeof(float)) {
      return 9;
    }
    size_t nvalid = 0;
    for (size_t i = 0; i + 1 < pred[1].size(); i += 2) {
      nvalid += read_half(pred[1], i / 2) != 0;
    }
    if (nvalid > 16) {
      ET_LOG(Error, "predictor chunk %zu exceeds 16 decoder slots", chunk_index);
      return 10;
    }
    predictor_state[0] = read_float(pred[2], 0);
    for (size_t i = 0; i < 512; ++i) {
      predictor_state[i + 1] = read_float(pred[3], i);
    }
    if (nvalid) {
      std::vector<uint16_t> dec_input(15 * 512 + 16 * 512 + 16 * 512 * 26 + 32 * 5120 + 1);
      std::memcpy(dec_input.data(), enc[0].data(), enc[0].size());
      for (size_t i = 0; i < nvalid * 512; ++i) {
        dec_input[7680 + i] = half_bits(read_float(pred[0], i));
      }
      std::memcpy(dec_input.data() + 7680 + 16 * 512, fsmn.data(), fsmn.size());
      std::memcpy(dec_input.data() + 7680 + 16 * 512 + fsmn.size() / 2,
                  opt.data(), opt.size());
      dec_input.back() = half_bits(static_cast<float>(nvalid));
      std::vector<std::vector<uint8_t>> dec;
      const std::string& decoder = decoder_started ? FLAGS_decoder_next : FLAGS_decoder_first;
      if (!run_half_stage(modules, decoder, dec_input, dec, nullptr, nullptr) ||
          dec.size() != 49 || dec[0].size() != 16 * 8404 * 2) {
        return 11;
      }
      decoder_started = true;
      for (size_t row = 0; row < nvalid; ++row) {
        const uint16_t* logits = reinterpret_cast<const uint16_t*>(
            dec[0].data() + row * 8404 * sizeof(uint16_t));
        size_t best = 0;
        float score = -INFINITY;
        for (size_t col = 0; col < 8404; ++col) {
          const float value = fp16_to_float(logits[col]);
          if (value > score) {
            score = value;
            best = col;
          }
        }
        if (best >= tokens.size()) {
          ET_LOG(Error, "decoder produced out-of-range token %zu", best);
          return 12;
        }
        if (best != 0 && best != 1 && best != 2) {
          *out += tokens[best] == "<space>" ? " " : tokens[best];
        }
      }
      fsmn.clear();
      opt.clear();
      for (size_t i = 1; i < 17; ++i) {
        fsmn.insert(fsmn.end(), dec[i].begin(), dec[i].end());
      }
      for (size_t i = 17; i < 49; ++i) {
        opt.insert(opt.end(), dec[i].begin(), dec[i].end());
      }
      if (fsmn.size() != 16 * 512 * 26 * 2 || opt.size() != 32 * 5120 * 2) {
        ET_LOG(Error, "decoder cache size mismatch");
        return 13;
      }
    }
    ++chunk_index;
    return 0;
  };

  auto consume_chunk = [&](const std::vector<std::array<float, 560>>& feats,
                           bool is_final,
                           int64_t center) -> int {
    std::string chunk_text;
    const int status = process_paraformer_chunk(feats, is_final, &chunk_text);
    if (status != 0) {
      return status;
    }
    if (b_reset) {
      cur_text += chunk_text;
    } else {
      pf_chunks.push_back({center, chunk_text});
    }
    return 0;
  };

  auto decibel_frame = [&](int64_t frame) {
    const int64_t begin = frame * 160 - sample_base;
    const int64_t end =
        std::min<int64_t>(begin + 400, static_cast<int64_t>(sample_buf.size()));
    double acc = 0.0;
    for (int64_t j = std::max<int64_t>(begin, int64_t{0}); j < end; ++j) {
      acc += static_cast<double>(sample_buf[j]) * sample_buf[j];
    }
    return static_cast<float>(10.0 * std::log10(acc + 1e-6));
  };
  std::vector<float> samples;
  bool final = false;
  int rc = 0;
  while (audio.next(samples, final)) {
    sample_buf.insert(sample_buf.end(), samples.begin(), samples.end());

    auto vad_features = vad_frontend.Accept(samples.data(), samples.size(), final);
    vad_pending.insert(vad_pending.end(), vad_features.begin(), vad_features.end());
    while (vad_pending.size() >= static_cast<size_t>(FLAGS_vad_chunk) ||
           (final && !vad_pending.empty())) {
      const size_t valid = std::min<size_t>(FLAGS_vad_chunk, vad_pending.size());
      std::vector<float> psil;
      if (!run_vad_encoder(vad_module, vad_caches, vad_pending, valid, psil)) {
        return 7;
      }
      vad_pending.erase(vad_pending.begin(), vad_pending.begin() + valid);
      std::vector<float> decibel(valid);
      for (size_t j = 0; j < valid; ++j) {
        decibel[j] = decibel_frame(vad_frames + j);
      }
      const bool is_last = final && vad_pending.empty();
      auto got = vad_decoder.Accept(decibel.data(), psil.data(), valid, is_last);
      for (const auto& seg : got) {
        segments.push_back(seg);
        if (b_reset) {
          segment_texts.push_back(cur_text);
          cur_text.clear();
          reset_paraformer();
        } else if (b_realtime) {
          std::string seg_text;
          while (pf_flush < pf_chunks.size() &&
                 pf_chunks[pf_flush].center_ms < seg.end_ms) {
            seg_text += pf_chunks[pf_flush].text;
            ++pf_flush;
          }
          segment_texts.push_back(seg_text);
          if (!b_diarize) {
            std::printf(
                "SEG %lld %lld %s\n",
                (long long)seg.start_ms,
                (long long)seg.end_ms,
                seg_text.c_str());
            std::fflush(stdout);
          }
        }
        if (b_diarize && (seg.end_ms - seg.start_ms) >= FLAGS_spk_min_ms) {
          int64_t a = seg.start_ms * 16 - sample_base;
          int64_t b = std::min<int64_t>(
              seg.end_ms * 16 - sample_base, static_cast<int64_t>(sample_buf.size()));
          a = std::max<int64_t>(a, int64_t{0});
          if (b > a) {
            auto emb = campplus_embedding(
                *spk_module, sample_buf.data() + a, b - a, spk_T);
            if (!emb.empty()) {
              diar_embs.push_back(std::move(emb));
              diar_index.push_back(segments.size() - 1);
              if (std::getenv("SPK_DEBUG") != nullptr) {
                std::fprintf(stderr, "EMB %zu\n", diar_index.size());
                std::fflush(stderr);
              }
            }
          }
        }
      }
      vad_frames += valid;
    }

    const int64_t keep = 80 * 16000;
    if (static_cast<int64_t>(sample_buf.size()) > keep) {
      const int64_t drop = static_cast<int64_t>(sample_buf.size()) - keep;
      sample_buf.erase(sample_buf.begin(), sample_buf.begin() + drop);
      sample_base += drop;
    }

    pf_queue.push_back(
        {audio_chunks * 600 + 300,
         frontend.Accept(samples.data(), samples.size(), final)});
    ++audio_chunks;
    while (static_cast<int>(pf_queue.size()) > FLAGS_vad_delay_chunks) {
      if ((rc = consume_chunk(
               pf_queue.front().second, false, pf_queue.front().first)) != 0) {
        return rc;
      }
      pf_queue.pop_front();
    }
  }
  while (!pf_queue.empty()) {
    const bool last = pf_queue.size() == 1;
    if ((rc = consume_chunk(
             pf_queue.front().second, last, pf_queue.front().first)) != 0) {
      return rc;
    }
    pf_queue.pop_front();
  }

  if (b_realtime) {
    for (; pf_flush < pf_chunks.size(); ++pf_flush) {
      cur_text += pf_chunks[pf_flush].text;
    }
  } else if (!b_reset) {
    segment_texts.assign(segments.size(), std::string());
    if (segments.empty()) {
      for (const PfChunk& chunk : pf_chunks) {
        cur_text += chunk.text;
      }
    } else {
      // Every chunk goes to the segment nearest its center, so no token is
      // dropped at a boundary; only silence chunks fall in a gap and are empty.
      for (const PfChunk& chunk : pf_chunks) {
        size_t best = 0;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        for (size_t i = 0; i < segments.size(); ++i) {
          const int64_t s = segments[i].start_ms;
          const int64_t e = segments[i].end_ms;
          const int64_t distance = chunk.center_ms < s
              ? s - chunk.center_ms
              : (chunk.center_ms >= e ? chunk.center_ms - e : 0);
          if (distance < best_distance) {
            best_distance = distance;
            best = i;
          }
        }
        segment_texts[best] += chunk.text;
      }
    }
  }

  std::vector<int> spk_labels(segments.size(), -1);
  if (b_diarize && !diar_embs.empty()) {
    const auto labels =
        cluster_offline(diar_embs, static_cast<float>(FLAGS_spk_threshold));
    for (size_t k = 0; k < diar_index.size(); ++k) {
      spk_labels[diar_index[k]] = labels[k];
    }
  }

  if (b_diarize || !b_realtime) {
    for (size_t i = 0; i < segments.size(); ++i) {
      std::printf(
          "SEG %lld %lld",
          (long long)segments[i].start_ms,
          (long long)segments[i].end_ms);
      if (spk_labels[i] >= 0) {
        std::printf(" SPK%d", spk_labels[i]);
      }
      std::printf(" %s\n", segment_texts[i].c_str());
    }
  }
  if (!cur_text.empty()) {
    std::printf("TAIL %s\n", cur_text.c_str());
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start_time).count();
  std::printf("SEGMENTS %zu\n", segments.size());
  std::printf("TIMING audio=%.3fs elapsed=%.3fs rtf=%.3f\n",
              audio.duration_seconds(), seconds, seconds / audio.duration_seconds());
  return 0;
}

// Stateful counterpart used by the TCP protocol. The file path mode above is
// intentionally kept unchanged; this session consumes raw PCM incrementally.
class StreamingSession {
 public:
  explicit StreamingSession(
      std::unordered_map<std::string, std::unique_ptr<Module>>& modules)
      : modules_(modules) {
    std::vector<uint8_t> cmvn_bytes;
    if (!read_file(FLAGS_cmvn, cmvn_bytes) || cmvn_bytes.size() != 1120 * sizeof(float)) {
      return;
    }
    std::vector<float> cmvn(1120);
    std::memcpy(cmvn.data(), cmvn_bytes.data(), cmvn_bytes.size());
    std::ifstream token_file(FLAGS_tokens);
    if (!token_file) {
      return;
    }
    tokens_ = nlohmann::json::parse(token_file).get<std::vector<std::string>>();
    frontend_ = std::make_unique<paraformer::AudioFrontend>(cmvn);
    if (!FLAGS_vad_model.empty()) {
      std::vector<uint8_t> vad_cmvn_bytes;
      if (!read_file(FLAGS_cmvn_vad, vad_cmvn_bytes) ||
          vad_cmvn_bytes.size() != 800 * sizeof(float)) {
        return;
      }
      std::vector<float> vad_cmvn(800);
      std::memcpy(vad_cmvn.data(), vad_cmvn_bytes.data(), vad_cmvn_bytes.size());
      vad_module_ = std::make_unique<Module>(FLAGS_vad_model, Module::LoadMode::Mmap);
      if (vad_module_->load() != Error::Ok || vad_module_->load_forward() != Error::Ok) {
        return;
      }
      vad_frontend_ = std::make_unique<fsmn_vad::VadFrontend>(vad_cmvn);
      vad_caches_.assign(4 * kVadProj * kVadCacheFrames, 0);
      vad_enabled_ = true;
    }
    if (!FLAGS_spk_model.empty()) {
      spk_module_ = std::make_unique<Module>(FLAGS_spk_model, Module::LoadMode::Mmap);
      if (spk_module_->load() != Error::Ok || spk_module_->load_forward() != Error::Ok) {
        return;
      }
      auto smeta = spk_module_->method_meta("forward");
      if (smeta.ok() && smeta.get().num_inputs() >= 1) {
        auto ti = smeta.get().input_tensor_meta(0);
        if (ti.ok() && ti.get().sizes().size() == 3) {
          spk_T_ = ti.get().sizes()[1];
        }
      }
    }
    valid_ = true;
  }

  bool valid() const { return valid_; }
  const std::string& text() const { return text_; }
  size_t audio_bytes() const { return audio_bytes_; }
  size_t chunks() const { return chunk_index_; }

  // Completed VAD segments (start/end ms and the text assigned to them), in
  // order. Drained by the TCP loop after each feed.
  std::vector<DiarSegment> take_segments() {
    std::vector<DiarSegment> out;
    out.swap(segments_out_);
    return out;
  }

  int feed(const uint8_t* pcm_bytes, size_t byte_count, bool final) {
    if (!valid_ || finalized_ || (byte_count & 1)) {
      return 2;
    }
    std::vector<float> samples(byte_count / 2);
    for (size_t i = 0; i < samples.size(); ++i) {
      int16_t sample;
      std::memcpy(&sample, pcm_bytes + i * 2, sizeof(sample));
      samples[i] = sample / 32768.0f;
    }
    audio_bytes_ += byte_count;
    if (final) {
      finalized_ = true;
    }
    // Keep the fbank bounded like AudioReader and batch the encoder's ten
    // frames per invocation. Large TCP payloads must not be handed to the
    // online fbank in one call: its bounded feature queue is not a request
    // buffer.
    for (size_t offset = 0; offset < samples.size() || (final && offset == samples.size());) {
      const size_t count = std::min<size_t>(9600, samples.size() - offset);
      const bool sub_final = final && offset + count == samples.size();
      auto features = frontend_->Accept(samples.data() + offset, count, sub_final);
      pending_features_.insert(pending_features_.end(), features.begin(), features.end());
      while (pending_features_.size() >= 10) {
        std::vector<std::array<float, 560>> batch(
            pending_features_.begin(), pending_features_.begin() + 10);
        pending_features_.erase(pending_features_.begin(), pending_features_.begin() + 10);
        const int status = record_chunk(batch, false);
        if (status != 0) {
          return status;
        }
      }
      if (sub_final) {
        if (!pending_features_.empty()) {
          std::vector<std::array<float, 560>> batch(
              pending_features_.begin(), pending_features_.end());
          pending_features_.clear();
          const int status = record_chunk(batch, true);
          if (status != 0) {
            return status;
          }
        }
        break;
      }
      offset += count;
    }

    if (vad_enabled_) {
      if (feed_vad(samples, final) != 0) {
        return 3;
      }
    }
    return 0;
  }

 private:
  int process_features(const std::vector<std::array<float, 560>>& features, bool final) {
    if (features.empty() || features.size() > 10) {
      return 1;
    }
    if (chunk_index_ * 10 >= (size_t{1} << 40)) {
      return 1;
    }
    enc_input_.resize(kEncoderBaseValues);
    std::vector<uint16_t>& enc_input = enc_input_;
    const int64_t position = static_cast<int64_t>(chunk_index_ * 10);
    enc_input[0] = half_bits(static_cast<float>(std::min<int64_t>(position, 2048)));
    enc_input[1] = half_bits(static_cast<float>(features.size()));
    enc_input[2] = half_bits(static_cast<float>(std::min<size_t>(chunk_index_ * 10, 40)));
    for (size_t row = 0; row < features.size(); ++row) {
      for (size_t col = 0; col < 560; ++col) {
        enc_input[3 + row * 560 + col] = half_bits(features[row][col]);
      }
    }
    std::memcpy(enc_input.data() + 3 + 10 * 560, enc_state_.data(), enc_state_.size());
    std::memcpy(enc_input.data() + 3 + 10 * 560 + enc_state_.size() / 2,
                overlap_.data(), overlap_.size());
    for (size_t digit = 0; digit < 4; ++digit) {
      enc_input[enc_input.size() - 4 + digit] =
          half_bits(static_cast<float>((position >> (digit * 10)) & 1023));
    }
    std::vector<std::vector<uint8_t>> enc;
    if (!run_half_stage(modules_, FLAGS_model_path, enc_input, enc, &position, &encoder_timing_) ||
        enc.size() != 3 || enc[0].size() != 15 * 512 * 2 ||
        enc[1].size() != enc_state_.size() || enc[2].size() != overlap_.size()) {
      return 2;
    }
    enc_state_ = std::move(enc[1]);
    overlap_ = std::move(enc[2]);
    std::vector<uint16_t> pred_input(8195);
    std::memcpy(pred_input.data(), enc[0].data(), enc[0].size());
    pred_input[7680] = half_bits(predictor_state_[0]);
    for (size_t i = 0; i < 512; ++i) {
      pred_input[7681 + i] = half_bits(predictor_state_[i + 1]);
    }
    pred_input[8194] = half_bits(final ? 1.0f : 0.0f);
    std::vector<std::vector<uint8_t>> pred;
    if (!run_half_stage(modules_, FLAGS_predictor, pred_input, pred, nullptr, &predictor_timing_) ||
        pred.size() != 4 || pred[0].size() < 16 * 512 * sizeof(float) || pred[1].size() % 2 ||
        pred[2].size() < sizeof(float) || pred[3].size() < 512 * sizeof(float)) {
      return 3;
    }
    size_t nvalid = 0;
    for (size_t i = 0; i + 1 < pred[1].size(); i += 2) {
      nvalid += read_half(pred[1], i / 2) != 0;
    }
    if (nvalid > 16) {
      return 4;
    }
    predictor_state_[0] = read_float(pred[2], 0);
    for (size_t i = 0; i < 512; ++i) {
      predictor_state_[i + 1] = read_float(pred[3], i);
    }
    if (nvalid) {
      std::vector<uint16_t> dec_input(15 * 512 + 16 * 512 + 16 * 512 * 26 + 32 * 5120 + 1);
      std::memcpy(dec_input.data(), enc[0].data(), enc[0].size());
      for (size_t i = 0; i < nvalid * 512; ++i) {
        dec_input[7680 + i] = half_bits(read_float(pred[0], i));
      }
      std::memcpy(dec_input.data() + 7680 + 16 * 512, fsmn_.data(), fsmn_.size());
      std::memcpy(dec_input.data() + 7680 + 16 * 512 + fsmn_.size() / 2,
                  opt_.data(), opt_.size());
      dec_input.back() = half_bits(static_cast<float>(nvalid));
      std::vector<std::vector<uint8_t>> dec;
      const std::string& decoder = decoder_started_ ? FLAGS_decoder_next : FLAGS_decoder_first;
      StageTiming& timing = decoder_started_ ? decoder_next_timing_ : decoder_first_timing_;
      if (!run_half_stage(modules_, decoder, dec_input, dec, nullptr, &timing) ||
          dec.size() != 49 || dec[0].size() != 16 * 8404 * 2) {
        return 5;
      }
      decoder_started_ = true;
      for (size_t row = 0; row < nvalid; ++row) {
        const uint16_t* logits = reinterpret_cast<const uint16_t*>(
            dec[0].data() + row * 8404 * sizeof(uint16_t));
        size_t best = 0;
        float score = -INFINITY;
        for (size_t col = 0; col < 8404; ++col) {
          const float value = fp16_to_float(logits[col]);
          if (value > score) {
            score = value;
            best = col;
          }
        }
        if (best >= tokens_.size()) {
          return 6;
        }
        if (best != 0 && best != 1 && best != 2) {
          text_ += tokens_[best] == "<space>" ? " " : tokens_[best];
        }
      }
      fsmn_.clear();
      opt_.clear();
      for (size_t i = 1; i < 17; ++i) {
        fsmn_.insert(fsmn_.end(), dec[i].begin(), dec[i].end());
      }
      for (size_t i = 17; i < 49; ++i) {
        opt_.insert(opt_.end(), dec[i].begin(), dec[i].end());
      }
    }
    ++chunk_index_;
    return 0;
  }

  int record_chunk(const std::vector<std::array<float, 560>>& batch, bool final) {
    const int64_t center = static_cast<int64_t>(chunk_index_) * 600 + 300;
    const size_t before = text_.size();
    const int status = process_features(batch, final);
    if (status != 0) {
      return status;
    }
    if (vad_enabled_) {
      pf_chunks_.push_back({center, text_.substr(before)});
    }
    return 0;
  }

  int feed_vad(const std::vector<float>& samples, bool final) {
    all_samples_.insert(all_samples_.end(), samples.begin(), samples.end());
    for (size_t offset = 0; offset < samples.size() || (final && offset == samples.size());) {
      const size_t count = std::min<size_t>(9600, samples.size() - offset);
      const bool sub_final = final && offset + count == samples.size();
      auto features = vad_frontend_->Accept(samples.data() + offset, count, sub_final);
      vad_pending_.insert(vad_pending_.end(), features.begin(), features.end());
      while (vad_pending_.size() >= static_cast<size_t>(FLAGS_vad_chunk) ||
             (sub_final && !vad_pending_.empty())) {
        const size_t valid = std::min<size_t>(FLAGS_vad_chunk, vad_pending_.size());
        std::vector<float> psil;
        if (!run_vad_encoder(*vad_module_, vad_caches_, vad_pending_, valid, psil)) {
          return 1;
        }
        vad_pending_.erase(vad_pending_.begin(), vad_pending_.begin() + valid);
        std::vector<float> decibel(valid);
        for (size_t j = 0; j < valid; ++j) {
          decibel[j] = decibel_at(all_samples_, vad_frames_ + j);
        }
        const bool is_last = sub_final && vad_pending_.empty();
        auto got = vad_decoder_.Accept(decibel.data(), psil.data(), valid, is_last);
        for (const auto& seg : got) {
          std::string seg_text;
          while (pf_flush_ < pf_chunks_.size() &&
                 pf_chunks_[pf_flush_].center_ms < seg.end_ms) {
            seg_text += pf_chunks_[pf_flush_].text;
            ++pf_flush_;
          }
          int speaker = -1;
          if (spk_module_ != nullptr &&
              (seg.end_ms - seg.start_ms) >= FLAGS_spk_min_ms) {
            const int64_t a = seg.start_ms * 16;
            const int64_t b = std::min<int64_t>(seg.end_ms * 16, all_samples_.size());
            if (a >= 0 && b > a) {
              auto emb = campplus_embedding(
                  *spk_module_, all_samples_.data() + a, b - a, spk_T_);
              if (!emb.empty()) {
                float best = -2.0f;
                int bi = -1;
                for (size_t c = 0; c < spk_centroids_.size(); ++c) {
                  const float s = cosine_similarity(emb, spk_centroids_[c]);
                  if (s > best) {
                    best = s;
                    bi = static_cast<int>(c);
                  }
                }
                if (bi >= 0 && best >= static_cast<float>(FLAGS_spk_threshold)) {
                  speaker = bi;
                  auto& centroid = spk_centroids_[bi];
                  for (size_t d = 0; d < centroid.size() && d < emb.size(); ++d) {
                    centroid[d] = 0.7f * centroid[d] + 0.3f * emb[d];
                  }
                  normalize_vector(centroid);
                } else {
                  normalize_vector(emb);
                  speaker = static_cast<int>(spk_centroids_.size());
                  spk_centroids_.push_back(std::move(emb));
                }
              }
            }
          }
          segments_out_.push_back({seg, seg_text, speaker});
        }
        vad_frames_ += valid;
      }
      if (sub_final) {
        break;
      }
      offset += count;
    }
    return 0;
  }

  std::unordered_map<std::string, std::unique_ptr<Module>>& modules_;
  bool valid_ = false;
  bool finalized_ = false;
  bool vad_enabled_ = false;
  std::unique_ptr<Module> vad_module_;
  std::unique_ptr<fsmn_vad::VadFrontend> vad_frontend_;
  std::vector<float> vad_caches_;
  std::vector<std::array<float, kVadFeat>> vad_pending_;
  fsmn_vad::VadDecoder vad_decoder_;
  int64_t vad_frames_ = 0;
  std::vector<float> all_samples_;
  std::vector<PfChunk> pf_chunks_;
  size_t pf_flush_ = 0;
  std::vector<DiarSegment> segments_out_;
  std::unique_ptr<Module> spk_module_;
  int spk_T_ = 300;
  std::vector<std::vector<float>> spk_centroids_;
  bool decoder_started_ = false;
  size_t audio_bytes_ = 0;
  size_t chunk_index_ = 0;
  std::unique_ptr<paraformer::AudioFrontend> frontend_;
  std::vector<std::string> tokens_;
  std::vector<std::array<float, 560>> pending_features_;
  std::vector<uint16_t> enc_input_;
  std::vector<uint8_t> enc_state_ = std::vector<uint8_t>(2 * 50 * 4 * 50 * 128 * 2);
  std::vector<uint8_t> overlap_ = std::vector<uint8_t>(5 * 560 * 2);
  std::vector<uint8_t> fsmn_ = std::vector<uint8_t>(16 * 512 * 26 * 2);
  std::vector<uint8_t> opt_ = std::vector<uint8_t>(32 * 5120 * 2);
  std::array<float, 513> predictor_state_{};
  std::string text_;
  StageTiming encoder_timing_, predictor_timing_, decoder_first_timing_, decoder_next_timing_;
};

bool recv_all(int fd, void* data, size_t size) {
  auto* bytes = static_cast<uint8_t*>(data);
  while (size) {
    const ssize_t got = recv(fd, bytes, size, 0);
    if (got <= 0) {
      return false;
    }
    bytes += got;
    size -= static_cast<size_t>(got);
  }
  return true;
}

bool send_all(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  while (size) {
    const ssize_t sent = send(fd, bytes, size, MSG_NOSIGNAL);
    if (sent <= 0) {
      return false;
    }
    bytes += sent;
    size -= static_cast<size_t>(sent);
  }
  return true;
}

int make_tcp_listener(int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  int reuse = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
      listen(fd, 1) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

bool send_frame(int fd, uint8_t type, const void* data, uint32_t length) {
  return send_all(fd, &type, sizeof(type)) && send_all(fd, &length, sizeof(length)) &&
      (!length || send_all(fd, data, length));
}

int serve_tcp_audio(int port,
                    std::unordered_map<std::string, std::unique_ptr<Module>>& modules) {
  if (port < 1 || port > 65535) {
    ET_LOG(Error, "--tcp_port must be between 1 and 65535");
    return 2;
  }
  const int listener = make_tcp_listener(port);
  if (listener < 0) {
    ET_LOG(Error, "cannot listen on 127.0.0.1:%d: %s", port, std::strerror(errno));
    return 3;
  }
  std::printf("READY tcp=127.0.0.1:%d\n", port);
  std::fflush(stdout);
  constexpr uint32_t kMaxAudioBytes = 256u * 1024u * 1024u;
  while (true) {
    const int client = accept(listener, nullptr, nullptr);
    if (client < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    uint8_t frame_type = 0;
    uint32_t frame_length = 0;
    std::unique_ptr<StreamingSession> session;
    std::chrono::steady_clock::time_point started;
    bool ended = false;
    while (!ended && recv_all(client, &frame_type, sizeof(frame_type)) &&
           recv_all(client, &frame_length, sizeof(frame_length))) {
      if (frame_length > kMaxAudioBytes || (frame_type != 1 && frame_type != 2 && frame_type != 3)) {
        const uint32_t status = 4;
        send_frame(client, 0x7f, &status, sizeof(status));
        break;
      }
      std::vector<uint8_t> payload(frame_length);
      if (frame_length && !recv_all(client, payload.data(), payload.size())) {
        break;
      }
      if (frame_type == 1) {
        if (session) {
          const uint32_t status = 5;
          send_frame(client, 0x7f, &status, sizeof(status));
          break;
        }
        session = std::make_unique<StreamingSession>(modules);
        started = std::chrono::steady_clock::now();
        const uint32_t status = session->valid() ? 0 : 6;
        send_frame(client, 0x80, &status, sizeof(status));
      } else if (frame_type == 2) {
        if (!session || payload.empty() || session->audio_bytes() + payload.size() > kMaxAudioBytes ||
            session->feed(payload.data(), payload.size(), false) != 0) {
          const uint32_t status = 7;
          send_frame(client, 0x7f, &status, sizeof(status));
          break;
        }
        const uint32_t text_length = static_cast<uint32_t>(session->text().size());
        send_frame(client, 0x81, session->text().data(), text_length);
        for (const auto& segment : session->take_segments()) {
          const uint32_t start = static_cast<uint32_t>(segment.span.start_ms);
          const uint32_t end = static_cast<uint32_t>(segment.span.end_ms);
          const int32_t speaker = static_cast<int32_t>(segment.speaker);
          const uint32_t length = static_cast<uint32_t>(segment.text.size());
          std::vector<uint8_t> payload(16 + length);
          std::memcpy(payload.data(), &start, sizeof(start));
          std::memcpy(payload.data() + 4, &end, sizeof(end));
          std::memcpy(payload.data() + 8, &speaker, sizeof(speaker));
          std::memcpy(payload.data() + 12, &length, sizeof(length));
          std::memcpy(payload.data() + 16, segment.text.data(), length);
          send_frame(client, 0x83, payload.data(), static_cast<uint32_t>(payload.size()));
        }
      } else {
        if (!session || session->feed(nullptr, 0, true) != 0 || session->audio_bytes() == 0) {
          const uint32_t status = 8;
          send_frame(client, 0x7f, &status, sizeof(status));
          break;
        }
        const uint32_t elapsed_ms = static_cast<uint32_t>(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() * 1000.0 + 0.5);
        const uint32_t status = 0;
        const uint32_t audio_ms = static_cast<uint32_t>(session->audio_bytes() / 32);
        const uint32_t text_length = static_cast<uint32_t>(session->text().size());
        std::vector<uint8_t> result(16 + text_length);
        std::memcpy(result.data(), &status, sizeof(uint32_t));
        std::memcpy(result.data() + 4, &elapsed_ms, sizeof(uint32_t));
        std::memcpy(result.data() + 8, &audio_ms, sizeof(uint32_t));
        std::memcpy(result.data() + 12, &text_length, sizeof(uint32_t));
        std::memcpy(result.data() + 16, session->text().data(), text_length);
        for (const auto& segment : session->take_segments()) {
          const uint32_t start = static_cast<uint32_t>(segment.span.start_ms);
          const uint32_t end = static_cast<uint32_t>(segment.span.end_ms);
          const int32_t speaker = static_cast<int32_t>(segment.speaker);
          const uint32_t length = static_cast<uint32_t>(segment.text.size());
          std::vector<uint8_t> payload(16 + length);
          std::memcpy(payload.data(), &start, sizeof(start));
          std::memcpy(payload.data() + 4, &end, sizeof(end));
          std::memcpy(payload.data() + 8, &speaker, sizeof(speaker));
          std::memcpy(payload.data() + 12, &length, sizeof(length));
          std::memcpy(payload.data() + 16, segment.text.data(), length);
          send_frame(client, 0x83, payload.data(), static_cast<uint32_t>(payload.size()));
        }
        send_frame(client, 0x82, result.data(), static_cast<uint32_t>(result.size()));
        ended = true;
      }
    }
    close(client);
  }
  close(listener);
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_tcp_port != 0) {
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    return serve_tcp_audio(FLAGS_tcp_port, modules);
  }

  if (!FLAGS_audio.empty()) {
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    if (!FLAGS_vad_model.empty()) {
      return transcribe_audio_vad(FLAGS_audio, modules);
    }
    return transcribe_audio(FLAGS_audio, modules);
  }

  if (FLAGS_serve_audio) {
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    std::string audio_path;
    std::puts("READY");
    std::fflush(stdout);
    while (std::getline(std::cin, audio_path)) {
      if (audio_path == "QUIT") {
        break;
      }
      const int status = transcribe_audio(audio_path, modules);
      std::printf("DONE %d\n", status);
      std::fflush(stdout);
    }
    return 0;
  }

  if (FLAGS_serve) {
    std::unordered_map<std::string, std::unique_ptr<Module>> modules;
    std::string line;
    std::puts("READY");
    std::fflush(stdout);
    while (std::getline(std::cin, line)) {
      if (line == "QUIT") {
        break;
      }
      const size_t first = line.find('\t');
      const size_t second = first == std::string::npos ? first : line.find('\t', first + 1);
      int status = 2;
      if (second != std::string::npos) {
        const std::string path = line.substr(0, first);
        const std::string inputs = line.substr(first + 1, second - first - 1);
        const std::string output = line.substr(second + 1);
        auto& module = modules[path];
        if (!module) {
          module = std::make_unique<Module>(path, Module::LoadMode::Mmap);
          if (module->load() != Error::Ok || module->load_forward() != Error::Ok) {
            module.reset();
            status = 3;
          }
        }
        if (module) {
          auto meta = module->method_meta("forward");
          status = meta.ok() ? execute_stage(*module, meta.get(), inputs, output, 1) : 5;
        }
      }
      std::printf("DONE %d\n", status);
      std::fflush(stdout);
    }
    return 0;
  }

  if (FLAGS_model_path.empty()) {
    ET_LOG(Error, "--model_path is required");
    return 2;
  }
  auto module = std::make_unique<Module>(FLAGS_model_path, Module::LoadMode::Mmap);
  if (module->load() != Error::Ok) {
    ET_LOG(Error, "failed to load %s", FLAGS_model_path.c_str());
    return 3;
  }
  if (module->load_forward() != Error::Ok) {
    ET_LOG(Error, "failed to load forward of %s", FLAGS_model_path.c_str());
    return 4;
  }
  auto mmeta = module->method_meta("forward");
  if (!mmeta.ok()) {
    ET_LOG(Error, "no forward method in %s", FLAGS_model_path.c_str());
    return 5;
  }
  const MethodMeta& meta = mmeta.get();
  std::printf("MODEL %s\n", FLAGS_model_path.c_str());
  std::printf("INPUTS %zu\n", meta.num_inputs());
  for (size_t i = 0; i < meta.num_inputs(); i++) {
    auto ti = meta.input_tensor_meta(i);
    if (!ti.ok()) {
      std::printf("  in %zu non-tensor\n", i);
      continue;
    }
    std::printf("  in %zu %s %s %s\n", i, shape_str(ti.get()).c_str(),
                dtype_name(ti.get().scalar_type()),
                std::string(ti.get().name()).c_str());
  }
  std::printf("OUTPUTS %zu\n", meta.num_outputs());
  for (size_t i = 0; i < meta.num_outputs(); i++) {
    auto ti = meta.output_tensor_meta(i);
    if (!ti.ok()) {
      std::printf("  out %zu non-tensor\n", i);
      continue;
    }
    std::printf("  out %zu %s %s %s\n", i, shape_str(ti.get()).c_str(),
                dtype_name(ti.get().scalar_type()),
                std::string(ti.get().name()).c_str());
  }
  std::fflush(stdout);
  return FLAGS_sig ? 0 : execute_stage(*module, meta, FLAGS_inputs, FLAGS_output_file,
                                       FLAGS_num_executions);
}
