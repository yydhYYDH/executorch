/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fsmn_vad_frontend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace fsmn_vad {
namespace {

constexpr int kNumBins = 80;
constexpr int kLfrM = 5;
constexpr int kLfrN = 1;

knf::FbankOptions FbankOptions() {
  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = 16000;
  opts.frame_opts.frame_length_ms = 25;
  opts.frame_opts.frame_shift_ms = 10;
  opts.frame_opts.window_type = "hamming";
  opts.frame_opts.dither = 0;
  opts.frame_opts.max_feature_vectors = 256;
  opts.mel_opts.num_bins = kNumBins;
  opts.energy_floor = 0;
  return opts;
}

} // namespace

VadFrontend::VadFrontend(const std::vector<float>& cmvn)
    : fbank_(FbankOptions()) {
  if (cmvn.size() != 800) {
    throw std::invalid_argument("CMVN must contain 800 float32 values");
  }
  std::copy_n(cmvn.begin(), 400, means_.begin());
  std::copy_n(cmvn.begin() + 400, 400, vars_.begin());
}

void VadFrontend::Emit(std::vector<std::array<float, 400>>& output) {
  std::array<float, 400> features;
  const size_t available = std::min<size_t>(pending_.size(), kLfrM);
  for (size_t frame = 0; frame < kLfrM; ++frame) {
    const std::array<float, 80>& source =
        pending_[std::min(frame, available - 1)];
    std::copy(source.begin(), source.end(), features.begin() + frame * kNumBins);
  }
  for (size_t i = 0; i < features.size(); ++i) {
    features[i] = (features[i] + means_[i]) * vars_[i];
  }
  output.push_back(features);
}

std::vector<std::array<float, 400>> VadFrontend::Accept(
    const float* samples,
    size_t count,
    bool final) {
  if (finished_) {
    throw std::logic_error("audio frontend already finalized");
  }
  std::vector<float> scaled(count);
  for (size_t i = 0; i < count; ++i) {
    scaled[i] = samples[i] * 32768.0f;
  }
  fbank_.AcceptWaveform(16000, scaled.data(), static_cast<int32_t>(count));
  if (final) {
    fbank_.InputFinished();
    finished_ = true;
  }
  const int32_t ready = fbank_.NumFramesReady();
  for (; next_frame_ < ready; ++next_frame_) {
    std::array<float, 80> frame;
    std::copy_n(fbank_.GetFrame(next_frame_), kNumBins, frame.begin());
    if (!initialized_) {
      // Left context: (lfr_m - 1) / 2 repeats of the first frame.
      for (int i = 0; i < (kLfrM - 1) / 2; ++i) {
        pending_.push_back(frame);
      }
      initialized_ = true;
    }
    pending_.push_back(frame);
  }
  std::vector<std::array<float, 400>> output;
  while (pending_.size() >= static_cast<size_t>(kLfrM)) {
    Emit(output);
    ++emitted_;
    for (int i = 0; i < kLfrN; ++i) {
      pending_.pop_front();
    }
  }
  if (final) {
    // Right context: repeat the last frame until every input frame has an
    // output row (the offline frontend pads (lfr_m - 1) / 2 rows this way).
    while (emitted_ < static_cast<size_t>(next_frame_) && !pending_.empty()) {
      Emit(output);
      ++emitted_;
      pending_.pop_front();
    }
  }
  return output;
}

} // namespace fsmn_vad
