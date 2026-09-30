/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace fsmn_vad {

// WavFrontendOnline for the FSMN-Monophone VAD: 16 kHz, hamming, 25 ms frame,
// 10 ms shift, dither 0, 80 mel bins, LFR m=5 n=1, then CMVN. Each emitted row
// is 400 values; row i is fbank frames [i-2, i+2] with the first and last frame
// repeated at the edges, then (x + mean) * var.
class VadFrontend {
 public:
  explicit VadFrontend(const std::vector<float>& cmvn);

  std::vector<std::array<float, 400>> Accept(
      const float* samples,
      size_t count,
      bool final);

 private:
  void Emit(std::vector<std::array<float, 400>>& output);

  std::array<float, 400> means_;
  std::array<float, 400> vars_;
  knf::OnlineFbank fbank_;
  std::deque<std::array<float, 80>> pending_;
  int32_t next_frame_ = 0;
  size_t emitted_ = 0;
  bool initialized_ = false;
  bool finished_ = false;
};

} // namespace fsmn_vad
