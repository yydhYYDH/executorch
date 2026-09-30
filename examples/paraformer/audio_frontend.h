#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace paraformer {

class AudioFrontend {
 public:
  explicit AudioFrontend(const std::vector<float>& cmvn);

  std::vector<std::array<float, 560>> Accept(
      const float* samples,
      size_t count,
      bool final);

 private:
  void Emit(std::vector<std::array<float, 560>>& output);

  std::array<float, 560> means_;
  std::array<float, 560> vars_;
  knf::OnlineFbank fbank_;
  std::deque<std::array<float, 80>> pending_;
  int32_t next_frame_ = 0;
  bool initialized_ = false;
  bool finished_ = false;
};

} // namespace paraformer
