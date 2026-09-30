#include "audio_frontend.h"

#include <algorithm>
#include <stdexcept>

namespace paraformer {
namespace {

knf::FbankOptions FbankOptions() {
  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = 16000;
  opts.frame_opts.frame_length_ms = 25;
  opts.frame_opts.frame_shift_ms = 10;
  opts.frame_opts.window_type = "hamming";
  opts.frame_opts.dither = 0;
  opts.frame_opts.max_feature_vectors = 256;
  opts.mel_opts.num_bins = 80;
  opts.energy_floor = 0;
  return opts;
}

} // namespace

AudioFrontend::AudioFrontend(const std::vector<float>& cmvn)
    : fbank_(FbankOptions()) {
  if (cmvn.size() != 1120) {
    throw std::invalid_argument("CMVN must contain 1120 float32 values");
  }
  std::copy_n(cmvn.begin(), 560, means_.begin());
  std::copy_n(cmvn.begin() + 560, 560, vars_.begin());
}

void AudioFrontend::Emit(std::vector<std::array<float, 560>>& output) {
  std::array<float, 560> features;
  for (size_t frame = 0; frame < 7; ++frame) {
    std::copy(pending_[frame].begin(), pending_[frame].end(),
              features.begin() + frame * 80);
  }
  for (size_t i = 0; i < features.size(); ++i) {
    features[i] = (features[i] + means_[i]) * vars_[i];
  }
  output.push_back(features);
  for (int i = 0; i < 6; ++i) {
    pending_.pop_front();
  }
}

std::vector<std::array<float, 560>> AudioFrontend::Accept(
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
    std::copy_n(fbank_.GetFrame(next_frame_), 80, frame.begin());
    if (!initialized_) {
      for (int i = 0; i < 3; ++i) {
        pending_.push_back(frame);
      }
      initialized_ = true;
    }
    pending_.push_back(frame);
  }
  std::vector<std::array<float, 560>> output;
  while (pending_.size() >= 7) {
    Emit(output);
  }
  if (final && pending_.size() > 3) {
    while (pending_.size() < 7) {
      pending_.push_back(pending_.back());
    }
    Emit(output);
  }
  return output;
}

} // namespace paraformer
