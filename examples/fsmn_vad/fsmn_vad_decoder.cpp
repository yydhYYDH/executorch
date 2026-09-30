/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fsmn_vad_decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fsmn_vad {

VadDecoder::VadDecoder() {
  win_size_frame_ = static_cast<int>(kWindowSizeMs / kFrameInMs);
  win_state_.assign(win_size_frame_, 0);
  sil_to_speech_frmcnt_thres_ = static_cast<int>(kSilToSpeechMs / kFrameInMs);
  speech_to_sil_frmcnt_thres_ = static_cast<int>(kSpeechToSilMs2 / kFrameInMs);
}

int64_t VadDecoder::LatencyFrmNumAtStartPoint() const {
  int64_t latency = win_size_frame_;
  if (kDoExtend) {
    latency += kLookBackStartMs / kFrameInMs;
  }
  return latency;
}

std::vector<VadSegment> VadDecoder::Accept(
    const float* decibel,
    const float* psil,
    int n,
    bool final) {
  for (int i = 0; i < n; ++i) {
    decibel_.push_back(decibel[i]);
    psil_.push_back(psil[i]);
  }
  frm_cnt_ += n;
  if (!final) {
    DetectCommonFrames(n);
  } else {
    DetectLastFrames(n);
  }
  return Collect();
}

std::vector<VadSegment> VadDecoder::Collect() {
  std::vector<VadSegment> out;
  for (Seg& seg : out_buf_) {
    if (seg.has_start && seg.has_end && !seg.emitted) {
      seg.emitted = true;
      out.push_back({seg.start_ms, seg.end_ms});
    }
  }
  out_buf_.erase(
      std::remove_if(
          out_buf_.begin(),
          out_buf_.end(),
          [](const Seg& s) { return s.emitted; }),
      out_buf_.end());
  return out;
}

VadDecoder::Frm VadDecoder::GetFrameState(int64_t t) {
  if (t < 0 || t >= static_cast<int64_t>(decibel_.size())) {
    return Frm::Sil;
  }
  const float cur_decibel = decibel_[t];
  const float cur_snr = cur_decibel - noise_average_decibel_;
  if (cur_decibel < kDecibelThres) {
    return Frm::Sil;
  }
  const float sum_score = psil_[t];
  const float noise_prob =
      sum_score > 0.0f ? std::log(sum_score) * kSpeech2NoiseRatio
                       : -std::numeric_limits<float>::infinity();
  const float speech_sum = 1.0f - sum_score;
  const float speech_prob = speech_sum > 0.0f
      ? std::log(speech_sum)
      : -std::numeric_limits<float>::infinity();
  if (std::exp(speech_prob) >= std::exp(noise_prob) + kSpeechNoiseThres &&
      cur_snr >= kSnrThres && cur_decibel >= kDecibelThres) {
    return Frm::Speech;
  }
  if (noise_average_decibel_ < -99.9f) {
    noise_average_decibel_ = cur_decibel;
  } else {
    noise_average_decibel_ =
        (cur_decibel + noise_average_decibel_ * 99.0f) / 100.0f;
  }
  return Frm::Sil;
}

VadDecoder::Change VadDecoder::WindowDetect(Frm state) {
  int cur = 0;
  if (state == Frm::Speech) {
    cur = 1;
  } else if (state == Frm::Sil) {
    cur = 0;
  } else {
    return Change::Invalid;
  }
  win_sum_ -= win_state_[cur_win_pos_];
  win_sum_ += cur;
  win_state_[cur_win_pos_] = cur;
  cur_win_pos_ = (cur_win_pos_ + 1) % win_size_frame_;

  if (pre_frame_state_ == Frm::Sil && win_sum_ >= sil_to_speech_frmcnt_thres_) {
    pre_frame_state_ = Frm::Speech;
    return Change::Sil2Speech;
  }
  if (pre_frame_state_ == Frm::Speech &&
      win_sum_ <= speech_to_sil_frmcnt_thres_) {
    pre_frame_state_ = Frm::Sil;
    return Change::Speech2Sil;
  }
  if (pre_frame_state_ == Frm::Sil) {
    return Change::Sil2Sil;
  }
  if (pre_frame_state_ == Frm::Speech) {
    return Change::Speech2Speech;
  }
  return Change::Invalid;
}

void VadDecoder::DetectCommonFrames(int block) {
  if (vad_state_ == 3) {
    return;
  }
  for (int i = block - 1; i >= 0; --i) {
    const int64_t idx = frm_cnt_ - 1 - i;
    const Frm state = GetFrameState(idx - last_drop_frames_);
    DetectOneFrame(state, idx, false);
  }
}

void VadDecoder::DetectLastFrames(int block) {
  if (vad_state_ == 3) {
    return;
  }
  for (int i = block - 1; i >= 0; --i) {
    const int64_t idx = frm_cnt_ - 1 - i;
    const Frm state = GetFrameState(idx - last_drop_frames_);
    if (i != 0) {
      DetectOneFrame(state, idx, false);
    } else {
      DetectOneFrame(state, frm_cnt_ - 1, true);
    }
  }
}

void VadDecoder::DetectOneFrame(
    Frm cur_frm_state,
    int64_t cur_frm_idx,
    bool is_final_frame) {
  const Change change = WindowDetect(cur_frm_state);
  const int64_t frm_shift_in_ms = kFrameInMs;

  if (change == Change::Sil2Speech) {
    continous_silence_frame_count_ = 0;
    int64_t start_frame = 0;
    if (vad_state_ == 1) {
      start_frame = std::max(
          data_buf_start_frame_, cur_frm_idx - LatencyFrmNumAtStartPoint());
      OnVoiceStart(start_frame, false);
      vad_state_ = 2;
      for (int64_t t = start_frame + 1; t <= cur_frm_idx; ++t) {
        OnVoiceDetected(t);
      }
    } else if (vad_state_ == 2) {
      for (int64_t t = latest_confirmed_speech_frame_ + 1; t < cur_frm_idx;
           ++t) {
        OnVoiceDetected(t);
      }
      if (cur_frm_idx - confirmed_start_frame_ + 1 >
          kMaxSingleSegmentMs / frm_shift_in_ms) {
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_ = 3;
      } else if (!is_final_frame) {
        OnVoiceDetected(cur_frm_idx);
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
    }
  } else if (change == Change::Speech2Sil) {
    continous_silence_frame_count_ = 0;
    if (vad_state_ == 2) {
      if (cur_frm_idx - confirmed_start_frame_ + 1 >
          kMaxSingleSegmentMs / frm_shift_in_ms) {
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_ = 3;
      } else if (!is_final_frame) {
        OnVoiceDetected(cur_frm_idx);
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
    }
  } else if (change == Change::Speech2Speech) {
    continous_silence_frame_count_ = 0;
    if (vad_state_ == 2) {
      if (cur_frm_idx - confirmed_start_frame_ + 1 >
          kMaxSingleSegmentMs / frm_shift_in_ms) {
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_ = 3;
      } else if (!is_final_frame) {
        OnVoiceDetected(cur_frm_idx);
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
    }
  } else if (change == Change::Sil2Sil) {
    continous_silence_frame_count_ += 1;
    if (vad_state_ == 1) {
      if (is_final_frame && number_end_time_detected_ == 0) {
        for (int64_t t = lastest_confirmed_silence_frame_ + 1; t < cur_frm_idx;
             ++t) {
          OnSilenceDetected(t);
        }
        OnVoiceStart(0, true);
        OnVoiceEnd(0, true, false);
        vad_state_ = 3;
      } else {
        if (cur_frm_idx >= LatencyFrmNumAtStartPoint()) {
          OnSilenceDetected(
              cur_frm_idx - LatencyFrmNumAtStartPoint());
        }
      }
    } else if (vad_state_ == 2) {
      if (continous_silence_frame_count_ * frm_shift_in_ms >=
          kMaxEndSilFrameCntThresh) {
        int64_t lookback_frame = kMaxEndSilFrameCntThresh / frm_shift_in_ms;
        if (kDoExtend) {
          lookback_frame -= kLookAheadEndMs / frm_shift_in_ms;
          lookback_frame -= 1;
          lookback_frame = std::max<int64_t>(0, lookback_frame);
        }
        OnVoiceEnd(cur_frm_idx - lookback_frame, false, false);
        vad_state_ = 3;
      } else if (
          cur_frm_idx - confirmed_start_frame_ + 1 >
          kMaxSingleSegmentMs / frm_shift_in_ms) {
        OnVoiceEnd(cur_frm_idx, false, false);
        vad_state_ = 3;
      } else if (kDoExtend && !is_final_frame) {
        if (continous_silence_frame_count_ <= kLookAheadEndMs / frm_shift_in_ms) {
          OnVoiceDetected(cur_frm_idx);
        }
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx);
      }
    }
  }

  if (vad_state_ == 3 && kDetectModeMultiple) {
    ResetDetection();
  }
}

void VadDecoder::OnSilenceDetected(int64_t valid_frame) {
  lastest_confirmed_silence_frame_ = valid_frame;
  if (vad_state_ == 1) {
    PopDataBufTillFrame(valid_frame);
  }
}

void VadDecoder::OnVoiceDetected(int64_t valid_frame) {
  latest_confirmed_speech_frame_ = valid_frame;
  PopDataToOutputBuf(valid_frame, 1, false, false);
}

void VadDecoder::OnVoiceStart(int64_t start_frame, bool fake_result) {
  if (confirmed_start_frame_ != -1) {
    return;
  }
  confirmed_start_frame_ = start_frame;
  if (!fake_result && vad_state_ == 1) {
    PopDataToOutputBuf(start_frame, 1, true, false);
  }
}

void VadDecoder::OnVoiceEnd(
    int64_t end_frame,
    bool fake_result,
    bool is_last_frame) {
  for (int64_t t = latest_confirmed_speech_frame_ + 1; t < end_frame; ++t) {
    OnVoiceDetected(t);
  }
  if (confirmed_end_frame_ != -1) {
    return;
  }
  confirmed_end_frame_ = end_frame;
  if (!fake_result) {
    PopDataToOutputBuf(end_frame, 1, false, true);
  }
  number_end_time_detected_ += 1;
}

void VadDecoder::MaybeOnVoiceEndIfLastFrame(
    bool is_final_frame,
    int64_t cur_frm_idx) {
  if (is_final_frame) {
    OnVoiceEnd(cur_frm_idx, false, true);
    vad_state_ = 3;
  }
}

void VadDecoder::PopDataBufTillFrame(int64_t frame_idx) {
  while (data_buf_start_frame_ < frame_idx) {
    data_buf_start_frame_ += 1;
  }
}

void VadDecoder::PopDataToOutputBuf(
    int64_t start_frm,
    int64_t frm_cnt,
    bool first_frm_is_start_point,
    bool last_frm_is_end_point) {
  PopDataBufTillFrame(start_frm);
  if (out_buf_.empty() || first_frm_is_start_point) {
    out_buf_.push_back(Seg{});
    out_buf_.back().start_ms = start_frm * kFrameInMs;
    out_buf_.back().end_ms = out_buf_.back().start_ms;
  }
  Seg& cur = out_buf_.back();
  data_buf_start_frame_ += frm_cnt;
  cur.end_ms = (start_frm + frm_cnt) * kFrameInMs;
  if (first_frm_is_start_point) {
    cur.has_start = true;
  }
  if (last_frm_is_end_point) {
    cur.has_end = true;
  }
}

void VadDecoder::DropCachedFrames(int64_t drop_frames) {
  const int64_t real = drop_frames - last_drop_frames_;
  if (real <= 0) {
    return;
  }
  decibel_.erase(decibel_.begin(), decibel_.begin() + real);
  psil_.erase(psil_.begin(), psil_.begin() + real);
  last_drop_frames_ = drop_frames;
}

void VadDecoder::ResetDetection() {
  const int64_t ended_at = confirmed_end_frame_;
  continous_silence_frame_count_ = 0;
  latest_confirmed_speech_frame_ = 0;
  lastest_confirmed_silence_frame_ = -1;
  confirmed_start_frame_ = -1;
  confirmed_end_frame_ = -1;
  vad_state_ = 1;
  win_sum_ = 0;
  std::fill(win_state_.begin(), win_state_.end(), 0);
  cur_win_pos_ = 0;
  pre_frame_state_ = Frm::Sil;

  if (ended_at != -1) {
    DropCachedFrames(ended_at + 1);
  }
}

} // namespace fsmn_vad
