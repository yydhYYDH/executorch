/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace fsmn_vad {

struct VadSegment {
  int64_t start_ms = 0;
  int64_t end_ms = 0;
};

// Port of FunASR's FsmnVADStreaming endpoint state machine (the static
// threshold path, dynamic_silence=False). It consumes the per-frame silence
// probability and the per-frame waveform decibel, and yields completed speech
// segments [start_ms, end_ms] as they are detected. The waveform buffer, score
// buffer and scorer machinery of the Python model are not needed here because
// only timestamps are produced; the frame index arithmetic is kept identical so
// the segments match FunASR.
class VadDecoder {
 public:
  // `max_end_silence_ms` is the FunASR max_end_silence_time: how long the
  // speaker may pause before the current utterance is closed. Larger values
  // give longer, fewer segments.
  explicit VadDecoder(int64_t max_end_silence_ms = 800);

  // Feed `n` consecutive frames. `decibel[t]` is 10*log10(sum of the 25 ms
  // window squared) and `psil[t]` is the encoder's silence probability
  // (class 0). `final` runs the last-frame path and flushes.
  std::vector<VadSegment> Accept(
      const float* decibel,
      const float* psil,
      int n,
      bool final);

 private:
  enum class Frm { Invalid = -1, Speech = 1, Sil = 0 };
  enum class Change {
    Speech2Speech,
    Speech2Sil,
    Sil2Sil,
    Sil2Speech,
    NoBegin,
    Invalid,
  };
  struct Seg {
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    bool has_start = false;
    bool has_end = false;
    bool emitted = false;
  };

  static Frm FrameState(bool speech) { return speech ? Frm::Speech : Frm::Sil; }

  Frm GetFrameState(int64_t t);
  Change WindowDetect(Frm state);
  void DetectCommonFrames(int block);
  void DetectLastFrames(int block);
  void DetectOneFrame(Frm state, int64_t cur_frm_idx, bool is_final_frame);
  void OnSilenceDetected(int64_t valid_frame);
  void OnVoiceDetected(int64_t valid_frame);
  void OnVoiceStart(int64_t start_frame, bool fake_result);
  void OnVoiceEnd(int64_t end_frame, bool fake_result, bool is_last_frame);
  void MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int64_t cur_frm_idx);
  void PopDataBufTillFrame(int64_t frame_idx);
  void PopDataToOutputBuf(
      int64_t start_frm,
      int64_t frm_cnt,
      bool first_frm_is_start_point,
      bool last_frm_is_end_point);
  void DropCachedFrames(int64_t drop_frames);
  void ResetDetection();
  int64_t LatencyFrmNumAtStartPoint() const;
  std::vector<VadSegment> Collect();

  // VADXOptions (static path).
  static constexpr int64_t kFrameInMs = 10;
  static constexpr int64_t kMaxEndSilenceMs = 800;
  static constexpr int64_t kSpeechToSilMs = 150;
  static constexpr int64_t kMaxEndSilFrameCntThresh =
      kMaxEndSilenceMs - kSpeechToSilMs;
  static constexpr int64_t kMaxSingleSegmentMs = 60000;
  static constexpr int64_t kLookAheadEndMs = 100;
  static constexpr int64_t kLookBackStartMs = 200;
  static constexpr float kSpeechNoiseThres = 0.6f;
  static constexpr float kSpeech2NoiseRatio = 1.0f;
  static constexpr float kDecibelThres = -100.0f;
  static constexpr float kSnrThres = -100.0f;
  static constexpr int64_t kWindowSizeMs = 200;
  static constexpr int64_t kSilToSpeechMs = 150;
  static constexpr int64_t kSpeechToSilMs2 = 150;
  static constexpr bool kDetectModeMultiple = true;
  static constexpr bool kDoExtend = true;

  // WindowDetector.
  int win_size_frame_ = 0;
  int win_sum_ = 0;
  std::vector<int> win_state_;
  int cur_win_pos_ = 0;
  Frm pre_frame_state_ = Frm::Sil;
  int sil_to_speech_frmcnt_thres_ = 0;
  int speech_to_sil_frmcnt_thres_ = 0;
  int64_t max_end_sil_frame_cnt_thresh_ = kMaxEndSilenceMs - kSpeechToSilMs;

  // Stats.
  int64_t data_buf_start_frame_ = 0;
  int64_t frm_cnt_ = 0;
  int64_t latest_confirmed_speech_frame_ = 0;
  int64_t lastest_confirmed_silence_frame_ = -1;
  int64_t continous_silence_frame_count_ = 0;
  int vad_state_ = 1; // 1 start-not-detected, 2 in-speech, 3 end-detected
  int64_t confirmed_start_frame_ = -1;
  int64_t confirmed_end_frame_ = -1;
  int64_t number_end_time_detected_ = 0;
  float noise_average_decibel_ = -100.0f;
  int64_t last_drop_frames_ = 0;
  std::vector<float> decibel_;
  std::vector<float> psil_;
  std::vector<Seg> out_buf_;
};

} // namespace fsmn_vad
