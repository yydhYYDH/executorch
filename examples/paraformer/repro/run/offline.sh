#!/usr/bin/env bash
# Offline: run a whole meeting recording on the phone in one shot.
#
#   bash run/offline.sh meeting.wav [--spk] [--vad]
#
# The runner computes fbank/LFR/CMVN on the device and drives encoder,
# predictor and decoder chunk by chunk.  Prints the transcript, and (with VAD /
# speaker enabled) one SEG line per utterance with its speaker.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
if [[ -f "$REPRO/config.sh" ]]; then source "$REPRO/config.sh"
else source "$REPRO/config.example.sh"; fi

AUDIO="${1:?usage: offline.sh AUDIO [--spk] [--vad]}"; shift || true
EXTRA=""
for arg in "$@"; do
  case "$arg" in
    --spk) EXTRA="$EXTRA --spk_model camppluscpu/campplus_cpu.pte" ;;
    --vad) EXTRA="$EXTRA --vad_model fsmn/fsmn_vad_enc_T20.pte --cmvn_vad fsmn/cmvn.bin" ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

REMOTE_AUDIO="/data/local/tmp/$(basename "$AUDIO")"
scp -q "$AUDIO" "$PHONE:$REMOTE_AUDIO"
ssh "$PHONE" "cd '$PHONE_DIR' && \
  LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH=\$PWD/skel \
  ./paraformer_runner_spk --audio '$REMOTE_AUDIO' \
    --cmvn cmvn.bin --tokens tokens.json \
    --model_path encp_one.pte --predictor pred15.pte \
    --decoder_first decoder16_first.pte --decoder_next decoder16_next.pte \
    $EXTRA"
