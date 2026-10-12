#!/usr/bin/env bash
# Export every PTE this project needs (VAD + Paraformer + CAM++).
#
#   cp config.example.sh config.sh && $EDITOR config.sh
#   bash reproduce.sh                 # uses the shipped calibration constants
#   bash reproduce.sh --recalibrate   # re-measure them from the model instead
#   bash reproduce.sh --download      # fetch the models first if missing
#
# Produces $OUT_DIR/{encp_one.pte,pred15.pte,decoder16_first.pte,
#                    decoder16_next.pte,cmvn.bin,tokens.json,
#                    fsmn/,campplus_cpu.pte}
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

if [[ -f "$HERE/config.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HERE/config.sh"
else
  echo "note: no config.sh; using config.example.sh defaults" >&2
  # shellcheck source=/dev/null
  source "$HERE/config.example.sh"
fi

RECALIBRATE=0
DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --recalibrate) RECALIBRATE=1 ;;
    --download) DOWNLOAD=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR" "$OUT_DIR/diag"

if [[ "$DOWNLOAD" == 1 ]]; then
  python "$HERE/download_models.py" --into "$MODELS_DIR"
fi

for d in "$PARAFORMER_DIR" "$FSMN_VAD_DIR" "$CAMPP_PLUS_DIR"; do
  [[ -d "$d" ]] || { echo "missing model dir: $d (run with --download)" >&2; exit 1; }
done

echo "== 1/6 FSMN-VAD =="
python "$ROOT/examples/fsmn_vad/export_fsmn_vad.py" "$FSMN_VAD_DIR" \
  --chunk 20 --out "$OUT_DIR/fsmn/fsmn_vad_enc_T20.pte"
python "$ROOT/examples/fsmn_vad/prepare_assets.py" "$FSMN_VAD_DIR" "$OUT_DIR/fsmn"

if [[ "$RECALIBRATE" == 1 ]]; then
  echo "== 2/6 calibrate (re-measure per-layer ranges) =="
  python "$HERE/export/calibrate.py" \
    --model-dir "$PARAFORMER_DIR" --calib-dir "$HERE/calib"
else
  echo "== 2/6 calibration: using shipped $HERE/calib =="
fi

echo "== 3/6 Paraformer encoder =="
python "$HERE/export/export_encoder.py" \
  --model-dir "$PARAFORMER_DIR" --calib-dir "$HERE/calib" \
  --out "$OUT_DIR/encp_one.pte" --diag-dir "$OUT_DIR/diag"

echo "== 4/6 Paraformer predictor + decoders =="
python "$HERE/export/make_example_inputs.py" \
  --model-dir "$PARAFORMER_DIR" --out-dir "$OUT_DIR/diag"
python "$HERE/export/export_predictor.py" \
  --model-dir "$PARAFORMER_DIR" --out "$OUT_DIR/pred15.pte"
python "$HERE/export/export_decoder.py" \
  --model-dir "$PARAFORMER_DIR" --input "$OUT_DIR/diag/decoder_input.bin" \
  --first --out "$OUT_DIR/decoder16_first.pte"
python "$HERE/export/export_decoder.py" \
  --model-dir "$PARAFORMER_DIR" --input "$OUT_DIR/diag/decoder_input.bin" \
  --out "$OUT_DIR/decoder16_next.pte"

echo "== 5/6 frontend assets =="
python "$ROOT/examples/paraformer/prepare_assets.py" "$PARAFORMER_DIR" "$OUT_DIR"

echo "== 6/6 CAM++ =="
python "$HERE/export/export_campplus.py" \
  --checkpoint "$CAMPP_PLUS_DIR/campplus_cn_common.bin" \
  --out "$OUT_DIR/campplus_cpu.pte"

echo
echo "done. PTEs are in $OUT_DIR"
ls -la "$OUT_DIR" "$OUT_DIR/fsmn"
echo "next: build the runner, then run deploy/push_assets.sh"
