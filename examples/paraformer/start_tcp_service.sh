#!/system/bin/sh

set -eu

BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PORT=${PORT:-9000}
LOG=${LOG:-"$BASE_DIR/paraformer_tcp_stream.log"}
VAD_MODEL=${VAD_MODEL:-"$BASE_DIR/fsmn/fsmn_vad_enc_T20.pte"}
VAD_CMVN=${VAD_CMVN:-"$BASE_DIR/fsmn/cmvn.bin"}

# Prefer the VAD+CAM++ build, then the VAD build.
if [ -z "${RUNNER:-}" ]; then
  if [ -x "$BASE_DIR/paraformer_runner_spk" ]; then
    RUNNER="$BASE_DIR/paraformer_runner_spk"
  else
    RUNNER="$BASE_DIR/paraformer_runner_vad"
  fi
fi
# Prefer the encoder with the fp16 score clamp; it is the one that does not
# overflow on loud far-field meetings.
if [ -z "${MODEL:-}" ]; then
  if [ -f "$BASE_DIR/encp_one_min.pte" ]; then
    MODEL="$BASE_DIR/encp_one_min.pte"
  else
    MODEL="$BASE_DIR/encp_one.pte"
  fi
fi
RUNNER_NAME=${RUNNER##*/}

if [ ! -x "$RUNNER" ]; then
  echo "runner is not executable: $RUNNER" >&2
  exit 1
fi

for path in cmvn.bin tokens.json pred15.pte decoder16_first.pte decoder16_next.pte; do
  if [ ! -f "$BASE_DIR/$path" ]; then
    echo "missing asset: $BASE_DIR/$path" >&2
    exit 1
  fi
done
if [ ! -f "$MODEL" ]; then
  echo "missing encoder: $MODEL" >&2
  exit 1
fi

# FSMN-VAD endpointing is enabled when its PTE and CMVN are present.
VAD_FLAGS=""
if [ -f "$VAD_MODEL" ] && [ -f "$VAD_CMVN" ]; then
  VAD_FLAGS="--vad_model $VAD_MODEL --cmvn_vad $VAD_CMVN"
fi

# CAM++ speaker tagging is enabled when its CPU PTE is present.
SPK_MODEL=${SPK_MODEL:-"$BASE_DIR/camppluscpu/campplus_cpu.pte"}
SPK_FLAGS=""
if [ -f "$SPK_MODEL" ]; then
  SPK_FLAGS="--spk_model $SPK_MODEL"
fi

# Stop only this service instance. Other HTP users, such as llama-bench, are
# intentionally left untouched.
for pid in $(ps -A -o pid= -o args= | awk -v runner="$RUNNER_NAME" -v port="$PORT" \
    '$0 ~ runner && $0 ~ "--tcp_port " port {print $1}'); do
  kill "$pid" 2>/dev/null || true
done
sleep 1

cd "$BASE_DIR"
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH="$BASE_DIR/skel"

nohup "$RUNNER" \
  --tcp_port "$PORT" \
  --tcp_tmp_dir "$BASE_DIR" \
  --cmvn "$BASE_DIR/cmvn.bin" \
  --tokens "$BASE_DIR/tokens.json" \
  --model_path "$MODEL" \
  --predictor "$BASE_DIR/pred15.pte" \
  --decoder_first "$BASE_DIR/decoder16_first.pte" \
  --decoder_next "$BASE_DIR/decoder16_next.pte" \
  $VAD_FLAGS \
  $SPK_FLAGS \
  >> "$LOG" 2>&1 </dev/null &

echo "started pid=$! port=$PORT log=$LOG vad=${VAD_FLAGS:-off} spk=${SPK_FLAGS:-off}"
