#!/system/bin/sh

set -eu

BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PORT=${PORT:-9000}
RUNNER=${RUNNER:-"$BASE_DIR/paraformer_runner_vad"}
LOG=${LOG:-"$BASE_DIR/paraformer_tcp_stream.log"}
VAD_MODEL=${VAD_MODEL:-"$BASE_DIR/fsmn/fsmn_vad_enc_T20.pte"}
VAD_CMVN=${VAD_CMVN:-"$BASE_DIR/fsmn/cmvn.bin"}
RUNNER_NAME=${RUNNER##*/}

if [ ! -x "$RUNNER" ]; then
  echo "runner is not executable: $RUNNER" >&2
  exit 1
fi

for path in cmvn.bin tokens.json encp_one.pte pred15.pte decoder16_first.pte decoder16_next.pte; do
  if [ ! -f "$BASE_DIR/$path" ]; then
    echo "missing asset: $BASE_DIR/$path" >&2
    exit 1
  fi
done

# FSMN-VAD endpointing is enabled when its PTE and CMVN are present.
VAD_FLAGS=""
if [ -f "$VAD_MODEL" ] && [ -f "$VAD_CMVN" ]; then
  VAD_FLAGS="--vad_model $VAD_MODEL --cmvn_vad $VAD_CMVN"
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
  --model_path "$BASE_DIR/encp_one.pte" \
  --predictor "$BASE_DIR/pred15.pte" \
  --decoder_first "$BASE_DIR/decoder16_first.pte" \
  --decoder_next "$BASE_DIR/decoder16_next.pte" \
  $VAD_FLAGS \
  >> "$LOG" 2>&1 </dev/null &

echo "started pid=$! port=$PORT log=$LOG vad=${VAD_FLAGS:-off}"
