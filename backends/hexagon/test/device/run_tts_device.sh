#!/data/data/com.termux/files/usr/bin/bash
# TTS CodecDecoder on the device.  Modelled on the verified upsbilin/run.sh:
# preflight every file, print the exact command, log the runner and skel
# identity, and require a NON-EMPTY output before declaring success (skill 8.98:
# a clean exit is evidence the dispatcher worked and nothing about the values).
set -u
MYD=/data/data/com.termux/files/home/csm/execuTorch-ds/ttsdev
RUNNERD=/data/data/com.termux/files/home/csm/execuTorch-ds/devloop
SKELD=$RUNNERD/skel
TAG=tts

cd "$MYD" || exit 1
echo "== identity (skill 36.2: size + md5 of runner and skel, every run) =="
for f in "$RUNNERD/executor_runner" "$SKELD/libhex-htp-skel-v79.so"; do
  echo "IDENT $(basename $f) bytes=$(wc -c < "$f") md5=$(md5sum "$f" | cut -d' ' -f1)"
done

echo "== preflight =="
for f in "$MYD/$TAG.pte" "$MYD/${TAG}_in0.bin" "$MYD/${TAG}_in1.bin" "$RUNNERD/executor_runner" "$SKELD/libhex-htp-skel-v79.so"; do
  if [ ! -f "$f" ]; then
    echo "PREFLIGHT_MISSING=$f"
    exit 20
  fi
  echo "PREFLIGHT_OK=$f BYTES=$(wc -c < "$f")"
done

export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH="$SKELD"

run_one() {
  local tag="$1"
  local inbin="$2"
  local outb="$MYD/out_$tag"
  rm -f "$outb"-*.bin "$MYD/run_$tag.log"
  echo "== RUN $tag =="
  echo "CMD: $RUNNERD/executor_runner --model_path $MYD/$TAG.pte --inputs $inbin --output_file $outb --print_output none --num_executions 1"
  local s e rc b found o
  s=$(date +%s)
  timeout 900 "$RUNNERD/executor_runner" --model_path "$MYD/$TAG.pte" --inputs "$inbin" --output_file "$outb" --print_output none --num_executions 1 > "$MYD/run_$tag.log" 2>&1
  rc=$?
  e=$(date +%s)
  echo "RUNNER_EXIT=$rc ELAPSED_S=$((e-s))"
  echo "-- hexagon lines --"
  grep -aE 'enter|exit|ops=|error|fail|arena' "$MYD/run_$tag.log" | head -60
  found=0
  for o in "$outb"-*.bin; do
    if [ -f "$o" ]; then
      b=$(wc -c < "$o")
      echo "OUTPUT=$o BYTES=$b SHA256=$(sha256sum "$o" | cut -d' ' -f1)"
      if [ "$b" -gt 0 ]; then
        found=1
      else
        echo "OUTPUT_EMPTY=$o"
      fi
    fi
  done
  if [ "$found" -eq 0 ]; then
    echo "VERDICT=$tag FAIL_NO_NONEMPTY_OUTPUT"
    echo "--- tail of the runner log ---"
    tail -25 "$MYD/run_$tag.log"
    return 21
  fi
  echo "VERDICT=$tag OK"
  return 0
}

run_one tts "$MYD/${TAG}_in0.bin"
rc0=$?
run_one tts2 "$MYD/${TAG}_in1.bin"
rc2=$?

echo "== control: a known-good model, to prove the runner and skel are healthy =="
G=$RUNNERD
if [ -f "$G/cnnB.pte" ] && [ -f "$G/cnnB_in0.bin" ]; then
  rm -f "$MYD/ctl_cnnB"-*.bin
  timeout 300 "$RUNNERD/executor_runner" --model_path "$G/cnnB.pte" --inputs "$G/cnnB_in0.bin" --output_file "$MYD/ctl_cnnB" --print_output none --num_executions 1 > "$MYD/ctl_cnnB.log" 2>&1
  echo "CONTROL_CNNB_EXIT=$?"
  for o in "$MYD/ctl_cnnB"-*.bin; do
    if [ -f "$o" ]; then
      echo "CONTROL_CNNB_OUTPUT bytes=$(wc -c < "$o") md5=$(md5sum "$o" | cut -d' ' -f1)"
    fi
  done
else
  echo "CONTROL_CNNB_SKIPPED (no cnnB.pte in devloop)"
fi

echo "ALL_DONE rc_tts=$rc0 rc_tts2=$rc2"
