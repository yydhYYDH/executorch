#!/data/data/com.termux/files/usr/bin/bash
# The phone-side runner loop for the W8A16 prefill cases.
#
# The runner and the skeleton are the pair already deployed on this phone, reached
# by absolute path, so nothing in the shared directory is overwritten and the
# binary pair is named in the report rather than silently inherited. Every case
# names its own .pte, writes its own --output_file, and prints the exit code and
# the byte count. The byte count is printed because a run that wrote nothing is a
# fact worth having; it is not a result. The numbers are compared on the host
# afterwards, against references the exporting process produced.
set -u
D=${1:?device dir}
R=${2:-/data/data/com.termux/files/home/csm/execuTorch-ds/executor_runner}
S=${3:-/data/data/com.termux/files/home/csm/execuTorch-ds/skel}
shift 3
cd "$D" || exit 1
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH="$S"
for tag in "$@"; do
  rm -f "$D/out_$tag"-*.bin "$D/run_$tag.log"
  timeout 300 "$R" --model_path "$D/$tag.pte" --inputs "$D/${tag}_in0.bin" \
      --output_file "$D/out_$tag" --print_output none --num_executions 1 \
      > "$D/run_$tag.log" 2>&1
  rc=$?
  echo "CASE=$tag RUNNER_EXIT=$rc"
  found=0
  for output in "$D/out_$tag"-*.bin; do
    if [ -f "$output" ]; then
      found=1
      echo "CASE=$tag OUTPUT=$output OUTPUT_BYTES=$(wc -c < "$output") SHA256=$(sha256sum "$output" | cut -c1-32)"
    fi
  done
  if [ "$found" -ne 1 ]; then
    echo "CASE=$tag OUTPUT_MISSING=1"
  fi
  grep -oE "enter d0: ops=[0-9]+" "$D/run_$tag.log" | head -1 | sed "s/^/CASE=$tag /"
  grep -oE "exit d0: (ok|failed)" "$D/run_$tag.log" | head -1 | sed "s/^/CASE=$tag /"
  tail -3 "$D/run_$tag.log"
done
