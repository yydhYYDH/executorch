#!/usr/bin/env bash
# Push the runner, the PTEs, the frontend assets and the Hexagon skel to the
# phone.  Layout matches what examples/paraformer/start_tcp_service.sh expects.
#
#   RUNNER_BIN=cmake-out/paraformer_runner bash deploy/push_assets.sh
#
# Requires config.sh (PHONE, PHONE_DIR, OUT_DIR) and a skel built for the same
# Hexagon revision as the PTEs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
if [[ -f "$REPRO/config.sh" ]]; then source "$REPRO/config.sh"
else source "$REPRO/config.example.sh"; fi

RUNNER_BIN="${RUNNER_BIN:-$REPRO/../../../cmake-out/paraformer_runner}"
SKEL_DIR="${SKEL_DIR:-}"
[[ -f "$RUNNER_BIN" ]] || { echo "runner not found: $RUNNER_BIN (set RUNNER_BIN)" >&2; exit 1; }
[[ -d "$OUT_DIR" ]] || { echo "no exports in OUT_DIR=$OUT_DIR (run reproduce.sh)" >&2; exit 1; }

# ssh/scp here flaps; retry the transport, never a partial copy.
retry() { local n=$1; shift; for i in $(seq 1 "$n"); do "$@" && return 0; sleep 5; done; return 1; }

ssh "$PHONE" "mkdir -p '$PHONE_DIR/fsmn' '$PHONE_DIR/camppluscpu'"
retry 5 rsync -a --partial --inplace "$RUNNER_BIN" "$PHONE:$PHONE_DIR/paraformer_runner_spk"
retry 5 rsync -a --partial --inplace "$OUT_DIR/encp_one.pte" "$OUT_DIR/pred15.pte" \
  "$OUT_DIR/decoder16_first.pte" "$OUT_DIR/decoder16_next.pte" \
  "$OUT_DIR/cmvn.bin" "$OUT_DIR/tokens.json" "$PHONE:$PHONE_DIR/"
retry 5 rsync -a --partial --inplace "$OUT_DIR/fsmn/" "$PHONE:$PHONE_DIR/fsmn/"
retry 5 rsync -a --partial --inplace "$OUT_DIR/campplus_cpu.pte" "$PHONE:$PHONE_DIR/camppluscpu/"
if [[ -n "$SKEL_DIR" ]]; then
  retry 5 rsync -a --partial --inplace "$SKEL_DIR/" "$PHONE:$PHONE_DIR/skel/"
fi

# Verify the encoder landed intact (silent rsync failures have burned us).
local_sum=$(md5sum "$OUT_DIR/encp_one.pte" | cut -d' ' -f1)
remote_sum=$(ssh "$PHONE" "md5sum '$PHONE_DIR/encp_one.pte'" | cut -d' ' -f1)
[[ "$local_sum" == "$remote_sum" ]] || { echo "encp_one.pte md5 mismatch" >&2; exit 1; }
ssh "$PHONE" "chmod +x '$PHONE_DIR/paraformer_runner_spk'"
echo "pushed to $PHONE:$PHONE_DIR (encp_one.pte md5 $local_sum ok)"
