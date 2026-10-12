#!/usr/bin/env bash
# Frontend: serve the browser UI and relay audio to the runner.
#
#   bash run/frontend.sh                 # local: runner at 127.0.0.1:9000
#   RUNNER_HOST=127.0.0.1 bash run/frontend.sh
#
# Open http://127.0.0.1:$BRIDGE_PORT/ afterwards.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$REPRO/../../.." && pwd)"
if [[ -f "$REPRO/config.sh" ]]; then source "$REPRO/config.sh"
else source "$REPRO/config.example.sh"; fi

RUNNER_HOST="${RUNNER_HOST:-127.0.0.1}"
echo "bridge: listen 0.0.0.0:$BRIDGE_PORT -> runner $RUNNER_HOST:$RUNNER_PORT"
exec python3 "$ROOT/examples/paraformer/web/server.py" \
  --runner-host "$RUNNER_HOST" --runner-port "$RUNNER_PORT" \
  --listen-host 0.0.0.0 --listen-port "$BRIDGE_PORT"
