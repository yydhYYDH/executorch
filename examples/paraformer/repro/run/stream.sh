#!/usr/bin/env bash
# Streaming: start the on-device TCP service, then bridge it.
#
#   bash run/stream.sh              # start runner + (if SERVER set) reverse tunnel
#   bash run/stream.sh --local      # phone is directly reachable; no tunnel
#
# After this, either run run/frontend.sh or point examples/paraformer/web/server.py
# at $RUNNER_HOST:$RUNNER_PORT yourself.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/.." && pwd)"
if [[ -f "$REPRO/config.sh" ]]; then source "$REPRO/config.sh"
else source "$REPRO/config.example.sh"; fi

LOCAL=0
[[ "${1:-}" == "--local" ]] && LOCAL=1

echo "starting runner on $PHONE:$PHONE_DIR (port $RUNNER_PORT)"
ssh "$PHONE" "cd '$PHONE_DIR' && PORT=$RUNNER_PORT ./start_tcp_service.sh"

if [[ "$LOCAL" == 0 ]]; then
  echo "keeping 127.0.0.1:$RUNNER_PORT reachable on $SERVER via reverse tunnel"
  echo "run this in a persistent shell (or a systemd unit, see DEPLOY.md):"
  echo "  autossh -M 0 -NT -o ServerAliveInterval=30 -R 127.0.0.1:$RUNNER_PORT:127.0.0.1:$RUNNER_PORT $SERVER"
else
  echo "no tunnel: point the bridge at $PHONE:$RUNNER_PORT"
fi
