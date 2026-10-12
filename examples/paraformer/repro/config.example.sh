# Copy to config.sh and edit the paths.  Sourced by reproduce.sh and deploy/run
# scripts; every value can also be overridden on the command line.

# --- host paths ---
# download_models.py symlinks the ModelScope snapshots here.
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PARAFORMER_DIR="${PARAFORMER_DIR:-$MODELS_DIR/paraformer}"
FSMN_VAD_DIR="${FSMN_VAD_DIR:-$MODELS_DIR/fsmn-vad}"
CAMPP_PLUS_DIR="${CAMPP_PLUS_DIR:-$MODELS_DIR/campplus}"

# Where PTEs and frontend assets are written.
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/out}"

# --- device (phone) ---
PHONE="${PHONE:-oneplus13-reverse}"
PHONE_DIR="${PHONE_DIR:-/data/data/com.termux/files/home/csm/execuTorch-ds}"

# --- server (optional: public frontend) ---
SERVER="${SERVER:-chengdu}"
RUNNER_PORT="${RUNNER_PORT:-9000}"
BRIDGE_PORT="${BRIDGE_PORT:-8020}"
