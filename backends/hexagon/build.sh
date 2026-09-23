#!/usr/bin/env bash
#
# One-shot build for the Hexagon backend. Produces the two artifacts that have
# to exist before anything can run on a device:
#
#   skel    cross-compiled by the Hexagon toolchain, loaded by the DSP
#   runner  Android arm64, loaded by the phone's CPU
#
# Usage:
#   ./build.sh              build both
#   ./build.sh skel         build the skel only
#   ./build.sh runner       build the runner only
#   ./build.sh deploy       build both, then push to the device
#
# Everything below is overridable from the environment.

set -euo pipefail

# --- configuration -----------------------------------------------------------

ET_ROOT=${ET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-}
HEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT:-${HEXAGON_SDK_ROOT}/tools/HEXAGON_Tools/19.0.04}
ANDROID_NDK=${ANDROID_NDK:-}

SKEL_ARCH=${SKEL_ARCH:-v79}
# The generated Command_generated.h needs a flatbuffers new enough for the
# three-argument VerifyField. The default the CMakeLists computes can land on an
# older copy elsewhere on the machine and fail to compile, so pin it.
FLATBUFFERS_ROOT=${FLATBUFFERS_ROOT:-${ET_ROOT}/third-party/flatbuffers}

SKEL_BUILD_DIR=${SKEL_BUILD_DIR:-/tmp/skelbuild}
HOST_BUILD_DIR=${HOST_BUILD_DIR:-${ET_ROOT}/cmake-out-android-arm64-v8a}
# The main build's hexagon_idl target writes the generated FastRPC sources here.
IDL_DIR=${IDL_DIR:-${HOST_BUILD_DIR}/backends/hexagon/idl}

CMAKE=${CMAKE:-cmake}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}

# Optional, for ./build.sh deploy
DEVICE=${DEVICE:-}
DEVICE_DIR=${DEVICE_DIR:-/data/local/tmp/hexagon}
MODEL=${MODEL:-}

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- skel --------------------------------------------------------------------

build_skel() {
  say "building the skel for ${SKEL_ARCH}"
  [ -d "${HEXAGON_SDK_ROOT}" ]   || die "HEXAGON_SDK_ROOT not found: ${HEXAGON_SDK_ROOT}"
  [ -d "${HEXAGON_TOOLS_ROOT}" ] || die "HEXAGON_TOOLS_ROOT not found: ${HEXAGON_TOOLS_ROOT}"
  [ -d "${IDL_DIR}" ]            || die "IDL_DIR not found: ${IDL_DIR}"

  mkdir -p "${SKEL_BUILD_DIR}"
  cd "${SKEL_BUILD_DIR}"
  # The toolchain file is what selects hexagon-clang++; without it CMake picks
  # the host compiler, which then rejects -mhvx and -mhmx.
  "${CMAKE}" "${ET_ROOT}/backends/hexagon/skel" \
    -DCMAKE_TOOLCHAIN_FILE="${HEXAGON_SDK_ROOT}/build/cmake/hexagon_toolchain.cmake" \
    -DHEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}" \
    -DHEXAGON_TOOLS_ROOT="${HEXAGON_TOOLS_ROOT}" \
    -DMNN_OPS_ROOT="${ET_ROOT}/backends/hexagon/third-party/mnn-htp-ops" \
    -DHEXAGON_FLATBUFFERS_ROOT="${FLATBUFFERS_ROOT}" \
    -DSKEL_ARCH="${SKEL_ARCH}" \
    -DIDL_DIR="${IDL_DIR}"
  "${CMAKE}" --build . -j"${JOBS}"

  verify_skel_opt
}

# CMake is configured without CMAKE_BUILD_TYPE here, so nothing adds optimization
# on its own and the explicit -O2 in skel/CMakeLists.txt is the only thing that
# does. Without it the DSP aborts with 0x8000040d before reading a byte, because
# an unoptimized fp16 branch chain asks for a 14848-byte stack frame that the RPC
# thread cannot provide.
verify_skel_opt() {
  local flags="${SKEL_BUILD_DIR}/CMakeFiles/hex-htp-${SKEL_ARCH}.dir/flags.make"
  [ -f "${flags}" ] || die "cannot find ${flags} to verify the build flags"

  if ! grep -q -- '-O[123s]' "${flags}"; then
    printf '\n' >&2
    grep -m1 CXX_FLAGS "${flags}" >&2 || true
    die "the skel was built without optimization; the DSP will abort with 0x8000040d.
     Check that skel/CMakeLists.txt still contains add_compile_options(-O2)."
  fi
  printf 'optimization flag present: %s\n' "$(grep -m1 CXX_FLAGS "${flags}" | tr -s ' ')"
}

# --- runner ------------------------------------------------------------------

build_runner() {
  say "building the android runner"
  [ -f "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" ] \
    || die "ANDROID_NDK not found: ${ANDROID_NDK}"

  cd "${ET_ROOT}"
  # Reconfiguring is cheap and keeps an existing build dir in sync with a changed
  # CMakeLists; EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL is required alongside the
  # runner or the configure step fails.
  "${CMAKE}" -B "${HOST_BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-26 \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_HEXAGON=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON
  "${CMAKE}" --build "${HOST_BUILD_DIR}" --target executor_runner -j"${JOBS}"

  ls -l "${HOST_BUILD_DIR}/executor_runner"
}

# --- deploy ------------------------------------------------------------------

deploy() {
  [ -n "${DEVICE}" ] || die "DEVICE is not set"
  say "deploying to ${DEVICE}:${DEVICE_DIR}"
  local skel_so="${SKEL_BUILD_DIR}/ship/libhex-htp-skel-${SKEL_ARCH}.so"
  [ -f "${skel_so}" ] || die "skel not built: ${skel_so}"

  ssh "${DEVICE}" "mkdir -p '${DEVICE_DIR}/skel'"
  scp "${skel_so}"                              "${DEVICE}:${DEVICE_DIR}/skel/"
  scp "${HOST_BUILD_DIR}/executor_runner"       "${DEVICE}:${DEVICE_DIR}/"
  [ -n "${MODEL}" ] && scp "${MODEL}"          "${DEVICE}:${DEVICE_DIR}/"

  # The runner is ~160 MB and a stale copy on the device is the usual reason a
  # rebuild appears to change nothing, so confirm what actually landed.
  say "on the device"
  ssh "${DEVICE}" "cd '${DEVICE_DIR}' && ls -l skel/libhex-htp-skel-${SKEL_ARCH}.so executor_runner"

  cat <<EOF

Run it on the phone:

  ssh ${DEVICE}
  cd ${DEVICE_DIR}
  export LD_LIBRARY_PATH=\$PWD/libs:/system/lib64:/vendor/lib64
  export ADSP_LIBRARY_PATH=\$PWD/skel
  ./executor_runner --model_path <model>.pte --output_file devout --print_output none
EOF
}

# --- main --------------------------------------------------------------------

case "${1:-all}" in
  skel)   build_skel ;;
  runner) build_runner ;;
  all)    build_skel; build_runner ;;
  deploy) build_skel; build_runner; deploy ;;
  *)      die "unknown target '\$1'; expected skel, runner, all or deploy" ;;
esac

say "done"
