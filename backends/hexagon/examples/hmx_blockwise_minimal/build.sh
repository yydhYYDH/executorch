#!/usr/bin/env bash
set -euo pipefail

example_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$example_dir/../../../.." && pwd)
sdk_dir=${HEXAGON_SDK_ROOT:-/home/yydh/hexagon-tutorial/tools/hexagon-sdk}
tools_dir=${HEXAGON_TOOLS_ROOT:-$sdk_dir/tools/HEXAGON_Tools/19.0.04/Tools}
build_dir=${HMX_EXAMPLE_BUILD_DIR:-$repo_dir/.tmp/scratch/mnv2/hmx_blockwise_minimal/example}
mkdir -p "$build_dir"

"$tools_dir/bin/hexagon-clang" -mv79 -mhmx -O2 -shared -fPIC \
  -Wall -Wextra -Werror \
  -I "$sdk_dir/incs" -I "$sdk_dir/incs/stddef" \
  -I "$sdk_dir/rtos/qurt/computev79/include/qurt" \
  "$example_dir/matmul.c" "$example_dir/main.c" \
  -o "$build_dir/libhmx_blockwise_example.so"
"$tools_dir/bin/hexagon-llvm-objdump" -d --mcpu=hexagonv79 --mattr=+hmx \
  "$build_dir/libhmx_blockwise_example.so" > "$build_dir/example.asm"
printf '%s\n' "$build_dir/libhmx_blockwise_example.so"
