#!/usr/bin/env bash
set -euo pipefail

example_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$example_dir/../../../.." && pwd)
sdk_dir=${HEXAGON_SDK_ROOT:-/home/yydh/hexagon-tutorial/tools/hexagon-sdk}
tools_dir=${HEXAGON_TOOLS_ROOT:-$sdk_dir/tools/HEXAGON_Tools/19.0.04/Tools}
build_dir=${HMX_EXAMPLE_BUILD_DIR:-$repo_dir/.tmp/scratch/mnv2/hmx_quant_params/example}
mkdir -p "$build_dir/lowbit"

"$tools_dir/bin/hexagon-clang" -mv79 -mhmx -O2 -shared -fPIC \
  -Wall -Wextra -Werror \
  -I "$sdk_dir/incs" -I "$sdk_dir/incs/stddef" \
  -I "$sdk_dir/rtos/qurt/computev79/include/qurt" \
  "$example_dir/conversion.c" "$example_dir/main.c" \
  "$example_dir/../hmx_16a8w_minimal/matmul.c" \
  "$example_dir/../hmx_16a16w_minimal/matmul.c" \
  -o "$build_dir/libhmx_quant_params.so"
"$tools_dir/bin/hexagon-clang" -mv79 -mhmx -O2 -shared -fPIC \
  -Wall -Wextra -Werror \
  -I "$sdk_dir/incs" -I "$sdk_dir/incs/stddef" \
  -I "$sdk_dir/rtos/qurt/computev79/include/qurt" \
  "$example_dir/conversion.c" "$example_dir/lowbit_main.c" \
  "$example_dir/../hmx_int8_minimal/matmul.c" \
  "$example_dir/../hmx_int4_minimal/matmul.c" \
  -o "$build_dir/lowbit/libhmx_lowbit_quant_params.so"
"$tools_dir/bin/hexagon-llvm-objdump" -d --mcpu=hexagonv79 --mattr=+hmx \
  "$build_dir/libhmx_quant_params.so" > "$build_dir/example.asm"
cc -O2 -shared -fPIC -Wall -Wextra -Werror "$example_dir/conversion.c" \
  -o "$build_dir/libconversion_host.so"
cp "$build_dir/libconversion_host.so" "$build_dir/lowbit/"
printf '%s\n' "$build_dir/libhmx_quant_params.so"
