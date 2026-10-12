#!/usr/bin/env bash
set -euo pipefail
example_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$example_dir/../../../.." && pwd)
sdk_dir=${HEXAGON_SDK_ROOT:-/home/yydh/hexagon-tutorial/tools/hexagon-sdk}
tools_dir=${HEXAGON_TOOLS_ROOT:-$sdk_dir/tools/HEXAGON_Tools/19.0.04/Tools}
build_dir=${CONVERSION_BUILD_DIR:-$repo_dir/.tmp/scratch/mnv2/integer_float_conversion}
host_cc=${CONVERSION_HOST_CC:-/home/yydh/android-ndk-r28c/toolchains/llvm/prebuilt/linux-x86_64/bin/clang}
mkdir -p "$build_dir"
"$host_cc" -std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC \
  "$example_dir/conversion.c" -o "$build_dir/libconversion_host.so"
"$host_cc" -std=c11 -O2 -Wall -Wextra -Werror \
  "$example_dir/conversion.c" "$example_dir/main.c" -o "$build_dir/conversion_host"
"$tools_dir/bin/hexagon-clang" -mv79 -O2 -Wall -Wextra -Werror -shared -fPIC \
  -I "$sdk_dir/incs" -I "$sdk_dir/incs/stddef" \
  "$example_dir/conversion.c" "$example_dir/main.c" \
  -o "$build_dir/libinteger_float_conversion.so"
"$tools_dir/bin/hexagon-llvm-objdump" -d --mcpu=hexagonv79 --mattr=+hmx \
  "$build_dir/libinteger_float_conversion.so" > "$build_dir/conversion.asm"
