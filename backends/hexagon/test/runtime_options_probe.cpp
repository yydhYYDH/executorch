/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Reads the hexagon runtime options the way init() reads them, and prints what
// it resolved. Built and driven by test/test_runtime_options.py: the options are
// the part of this backend that runs on the host during init() rather than on
// the DSP, so they are also the part that can be exercised here without a
// device.
//
//   runtime_options_probe [key[=kind:]value ...]
//
// `kind` is one of b, i or s, so a test can hand over a value of the wrong type
// on purpose. Without it the key decides: the counts and indices are ints, the
// rest are flags. A key on its own is not valid -- a value has to be there.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <executorch/backends/hexagon/runtime/HexagonBackendOptions.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/memory_allocator.h>

namespace {

using executorch::backends::hexagon::HexagonBackendOptions;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::Error;
using executorch::runtime::kMaxOptionValueLength;
using executorch::runtime::Span;
using executorch::runtime::MemoryAllocator;

const char* ErrorName(Error error) {
  switch (error) {
    case Error::Ok:
      return "ok";
    case Error::InvalidArgument:
      return "invalid_argument";
    case Error::NotFound:
      return "not_found";
    default:
      return "other";
  }
}

// The options whose value is a count or an index rather than a flag.
bool IsIntKey(const std::string& key) {
  static const char* const kIntKeys[] = {
      "delegate",
      "cmd_start",
      "cmd_limit",
      "stop_after",
      "watchdog_seconds",
      "tile_budget",
  };
  for (const char* candidate : kIntKeys) {
    if (key == candidate) {
      return true;
    }
  }
  return false;
}

struct Parsed {
  std::string key;
  char kind;
  bool boolean = false;
  int integer = 0;
  std::string text;
};

} // namespace

int main(int argc, char** argv) {
  // Logging and timestamps go through the platform layer, which the runtime
  // initializes at startup and this probe has to initialize for itself.
  et_pal_init();

  std::vector<Parsed> parsed;
  for (int i = 1; i < argc; i++) {
    const std::string argument(argv[i]);
    const size_t at = argument.find('=');
    if (at == std::string::npos) {
      std::fprintf(stderr, "not a key=value option: %s\n", argv[i]);
      return 2;
    }
    Parsed entry;
    entry.key = argument.substr(0, at);
    std::string value = argument.substr(at + 1);
    if (value.size() > 1 && value[1] == ':') {
      entry.kind = value[0];
      value = value.substr(2);
    } else {
      entry.kind = IsIntKey(entry.key) ? 'i' : 'b';
    }
    if (entry.kind == 'b') {
      entry.boolean = std::atoi(value.c_str()) != 0;
    } else if (entry.kind == 'i') {
      entry.integer = std::atoi(value.c_str());
    } else {
      entry.text = value;
    }
    parsed.push_back(std::move(entry));
  }

  // The context only views these, so they are built after `parsed` is full and
  // no longer reallocating: a key or a string that moved would leave the
  // context reading freed memory.
  std::vector<BackendOption> options(parsed.size());
  for (size_t i = 0; i < parsed.size(); i++) {
    std::snprintf(options[i].key, sizeof(options[i].key), "%s", parsed[i].key.c_str());
    switch (parsed[i].kind) {
      case 'b':
        options[i].value = parsed[i].boolean;
        break;
      case 'i':
        options[i].value = parsed[i].integer;
        break;
      default: {
        std::array<char, kMaxOptionValueLength> text{};
        std::snprintf(text.data(), text.size(), "%s", parsed[i].text.c_str());
        options[i].value = text;
        break;
      }
    }
  }

  uint8_t arena[1024];
  MemoryAllocator allocator(sizeof(arena), arena);
  BackendInitContext context(
      &allocator,
      nullptr,
      "forward",
      nullptr,
      Span<const BackendOption>(options.data(), options.size()));

  auto resolved = HexagonBackendOptions().resolve(context);
  if (!resolved.ok()) {
    std::printf(
        "error %s %d\n",
        ErrorName(resolved.error()),
        static_cast<int>(resolved.error()));
    return 0;
  }
  const auto& value = resolved.get();
  std::printf("ok\n");
  std::printf("trace %d\n", value.trace ? 1 : 0);
  std::printf("delegate %d\n", value.delegate);
  std::printf("cmd_start %d\n", value.cmd_start);
  std::printf("cmd_limit %d\n", value.cmd_limit);
  std::printf("stop_after %d\n", value.stop_after);
  std::printf("fake_cache %d\n", value.fake_cache ? 1 : 0);
  std::printf("phase %d\n", value.phase ? 1 : 0);
  std::printf("kv_state %d\n", value.kv_state ? 1 : 0);
  std::printf("watchdog_seconds %d\n", value.watchdog_seconds);
  std::printf("tile_budget %d\n", value.tile_budget);
  std::printf("acct %d\n", value.acct ? 1 : 0);
  return 0;
}
