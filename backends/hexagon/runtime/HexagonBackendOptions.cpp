/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/hexagon/runtime/HexagonBackendOptions.h>

#include <cstdlib>

#include <executorch/runtime/platform/log.h>

namespace executorch::backends::hexagon {

namespace {

using executorch::runtime::BackendInitContext;
using executorch::runtime::Error;
using executorch::runtime::Result;

// Keys, as a runner names them in LoadBackendOptionsMap. The environment
// variable each one has always had is named beside it.
// Must stay C arrays (not const char*) so a wrong type is a compile error.
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kTraceKey[] = "trace"; // HEXAGON_TRACE
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kDelegateKey[] = "delegate"; // HEXAGON_DELEGATE
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kCmdStartKey[] = "cmd_start"; // HEXAGON_CMD_START
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kCmdLimitKey[] = "cmd_limit"; // HEXAGON_CMD_LIMIT
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kStopAfterKey[] = "stop_after"; // HEXAGON_STOP_AFTER
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kFakeCacheKey[] = "fake_cache"; // HEXAGON_FAKE_CACHE
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kPhaseKey[] = "phase"; // HEXAGON_PHASE
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kKvStateKey[] = "kv_state"; // HEXAGON_KV_STATE
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kWatchdogSecondsKey[] = "watchdog_seconds"; // HEXAGON_WATCHDOG_SECONDS
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kTileBudgetKey[] = "tile_budget"; // HEXAGON_HMX_TILE_BUDGET
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kAcctKey[] = "acct"; // HEXAGON_ACCT

// The environment form of these knobs, read only when no spec names them.
// atoi semantics are the ones these reads always had: an unset variable and one
// holding something that is not a number both fall back.
int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::atoi(value);
}

bool EnvFlag(const char* name, bool fallback) {
  return EnvInt(name, fallback ? 1 : 0) != 0;
}

// One option: the load-time spec if it has one, else what the environment said.
// A spec that is present but cannot be used is an error, not a fallback: a
// caller that asked for a trace and silently did not get one cannot tell that
// apart from a trace with nothing in it.
template <typename T>
Error ResolveOption(
    const BackendInitContext& context,
    const char* key,
    const char* expected,
    T fallback,
    T* out) {
  auto spec = context.get_runtime_spec<T>(key);
  if (spec.ok()) {
    *out = spec.get();
    return Error::Ok;
  }
  if (spec.error() == Error::NotFound) {
    *out = fallback;
    return Error::Ok;
  }
  ET_LOG(Error, "hexagon: backend option %s must be %s.", key, expected);
  return spec.error();
}

} // namespace

Result<HexagonRuntimeOptions> HexagonBackendOptions::resolve(
    const BackendInitContext& context) const {
  HexagonRuntimeOptions options;
  Error err = Error::Ok;

  if ((err = ResolveOption(
           context,
           kTraceKey,
           "a bool",
           EnvFlag("HEXAGON_TRACE", false),
           &options.trace)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kDelegateKey,
           "an int",
           EnvInt("HEXAGON_DELEGATE", -1),
           &options.delegate)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kCmdStartKey,
           "an int",
           EnvInt("HEXAGON_CMD_START", 0),
           &options.cmd_start)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kCmdLimitKey,
           "an int",
           EnvInt("HEXAGON_CMD_LIMIT", 0),
           &options.cmd_limit)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kStopAfterKey,
           "an int",
           EnvInt("HEXAGON_STOP_AFTER", -1),
           &options.stop_after)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kFakeCacheKey,
           "a bool",
           EnvFlag("HEXAGON_FAKE_CACHE", false),
           &options.fake_cache)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kPhaseKey,
           "a bool",
           EnvFlag("HEXAGON_PHASE", false),
           &options.phase)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kKvStateKey,
           "a bool",
           EnvFlag("HEXAGON_KV_STATE", true),
           &options.kv_state)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kWatchdogSecondsKey,
           "an int",
           EnvInt("HEXAGON_WATCHDOG_SECONDS", 15),
           &options.watchdog_seconds)) != Error::Ok) {
    return err;
  }
  // -1 is "no override": the exporter's own answer stays in the command.
  if ((err = ResolveOption(
           context,
           kTileBudgetKey,
           "an int",
           EnvInt("HEXAGON_HMX_TILE_BUDGET", -1),
           &options.tile_budget)) != Error::Ok) {
    return err;
  }
  if ((err = ResolveOption(
           context,
           kAcctKey,
           "a bool",
           EnvFlag("HEXAGON_ACCT", false),
           &options.acct)) != Error::Ok) {
    return err;
  }

  // Ranges the values have to be in to mean anything. Reported by name so a
  // mistyped option is not read as a trace that happens to be empty.
  if (options.delegate < -1) {
    ET_LOG(
        Error,
        "hexagon: delegate must be -1 (every one) or more, got %d.",
        options.delegate);
    return Error::InvalidArgument;
  }
  if (options.cmd_start < 0 || options.cmd_limit < 0) {
    ET_LOG(
        Error,
        "hexagon: cmd_start and cmd_limit are command counts and cannot be "
        "negative, got %d and %d.",
        options.cmd_start,
        options.cmd_limit);
    return Error::InvalidArgument;
  }
  if (options.stop_after < -1) {
    ET_LOG(
        Error,
        "hexagon: stop_after must be -1 (never) or a delegate index, got %d.",
        options.stop_after);
    return Error::InvalidArgument;
  }
  if (options.watchdog_seconds < 0) {
    ET_LOG(
        Error,
        "hexagon: watchdog_seconds must be 0 (disabled) or more, got %d.",
        options.watchdog_seconds);
    return Error::InvalidArgument;
  }
  if (options.tile_budget < -1) {
    ET_LOG(
        Error,
        "hexagon: tile_budget must be -1 (keep what the blob asked for) or "
        "more, got %d.",
        options.tile_budget);
    return Error::InvalidArgument;
  }
  return options;
}

} // namespace executorch::backends::hexagon
