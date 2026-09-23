/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::hexagon {

// The knobs that change how a delegate runs rather than what its blob holds.
//
// Every one of these is a diagnostic or a cap: none of them changes a byte of
// the blob, so a runner is free to set them per device and per load. The
// choices that do change the blob -- whether weights are pre-tiled for the HMX
// unit, which attention entry point to use, which weights the .pte carries --
// are compile specs instead, because flipping one of those at load time would
// run a program against a layout it was never compiled for. See
// hexagon_compat.h for the check that keeps the two in step.
struct HexagonRuntimeOptions {
  // Per-command trace to stderr, and which delegate to apply it to.
  bool trace = false;
  int delegate = -1;
  // Bracket the command group: start at this command, run this many of them.
  int cmd_start = 0;
  int cmd_limit = 0;
  // Stop the process once this delegate has run, for a bisect by delegate.
  int stop_after = -1;
  // Fill an attention command's empty cache operands from its key and value.
  bool fake_cache = false;
  // Timeline of the init and execute phases.
  bool phase = false;
  // Keep the packed attention cache in the persistent state block.
  bool kv_state = true;
  // Print the DSP trace on a timer, in seconds. Zero disables it.
  int watchdog_seconds = 15;
  // Cap on the 32-column weight tiles the HMX unit keeps resident. Negative
  // leaves whatever the blob's command asked for, which is what the exporter
  // wrote and what a blob with no opinion carries.
  int tile_budget = -1;
  // Host-side accounting of what a round trip costs outside the DSP.
  bool acct = false;
};

class HexagonBackendOptions {
 public:
  // Reads the options for one delegate: a value in the load-time spec wins,
  // then the environment variable this knob has always had, then the default
  // above. A process that sets none of them behaves exactly as it did before
  // options existed.
  //
  // A spec whose value cannot be used is an error rather than a fallback: a
  // caller that asked for a trace and silently did not get one has no way to
  // tell that apart from a trace with nothing in it. Returns
  // Error::InvalidArgument, having logged which key was wrong.
  runtime::Result<HexagonRuntimeOptions> resolve(
      const ET_RUNTIME_NAMESPACE::BackendInitContext& context) const;
};

} // namespace executorch::backends::hexagon
