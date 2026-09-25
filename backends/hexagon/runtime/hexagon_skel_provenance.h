// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

#include "hexagon_skel_source_id.h"

// The only question the loader asks about a skel it just opened: does the
// source closure the skel was built from still match the one this runner was
// built from? It is pure so a host-only test can drive the same comparison the
// device path uses.
inline bool skel_source_id_matches(const uint32_t (&reported)[4]) {
  const uint32_t expected[4] = {
      HEXAGON_SKEL_SOURCE_ID_0,
      HEXAGON_SKEL_SOURCE_ID_1,
      HEXAGON_SKEL_SOURCE_ID_2,
      HEXAGON_SKEL_SOURCE_ID_3,
  };
  for (size_t i = 0; i < 4; ++i) {
    if (reported[i] != expected[i]) {
      return false;
    }
  }
  return true;
}
