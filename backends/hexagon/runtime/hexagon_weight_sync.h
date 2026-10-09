/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/hexagon/serialization/hexagon_schema.h>

namespace executorch::backends::hexagon {

class HexagonWeightSyncPolicy {
 public:
  HexagonWeightSyncPolicy(const HexagonOp* ops, size_t count) {
    const auto weights = static_cast<uint32_t>(HexagonTensorSpace::kWeights);
    for (size_t i = 0; i < count; ++i) {
      for (uint32_t j = 0; j < ops[i].n_outputs; ++j) {
        writable_ |= ops[i].outputs[j].space == weights;
      }
      for (uint32_t j = 0; j < ops[i].n_inputs; ++j) {
        writable_ |= ops[i].inputs[j].space == weights &&
            (ops[i].in_place & (1u << j)) != 0;
      }
    }
  }

  bool should_sync(const HexagonTensorRef& ref, bool first_execution) const {
    const auto space = static_cast<HexagonTensorSpace>(ref.space);
    return space != HexagonTensorSpace::kAbsent &&
        (space != HexagonTensorSpace::kWeights || first_execution || writable_);
  }

 private:
  bool writable_ = false;
};

} // namespace executorch::backends::hexagon
