/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/hexagon/runtime/hexagon_weight_sync.h>
#include <cassert>

using namespace executorch::backends::hexagon;

int main() {
  const HexagonTensorRef weight{0, 0, 0, 128};
  const HexagonTensorRef input{1, 0, 0, 128};
  const HexagonTensorRef activation{3, 0, 0, 128};
  const HexagonTensorRef absent{0xFFFFFFFF, 0, 0, 0};
  HexagonOp ops[2]{};
  ops[0].n_inputs = 2;
  ops[0].inputs[0] = input;
  ops[0].inputs[1] = weight;
  ops[1].n_inputs = 1;
  ops[1].inputs[0] = weight;
  HexagonWeightSyncPolicy readonly(ops, 2);
  assert(readonly.should_sync(weight, true));
  assert(!readonly.should_sync(weight, false));
  assert(readonly.should_sync(input, false));
  assert(readonly.should_sync(activation, false));
  assert(!readonly.should_sync(absent, true));
  assert(!readonly.should_sync(absent, false));

  // A write through another weight reference may alias this input.
  ops[1].n_outputs = 1;
  ops[1].outputs[0] = HexagonTensorRef{0, 9, 64, 32};
  HexagonWeightSyncPolicy written(ops, 2);
  assert(written.should_sync(weight, false));
  ops[1].n_outputs = 0;
  ops[0].in_place = 1u << 1;
  HexagonWeightSyncPolicy in_place(ops, 2);
  assert(in_place.should_sync(weight, false));
  ops[0].in_place = 1u;
  HexagonWeightSyncPolicy input_in_place(ops, 2);
  assert(!input_in_place.should_sync(weight, false));
  assert(input_in_place.should_sync(input, false));
}
