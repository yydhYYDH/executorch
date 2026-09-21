/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/runner_util/inputs.h>

#include <vector>

#include <ATen/ATen.h> // @manual=//caffe2/aten:ATen-core
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>

using executorch::runtime::Error;
using executorch::runtime::aten::Method;
using executorch::runtime::aten::TensorInfo;

namespace executorch {
namespace extension {
namespace internal {

Error fill_and_set_input(
    Method& method,
    TensorInfo& tensor_meta,
    size_t input_index,
    void* data_ptr,
    bool fill_tensor,
    executorch::runtime::Span<const int32_t> dynamic_sizes) {
  // Convert the sizes array from int32_t to int64_t.
  std::vector<int64_t> sizes;
  auto shape = dynamic_sizes.empty() ? tensor_meta.sizes() : dynamic_sizes;
  for (auto s : shape) {
    sizes.push_back(s);
  }
  at::Tensor t = at::from_blob(
      data_ptr, sizes, at::TensorOptions(tensor_meta.scalar_type()));

  if (fill_tensor) {
    t.fill_(1.0f);
  }

  return method.set_input(t, input_index);
}

} // namespace internal
} // namespace extension
} // namespace executorch
