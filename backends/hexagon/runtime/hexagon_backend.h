/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>

#include <memory>

#include <executorch/backends/hexagon/runtime/hexagon_driver.h>

namespace executorch::backends::hexagon {

// Registered under "HexagonBackend"; the name has to match what the partitioner
// stamps into the .pte.
class HexagonBackend final : public runtime::BackendInterface {
 public:
  bool is_available() const override;

  runtime::Result<runtime::DelegateHandle*> init(
      runtime::BackendInitContext& context,
      runtime::FreeableBuffer* processed,
      runtime::ArrayRef<runtime::CompileSpec> compile_specs) const override;

  runtime::Error execute(
      runtime::BackendExecutionContext& context,
      runtime::DelegateHandle* handle,
      runtime::Span<runtime::EValue*> args) const override;

  void destroy(runtime::DelegateHandle* handle) const override;
};

} // namespace executorch::backends::hexagon
