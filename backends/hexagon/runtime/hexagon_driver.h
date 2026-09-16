/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::hexagon {

using runtime::Error;
using runtime::Result;

// Domain ids, see the Hexagon SDK's remote.h.
constexpr int kAdspDomainId = 0;
constexpr int kCdspDomainId = 3;

// One FastRPC session to the DSP plus the shared memory that command buffers,
// weights and activations live in.
//
// The DSP only ever sees buffer file descriptors, so every buffer that takes
// part in an inference comes from Alloc() and crosses as a fd.
class HexagonDriver final {
 public:
  static Result<HexagonDriver> Create(int domain_id = kCdspDomainId);

  ~HexagonDriver();

  HexagonDriver(const HexagonDriver&) = delete;
  HexagonDriver& operator=(const HexagonDriver&) = delete;
  HexagonDriver(HexagonDriver&&) noexcept;
  HexagonDriver& operator=(HexagonDriver&&) noexcept;

  // Shared memory, host-writable, mapped for the DSP, with a stable fd.
  Result<void*> Alloc(size_t bytes, size_t alignment = 4096);
  void Free(void* ptr);
  Result<int> ToFd(void* ptr);

  // Byte offset of ptr within its FastRPC mapping. The DSP resolves an fd to
  // the base of that mapping, so every offset handed to it is measured from
  // here rather than from ptr.
  Result<uint64_t> MappingOffset(void* ptr);

  // Reconciles host and DSP views of a buffer. Both are no-ops when the arena
  // is uncached (see Alloc).
  Error Flush(void* ptr, size_t bytes);
  Error Invalidate(void* ptr, size_t bytes);

  // Runs one batch: count commands are read from the group buffer at
  // (group_fd, group_offset), and the sync group names the tensors to
  // invalidate beforehand and flush afterwards.
  Error ExecuteCommandGroup(
      int group_fd,
      int group_offset,
      uint32_t count,
      int sync_fd,
      int sync_offset,
      int sync_size);

  Error PowerAcquire();
  Error PowerRelease();

  uint32_t skel_arch() const {
    return skel_arch_;
  }
  int domain_id() const {
    return domain_id_;
  }
  // False when the device's libcdsprpc.so has no cache maintenance entry
  // points, in which case the arena is allocated uncached.
  bool cached() const {
    return cached_;
  }

 private:
  HexagonDriver() = default;

  Error Open();
  void Close();
  // rpcmem_cache_flush/invalidate are on the device but absent from the SDK's
  // link-time stub, so they are resolved dynamically.
  void ResolveCacheOps();

  struct Allocation {
    void* base; // what rpcmem_alloc returned
    size_t size;
    int fd;
  };

  using CacheOpFn = int (*)(void*, int);

  int domain_id_ = kCdspDomainId;
  uint64_t handle_ = 0; // remote_handle64, 0 when closed
  uint32_t skel_arch_ = 0;
  bool cached_ = false;
  CacheOpFn cache_flush_ = nullptr;
  CacheOpFn cache_invalidate_ = nullptr;
  std::unordered_map<void*, Allocation> allocations_;
};

} // namespace executorch::backends::hexagon
