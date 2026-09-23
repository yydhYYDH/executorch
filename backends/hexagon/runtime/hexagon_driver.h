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
#include <memory>
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

// A DSP-visible buffer, plus the two numbers the DSP needs to address into it:
// the fd it resolves through HAP_mmap_get and the offset of ptr inside that
// mapping.
struct Arena {
  void* ptr = nullptr;
  size_t bytes = 0;
  int fd = -1;
  uint64_t bias = 0;
};

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
  Result<Arena> Alloc(size_t bytes, size_t alignment = 4096);
  void Free(void* ptr);

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

  // ExecuteCommandGroup, plus the DSP-side per-command trace written into the
  // buffer at (profile_fd, profile_offset). The trace is readable even when the
  // call fails, which is the point of it: a DSP fault leaves nothing else behind.
  Error ExecuteCommandGroupTraced(
      int group_fd,
      int group_offset,
      uint32_t count,
      int sync_fd,
      int sync_offset,
      int sync_size,
      int profile_fd,
      int profile_offset,
      int profile_size);

  Error PowerAcquire();
  Error PowerRelease();

  // Closes the FastRPC session and frees every buffer Alloc() handed out.
  // HexagonBackend::destroy() is the caller that matters: the delegate's driver
  // comes from the runtime allocator, which never constructs it, so the
  // destructor that would otherwise close the session never runs. Idempotent,
  // and safe after Free().
  void Close();

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

// Buffers keyed by size and handed to every delegate that asks for that size.
//
// A delegate's scratch is dead between execute() calls: the method inputs are
// copied in, the kernels fill the activations and the method outputs, and the
// outputs are copied back out, all inside one execute. Nothing written there
// has to survive, and a graph runs one delegate at a time, so delegates whose
// scratch is the same size can overlap on one buffer. What a delegate keeps for
// itself -- descriptors, sync group, command group, weights -- is tens of
// kilobytes, which is what makes the DSP's shared-memory ceiling survivable for
// a model with hundreds of delegates.
class SharedArenaPool final {
 public:
  static SharedArenaPool& Get();

  // The buffer for a size; everyone asking for that size shares it.
  Result<Arena> Acquire(size_t bytes);

 private:
  SharedArenaPool() = default;

  // The pool has its own session: a pooled buffer outlives the delegate that
  // first asked for it, so it cannot belong to that delegate's driver.
  std::unique_ptr<HexagonDriver> driver_;
  std::unordered_map<size_t, Arena> arenas_;
};

} // namespace executorch::backends::hexagon
