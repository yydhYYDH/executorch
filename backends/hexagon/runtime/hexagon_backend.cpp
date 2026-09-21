/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/hexagon/runtime/hexagon_backend.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <limits>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <flatbuffers/flatbuffers.h>
// Regenerated from the vendored Command.fbs with the flatbuffers version this
// tree vendors; the checked-in copy was produced by an older codegen whose
// accessors do not compile against it. The wire format is unchanged.
#include <Command_generated.h>

#include <executorch/backends/hexagon/serialization/hexagon_schema.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::hexagon {

namespace {

using runtime::ArrayRef;
using runtime::BackendExecutionContext;
using runtime::BackendInitContext;
using runtime::CompileSpec;
using runtime::DelegateHandle;
using runtime::Error;
using runtime::EValue;
using runtime::FreeableBuffer;
using runtime::MemoryAllocator;
using runtime::Result;
using runtime::Span;
using runtime::resize_tensor;

// Diagnostic knobs, all off unless set in the environment:
//   HEXAGON_TRACE=1        print the per-command trace
//   HEXAGON_DELEGATE=n     which delegate, by execute order (-1 = every one)
//   HEXAGON_CMD_START=s    first command of that delegate to run
//   HEXAGON_CMD_LIMIT=k    how many commands to run from there
//   HEXAGON_STOP_AFTER=n   exit once delegate n has run
//   HEXAGON_FAKE_CACHE=1   fill an empty attention past-key/value operand from
//                          the key and value operands, which is what the fixed
//                          emitter does, so the null-pointer write can be
//                          isolated from every other defect in the same blob
// start/limit cut the group array at a command boundary, the cheapest way to
// bracket the command that kills the DSP without touching the exported blob. A
// suffix (start > 0) is how one command is run on its own.
struct TraceConfig {
  bool trace = false;
  int delegate = -1;
  int start = 0;
  int limit = 0;
  int stop_after = -1;
  bool fake_cache = false;
  // HEXAGON_PHASE=1: emit a steady-clock timeline of the init and execute
  // phases, so the host-side setup and per-inference overhead can be split.
  bool phase = false;
};

int EnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::atoi(value);
}

const TraceConfig& Trace() {
  static const TraceConfig config = [] {
    TraceConfig c;
    c.trace = EnvInt("HEXAGON_TRACE", 0) != 0;
    c.delegate = EnvInt("HEXAGON_DELEGATE", -1);
    c.start = EnvInt("HEXAGON_CMD_START", 0);
    c.limit = EnvInt("HEXAGON_CMD_LIMIT", 0);
    c.stop_after = EnvInt("HEXAGON_STOP_AFTER", -1);
    c.fake_cache = EnvInt("HEXAGON_FAKE_CACHE", 0) != 0;
    c.phase = EnvInt("HEXAGON_PHASE", 0) != 0;
    return c;
  }();
  return config;
}

// Layout owned by execute_command.cc: four header ints at kProbeBaseInts inside
// the profile buffer, then one eight-int record per command index.
constexpr int kProbeBaseInts = 1024;
constexpr int kProbeHeaderInts = 4;
constexpr int kProbeRecordInts = 8;
constexpr int kProbeRecords = 508;
// Sixteen stage records after the command records, four ints each.
constexpr int kProbeStages = 80;
constexpr int kProbeBytes =
    (kProbeBaseInts + kProbeHeaderInts + kProbeRecords * kProbeRecordInts + kProbeStages * 4) * 4;
constexpr int32_t kProbeMagic = 0x48455850; // "HEXP"

// DSP_OP_FLASH_ATTN, the one command whose empty cache slots are fatal.
constexpr int32_t kFlashAttnOp = 18;

// A run of bytes inside the arena.
// Round-to-nearest-even fp32 to fp16, matching what a __fp16 cast produces on
// the device. Written out because the host build has no __fp16.
uint16_t float_to_half_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t biased = (bits >> 23) & 0xFFu;
  uint32_t mantissa = bits & 0x7FFFFFu;
  int32_t exponent = static_cast<int32_t>(biased) - 127 + 15;
  if (biased != 0xFFu && exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half = mantissa >> shift;
    if ((mantissa >> (shift - 1)) & 1u) {
      ++half;
    }
    return static_cast<uint16_t>(sign | half);
  }
  if (biased == 0xFFu) {
    // Infinities and NaN keep their payload in the top mantissa bit.
    return static_cast<uint16_t>(
        sign | 0x7C00u | (mantissa != 0 ? 0x200u : 0u));
  }
  if (exponent >= 31) {
    // Finite but past fp16's range, so it saturates to infinity.
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  mantissa += 0x1000u;
  if (mantissa & 0x800000u) {
    mantissa = 0;
    if (++exponent >= 31) {
      return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

// The same conversion, four lanes at a time, because the scalar form is the
// whole cost of handing an fp32 tensor to an fp16 kernel: a layer's residual
// stream is 64x1024 values and the loop below ran one branchy scalar per value.
// The scalar form rounds by adding 0x1000 to the mantissa and carrying into the
// exponent when that overflows, and both steps are the same integer arithmetic
// on all four lanes, so the vector form copies it exactly instead of using
// vcvt_f16_f32 (which rounds to nearest even and would change the bits a
// re-run produces). Lanes fp16 cannot reach -- subnormal, infinity, overflow --
// go back to the scalar form; on the ViT's residuals they are rare.
void narrow_fp32_to_fp16(const float* from, uint16_t* to, size_t elements) {
#if defined(__aarch64__)
  const uint32x4_t mantissa_mask = vdupq_n_u32(0x7FFFFFu);
  const uint32x4_t round_addend = vdupq_n_u32(0x1000u);
  const uint32x4_t exponent_mask = vdupq_n_u32(0xFFu);
  const uint32x4_t sign_mask = vdupq_n_u32(0x8000u);
  const uint32x4_t low_exponent = vdupq_n_u32(112u);
  const uint32x4_t high_exponent = vdupq_n_u32(143u);
  const uint32x4_t half_exponent_bias = vdupq_n_u32(112u);
  const uint32x4_t half_mantissa_mask = vdupq_n_u32(0x3FFu);
  const uint32x4_t max_exponent = vdupq_n_u32(31u);
  const uint32x4_t infinity = vdupq_n_u32(0x7C00u);

  size_t i = 0;
  for (; i + 4 <= elements; i += 4) {
    const uint32x4_t bits = vld1q_u32(reinterpret_cast<const uint32_t*>(from + i));
    const uint32x4_t exponent = vandq_u32(vshrq_n_u32(bits, 23), exponent_mask);
    const uint32x4_t sign = vandq_u32(vshrq_n_u32(bits, 16), sign_mask);
    const uint32x4_t rounded = vaddq_u32(vandq_u32(bits, mantissa_mask), round_addend);
    const uint32x4_t half_exp = vaddq_u32(
        vsubq_u32(exponent, half_exponent_bias), vshrq_n_u32(rounded, 23));
    uint32x4_t half = vorrq_u32(
        sign,
        vorrq_u32(
            vshlq_n_u32(half_exp, 10),
            vandq_u32(vshrq_n_u32(rounded, 13), half_mantissa_mask)));
    half = vbslq_u32(vcgeq_u32(half_exp, max_exponent), vorrq_u32(sign, infinity), half);

    const uint32x4_t unusual =
        vorrq_u32(vcleq_u32(exponent, low_exponent), vcgeq_u32(exponent, high_exponent));
    if (vmaxvq_u32(unusual) != 0) {
      for (int lane = 0; lane < 4; lane++) {
        to[i + lane] = float_to_half_bits(from[i + lane]);
      }
      continue;
    }
    vst1_u16(to + i, vmovn_u32(half));
  }
  for (; i < elements; i++) {
    to[i] = float_to_half_bits(from[i]);
  }
#else
  for (size_t i = 0; i < elements; i++) {
    to[i] = float_to_half_bits(from[i]);
  }
#endif
}

// The inverse of the narrowing above. The algorithm was checked against numpy
// over every finite fp16 pattern before it was written down here.
float half_bits_to_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t biased = (bits >> 10) & 0x1Fu;
  const uint32_t mantissa = bits & 0x3FFu;
  uint32_t out = 0;
  if (biased == 0) {
    if (mantissa == 0) {
      out = sign;
    } else {
      // Subnormal: shift the leading bit up to where the implicit one sits.
      uint32_t value = mantissa;
      uint32_t shift = 0;
      while ((value & 0x400u) == 0) {
        value <<= 1;
        ++shift;
      }
      out = sign | ((113u - shift) << 23) | ((value & 0x3FFu) << 13);
    }
  } else if (biased == 0x1Fu) {
    out = sign | 0x7F800000u | (mantissa != 0 ? 0x400000u : 0u);
  } else {
    out = sign | ((biased - 15u + 127u) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &out, sizeof(result));
  return result;
}

struct Region {
  size_t offset = 0;
  size_t size = 0;
};

struct HexagonDelegate {
  HexagonDriver driver;
  // Written once at init and read on every execute: the command descriptors,
  // the sync group, the command group and the weights. Private to this
  // delegate.
  Arena resident;
  // The per-execute half: the method inputs, the activations and the method
  // outputs. Shared through the pool with every delegate of the same size,
  // because nothing written here outlives an execute.
  Arena scratch;

  // Host-written once at init.
  Region group; // command group array
  Region commands; // the command descriptors
  Region sync; // SyncGroup flatbuffer

  // The four tensor sections, in blob order. An offset is measured from the
  // start of the block its section lives in: weights in resident, the other
  // three in scratch.
  Region weights;
  Region input_section;
  Region activations;
  Region output_section;

  uint32_t n_ops = 0;
  uint32_t activations_bytes = 0;
  std::vector<HexagonOp> ops;
  std::vector<size_t> command_offsets;
  // Order this delegate was created in, for the trace only.
  int index = 0;
  // Host-visible landing zone for the DSP-side per-command trace. Empty unless
  // the environment asked for a trace.
  Arena probe;
  // Block-relative location of each method input/output, in signature order.
  std::vector<Region> inputs;
  std::vector<Region> outputs;

  // Params the emitter could not know, resolved from an input on every
  // execute. Only ops with HexagonOp::patch_param set appear here.
  struct Patch {
    uint8_t* slot; // param slot to overwrite, in the resident block
    const uint8_t* source; // where the value is read from, in either block
    HexagonTensorRef source_ref;
    uint32_t scale; // multiplier between the two
    // Ints from slot to a second param that has to move with the position, or
    // zero. Attention carries the sequence extent seven ints past start_pos.
    uint32_t extent_offset;
  };
  std::vector<Patch> patches;

  struct DynamicPatch {
    uint8_t* slot;
    int32_t scale;
    int32_t add;
  };
  uint32_t dynamic_input = 0;
  uint32_t dynamic_axis = 0;
  uint32_t dynamic_max_length = 0;
  uint32_t dynamic_example = 0;
  std::vector<DynamicPatch> dynamic_patches;
  struct DynamicLayout {
    HexagonTensorSpace space;
    uint32_t index;
    int64_t c0;
    int64_t c1;
    int64_t c2;
  };
  std::vector<DynamicLayout> dynamic_layouts;
  struct RuntimeLayout {
    HexagonTensorSpace space;
    uint32_t index;
    size_t offset;
    size_t size;
  };
  std::vector<RuntimeLayout> runtime_layouts;

  // Method inputs the subgraph writes to, by signature index. Their scratch
  // slots are copied back to the caller once the command group has run.
  std::vector<uint32_t> in_place_inputs;
};

// Read back what the DSP managed to write before it faulted. Phase 0 on the
// last written record is the crash signature: the command was entered and never
// returned from.
void PrintProbe(const HexagonDelegate& delegate) {
  if (delegate.probe.ptr == nullptr) {
    return;
  }
  const int32_t* probe = static_cast<const int32_t*>(delegate.probe.ptr);
  const int32_t* header = probe + kProbeBaseInts;
  if (header[0] != kProbeMagic) {
    std::fprintf(
        stderr,
        "[hexagon] probe d%d: no record; the fault came before the command loop\n",
        delegate.index);
    return;
  }
  std::fprintf(stderr, "[hexagon] probe d%d: sent count=%d last=%d\n", delegate.index, header[1], header[3]);
  for (int i = 0; i < header[1] && i < kProbeRecords; i++) {
    const int32_t* rec = probe + kProbeBaseInts + kProbeHeaderInts + i * kProbeRecordInts;
    if (rec[0] != i + 1) {
      continue;
    }
    const char* phase = rec[1] == 0 ? "ENTER (no return)" : (rec[1] == 1 ? "done" : "failed");
    std::fprintf(
        stderr,
        "[hexagon] probe d%d cmd %d: op=%d fd=%d off=%d size=%d %s ret=%d\n",
        delegate.index,
        i,
        rec[5],
        rec[2],
        rec[3],
        rec[4],
        phase,
        rec[6]);
  }
}

// Kernel microseconds per DSP op type, accumulated on the DSP side by
// execute_command.cc into the low slots of the profile buffer. This is the time
// inside the op switch, so it excludes the RPC and the per-command loop.
void PrintOpTimes(const HexagonDelegate& delegate) {
  if (delegate.probe.ptr == nullptr) {
    return;
  }
  const int32_t* slots = static_cast<const int32_t*>(delegate.probe.ptr);
  for (int slot = 0; slot < 256; slot++) {
    if (slots[slot] != 0) {
      std::fprintf(stderr, "[hexagon] optime d%d slot=%d: %d us\n", delegate.index, slot, slots[slot]);
    }
  }
}

// How far inside the kernel the crash happened. A stage is written by the kernel
// itself and flushed, so the last one present is where it was when it died.
const char* StageName(int stage) {
  switch (stage) {
    case 1: return "flash_attn entered";
    case 2: return "pre push_kv";
    case 3: return "post push_kv";
    case 4: return "pre sync_attention";
    case 5: return "post sync_attention";
    case 6: return "push_kv entered";
    case 7: return "push_kv pre-write";
    case 8: return "push_kv post-write";
    case 9: return "past push block";
    case 10: return "push_kv vtcm reserved";
    case 11: return "push_kv jobs submitted";
    case 12: return "push_kv jobs done";
    case 13: return "chunk entered";
    case 14: return "chunk K written";
    case 15: return "chunk V written";
    case 16: return "pre clear v tail";
    case 17: return "chunk enter";
    case 18: return "chunk K done";
    case 19: return "chunk V done";
    case 20: return "chunk exit";
    case 21: return "worker loop exit";
    case 22: return "rpc stack left";
    case 23: return "worker stack left";
    case 24: return "push_kv tasks";
    case 25: return "push_kv heads/dim/maxkv";
    case 26: return "push_kv icP/ocP/chunks";
    case 27: return "push_kv c4/tokoff/seqlen";
    case 28: return "push_kv pastK/K/stride";
    case 29: return "cache write maxK/maxV/cap";
    case 30: return "clamp build";
    case 31: return "dsp worker pool state";
    case 32: return "attn run_tasks entered";
    case 33: return "pre hmx_queue_begin";
    case 34: return "post hmx_queue_begin";
    case 35: return "attn submits done";
    case 36: return "pre synctoken wait";
    case 37: return "post synctoken wait";
    case 38: return "post hmx_queue_end";
    case 39: return "process_head entered";
    case 40: return "pre causal QK";
    case 41: return "post causal QK";
    case 42: return "post causal softmax";
    case 43: return "pre causal SV";
    case 44: return "post causal SV";
    case 45: return "attn worker exit";
    case 47: return "attn_hmx_matmul entered";
    case 48: return "matmul vtcm sized";
    case 49: return "matmul pre compute";
    case 50: return "matmul post store";
    case 51: return "matmul exit";
    case 52: return "queue thread job start";
    case 53: return "queue thread job end";
    case 54: return "pre hmx resource lock";
    case 55: return "post hmx resource lock";
    case 56: return "hmx resource end";
    case 57: return "submit blocking wait";
    case 64: return "matmul enters";
    case 65: return "matmul exits";
    case 66: return "queue idle wakes";
    case 67: return "queue jobs done";
    case 68: return "submit completions";
    case 69: return "submit sem acquires";
    case 70: return "worker loop iters";
    case 71: return "worker exits";
    default: return "unknown";
  }
}

void PrintStages(const HexagonDelegate& delegate) {
  if (delegate.probe.ptr == nullptr) {
    return;
  }
  const int32_t* probe = static_cast<const int32_t*>(delegate.probe.ptr);
  const int32_t* base =
      probe + kProbeBaseInts + kProbeHeaderInts + kProbeRecords * kProbeRecordInts;
  for (int i = 0; i < kProbeStages; i++) {
    const int32_t* rec = base + i * 4;
    if (rec[0] != i + 1) {
      continue;
    }
    std::fprintf(
        stderr,
        "[hexagon] stage d%d %d %s: (%d, %d, %d)\n",
        delegate.index,
        i + 1,
        StageName(i + 1),
        rec[1],
        rec[2],
        rec[3]);
  }
}

// A blocked invoke never returns, and nothing can interrupt the DSP once its side stops
// answering: the calling thread sits in fastrpc_wait_for_completion until the process dies.
// Whatever the DSP already wrote into the probe ring is then the only evidence left, so read
// it out on a timer instead of after the return. HEXAGON_WATCHDOG_SECONDS (default 15, 0
// disables) sets the interval; only a traced delegate has a probe ring to read.
class ProbeWatchdog {
 public:
  ProbeWatchdog(const HexagonDelegate* delegate, bool traced) : delegate_(delegate) {
    const char* env = std::getenv("HEXAGON_WATCHDOG_SECONDS");
    seconds_ = env != nullptr ? std::atoi(env) : 15;
    if (!traced || seconds_ <= 0 || delegate_->probe.ptr == nullptr) {
      return;
    }
    thread_ = std::thread([this] { Loop(); });
  }
  ~ProbeWatchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_.store(true);
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void Loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (int elapsed = seconds_; !stop_.load(); elapsed += seconds_) {
      // Waiting has to be interruptible: a fixed 100 ms sleep charged the destructor
      // whatever was left of the slice on every invoke, traced or not.
      if (cv_.wait_for(lock, std::chrono::seconds(seconds_), [this] { return stop_.load(); })) {
        return;
      }
      std::fprintf(
          stderr,
          "[hexagon] watchdog d%d: still inside the command group after %d s, DSP trace so far:\n",
          delegate_->index,
          elapsed);
      PrintProbe(*delegate_);
      PrintStages(*delegate_);
      std::fflush(nullptr);
    }
  }

  const HexagonDelegate* delegate_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  int seconds_ = 15;
  std::thread thread_;
};

// The operand list of every command in the group, read back out of the
// descriptors the DSP is handed, so the crashing command can be named exactly.
void PrintCommands(const HexagonDelegate& delegate) {
  const uint8_t* resident = static_cast<const uint8_t*>(delegate.resident.ptr);
  const int32_t* group = reinterpret_cast<const int32_t*>(resident + delegate.group.offset);
  for (uint32_t i = 0; i < delegate.n_ops; i++) {
    const int32_t* entry = group + 2 + i * 3;
    const auto* command = flatbuffers::GetRoot<DSPCOMMAND::Command>(
        resident + entry[1] - (int32_t)delegate.resident.bias);
    if (command == nullptr) {
      continue;
    }
    std::fprintf(stderr, "[hexagon] cmd d%d %u: type=%d", delegate.index, i, command->type());
    const auto* inputs = command->inputs();
    for (uint32_t j = 0; inputs != nullptr && j < inputs->size(); j++) {
      const auto* tensor = inputs->Get(j);
      std::fprintf(stderr, " in%u=(fd%d,%d,%d)", j, tensor->fd(), tensor->offset(), tensor->size());
    }
    const auto* outputs = command->outputs();
    for (uint32_t j = 0; outputs != nullptr && j < outputs->size(); j++) {
      const auto* tensor = outputs->Get(j);
      std::fprintf(stderr, " out%u=(fd%d,%d,%d)", j, tensor->fd(), tensor->offset(), tensor->size());
    }
    const auto* params = command->params();
    std::fprintf(stderr, " params=[");
    for (uint32_t j = 0; params != nullptr && j < params->size(); j++) {
      std::fprintf(stderr, "%s%d", j == 0 ? "" : ",", params->Get(j));
    }
    std::fprintf(stderr, "]\n");
  }
}

// Absolute steady-clock milliseconds. The runner prints its own phase stamps
// off the same clock, so both timelines line up in one process run.
double PhaseNowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Where a section lives in the arena. Every tensor address is this plus the
// offset the AOT step recorded.
size_t SectionBase(const HexagonDelegate& delegate, HexagonTensorSpace space) {
  switch (space) {
    case HexagonTensorSpace::kWeights:
      return delegate.weights.offset;
    case HexagonTensorSpace::kInput:
      return delegate.input_section.offset;
    case HexagonTensorSpace::kActivation:
      return delegate.activations.offset;
    case HexagonTensorSpace::kOutput:
      return delegate.output_section.offset;
    case HexagonTensorSpace::kAbsent:
      break;
  }
  return 0;
}

// Weights are resident and private to the delegate; every other operand is
// scratch, and is addressed through the pooled block's fd.
const Arena& BlockOf(const HexagonDelegate& delegate, HexagonTensorSpace space) {
  return space == HexagonTensorSpace::kWeights ? delegate.resident
                                               : delegate.scratch;
}

bool IsKnownSpace(uint32_t space) {
  return space <= static_cast<uint32_t>(HexagonTensorSpace::kActivation) ||
      space == static_cast<uint32_t>(HexagonTensorSpace::kAbsent);
}

bool IsAbsent(const HexagonTensorRef& ref) {
  return ref.space == static_cast<uint32_t>(HexagonTensorSpace::kAbsent);
}

// An absent operand goes out as fd = -1, which the dispatcher maps to a null
// pointer; anything else is addressed through its section.
flatbuffers::Offset<DSPCOMMAND::Tensor> MakeTensor(
    flatbuffers::FlatBufferBuilder& builder,
    const HexagonDelegate& delegate,
    const HexagonTensorRef& ref) {
  if (IsAbsent(ref)) {
    return DSPCOMMAND::CreateTensor(builder, -1, 0, 0);
  }
  const size_t base =
      SectionBase(delegate, (HexagonTensorSpace)ref.space) + ref.offset;
  const Arena& block = BlockOf(delegate, (HexagonTensorSpace)ref.space);
  return DSPCOMMAND::CreateTensor(
      builder, block.fd, (int32_t)(base + block.bias), (int32_t)ref.size);
}

const HexagonDelegate::DynamicLayout* FindDynamicLayout(
    const HexagonDelegate& delegate,
    HexagonTensorSpace space,
    uint32_t index) {
  for (const auto& layout : delegate.dynamic_layouts) {
    if (layout.space == space && layout.index == index) {
      return &layout;
    }
  }
  return nullptr;
}

bool EvalDynamicBytes(
    const HexagonDelegate::DynamicLayout& layout,
    int64_t length,
    size_t* bytes) {
  const __int128 value = static_cast<__int128>(layout.c0) +
      static_cast<__int128>(layout.c1) * length +
      static_cast<__int128>(layout.c2) * length * length;
  if (value < 0 || value > std::numeric_limits<size_t>::max()) {
    return false;
  }
  *bytes = static_cast<size_t>(value);
  return true;
}

const HexagonDelegate::RuntimeLayout* FindRuntimeLayout(
    const HexagonDelegate& delegate,
    HexagonTensorSpace space,
    uint32_t index) {
  for (const auto& layout : delegate.runtime_layouts) {
    if (layout.space == space && layout.index == index) {
      return &layout;
    }
  }
  return nullptr;
}

bool AddLayoutBlock(
    std::vector<std::pair<size_t, size_t>>& free,
    size_t requested,
    size_t* arena_end,
    size_t* offset) {
  auto candidate = free.end();
  for (auto it = free.begin(); it != free.end(); ++it) {
    if (it->second >= requested &&
        (candidate == free.end() || it->second < candidate->second)) {
      candidate = it;
    }
  }
  if (candidate != free.end()) {
    *offset = candidate->first;
    free.erase(candidate);
    return true;
  }
  for (auto it = free.begin(); it != free.end(); ++it) {
    if (it->first + it->second == *arena_end) {
      *offset = it->first;
      free.erase(it);
      *arena_end = *offset + requested;
      return true;
    }
  }
  *offset = AlignUp(*arena_end, kHexagonAlignment);
  *arena_end = *offset + requested;
  return true;
}

Error ResizeDynamicDelegate(HexagonDelegate* delegate, int64_t length) {
  if (delegate->dynamic_layouts.empty()) {
    return Error::Ok;
  }

  delegate->runtime_layouts.clear();
  delegate->runtime_layouts.reserve(delegate->dynamic_layouts.size());
  for (const auto& layout : delegate->dynamic_layouts) {
    size_t size = 0;
    if (!EvalDynamicBytes(layout, length, &size)) {
      ET_LOG(Error, "hexagon: dynamic layout size overflow");
      return Error::InvalidArgument;
    }
    delegate->runtime_layouts.push_back(
        {layout.space, layout.index, 0, size});
  }

  auto size_for = [&](HexagonTensorSpace space, uint32_t index, size_t fallback) {
    const auto* layout = FindRuntimeLayout(*delegate, space, index);
    return layout == nullptr ? fallback : layout->size;
  };

  size_t input_bytes = 0;
  for (uint32_t i = 0; i < delegate->inputs.size(); i++) {
    auto* layout = const_cast<HexagonDelegate::RuntimeLayout*>(
        FindRuntimeLayout(*delegate, HexagonTensorSpace::kInput, i));
    const size_t size =
        size_for(HexagonTensorSpace::kInput, i, delegate->inputs[i].size);
    if (layout != nullptr) {
      layout->offset = input_bytes;
    }
    delegate->inputs[i] = Region{input_bytes, size};
    input_bytes = AlignUp(input_bytes + size, kHexagonAlignment);
  }

  struct Lifetime {
    uint32_t index;
    uint32_t first;
    uint32_t last;
    size_t size;
  };
  std::vector<Lifetime> lifetimes;
  for (uint32_t op_index = 0; op_index < delegate->ops.size(); op_index++) {
    const auto& op = delegate->ops[op_index];
    auto visit = [&](const HexagonTensorRef& ref) {
      if (ref.space != static_cast<uint32_t>(HexagonTensorSpace::kActivation)) {
        return;
      }
      auto it = std::find_if(
          lifetimes.begin(), lifetimes.end(),
          [&](const Lifetime& value) { return value.index == ref.index; });
      const size_t size = size_for(
          HexagonTensorSpace::kActivation, ref.index, ref.size);
      if (it == lifetimes.end()) {
        lifetimes.push_back({ref.index, op_index, op_index, size});
      } else {
        it->first = std::min(it->first, op_index);
        it->last = std::max(it->last, op_index);
        it->size = std::max(it->size, size);
      }
    };
    for (uint32_t i = 0; i < op.n_inputs; i++) {
      visit(op.inputs[i]);
    }
    for (uint32_t i = 0; i < op.n_outputs; i++) {
      visit(op.outputs[i]);
    }
  }
  std::sort(
      lifetimes.begin(), lifetimes.end(),
      [](const Lifetime& lhs, const Lifetime& rhs) {
        return std::tie(lhs.first, lhs.index) < std::tie(rhs.first, rhs.index);
      });

  std::vector<std::tuple<uint32_t, size_t, size_t>> active;
  std::vector<std::pair<size_t, size_t>> free;
  size_t activation_bytes = 0;
  for (const auto& lifetime : lifetimes) {
    for (auto it = active.begin(); it != active.end();) {
      if (std::get<0>(*it) < lifetime.first) {
        free.emplace_back(std::get<1>(*it), std::get<2>(*it));
        it = active.erase(it);
      } else {
        ++it;
      }
    }
    size_t offset = 0;
    AddLayoutBlock(free, lifetime.size, &activation_bytes, &offset);
    auto* layout = const_cast<HexagonDelegate::RuntimeLayout*>(
        FindRuntimeLayout(*delegate, HexagonTensorSpace::kActivation, lifetime.index));
    if (layout != nullptr) {
      layout->offset = offset;
      layout->size = lifetime.size;
    }
    active.emplace_back(lifetime.last, offset, lifetime.size);
  }

  size_t cursor = 0;
  delegate->input_section = Region{cursor, input_bytes};
  cursor = AlignUp(cursor + input_bytes, kHexagonAlignment);
  delegate->activations = Region{cursor, activation_bytes};
  cursor = AlignUp(cursor + activation_bytes, kHexagonAlignment);

  size_t output_bytes = 0;
  for (uint32_t i = 0; i < delegate->outputs.size(); i++) {
    auto* layout = const_cast<HexagonDelegate::RuntimeLayout*>(
        FindRuntimeLayout(*delegate, HexagonTensorSpace::kOutput, i));
    const size_t size = size_for(
        HexagonTensorSpace::kOutput, i, delegate->outputs[i].size);
    if (layout != nullptr) {
      layout->offset = output_bytes;
    }
    delegate->outputs[i] = Region{output_bytes, size};
    output_bytes = AlignUp(output_bytes + size, kHexagonAlignment);
  }
  delegate->output_section = Region{cursor, output_bytes};
  for (uint32_t i = 0; i < delegate->outputs.size(); i++) {
    delegate->outputs[i].offset += delegate->output_section.offset;
  }
  cursor = AlignUp(cursor + output_bytes, kHexagonAlignment);
  const size_t scratch_bytes = std::max<size_t>(cursor, kHexagonAlignment);

  auto pooled = SharedArenaPool::Get().Acquire(scratch_bytes);
  if (!pooled.ok()) {
    return pooled.error();
  }
  delegate->scratch = pooled.get();
  for (auto& patch : delegate->patches) {
    const auto space = static_cast<HexagonTensorSpace>(patch.source_ref.space);
    const auto* layout = FindRuntimeLayout(*delegate, space, patch.source_ref.index);
    const size_t offset = layout == nullptr ? patch.source_ref.offset : layout->offset;
    patch.source = static_cast<const uint8_t*>(BlockOf(*delegate, space).ptr) +
        SectionBase(*delegate, space) + offset;
  }

  auto update_tensor = [&](DSPCOMMAND::Tensor* tensor, const HexagonTensorRef& ref) {
    if (IsAbsent(ref)) {
      return;
    }
    const auto* layout = FindRuntimeLayout(*delegate,
        static_cast<HexagonTensorSpace>(ref.space), ref.index);
    const size_t offset = layout == nullptr ? ref.offset : layout->offset;
    const size_t size = layout == nullptr ? ref.size : layout->size;
    const auto space = static_cast<HexagonTensorSpace>(ref.space);
    const Arena& block = BlockOf(*delegate, space);
    tensor->mutate_fd(block.fd);
    tensor->mutate_offset(static_cast<int32_t>(SectionBase(*delegate, space) + offset + block.bias));
    tensor->mutate_size(static_cast<int32_t>(size));
  };

  for (uint32_t i = 0; i < delegate->ops.size(); i++) {
    auto* command = flatbuffers::GetMutableRoot<DSPCOMMAND::Command>(
        static_cast<uint8_t*>(delegate->resident.ptr) + delegate->command_offsets[i]);
    auto* inputs = command->mutable_inputs();
    for (uint32_t j = 0; j < delegate->ops[i].n_inputs; j++) {
      update_tensor(const_cast<DSPCOMMAND::Tensor*>(inputs->Get(j)), delegate->ops[i].inputs[j]);
    }
    auto* outputs = command->mutable_outputs();
    for (uint32_t j = 0; j < delegate->ops[i].n_outputs; j++) {
      update_tensor(const_cast<DSPCOMMAND::Tensor*>(outputs->Get(j)), delegate->ops[i].outputs[j]);
    }
    // The exported KV tensors retain the model's upper-bound cache shape, while
    // a prefill call only makes length + 1 positions valid (the extra position
    // is the inclusive cache endpoint used by the attention kernel). Patch the
    // logical extent independently of the backing allocation.
    if (delegate->ops[i].type == kFlashAttnOp &&
        delegate->ops[i].n_inputs > 2 && delegate->ops[i].n_params > 8) {
      auto* params = command->mutable_params();
      params->Mutate(8, static_cast<int32_t>(length + 1));
    }
    // The prefill graph returns the last token through a raster blit.  Its
    // traced byte offset is the example length's last row, so update it with
    // the actual request length before the command group is rebuilt.
    if (delegate->ops[i].type == 3 && delegate->ops[i].n_params > 8 &&
        delegate->ops[i].params[4] > 0 && delegate->ops[i].params[7] == 1 &&
        delegate->ops[i].params[8] > 0 &&
        delegate->ops[i].params[4] % delegate->ops[i].params[8] == 0) {
      auto* params = command->mutable_params();
      params->Mutate(4, static_cast<int32_t>((length - 1) * delegate->ops[i].params[8]));
      if (delegate->ops[i].n_params > 10 && delegate->dynamic_example != 0) {
        params->Mutate(10, static_cast<int32_t>(
            static_cast<int64_t>(delegate->ops[i].params[10]) * length /
            delegate->dynamic_example));
      }
    }
    if (delegate->ops[i].type == 3 && delegate->ops[i].n_params > 8 &&
        delegate->ops[i].params[7] > 1 &&
        (delegate->ops[i].params[8] == 64 || delegate->ops[i].params[8] == 128) &&
        delegate->ops[i].params[7] % 16 == 0) {
      auto* params = command->mutable_params();
      const int32_t rows = delegate->dynamic_example == 0
          ? static_cast<int32_t>(length * 16)
          : static_cast<int32_t>(static_cast<int64_t>(delegate->ops[i].params[7]) * length /
                                 delegate->dynamic_example);
      params->Mutate(7, rows);
      if (delegate->ops[i].n_params > 19 && delegate->ops[i].params[19] == delegate->ops[i].params[7]) {
        params->Mutate(19, rows);
      }
    }
    if (delegate->ops[i].type == 3 && delegate->ops[i].n_params > 8 &&
        delegate->ops[i].params[7] == 1 && delegate->ops[i].params[4] == 0 &&
        delegate->ops[i].params[8] > 1 && delegate->dynamic_example != 0) {
      auto* params = command->mutable_params();
      params->Mutate(8, static_cast<int32_t>(
          static_cast<int64_t>(delegate->ops[i].params[8]) * length /
          delegate->dynamic_example));
    }
  }

  flatbuffers::FlatBufferBuilder builder(1u << 20);
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> sync_in;
  std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> sync_out;
  for (const auto& op : delegate->ops) {
    for (uint32_t j = 0; j < op.n_inputs; j++) {
      if (!IsAbsent(op.inputs[j])) {
        const auto& ref = op.inputs[j];
        const auto* layout = FindRuntimeLayout(*delegate,
            static_cast<HexagonTensorSpace>(ref.space), ref.index);
        const size_t offset = layout == nullptr ? ref.offset : layout->offset;
        const size_t size = layout == nullptr ? ref.size : layout->size;
        const auto space = static_cast<HexagonTensorSpace>(ref.space);
        const Arena& block = BlockOf(*delegate, space);
        sync_in.push_back(DSPCOMMAND::CreateTensor(
            builder, block.fd,
            static_cast<int32_t>(SectionBase(*delegate, space) + offset + block.bias),
            static_cast<int32_t>(size)));
      }
    }
    for (uint32_t j = 0; j < op.n_outputs; j++) {
      if (!IsAbsent(op.outputs[j])) {
        const auto& ref = op.outputs[j];
        const auto* layout = FindRuntimeLayout(*delegate,
            static_cast<HexagonTensorSpace>(ref.space), ref.index);
        const size_t offset = layout == nullptr ? ref.offset : layout->offset;
        const size_t size = layout == nullptr ? ref.size : layout->size;
        const auto space = static_cast<HexagonTensorSpace>(ref.space);
        const Arena& block = BlockOf(*delegate, space);
        sync_out.push_back(DSPCOMMAND::CreateTensor(
            builder, block.fd,
            static_cast<int32_t>(SectionBase(*delegate, space) + offset + block.bias),
            static_cast<int32_t>(size)));
      }
    }
  }
  builder.Finish(DSPCOMMAND::CreateSyncGroup(
      builder, builder.CreateVector(sync_in), builder.CreateVector(sync_out)));
  if (builder.GetSize() > delegate->sync.size) {
    ET_LOG(Error, "hexagon: resized sync group exceeds budget");
    return Error::Internal;
  }
  delegate->sync.size = builder.GetSize();
  std::memcpy(
      static_cast<uint8_t*>(delegate->resident.ptr) + delegate->sync.offset,
      builder.GetBufferPointer(), delegate->sync.size);
  delegate->activations_bytes = activation_bytes;
  return Error::Ok;
}

} // namespace

bool HexagonBackend::is_available() const {
  return true;
}

Result<DelegateHandle*> HexagonBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)compile_specs;

  if (processed == nullptr || processed->size() < sizeof(HexagonBlobHeader)) {
    ET_LOG(Error, "hexagon: blob too small");
    return Error::DelegateInvalidCompatibility;
  }

  const auto* header =
      reinterpret_cast<const HexagonBlobHeader*>(processed->data());
  if (header->magic != kHexagonBlobMagic ||
      header->version != kHexagonBlobVersion) {
    ET_LOG(
        Error,
        "hexagon: bad blob magic 0x%08x or version %u",
        header->magic,
        header->version);
    return Error::DelegateInvalidCompatibility;
  }
  if (header->n_ops > 4096 || header->n_inputs > 64 ||
      header->n_outputs > 64) {
    ET_LOG(Error, "hexagon: implausible blob dimensions");
    return Error::DelegateInvalidCompatibility;
  }

  const size_t ops_bytes = sizeof(HexagonOp) * header->n_ops;
  if (processed->size() < sizeof(HexagonBlobHeader) + ops_bytes) {
    ET_LOG(Error, "hexagon: truncated blob");
    return Error::DelegateInvalidCompatibility;
  }

  MemoryAllocator* allocator = context.get_runtime_allocator();
  auto* delegate = allocator->allocateInstance<HexagonDelegate>();
  if (delegate == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  // Assigned here rather than at the end of init: the phase timeline needs an
  // ordinal from the first stage on.
  {
    static int next_delegate = 0;
    delegate->index = next_delegate++;
  }
  const bool phase = Trace().phase;
  auto stamp = [&](const char* what) {
    if (phase) {
      std::fprintf(
          stderr, "[phase] t=%.1f init d%d %s\n", PhaseNowMs(), delegate->index, what);
    }
  };
  stamp("alloc_delegate");

  auto driver = HexagonDriver::Create();
  if (!driver.ok()) {
    ET_LOG(Error, "hexagon: no DSP session");
    return driver.error();
  }
  delegate->driver = std::move(driver.get());
  stamp("driver_open");
  ET_LOG(Info, "hexagon: skel arch V%02X, arena %s", delegate->driver.skel_arch(),
      delegate->driver.cached() ? "cached" : "uncached");

  // The descriptors are built here rather than measured, so they get a budget:
  // twice the size of the records they came from, plus slack for per-op
  // flatbuffer overhead. The sync group duplicates every operand, at roughly
  // 32 bytes each.
  const size_t command_budget = AlignUp(ops_bytes * 2 + 8192, kHexagonAlignment);
  const size_t sync_budget = AlignUp(
      (header->n_ops * (kMaxOpInputs + kMaxOpOutputs) + 16) * 32, kHexagonAlignment);
  const size_t group_bytes = 8 + header->n_ops * 3 * sizeof(int32_t);

  size_t resident_bytes = command_budget + sync_budget;
  resident_bytes = AlignUp(resident_bytes + header->weights_bytes, kHexagonAlignment);
  resident_bytes = AlignUp(resident_bytes + group_bytes, kHexagonAlignment);

  size_t scratch_bytes = 0;
  scratch_bytes = AlignUp(scratch_bytes + header->inputs_bytes, kHexagonAlignment);
  scratch_bytes = AlignUp(scratch_bytes + header->activations_bytes, kHexagonAlignment);
  scratch_bytes = AlignUp(scratch_bytes + header->outputs_bytes, kHexagonAlignment);
  // rpcmem rejects a zero-length request, and a subgraph with no operand of its
  // own is still a legal subgraph.
  scratch_bytes = scratch_bytes == 0 ? kHexagonAlignment : scratch_bytes;

  std::fprintf(
      stderr,
      "[hexagon] arena: resident %zu, scratch %zu (weights %zu, activations %zu)\n",
      resident_bytes,
      scratch_bytes,
      (size_t)header->weights_bytes,
      (size_t)header->activations_bytes);
  auto arena = delegate->driver.Alloc(resident_bytes);
  if (!arena.ok()) {
    std::fprintf(stderr, "[hexagon] rpcmem allocation failed\n");
    return arena.error();
  }
  delegate->resident = arena.get();
  stamp("resident_alloc");
  // No zero-fill. Every byte of this block is written before it is read: the
  // weights section by the memcpy below, the command and sync sections by the
  // descriptor builder, and the group array by its own memset. Zeroing it first
  // cost a full pass over the weights -- 659 MB, a third of a second, on every
  // load -- to clear memory that the next few lines overwrite.
  // (The scratch block is still zeroed by the pool: it holds activation slots,
  // which the blob does not carry, so the first execution of a model would
  // otherwise read whatever the previous delegate left behind.)

  auto pooled = SharedArenaPool::Get().Acquire(scratch_bytes);
  if (!pooled.ok()) {
    std::fprintf(stderr, "[hexagon] rpcmem allocation failed\n");
    return pooled.error();
  }
  delegate->scratch = pooled.get();
  stamp("scratch_alloc");

  const TraceConfig& config = Trace();
  if (config.trace || config.delegate >= 0) {
    auto probe = delegate->driver.Alloc(kProbeBytes);
    if (probe.ok()) {
      delegate->probe = probe.get();
      std::memset(delegate->probe.ptr, 0, kProbeBytes);
    }
  }

  const auto* ops = reinterpret_cast<const HexagonOp*>(
      reinterpret_cast<const uint8_t*>(processed->data()) +
      sizeof(HexagonBlobHeader));
  delegate->ops.assign(ops, ops + header->n_ops);
  delegate->command_offsets.reserve(header->n_ops);
  const uint8_t* blob = reinterpret_cast<const uint8_t*>(processed->data());
  const size_t weights_blob_offset = sizeof(HexagonBlobHeader) + ops_bytes;
  // Only the weights are on disk. Every other section is a size the runtime
  // works from -- the input and output sizes are arena budgets it copies to and
  // from the caller, and the activation section is scratch it reserves for
  // itself and never reads out of the blob -- so only the weights have to be
  // present. A blob written before the padding was dropped still carries the
  // zeros, which this accepts too: it is a lower bound, not an equality.
  const size_t sections_total = static_cast<size_t>(header->weights_bytes);
  if (weights_blob_offset + sections_total > processed->size()) {
    ET_LOG(Error, "hexagon: tensor sections out of bounds");
    return Error::DelegateInvalidCompatibility;
  }

  const size_t trailer_offset = weights_blob_offset + sections_total;
  if (processed->size() > trailer_offset) {
    if (processed->size() - trailer_offset < sizeof(HexagonDynamicTrailer)) {
      ET_LOG(Error, "hexagon: truncated dynamic trailer");
      return Error::DelegateInvalidCompatibility;
    }
    const auto* trailer = reinterpret_cast<const HexagonDynamicTrailer*>(
        blob + trailer_offset);
    if (trailer->magic != kHexagonDynamicTrailerMagic ||
        (trailer->version < 1 || trailer->version > 3) ||
        trailer->input_index >= header->n_inputs ||
        trailer->max_length == 0) {
      ET_LOG(Error, "hexagon: invalid dynamic trailer");
      return Error::DelegateInvalidCompatibility;
    }
    const size_t trailer_bytes = trailer->version >= 3 ? sizeof(HexagonDynamicTrailerV3) : sizeof(HexagonDynamicTrailer);
    if (processed->size() - trailer_offset < trailer_bytes) {
      ET_LOG(Error, "hexagon: truncated dynamic trailer v%u", trailer->version);
      return Error::DelegateInvalidCompatibility;
    }
    const size_t patches_bytes =
        static_cast<size_t>(trailer->n_patches) * sizeof(HexagonDynamicPatch);
    const size_t layout_header_bytes =
        trailer->version >= 2 ? sizeof(HexagonDynamicLayoutHeader) : 0;
    if (trailer_bytes + patches_bytes + layout_header_bytes >
        processed->size() - trailer_offset) {
      ET_LOG(Error, "hexagon: dynamic patch records out of bounds");
      return Error::DelegateInvalidCompatibility;
    }
    delegate->dynamic_input = trailer->input_index;
    delegate->dynamic_axis = trailer->axis;
    delegate->dynamic_max_length = trailer->max_length;
    delegate->dynamic_example = trailer->version >= 3
        ? reinterpret_cast<const HexagonDynamicTrailerV3*>(trailer)->example_length : 0;
    const auto* records = reinterpret_cast<const HexagonDynamicPatch*>(
        blob + trailer_offset + trailer_bytes);
    delegate->dynamic_patches.reserve(trailer->n_patches);
    for (uint32_t i = 0; i < trailer->n_patches; i++) {
      if (records[i].op_index < 0 ||
          static_cast<uint32_t>(records[i].op_index) >= header->n_ops ||
          records[i].param_index < 0 || records[i].scale <= 0) {
        ET_LOG(Error, "hexagon: invalid dynamic patch %u", i);
        return Error::DelegateInvalidCompatibility;
      }
    }
    if (trailer->version >= 2) {
      const auto* layout_header = reinterpret_cast<const HexagonDynamicLayoutHeader*>(
          blob + trailer_offset + trailer_bytes + patches_bytes);
      const size_t layout_bytes = static_cast<size_t>(layout_header->n_layouts) *
          sizeof(HexagonDynamicLayout);
      if (layout_header->n_layouts > 100000 ||
          trailer_bytes + patches_bytes +
                  sizeof(HexagonDynamicLayoutHeader) + layout_bytes >
              processed->size() - trailer_offset) {
        ET_LOG(Error, "hexagon: dynamic layout records out of bounds");
        return Error::DelegateInvalidCompatibility;
      }
      const auto* layouts = reinterpret_cast<const HexagonDynamicLayout*>(
          reinterpret_cast<const uint8_t*>(layout_header) +
          sizeof(HexagonDynamicLayoutHeader));
      delegate->dynamic_layouts.reserve(layout_header->n_layouts);
      for (uint32_t i = 0; i < layout_header->n_layouts; i++) {
        if (layouts[i].space > static_cast<uint32_t>(HexagonTensorSpace::kActivation) ||
            layouts[i].c0 < 0 || layouts[i].c1 < 0 || layouts[i].c2 < 0) {
          ET_LOG(Error, "hexagon: invalid dynamic layout %u", i);
          return Error::DelegateInvalidCompatibility;
        }
        delegate->dynamic_layouts.push_back({
            static_cast<HexagonTensorSpace>(layouts[i].space),
            layouts[i].index,
            layouts[i].c0,
            layouts[i].c1,
            layouts[i].c2});
      }
    }
    if (delegate->dynamic_axis > 7) {
      ET_LOG(Error, "hexagon: dynamic axis %u is invalid", delegate->dynamic_axis);
      return Error::DelegateInvalidCompatibility;
    }
  }

  // The host-written regions go first, each bounded by its own budget: an
  // overrun has to fail at the write site, not silently land on the weights.
  size_t cursor = 0;
  delegate->commands.offset = cursor;
  delegate->commands.size = command_budget;
  cursor += command_budget;

  delegate->sync.offset = cursor;
  delegate->sync.size = sync_budget;
  cursor += sync_budget;

  auto place = [&](Region& region, size_t bytes) {
    region.offset = cursor;
    region.size = bytes;
    cursor = AlignUp(cursor + bytes, kHexagonAlignment);
  };
  place(delegate->weights, header->weights_bytes);

  // The group array the DSP walks: a header then 3 int32 per command.
  delegate->group.offset = cursor;
  delegate->group.size = group_bytes;
  auto* group = reinterpret_cast<int32_t*>(
      static_cast<uint8_t*>(delegate->resident.ptr) + cursor);
  std::memset(group, 0, delegate->group.size);
  cursor = AlignUp(cursor + delegate->group.size, kHexagonAlignment);

  if (cursor > delegate->resident.bytes) {
    ET_LOG(
        Error,
        "hexagon: layout needs %zu bytes, resident block has %zu",
        cursor,
        delegate->resident.bytes);
    return Error::Internal;
  }

  std::memcpy(
      static_cast<uint8_t*>(delegate->resident.ptr) + delegate->weights.offset,
      blob + weights_blob_offset,
      header->weights_bytes);
  stamp("weights_copy");

  cursor = 0;
  place(delegate->input_section, header->inputs_bytes);
  place(delegate->activations, header->activations_bytes);
  place(delegate->output_section, header->outputs_bytes);

  if (cursor > delegate->scratch.bytes) {
    ET_LOG(
        Error,
        "hexagon: scratch needs %zu bytes, pooled block has %zu",
        cursor,
        delegate->scratch.bytes);
    return Error::Internal;
  }

  delegate->inputs.assign(header->n_inputs, Region{});
  delegate->outputs.assign(header->n_outputs, Region{});

  // Bind each method argument to the slot its tensors already refer to.
  for (uint32_t i = 0; i < header->n_ops; i++) {
    for (uint32_t j = 0; j < ops[i].n_inputs; j++) {
      const auto& ref = ops[i].inputs[j];
      if (!IsKnownSpace(ref.space)) {
        ET_LOG(Error, "hexagon: op %u input %u has bad space %u", i, j, ref.space);
        return Error::DelegateInvalidCompatibility;
      }
      if (ref.space == static_cast<uint32_t>(HexagonTensorSpace::kInput)) {
        if (ref.index >= header->n_inputs) {
          ET_LOG(Error, "hexagon: op %u input index %u out of range", i, ref.index);
          return Error::DelegateInvalidCompatibility;
        }
        delegate->inputs[ref.index] =
            Region{delegate->input_section.offset + ref.offset, (size_t)ref.size};
      }
    }
    for (uint32_t j = 0; j < ops[i].n_outputs; j++) {
      const auto& ref = ops[i].outputs[j];
      if (!IsKnownSpace(ref.space)) {
        ET_LOG(Error, "hexagon: op %u output %u has bad space %u", i, j, ref.space);
        return Error::DelegateInvalidCompatibility;
      }
      if (ref.space == static_cast<uint32_t>(HexagonTensorSpace::kOutput)) {
        if (ref.index >= header->n_outputs) {
          ET_LOG(Error, "hexagon: op %u output index %u out of range", i, ref.index);
          return Error::DelegateInvalidCompatibility;
        }
        delegate->outputs[ref.index] =
            Region{delegate->output_section.offset + ref.offset, (size_t)ref.size};
      }
    }
  }

  // Build the descriptors and the sync group on the host, then copy them in.
  // FlatBufferBuilder cannot write straight into the arena without knowing its
  // size up front, and flatbuffer offsets are self-relative, so relocating a
  // finished buffer is safe.
  flatbuffers::FlatBufferBuilder builder(1u << 20);
  size_t command_cursor = delegate->commands.offset;

  for (uint32_t i = 0; i < header->n_ops; i++) {
    HexagonOp op = delegate->ops[i];
    // HEXAGON_FAKE_CACHE: an attention command emitted before the emitter was
    // fixed still points its past key and value slots at fd -1, which the
    // dispatcher turns into the null pointers htp_ops_push_kv writes through.
    // Substituting the key and value operands is what the fixed emitter does.
    if (config.fake_cache && op.type == kFlashAttnOp && op.n_inputs >= 6 &&
        IsAbsent(op.inputs[4]) && IsAbsent(op.inputs[5])) {
      op.inputs[4] = op.inputs[1];
      op.inputs[5] = op.inputs[2];
      std::fprintf(stderr, "[hexagon] d%d op %u: past K/V filled from K/V\n", delegate->index, i);
    }
    delegate->ops[i] = op;

    // Reset before the tensors are built, not after: the offsets they return
    // index into this builder, so clearing later leaves CreateCommand holding
    // offsets into a buffer that no longer exists.
    builder.Clear();
    delegate->command_offsets.push_back(command_cursor);

    std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputs;
    inputs.reserve(op.n_inputs);
    for (uint32_t j = 0; j < op.n_inputs; j++) {
      inputs.push_back(MakeTensor(builder, *delegate, op.inputs[j]));
    }

    std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputs;
    outputs.reserve(op.n_outputs);
    for (uint32_t j = 0; j < op.n_outputs; j++) {
      outputs.push_back(MakeTensor(builder, *delegate, op.outputs[j]));
    }

    std::vector<int32_t> params(op.params, op.params + op.n_params);

    for (uint32_t j = 0; j < op.n_inputs; j++) {
      if ((op.in_place & (1u << j)) == 0) {
        continue;
      }
      // Only a method input has a caller-visible buffer to write back to.
      if (op.inputs[j].space !=
          static_cast<uint32_t>(HexagonTensorSpace::kInput)) {
        ET_LOG(
            Error, "hexagon: op %u in-place input %u is not a method input", i, j);
        return Error::DelegateInvalidCompatibility;
      }
      delegate->in_place_inputs.push_back(op.inputs[j].index);
    }

    builder.Finish(DSPCOMMAND::CreateCommand(
        builder,
        (int32_t)op.type,
        builder.CreateVector(inputs),
        builder.CreateVector(outputs),
        builder.CreateVector(params)));
    const size_t size = builder.GetSize();

    // Resolve the patch slot now: the descriptor is written once, but the
    // command group offset it lands at is only known here.
    if (op.patch_param != kNoOpPatch) {
      if (op.patch_param >= op.n_params || op.patch_input >= op.n_inputs) {
        ET_LOG(
            Error,
            "hexagon: op %u patch (%u, %u) out of range",
            i,
            op.patch_param,
            op.patch_input);
        return Error::DelegateInvalidCompatibility;
      }
      const auto* command =
          flatbuffers::GetRoot<DSPCOMMAND::Command>(builder.GetBufferPointer());
      const size_t delta =
          reinterpret_cast<const uint8_t*>(command->params()->data()) -
          builder.GetBufferPointer() + op.patch_param * sizeof(int32_t);
      const HexagonTensorRef& src = op.inputs[op.patch_input];
      // kAbsent would name a operand with no storage, so it cannot supply a value.
      if (src.space > static_cast<uint32_t>(HexagonTensorSpace::kActivation)) {
        ET_LOG(Error, "hexagon: op %u patch input has space %u", i, src.space);
        return Error::DelegateInvalidCompatibility;
      }
      const auto space = static_cast<HexagonTensorSpace>(src.space);
      // The attention extent (max_kv_len) is not the token count but the
      // position plus it: ResizeDynamicDelegate only sees the tokens in this
      // call, so a decode step at position p would keep asking for two rows.
      const uint32_t extent_offset =
          (op.type == kFlashAttnOp && op.patch_param == 1 && op.n_params > 8) ? 7u
                                                                             : 0u;
      delegate->patches.push_back(
          {static_cast<uint8_t*>(delegate->resident.ptr) + command_cursor + delta,
           static_cast<const uint8_t*>(BlockOf(*delegate, space).ptr) +
               SectionBase(*delegate, space) + src.offset,
           src,
           op.patch_scale,
           extent_offset});
    }

    if (command_cursor + size >
        delegate->commands.offset + delegate->commands.size) {
      ET_LOG(
          Error,
          "hexagon: command budget of %zu bytes exceeded",
          delegate->commands.size);
      return Error::Internal;
    }
    std::memcpy(
        static_cast<uint8_t*>(delegate->resident.ptr) + command_cursor,
        builder.GetBufferPointer(),
        size);

    if (processed->size() > trailer_offset) {
      const auto* trailer = reinterpret_cast<const HexagonDynamicTrailer*>(
          blob + trailer_offset);
      const size_t trailer_bytes = trailer->version >= 3
          ? sizeof(HexagonDynamicTrailerV3) : sizeof(HexagonDynamicTrailer);
      const auto* records = reinterpret_cast<const HexagonDynamicPatch*>(
          blob + trailer_offset + trailer_bytes);
      for (uint32_t p = 0; p < trailer->n_patches; p++) {
        if (records[p].op_index != static_cast<int32_t>(i)) {
          continue;
        }
        if (records[p].param_index >= static_cast<int32_t>(op.n_params)) {
          ET_LOG(Error, "hexagon: dynamic patch param out of range");
          return Error::DelegateInvalidCompatibility;
        }
        const auto* command = flatbuffers::GetRoot<DSPCOMMAND::Command>(
            static_cast<uint8_t*>(delegate->resident.ptr) + command_cursor);
        const size_t delta =
            reinterpret_cast<const uint8_t*>(command->params()->data()) -
            static_cast<uint8_t*>(delegate->resident.ptr) - command_cursor +
            static_cast<size_t>(records[p].param_index) * sizeof(int32_t);
        delegate->dynamic_patches.push_back(
            {static_cast<uint8_t*>(delegate->resident.ptr) + command_cursor +
                 delta,
             records[p].scale,
             records[p].add});
      }
    }

    // The DSP reads entries from group_ptr + 8, so they start at int index 2,
    // not 1; anywhere else shifts every (fd, offset) pair by four bytes.
    group[2 + i * 3 + 0] = delegate->resident.fd;
    group[2 + i * 3 + 1] = (int32_t)(command_cursor + delegate->resident.bias);
    // size <= 0 tells the DSP to invalidate the descriptor before reading it,
    // which is right: the host wrote it once and never touches it again.
    group[2 + i * 3 + 2] = 0;

    command_cursor = AlignUp(command_cursor + size, kHexagonAlignment);
  }

  stamp("commands_built");

  // The sync group names everything the DSP invalidates on the way in and
  // flushes on the way out.
  {
    // Reset before the tensors are built, for the same reason as the command
    // loop: CreateSyncGroup takes offsets into this builder, and a later Clear
    // leaves them dangling, which silently corrupts the cache maintenance the
    // DSP derives from this group.
    builder.Clear();

    std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> sync_in;
    std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> sync_out;
    // Only real buffers need invalidating or flushing; an absent operand has
    // nothing to reconcile.
    for (uint32_t i = 0; i < header->n_ops; i++) {
      for (uint32_t j = 0; j < ops[i].n_inputs; j++) {
        if (!IsAbsent(ops[i].inputs[j])) {
          sync_in.push_back(MakeTensor(builder, *delegate, ops[i].inputs[j]));
        }
      }
      for (uint32_t j = 0; j < ops[i].n_outputs; j++) {
        if (!IsAbsent(ops[i].outputs[j])) {
          sync_out.push_back(MakeTensor(builder, *delegate, ops[i].outputs[j]));
        }
      }
    }

    builder.Finish(DSPCOMMAND::CreateSyncGroup(
        builder, builder.CreateVector(sync_in), builder.CreateVector(sync_out)));

    if (builder.GetSize() > delegate->sync.size) {
      ET_LOG(
          Error,
          "hexagon: sync group needs %zu bytes, budget is %zu",
          static_cast<size_t>(builder.GetSize()),
          delegate->sync.size);
      return Error::Internal;
    }
    delegate->sync.size = builder.GetSize();
    std::memcpy(
        static_cast<uint8_t*>(delegate->resident.ptr) + delegate->sync.offset,
        builder.GetBufferPointer(),
        delegate->sync.size);
  }

  stamp("sync_built");
  delegate->n_ops = header->n_ops;
  delegate->activations_bytes = header->activations_bytes;

  if (config.trace) {
    std::fprintf(
        stderr,
        "[hexagon] init d%d: ops=%u act=%u types=",
        delegate->index,
        delegate->n_ops,
        delegate->activations_bytes);
    for (uint32_t i = 0; i < delegate->n_ops; i++) {
      std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", (int)ops[i].type);
    }
    std::fprintf(stderr, "\n");
  }

  // The command group, descriptors, sync group and weights are all host-written
  // and never touched again, so one flush at init covers them.
  ET_CHECK_OK_OR_RETURN_ERROR(
      delegate->driver.Flush(delegate->resident.ptr, delegate->resident.bytes));
  stamp("resident_flush");

  ET_LOG(
      Info,
      "hexagon: ready, %u ops, %u in, %u out, resident %zu bytes",
      delegate->n_ops,
      header->n_inputs,
      header->n_outputs,
      delegate->resident.bytes);
  stamp("init_done");

  return delegate;
}

// Host-side accounting: the DSP reports kernel time, so whatever else the
// round trip costs is what happens inside this function.
double AcctNowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

Error HexagonBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  (void)context;

  auto* delegate = static_cast<HexagonDelegate*>(handle);
  if (delegate == nullptr) {
    return Error::InvalidArgument;
  }
  if (args.size() != delegate->inputs.size() + delegate->outputs.size()) {
    ET_LOG(
        Error,
        "hexagon: expected %zu args, got %zu",
        delegate->inputs.size() + delegate->outputs.size(),
        args.size());
    return Error::InvalidArgument;
  }

  auto* const resident = static_cast<uint8_t*>(delegate->resident.ptr);
  uint8_t* scratch = static_cast<uint8_t*>(delegate->scratch.ptr);

  const bool acct = EnvInt("HEXAGON_ACCT", 0) != 0;
  const bool phase = Trace().phase;
  const double t_begin = acct ? AcctNowMs() : 0.0;
  if (phase) {
    std::fprintf(
        stderr, "[phase] t=%.1f exec d%d enter\n", PhaseNowMs(), delegate->index);
  }
  size_t input_bytes = 0;
  int64_t dynamic_length = 0;
  if (delegate->dynamic_max_length != 0) {
    if (delegate->dynamic_input >= delegate->inputs.size() ||
        !args[delegate->dynamic_input]->isTensor()) {
      ET_LOG(Error, "hexagon: dynamic sequence input is not a tensor");
      return Error::InvalidArgument;
    }
    const auto& sequence = args[delegate->dynamic_input]->toTensor();
    if (delegate->dynamic_axis >= sequence.dim() ||
        sequence.sizes()[delegate->dynamic_axis] <= 0 ||
        static_cast<uint32_t>(sequence.sizes()[delegate->dynamic_axis]) >
            delegate->dynamic_max_length) {
      ET_LOG(Error, "hexagon: dynamic sequence length exceeds max");
      return Error::InvalidArgument;
    }
    dynamic_length = sequence.sizes()[delegate->dynamic_axis];
    if (!delegate->dynamic_layouts.empty()) {
      Error resize_status = ResizeDynamicDelegate(delegate, dynamic_length);
      if (resize_status != Error::Ok) {
        return resize_status;
      }
      scratch = static_cast<uint8_t*>(delegate->scratch.ptr);
    }
    for (size_t i = 0; i < delegate->outputs.size(); i++) {
      if (!args[delegate->inputs.size() + i]->isTensor()) {
        continue;
      }
      auto& value = args[delegate->inputs.size() + i]->toTensor();
      const size_t target_bytes = delegate->outputs[i].size;
      const size_t itemsize =
          value.scalar_type() == runtime::etensor::ScalarType::Float ? 4 : 2;
      const size_t logical_target =
          value.scalar_type() == runtime::etensor::ScalarType::Float
          && target_bytes * 2 == value.nbytes() ? target_bytes * 2
                                                : target_bytes;
      if (value.nbytes() != logical_target && value.dim() != 0) {
        std::vector<executorch::aten::SizesType> current_sizes(
            value.sizes().begin(), value.sizes().end());
        bool resized = false;
        for (size_t axis = 0; axis < current_sizes.size(); axis++) {
          if (current_sizes[axis] == dynamic_length) {
            continue;
          }
          const int64_t old = current_sizes[axis];
          current_sizes[axis] = static_cast<executorch::aten::SizesType>(
              dynamic_length);
          size_t candidate_numel = 1;
          for (auto dim : current_sizes) {
            candidate_numel *= static_cast<size_t>(dim);
          }
          if (candidate_numel * itemsize == logical_target) {
            Error resize_error = resize_tensor(
                value,
                runtime::ArrayRef<executorch::aten::SizesType>(
                    current_sizes.data(), current_sizes.size()));
            if (resize_error != Error::Ok) {
              ET_LOG(Error, "hexagon: dynamic output resize failed: %u",
                  static_cast<uint32_t>(resize_error));
              return resize_error;
            }
            resized = true;
            break;
          }
          current_sizes[axis] = old;
        }
        if (!resized) {
          ET_LOG(Error, "hexagon: cannot infer dynamic output shape");
          return Error::InvalidArgument;
        }
      }
    }
  }

  for (size_t i = 0; i < delegate->inputs.size(); i++) {
    // A non-tensor input still has an arena slot: a graph hands the attention
    // its start position as a plain int, and the command reads it back out of
    // that slot. Leaving it untouched pins the position at zero, so every
    // decode step attends at the start and the cache never advances.
    if (!args[i]->isTensor()) {
      if (args[i]->isInt() && delegate->inputs[i].size >= sizeof(int64_t)) {
        const int64_t value = args[i]->toInt();
        std::memcpy(
            scratch + delegate->inputs[i].offset, &value, sizeof(value));
      }
      continue;
    }
    const auto& tensor = args[i]->toTensor();
    const size_t nbytes = tensor.nbytes();

    // A tensor no command reads gets no slot either. The blob carries an input's
    // offset and size inside its operand references and nowhere else, so an
    // input the emitter folded away -- addmm with beta=0 drops its bias -- is
    // one the blob does not describe: there is no offset to copy to and no size
    // to respect. Skipping is the only answer, and it is not an error: the value
    // could never reach the graph anyway. A slot that exists but is too small is
    // a different case and still fails below.
    if (delegate->inputs[i].size == 0) {
      ET_LOG(
          Info,
          "hexagon: input %zu is %zu bytes, no slot in the blob (no command reads it); skipped",
          i,
          nbytes);
      continue;
    }
    uint8_t* const dst = scratch + delegate->inputs[i].offset;

    // Every tensor in the command is described as 2 bytes per element, because
    // the kernels are fp16. A fp32 input whose slot is exactly half its size is
    // narrowed on the way in instead of being rejected, so a region the graph
    // computes in fp32 can still be handed to an fp16 kernel.
    const size_t elements = static_cast<size_t>(tensor.numel());
    if (tensor.scalar_type() == runtime::etensor::ScalarType::Float &&
        delegate->inputs[i].size == elements * 2) {
      const float* const from =
          static_cast<const float*>(tensor.const_data_ptr());
      narrow_fp32_to_fp16(from, reinterpret_cast<uint16_t*>(dst), elements);
      continue;
    }

    if (nbytes > delegate->inputs[i].size) {
      ET_LOG(
          Error,
          "hexagon: input %zu is %zu bytes, slot holds %zu",
          i,
          nbytes,
          delegate->inputs[i].size);
      return Error::InvalidArgument;
    }
    std::memcpy(dst, tensor.const_data_ptr(), nbytes);
    input_bytes += nbytes;
  }

  const double t_input = acct ? AcctNowMs() : 0.0;

  if (delegate->dynamic_max_length != 0) {
    const int64_t length = dynamic_length;
    for (const auto& patch : delegate->dynamic_patches) {
      const int64_t value = length * patch.scale + patch.add;
      if (value < 0 || value > INT32_MAX) {
      ET_LOG(Error, "hexagon: dynamic patch value overflows int32");
      return Error::InvalidArgument;
      }
      const int32_t narrowed = static_cast<int32_t>(value);
      std::memcpy(patch.slot, &narrowed, sizeof(narrowed));
    }
  }

  // The inputs are in the arena by now, so a patched param can read back what
  // the caller just handed us. This has to precede the flush below.
  for (const auto& patch : delegate->patches) {
    int32_t value = 0;
    std::memcpy(&value, patch.source, sizeof(value));
    // Widened before scaling: a position times a cache row is still a byte
    // offset, and it overflows int32 well before either factor does.
    value = static_cast<int32_t>(static_cast<int64_t>(value) * patch.scale);
    std::memcpy(patch.slot, &value, sizeof(value));
    if (patch.extent_offset != 0 && delegate->dynamic_max_length != 0) {
      // Inclusive extent: the token at this position plus the ones handed in now.
      const int32_t extent =
          static_cast<int32_t>(value + dynamic_length + 1);
      std::memcpy(
          patch.slot + patch.extent_offset * sizeof(int32_t),
          &extent,
          sizeof(extent));
    }
  }

  // The inputs are one contiguous section, so one flush covers them all. The
  // patched slots live in the command section, so flush that as well.
  ET_CHECK_OK_OR_RETURN_ERROR(delegate->driver.Flush(
      scratch + delegate->input_section.offset, delegate->input_section.size));
  if (!delegate->patches.empty() || !delegate->dynamic_patches.empty()) {
    ET_CHECK_OK_OR_RETURN_ERROR(delegate->driver.Flush(
        resident + delegate->commands.offset, delegate->commands.size));
  }
  const double t_flush = acct ? AcctNowMs() : 0.0;

  const TraceConfig& config = Trace();
  static int next_execute = 0;
  const int exec_index = next_execute++;
  const bool traced = config.trace && (config.delegate < 0 || config.delegate == exec_index);
  uint32_t first = 0;
  uint32_t count = delegate->n_ops;
  if (config.delegate < 0 || config.delegate == exec_index) {
    first = (uint32_t)std::min<int>(std::max(config.start, 0), (int)delegate->n_ops);
    count = delegate->n_ops - first;
    if (config.limit > 0) {
      count = (uint32_t)std::min<int>(config.limit, (int)count);
    }
  }
  const int32_t group_offset = (int32_t)(
      delegate->group.offset + delegate->resident.bias + first * 3 * (int)sizeof(int32_t));

  std::fprintf(
      stderr,
      "[hexagon] enter d%d: ops=%u act=%u scratch=%zu first=%u count=%u\n",
      exec_index, delegate->n_ops, delegate->activations_bytes,
      delegate->scratch.bytes, first, count);
  if (traced) {
    PrintCommands(*delegate);
  }

  const double t_call0 = acct ? AcctNowMs() : 0.0;
  if (phase) {
    std::fprintf(
        stderr, "[phase] t=%.1f exec d%d pre_call\n", PhaseNowMs(), delegate->index);
  }
  Error group_error = Error::Ok;
  {
    ProbeWatchdog watchdog(delegate, traced);
    if (traced && delegate->probe.ptr != nullptr) {
      group_error = delegate->driver.ExecuteCommandGroupTraced(
          delegate->resident.fd,
          group_offset,
          count,
          delegate->resident.fd,
          (int)(delegate->sync.offset + delegate->resident.bias),
          (int)delegate->sync.size,
          delegate->probe.fd,
          (int)delegate->probe.bias,
          kProbeBytes);
    } else {
      group_error = delegate->driver.ExecuteCommandGroup(
          delegate->resident.fd,
          group_offset,
          count,
          delegate->resident.fd,
          (int)(delegate->sync.offset + delegate->resident.bias),
          (int)delegate->sync.size);
    }
  }

  std::fprintf(
      stderr,
      "[hexagon] exit d%d: %s\n",
      exec_index,
      group_error == Error::Ok ? "ok" : "failed");
  double t_call1 = 0.0;
  if (acct) {
    t_call1 = AcctNowMs();
    std::fprintf(
        stderr, "[phase] t=%.1f exec d%d post_call\n", PhaseNowMs(), delegate->index);
    std::fprintf(
        stderr,
        "[hexagon] acct d%d ops=%u in=%zuB in_ms=%.3f flush_ms=%.3f "
        "call_ms=%.3f total_ms=%.3f\n",
        exec_index,
        delegate->n_ops,
        input_bytes,
        t_input - t_begin,
        t_flush - t_input,
        t_call1 - t_call0,
        t_call1 - t_begin);
  }
  if (traced) {
    PrintProbe(*delegate);
    PrintOpTimes(*delegate);
    PrintStages(*delegate);
  }
  if (config.stop_after == exec_index) {
    std::fprintf(stderr, "[hexagon] stopping after d%d by request\n", exec_index);
    std::fflush(nullptr);
    std::exit(0);
  }
  if (group_error != Error::Ok) {
    // A failed group leaves the DSP side of this session unusable, and the
    // caller may abort before destroy() runs, so the session is released here:
    // the skel's global backend keeps its VTCM context until htp_ops_close.
    // Close() is idempotent, so the later destroy() on the normal path is fine.
    // This invoke can block, so it relies on the driver arming an RPC timeout.
    delegate->driver.Close();
  }
  ET_CHECK_OK_OR_RETURN_ERROR(group_error);

  // A subgraph that advances a KV cache in place has to hand the updated buffer
  // back, or the next execute() copies the old one in and every step after the
  // first attends over stale keys.
  for (const uint32_t index : delegate->in_place_inputs) {
    std::memcpy(
        args[index]->toTensor().mutable_data_ptr(),
        scratch + delegate->inputs[index].offset,
        delegate->inputs[index].size);
  }
  Error invalidate_error = delegate->driver.Invalidate(
      scratch + delegate->output_section.offset, delegate->output_section.size);
  if (invalidate_error != Error::Ok) {
    ET_LOG(Error, "hexagon: output invalidate failed: %u",
        static_cast<uint32_t>(invalidate_error));
    return invalidate_error;
  }

  for (size_t i = 0; i < delegate->outputs.size(); i++) {
    const auto& out = delegate->outputs[i];
    auto* arg = args[delegate->inputs.size() + i];
    if (!arg->isTensor()) {
      continue;
    }
    auto& tensor = arg->toTensor();
    if (dynamic_length > 0 && tensor.dim() > delegate->dynamic_axis &&
        tensor.sizes()[delegate->dynamic_axis] ==
            static_cast<int64_t>(delegate->dynamic_max_length)) {
      std::vector<executorch::aten::SizesType> current_sizes(
          tensor.sizes().begin(), tensor.sizes().end());
      current_sizes[delegate->dynamic_axis] =
          static_cast<executorch::aten::SizesType>(dynamic_length);
      ET_CHECK_OK_OR_RETURN_ERROR(resize_tensor(
          tensor,
          runtime::ArrayRef<executorch::aten::SizesType>(
              current_sizes.data(), current_sizes.size())));
    }
    // The mirror of the narrow on the way in: a subgraph whose declared result
    // is fp32 still writes fp16, because that is what the kernels produce. Its
    // slot is half the caller's buffer, and the bytes have to be widened back
    // or the caller reads fp16 patterns as floats. attention is the op that
    // lands here today.
    const size_t elements = static_cast<size_t>(tensor.numel());
    if (tensor.scalar_type() == runtime::etensor::ScalarType::Float &&
        out.size == elements * 2) {
      const uint16_t* const from =
          reinterpret_cast<const uint16_t*>(scratch + out.offset);
      float* const to = static_cast<float*>(tensor.mutable_data_ptr());
      for (size_t element = 0; element < elements; element++) {
        to[element] = half_bits_to_float(from[element]);
      }
      continue;
    }

    if (tensor.nbytes() > out.size) {
      ET_LOG(
          Error,
          "hexagon: output %zu is %zu bytes, slot holds %zu",
          i,
          tensor.nbytes(),
          out.size);
      return Error::InvalidArgument;
    }
    std::memcpy(
        tensor.mutable_data_ptr(), scratch + out.offset, tensor.nbytes());
  }

  if (acct) {
    const double t_end = AcctNowMs();
    std::fprintf(
        stderr,
        "[hexagon] acct2 d%d outcopy_ms=%.3f span_ms=%.3f\n",
        exec_index,
        t_end - t_call1,
        t_end - t_begin);
  }
  if (phase) {
    std::fprintf(
        stderr, "[phase] t=%.1f exec d%d end\n", PhaseNowMs(), delegate->index);
  }

  return Error::Ok;
}

void HexagonBackend::destroy(DelegateHandle* handle) const {
  auto* delegate = static_cast<HexagonDelegate*>(handle);
  if (delegate == nullptr) {
    return;
  }
  if (delegate->probe.ptr != nullptr) {
    delegate->driver.Free(delegate->probe.ptr);
    delegate->probe.ptr = nullptr;
  }
  if (delegate->resident.ptr != nullptr) {
    delegate->driver.Free(delegate->resident.ptr);
    delegate->resident.ptr = nullptr;
  }
  // The delegate itself lives in the runtime allocator and is reclaimed with
  // the program, and the scratch block belongs to the pool, so the resident
  // block is the only thing released here.
  //
  // Closing is what releases the DSP side: the allocator only hands out raw
  // bytes, so the delegate's driver is never constructed and ~HexagonDriver()
  // never runs, which would otherwise leave the FastRPC session and the skel's
  // global backend (init_backend's reference count, the VTCM context) held.
  // Close() is idempotent, so the Free() calls above and the destructor that
  // does not run are both still correct.
  delegate->driver.Close();
}

namespace {
HexagonBackend g_hexagon_backend;
} // namespace

} // namespace executorch::backends::hexagon

// NOLINTNEXTLINE
static const executorch::runtime::Backend kHexagonBackendRegistration{
    "HexagonBackend",
    &executorch::backends::hexagon::g_hexagon_backend};
// NOLINTNEXTLINE
static const auto kHexagonBackendRegistered =
    executorch::runtime::register_backend(kHexagonBackendRegistration);
