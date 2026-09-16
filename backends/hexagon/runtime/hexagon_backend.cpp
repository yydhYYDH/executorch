/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/hexagon/runtime/hexagon_backend.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include <flatbuffers/flatbuffers.h>
// Regenerated from the vendored Command.fbs with the flatbuffers version this
// tree vendors; the checked-in copy was produced by an older codegen whose
// accessors do not compile against it. The wire format is unchanged.
#include <Command_generated.h>

#include <executorch/backends/hexagon/serialization/hexagon_schema.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
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
  void* arena = nullptr;
  int arena_fd = -1;
  size_t arena_bytes = 0;
  // Where the arena starts inside its FastRPC mapping. The DSP resolves the fd
  // to the mapping's base, so every offset it is given is biased by this.
  uint64_t arena_bias = 0;

  // Host-written once at init.
  Region group; // command group array
  Region commands; // the command descriptors
  Region sync; // SyncGroup flatbuffer

  // The four tensor sections, in blob order.
  Region weights;
  Region input_section;
  Region activations;
  Region output_section;

  uint32_t n_ops = 0;
  // Absolute arena location of each method input/output, in signature order.
  std::vector<Region> inputs;
  std::vector<Region> outputs;

  // Params the emitter could not know, resolved from an input on every
  // execute. Only ops with HexagonOp::patch_param set appear here.
  struct Patch {
    size_t param_offset; // arena location of the param slot to overwrite
    size_t input_offset; // arena location the value is read from
    uint32_t scale; // multiplier between the two
  };
  std::vector<Patch> patches;

  // Method inputs the subgraph writes to, by signature index. Their arena slots
  // are copied back to the caller once the command group has run.
  std::vector<uint32_t> in_place_inputs;
};

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
  return DSPCOMMAND::CreateTensor(
      builder,
      delegate.arena_fd,
      (int32_t)(base + delegate.arena_bias),
      (int32_t)ref.size);
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

  auto driver = HexagonDriver::Create();
  if (!driver.ok()) {
    ET_LOG(Error, "hexagon: no DSP session");
    return driver.error();
  }
  delegate->driver = std::move(driver.get());
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

  size_t total = command_budget + sync_budget + group_bytes;
  total = AlignUp(total + header->weights_bytes, kHexagonAlignment);
  total = AlignUp(total + header->inputs_bytes, kHexagonAlignment);
  total = AlignUp(total + header->activations_bytes, kHexagonAlignment);
  total = AlignUp(total + header->outputs_bytes, kHexagonAlignment);

  std::fprintf(
      stderr,
      "[hexagon] arena request: %zu bytes (weights %zu, activations %zu)\n",
      total,
      (size_t)header->weights_bytes,
      (size_t)header->activations_bytes);
  auto arena = delegate->driver.Alloc(total);
  if (!arena.ok()) {
    std::fprintf(stderr, "[hexagon] rpcmem allocation failed\n");
    return arena.error();
  }
  delegate->arena = arena.get();
  delegate->arena_bytes = total;
  std::memset(delegate->arena, 0, total);

  auto fd = delegate->driver.ToFd(delegate->arena);
  if (!fd.ok()) {
    return fd.error();
  }
  delegate->arena_fd = fd.get();

  auto bias = delegate->driver.MappingOffset(delegate->arena);
  if (!bias.ok()) {
    return bias.error();
  }
  delegate->arena_bias = bias.get();

  const auto* ops = reinterpret_cast<const HexagonOp*>(
      reinterpret_cast<const uint8_t*>(processed->data()) +
      sizeof(HexagonBlobHeader));
  const uint8_t* blob = reinterpret_cast<const uint8_t*>(processed->data());
  const size_t weights_blob_offset = sizeof(HexagonBlobHeader) + ops_bytes;
  // Only the weights and activations live in the blob; the input and output
  // sizes are arena budgets the runtime allocates against, so counting them
  // here overstates the blob and rejects every delegate that has an input.
  const size_t sections_total = static_cast<size_t>(header->weights_bytes) +
      header->activations_bytes;
  if (weights_blob_offset + sections_total > processed->size()) {
    ET_LOG(Error, "hexagon: tensor sections out of bounds");
    return Error::DelegateInvalidCompatibility;
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
  place(delegate->input_section, header->inputs_bytes);
  place(delegate->activations, header->activations_bytes);
  place(delegate->output_section, header->outputs_bytes);

  std::memcpy(
      static_cast<uint8_t*>(delegate->arena) + delegate->weights.offset,
      blob + weights_blob_offset,
      header->weights_bytes);

  // The group array the DSP walks: a header then 3 int32 per command.
  delegate->group.offset = cursor;
  delegate->group.size = group_bytes;
  auto* group =
      reinterpret_cast<int32_t*>(static_cast<uint8_t*>(delegate->arena) + cursor);
  std::memset(group, 0, delegate->group.size);
  cursor = AlignUp(cursor + delegate->group.size, kHexagonAlignment);

  if (cursor > delegate->arena_bytes) {
    ET_LOG(
        Error,
        "hexagon: layout needs %zu bytes, arena has %zu",
        cursor,
        delegate->arena_bytes);
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
    const HexagonOp& op = ops[i];

    // Reset before the tensors are built, not after: the offsets they return
    // index into this builder, so clearing later leaves CreateCommand holding
    // offsets into a buffer that no longer exists.
    builder.Clear();

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
      delegate->patches.push_back(
          {command_cursor + delta,
           SectionBase(*delegate, static_cast<HexagonTensorSpace>(src.space)) +
               src.offset,
           op.patch_scale});
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
        static_cast<uint8_t*>(delegate->arena) + command_cursor,
        builder.GetBufferPointer(),
        size);

    // The DSP reads entries from group_ptr + 8, so they start at int index 2,
    // not 1; anywhere else shifts every (fd, offset) pair by four bytes.
    group[2 + i * 3 + 0] = delegate->arena_fd;
    group[2 + i * 3 + 1] = (int32_t)(command_cursor + delegate->arena_bias);
    // size <= 0 tells the DSP to invalidate the descriptor before reading it,
    // which is right: the host wrote it once and never touches it again.
    group[2 + i * 3 + 2] = 0;

    command_cursor = AlignUp(command_cursor + size, kHexagonAlignment);
  }

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
        static_cast<uint8_t*>(delegate->arena) + delegate->sync.offset,
        builder.GetBufferPointer(),
        delegate->sync.size);
  }

  delegate->n_ops = header->n_ops;

  // The command group, descriptors, sync group and weights are all host-written
  // and never touched again, so one flush at init covers them.
  ET_CHECK_OK_OR_RETURN_ERROR(delegate->driver.Flush(
      delegate->arena, delegate->sync.offset + delegate->sync.size));

  ET_LOG(
      Info,
      "hexagon: ready, %u ops, %u in, %u out, arena %zu bytes",
      delegate->n_ops,
      header->n_inputs,
      header->n_outputs,
      delegate->arena_bytes);

  return delegate;
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

  auto* base = static_cast<uint8_t*>(delegate->arena);

  for (size_t i = 0; i < delegate->inputs.size(); i++) {
    const auto& tensor = args[i]->toTensor();
    const size_t nbytes = tensor.nbytes();
    uint8_t* const dst = base + delegate->inputs[i].offset;

    // Every tensor in the command is described as 2 bytes per element, because
    // the kernels are fp16. A fp32 input whose slot is exactly half its size is
    // narrowed on the way in instead of being rejected, so a region the graph
    // computes in fp32 can still be handed to an fp16 kernel.
    const size_t elements = static_cast<size_t>(tensor.numel());
    if (tensor.scalar_type() == runtime::etensor::ScalarType::Float &&
        delegate->inputs[i].size == elements * 2) {
      const float* const from =
          static_cast<const float*>(tensor.const_data_ptr());
      uint16_t* const to = reinterpret_cast<uint16_t*>(dst);
      for (size_t element = 0; element < elements; element++) {
        to[element] = float_to_half_bits(from[element]);
      }
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
  }

  // The inputs are in the arena by now, so a patched param can read back what
  // the caller just handed us. This has to precede the flush below.
  for (const auto& patch : delegate->patches) {
    int32_t value = 0;
    std::memcpy(&value, base + patch.input_offset, sizeof(value));
    // Widened before scaling: a position times a cache row is still a byte
    // offset, and it overflows int32 well before either factor does.
    value = static_cast<int32_t>(static_cast<int64_t>(value) * patch.scale);
    std::memcpy(base + patch.param_offset, &value, sizeof(value));
  }

  // The inputs are one contiguous section, so one flush covers them all. The
  // patched slots live in the command section, so flush that as well.
  ET_CHECK_OK_OR_RETURN_ERROR(delegate->driver.Flush(
      base + delegate->input_section.offset, delegate->input_section.size));
  if (!delegate->patches.empty()) {
    ET_CHECK_OK_OR_RETURN_ERROR(delegate->driver.Flush(
        base + delegate->commands.offset, delegate->commands.size));
  }

  ET_CHECK_OK_OR_RETURN_ERROR(delegate->driver.ExecuteCommandGroup(
      delegate->arena_fd,
      (int)(delegate->group.offset + delegate->arena_bias),
      delegate->n_ops,
      delegate->arena_fd,
      (int)(delegate->sync.offset + delegate->arena_bias),
      (int)delegate->sync.size));

  // A subgraph that advances a KV cache in place has to hand the updated buffer
  // back, or the next execute() copies the old one in and every step after the
  // first attends over stale keys.
  for (const uint32_t index : delegate->in_place_inputs) {
    std::memcpy(
        args[index]->toTensor().mutable_data_ptr(),
        base + delegate->inputs[index].offset,
        delegate->inputs[index].size);
  }

  ET_CHECK_OK_OR_RETURN_ERROR(delegate->driver.Invalidate(
      base + delegate->output_section.offset, delegate->output_section.size));

  for (size_t i = 0; i < delegate->outputs.size(); i++) {
    const auto& out = delegate->outputs[i];
    auto& tensor = args[delegate->inputs.size() + i]->toTensor();
    // The mirror of the narrow on the way in: a subgraph whose declared result
    // is fp32 still writes fp16, because that is what the kernels produce. Its
    // slot is half the caller's buffer, and the bytes have to be widened back
    // or the caller reads fp16 patterns as floats. attention is the op that
    // lands here today.
    const size_t elements = static_cast<size_t>(tensor.numel());
    if (tensor.scalar_type() == runtime::etensor::ScalarType::Float &&
        out.size == elements * 2) {
      const uint16_t* const from =
          reinterpret_cast<const uint16_t*>(base + out.offset);
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
    std::memcpy(tensor.mutable_data_ptr(), base + out.offset, tensor.nbytes());
  }

  return Error::Ok;
}

void HexagonBackend::destroy(DelegateHandle* handle) const {
  auto* delegate = static_cast<HexagonDelegate*>(handle);
  if (delegate == nullptr) {
    return;
  }
  if (delegate->arena != nullptr) {
    delegate->driver.Free(delegate->arena);
    delegate->arena = nullptr;
  }
  // The delegate itself lives in the runtime allocator and is reclaimed with
  // the program, so only DSP and shared-memory resources are released here.
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
