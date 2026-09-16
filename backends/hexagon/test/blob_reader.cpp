// Dumps a serialized Hexagon delegate blob as text.
//
// The AOT writer (serialization/blob.py) and this reader are two independent
// implementations of one packed on-disk format, and nothing at runtime compares
// them. A field one side forgets, or a struct whose layout changes on one side
// only, shows up as wrong numbers on a device rather than as an error.
//
// test_blob_roundtrip.py builds a blob that exercises every field, runs this
// over it, and compares the result against what it wrote. This file reads the
// structs from the same header the runtime includes, so it cannot drift from
// them.
//
// Deliberately dependency-free: the format is plain packed structs, so this
// needs no ExecuTorch, no Hexagon SDK, no FlatBuffers and no gtest, and the test
// therefore runs wherever a host C++ compiler exists.

// First, deliberately: a header that compiles only because something else was
// included before it is broken, and this ordering is what catches that.
#include "hexagon_schema.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

using namespace executorch::backends::hexagon;

namespace {

void print_sections(const uint8_t* sections, const HexagonBlobHeader* header) {
  for (const auto& named : {std::pair<const char*, uint32_t>{"weights", header->weights_bytes},
                            {"activations", header->activations_bytes}}) {
    std::printf("section %s %u ", named.first, named.second);
    for (uint32_t i = 0; i < named.second; i++) {
      std::printf("%02x", sections[i]);
    }
    std::printf("\n");
    sections += named.second;
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: blob_reader <blob>\n");
    return 2;
  }

  std::FILE* file = std::fopen(argv[1], "rb");
  if (file == nullptr) {
    std::perror("blob_reader: open");
    return 2;
  }
  std::fseek(file, 0, SEEK_END);
  const long length = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  std::vector<uint8_t> blob(static_cast<size_t>(length));
  const size_t read = std::fread(blob.data(), 1, blob.size(), file);
  std::fclose(file);
  if (read != blob.size()) {
    std::fprintf(stderr, "blob_reader: short read\n");
    return 2;
  }

  // Reported first so a layout mismatch is the first thing the test sees.
  std::printf("size HexagonBlobHeader %zu\n", sizeof(HexagonBlobHeader));
  std::printf("size HexagonTensorRef %zu\n", sizeof(HexagonTensorRef));
  std::printf("size HexagonOp %zu\n", sizeof(HexagonOp));
  // Anything but a multiple of the Python struct would mean the two writers
  // disagree about padding before a single field is compared.
  static_assert(sizeof(HexagonOp) == 480, "HexagonOp layout drifted");
  const auto offset_of = [](const char* field, size_t value) {
    std::printf("offsetof HexagonOp %s %zu\n", field, value);
  };
  offset_of("params", offsetof(HexagonOp, params));
  offset_of("patch_param", offsetof(HexagonOp, patch_param));
  offset_of("patch_input", offsetof(HexagonOp, patch_input));
  offset_of("patch_scale", offsetof(HexagonOp, patch_scale));
  offset_of("in_place", offsetof(HexagonOp, in_place));
  offset_of("inputs", offsetof(HexagonOp, inputs));
  offset_of("outputs", offsetof(HexagonOp, outputs));

  if (blob.size() < sizeof(HexagonBlobHeader)) {
    std::fprintf(stderr, "blob_reader: truncated header\n");
    return 1;
  }
  const auto* header = reinterpret_cast<const HexagonBlobHeader*>(blob.data());
  std::printf("header magic %u\n", header->magic);
  std::printf("header version %u\n", header->version);
  std::printf("header n_ops %u\n", header->n_ops);
  std::printf("header n_inputs %u\n", header->n_inputs);
  std::printf("header n_outputs %u\n", header->n_outputs);
  std::printf("header weights_bytes %u\n", header->weights_bytes);
  std::printf("header inputs_bytes %u\n", header->inputs_bytes);
  std::printf("header activations_bytes %u\n", header->activations_bytes);
  std::printf("header outputs_bytes %u\n", header->outputs_bytes);

  const size_t ops_bytes = sizeof(HexagonOp) * header->n_ops;
  if (blob.size() < sizeof(HexagonBlobHeader) + ops_bytes) {
    std::fprintf(stderr, "blob_reader: truncated ops\n");
    return 1;
  }
  const auto* ops =
      reinterpret_cast<const HexagonOp*>(blob.data() + sizeof(HexagonBlobHeader));

  for (uint32_t i = 0; i < header->n_ops; i++) {
    const HexagonOp& op = ops[i];
    std::printf("op %u type %d\n", i, static_cast<int>(op.type));
    std::printf("op %u n_inputs %u\n", i, op.n_inputs);
    std::printf("op %u n_outputs %u\n", i, op.n_outputs);
    std::printf("op %u n_params %u\n", i, op.n_params);
    // Every slot, not just the live ones: the padding is part of the format.
    for (uint32_t j = 0; j < kMaxOpParams; j++) {
      std::printf("op %u param %u %d\n", i, j, static_cast<int>(op.params[j]));
    }
    std::printf("op %u patch_param %u\n", i, op.patch_param);
    std::printf("op %u patch_input %u\n", i, op.patch_input);
    std::printf("op %u patch_scale %u\n", i, op.patch_scale);
    std::printf("op %u in_place %u\n", i, op.in_place);
    for (uint32_t j = 0; j < kMaxOpInputs; j++) {
      std::printf(
          "op %u input %u %u %u %llu %llu\n",
          i,
          j,
          op.inputs[j].space,
          op.inputs[j].index,
          static_cast<unsigned long long>(op.inputs[j].offset),
          static_cast<unsigned long long>(op.inputs[j].size));
    }
    for (uint32_t j = 0; j < kMaxOpOutputs; j++) {
      std::printf(
          "op %u output %u %u %u %llu %llu\n",
          i,
          j,
          op.outputs[j].space,
          op.outputs[j].index,
          static_cast<unsigned long long>(op.outputs[j].offset),
          static_cast<unsigned long long>(op.outputs[j].size));
    }
  }

  const size_t sections_at = sizeof(HexagonBlobHeader) + ops_bytes;
  if (sections_at + header->weights_bytes + header->activations_bytes > blob.size()) {
    std::fprintf(stderr, "blob_reader: truncated sections\n");
    return 1;
  }
  print_sections(blob.data() + sections_at, header);
  return 0;
}
