#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsp/ops.h"
#include "hexagon_schema.h"

extern "C" int htp_ops_pool2d_fp16(
    uint8_t *output, uint8_t *input, int32_t batch, int32_t ih, int32_t iw,
    int32_t oh, int32_t ow, int32_t c4, int32_t kernel_y, int32_t kernel_x,
    int32_t stride_y, int32_t stride_x, int32_t pad_y, int32_t pad_x,
    int32_t pad_type, int32_t count_type, int32_t pool_type);
extern "C" int htp_ops_raster_blit(
    uint8_t *dst, uint8_t **src, int src_number, uint8_t *region,
    int32_t region_count, int32_t bytes);
extern "C" int htp_ops_reduction(
    uint8_t *dst, const uint8_t *src, int32_t outside, int32_t reduce,
    int32_t op_type, int32_t inside, int32_t bytes);

using namespace executorch::backends::hexagon;

struct BlobFixture {
  const char *tag;
  const unsigned char *blob;
  unsigned long long blob_bytes;
  const unsigned char *inputs;
  unsigned int inputs_size;
  long long runtime_length;
};

#include "blob_fixture.h"

enum { kPool2d = 1, kRasterBlit = 3, kReduction = 29, kVectorBytes = 128 };
enum { kMaxArenaBytes = 4 * 1024 * 1024 };

struct Bases {
  uint64_t weights, inputs, activations, outputs;
};

static Bases g_bases;
static uint8_t *g_arena;
static const char *g_tag;

static uint64_t align_up(uint64_t value) {
  return (value + kHexagonAlignment - 1) & ~(uint64_t)(kHexagonAlignment - 1);
}

static Bases bases_of(const HexagonBlobHeader *header) {
  Bases bases;
  bases.weights = 0;
  bases.inputs = align_up(header->weights_bytes);
  bases.activations = align_up(bases.inputs + header->inputs_bytes);
  bases.outputs = align_up(bases.activations + header->activations_bytes);
  return bases;
}

static uint8_t *address(const HexagonBlobHeader *header, const HexagonTensorRef &ref) {
  uint64_t base;
  switch (ref.space) {
    case (uint32_t)HexagonTensorSpace::kWeights:
      base = g_bases.weights;
      break;
    case (uint32_t)HexagonTensorSpace::kInput:
      base = g_bases.inputs;
      break;
    case (uint32_t)HexagonTensorSpace::kActivation:
      base = g_bases.activations;
      break;
    default:
      base = g_bases.outputs;
      break;
  }
  (void)header;
  return g_arena + base + ref.offset;
}

static const HexagonTensorRef *slot(
    const HexagonOp *ops, uint32_t n_ops, uint32_t space, uint32_t index) {
  for (uint32_t i = 0; i < n_ops; ++i) {
    for (uint32_t k = 0; k < ops[i].n_inputs; ++k) {
      if (ops[i].inputs[k].space == space && ops[i].inputs[k].index == index) {
        return &ops[i].inputs[k];
      }
    }
    for (uint32_t k = 0; k < ops[i].n_outputs; ++k) {
      if (ops[i].outputs[k].space == space && ops[i].outputs[k].index == index) {
        return &ops[i].outputs[k];
      }
    }
  }
  return nullptr;
}

static void execute_op(const HexagonOp &op, const HexagonBlobHeader *header) {
  int32_t params[kMaxOpParams];
  memcpy(params, op.params, sizeof(params));
  if (op.type == kPool2d) {
    int ret = htp_ops_pool2d_fp16(
        address(header, op.outputs[0]), address(header, op.inputs[0]), params[0],
        params[1], params[2], params[3], params[4], params[5], params[6],
        params[7], params[8], params[9], params[10], params[11], params[12],
        params[13], params[14]);
    if (ret != 0) printf("%s pool returned %d\n", g_tag, ret);
    return;
  }
  if (op.type == kReduction) {
    int ret = htp_ops_reduction(
        address(header, op.outputs[0]), address(header, op.inputs[0]), params[0],
        params[1], params[2], params[3], params[4]);
    if (ret != 0) printf("%s reduction returned %d\n", g_tag, ret);
    return;
  }
  if (op.type == kRasterBlit) {
    uint8_t *sources[kMaxOpInputs];
    for (uint32_t i = 0; i < op.n_inputs; ++i) {
      sources[i] = address(header, op.inputs[i]);
    }
    int ret = htp_ops_raster_blit(
        address(header, op.outputs[0]), sources, (int)op.n_inputs,
        (uint8_t *)&params[3], params[0], params[1]);
    if (ret != 0) printf("%s blit returned %d\n", g_tag, ret);
    return;
  }
  printf("%s unsupported op %u\n", g_tag, op.type);
}

static int run_blob(const unsigned char *blob, unsigned long long blob_bytes,
                    uint8_t *arena, unsigned long long arena_bytes,
                    const unsigned char *input_data, unsigned long long input_bytes,
                    const char *tag) {
  const HexagonBlobHeader *header = (const HexagonBlobHeader *)blob;
  if (header->magic != kHexagonBlobMagic || header->version != kHexagonBlobVersion) {
    printf("BADBLOB\n");
    return 1;
  }
  if (blob_bytes < sizeof(*header) +
                       (unsigned long long)header->n_ops * sizeof(HexagonOp)) {
    printf("%s short blob\n", tag);
    return 1;
  }
  g_arena = arena;
  g_bases = bases_of(header);
  g_tag = tag;
  memset(arena, 0, arena_bytes);
  const HexagonOp *ops =
      (const HexagonOp *)(blob + sizeof(HexagonBlobHeader));
  const uint8_t *sections =
      blob + sizeof(HexagonBlobHeader) + header->n_ops * sizeof(HexagonOp);
  HexagonTensorRef weights{(uint32_t)HexagonTensorSpace::kWeights, 0, 0, 0};
  memcpy(address(header, weights), sections, header->weights_bytes);

  uint64_t at = 0;
  for (uint32_t index = 0; index < header->n_inputs; ++index) {
    const HexagonTensorRef *ref =
        slot(ops, header->n_ops, (uint32_t)HexagonTensorSpace::kInput, index);
    if (ref == nullptr || at + ref->size > input_bytes) {
      printf("%s bad input %u\n", tag, index);
      return 1;
    }
    memcpy(address(header, *ref), input_data + at, ref->size);
    at += ref->size;
  }
  for (uint32_t i = 0; i < header->n_ops; ++i) {
    execute_op(ops[i], header);
  }
  for (uint32_t index = 0; index < header->n_outputs; ++index) {
    const HexagonTensorRef *ref =
        slot(ops, header->n_ops, (uint32_t)HexagonTensorSpace::kOutput, index);
    if (ref == nullptr) {
      printf("%s missing output %u\n", tag, index);
      return 1;
    }
    const uint8_t *raw = address(header, *ref);
    printf("%s%u", tag, index);
    for (uint64_t b = 0; b + 1 < ref->size; b += 2) {
      printf(" %02x%02x", raw[b + 1], raw[b]);
    }
    printf("\n");
  }
  return 0;
}

int main(void) {
  static uint8_t arena[kMaxArenaBytes] __attribute__((aligned(kVectorBytes)));
  static_assert(__alignof__(arena) == kVectorBytes, "pool arena alignment");
  int status = 0;
  for (unsigned i = 0; i < kFixtureCount; ++i) {
    const BlobFixture &fixture = kFixtures[i];
    status |= run_blob(
        fixture.blob, fixture.blob_bytes, arena, sizeof(arena), fixture.inputs,
        fixture.inputs_size, fixture.tag);
  }
  return status;
}
