/* Executes a real blob's command stream on the DSP, inside hexagon-sim.
 *
 * The host interpreter in blob_interpreter.py runs the same bytes, so a
 * disagreement is either the host model of an op or its DSP counterpart, and
 * the blob both of them read is what the emitter produced.
 *
 * The op layout comes from the backend's own hexagon_schema.h rather than a
 * C-side transcription, and the section bases are derived here from the header
 * rather than passed in, so the alignment rule is checked rather than assumed.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hexagon_schema.h"

using namespace executorch::backends::hexagon;

/* One blob to run, from the generated blob_fixture.h. The struct is declared
 * here and the header included below it so the header can name the type. */
struct BlobFixture {
  const char *tag;
  const unsigned char *blob;
  const unsigned char *inputs;
  unsigned int inputs_size;
};

extern "C" int htp_ops_binary_elementwise(uint8_t *dst, uint8_t *src0, uint8_t *src1,
                                          int32_t outSize, int32_t in0Size,
                                          int32_t in1Size, int32_t opType,
                                          int32_t bytes, int32_t inputBytes,
                                          int32_t inputIsFloat, int32_t outputIsFloat,
                                          const int32_t *broadcastParams,
                                          int32_t broadcastParamCount);
extern "C" int htp_ops_softmax(uint8_t *dst, const uint8_t *src, int32_t outside,
                                int32_t channel, int32_t inside, int32_t bytes);
extern "C" int htp_ops_reduction(uint8_t *dst, const uint8_t *src, int32_t outside,
                                 int32_t reduce, int32_t inside, int32_t opType,
                                 int32_t bytes);
extern "C" int htp_ops_layer_norm(uint8_t *dst, uint8_t *src, uint8_t *gamma,
                                  uint8_t *beta, int32_t outterSize,
                                  int32_t innerSize, float epsilon,
                                  int32_t RMSNorm);
extern "C" int htp_ops_batch_matmul(uint8_t *dst, uint8_t *src0, uint8_t *src1,
                                    uint8_t *iter0, uint8_t *iter1,
                                    uint8_t *iter2, int32_t bytes,
                                    uint8_t *param);
extern "C" int htp_ops_raster_blit(uint8_t *dst, uint8_t **src, int src_number,
                                   uint8_t *region, int32_t regionCount,
                                   int32_t bytes);
extern "C" int htp_ops_unary(uint8_t *dst, uint8_t *src, int32_t size,
                             int32_t opType, int32_t bytes);
extern "C" int htp_ops_flash_attn(uint8_t *o, uint8_t *q, uint8_t *k, uint8_t *v,
                                  uint8_t *mask, uint8_t *workspace,
                                  uint8_t *pastK, uint8_t *pastV, int32_t qo_len,
                                  int32_t seq_current, int32_t seq_add,
                                  int32_t n_heads, int32_t n_kv_heads,
                                  int32_t head_dim, float scale,
                                  int32_t mask_stride, int32_t max_kv_len,
                                  int32_t value_c4);

enum {
  kRasterBlit = 3,
  kUnary = 4,
  kLayerNorm = 8,
  kFlashAttn = 18,
  kBinaryElementwise = 19,
  kSoftmax = 28,
  kReduction = 29,
  kBatchMatmul = 38,
};

#include "blob_fixture.h"

/* Section bases, relative to the arena. Derived from the header rather than
 * passed in, so this is an independent implementation of the layout rule the
 * host interpreter implements too. */
struct Bases {
  uint64_t weights, inputs, activations, outputs;
};

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

static bool absent(const HexagonTensorRef &ref) {
  return ref.space == (uint32_t)HexagonTensorSpace::kAbsent;
}

static uint8_t *g_arena = nullptr;
static Bases g_bases;

static uint8_t *address(const HexagonBlobHeader *header, const HexagonTensorRef &ref) {
  (void)header;
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
  return g_arena + base + ref.offset;
}

/* The ref the emitters used for one method input or output, the same search the
 * host interpreter does. */
static const HexagonTensorRef *slot(const HexagonOp *ops, uint32_t n_ops,
                                    uint32_t space, uint32_t index) {
  for (uint32_t i = 0; i < n_ops; ++i) {
    for (uint32_t k = 0; k < ops[i].n_inputs; ++k)
      if (ops[i].inputs[k].space == space && ops[i].inputs[k].index == index)
        return &ops[i].inputs[k];
    for (uint32_t k = 0; k < ops[i].n_outputs; ++k)
      if (ops[i].outputs[k].space == space && ops[i].outputs[k].index == index)
        return &ops[i].outputs[k];
  }
  return nullptr;
}

static void execute_op(const HexagonOp &op, const HexagonBlobHeader *header) {
  int32_t params[kMaxOpParams];
  memcpy(params, op.params, sizeof(params));
  if (op.patch_param != kNoOpPatch) {
    int32_t value;
    memcpy(&value, address(header, op.inputs[op.patch_input]), 4);
    params[op.patch_param] = value * (int32_t)op.patch_scale;
  }

  if (op.type == kUnary) {
    htp_ops_unary(address(header, op.outputs[0]), address(header, op.inputs[0]),
                  params[0], params[1], params[2]);
    return;
  }
  if (op.type == kBatchMatmul) {
    htp_ops_batch_matmul(address(header, op.outputs[0]),
                         address(header, op.inputs[0]),
                         address(header, op.inputs[1]), NULL, NULL, NULL,
                         params[0], (uint8_t *)&params[1]);
    return;
  }
  if (op.type == kSoftmax) {
    htp_ops_softmax(address(header, op.outputs[0]), address(header, op.inputs[0]),
                    params[0], params[1], params[2], params[3]);
    return;
  }
  if (op.type == kFlashAttn) {
    /* params[6] carries the scale as its float bits, the way the emitter packs
     * it. The two slots after the mask are the past keys and values: the cache
     * this op was handed, which the emitters pass rather than leaving empty,
     * because the kernel writes through them before it computes anything. */
    float scale;
    memcpy(&scale, &params[6], 4);
    htp_ops_flash_attn(
        address(header, op.outputs[0]), address(header, op.inputs[0]),
        address(header, op.inputs[1]), address(header, op.inputs[2]),
        absent(op.inputs[3]) ? nullptr : address(header, op.inputs[3]),
        address(header, op.outputs[1]),
        absent(op.inputs[4]) ? nullptr : address(header, op.inputs[4]),
        absent(op.inputs[5]) ? nullptr : address(header, op.inputs[5]),
        params[0], params[1], params[2], params[3], params[4], params[5], scale,
        params[7], params[8], params[11]);
    return;
  }
  if (op.type == kReduction) {
    htp_ops_reduction(address(header, op.outputs[0]), address(header, op.inputs[0]),
                      params[0], params[1], params[2], params[3], params[4]);
    return;
  }
  if (op.type == kLayerNorm) {
    float epsilon;
    memcpy(&epsilon, &params[2], 4);
    htp_ops_layer_norm(
        address(header, op.outputs[0]), address(header, op.inputs[0]),
        absent(op.inputs[1]) ? nullptr : address(header, op.inputs[1]),
        absent(op.inputs[2]) ? nullptr : address(header, op.inputs[2]),
        params[0], params[1], epsilon, params[3]);
    return;
  }
  if (op.type == kBinaryElementwise) {
    /* The broadcast tail is a rank, the output extents and one stride list per
     * operand, so its length follows from the rank. */
    /* The count is a fixed 25 and not a function of the rank: the kernel
     * rejects anything shorter and then quietly takes the non-broadcast path,
     * which reports success and writes nothing. The extents and the two stride
     * lists are each padded to eight words whether or not the rank needs them. */
    enum { kBroadcastParams = 25 };
    htp_ops_binary_elementwise(
        address(header, op.outputs[0]), address(header, op.inputs[0]),
        address(header, op.inputs[1]), params[0], params[1], params[2], params[3],
        params[4], params[5], params[6], params[7], &params[8], kBroadcastParams);
    return;
  }
  if (op.type == kRasterBlit) {
    uint8_t *sources[kMaxOpInputs];
    for (uint32_t i = 0; i < op.n_inputs; ++i)
      sources[i] = address(header, op.inputs[i]);
    htp_ops_raster_blit(address(header, op.outputs[0]), sources,
                        (int)op.n_inputs, (uint8_t *)&params[3], params[0],
                        params[1]);
    return;
  }
  printf("UNSUPPORTED %u\n", op.type);
}

/* Runs one blob against one arena and prints its method outputs. */
static int run_blob(const unsigned char *blob, uint8_t *arena, uint64_t arena_bytes,
                    const unsigned char *input_data, uint64_t input_bytes,
                    const char *tag) {
  const HexagonBlobHeader *header = (const HexagonBlobHeader *)blob;
  if (header->magic != kHexagonBlobMagic || header->version != kHexagonBlobVersion) {
    printf("BADBLOB\n");
    return 1;
  }
  g_arena = arena;
  g_bases = bases_of(header);
  memset(arena, 0, arena_bytes);

  const HexagonOp *ops = (const HexagonOp *)(blob + sizeof(HexagonBlobHeader));
  const uint8_t *sections =
      blob + sizeof(HexagonBlobHeader) + header->n_ops * sizeof(HexagonOp);

  /* The two sections the blob carries. */
  const HexagonTensorRef weights{(uint32_t)HexagonTensorSpace::kWeights, 0, 0, 0};
  const HexagonTensorRef activations{(uint32_t)HexagonTensorSpace::kActivation, 0, 0, 0};
  memcpy(address(header, weights), sections, header->weights_bytes);
  memcpy(address(header, activations), sections + header->weights_bytes,
         header->activations_bytes);

  /* The method inputs, from the fixture, in method-input order. */
  uint64_t at = 0;
  for (uint32_t index = 0; index < header->n_inputs; ++index) {
    const HexagonTensorRef *ref =
        slot(ops, header->n_ops, (uint32_t)HexagonTensorSpace::kInput, index);
    if (ref == nullptr || at + ref->size > input_bytes) {
      printf("BADINPUT %u\n", index);
      return 1;
    }
    memcpy(address(header, *ref), input_data + at, ref->size);
    at += ref->size;
  }

  for (uint32_t i = 0; i < header->n_ops; ++i) execute_op(ops[i], header);

  for (uint32_t index = 0; index < header->n_outputs; ++index) {
    const HexagonTensorRef *ref =
        slot(ops, header->n_ops, (uint32_t)HexagonTensorSpace::kOutput, index);
    if (ref == nullptr) {
      printf("%s no slot for output %u\n", tag, index);
      continue;
    }
    const uint8_t *raw = address(header, *ref);
    printf("%s%u", tag, index);
    /* Little-endian, so print the high byte first to show the fp16 value. */
    for (uint64_t b = 0; b + 1 < ref->size; b += 2)
      printf(" %02x%02x", raw[b + 1], raw[b]);
    printf("\n");
  }
  return 0;
}

int main(void) {
  /* One arena, reused: run_blob zeroes it. Sized for the largest fixture. */
  static uint8_t arena[kMaxArenaBytes];
  int status = 0;
  for (unsigned i = 0; i < kFixtureCount; ++i) {
    const BlobFixture &fixture = kFixtures[i];
    status |= run_blob(fixture.blob, arena, sizeof(arena), fixture.inputs,
                       fixture.inputs_size, fixture.tag);
  }
  return status;
}
