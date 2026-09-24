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

#include "dsp/hmx_mgr.h"
#include "dsp/ops.h"
#include "dsp/hmx_queue.h"
#include "dsp/power.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "hexagon_schema.h"
#include "region_ops.h"

using namespace executorch::backends::hexagon;

/* One blob to run, from the generated blob_fixture.h. The struct is declared
 * here and the header included below it so the header can name the type. */
struct BlobFixture {
  const char *tag;
  const unsigned char *blob;
  unsigned long long blob_bytes;
  const unsigned char *inputs;
  unsigned int inputs_size;
  /* The length the caller would hand this blob at run time, for the cases whose
   * blob carries a dynamic trailer. The runtime reads it off the shape of the
   * input the trailer names (hexagon_backend.cpp:2006-2021); a fixture has no
   * shapes, so the test states the same number here and the patches below turn
   * it into params exactly the way that code does. Zero for a static blob. */
  long long runtime_length;
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
extern "C" int htp_ops_unary_clamp(uint8_t *dst, uint8_t *src, int32_t size,
                                   int32_t min_bits, int32_t max_bits);
extern "C" int htp_ops_unary_row_guard(uint8_t *dst, uint8_t *mask, uint8_t *src,
                                       int32_t size, int32_t row_len, int32_t pad);
extern "C" int htp_ops_unary_scale(uint8_t *dst, uint8_t *src, int32_t size,
                                   int32_t scale_bits);
extern "C" int htp_ops_conv_depthwise2d_fp16(
    uint8_t *dst, uint8_t *src, uint8_t *weight, uint8_t *bias, int32_t batch,
    int32_t ih, int32_t iw, int32_t oh, int32_t ow, int32_t c4, int32_t kernelY,
    int32_t kernelX, int32_t strideY, int32_t strideX, int32_t padY, int32_t padX,
    int32_t dilateY, int32_t dilateX, int32_t relu, int32_t relu6);
extern "C" int htp_ops_im2col_convolution_fp16(uint8_t *output, uint8_t *input,
                                               uint8_t *weight, uint8_t *bias,
                                               const HmxIm2ColConvParam *params);
extern "C" int htp_ops_zero(uint8_t *dst, int32_t size);
extern "C" int htp_ops_flash_attn(uint8_t *o, uint8_t *q, uint8_t *k, uint8_t *v,
                                  uint8_t *mask, uint8_t *workspace,
                                  uint8_t *pastK, uint8_t *pastV, int32_t qo_len,
                                  int32_t seq_current, int32_t seq_add,
                                  int32_t n_heads, int32_t n_kv_heads,
                                  int32_t head_dim, float scale,
                                  int32_t mask_stride, int32_t max_kv_len,
                                  int32_t value_c4);
extern "C" int htp_ops_pool2d_fp16(uint8_t *output, uint8_t *input, int32_t batch,
                                   int32_t ih, int32_t iw, int32_t oh, int32_t ow,
                                   int32_t c4, int32_t kernelY, int32_t kernelX,
                                   int32_t strideY, int32_t strideX, int32_t padY,
                                   int32_t padX, int32_t padType, int32_t countType,
                                   int32_t poolType);
extern "C" int htp_ops_shared_gather(uint8_t *dst, uint8_t *indices, uint8_t *weight,
                                     int32_t selectSize, int32_t ic, int32_t oc,
                                     int32_t bytes, int32_t isInt4,
                                     int32_t scaleBlockNum, int32_t scaleAsymmetric);
/* The one kernel here that fills two outputs. It is not in `dsp/ops.h`, and it
 * is the only entry point in this file whose second output has no reader: the
 * emitter allocates the positions a slot of their own because the kernel
 * refuses to run without one, and nothing copies them out. A null there is a -1
 * rather than a wrong number, which is why the return value is printed. */
extern "C" int htp_ops_topkv2_k1_fp16(uint8_t *values, uint8_t *indices,
                                      uint8_t *input, int32_t rowSize,
                                      int32_t rows);
extern "C" int htp_ops_matmul_q4a16_gemv_i8(uint8_t *output, uint8_t *activation,
                                            uint8_t *weight, uint8_t *bias, int32_t k,
                                            int32_t n, int32_t scale_block_num,
                                            int32_t scale_asymmetric);
/* `hmx_matmulw8a16block_gemv_i8` below and `htp_ops_vision_attention_fp16` further
 * down are the two the vendored `dsp/ops.h` declares itself, so they are the two
 * this file must not: Hexagon's `int32_t` is `long`, and a second prototype that
 * spells the same parameter `int32_t` where ops.h spells it `int` is a redeclaration
 * with different types -- which the compiler rejects, but only once something here
 * includes ops.h. The GEMV entry was written before this file included it. */
/* The two GEMV entries stage their operands in VTCM and take the size-zero path
 * out of `matmul_q4block_gemv_i8.c:323-325` when they cannot: they return -1
 * before writing a byte, so the fixture's output is whatever the arena already
 * held. On the device the delegate's own setup acquires it, and the dispatcher
 * re-acquires when it has not (`execute_command.cc:995`); a runner that drives
 * the kernels itself has to say so, which is what main() does below. */
extern "C" void vtcm_manager_setup();
extern "C" int vtcm_manager_acquire();
extern "C" unsigned int vtcm_manager_get_vtcm_size();
/* An HVX vector at `-mhvx-length=128b`, which is what every kernel here assumes
 * of the buffers it is handed. The static assertions below the buffers read this
 * back out of the declarations. */
enum { kVectorBytes = 128 };

enum {
  kPool2d = 1,
  kDepthwise = 2,
  kRasterBlit = 3,
  kUnary = 4,
  kLayerNorm = 8,
  kIm2Col = 12,
  kFlashAttn = 18,
  kBinaryElementwise = 19,
  kSharedGather = 23,
  kZero = 24,
  kTopkV2K1 = 27,
  kSoftmax = 28,
  kReduction = 29,
  kBatchMatmul = 38,
  kQ4A16Gemv = 41,
  kVisionAttention = 43,
  kW8A16Gemv = 45,
};

/* The three unary types that arrive at an entry point of their own, from
 * unary_ops.cc:15-33. */
enum {
  kUnaryClamp = 15,
  kUnaryRowGuard = 16,
  kUnaryScale = 17,
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
/* The fixture's tag, for the diagnostics below: a kernel that refuses says so
 * on stdout, where nothing else distinguishes it from an empty answer. */
static const char *g_tag = "";

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

/* The dynamic trailer's patches, and the run-time length they are computed
 * from. The runtime applies these before it issues anything
 * (hexagon_backend.cpp:2148-2156): every record names one command's parameter
 * slot and overwrites it with `length * scale + add`. The host interpreter does
 * not read the trailer, so until this existed nothing but the device could tell
 * a patched command from one that describes the exported bound. */
static const HexagonDynamicPatch *g_patches = nullptr;
static uint32_t g_patch_count = 0;
static int64_t g_runtime_length = 0;

/* Where the trailer starts: past the ops, past the weights the file carries,
 * and past the external-weight records when the blob left some behind. */
static const uint8_t *trailer_of(const uint8_t *blob, const HexagonBlobHeader *header) {
  const uint8_t *at = blob + sizeof(HexagonBlobHeader) +
                      (size_t)header->n_ops * sizeof(HexagonOp) + header->weights_bytes;
  if (header->version == kHexagonBlobVersionExternalWeights) {
    HexagonExternalWeightsTrailer trailer;
    memcpy(&trailer, at, sizeof(trailer));
    at += sizeof(trailer) +
          (size_t)trailer.n_ext * sizeof(HexagonExternalWeight);
  }
  return at;
}

/* Reads the trailer, if the blob has one. Returns false when it does not. */
static bool load_patches(const uint8_t *blob, const HexagonBlobHeader *header,
                         uint64_t blob_bytes) {
  g_patches = nullptr;
  g_patch_count = 0;
  const uint8_t *at = trailer_of(blob, header);
  if ((uint64_t)(at - blob) + sizeof(HexagonDynamicTrailerV3) > blob_bytes) return false;
  HexagonDynamicTrailerV3 dynamic;
  memcpy(&dynamic, at, sizeof(dynamic));
  if (dynamic.base.magic != kHexagonDynamicTrailerMagic) return false;
  (void)dynamic.example_length;
  const size_t patches_at = sizeof(HexagonDynamicTrailerV3);
  if ((uint64_t)(at - blob) + patches_at +
          (size_t)dynamic.base.n_patches * sizeof(HexagonDynamicPatch) >
      blob_bytes)
    return false;
  g_patches = (const HexagonDynamicPatch *)(at + patches_at);
  g_patch_count = dynamic.base.n_patches;
  return true;
}

static void execute_op(const HexagonOp &op, const HexagonBlobHeader *header,
                       uint32_t op_index) {
  int32_t params[kMaxOpParams];
  memcpy(params, op.params, sizeof(params));
  /* The runtime's order: the length patches first, then the operands' own ones,
   * which read back what the caller just handed in. */
  for (uint32_t i = 0; i < g_patch_count; ++i) {
    if ((uint32_t)g_patches[i].op_index != op_index) continue;
    if (g_patches[i].param_index < 0 ||
        (uint32_t)g_patches[i].param_index >= kMaxOpParams)
      continue;
    const int64_t patched =
        g_runtime_length * g_patches[i].scale + g_patches[i].add;
    params[g_patches[i].param_index] = (int32_t)patched;
  }
  if (op.patch_param != kNoOpPatch) {
    int32_t value;
    memcpy(&value, address(header, op.inputs[op.patch_input]), 4);
    params[op.patch_param] = value * (int32_t)op.patch_scale;
  }

  if (op.type == kDepthwise) {
    htp_ops_conv_depthwise2d_fp16(
        address(header, op.outputs[0]), address(header, op.inputs[0]),
        address(header, op.inputs[1]), address(header, op.inputs[2]), params[0],
        params[1], params[2], params[3], params[4], params[5], params[6], params[7],
        params[8], params[9], params[10], params[11], params[12], params[13],
        params[14], params[15]);
    return;
  }
  if (op.type == kIm2Col) {
    /* The command's parameters are the kernel's own struct, in its field order. */
    htp_ops_im2col_convolution_fp16(address(header, op.outputs[0]),
                                    address(header, op.inputs[0]),
                                    address(header, op.inputs[1]),
                                    address(header, op.inputs[2]),
                                    (const HmxIm2ColConvParam *)params);
    return;
  }
  if (op.type == kZero) {
    htp_ops_zero(address(header, op.outputs[0]), params[0]);
    return;
  }
  if (op.type == kUnary) {
    /* The dispatcher does not hand every unary to the same entry point
     * (execute_command.cc:380-398): clamp, the masked-row guard and the scale
     * take their operands where the rest take an op type. Calling the generic
     * one for all of them would leave clamp's bounds at the zero its task state
     * starts at -- clamp(x, 0, 0) -- which is a wrong number rather than an
     * error, so the split is mirrored here. */
    if (params[1] == kUnaryClamp) {
      htp_ops_unary_clamp(address(header, op.outputs[0]),
                          address(header, op.inputs[0]), params[0], params[3],
                          params[4]);
    } else if (params[1] == kUnaryRowGuard) {
      htp_ops_unary_row_guard(address(header, op.outputs[0]),
                              address(header, op.inputs[0]),
                              address(header, op.inputs[1]), params[0], params[3],
                              params[4]);
    } else if (params[1] == kUnaryScale) {
      htp_ops_unary_scale(address(header, op.outputs[0]),
                          address(header, op.inputs[0]), params[0], params[3]);
    } else {
      htp_ops_unary(address(header, op.outputs[0]), address(header, op.inputs[0]),
                    params[0], params[1], params[2]);
    }
    return;
  }
  if (op.type == kPool2d) {
    /* One operand: the packed activation the emitter's first blit produced,
     * and the pooled one it writes for the second to read back. */
    htp_ops_pool2d_fp16(address(header, op.outputs[0]), address(header, op.inputs[0]),
                        params[0], params[1], params[2], params[3], params[4],
                        params[5], params[6], params[7], params[8], params[9],
                        params[10], params[11], params[12], params[13], params[14]);
    return;
  }
  if (op.type == kTopkV2K1) {
    /* Values, positions, operand, in that order: the positions are the second
     * output the emitter allocates and nothing reads, and the two params are
     * the row length and the row count. */
    int ret = htp_ops_topkv2_k1_fp16(address(header, op.outputs[0]),
                                     address(header, op.outputs[1]),
                                     address(header, op.inputs[0]), params[0],
                                     params[1]);
    if (ret != 0) printf("%s topk returned %d\n", g_tag, ret);
    return;
  }
  if (op.type == kSharedGather) {
    /* Indices first, table second -- the order the emitter binds them in, which
     * is the order execute_command.cc:807 hands mapped_ptrs[0] and [1] to. */
    htp_ops_shared_gather(address(header, op.outputs[0]),
                          address(header, op.inputs[0]),
                          address(header, op.inputs[1]), params[0], params[1],
                          params[2], params[3], params[4], 1, 0);
    return;
  }
  if (op.type == kQ4A16Gemv) {
    int ret = htp_ops_matmul_q4a16_gemv_i8(
        address(header, op.outputs[0]), address(header, op.inputs[0]),
        address(header, op.inputs[1]),
        absent(op.inputs[2]) ? nullptr : address(header, op.inputs[2]), params[1],
        params[2], params[8], params[9]);
    if (ret != 0) printf("%s q4a16 gemv returned %d\n", g_tag, ret);
    return;
  }
  if (op.type == kW8A16Gemv) {
    int ret = hmx_matmulw8a16block_gemv_i8(
        address(header, op.outputs[0]), address(header, op.inputs[0]),
        address(header, op.inputs[1]), address(header, op.inputs[2]),
        absent(op.inputs[3]) ? nullptr : address(header, op.inputs[3]), params[1],
        params[2], params[8]);
    if (ret != 0) printf("%s w8a16 gemv returned %d\n", g_tag, ret);
    return;
  }
  if (op.type == kVisionAttention) {
    /* A scalar walk over the score matrix (attention_entry.cc:24-67), not the
     * flash variant the suite excludes: the flash kernel starts a worker pool,
     * which the simulated QuRT cannot, while this one never reaches the pool at
     * all. Its prototype is `dsp/ops.h`'s rather than a local one, like the w8a16
     * GEMV entry's. */
    float scale;
    memcpy(&scale, &params[4], 4);
    htp_ops_vision_attention_fp16(
        address(header, op.outputs[0]), address(header, op.inputs[0]),
        address(header, op.inputs[1]), address(header, op.inputs[2]),
        op.n_inputs > 3 && !absent(op.inputs[3]) ? address(header, op.inputs[3])
                                                 : nullptr,
        address(header, op.outputs[1]), params[0], params[1], params[2], params[3],
        scale, params[5], params[6]);
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
static int run_blob(const unsigned char *blob, uint64_t blob_bytes, uint8_t *arena,
                    uint64_t arena_bytes, const unsigned char *input_data,
                    uint64_t input_bytes, const char *tag, int64_t runtime_length) {
  const HexagonBlobHeader *header = (const HexagonBlobHeader *)blob;
  if (header->magic != kHexagonBlobMagic || header->version != kHexagonBlobVersion) {
    printf("BADBLOB\n");
    return 1;
  }
  g_arena = arena;
  g_bases = bases_of(header);
  g_tag = tag;
  memset(arena, 0, arena_bytes);
  g_runtime_length = runtime_length;
  load_patches(blob, header, blob_bytes);

  const HexagonOp *ops = (const HexagonOp *)(blob + sizeof(HexagonBlobHeader));
  const uint8_t *sections =
      blob + sizeof(HexagonBlobHeader) + header->n_ops * sizeof(HexagonOp);

  /* The one section the blob carries. The activations are scratch the runtime
   * reserves from the header size and never reads out of the blob, so the arena
   * is left as the memset above made it. */
  const HexagonTensorRef weights{(uint32_t)HexagonTensorSpace::kWeights, 0, 0, 0};
  memcpy(address(header, weights), sections, header->weights_bytes);

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

  for (uint32_t i = 0; i < header->n_ops; ++i) execute_op(ops[i], header, i);

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
  /* The convolution kernels reach VTCM and the HMX unit, which the device build
   * brings up when the backend is initialized. */
  power_setup();
  power_acquire();
  hmx_manager_setup();
  hmx_queue_setup();
  worker_pool_global_init();

  /* One arena, reused: run_blob zeroes it. Sized for the largest fixture, and
   * aligned the way the kernels' vector accesses assume: an HVX load or store
   * wants its address aligned to the vector length, and a byte array gives the
   * linker no reason to -- which is a wrong answer rather than a crash, and one
   * that a change to the fixture list can flip. The section bases are derived
   * from the arena's own address and the blob lays every tensor out on a 128-byte
   * boundary, so the arena has to start on one. Keep the attribute, and see
   * backends/hexagon/test/README.md before changing either side.
   *
   * The assertion is what makes dropping the attribute a build failure rather
   * than a hazard to remember. It cannot be written on the address (not a
   * constant expression) or on the array's type (an array's type alignment is
   * its element's), so it reads the attribute back through `__alignof__` on the
   * object, which the Hexagon compiler answers with the attribute's value. */
  static uint8_t arena[kMaxArenaBytes] __attribute__((aligned(kVectorBytes)));
  static_assert(
      __alignof__(arena) == kVectorBytes,
      "the kernels read this arena a vector at a time");
  /* The blob's own rule for where a section starts has to be the alignment the
   * kernels want, or the bases this file derives from the arena's address would
   * put an operand off a vector boundary. */
  static_assert(kHexagonAlignment == kVectorBytes,
                "the section layout and the vector length have to agree");
  int status = 0;
  /* A device does not run these kernels without VTCM: the delegate's setup
   * acquires it and the dispatcher re-acquires it when it is missing
   * (`execute_command.cc:995`), but this main() is neither of those. Without the
   * call both GEMV entries return -1 before writing anything -- an empty answer
   * that is indistinguishable, in the output slot, from a wrong one. The
   * simulator hands out the same 8 MiB a device has. */
  vtcm_manager_setup();
  printf("vtcm acquired=%d size=%u\n", vtcm_manager_acquire(),
         vtcm_manager_get_vtcm_size());
  for (unsigned i = 0; i < kFixtureCount; ++i) {
    const BlobFixture &fixture = kFixtures[i];
    status |= run_blob(fixture.blob, fixture.blob_bytes, arena, sizeof(arena),
                       fixture.inputs, fixture.inputs_size, fixture.tag,
                       fixture.runtime_length);
  }
  return status;
}
