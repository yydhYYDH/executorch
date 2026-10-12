# HMX conversion parameters from logical quantization values

QNN 2.41 / SM8750 v79, verified 2026-10-12. The shared encoder removes captured
coefficient constants from the A16W8 and A16W16 compute libraries. It accepts
new FP32 scales and output zero points; the new tile APIs also accept a new
activation zero point. No QNN library or capture file is needed to run them.

## 2026-10-12: A8W8, A8W4 and A16W4 runtime parameters

These three examples now have `*_tile_quantized` APIs taking
`(A, W, sa, sw, sq, za, zw, zq, Q, Y, workspace)`:

| API | M/K/N | Output | Accumulator convention |
|---|---|---|---|
| `hmx_matmul_u8s8_tile_quantized` | 64/32/64 | uint8 | signed dot product |
| `hmx_matmul_u8s4_tile_quantized` | 64/32/128 | uint8 | signed dot product ×16 |
| `hmx_matmul_u16s4_tile_quantized` | 32/32/64 | uint16 | dot product truncated by 256 before conversion |

All three take row-major arrays. The weight container is **int8**, including
INT4 examples: each logical `Wq` is a byte, and packing first centers it by
`Wq-zw`. `zw` may be -128..127; the centered value must fit -128..127 for
A8W8 or -8..7 for either INT4 path. A QNN symmetric signed-weight model has
`zw=0`; centering a caller's nonzero-zw container is an API feature, not a
claim that QNN exports every asymmetric weight configuration. Activation and
output zero points can be any value in their uint8/uint16 ranges.

Use positive finite FP32 `sa/sw/sq`. U8 normal conversion supports encoded
shifts 0..31; U16 supports 8..22. Non-normal intermediate gains, unsupported
shifts/modes and unrepresentable centered weights return **1**, without
changing Q/Y/workspace or issuing HMX instructions. Success returns **0**.
These remain fixed tile examples, with no bias/per-channel or LPBQ support.
Workspace must be VTCM, 2048-byte aligned, with the example's documented size;
the caller owns the HMX power/resource lease and lock. The previous fixed
parameter APIs remain available and use the new implementation.

On PJZ110/SM8750/v79, **276 cases** (92 per family; 1,318,912 Q and the same
number of Y values) passed bit-exact against the independent integer oracle.
Cases alternate all three HMX families in one lease, exercise random FP32
scales and all zero-point edges, full signed weight endpoints, shift endpoints,
saturation, zero columns and isolated +/-1 weights. Six rejected kernel calls
also preserve sentinel output/workspace bytes. Host rejection tests cover
invalid scales and accumulator conventions.

Three newly exported, **unmodified QNN 2.41** models (cases 3/4/5, one per
family, non-power-of-two scales) agree with the HMX integer oracle and
FP32 dequantization at all **14,336 Q/Y** values. These use already centered
weights in QNN and retain one delegate with no external compute nodes.
This is tested-model evidence, not every QNN mode/shape. Running the standalone
examples requires no QNN or coefficient captures; QNN is an optional oracle.

Reproduce from the repository root:

```bash
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_quant_params/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_quant_params/run_lowbit_device.py

# Optional QNN comparison: repeat with --case 4 and 5 in separate output dirs.
.tmp/locks/lockrun heavy -- env QNN_SDK_ROOT=/home/yydh/env/qairt/2.41.0.251128 \
  LD_LIBRARY_PATH=/home/yydh/env/qairt/2.41.0.251128/lib/x86_64-linux-clang \
  PYTHONPATH=/home/yydh /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_quant_params/export_lowbit_qnn.py \
  --case 3 --output-dir .tmp/scratch/mnv2/lowbit_quant/qnn3
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_quant_params/run_lowbit_qnn.py \
  --fixture .tmp/scratch/mnv2/lowbit_quant/qnn3
```

Device evidence: `.tmp/scratch/mnv2/hmx_quant_params/example/lowbit/checks.json`,
raw `output.u16` (U8 Q widened only for the fixture file), `output.f32`,
`coefficients.u32`, hashes and logs beside it. QNN evidence:
`.tmp/scratch/mnv2/lowbit_quant/qnn{3,4,5}/qnn_checks.json`.
Standalone library SHA256:
`f41c71187b7fce539b6ba0fa1b23d85c7c073aa87b8664aa5a4f59d01a2d314f`;
Q file SHA256:
`15846c901d6c48f32e721f5bf7ff61b95d486d2cf25b1869ae6777f948376ea3`.

### U8 coefficient and rounding

The ordinary U8 path uses **12 multiplier bits**, not 11. The extra LSB is
coefficient bit 31; it is not part of the output zero point:

```text
b = bits(FP32(FP32(sa*sw)/sq)) + 0x800
mult = ((b >> 12) & 0x7ff) | 0x800
shift = exponent(b) - 127 + 24 - d     # d=0 A8W8, d=4 A8W4
word = ((mult >> 1) & 0x3ff) | ((mult & 1) << 31)
     | (shift << 10) | (zq << 23) | 0x400000
correction[n] = -za * sum_k(Wq[k,n]-zw) * 2^d
```

QNN 2.41 x86 encoder loop `0x131e578..0x131e5f1` establishes this encoding;
full U8 worker `0x32ceff0` decodes the extra bit, and math helper `0x32cd360`
shows the truncation/sticky step. The historical captured A8W8 scales
`sa=0.003786046290770173`, `sw=0.004545191302895546`,
`sq=0.005781101528555155`, `zq=115` generate `0xb9c03e18`, matching the
actual device table. The small older `mxconvert_to_u8` helper does not include
this full precision mode and must not be used as its oracle.

For the covered normal range, with centered dot product `dot`:

```text
x = dot * 2^d * 2^shift
t = floor(x / 2^19)
t = t OR (floor(x / 2^20) != 0)       # discarded information becomes sticky
Q = clamp(floor((t*mult + (16*zq+8)*2^12) / 2^16), 0, 255)
Y = FP32((Q-zq) * sq)
```

A16W4 uses the shared U16 coefficient encoder with d=8. Its column correction
is `-round_half_away(za*sum(Wq-zw)/256)`, and its normal conversion follows the
existing U16 integer oracle. Do not replace either oracle with ideal FP32
matmul followed by generic round-to-nearest; that omits HMX truncation.

## Original A16W8/A16W16 evidence and scope

| Evidence | Result |
|---|---|
| Generated coefficient words vs historical QNN per-tensor models | 39 / 39 pairs bit-exact |
| Host encoder vs independent Python bitfield reference | 71 / 71 pairs bit-exact |
| DSP-generated words vs Python and historical QNN pairs | 71 / 71 and 39 / 39 respectively, bit-exact |
| DSP HMX outputs vs independent integer reference | 71 cases × 2,048 = 145,408 Q values, zero mismatches |
| New parameter sweep | 32 cases covering both weight families, shifts 8..22, za/zq including 0 and 65535 |

The 39 comparisons use **historical actual device-captured coefficients**, not
newly run whole-model QNN outputs. The 71 HMX runs use newly generated tables
and deterministic tile inputs; they do not establish full-model equivalence
for arbitrary shapes or QNN modes. The existing fixed-model A16W8 numerical
test is also rerun after replacing the embedded coefficients.

Supported: positive finite FP32 scales whose intermediate gain is normal,
normal U16 conversion mode, accumulator scaling d=8 or 16, encoded shift 8..22,
arbitrary valid uint16 activation/output zero points. Unsupported modes or
scale ranges return 1 before issuing HMX instructions. This does not claim
the full QNN scale range, negative scales, subnormal gains, all-zero channels,
per-channel preparation precision, bias-enabled operators, or LPBQ's exact
parameter preparation. LPBQ's residual is recorded in its own README.

## What the coefficients depend on

For affine quantization:

```text
Areal = sa * (Aq - za)
Wreal = sw * (Wq - zw)
gain  = sa * sw / sq
Q     ≈ zq + gain * sum_k((Aq-za)*(Wq-zw))
```

The table contains **two distinct parts per output column**:

| Part | Inputs |
|---|---|
| Coefficient pair (multiplier, shift, output bias/mode) | sa, sw, sq, zq, accumulator scaling d, conversion mode, exact FP arithmetic |
| Accumulator correction | za, sum of effective centered weights over K, d, correction rounding; bias if present |

M does not directly enter either arithmetic formula. N determines how many
entries exist and how they are banked. K affects weight sums and accumulation;
it can also cause QNN to choose another kernel/accumulator mode. Holding the
quantization parameters and mode fixed does not imply that changing shape
never changes table layout or runtime configuration.

Changing Aq values alone does not change the table for static quantization.
Changing weights with fixed scales requires regenerating the correction half,
not necessarily the coefficient half. Changing scales or zq requires encoding
a new coefficient pair. Changing za requires a new correction. Per-channel
sw generally requires one pair per column.

## Exact observed per-tensor coefficient generation

The matching calculation is deliberately **two FP32 operations**:

```text
product = FP32(sa * sw)
gain    = FP32(product / sq)
b       = uint32_bits(gain) + 2
mult    = ((b >> 2) & 0x1fffff) | 0x200000
shift   = exponent_bits(b) - 127 + 16 + d
bias    = 16*zq + 8
```

d=8 for byte weights; d=16 for the split signed-byte INT16 path. The extra d
compensates for the hardware accumulator's `2^-d` scaling before conversion.
The `+2` and mantissa extraction implement the observed 22-bit conversion
multiplier quantization. They are not an arbitrary numerical tolerance.

For `encoded = mult ^ 0x200000`, pack:

```text
w0 = ((encoded & 0x7ff) >> 1) | ((encoded & 1) << 31)
   | (shift << 10) | ((bias & 0xfff) << 19)
w1 = ((encoded >> 11) & 0x3ff) | (((encoded >> 21) & 1) << 16)
   | ((bias & 0xff000) << 11)
```

The function in [conversion.c](conversion.c) forces the intermediate product
to FP32. FP64 intermediates matched only 36/39 QNN pairs; multiplication by a
precomputed FP32 reciprocal matched 37/39. The multiply-then-divide sequence
matched all 39, including three cases whose mantissa LSB distinguishes these
policies. DSP execution independently reproduces the same bits.

Static encoder evidence is in QNN 2.41 x86 `libHtpPrepare.so`: `0x131e28e`,
bit increment/extraction loop `0x131e387`, and bitfield stores through
`0x131e42d`. These addresses are binary-specific. The tested original v79 skel
SHA256 is `5d6900ed5c2edb2c81517adf251ca34a0eae43d41c7aeb799d855c66c84f0400`.

## Weight zero point and the correction half

The A16W8/A16W16 tile APIs accept **already centered native weights**, not an arbitrary
affine QNN weight container. Set `Wnative=Wq-zw` before packing, provided the
result is representable in the selected native format. For the original
A16W8 UINT8 container zw=128; for the original INT16 container zw=0.

If weights cannot be centered into the hardware format, changing coefficient
words alone is insufficient. Algebraically, uncentered multiplication requires:

```text
sum((Aq-za)*(Wq-zw))
 = sum(Aq*Wq) - zw*sum(Aq) - za*sum(Wq) + K*za*zw
```

The `-zw*sum(Aq)` term depends on each input row, so it cannot be hidden in a
static per-column table. It needs different packing or additional computation.
An arbitrary weight zero point is not a supported parameter of these APIs.

For current native weights, correction is `-round_half_away(za*sum(W)/2^d)`.
This matches all 1,248 available entries across the same 39 per-tensor capture
records. That sample does not distinguish all exact-half policies; LPBQ's
separate table was shown to use `floor((-za*sum(W)+127)/256)`. Do not silently
transfer one family's half-tie rule to every QNN mode. The INT16 native format
also has the existing upper representability boundary 32639; correction uses
the same effective values as the HMX packer.

The tested normal-mode output contract is:

```text
t = floor(sum(Aq*Wnative)/2^d) + correction
Q = clamp(floor(((t*2^(shift-7)+(t!=0))*mult
                 + (16*zq+8)*2^26)/2^30), 0, 65535)
```

This includes the observed sticky term and output rounding bias. A float
matmul followed by a generic `round` is not the exact reference for it.

## API

Generate the coefficient half on the host or DSP:

```c
uint32_t coefficient[2];
int rc = hmx_u16_conversion_params(sa, sw, sq, zq, 8, coefficient);
```

Compute with generated coefficients and rebuilt weight-sum corrections:

```c
int rc = hmx_matmul_a16w8_tile_quantized(
    a_u16, w_centered_i8, sa, sw, sq, za, zq, q_u16, workspace);
// Analogous hmx_matmul_a16w16_tile_quantized(..., w_centered_i16, ...).
```

Both fixed M32 K32 N64 tile APIs retain the existing VTCM/power/lock contract.
They validate coefficients before issuing HMX and return 1 for unsupported
parameters. Their original `void` entry points remain fixed-model wrappers,
but those wrappers now invoke this generator instead of embedding table words.
No production backend is changed.

## Reproduce

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_quant_params/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_quant_params/run_device.py
```

This default needs no capture and runs 32 new parameter cases. To include the
39 historical QNN coefficient comparisons:

```bash
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_quant_params/run_device.py \
  --captures .tmp/scratch/mnv2/qnn_quant_capture
```

The optional capture input is verification evidence; it is not sent to HMX as
table words. The DSP fixture contains only sa/sw/sq/za/zq/d. Results are in
`.tmp/scratch/mnv2/hmx_quant_params/example/checks.json`, full `output.u16`,
`coefficients.u32`, disassembly, deployment hashes and device logs. Additional
host comparison evidence is in `.tmp/scratch/mnv2/qnn_conversion_general/`.
