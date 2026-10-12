# Standalone HMX LPBQ INT4 byte-weight example

Verified on PJZ110 / SM8750 / HTP v79, 2026-10-12. This example follows the
QNN 2.41 `use_16a4w_block` **byte-weight** route: fold each quantized block scale
into an INT8 weight, accumulate all K blocks in HMX, then convert once. The DSP
library needs no QNN library, captured buffers, or captured coefficient words.

The older [hmx_blockwise_minimal](../hmx_blockwise_minimal/README.md) implements
a different arithmetic contract: convert and saturate each nibble-weight block,
then combine on the CPU. Its passing tests do not establish QNN LPBQ equivalence.

## Verified scope

| Check | Result |
|---|---|
| Standalone DSP vs independent NumPy contract | 12 cases, 24,576 Q and FP32 Y values, zero mismatches |
| Known-parameter QNN Linear, M32 K128 N64 | 2,048 Q and FP32 Y values, zero mismatches against the standalone example |
| QNN entry probe | Actual byte-weight entry `0x2b25c0`, `hmx_v75_convhbh1x1_stride1`; original/probe/original outputs identical |
| Older random QNN Linear, M128 K128 N128 | 16,384 Q and FP32 Y values exact; all 512 table words exact after scale normalization fix |
| New randomized QNN Linear, M32 K128 N128, seeds 71/72/73 | 12,288 Q and FP32 Y values exact against original QNN and standalone DSP |
| Host-prepared C weights/table → DSP HMX | 12 base cases, old QNN / 16,384 Q/Y and fresh QNN / 4,096 Q/Y exact; DSP-prepared path regression also passed |
| Simulator / speed / production backend | Not tested or changed in this example |

Output equality for the known-parameter model does not mean its conversion
table is byte-identical: 8 / 256 generated words differ from the captured QNN
table, without changing this model's outputs. No captured words are substituted
to make tests pass. See the rounding investigation below.

## Interface and arithmetic

`hmx_matmul_u16s4_lpbq_tile` accepts row-major A uint16 `[32,128]`, logical W
INT4 `[-8,7]` stored one value per INT8 byte `[128,64]`, FP32 scales `[64,4]`,
activation/output scales and zero points, a layer-wide normalization scale, and returns Q uint16 and dequantized
FP32 Y `[32,64]`. K block size is 32. The caller supplies 16 KiB of
2048-byte-aligned VTCM and an active HMX power/lease/lock.

Weight preparation matches
`backends/qualcomm/builders/node_visitor.py::make_qnn_per_block_config`:

```text
sw[n]       = max_b(scales[n,b]) / 16                 // FP32
m[n,b]      = clamp(round_to_even(scales[n,b]/sw[n]), 1, 16)
Wbyte[k,n]  = W4[k,n] * m[n,k/32]                    // signed INT8
correction  = floor((-za * sum_k(Wbyte[k,n]) + 127) / 256)
```

The scale multiplier is **1..16**, including for a zero block scale when other
blocks have positive scales. It is not 0..16. INT4 `-8 * 16 = -128` fits the
byte-weight path.

QNN's host preparation uses a layer-wide scale. The independent implementation
reproduces the operation order, with each arithmetic step rounded to FP32:

```text
global_weight_scale = max_n(sw[n])              // over the entire layer
normalized[n]       = FP32(sw[n] / global_weight_scale)
product             = FP32(global_weight_scale * sa)
global_gain         = FP32(product / sq)
gain[n]             = FP32(normalized[n] * global_gain)
```

Pass the same `global_weight_scale` to every N64 tile of a larger layer.
Passing 0 selects the maximum of the current tile, suitable for a whole N64
layer; selecting a maximum independently for each tile changes rounding.
The implementation encodes a 22-bit multiplier and shift from `(bits(gain)+2)`
and stores the output bias `16*zq+8`. The complete integer
oracle, including the HMX sticky term, is in [verify.py](verify.py):

```text
t    = floor(sum_k(Aq[k]*Wbyte[k])/256) + correction
Q    = clamp(floor(((t*2^(shift-7) + (t!=0))*mult
                   + (16*zq+8)*2^26) / 2^30), 0, 65535)
Y    = FP32((int(Q)-zq)*sq)
```

The correction's exact-half rule rounds toward negative infinity; the `+127`
is deliberate. It was checked against all 128 columns of the random QNN
model's captured correction table. Ordinary floating matmul followed by round
does not reproduce this conversion contract.

Supported parameters require positive finite `sa`/`sq`, nonnegative finite
block scales with a positive normal `max_scale/16` for each column, and an
encoded shift in 8..22. Unsupported parameters return 1 before issuing HMX
instructions. This fixed tile API does not implement arbitrary shapes,
all-zero scale channels, or other QNN conversion modes.

## Host preparation and prepared DSP execution

`prepare.c` contains portable C `hmx_lpbq_prepare`. It produces 8,192 bytes
of native byte weights and 1,024 bytes of conversion table for one K128/N64
tile; it needs neither QNN nor HMX/VTCM. `build.sh` builds it for the DSP and
as an x86 host library, `libhmx_lpbq_prepare.so`, with normal FP32 arithmetic.
The host transport helper `prepare_host.py` calls the C API, checks every
weight/table entry against the independent NumPy contract, and serializes the
result. These are newly generated parameters, not captured QNN data.

The DSP API `hmx_matmul_u16s4_lpbq_prepared_tile` accepts Aq, prepared weights,
table, sq/zq, outputs and VTCM workspace. It copies the prepared buffers into
VTCM, packs activation blocks and performs HMX accumulation/conversion. It
does not compute scale ratios, expand logical INT4, or generate coefficients.
Output unpacking and FP32 dequantization remain DSP scalar C. The original
`hmx_matmul_u16s4_lpbq_tile` remains available and prepares on the DSP before
calling the same compute API.

For static weights and fixed quantization parameters, cache the prepared data
and reuse it for new activation **values**. Changing W, block scales, sa/sq,
za/zq requires corresponding preparation again. Every N tile must use the
same layer-wide `global_weight_scale`. The prepared API trusts these buffers;
its sq/zq and activation quantization must match their preparation parameters.
The example copies buffers into VTCM per invocation; no latency improvement or
persistent VTCM cache is claimed.

```bash
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_lpbq_minimal/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_lpbq_minimal/run_device.py --host-prepare \
  --qnn-fixture .tmp/scratch/mnv2/lpbq_fix
```

Omit `--qnn-fixture` to run just the 12 independent cases. `prepared.bin`
contains a little-endian case count followed by 8,192 weight bytes and 256
uint32 table words per case. The DSP FARF marker `host_prepared 1`, deployed
hashes, and `checks.json` identify this path. The host/DSP split retains the
normal-mode and N64 QNN preparation boundaries described below.

## Layout and instruction sequence

| VTCM byte offset | Size | Contents |
|---:|---:|---|
| 0 | 2,048 | One packed K32 activation block, reused |
| 2,048 | 8,192 | Four effective byte-weight blocks |
| 10,240 | 4,096 | Native HMX output |
| 14,336 | 1,024 | Four generated conversion banks |

For local `k=0..31`, block `b`, row `r`, column `n`:

```text
A32[(r/2)*64 + k*2 + r%2] = A[r,32*b+k]
Wnative[b*2048 + (n/32)*1024 + (k/4)*128 + (n%32)*4 + k%4]
table_word = ((n/32)*2 + n%2)*64 + (n%32/2)*2
Qnative[(n/32)*1024 + (r/2)*64 + (n%32)*2 + r%2]
```

The kernel clears once, loads four generated bias banks at byte offsets
`0x000,0x101,0x202,0x303`, then issues four packets:

```asm
{ activation.ub = mxmem(A32,0x77c)
  weight.b = mxmem(Wblock,0x7ff):deep }
```

There is no accumulator reset or intermediate output conversion between
blocks. Finally `cvt.uh = acc(...):2x2` uses selectors `0x401,0x120c` for the
first N32 group and `0x2401,0x320c` for the second, with `mxmem(...,0x700) = cvt`.
DSP scalar C unpacks the final output and dequantizes it.

## Reproduce without QNN or capture

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_lpbq_minimal/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_lpbq_minimal/run_device.py
```

The 12 cases cover activation zero point, several ranges, full UINT16 values,
zero weights, both saturation directions, a single-element layout probe, zero
block scales, scale ties, and cancellation across blocks. Cancellation exposes
implementations that prematurely saturate individual block outputs. VTCM is
poisoned differently between successive invocations.

Build overrides: `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`,
`HMX_EXAMPLE_BUILD_DIR`. Device options: `--host`, `--sdk`, `--build-dir`,
`--remote-dir`. The script checks deployed hashes, the DSP main return value,
every Q/Y element, and saves disassembly and FARF evidence. It waits for
llama-server/llama-bench; llama-cli does not block this experiment.

## Optional original-QNN comparison, also without capture

`export_qnn.py` creates a QNN LPBQ Linear from the same logical integer weights,
block scales and quantization parameters as standalone case 3, repeated to N128 by default. It requires
exactly one delegate and excludes external compute. `run_qnn_reference.py`
uses the **unmodified** SDK skel, verifies its hash, and runs a regular runner;
it performs no patching, interception or buffer capture.

```bash
export PYTHONPATH=/home/yydh
export QNN_SDK_ROOT=/home/yydh/env/qairt/2.41.0.251128
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang"
.tmp/locks/lockrun heavy -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_lpbq_minimal/export_qnn.py
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_lpbq_minimal/run_qnn_reference.py \
  --fixture .tmp/scratch/mnv2/hmx_lpbq_minimal/qnn_reference
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python \
  backends/hexagon/examples/hmx_lpbq_minimal/run_device.py \
  --qnn-fixture .tmp/scratch/mnv2/hmx_lpbq_minimal/qnn_reference
```

The comparison runner defaults to the existing phone QNN runner/skel directories;
override `--runner-dir`, `--skel-dir`, `--sdk` for another installation. The
fixture passed to the standalone DSP contains only logical A/W/scales/zero
points, never a QNN conversion table. `qnn_before.raw` is just the reference
output filename; the public script runs an ordinary, unpatched QNN reference.

Evidence is under `.tmp/scratch/mnv2/hmx_lpbq_minimal/`. The earlier dispatch
probe, before/probe/after raw outputs, and random-model discrepancy are under
`.tmp/scratch/mnv2/lpbq_fix/`; scratch is not versioned.

## Conversion-precision diagnosis and fix

The original direct FP64-to-FP32 gain policy differed from the random 128³
QNN model in 14/128 multiplier columns and 4/16,384 outputs. Replacing it with
the FP32 normalization sequence above makes all 512 conversion-table words,
16,384 Q values and 16,384 FP32 Y values identical. The device library SHA is
`c3ae41016e36ef9950b31b6e4c9945732e37a4bccfa0cb9f0b105a2f6c33ec23`.

The executed preparation path was measured with GDB in SDK 2.41
`libQnnHtp.so`: `normalize_channel_scales` core `0x19eeee8`, then
`conv_combine_scales` core `0x131d4c6`, during `GraphPrepare::const_prop`.
The model context already contains the captured DSP conversion table verbatim.
The separate DSP QF32 function was not the source of this AOT model's table.

An additional N64 randomized export exposes an unresolved QNN preparation
boundary: the normalization input has 128 FP32 values, only the first 64 match
the model. The tail includes large unrelated values; its maximum
`2.5313279e13` is passed to normalization instead of the logical maximum
`0.0062359297`. The resulting model differs by 1 Q value, and 17 coefficient
words differ from logical-scale preparation. The source of that tail is not
yet established. Using the directly measured scalar reproduces all table
words from the same instrumented export; another export of identical logical
parameters has 11 different words. Reconstructing that boundary solely from
logical parameters is not claimed. The previously
verified N64 known-parameter model still has 8 table-word differences, without
output differences. Fresh arbitrary-parameter reference exports therefore
use N128 by default, where all normalization channels are defined.

Detailed findings and commands: [LPBQ preparation investigation](../../QNN_LPBQ_PREPARATION.md).
