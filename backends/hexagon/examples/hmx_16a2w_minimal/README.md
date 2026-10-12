# A16W2 native HMX minimal example

Implemented and device-checked on 2026-10-12, OnePlus 13 / SM8750 / HTP v79,
QNN 2.41.0.251128. The standalone example uses no QNN runtime or captured
numeric constants. It implements one fixed **M32 × K32 × N64** tile with
UINT16 activations, native signed INT2 weights and UINT16 output; it also
returns the output dequantized to FP32.

## Build and run without QNN

From the repository root:

```bash
QUANT_PY=/home/yydh/miniconda3/envs/et/bin/python
export HEXAGON_SDK_ROOT=/home/yydh/hexagon-tutorial/tools/hexagon-sdk
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools"
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_16a2w_minimal/build.sh
.tmp/locks/lockrun device -- "$QUANT_PY" backends/hexagon/examples/hmx_16a2w_minimal/run_device.py
```

The runner defaults to SSH `oneplus13-reverse`; override `--host`, `--sdk`,
`--build-dir` or `--remote-dir` as needed. It waits for `llama-server` and
`llama-bench`, checks deployment hashes, launches the SDK DSP helper, pulls
fresh complete outputs and validates them. A failed DSP invocation is not
silently retried. No simulator test is needed.

The default run generates 40 deterministic cases. They cover all four INT2
values, zero columns, isolated weights, positive and negative accumulation,
per-channel scales, activation/output zero-point edges and saturation. The
runner alternates the parameterized and prepared APIs. It also rejects 10
invalid preparation inputs on the host and checks that neither output buffer
was modified.

## Calling the kernel

See [matmul.h](matmul.h), [prepare.c](prepare.c) and [matmul.c](matmul.c).
Inputs and returned Q/Y are row-major:

```c
uint8_t packed_weight[HMX_A16W2_WEIGHT_BYTES];
uint32_t table[HMX_A16W2_TABLE_BYTES / sizeof(uint32_t)];

int status = hmx_a16w2_prepare(
    w, weight_scales, sa, sq, za, zw, zq, packed_weight, table);
if (status == 0) {
  hmx_matmul_u16s2_prepared_tile(
      a, packed_weight, table, sq, zq, q, y, workspace);
}
```

`w` is a logical `int8_t[32*64]` container, not prepacked bytes. Each centered
value `w[k,n]-zw` must be in `[-2,1]`. `weight_scales[64]` supplies positive
finite FP32 scales. `sa` and `sq` are positive finite activation/output scales;
`za` and `zq` are UINT16 zero points. `zw` is in `[-128,127]`, and only values
representable in the logical container are accepted. QNN's tested INT2 export
uses symmetric weights (`zw=0`); the example's nonzero `zw` support merely
centers the caller's container before packing.

Preparation is portable C and can run on the host or DSP. It returns 1 for
unsupported parameters before writing either prepared output. It currently
supports the shared encoder's normal UINT16 conversion with shift 8..22;
zero scales, extreme shifts, bias and alternate conversion modes are outside
the contract. The prepared API assumes valid parameters produced by the
encoder, including matching `sq` and `zq`.

The caller must own HMX power, a compute-resource lease and the HMX lock.
`workspace` must be VTCM, 2048-byte aligned and at least 16384 bytes.
[main.c](main.c) demonstrates acquisition and cleanup. The convenience
`hmx_matmul_u16s2_tile` prepares parameters on the DSP before calling the
prepared API. Both APIs currently repack A and copy 512 weight bytes plus
1024 table bytes into VTCM on each call; no residency or performance result
is claimed.

## Native instructions and layouts

The computation uses one native INT2 load packet:

```asm
mxclracc
{ activation.ub = mxmem(A, 0x77c)
  weight.c = mxmem(W, 0x1ff):deep }
```

Although spelled `activation.ub`, this selector loads the packed UINT16
activation tile. It does not quantize the inputs to UINT8. There is no INT2
expansion into INT8/FP16 in this implementation.

Activation index, in UINT16 elements:

```text
A_native[(r/2)*64 + 2*k + r%2] = A[r,k]
```

Let `Wc = W-zw`. For `k=0,16`, `n=0..63`, `i=0..3`, the 512-byte weight layout is:

```text
index = (n/32)*256 + (k/16)*128 + (n%32)*4 + i
packed[index] = sum_s=0..3 ((Wc[k+i+4*s,n] & 3) << (2*s))
```

Thus one byte holds four 2-bit two's-complement weights, ordered by K offsets
`i, i+4, i+8, i+12`, rather than four adjacent K positions.

Each 32-column output group uses two conversion blocks, selecting even then
odd columns:

```asm
bias = mxmem2(table_group)
cvt.uh = acc(0x401):2x2
bias = mxmem2(table_group + 256 bytes)
cvt.uh = acc(0x20c):2x2
mxmem(output_group, 0x700) = cvt
```

The converted output index, in UINT16 elements, is
`(n/32)*1024 + (r/2)*64 + (n%32)*2 + r%2`.
It is unpacked to row-major Q; FP32 Y is `(Q-zq)*sq`.

## Parameter generation and numerical semantics

Per-channel gain uses the following **FP32 operation order**, with a rounding
boundary at every operation:

```text
global = max(weight_scales)
base = FP32(FP32(global * sa) / sq)
gain[n] = FP32(FP32(weight_scales[n] / global) * base)
```

The shared [conversion encoder](../hmx_quant_params/README.md) creates each
channel's coefficient pair with `(gain[n], 1, 1, zq, divisor_shift=8)`.
Directly evaluating `FP32(FP32(sa*sw[n])/sq)` does not always give QNN's bits:
it matched 57/64 channel pairs in the captured random model; the normalized
order matched all 64 and the complete 1024-byte table.

The activation zero-point correction is
`corr[n] = -round_half_away_from_zero(za * sum_k(Wc[k,n]) / 256)`.
For channel `n`, the table word index is
`((n/32)*2+n%2)*64 + 2*((n%32)/2)`.
The coefficient pair occupies `index,+1`, correction occupies `index+32`,
and `index+33` is zero. This is four 256-byte conversion blocks total.

The [independent oracle](verify.py) models the native accumulator/conversion:

```text
t = floor(sum_k(A[r,k] * Wc[k,n]) / 256) + corr[n]
Q = clamp_u16(floor(
    ((t * 2^(shift[n]-7) + (t != 0)) * mantissa[n]
      + (16*zq+8) * 2^26) / 2^30))
Y = FP32(FP32(Q-zq) * sq)
```

This deliberately models QNN/HMX truncation and conversion. A generic float
or PyTorch QDQ matmul is not a bit-exact oracle for these operations.

## Optional fresh QNN comparison

INT2 export requires `is_linear_per_channel=True`; per-tensor configuration
is rejected by the current quantizer. These scripts export a single delegate
with controlled logical parameters, run the **unmodified** QNN skel and compare
its complete output. They need no instrumentation/capture session:

```bash
export PYTHONPATH=/home/yydh
export QNN_SDK_ROOT=/home/yydh/env/qairt/2.41.0.251128
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang"
QUANT_PY=/home/yydh/miniconda3/envs/et/bin/python
QNN_FIXTURE=.tmp/scratch/mnv2/hmx_16a2w_minimal/qnn_reference
.tmp/locks/lockrun heavy -- "$QUANT_PY" backends/hexagon/examples/hmx_16a2w_minimal/export_qnn.py --output-dir "$QNN_FIXTURE" --case 3
.tmp/locks/lockrun device -- "$QUANT_PY" backends/hexagon/examples/hmx_16a2w_minimal/run_qnn_reference.py --fixture "$QNN_FIXTURE"
.tmp/locks/lockrun device -- "$QUANT_PY" backends/hexagon/examples/hmx_16a2w_minimal/run_device.py --qnn-fixture "$QNN_FIXTURE"
```

The reference runner defaults to existing phone `qnnb/qnn_executor_runner` and
`matmul_rebenchmark/dispatch_probe/baseline` skel; override `--runner-dir` and
`--skel-dir` for another deployment. Hashes of the model, inputs, runner and
skel are recorded. `--qnn-fixture` may be repeated.

## Device and capture evidence

Scratch root: `.tmp/scratch/mnv2/hmx_16a2w_minimal/`.
The evidence tiers are separate:

- **Runtime dispatch:** instrumented private QNN skel confirmed entry
  `0x2b8280` (`hch`) and deep branch `0x2b84c0` for both M32/K32/N64 and
  M128/K128/N128. Captured selectors are A=`0x77c`, W=`0x1ff`, store=`0x700`;
  flags=`0xa0`. Seven candidate entry/deep paths were instrumented. Original
  before, private probe and original after outputs were byte-identical for
  both models. See `fullcapture_device_results.json`, `fullcapture/manifest.json`
  and `fullcapture_use_16a2w_*.farf.log`.
- **Preparation/layout:** for the first M32/K32/N64 captured tile, the generated
  activation tile, all 512 weight bytes and all 256 table words match the
  captured buffers. `analyze.py` and `captured_{a,w,table}.bin` preserve the
  derivation; `capture_layout_checks.json` records zero mismatches for all
  three buffers. The capture establishes this tile, not every larger layout.
- **Standalone DSP numerics:** 40 constructed cases plus two QNN fixture cases
  passed; all 86,016 Q values and all 86,016 FP32 Y values match the integer
  oracle, with complete weight/table comparison. See `example/checks.json`
  and `combined.log`. Host invalid-input checks are recorded separately.
- **Original QNN comparison:** both fixture cases match standalone DSP Q/Y
  bit-for-bit (4,096 values each). One is the original random calibrated model
  (`qnn_initial/`); the other is a fresh controlled per-channel export
  (`qnn3/`, case 3), using unmodified QNN. The latter also has
  `qnn3/qnn_checks.json` and `qnn3/qnn_reference.log`.

The tested standalone SO SHA256 is
`0f212db9b3e56ad29b498deb38e14aac5ed319642a059c1ecc11028331e6b112`;
the 42-case UINT16 output SHA256 is
`cc313d8ab35342689c1e651cdd0185cfb5c17f70be1ee91124fdc5ba78169301`.

The original QNN v79 skel SHA256 is
`5d6900ed5c2edb2c81517adf251ca34a0eae43d41c7aeb799d855c66c84f0400`.
Only this SDK/architecture and the stated fixed tile are verified. Larger
shapes, cross-version QNN behavior and production backend integration remain
outside this minimal example.
