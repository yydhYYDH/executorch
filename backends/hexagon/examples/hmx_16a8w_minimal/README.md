# Standalone HMX A16W8 tile

**Variable-parameter follow-up (2026-10-12):** the compute library now generates
conversion coefficients from FP32 `sa/sw/sq` and `zq`; the captured coefficient
pair is no longer embedded in `matmul.c`. New
`hmx_matmul_a16w8_tile_quantized` accepts activation/output zero points as well.
The original entry remains a wrapper using the old model's scale values.
See [parameter generation](../hmx_quant_params/README.md) for 71 device cases,
39 bit-exact historical QNN coefficient comparisons, API and range limits.
The fixed-parameter reference below describes the original tests.

**Device verified, bit-exact against its scalar contract.** On PJZ110 / SM8750 /
HTP v79, 2026-10-12: 12 cases, 24,576 uint16 outputs, zero mismatches in both
DSP and independent NumPy host references. This example runs HMX directly through
the Hexagon SDK launcher without loading QNN or ExecuTorch, retaining QNN's
captured conversion coefficients.

## Interface and arithmetic

Fixed shape `A[32,32] @ B[32,64]`, row-major:

* A: uint16, activation zero point 33401.
* B: **signed int8 native weights**, zero point 0. Subtract 128 if starting from
  QNN's exported uint8 weight container.
* Q: uint16, output zero point 29520.
* Workspace: 16 KiB VTCM, aligned to 2048 bytes, with HMX powered and locked.

For the fixed conversion words `0xa8403e00`, `0x39800212`:

```text
corr[n] = round_half_away(33401 * sum_k B[k,n] / 256)
t[r,n]  = floor(sum_k A[r,k] * B[k,n] / 256) - corr[n]
Q[r,n]  = clamp(floor(((256*t[r,n] + (t[r,n] != 0)) * 3183617
                       + 472328 * 2^26) / 2^30), 0, 65535)
```

The multiplier is `3183617 / 2^22`; the output bias is `472328 / 16 = 29520.5`.
The `(t != 0)` term models the conversion's sticky bit for this fixed mode and
tile range. Replacing this rule by `round(0.759033*t)+29520` creates one-count
errors near rounding boundaries. The integer rule was traced through QNN 2.41's
**host-side HMX emulation**, then checked against phone outputs. The device
outputs, rather than the emulation alone, establish the verified contract.

The original main/reference tests use fixed scales. The new parameter API
generates normal-branch coefficients for supported scales; see the linked
parameter-generation README. The older scope below does not describe that API.
Originally, the example did not generate tables for arbitrary scales
or claim a full-model byte comparison against QNN.

## Instruction sequence

The sequence follows QNN's captured A16W8 deep entry `0x2b4240`:

1. Pack A in row pairs and signed B in N32 blocks and K4 groups.
2. Load all four conversion banks with address operands `base+0x000`,
   `base+0x101`, `base+0x202`, `base+0x303`.
3. Issue `activation.ub = mxmem(A,0x77c)` together with
   `weight.b = mxmem(B,0x7ff):deep`.
4. Convert using `0x401`, `0x120c`, store the first N32 block; then convert
   using `0x2401`, `0x320c` and store the second.
5. Conversions use `:2x2`; stores use ordinary `mxmem(out,0x700)=cvt`.
   Unpack row pairs with no extra 32-word offset.

All four conversion banks are overwritten each call. `mxclracc` alone does not
reset them. Tests alternate `0x5a` and `0xa5` workspace fills, exposing missing
output writes. The old example's `:dilate` load, `:2x2` store and extra unpack
offset left its first columns reading untouched memory. Original diagnostic and
ablations: [QNN_A16W8_DIAGNOSIS.md](../../QNN_A16W8_DIAGNOSIS.md).

## Reproduce

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_16a8w_minimal/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python backends/hexagon/examples/hmx_16a8w_minimal/run_device.py
```

Defaults: Hexagon SDK `/home/yydh/hexagon-tutorial/tools/hexagon-sdk`, tools
19.0.04. Build overrides: `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`,
`HMX_EXAMPLE_BUILD_DIR`. Device options: `--sdk`, `--build-dir`, `--host`,
`--remote-dir`.

The script verifies deployment hashes, waits for llama-server/llama-bench,
requires DSP main return value zero, retrieves all output tiles and compares
all elements with NumPy int64 arithmetic. Artifacts live in
`.tmp/scratch/mnv2/hmx_16a8w_minimal/example/`: `output.u16`, `checks.json`,
`device_hashes.txt`, `device.log`, `example.farf.log`, `example.asm`.

| Cases | Coverage | Mismatches |
|---|---|---:|
| 0..2 | original random weights, A at/near zero point | 0 / 6144 |
| 3 | full uint16 A and random signed int8 B | 0 / 2048 |
| 4..7 | A=0/65535, B=-128/127, saturation at both ends | 0 / 8192 |
| 8..9 | zero weights, constant/varying A | 0 / 4096 |
| 10..11 | rounding boundaries, t=-1218/+1218 | 0 / 4096 |

Output SHA256:
`e466f20655cf017889fd967d3d6614e68564ab9443e81edad111337980aadd37`.
Deployed library SHA256:
`2b25e687e570ee32dd6098cd0b855c04168706bd5ac7116fc5590ccdb06e9481`.
Each run records hashes for the library it deploys.

No simulator, performance comparison, arbitrary shape, multi-tile accumulation,
FP16A8W implementation or LPBQ scale processing is covered here.
