# QNN quantized matmul standalone-example status

Checked against the example sources and device evidence on 2026-10-12.
“No capture needed” means the example can construct its own inputs and parameters
and run without a capture session. It does not automatically mean arbitrary
QNN model outputs are bit-exact.

| Configuration | Example | Standalone phone numerics | Captured numeric constants needed? | Remaining scope |
|---|---|---|---|---|
| `use_16a16w` | [hmx_16a16w_minimal](examples/hmx_16a16w_minimal/README.md) | 3 cases passed | **No: shared encoder now generates the pair** | Extreme shifts/other modes; per-channel preparation |
| `use_16a8w` | [hmx_16a8w_minimal](examples/hmx_16a8w_minimal/README.md) | 12 cases / 24,576 Q passed | **No: shared encoder now generates the pair** | Extreme shifts/other modes; per-channel preparation |
| `use_16a4w` | [hmx_int4_minimal](examples/hmx_int4_minimal/README.md), A16W4 branch | 92 parameterized cases / 188,416 Q/Y exact; fresh QNN model / 2,048 Q/Y exact | **No; shared encoder generates coefficients** | Extreme shifts/other modes; bias/per-channel/shapes |
| `use_16a4w_block` | **[hmx_lpbq_minimal](examples/hmx_lpbq_minimal/README.md)** | 12 cases / 24,576 Q/Y passed; old random QNN / 16,384 Q/Y and 512 table words exact; 3 new randomized N128 models / 12,288 Q/Y exact | **No; generates coefficients from logical scales/zero points** | N64 QNN preparation exposes abnormal padded channel scales; zero-scale channels/extreme modes unsupported |
| `use_8a8w` | [hmx_int8_minimal](examples/hmx_int8_minimal/README.md) | 92 parameterized cases / 376,832 Q/Y exact; fresh QNN model / 4,096 Q/Y exact | **No; shared encoder generates coefficients** | Extreme shifts/other modes; bias/per-channel/shapes |
| `use_8a4w` | [hmx_int4_minimal](examples/hmx_int4_minimal/README.md), A8W4 branch | 92 parameterized cases / 753,664 Q/Y exact; fresh QNN model / 8,192 Q/Y exact | **No; shared encoder generates coefficients** | Extreme shifts/other modes; bias/per-channel/shapes |
| `use_16a2w` | **[hmx_16a2w_minimal](examples/hmx_16a2w_minimal/README.md)** | 42 cases / 86,016 Q/Y exact; two original QNN fixtures / 4,096 Q/Y exact | **No; per-channel encoder generates coefficients** | Fixed M32/K32/N64; extreme shifts/other modes; bias/larger shapes |
| `use_fp16a8w` | [hmx_fp16a8w_minimal](examples/hmx_fp16a8w_minimal/README.md) | 10 cases / 20,480 FP16/FP32 passed | No | Full QNN model comparison; native mixed-precision options |

Thus 8/8 configurations have a corresponding standalone kernel-family example.
All eight run without a QNN runtime or a new capture. All eight now generate
or choose their own numeric parameters. A16W8 and A16W16 have new parameter
APIs with generated coefficient words; the common encoder matches all 39
historical QNN per-tensor coefficient pairs, and 71 variable-parameter tile
cases passed on the DSP. A8W8/A8W4/A16W4 also have parameter APIs accepting
`sa/sw/sq/za/zw/zq`; 276 variable-parameter tile cases and three fresh original
QNN model comparisons passed (1,318,912 and 14,336 Q/Y values respectively). See [the encoder README](examples/hmx_quant_params/README.md). That distinction is separate from whether the example
is general over shapes, scales or conversion modes. A16W2 now confirms the
native `weight.c:deep` path and supports per-channel scales with the observed
FP32 layer-normalization order; see its README for the fixed-tile contract
and separate runtime-capture / standalone-device evidence.

The old [hmx_blockwise_minimal](examples/hmx_blockwise_minimal/README.md) remains
a passing **independent per-block algorithm**, not a QNN LPBQ reproduction.
Its nibble-weight HMX calls convert/saturate each block, then the CPU adds
partials. QNN LPBQ instead expands INT4×block multiplier into byte weights,
accumulates all blocks, and converts once. The new example implements that
route and includes a block-cancellation case that tests this distinction.

On QNN 2.41 v79, the LPBQ dispatch is `0x2b25c0`, shared with A16W8's byte
kernel; it uses `weight.b` with selector `0x7ff`. The new implementation uses
the corresponding `:deep` packets and `cvt.uh:2x2`. It does not reinterpret
packed INT4 nibbles as byte weights.

The optional new QNN export/run scripts use an unmodified skel and require no
capture. Read the [LPBQ README](examples/hmx_lpbq_minimal/README.md) for commands,
layouts, conversion equations, device artifacts and the N64 preparation
boundary. No production backend or performance change is claimed here.
