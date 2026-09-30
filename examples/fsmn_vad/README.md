# FSMN-VAD Runner

Runs the FunASR [FSMN-Monophone VAD](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
streaming encoder on device. The runner reads a 16 kHz mono 16-bit WAV or raw
PCM, computes fbank + LFR (m=5, n=1) + CMVN on the device, drives the encoder's
four per-layer memory caches over fixed 20-frame chunks, and writes the
per-frame silence probability (class 0) and optionally the whole 248-way
softmax row. The VAD decision state machine is applied on the host with FunASR's
own decoder, so the device is compared on the same post-processing path as the
reference.

## Model

The encoder is a small FSMN: two input affine layers, four
`BasicBlock`s (a 250->128 projection, a depthwise 20-tap left-memory Conv2d, a
128->250 affine and a ReLU) and two output affine layers followed by softmax.
The whole graph lowers to a **single Hexagon delegate** with no portable compute.

The graph signature is `(x, c0, c1, c2, c3) -> (probs, c0', c1', c2', c3')`
with `x` `[1, T, 400]`, each cache `[1, 128, 19, 1]` and `probs` `[1, T, 248]`.
The encoder is causal (left context only), so the per-frame output is
independent of the chunk size used to feed it.

### 250-width matmul workaround

The DSP HMX matmul aborts (`0x8000040d`) for an output width of 250 with more
than one row. Every 250-wide linear layer is therefore zero-padded to 256 at
export; the extra rows/columns carry zero weight and zero activation, so the
scores are bit-for-bit unchanged on the host. `export_fsmn_vad.py` applies this
by default.

## Export

```bash
python examples/fsmn_vad/export_fsmn_vad.py /path/to/fsmn-vad \
  --chunk 20 --out fsmn_vad_enc_T20.pte
```

This also verifies that the lowered graph contains exactly one Hexagon
delegate. Pass `--cpu` to write a portable, non-delegated `.pte` for host
validation.

## Assets

`prepare_assets.py` writes the 2 x 400 CMVN as float32.

```bash
python examples/fsmn_vad/prepare_assets.py MODEL_DIR ASSET_DIR
```

## Build

```bash
-DEXECUTORCH_BUILD_FSMN_VAD_RUNNER=ON
```

reuses the `kaldi-native-fbank` sources vendored under `examples/paraformer`.

## Run on device

```bash
LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH=$PWD/skel \
  ./fsmn_vad_runner \
  --model_path fsmn_vad_enc_T20.pte \
  --audio input.wav --cmvn cmvn.bin \
  --psil_out psil.bin
```

`--limit_chunks N` stops after N encoder invocations. `--scores_out` writes the
full `[frames, 248]` softmax rows.

## Decode

Replays the runner's scores through FunASR's `FsmnVADStreaming` with
`dynamic_silence=False` (the classic static thresholds), which reproduces the
offline `generate()` segments exactly:

```bash
python examples/fsmn_vad/decode_scores.py --model-dir MODEL_DIR \
  --wav input.wav --psil psil.bin --out segments.json
```
