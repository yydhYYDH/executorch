#!/usr/bin/env python3
"""Measure the per-layer range constants the encoder export scales against.

Runs the model once over its example WAV and records, for each of the 50 encoder
blocks, the maximum absolute value of (a) the block input, (b) the block output
as it leaves ``forward_chunk``, and (c) the feed-forward output.  These become
``fc_inmag.json`` (M), ``alpha_fc.json`` (alpha) and ``ffn_mag_fc.json`` (betas)
in ``--calib-dir``.  They are model constants: regenerating them on the same
FunASR checkpoint reproduces the shipped ``encp_one_min.pte`` bit for bit.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    args = parser.parse_args()
    args.calib_dir.mkdir(parents=True, exist_ok=True)
    wav = args.wav or (args.model_dir / "example" / "asr_example.wav")

    from funasr import AutoModel

    torch.set_grad_enabled(False)
    model = AutoModel(model=str(args.model_dir), device="cpu", disable_update=True)
    net = model.model.eval()
    blocks = list(net.encoder.encoders0) + list(net.encoder.encoders)
    print(f"encoder blocks: {len(blocks)}", flush=True)

    n = len(blocks)
    input_mag = [0.0] * n
    output_mag = [0.0] * n
    ffn_mag = [0.0] * n

    for j, block in enumerate(blocks):
        chunk = block.forward_chunk
        ff = block.feed_forward.forward

        def capture_chunk(j=j, orig=chunk):
            def g(x, *a, **kw):
                input_mag[j] = max(input_mag[j], float(x.detach().float().abs().max()))
                out = orig(x, *a, **kw)
                t = out[0] if isinstance(out, (list, tuple)) else out
                if torch.is_tensor(t) and t.dim() == 3:
                    output_mag[j] = max(output_mag[j], float(t.detach().float().abs().max()))
                return out
            return g

        def capture_ffn(j=j, orig=ff):
            def g(*a, **kw):
                out = orig(*a, **kw)
                if torch.is_tensor(out):
                    ffn_mag[j] = max(ffn_mag[j], float(out.detach().float().abs().max()))
                return out
            return g

        block.forward_chunk = capture_chunk()
        block.feed_forward.forward = capture_ffn()

    result = model.generate(
        input=str(wav), disable_update=True, chunk_size=[0, 10, 5],
        encoder_chunk_look_back=4, decoder_chunk_look_back=1,
    )
    print(f"text {result[0]['text']}", flush=True)

    for name, values in (
        ("fc_inmag.json", input_mag),
        ("alpha_fc.json", output_mag),
        ("ffn_mag_fc.json", ffn_mag),
    ):
        path = args.calib_dir / name
        json.dump(values, open(path, "w"))
        top = max(values)
        over = sum(1 for v in values if v > 65504)
        print(f"{name}: min {min(values):.1f} max {top:.1f} >65504 {over}", flush=True)


if __name__ == "__main__":
    main()
