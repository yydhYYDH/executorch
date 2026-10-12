#!/usr/bin/env python3
"""Generate the traced example inputs the encoder and decoder exports need.

* Encoder: four real streaming chunks (default indices 0, 14, 28, 59) of the
  ``3 + 10*560 + 2*50*4*50*128 + 5*560`` fp16 input buffer, together with the
  running per-layer k/v cache the native encoder produces.  These are what
  ``export_encoder.py`` traces against.

* Decoder: a single ``15*512 + 16*512 + 16*512*26 + 32*5120 + 1`` fp16 buffer.
  Only its shape and homogeneity matter -- the graph is content independent.

The long WAV defaults to the model's own ``example/asr_example.wav`` repeated
six times, which is long enough to reach chunk 60.
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import torch


ENC_INPUT = 3 + 10 * 560 + 2 * 50 * 4 * 50 * 128 + 5 * 560
DEC_INPUT = 15 * 512 + 16 * 512 + 16 * 512 * 26 + 32 * 5120 + 1
CHUNK = [0, 10, 5]


def _flat_cache(opt):
    parts = []
    for entry in opt:
        parts.append(np.asarray(entry["k"].detach().cpu(), dtype=np.float16).reshape(-1))
        parts.append(np.asarray(entry["v"].detach().cpu(), dtype=np.float16).reshape(-1))
    return np.concatenate(parts)


def _make_long_wav(model_dir, repeats, out):
    src = model_dir / "example" / "asr_example.wav"
    with wave.open(str(src), "rb") as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        for _ in range(repeats):
            w.writeframes(frames)


def build_encoder_examples(model, wav, chunk_index, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = model.model.encoder.eval()
    captured = []
    original = encoder.forward_chunk

    def capture(*args, **kwargs):
        cache = kwargs.get("cache", args[2] if len(args) > 2 else None)
        xs = args[0].detach().clone()
        feats = cache["feats"].detach().clone()
        start_idx = int(cache["start_idx"])
        ncache = 0 if cache.get("opt") is None else int(cache["opt"][0]["k"].shape[2])
        result = original(*args, **kwargs)
        state = _flat_cache(cache["opt"]) if cache.get("opt") else np.zeros(
            2 * 50 * 4 * 50 * 128, dtype=np.float16
        )
        captured.append(
            {"xs": xs, "feats": feats, "start_idx": start_idx,
             "ncache": ncache, "state": state}
        )
        return result

    encoder.forward_chunk = capture
    try:
        audio = (np.fromfile(wav, dtype=np.int16).astype(np.float32) / 32768.0
                 if Path(wav).suffix.lower() == ".pcm" else str(wav))
        model.generate(
            input=audio, chunk_size=CHUNK,
            encoder_chunk_look_back=4, decoder_chunk_look_back=1, is_final=True,
        )
    finally:
        encoder.forward_chunk = original

    paths = {}
    for i in chunk_index:
        if i >= len(captured):
            raise RuntimeError(f"chunk {i} not reached; need a longer WAV (got {len(captured)})")
        c = captured[i]
        buf = np.zeros(ENC_INPUT, dtype=np.float16)
        n = c["xs"].shape[1]
        buf[0:3] = (c["start_idx"], n, c["ncache"])
        begin = 3
        buf[begin:begin + n * 560] = c["xs"].to(torch.float16).numpy().reshape(-1)
        begin += 10 * 560
        buf[begin:begin + c["state"].size] = c["state"]
        begin += 2 * 50 * 4 * 50 * 128
        buf[begin:] = c["feats"].to(torch.float16).numpy().reshape(-1)
        path = out_dir / f"enc_long_{i}.bin"
        buf.tofile(path)
        paths[i] = path
        print(f"wrote {path}")
    return paths


def build_decoder_input(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = np.zeros(DEC_INPUT - 1, dtype=np.float16)
    rng = np.random.default_rng(0)
    buf[: 15 * 512 + 16 * 512] = rng.standard_normal(15 * 512 + 16 * 512).astype(np.float16) * 0.1
    path = out_dir / "decoder_input.bin"
    buf.tofile(path)
    print(f"wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[0, 14, 28, 59])
    args = parser.parse_args()
    from funasr import AutoModel

    wav = args.wav
    if wav is None:
        wav = args.out_dir / "asr_example_6x.wav"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _make_long_wav(args.model_dir, args.repeats, wav)
    model = AutoModel(model=str(args.model_dir), device="cpu", disable_update=True)
    build_encoder_examples(model, wav, args.chunks, args.out_dir)
    build_decoder_input(args.out_dir)


if __name__ == "__main__":
    main()
