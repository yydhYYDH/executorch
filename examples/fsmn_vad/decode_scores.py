#!/usr/bin/env python3
"""Decode device FSMN-VAD scores into speech segments.

The runner writes either the full per-frame softmax scores or just the silence
probability (class 0). This replays those scores through FunASR's own
FsmnVADStreaming decoder, so the device and the reference share the exact same
post-processing; the device is only trusted for the neural encoder output.

Usage:
  python decode_scores.py --wav audio.wav --psil device_psil.bin --out seg.json
  python decode_scores.py --wav audio.wav --scores device_scores.bin --out seg.json
"""

import argparse
import json

import numpy as np
import torch
from funasr import AutoModel


def decode(wav, scores_path, psil_path, dtype, dynamic, model_dir="fsmn-vad"):
    m = AutoModel(model=model_dir, device="cpu", disable_update=True)
    inner = m.model
    kwargs = {} if dynamic else {"dynamic_silence": False}

    if not (scores_path or psil_path):
        return m.generate(input=wav, cache={}, disable_pbar=True, **kwargs)[0]["value"]

    captured = []
    real = inner.encoder
    orig = real.forward

    def probe(feats, cache=None):
        captured.append(feats.shape[1])
        return orig(feats, cache=cache)

    real.forward = probe
    m.generate(input=wav, cache={}, disable_pbar=True, **kwargs)
    real.forward = orig
    total = sum(captured)
    offsets = [0]

    if scores_path:
        data = np.fromfile(scores_path, dtype)
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        assert data.size % 248 == 0, data.size
        table = torch.from_numpy(data.reshape(-1, 248))
        assert table.shape[0] == total, (table.shape, total)

        def replay(feats, cache=None):
            t = feats.shape[1]
            i = offsets[0]
            offsets[0] += t
            return table[i : i + t].unsqueeze(0).to(feats.dtype)

    else:
        psil = np.fromfile(psil_path, np.float32)
        assert psil.size == total, (psil.size, total)

        def replay(feats, cache=None):
            t = feats.shape[1]
            i = offsets[0]
            offsets[0] += t
            s = torch.zeros(1, t, 248, dtype=feats.dtype)
            s[0, :, 0] = torch.from_numpy(psil[i : i + t])
            return s

    real.forward = replay
    result = m.generate(input=wav, cache={}, disable_pbar=True, **kwargs)[0]["value"]
    real.forward = orig
    assert offsets[0] == total
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="fsmn-vad")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--scores", default="")
    ap.add_argument("--psil", default="")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--dynamic", action="store_true", help="use the dynamic silence schedule")
    ap.add_argument("--compare", default="", help="reference segments .json to compare against")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    segments = decode(
        args.wav,
        args.scores,
        args.psil,
        np.float16 if args.dtype == "fp16" else np.float32,
        args.dynamic,
        args.model_dir,
    )
    print("SEGMENTS", len(segments))
    if args.compare:
        ref = json.load(open(args.compare))
        exact = ref == segments
        same = set(map(tuple, ref)) == set(map(tuple, segments))
        diffs = [
            (a, b) for a, b in zip(ref, segments) if a != b
        ]
        print(f"reference {len(ref)}  exact={exact}  same-set={same}  diffs={len(diffs)}")
        for a, b in diffs[:10]:
            print("  ref", a, "got", b)
    if args.out:
        json.dump(segments, open(args.out, "w"))


if __name__ == "__main__":
    main()
