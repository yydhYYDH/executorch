#!/usr/bin/env python3
"""Export the Paraformer streaming encoder to a Hexagon .pte.

This is the fused (mask-rewritten) export: the attention mask is folded into a
``where`` so the ``eq -> masked_fill`` chain never reaches the portable graph,
and the position encoding plus the per-chunk valid mask are precomputed on the
host and carried as graph inputs (``PREPARED_ENCODER``).  The result is the
``encp_one.pte`` build in ``encp_one_min.pte`` that ships on the device.

The per-layer range constants (alpha / input magnitude / ffn magnitude) come
from ``calibrate.py`` and are read out of ``--calib-dir``.  The traced examples
come from ``make_example_inputs.py`` (four real encoder chunks).  If the diag
inputs are absent this script regenerates them from the model's example WAV.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.backends.hexagon.default_fusion import default_fusion_passes  # noqa: E402

from make_example_inputs import _make_long_wav, build_encoder_examples  # noqa: E402


def load_encoder(model_dir, calib_dir):
    from funasr import AutoModel

    model = AutoModel(model=str(model_dir), device="cpu", disable_update=True)
    context = {
        "S": str(calib_dir).rstrip("/") + "/",
        "torch": torch,
        "json": json,
        "E": model.model.eval().encoder,
        "T": 10,
        "CH": [0, 10, 5],
        "LB": 4,
        "DT": torch.float16,
        "DBG": {"fix": True},
        "POSITION_DIGITS": True,
        "FUSE_MASK": True,
    }
    core = Path(__file__).with_name("hexagon_core.py").read_text()
    exec(compile(core, "hexagon_core.py", "exec"), context)
    return model, context


def build_samples(core, context, raw, chunk_index):
    encoder = context["W"]
    original_samples = {
        i: torch.cat(
            (
                torch.from_numpy(np.fromfile(raw[i], np.float16).copy()).view(1, -1),
                torch.tensor(
                    [[(i * 10 >> (10 * digit)) & 1023 for digit in range(4)]],
                    dtype=torch.float16,
                ),
            ),
            dim=1,
        )
        for i in chunk_index
    }
    samples, references = {}, {}
    for i, original_sample in original_samples.items():
        core["FUSE_MASK"] = False
        core["PREPARED_ENCODER"] = False
        references[i] = encoder(original_sample)
        digits = original_sample[0, -4:].to(torch.int64)
        position = sum(int(digits[d]) << (10 * d) for d in range(4))
        pos = position + torch.arange(1, 11).view(1, 10)
        encoding = encoder.enc.embed.encode(pos, 560, torch.float32).to(torch.float16)
        nvalid, ncache = original_sample[0, 1:3]
        past = (torch.arange(1, 51) > 50 - ncache).to(torch.float16)
        cur = torch.cat(
            (torch.ones(5, dtype=torch.float16),
             (torch.arange(1, 11) <= nvalid).to(torch.float16))
        )
        sample = torch.cat(
            (
                original_sample[:, :-4],
                encoding.reshape(1, -1),
                torch.cat((past, cur)).reshape(1, -1),
            ),
            dim=1,
        )
        samples[i] = sample
        core["FUSE_MASK"] = True
        core["PREPARED_ENCODER"] = True
        fused = encoder(sample)
        if not all(torch.equal(a, b) for a, b in zip(fused, references[i])):
            raise RuntimeError(f"mask rewrite changed host output at chunk {i}")
        print(f"original/fused exact {i}", flush=True)
    return samples, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--example-wav", type=Path, default=None)
    parser.add_argument("--diag-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("encp_one.pte"))
    parser.add_argument("--chunks", type=int, nargs="+", default=[0, 14, 28, 59])
    args = parser.parse_args()
    if args.diag_dir is None:
        args.diag_dir = args.out.parent / "diag"
    if args.example_wav is None:
        args.example_wav = args.diag_dir / "asr_example_6x.wav"
        args.diag_dir.mkdir(parents=True, exist_ok=True)
        _make_long_wav(args.model_dir, args.repeats, args.example_wav)

    torch.set_grad_enabled(False)
    import os

    os.environ["EXTRA_W2_ALL"] = "2.0"

    # Generate the traced inputs on a pristine model: load_encoder execs
    # hexagon_core, which monkeypatches every block's forward_chunk, so the
    # capture pass must finish before that happens.
    from funasr import AutoModel

    example_model = AutoModel(model=str(args.model_dir), device="cpu", disable_update=True)
    raw = build_encoder_examples(example_model, args.example_wav, args.chunks, args.diag_dir)
    del example_model

    model, context = load_encoder(args.model_dir, args.calib_dir)
    samples, references = build_samples(context, context, raw, args.chunks)

    exported = torch.export.export(context["W"], (samples[args.chunks[0]],), strict=False)
    for i, sample in samples.items():
        result = exported.module()(sample)
        if not all(
            torch.isfinite(a).all() and torch.equal(a, b)
            for a, b in zip(result, references[i])
        ):
            raise RuntimeError(f"host export changed output at {i}")
        print(f"original/exported exact {i}", flush=True)

    lowered = to_edge_transform_and_lower(
        exported,
        transform_passes=default_fusion_passes(),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(bytes(lowered.to_executorch().buffer))
    print(f"EXPORTED {args.out} {args.out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
