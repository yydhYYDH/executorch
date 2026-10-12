#!/usr/bin/env python3
"""Export the Paraformer streaming decoder to a Hexagon .pte.

One graph, two call modes.  ``--first`` exports the chunk that seeds the
self-attention and source-attention caches; the default exports the steady
state, which reads the caches back in.  Both take the same single fp16 buffer
``15*512 + 16*512 + 16*512*26 + 32*5120 + 1`` and return 49 tensors: the logits
(134464, viewed as 16 x 8404) followed by 16 compacted self-attention caches and
32 source-attention cache planes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from executorch.backends.hexagon.default_fusion import default_fusion_passes  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402

MEM, TGT, FS, KV = 15 * 512, 16 * 512, 16 * 512 * 26, 32 * 5120
SIZE = MEM + TGT + FS + KV + 1


class Decoder16(torch.nn.Module):
    def __init__(self, dec, first):
        super().__init__()
        self.dec = dec
        self.first = first

    def forward(self, buf):
        dec, FIRST = self.dec, self.first
        f = buf[0]
        memory = f[:MEM].view(1, 15, 512)
        x = f[MEM:MEM + TGT].view(1, 16, 512)
        valid = f[-1]
        mask = (torch.arange(16, dtype=torch.float16) < valid).to(torch.float16).view(1, 16, 1)
        positions = torch.arange(26, dtype=torch.float16)
        start = 16 - valid
        source = (positions - start).clamp(min=0, max=25).to(torch.int64)
        cache_mask = (positions >= start).to(torch.float16).view(1, 1, 26)
        fs = [f[MEM + TGT + i * 512 * 26:MEM + TGT + (i + 1) * 512 * 26].view(1, 512, 26)
              for i in range(16)]
        opt = []
        base = MEM + TGT + FS
        for i in range(16):
            k = f[base + (2 * i) * 5120:base + (2 * i + 1) * 5120].view(1, 4, 10, 128)
            v = f[base + (2 * i + 1) * 5120:base + (2 * i + 2) * 5120].view(1, 4, 10, 128)
            opt.append({"k": k, "v": v})
        x = x * mask
        fs_out = []
        opt_out = []
        for i, layer in enumerate(dec.decoders):
            residual = x
            feed = layer.feed_forward(layer.norm1(x))
            fsmn_input = layer.norm2(feed) * mask
            attention, full_cache = layer.self_attn(fsmn_input, None, None if FIRST else fs[i])
            compact = torch.index_select(full_cache, 2, source) * cache_mask
            fs_out.append(compact)
            x = residual + layer.dropout(attention)
            residual = x
            cross_input = layer.norm3(x)
            cross, cache = layer.src_attn.forward_chunk(
                cross_input, memory, None if FIRST else opt[i], [0, 10, 5], 1
            )
            opt_out.extend((cache["k"], cache["v"]))
            x = (residual + cross) * mask
        for layer in dec.decoders3:
            x, memory, _, _ = layer.forward_chunk(x, memory)
        y = dec.output_layer(dec.after_norm(x))
        return (y.reshape(-1), *fs_out, *opt_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="decoder_input.bin from make_example_inputs.py")
    parser.add_argument("--first", action="store_true", help="export the seeding chunk")
    parser.add_argument("--out", type=Path, default=Path("decoder16_next.pte"))
    args = parser.parse_args()

    from funasr import AutoModel

    torch.set_grad_enabled(False)
    am = AutoModel(model=str(args.model_dir), device="cpu", disable_update=True)
    dec = am.model.decoder.eval().to(torch.float16)
    first = args.first

    buf = np.fromfile(args.input, np.float16)
    assert buf.size == SIZE - 1, f"expected {SIZE - 1} fp16 values, got {buf.size}"
    buf = np.concatenate([buf, np.array([3], np.float16)])
    if not first:
        buf[MEM + TGT:MEM + TGT + FS] = 0
        buf[MEM + TGT + FS:-1] = 0
    x = torch.from_numpy(buf.copy()).view(1, -1)

    module = Decoder16(dec, first)
    with torch.no_grad():
        eager = module(x)
    ep = torch.export.export(module, (x.clone(),), strict=False)
    with torch.no_grad():
        exported = ep.module()(x)
    assert len(eager) == len(exported) == 49
    assert all(torch.isfinite(a).all() and torch.equal(a, b) for a, b in zip(eager, exported))
    print(f"host eager/exported exact; first tokens "
          f"{eager[0].reshape(16, -1)[:3].argmax(-1).tolist()}", flush=True)

    et = to_edge_transform_and_lower(
        ep, transform_passes=default_fusion_passes(), partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    raw = bytes(et.to_executorch().buffer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "wb").write(raw)
    print(f"exported {args.out} {len(raw)}", flush=True)


if __name__ == "__main__":
    main()
