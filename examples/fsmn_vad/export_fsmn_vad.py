#!/usr/bin/env python3
"""Export the FunASR FSMN-Monophone VAD encoder to an ExecuTorch .pte.

The graph is the streaming encoder only: input [1, T, 400] LFR features plus
four per-layer memory caches [1, 128, 19, 1], output [1, T, 248] softmax scores
and the updated caches. The host runner computes fbank/LFR/CMVN and drives the
caches; the VAD decision state machine is not part of the graph.

The DSP HMX matmul aborts (0x8000040d) on an output width of 250 with more than
one row, so the 250-wide linear layers are zero-padded to 256. The extra
rows/columns carry zero weight and zero activation, so the scores are unchanged.
"""

import argparse
from pathlib import Path

import torch
from funasr.models.fsmn_vad_streaming.encoder import FSMN, FSMNExport

from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import to_edge, to_edge_transform_and_lower

CFG = dict(
    input_dim=400,
    input_affine_dim=140,
    fsmn_layers=4,
    linear_dim=250,
    proj_dim=128,
    lorder=20,
    rorder=0,
    lstride=1,
    rstride=0,
    output_affine_dim=140,
    output_dim=248,
    use_softmax=True,
)


def pad_linear(lin, out_pad=0, in_pad=0):
    if in_pad:
        lin.weight.data = torch.cat(
            [lin.weight.data, torch.zeros(lin.weight.shape[0], in_pad)], dim=1
        )
    if out_pad:
        lin.weight.data = torch.cat(
            [lin.weight.data, torch.zeros(out_pad, lin.weight.shape[1])], dim=0
        )
        if lin.bias is not None:
            lin.bias.data = torch.cat([lin.bias.data, torch.zeros(out_pad)])
    lin.in_features = lin.weight.shape[1]
    lin.out_features = lin.weight.shape[0]


def pad_linear_dim(m, extra=6):
    pad_linear(m.in_linear2.linear, out_pad=extra)
    for block in m.fsmn:
        pad_linear(block.linear.linear, in_pad=extra)
        pad_linear(block.affine.linear, out_pad=extra)
    pad_linear(m.out_linear1.linear, in_pad=extra)


def build_export(model_dir, pad=True):
    m = FSMN(**CFG)
    sd = torch.load(Path(model_dir) / "model.pt", map_location="cpu")
    sd = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    if pad:
        pad_linear_dim(m)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return FSMNExport(m).eval()


def export(model_dir, chunk, out, heighten=True):
    exp = build_export(model_dir)
    x = torch.randn(1, chunk, CFG["input_dim"])
    caches = tuple(torch.zeros(1, CFG["proj_dim"], CFG["lorder"] - 1, 1) for _ in range(4))
    ep = torch.export.export(exp, (x, *caches), strict=False)
    if heighten:
        lowered = to_edge_transform_and_lower(ep, partitioner=[HexagonPartitioner()])
        prog = lowered.exported_program()
        nodes = [n for n in prog.graph.nodes if n.op == "call_function"]
        delegates = [n for n in nodes if "call_delegate" in str(n.target)]
        assert len(delegates) == 1, f"expected one delegate, got {len(delegates)}"
        pte = lowered.to_executorch()
    else:
        pte = to_edge(ep).to_executorch()
    Path(out).write_bytes(pte.buffer)
    print(f"wrote {out}: {len(pte.buffer)} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("fsmn_vad_enc.pte"))
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="write a portable, non-delegated .pte for host validation instead",
    )
    args = ap.parse_args()
    export(args.model_dir, args.chunk, args.out, heighten=not args.cpu)


if __name__ == "__main__":
    main()
