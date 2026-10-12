#!/usr/bin/env python3
"""Export the Paraformer streaming predictor (CIF) to a Hexagon .pte.

The predictor buffer is a single fp16 vector:
``OFF_H=7680  OFF_CA=7681  OFF_CH=8193  OFF_FL=8194  BUFN=8195``.
It returns the embeds (8192), the per-slot mask (16), the residual alpha (1)
and the hidden state (512); the device returns these as a mixed fp32/fp16 set,
so the runner infers each dtype from byte-count / element-count, never from the
input dtype.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from executorch.backends.hexagon.default_fusion import default_fusion_passes  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402

DT = torch.float16
NF, T, CD, K = 15, 10, 512, 16
OFF_H = NF * CD
OFF_CA = OFF_H + 1
OFF_CH = OFF_CA + CD
OFF_FL = OFF_CH + 1
BUFN = OFF_FL + 1


class Pred(nn.Module):
    def __init__(self, cp):
        super().__init__()
        self.cp = cp

    def forward(self, buf):
        cp = self.cp
        h = buf[0, 0:OFF_H].view(1, NF, CD)
        ca = buf[0, OFF_H:OFF_CA].view(1, 1)
        ch = buf[0, OFF_CA:OFF_CH].view(1, 1, CD)
        fl = buf[0, OFF_FL]
        q = cp.pad(h.transpose(1, 2))
        o = cp.cif_output(torch.relu(cp.cif_conv1d(q)).transpose(1, 2))
        a = torch.relu(torch.sigmoid(o).squeeze(-1) * cp.smooth_factor - cp.noise_threshold)
        pos = torch.arange(1, NF + 1, dtype=DT).unsqueeze(0)
        keep = ((pos <= 10.0).to(DT) + (fl > 0.5).to(DT)).clamp(max=1.0).to(DT)
        a = (a * keep).to(DT)
        A = torch.cat((ca, a, (fl * cp.tail_threshold).view(1, 1)), dim=1)[0]
        H = torch.cat((ch, h, torch.zeros(1, 1, CD, dtype=DT)), dim=1)[0]
        M = NF + 2
        cs = torch.cumsum(A, 0)
        csx = torch.cat((torch.zeros(1, dtype=DT), cs))
        cw = torch.cat((torch.zeros(1, CD, dtype=DT), torch.cumsum(A.unsqueeze(1) * H, 0)))
        tgt = (torch.arange(K) + 1).to(DT) * cp.threshold
        hit = (cs.unsqueeze(0) >= tgt.unsqueeze(1)).to(DT)
        before = torch.cumsum(hit, dim=1) - hit
        sel = hit * (1.0 - (before > 0).to(DT))
        idx = torch.arange(M).to(DT).reshape(1, M)
        e = (sel * idx).sum(dim=1).to(torch.int64).clamp(max=M - 1)
        valid = sel.sum(dim=1) > 0
        ep = torch.cat((torch.full((1,), -1, dtype=e.dtype), e[:-1]))
        s = torch.minimum(ep + 1, e)
        frame = (cw[e] - cw[s]) + H[e] * (tgt - csx[e]).unsqueeze(1)
        st = (valid.to(DT) * (e + 1).to(DT)).max().reshape(1).to(torch.int64)
        si = st.reshape(1).to(torch.int64)
        den = (cs[-1] - csx.index_select(0, si)[0]).clamp(min=1e-6)
        num = cw[-1] - cw.index_select(0, si)[0]
        chs = (num / den).reshape(1, CD)
        cas = ((cs[-1] - valid.sum().to(DT) * cp.threshold).clamp(min=0)).reshape(1, 1)
        msk = valid.reshape(K)
        return (frame * msk.unsqueeze(1)).reshape(-1).to(DT), msk.reshape(-1).to(DT), \
            cas.reshape(-1).to(DT), chs.reshape(-1).to(DT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("pred15.pte"))
    parser.add_argument("--ref-dir", type=Path, default=None)
    args = parser.parse_args()

    from funasr import AutoModel

    torch.set_grad_enabled(False)
    am = AutoModel(model=str(args.model_dir), device="cpu", disable_update=True)
    cp = am.model.predictor.eval().to(DT)

    g = torch.Generator().manual_seed(5)
    buf = torch.zeros(1, BUFN, dtype=DT)
    buf[0, 0:OFF_H] = (torch.randn(1, NF, CD, generator=g) * 0.3).reshape(-1)
    buf[0, OFF_FL] = 1.0
    with torch.no_grad():
        r = Pred(cp)(buf)
    print(f"BUFN {BUFN} outputs {[int(t.numel()) for t in r]}", flush=True)
    if args.ref_dir:
        args.ref_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(("embeds", "mask", "cas", "chs")):
            r[i].numpy().tofile(args.ref_dir / f"pred15_ref_{name}.bin")

    ep = torch.export.export(Pred(cp), (buf.clone(),), strict=False)
    et = to_edge_transform_and_lower(
        ep, transform_passes=default_fusion_passes(),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    bts = bytes(et.to_executorch().buffer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "wb").write(bts)
    print(f"OK {args.out} {len(bts)} bytes", flush=True)


if __name__ == "__main__":
    main()
