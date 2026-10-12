#!/usr/bin/env python3
"""Export CAM++ (speaker verification) to a CPU-runnable .pte with XNNPACK.

CAM++ runs on the host CPU tier, not the DSP: the batch-norm fold that has to
happen before an fp16 DSP arena breaks this graph, and the model is small.  The
checkpoint is FunASR's ``iic/speech_campplus_sv_zh-cn_16k-common``
(``campplus_cn_common.bin``); download it with modelscope and point
``--checkpoint`` at it.
"""

import argparse
import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from campplus_model import CAMPPlus  # noqa: E402

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: E402
    XnnpackPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("campplus_cpu.pte"))
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    state = torch.load(args.checkpoint, map_location="cpu")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(state, strict=True)
    model = model.eval()

    torch.manual_seed(0)
    x = torch.randn(1, args.frames, 80)
    with torch.no_grad():
        ref = model(x).clone()
    print("params", sum(p.numel() for p in model.parameters()))
    print("reference", tuple(ref.shape), "absmax", float(ref.abs().max()),
          "finite", int(torch.isfinite(ref).sum()), "/", ref.numel())

    exported = torch.export.export(model, (x,))
    with contextlib.redirect_stdout(io.StringIO()):
        lowered = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    gm = lowered.exported_program().graph_module
    delegates = [n for n in gm.graph.nodes
                 if getattr(n.target, "__name__", "") == "executorch_call_delegate"]
    print("delegates:", len(delegates))
    portable = [n for n in gm.graph.nodes
                if n.op == "call_function"
                and getattr(n.target, "__name__", "") != "executorch_call_delegate"]
    print("portable ops:", Counter(str(n.target) for n in portable).most_common())

    with contextlib.redirect_stdout(io.StringIO()):
        pte = bytes(lowered.to_executorch().buffer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pte)
    x.numpy().astype(np.float32).tofile(args.out.with_name("campplus_in0.bin"))
    ref.numpy().astype(np.float32).tofile(args.out.with_name("campplus_ref.npy"))
    print("wrote", args.out, len(pte), "B")


if __name__ == "__main__":
    main()
