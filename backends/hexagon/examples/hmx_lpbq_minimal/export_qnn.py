import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from verify import make_cases, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / ".tmp/scratch/mnv2/hmx_lpbq_minimal/qnn_reference",
    )
    parser.add_argument("--case", type=int, choices=range(12), default=3)
    parser.add_argument("--seed", type=int, help="Use randomized logical parameters")
    parser.add_argument("--columns", type=int, choices=(64, 128), default=128)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    a, w, scales, sa, sq, za, zq, _ = make_cases()[args.case]
    cols = args.columns
    if cols == 128:
        w = np.tile(w, (1, 2))
        scales = np.tile(scales, (2, 1))
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        za = int(rng.integers(8000, 55000))
        zq = int(rng.integers(10000, 55000))
        sa = np.float32(rng.uniform(1e-5, 4e-5))
        sq = np.float32(rng.uniform(1e-5, 5e-5))
        a = (za + rng.integers(-4000, 4001, size=(32, 128))).astype("<u2")
        w = rng.integers(-8, 8, size=(128, cols), dtype=np.int8)
        scales = rng.uniform(1e-4, 0.1, size=(cols, 4)).astype("<f4")
    x = torch.from_numpy((a.astype(np.float32) - za) * sa)
    torch.manual_seed(1234)
    model = torch.nn.Linear(128, cols, bias=False).eval()
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=QcomChipset.SM8750
    )
    quantizer.set_default_quant_config(
        QuantDtype.use_16a4w_block, is_linear_per_channel=False
    )
    quantizer.set_block_size_map({"linear": (1, 32)})
    graph = torch.export.export(model, (x,), strict=True).module()
    graph = prepare_pt2e(graph, quantizer)
    graph(x)
    graph = convert_pt2e(graph)
    # Fix the logical quantized tensors, so QNN and HMX receive the same case.
    graph._buffers["_frozen_param0"] = torch.from_numpy(w.T.copy())
    graph._buffers["_scale0"] = torch.from_numpy(scales.copy())
    graph._buffers["_zero_point0"] = torch.zeros((cols, 4), dtype=torch.int32)
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == torch.ops.torchao.dequantize_affine:
            values = list(node.args)
            values[5] = -8
            node.args = tuple(values)
        elif node.target in (
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ):
            values = list(node.args)
            input_quant = values[0].op == "placeholder" or (
                values[0].target
                == torch.ops.quantized_decomposed.quantize_per_tensor.default
                and values[0].args[0].op == "placeholder"
            )
            values[1] = float(sa if input_quant else sq)
            values[2] = za if input_quant else zq
            node.args = tuple(values)
    graph.graph.lint()
    graph.recompile()
    (root / "model.qdq.txt").write_text(str(graph.graph))
    graph(x)
    compile_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8750,
        backend_options=generate_htp_compiler_spec(use_fp16=False),
        profile_level=0,
        shared_buffer=False,
    )
    edge = to_edge_transform_and_lower_to_qnn(graph, (x,), compile_specs)
    lowered = edge.exported_program().graph_module
    calls = [node for node in lowered.graph.nodes if node.op == "call_function"]
    delegates = [
        node
        for node in calls
        if node.target == torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(delegates) == 1
    assert all(node in delegates or "getitem" in str(node.target) for node in calls)
    (root / "model.pte").write_bytes(edge.to_executorch().buffer)
    x.numpy().tofile(root / "input.f32")
    np.save(root / "Wq.npy", w.T.copy())
    np.save(root / "sc.npy", scales)
    np.save(
        root / "expected.q.npy",
        reference((a, w, scales, sa, sq, za, zq, np.float32(0)))[0],
    )
    metadata = {
        "case": args.case,
        "seed": args.seed,
        "shape_mkn": [32, 128, cols],
        "sa": float(sa),
        "sq": float(sq),
        "za": za,
        "zq": zq,
        "delegate_count": 1,
        "pte_sha256": hashlib.sha256((root / "model.pte").read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((root / "input.f32").read_bytes()).hexdigest(),
    }
    (root / "logical.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
