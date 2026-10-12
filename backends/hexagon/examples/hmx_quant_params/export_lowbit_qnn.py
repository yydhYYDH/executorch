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
from verify_lowbit import make_cases, inputs, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / ".tmp/scratch/mnv2/hmx_quant_params/lowbit_qnn_reference",
    )
    parser.add_argument("--case", type=int, default=3)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    case = make_cases()[args.case]
    sa, sw, sq, za, zw, zq, kind = case
    a, raw = inputs(case)
    w = (raw.astype(np.int16) - zw).astype(np.int8)
    m, k = a.shape
    cols = w.shape[1]
    x = torch.from_numpy((a.astype(np.float32) - za) * np.float32(sa))
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    model = torch.nn.Linear(k, cols, bias=False).eval()
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=QcomChipset.SM8750
    )
    quantizer.set_default_quant_config(
        [QuantDtype.use_8a8w, QuantDtype.use_8a4w, QuantDtype.use_16a4w][kind],
        is_linear_per_channel=False,
    )
    graph = torch.export.export(model, (x,), strict=True).module()
    graph = prepare_pt2e(graph, quantizer)
    graph(x)
    graph = convert_pt2e(graph)
    graph._buffers["_frozen_param0"] = torch.from_numpy(w.T.copy())
    for node in graph.graph.nodes:
        if node.op != "call_function" or node.target not in (
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ):
            continue
        values = list(node.args)
        if values[0].op == "get_attr":
            values[1], values[2] = float(sw), 0
            values[3], values[4] = (-128, 127) if kind == 0 else (-8, 7)
        else:
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
    expected_q, expected_y = reference(case)
    np.save(root / "expected.q.npy", expected_q)
    np.save(root / "expected.y.npy", expected_y)
    metadata = {
        "case": args.case,
        "parameters": list(case),
        "shape_mkn": [m, k, cols],
        "delegate_count": 1,
        "pte_sha256": hashlib.sha256((root / "model.pte").read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((root / "input.f32").read_bytes()).hexdigest(),
    }
    (root / "logical.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
