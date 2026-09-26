"""Positive controls: the five watched ops at geometries that MUST delegate.

A refusal list is worthless without the other half. Every control here is the
same op as one of the ten, at a geometry the support table says delegates, so a
partitioner that refused everything would fail this file rather than pass it.
The control is a real lowering with a real blob decoded: the command number is
read out of the blob, not inferred from a True verdict.
"""

from __future__ import annotations

import json

import torch

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from blob_interpreter import read_blob

DELEGATE = torch.ops.higher_order.executorch_call_delegate
OPNAMES = {v: k for k, v in vars(hexagon_ops).items() if k.startswith("DSP_OP_")}


def is_delegate(n) -> bool:
    return n.target is DELEGATE or "call_delegate" in str(n.target)


def decode(program, inputs):
    exported = torch.export.export(program, inputs)
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[HexagonPartitioner()]
    ).exported_program()
    gm = lowered.graph_module
    call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    delegates = [n for n in call_nodes if is_delegate(n)]
    portable = [n for n in call_nodes if not is_delegate(n)]
    cmds = []
    for d in delegates:
        sub = gm.get_submodule(str(d.args[0].target))
        _h, ops = read_blob(bytes(sub.processed_bytes))
        cmds.extend(OPNAMES.get(o.type, str(o.type)) for o in ops)
    return {
        "delegates": len(delegates),
        "portable": len(portable),
        "portable_targets": sorted(
            (getattr(n.target, "__name__", str(n.target))) for n in portable
        ),
        "commands": cmds,
    }


class AddF16(torch.nn.Module):
    def forward(self, a, b):
        return a + b


class SubF16(torch.nn.Module):
    def forward(self, a, b):
        return a - b


class CatF16(torch.nn.Module):
    def forward(self, a, b):
        return torch.cat([a, b], dim=-1)


class IndexF16(torch.nn.Module):
    """A row gather in the row the table documents: axis 0, one index, fp16 table.

    The table is a buffer, because `gather_table` requires a value the export can
    read: a method input has no bytes to tile at export, and that is a different
    refusal from the one the ten nodes hit.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randn(16, 8, dtype=torch.float16))

    def forward(self, idx):
        return torch.index_select(self.table, 0, idx)


class IndexTensorF16(torch.nn.Module):
    """`aten.index.Tensor` itself, at the geometry its row admits."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randn(16, 8, dtype=torch.float16))

    def forward(self, idx):
        return self.table[idx]


class IndexTensorI64(torch.nn.Module):
    """The same gather over an int64 table: a CONSTANT table, one index, axis 0.

    Every condition `gather_table` checks is satisfied except the width, so this
    isolates the width clause from the constant, the index and the axis.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.randint(0, 9, (16, 8), dtype=torch.int64))

    def forward(self, idx):
        return self.table[idx]


def main():
    f16 = torch.float16
    i32 = torch.int32
    cases = {
        "add_f16": (AddF16(), (torch.randn(4, 8, dtype=f16), torch.randn(4, 8, dtype=f16))),
        "sub_f16": (SubF16(), (torch.randn(4, 8, dtype=f16), torch.randn(4, 8, dtype=f16))),
        "cat_f16": (
            CatF16(),
            (torch.randn(2, 4, dtype=f16), torch.randn(2, 4, dtype=f16)),
        ),
        "index_select_f16_table_i32": (
            IndexF16(),
            (torch.tensor([0, 3, 5], dtype=i32),),
        ),
        "index_tensor_f16_table_i32": (
            IndexTensorF16(),
            (torch.tensor([0, 3, 5], dtype=i32),),
        ),
        "index_tensor_f16_table_i64": (
            IndexTensorF16(),
            (torch.tensor([0, 3, 5], dtype=torch.int64),),
        ),
        "index_tensor_int64_table_i64": (
            IndexTensorI64(),
            (torch.tensor([0, 3, 5], dtype=torch.int64),),
        ),
    }
    out = {}
    for name, (mod, inputs) in cases.items():
        out[name] = decode(mod, inputs)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
