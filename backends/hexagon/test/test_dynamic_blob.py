import struct

import pytest
import torch

from executorch.backends.hexagon.hexagon_backend import HexagonBackend
from executorch.exir import to_edge
from torch.export import Dim, export


def test_dynamic_mm_emits_sequence_patches():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.float16))

        def forward(self, value):
            return value @ self.weight

    program = export(
        Linear().eval(),
        (torch.randn(3, 4, dtype=torch.float16),),
        dynamic_shapes=({0: Dim("sequence", min=1, max=16)},),
    )
    blob = HexagonBackend.preprocess(
        to_edge(program).exported_program(), []
    ).processed_bytes

    trailer_magic = (0x44594E48).to_bytes(4, "little")
    trailer_offset = blob.find(trailer_magic)
    assert trailer_offset >= 0
    header = struct.unpack_from("<7I", blob, trailer_offset)
    assert header == (0x44594E48, 3, 0, 0, 16, 3, 3)
    patches = [
        struct.unpack_from("<4i", blob, trailer_offset + 28 + index * 16)
        for index in range(header[5])
    ]
    assert patches == [(0, 2, 1, 0), (0, 20, 8, 0), (0, 22, 4, 0)]
    n_layouts = struct.unpack_from("<I", blob, trailer_offset + 28 + 3 * 16)[0]
    assert n_layouts == 2


@pytest.mark.parametrize("operator", ["mm", "addmm"])
@pytest.mark.parametrize("width", [64, 256])
def test_dynamic_matmul_preserves_fallback_layout_and_element_counts(operator, width):
    from executorch.backends.hexagon.test.blob_interpreter import read_blob

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(width, width, dtype=torch.float16)
            )
            self.register_buffer("bias", torch.zeros(width, dtype=torch.float16))

        def forward(self, value):
            if operator == "addmm":
                return torch.addmm(self.bias, value, self.weight, beta=0)
            return torch.mm(value, self.weight)

    program = export(
        Linear().eval(),
        (torch.randn(4, width, dtype=torch.float16),),
        dynamic_shapes=({0: Dim("rows", min=2, max=8)},),
    )
    blob = HexagonBackend.preprocess(
        to_edge(program).exported_program(), []
    ).processed_bytes
    _, commands = read_blob(blob)
    command = next(command for command in commands if command.type == 38)
    assert bool(command.params[26] & 1) == (width == 256)

    trailer_offset = blob.find((0x44594E48).to_bytes(4, "little"))
    header = struct.unpack_from("<7I", blob, trailer_offset)
    patches = [
        struct.unpack_from("<4i", blob, trailer_offset + 28 + index * 16)
        for index in range(header[5])
    ]
    for rows in (2, 4, 8):
        params = command.params.copy()
        for op_index, param_index, scale, add in patches:
            if op_index == commands.index(command):
                params[param_index] = rows * scale + add
        assert params[2] == rows
        packed = struct.pack("<27i", *params[1:])
        assert struct.unpack_from("<3q", packed, 76) == (
            rows * width,
            rows * width,
            width * width,
        )
