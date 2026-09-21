import struct

import torch
from torch.export import Dim, export

from executorch.backends.hexagon.hexagon_backend import HexagonBackend
from executorch.exir import to_edge


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
    assert patches == [(0, 2, 1, 0), (0, 20, 8, 0), (0, 21, 4, 0)]
    n_layouts = struct.unpack_from("<I", blob, trailer_offset + 28 + 3 * 16)[0]
    assert n_layouts == 2
