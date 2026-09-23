# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The row gathers: the tiled table the kernel reads, and the command that reads it.

`htp_ops_shared_gather` is the one kernel that can read a table as large as an
LLM's token embedding, and its fp16 path is not a row gather over a row-major
table: the table is a grid of 32x32 tiles with the column pairs inside a tile
first. So there are two things to check, and they are checked separately. The
packing is checked against a transcription of the kernel's own addressing, and
against a handful of offsets derived from that source by hand, because a packer
and an un-tiler that are wrong in the same way agree with each other perfectly.
The command is then checked end to end: a real export through the real
partitioner produces a delegate whose blob carries a SHARED_GATHER command, and
the host interpreter runs that command against the reference.

What none of this covers is the DSP: the tiling here is a second implementation
of what shared_gather_ops.cc does, so agreement means the source was read the
same way twice, not that the hardware reads it that way. The README's
"unverified on device" list is the rest of it.
"""

import os
import pathlib
import struct
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import blob_interpreter  # noqa: E402
from blob_interpreter import execute, read_blob, SHARED_GATHER  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    pack_shared_gather_table,
    SHARED_GATHER_FP16,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import to_edge_transform_and_lower  # noqa: E402
from torch.export import Dim, export  # noqa: E402

#: How wide the kernel's tile grid is, and what one tile costs it.
TILE = 32
TILE_ELEMENTS = TILE * TILE

BACKEND_ID = "HexagonBackend"


def _tiled_bytes(oc: int, ic: int) -> int:
    """The table's size once tiled, which is oc * ic * 2 only when both divide 32."""
    return -(-oc // TILE) * -(-ic // TILE) * TILE_ELEMENTS * 2


def _distinct(oc: int, ic: int) -> np.ndarray:
    """An fp16 table whose entries are all different, so a wrong one is visible.

    Built from bit patterns in the normal range rather than sampled values: only
    2048 half floats below one are distinct, so sampling a table this size would
    repeat values and hide a packer that moved them.
    """
    count = oc * ic
    assert count <= 31744
    patterns = np.arange(0x3C00, 0x3C00 + count, dtype=np.uint16)
    return patterns.view(np.float16).reshape(oc, ic)


def _delegates(program):
    """The delegate submodules of a lowered graph, in node order."""
    graph_module = program.graph_module
    return [
        graph_module.get_submodule(str(node.args[0].target))
        for node in graph_module.graph.nodes
        if "call_delegate" in getattr(node.target, "__name__", "")
    ]


def _lowered(model, inputs, dynamic_shapes=None):
    """The lowered program and the blob of every delegate it produced."""
    program = export(model, tuple(inputs), dynamic_shapes=dynamic_shapes)
    lowered = to_edge_transform_and_lower(
        program, partitioner=[HexagonPartitioner()]
    ).exported_program()
    delegates = _delegates(lowered)
    for delegate in delegates:
        assert delegate.backend_id == BACKEND_ID
    return lowered, [bytes(delegate.processed_bytes) for delegate in delegates]


def _only_command(blob: bytes):
    """The single SHARED_GATHER command of a blob, with its header."""
    header, commands = read_blob(blob)
    assert header.n_ops == 1
    assert [command.type for command in commands] == [SHARED_GATHER]
    return header, commands[0]


def _reference(module, tokens: np.ndarray) -> np.ndarray:
    """What torch returns, at the width the arena holds."""
    with torch.no_grad():
        return module(torch.from_numpy(tokens)).numpy().astype(np.float16).reshape(-1)


def _commands(blobs):
    return [command for blob in blobs for command in read_blob(blob)[1]]


def _no_gather(blobs) -> bool:
    """Whether nothing emitted was a row gather.

    A refusal has to leave the graph runnable, and the ops around the gather may
    still be delegated, so what a test can ask for is that no command reads a
    table rather than that the delegate is absent.
    """
    return all(command.type != SHARED_GATHER for command in _commands(blobs))


@pytest.mark.parametrize(
    "oc, ic", [(64, 8), (33, 8), (64, 33), (1, 1), (5, 40), (96, 96), (40, 70), (2, 3)]
)
def test_the_packed_table_is_the_one_the_kernel_reads(oc, ic):
    """Pack, then un-tile the way the kernel reads, and get the table back.

    The two sides are written differently on purpose: the packer is a vectorized
    reshape and transpose, the un-tiler a loop copied off shared_gather_ops.cc
    line for line, including the short last column tile and its odd tail.
    """
    table = _distinct(oc, ic)
    packed = pack_shared_gather_table(table, oc, ic)
    assert len(packed) == _tiled_bytes(oc, ic)
    flat = np.frombuffer(packed, dtype=np.float16)
    read_back = blob_interpreter.untile_shared_gather(flat, oc, ic)
    assert np.array_equal(read_back, table)


@pytest.mark.parametrize("oc, ic", [(64, 16), (33, 35)])
def test_the_tile_offsets_are_the_ones_the_kernel_computes(oc, ic):
    """Offsets derived by hand from the kernel, not from the packer.

    shared_gather_ops.cc:296-311 reads element (index, c) of the tile at
    (index // 32, c // 32) at `((c % 32) // 2) * 64 + (index % 32) * 2 +
    ((c % 32) & 1)` inside that tile. Every term is spelled out here so a packer
    that transposed the two inner axes, or lifted the pair index out of the row,
    disagrees with this rather than with a second copy of itself.
    """
    table = _distinct(oc, ic)
    flat = np.frombuffer(pack_shared_gather_table(table, oc, ic), dtype=np.float16)
    tile_columns = -(-ic // TILE)
    for row, column in ((0, 0), (31, 31), (32, 0), (33, 5), (1, 34)):
        if row >= oc or column >= ic:
            continue
        inside = ((column % TILE) // 2) * 64 + (row % TILE) * 2 + (column % TILE) % 2
        at = ((row // TILE) * tile_columns + column // TILE) * TILE_ELEMENTS + inside
        assert flat[at] == table[row, column], (row, column, at)


def test_the_tiled_table_costs_the_padded_tile_grid():
    """The kernel pads to whole 32x32 tiles, and the file pays for the padding.

    1024 elements a tile is the kernel's addressing, not a choice: it reads the
    last column tile short but still strides past a whole one. So a table whose
    sides are not multiples of 32 is stored larger than its row-major size, which
    is what makes this a cost to state rather than a bug to fix.
    """
    assert _tiled_bytes(64, 64) == 64 * 64 * 2
    assert _tiled_bytes(64, 8) == 64 * 8 * 2 * 4
    assert _tiled_bytes(33, 35) == 2 * 2 * TILE_ELEMENTS * 2


class _Embedding(torch.nn.Module):
    def __init__(self, vocab=64, dim=8, padding_idx=None, dtype=torch.float32):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim, padding_idx=padding_idx).to(dtype)

    def forward(self, tokens):
        return self.emb(tokens)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_an_embedding_reaches_the_delegate_and_gathers_the_rows(dtype):
    """The whole path: export, partition, emit, and the numbers that come out.

    The positive assertions are the point. A lowering that silently kept the
    embedding on a portable kernel would still produce a program that runs, and
    one whose numbers still match the reference, so the delegate and the command
    are checked for before anything is compared.
    """
    torch.manual_seed(0)
    model = _Embedding(dtype=dtype).eval()
    tokens = np.array([3, 63, 0, 17, 5], dtype=np.int32)

    lowered, blobs = _lowered(model, (torch.from_numpy(tokens),))
    assert len(blobs) == 1
    header, command = _only_command(blobs[0])

    # The operands are the indices and then the table, which is the order
    # execute_command.cc:806-811 hands htp_ops_shared_gather its two pointers.
    indices, table = command.inputs
    assert indices.space is blob_interpreter.B.TensorSpace.INPUT
    assert indices.size == tokens.nbytes
    assert table.space is blob_interpreter.B.TensorSpace.WEIGHTS
    assert table.size == _tiled_bytes(64, 8)
    # selectSize, ic, oc, bytes, isInt4.
    assert command.params == [len(tokens), 8, 64, 2, SHARED_GATHER_FP16]

    # The row-major copy `preprocess` stored for the placeholder is not reachable
    # from any command, so it is not in the file: one table, tiled, not two.
    assert header.weights_bytes == _tiled_bytes(64, 8)

    got = np.frombuffer(execute(blobs[0], [tokens])[0], dtype=np.float16)
    assert np.array_equal(got, _reference(model, tokens))
    # The functional form is what the module's forward calls, so this is the same
    # reference arrived at twice.
    functional = torch.nn.functional.embedding(
        torch.from_numpy(tokens), model.emb.weight
    )
    assert np.array_equal(
        got, functional.detach().numpy().astype(np.float16).reshape(-1)
    )


def test_one_table_read_twice_is_stored_once():
    """Two gathers of the same table share one copy of its tiled bytes.

    A model can embed twice -- two token runs, or a prompt and a continuation --
    and the second command has to find the table the first one stored rather than
    a second copy of it, since it is the same bytes either way.
    """

    class _Twice(torch.nn.Module):
        def __init__(self, vocab=64, dim=8):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, dim)

        def forward(self, first, second):
            return self.emb(first) + self.emb(second)

    torch.manual_seed(0)
    model = _Twice().eval()
    first = np.array([1, 2, 3], dtype=np.int32)
    second = np.array([63, 0, 8], dtype=np.int32)
    _, blobs = _lowered(model, (torch.from_numpy(first), torch.from_numpy(second)))
    assert len(blobs) == 1
    header, commands = read_blob(blobs[0])
    gather = [command for command in commands if command.type == SHARED_GATHER]
    assert len(gather) == 2
    assert gather[0].inputs[1] == gather[1].inputs[1]
    assert header.weights_bytes == _tiled_bytes(64, 8)

    # The sum is the DSP's own: two fp16 rows added in fp16, as the arena holds
    # them, rather than the fp32 arithmetic the graph declares.
    with torch.no_grad():
        want = (
            model.emb(torch.from_numpy(first)).numpy().astype(np.float16)
            + model.emb(torch.from_numpy(second)).numpy().astype(np.float16)
        ).reshape(-1)
    got = np.frombuffer(execute(blobs[0], [first, second])[0], dtype=np.float16)
    assert np.array_equal(got, want)


def test_int64_indices_keep_the_embedding_on_a_portable_kernel():
    """The negative control for the test above: no int32, no delegate.

    The kernel reads `const int32_t[]`, so an int64 token tensor would be read as
    five low words interleaved with five zero high words. Refusing in the support
    check is what keeps that from being emitted at all, and the export has to
    succeed with the op left where it was.
    """
    torch.manual_seed(0)
    model = _Embedding().eval()
    tokens = np.array([3, 63, 0, 17, 5], dtype=np.int64)
    lowered, blobs = _lowered(model, (torch.from_numpy(tokens),))
    assert blobs == []
    assert any(
        "embedding" in str(node.target) for node in lowered.graph_module.graph.nodes
    )


def test_int64_tokens_reach_the_dsp_through_an_int32_cast():
    """What a model has to do today to delegate an embedding fed by int64 tokens.

    The cast is not delegated -- a cast out of int64 is a conversion the kernels
    have no command for -- so it stays on a portable kernel and splits the graph
    there. The embedding on the other side of the split is a command like any
    other, and what its kernel reads is four bytes per token.
    """

    class _Casting(_Embedding):
        def forward(self, tokens):
            return self.emb(tokens.to(torch.int32))

    torch.manual_seed(0)
    model = _Casting().eval()
    tokens = np.array([3, 63, 0, 17, 5], dtype=np.int64)

    lowered, blobs = _lowered(model, (torch.from_numpy(tokens),))
    assert len(blobs) == 1
    _, command = _only_command(blobs[0])
    assert command.params[0] == len(tokens)
    assert command.inputs[0].size == tokens.size * 4

    # The indices operand is the delegate's own input, and the cast is outside:
    # what reaches the command is the int32 tensor the cast produced.
    graph = _delegates(lowered)[0].original_module.graph_module.graph
    embedding = next(node for node in graph.nodes if "embedding" in str(node.target))
    assert embedding.args[1].op == "placeholder"
    assert embedding.args[1].meta["val"].dtype is torch.int32

    got = np.frombuffer(
        execute(blobs[0], [tokens.astype(np.int32)])[0], dtype=np.float16
    )
    assert np.array_equal(got, _reference(model, tokens))


def test_a_padding_index_comes_back_as_the_zero_row_it_is():
    """`embedding` ignores padding_idx in the forward, so nothing has to carry it.

    The op's padding_idx only shapes the gradient, and torch's own forward is an
    index_select; the row it names is zero because the initializer made it so. The
    kernel would return whatever the row holds, so the check is on the values
    rather than on the argument.
    """
    torch.manual_seed(0)
    model = _Embedding(padding_idx=7).eval()
    assert torch.count_nonzero(model.emb.weight[7]) == 0
    tokens = np.array([7, 3, 7, 0], dtype=np.int32)
    _, blobs = _lowered(model, (torch.from_numpy(tokens),))
    assert len(blobs) == 1
    got = np.frombuffer(execute(blobs[0], [tokens])[0], dtype=np.float16)
    assert np.array_equal(got, _reference(model, tokens))


def test_an_index_out_of_range_clears_the_row_where_torch_raises():
    """The divergence on invalid input, asserted rather than left implicit.

    shared_gather_ops.cc:292-295 clears a row whose index is outside [0, oc).
    Nothing this backend emits can produce such an index from a graph whose
    shapes agree, but the count and the table come from the graph while the index
    comes from the caller, so the boundary is worth pinning down: a token the
    vocabulary does not have reads as zeros on the DSP and raises on the CPU.
    """
    torch.manual_seed(0)
    model = _Embedding().eval()
    tokens = np.array([0, 64, -1], dtype=np.int32)
    _, blobs = _lowered(model, (torch.from_numpy(tokens),))
    got = np.frombuffer(execute(blobs[0], [tokens])[0], dtype=np.float16)
    assert not np.any(got[8:16])
    assert not np.any(got[16:24])
    assert np.array_equal(got[:8], _reference(model, tokens[:1]))
    with pytest.raises(IndexError):
        model(torch.from_numpy(tokens[1:2]))


def test_index_select_and_index_tensor_gather_rows_too():
    """The same read written another way, over axis 0 of the table.

    `torch.index_select(table, 0, index)` and `table[index]` are row gathers, so
    they reach the same command as `embedding`; anything that indexes another
    axis is refused (below) rather than read as if it were this one.
    """

    class _Select(torch.nn.Module):
        def __init__(self, rows=70, dim=40):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(rows, dim))

        def forward(self, index):
            return torch.index_select(self.table, 0, index)

    class _Index(torch.nn.Module):
        def __init__(self, rows=12, dim=5):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(rows, dim))

        def forward(self, index):
            return self.table[index]

    torch.manual_seed(0)
    select = _Select().eval()
    index = np.array([3, 69, 0, 7], dtype=np.int32)
    _, blobs = _lowered(select, (torch.from_numpy(index),))
    assert len(blobs) == 1
    header, command = _only_command(blobs[0])
    assert command.params == [4, 40, 70, 2, SHARED_GATHER_FP16]
    assert header.weights_bytes == _tiled_bytes(70, 40)
    got = np.frombuffer(execute(blobs[0], [index])[0], dtype=np.float16)
    assert np.array_equal(got, _reference(select, index))

    indexed = _Index().eval()
    index = np.array([11, 0, 4], dtype=np.int32)
    _, blobs = _lowered(indexed, (torch.from_numpy(index),))
    assert len(blobs) == 1
    _, command = _only_command(blobs[0])
    assert command.params == [3, 5, 12, 2, SHARED_GATHER_FP16]
    got = np.frombuffer(execute(blobs[0], [index])[0], dtype=np.float16)
    assert np.array_equal(got, _reference(indexed, index))


def test_a_gather_over_another_axis_is_refused():
    """Nothing here can describe a run down a column, so nothing is emitted."""

    class _ColumnSelect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(6, 9))

        def forward(self, index):
            return torch.index_select(self.table, 1, index)

    class _TwoAxis(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(6, 9))

        def forward(self, rows, columns):
            return self.table[rows, columns]

    torch.manual_seed(0)
    index = torch.zeros(3, dtype=torch.int32)
    _, blobs = _lowered(_ColumnSelect().eval(), (index,))
    assert _no_gather(blobs)
    _, blobs = _lowered(_TwoAxis().eval(), (index, index))
    assert _no_gather(blobs)


def test_a_table_the_layer_cannot_read_is_refused():
    """A table computed at run time has no bytes to tile.

    The kernel reads a tiled table, so the rearrange happens at export; a table
    the caller hands in has no export-time bytes, and emitting the row-major
    operand instead is the failure this refuses. What the assembly around it does
    is not the question -- a multiply of two visible tensors is still delegable --
    so what is asserted is that no command reads a table.
    """

    class _RuntimeTable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.randn(1))

        def forward(self, table, index):
            return torch.embedding(table * self.scale, index)

    torch.manual_seed(0)
    lowered, blobs = _lowered(
        _RuntimeTable().eval(),
        (torch.randn(16, 4), torch.zeros(3, dtype=torch.int32)),
    )
    assert _no_gather(blobs)
    assert any(
        "embedding" in str(node.target) for node in lowered.graph_module.graph.nodes
    )


def test_a_dynamic_sequence_patches_select_size():
    """selectSize is the run of tokens this call was handed, not the traced one.

    The blob carries the geometry for the length the graph was traced at plus a
    trailer that overwrites param 0 with the runtime length, which is the same
    mechanism the matmul uses for its rows. The command's own params cannot say
    this, and the interpreter here does not model the trailer, so this is the
    assertion that the count is patched at all.
    """
    torch.manual_seed(0)
    model = _Embedding().eval()
    traced = torch.zeros(3, dtype=torch.int32)
    _, blobs = _lowered(
        model,
        (traced,),
        dynamic_shapes={"tokens": {0: Dim("tokens", min=1, max=16)}},
    )
    assert len(blobs) == 1
    _, command = _only_command(blobs[0])
    assert command.params[0] == 3
    assert command.patch_param == 0xFFFFFFFF

    trailer_magic = (0x44594E48).to_bytes(4, "little")
    at = blobs[0].find(trailer_magic)
    assert at >= 0
    (
        magic,
        version,
        input_index,
        axis,
        max_length,
        patches,
        example,
    ) = struct.unpack_from("<7I", blobs[0], at)
    assert (magic, version, input_index, axis, max_length, example) == (
        0x44594E48,
        3,
        0,
        0,
        16,
        3,
    )
    assert patches == 1
    assert struct.unpack_from("<4i", blobs[0], at + 28) == (0, 0, 1, 0)


def test_the_command_type_and_the_table_kind_are_the_ones_the_c_uses():
    """The two numbers the emitter writes, re-read from the vendored sources.

    Every param is positional and unchecked, so a type that drifted by one would
    run a different kernel with these numbers rather than fail, and `isInt4` is
    the switch that picks between the tiled fp16 read and the quantized ones.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "third-party/mnn-htp-ops"
    values = {}
    next_value = 0
    for line in (root / "include/htp_command.h").read_text().splitlines():
        line = line.strip()
        if not line.startswith("DSP_OP_") or "," not in line:
            continue
        name, _, _ = line.partition(",")
        if "=" in name:
            name, _, explicit = name.partition("=")
            next_value = int(explicit.strip())
        values[next_value] = name.strip()
        next_value += 1
    assert values[SHARED_GATHER] == "DSP_OP_SHARED_GATHER"
    # 0 is the value the fp16 path is reached from: shared_gather_ops.cc:266-282
    # sends 2 to the int4 reader, 3 to the int8 one, and anything else that is not
    # zero to the int4 one, so the plain table is what 0 means.
    assert SHARED_GATHER_FP16 == 0
