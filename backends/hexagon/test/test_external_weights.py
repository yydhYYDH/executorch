# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weights that live in a .ptd next to the .pte, and the blob's record of them.

Two things are claimed when a weight leaves the .pte: the file the runtime opens
holds the same bytes the .pte used to, and the blob still says where each of them
goes. The first is checked by running the externalized blob through the host
interpreter with the round-tripped .ptd and comparing against the blob that kept
everything inline; the second by reading the trailer the way the runtime does and
comparing it against what the writer meant.

What none of this checks is a device: the DSP reads the weight out of the same
mapped arena either way, so what a .ptd saves is the size of the .pte and the
copy of it, not the rpcmem transfer.
"""

import os


import numpy as np
import pytest
import torch


from executorch.backends.hexagon.hexagon_backend import (
    HexagonCompileOptions,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as B
from executorch.backends.hexagon.test.blob_interpreter import (
    execute,
    read_blob,
    read_external_weights,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir._serialize.data_serializer import DataPayload
from executorch.extension.flat_tensor.serialize.serialize import (
    FlatTensorSerializer,
)
from torch.export import export

#: One command, one weight: small enough to export in a second, big enough that
#: the byte count either side of the boundary is worth counting.
ROWS = 4
IN_FEATURES = 64
OUT_FEATURES = 64
WEIGHT_BYTES = IN_FEATURES * OUT_FEATURES * 2


class _Project(torch.nn.Module):
    """A matmul whose weight the delegate owns, so the blob has bytes to store."""

    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n, dtype=torch.float16))

    def forward(self, x):
        return torch.mm(x, self.weight)


def _delegates(program):
    """The delegate submodules of a lowered graph, in node order."""
    graph_module = program.graph_module
    return [
        graph_module.get_submodule(str(node.args[0].target))
        for node in graph_module.graph.nodes
        if "call_delegate" in getattr(node.target, "__name__", "")
    ]


def _build(external_weights_max_bytes=None, hmx_prepack=False):
    """The model, its input, and the lowered edge program.

    Seeded, so two builds describe the same program and their blobs differ only
    in where the weights went. Prepacking is off unless a test asks for it: it
    rewrites the weight into the unit's tile order, and these tests compare the
    bytes that left the .pte with the bytes that stayed in.
    """
    torch.manual_seed(0)
    model = _Project(IN_FEATURES, OUT_FEATURES)
    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.float16)
    options = HexagonCompileOptions(
        hmx_prepack=hmx_prepack,
        external_weights_max_bytes=external_weights_max_bytes,
    )
    edge = to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner(options)],
    )
    return model, x, edge


def _only_delegate(edge):
    delegates = _delegates(edge.exported_program())
    assert len(delegates) == 1, f"expected one delegate, got {len(delegates)}"
    return delegates[0]


def _store_bytes(delegate):
    """The .ptd a runner would write, round-tripped back to key -> bytes.

    The file is made by the same serializer `to_executorch` uses, so this is the
    format the runtime's data map reads rather than a restatement of it.
    """
    store = delegate.named_data_store_output
    assert store is not None, "the delegate has nothing outside the .pte"
    assert len(store.external_data) == 1, store.external_data
    tag, entries = next(iter(store.external_data.items()))
    serializer = FlatTensorSerializer()
    written = bytes(
        serializer.serialize(DataPayload(buffers=store.buffers, named_data=entries))
    )
    assert written, "the .ptd came out empty"

    read_back = serializer.deserialize_to_named_data_store_output(written, tag)
    back = read_back.external_data[tag]
    assert set(back) == set(entries), (set(back), set(entries))
    return (
        tag,
        written,
        {
            key: bytes(read_back.buffers[entry.buffer_index])
            for key, entry in back.items()
        },
    )


def _inline_weights(blob: bytes) -> bytes:
    """The weights section as the file carries it."""
    header, _ = read_blob(blob)
    at = B.HEADER_SIZE + header.n_ops * B.OP_SIZE
    return blob[at : at + header.weights_bytes]


def test_a_threshold_moves_the_weight_out_of_the_pte():
    """The same program, once with everything inline and once with none of it."""
    _, _, inline_edge = _build(None)
    _, _, external_edge = _build(1)
    inline = _only_delegate(inline_edge).processed_bytes
    external = _only_delegate(external_edge).processed_bytes

    inline_header, _ = read_blob(inline)
    external_header, _ = read_blob(external)
    assert inline_header.version == B.BLOB_VERSION
    assert external_header.version == B.BLOB_VERSION_EXTERNAL_WEIGHTS

    weights = read_external_weights(external)
    assert len(weights.entries) == 1, weights.entries
    entry = weights.entries[0]
    assert entry.size == WEIGHT_BYTES
    assert entry.offset >= external_header.weights_bytes
    assert weights.weights_bytes == B._align_up(entry.offset + entry.size)

    # The .pte lost the weight and gained the trailer that names it.
    trailer_bytes = B._EXTERNAL_HEADER.size + B._EXTERNAL_WEIGHT.size
    assert len(external) == len(inline) - entry.size + trailer_bytes
    assert _only_delegate(inline_edge).named_data_store_output is None


def test_the_store_holds_the_bytes_the_pte_gave_up():
    """One key, its bytes, and the place the blob says they go."""
    _, _, inline_edge = _build(None)
    _, _, external_edge = _build(1)
    inline = _only_delegate(inline_edge).processed_bytes
    external = _only_delegate(external_edge).processed_bytes

    weights = read_external_weights(external)
    tag, written, supplied = _store_bytes(_only_delegate(external_edge))
    assert tag.startswith("hexagon_weights_")
    assert written
    assert list(supplied) == [entry.key for entry in weights.entries]

    # The key is derived from the content, so the same weight exported twice
    # reaches one entry rather than two.
    entry = weights.entries[0]
    assert entry.key == B.external_weight_key(supplied[entry.key])
    assert len(entry.key) <= B.EXTERNAL_KEY_MAX
    assert entry.key.isascii()

    # The weight the .pte gave up is the weight it used to carry.
    carried = _inline_weights(inline)
    assert supplied[entry.key] in carried, "the .ptd does not hold the same weight"


def test_the_external_blob_runs_the_same_as_the_inline_one():
    """The point of the exercise: the same numbers, from a smaller .pte."""
    model, x, inline_edge = _build(None)
    _, _, external_edge = _build(1)
    inline = _only_delegate(inline_edge).processed_bytes
    external = _only_delegate(external_edge).processed_bytes
    _, _, supplied = _store_bytes(_only_delegate(external_edge))

    expected = (x.float() @ model.weight.detach().float()).half().numpy().reshape(-1)
    inline_out = np.frombuffer(execute(inline, [x.numpy()])[0], dtype=np.float16)
    external_out = np.frombuffer(
        execute(external, [x.numpy()], supplied)[0], dtype=np.float16
    )
    assert np.array_equal(inline_out, external_out), "the weight moved and changed"
    assert np.array_equal(external_out, expected)


def test_a_blob_without_its_weight_is_refused():
    """Both failures the runtime has to report, on the same blob."""
    _, x, edge = _build(1)
    delegate = _only_delegate(edge)
    external = delegate.processed_bytes
    _, _, supplied = _store_bytes(delegate)
    entry = read_external_weights(external).entries[0]

    for named_data, message in (
        (None, "no data map has it"),
        ({entry.key: supplied[entry.key][:-2]}, "the blob expects"),
    ):
        try:
            execute(external, [x.numpy()], named_data)
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(
                f"a blob run without its weight should fail ({message})"
            )


def test_nothing_leaves_the_pte_until_the_threshold_is_below_a_weight():
    """No externalization means the bytes of the blob are what they always were."""
    _, _, untouched_edge = _build(None)
    _, _, above_edge = _build(WEIGHT_BYTES)
    untouched = _only_delegate(untouched_edge)
    above = _only_delegate(above_edge)

    assert above.processed_bytes == untouched.processed_bytes
    assert above.named_data_store_output is None
    weights = read_external_weights(above.processed_bytes)
    assert weights.entries == []
    header, _ = read_blob(above.processed_bytes)
    assert header.version == B.BLOB_VERSION
    assert weights.weights_bytes == header.weights_bytes


def _can_write_a_pte() -> bool:
    """Whether this checkout can serialize a .pte at all.

    The flatbuffer writer reads program.fbs as a package resource, which a
    source tree only has after an install step copies it in (`setup.py`, from
    `schema/`). Where it is missing, the .ptd above is still written and read;
    only the .pte wrapper around it is unavailable.
    """
    import importlib.resources

    from executorch.exir import _serialize

    return (importlib.resources.files(_serialize) / "program.fbs").is_file()


def test_the_pte_and_the_ptd_land_on_disk(tmp_path):
    """The whole-module path: one .pte beside one .ptd, holding the same bytes.

    The .pte is what a runner loads and the .ptd is what it has to find beside it,
    so the two sizes here are the claim the option makes: the same program, once
    with the weight inside the .pte and once with it outside.
    """
    if not _can_write_a_pte():
        pytest.skip(
            "this checkout has no program.fbs resource, so no .pte can be written"
        )
    _, x, edge = _build(1)
    delegate = _only_delegate(edge)
    tag, written, supplied = _store_bytes(delegate)
    entry = read_external_weights(delegate.processed_bytes).entries[0]

    program = edge.to_executorch()
    program.save(os.fspath(tmp_path / "model.pte"))
    program.write_tensor_data_to_file(os.fspath(tmp_path))

    pte = (tmp_path / "model.pte").read_bytes()
    assert pte, "the .pte came out empty"
    # The blob travels inside it, and the blob names the key the .ptd holds.
    assert entry.key.encode() in pte, "the .pte does not carry this delegate's blob"
    ptd = sorted(path.name for path in tmp_path.glob("*.ptd"))
    assert ptd == [f"{tag}.ptd"], ptd
    assert (tmp_path / f"{tag}.ptd").read_bytes() == written
    assert supplied[entry.key] in written

    # The same program with the weight left inside, for the size claim.
    _, _, inline = _build(None)
    inline.to_executorch().save(os.fspath(tmp_path / "inline.pte"))
    inline_pte = (tmp_path / "inline.pte").read_bytes()
    assert inline_pte
    assert (
        sorted(path.name for path in tmp_path.glob("*.ptd")) == ptd
    ), "the inline program wrote a .ptd it has no use for"
    # The accounting is exact: the .pte loses the weight and gains the trailer
    # that says where it went, one header plus one record per weight.
    trailer = B._EXTERNAL_HEADER.size + B._EXTERNAL_WEIGHT.size * len(
        read_external_weights(delegate.processed_bytes).entries
    )
    assert len(pte) == len(inline_pte) - entry.size + trailer
    assert len(pte) < len(inline_pte)
    assert len(written) >= entry.size
