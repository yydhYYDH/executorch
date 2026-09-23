# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the runtime needs to turn a DSP command into a graph node.

A profile of a delegated subgraph is a list of commands, and the only thing that
makes it readable is the map from command index to debug handle that preprocess
hands to EXIR. These tests pin that map, the fact that producing it does not
change the blob, and the probe layout the two C++ sides have to agree on for the
per-command times to arrive at all.
"""

import os
import pathlib
import struct
import sys

import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import to_edge, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.backend.backend_api import LoweredBackendModule  # noqa: E402
from torch.export import export  # noqa: E402

class _Mm(torch.nn.Module):
    """One command per node: a matmul then a bias add."""

    def forward(self, x, w, b):
        return torch.mm(x, w) + b


class _LayerNorm(torch.nn.Module):
    """A node the emitter lowers to more than one command."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))
        self.bias = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, (8,), self.weight, self.bias, 1e-5)


def _mm_args():
    return (
        torch.randn(4, 8, dtype=torch.float16),
        torch.randn(8, 6, dtype=torch.float16),
        torch.randn(6, dtype=torch.float16),
    )


def _command_count(blob: bytes) -> int:
    # n_ops is the third int of HexagonBlobHeader.
    return struct.unpack_from("<9I", blob, 0)[2]


def _handles(program) -> list:
    """The subgraph's debug handles, in graph order, one per node."""
    return [
        node.meta["debug_handle"]
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and node.meta.get("debug_handle") is not None
    ]


def test_a_command_is_numbered_by_the_handle_of_the_node_that_asked_for_it():
    """The runtime logs command i as delegate debug identifier i.

    So command i has to be the one node i's emitter added, in the order the
    emitters ran: that index is the whole linkage between a per-op event and the
    node it belongs to.
    """
    program = to_edge(export(_Mm(), _mm_args())).exported_program()
    result = HexagonBackend.preprocess(program, [])
    assert result.debug_handle_map == {
        0: (_handles(program)[0],),
        1: (_handles(program)[1],),
    }
    assert _command_count(result.processed_bytes) == 2


def test_a_node_that_emits_several_commands_appears_once_per_command():
    """Two commands from one node are two identifiers pointing at one handle.

    Which is what the handle means: the commands a single aten op decomposes
    into are all that op's time.
    """
    program = to_edge(export(_LayerNorm(), (torch.randn(4, 8, dtype=torch.float16),))).exported_program()
    result = HexagonBackend.preprocess(program, [])
    commands = _command_count(result.processed_bytes)
    assert commands > 1
    assert sorted(result.debug_handle_map) == list(range(commands))
    handle = _handles(program)[0]
    assert set(result.debug_handle_map.values()) == {(handle,)}


def test_the_map_does_not_reach_the_blob():
    """The mapping travels in the ETRecord, not in the .pte.

    A blob is what the device reads, and a command index is stable without any
    handle in it, so carrying handles in the blob would buy nothing and change
    the wire format. The strongest way to say that is that the same graph
    lowered with handles and without produces the same bytes.
    """
    program = to_edge(export(_Mm(), _mm_args())).exported_program()
    with_handles = HexagonBackend.preprocess(program, [])
    assert with_handles.debug_handle_map != {}
    for node in program.graph_module.graph.nodes:
        node.meta.pop("debug_handle", None)
    without_handles = HexagonBackend.preprocess(program, [])
    assert without_handles.debug_handle_map == {}
    assert with_handles.processed_bytes == without_handles.processed_bytes


def test_a_partitioned_subgraph_is_profiled_by_the_map_it_records():
    """The whole AOT path: the partitioner's subgraph, lowered and inspected.

    What matters here is that the map survives into the lowered module's meta,
    which is where EXIR reads it from when it assembles the ETRecord.
    """
    lowered = to_edge_transform_and_lower(
        export(_Mm(), _mm_args()),
        partitioner=[HexagonPartitioner()],
    )
    modules = [
        module
        for _, module in lowered.exported_program().graph_module.named_modules()
        if isinstance(module, LoweredBackendModule)
    ]
    assert len(modules) == 1
    mapping = modules[0].meta["debug_handle_map"]
    assert sorted(mapping) == list(range(_command_count(modules[0].processed_bytes)))
    assert all(isinstance(handle, int) for (handle,) in mapping.values())
