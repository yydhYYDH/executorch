# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A constant transpose read from a delegate output.

`permute_copy` of a weight is folded into the weights section at export, which
leaves the node with a weight reference and no entry in the producer table -- a
consumer reaches the fold through its own table, so nothing noticed. A node the
caller reads from a delegate output has no consumer: the delegate is asked for
the transposed weight and the emitter had no slot to write it to, so lowering
failed with "subgraph output 0 is ... which no emitter produced" (an LSTM with
batch > 1 splits exactly this partition off, because its consumer stays on the
portable kernels). An output slot is a region of the arena the runtime copies
out, so there the transpose is emitted as the blit it is.

Both halves are pinned: the split-off transpose has to reach the output slot
with the right numbers, and a transpose a consumer reads still has to be folded
rather than emitted.
"""

import os
import pathlib
import sys

import numpy as np
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as B  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_RASTER_BLIT, which is what a transpose lowers to.
_RASTER_BLIT = 3

#: DSP_OP_BATCH_MATMUL, which is what the product reads a folded weight with.
_BATCH_MATMUL = 38

#: The two-byte element every kernel reads.
_FP16_BYTES = 2


class _TransposedWeight(torch.nn.Module):
    """The transposed weight and nothing else: the delegate's whole output."""

    def __init__(self, weight) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(weight)

    def forward(self):
        return self.w.t()


class _Product(torch.nn.Module):
    """The transposed weight of the matmul that consumes it."""

    def __init__(self, weight) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(weight)

    def forward(self, x):
        return x @ self.w.t()


def _delegated_blob(module, example_inputs):
    program = to_edge_transform_and_lower(
        export(module.eval(), example_inputs),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"expected one delegate, got {calls}"
    return bytes(
        program.graph_module.get_submodule(calls[0].args[0].target)._processed_bytes
    )


def test_a_constant_transpose_at_a_delegate_output_is_written_out():
    """The output slot gets the transpose, with the weight's numbers in it."""
    weight = torch.randn(8, 4)
    blob = _delegated_blob(_TransposedWeight(weight), ())
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [_RASTER_BLIT]
    (out,) = commands[0].outputs
    assert out.space is B.TensorSpace.OUTPUT and out.index == 0
    assert out.size == weight.t().numel() * _FP16_BYTES

    got = np.frombuffer(execute(blob, [])[0], dtype=np.float16)
    expected = weight.t().to(torch.float16).numpy().reshape(-1)
    assert np.array_equal(got[: expected.size], expected)


def test_a_constant_transpose_a_consumer_reads_is_still_folded():
    """Fix scope: the fold stays, so a transpose nothing reads is not emitted."""
    weight = torch.randn(4, 8)
    x = torch.randn(2, 8)
    blob = _delegated_blob(_Product(weight), (x,))
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [_BATCH_MATMUL], (
        "the folded weight reached the kernels directly before this change; a "
        f"blit here means the fold was dropped: {commands}"
    )
