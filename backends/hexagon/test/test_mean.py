# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A mean over every dim, which the span test used to iterate.

``aten.mean.dim`` reads a missing dim as "every dim", and both the partitioner's
span test and the emitter iterated the argument, so ``list(None)`` raised a
TypeError out of ``is_node_supported`` and a model holding a whole-tensor mean
failed to lower with a Python error rather than with a refusal. Every dim is one
contiguous span -- the kernel reduces the whole buffer as ``[1][numel][1]`` --
so the missing dim is read as all of them in both places. These tests pin the
delegation and the numbers, not the absence of the exception.
"""



import numpy as np
import torch


import blob_interpreter
from blob_interpreter import execute, read_blob
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _mean_reduces_one_span,
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

#: DSP_OP_REDUCTION, the one command a mean lowers to.
_REDUCTION = 29

#: REDUCTION_MEAN (1 is sum, 2 is maximum) and the width the kernels read.
_MEAN = 3
_FP16_BYTES = 2


class _Mean(torch.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


def _mean_node(dim, shape, result_shape):
    """A mean node with the operand and result values its predicate reads."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=torch.float16)
    mean = graph.call_function(exir_ops.edge.aten.mean.dim, args=(source, dim, False))
    mean.meta["val"] = torch.empty(result_shape, dtype=torch.float16)
    return mean


def test_a_mean_over_every_dim_is_one_span():
    """The support check answers, and its answer is yes.

    A missing dim is every dim, which is the whole tensor as the reduce span
    with nothing outside it and nothing inside it; the answer for a dim set it
    cannot collapse is unchanged, so this is not a relaxation of the test.
    """
    node = _mean_node(None, (2, 3, 4), ())
    assert _mean_reduces_one_span(node)
    assert HexagonOperatorSupport().is_node_supported({}, node)
    assert _mean_reduces_one_span(_mean_node((0, 1), (2, 3, 4), (4,)))
    assert not _mean_reduces_one_span(_mean_node((0, 2), (2, 3, 4), (3,)))


def _delegated_blob(dim, shape):
    """The blob a whole-model mean over `shape` lowers to, plus its operands."""
    x = torch.randn(*shape, dtype=torch.float16)
    program = to_edge_transform_and_lower(
        export(_Mean(dim), (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the mean did not reach the delegate: {calls}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    return bytes(lowered._processed_bytes), x, _Mean(dim)(x)


def test_a_mean_over_every_dim_delegates_as_one_span_and_matches_torch():
    """The emitted command is the whole tensor as one span, and it is right."""
    for shape in ((20,), (2, 3, 4), (2, 3, 4, 5)):
        blob, x, expected = _delegated_blob(None, shape)
        _, commands = read_blob(blob)
        assert [command.type for command in commands] == [_REDUCTION]
        assert list(commands[0].params[:5]) == [
            1,
            x.numel(),
            1,
            _MEAN,
            _FP16_BYTES,
        ], f"a whole-tensor mean emitted {commands[0].params[:5]}"
        got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)[:1]
        np.testing.assert_allclose(
            got, expected.numpy().reshape(-1), rtol=2e-3, atol=2e-3
        )
