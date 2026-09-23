# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The two reduction overloads that name no dim, and the width gate on mean.

A reduction family has more than one entry point in the edge dialect, and the
backend only reaches the ones whose target is in `EMITTERS`. ``torch.mean(x)``
lowers to ``aten.mean.default`` while ``torch.mean(x, dim=1)`` lowers to
``aten.mean.dim``; only the second had an emitter, so a whole-tensor mean left
the graph with no error and no delegate. ``torch.max(x)`` has the same pair
against ``torch.amax(x)``. These tests pin both, and pin the dtype rule that
``sum.dim_IntList`` already carried and the two mean overloads did not: a mean
that requests a width is not the mean this kernel computes.

The interpreter is the second implementation of the kernels, so the numbers are
checked against torch eager rather than against the emitter's own arithmetic.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    _mean_reduces_one_span,
    HexagonOperatorSupport,
    HexagonPartitioner,
    mean_result_width_is_emittable,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

#: DSP_OP_REDUCTION, REDUCTION_MEAN and REDUCTION_MAXIMUM, and the width.
_REDUCTION = 29
_MAXIMUM = 2
_MEAN = 3
_FP16_BYTES = 2

_MEAN_DEFAULT = exir_ops.edge.aten.mean.default
_MEAN_DIM = exir_ops.edge.aten.mean.dim
_MAX_DEFAULT = exir_ops.edge.aten.max.default


def _program(model, x):
    return to_edge_transform_and_lower(
        export(model, (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _blob(program):
    calls = _delegates(program)
    assert len(calls) == 1, f"expected one delegate, got {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    return bytes(lowered._processed_bytes)


class _Whole(torch.nn.Module):
    """torch.mean(x) and torch.max(x): the overloads that name no dim."""

    def __init__(self, kind) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, x):
        return torch.mean(x) if self.kind == "mean" else torch.max(x)


class _MeanDim(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return torch.mean(x, **self.kwargs)


def _node(target, shape, result_shape, **kwargs):
    """A node of `target` with the operand and result values its gate reads."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=torch.float16)
    if target is _MEAN_DEFAULT:
        node = graph.call_function(target, args=(source,), kwargs=kwargs)
    else:
        node = graph.call_function(
            target, args=(source, kwargs.pop("dim"), False), kwargs=kwargs
        )
    node.meta["val"] = torch.empty(result_shape, dtype=torch.float16)
    return node


# --------------------------------------------------------------- the overloads


@pytest.mark.parametrize("kind", ["mean", "max"])
@pytest.mark.parametrize("shape", [(20,), (2, 3, 4), (2, 3, 4, 5)])
def test_a_reduction_over_every_dim_is_one_span_and_matches_torch(kind, shape):
    """Both overloads emit one REDUCTION over [1][numel][1] and agree with torch."""
    x = torch.randn(*shape, dtype=torch.float16)
    blob = _blob(_program(_Whole(kind), x))
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [_REDUCTION]
    wanted = _MEAN if kind == "mean" else _MAXIMUM
    assert list(commands[0].params[:5]) == [
        1,
        x.numel(),
        1,
        wanted,
        _FP16_BYTES,
    ], f"{kind} over every dim emitted {commands[0].params[:5]}"
    got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)[:1]
    expected = (torch.mean(x) if kind == "mean" else torch.max(x)).numpy().reshape(-1)
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def test_the_two_mean_overloads_are_the_same_command():
    """torch.mean(x) and torch.mean(x, dim=all) lower to the same three params.

    The overload without a dim reduces every axis; mean.dim asked for every axis
    is the same reduction, so the two must not differ in what they emit. Before
    mean.default had an emitter this could not be written -- it produced no
    command at all.
    """
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    whole = _blob(_program(_Whole("mean"), x))
    named = _blob(_program(_MeanDim(dim=None), x))
    assert list(read_blob(whole)[1][0].params[:5]) == [
        1,
        x.numel(),
        1,
        _MEAN,
        _FP16_BYTES,
    ]
    assert list(read_blob(whole)[1][0].params[:5]) == list(
        read_blob(named)[1][0].params[:5]
    )


def test_a_whole_tensor_mean_is_a_delegate_rather_than_a_silent_host_op():
    """The regression this test exists for: torch.mean(x) used to emit nothing.

    `aten.mean.default` was a target with no entry in EMITTERS, so the node was
    never in SUPPORTED_TARGETS and the partitioner left it where it was. That is
    not a refusal, it is an absence, and the only symptom is a missing delegate.
    """
    assert _MEAN_DEFAULT in hexagon_ops.EMITTERS
    assert _MAX_DEFAULT in hexagon_ops.EMITTERS
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    assert len(_delegates(_program(_Whole("mean"), x))) == 1
    assert len(_delegates(_program(_Whole("max"), x))) == 1


def test_both_mean_overloads_spell_out_which_dim_is_missing():
    """The span test reads the dim from either overload without iterating None.

    `_mean_reduces_one_span` used to index args[1], which does not exist on the
    overload that has no dim; the test is on the predicate rather than on a
    lowered graph because that is where the argument is read.
    """
    assert _mean_reduces_one_span(_node(_MEAN_DEFAULT, (2, 3, 4), ()))
    assert _mean_reduces_one_span(_node(_MEAN_DIM, (2, 3, 4), (4,), dim=None))
    assert _mean_reduces_one_span(_node(_MEAN_DIM, (2, 3, 4), (4,), dim=(0, 1)))
    # A dim set it cannot collapse is still refused on both.
    assert not _mean_reduces_one_span(_node(_MEAN_DIM, (2, 3, 4), (3,), dim=(0, 2)))
    # A rank-0 operand has no span to collapse.
    assert not _mean_reduces_one_span(_node(_MEAN_DEFAULT, (), ()))


# --------------------------------------------------------------- the width rule


def test_a_mean_into_another_width_stays_on_the_host():
    """The kernel stores fp16, so a mean that names a dtype is a different op.

    `sum.dim_IntList` has carried this rule since the fp32-accumulator defect
    was found (test_sum_amax.py::test_a_sum_into_another_width_stays_on_the_host).
    Both mean overloads skipped it, so `torch.mean(x, dim=1, dtype=float32)`
    reached the kernel, came back fp16 and was widened by the runtime -- a fp32
    tensor rounded to fp16, which is the same defect with a different op.
    """
    x = torch.randn(2, 3, 4, dtype=torch.float16)
    for model in (
        _MeanDim(dim=1, dtype=torch.float32),
        _MeanDim(dtype=torch.float32),
    ):
        assert _delegates(_program(model, x)) == []
        assert model(x).dtype is torch.float32
    assert _delegates(_program(_Whole("mean"), x)) != []


def test_the_width_gate_reads_the_keyword_slot_both_overloads_use():
    """dtype is keyword-only on both mean schemas, so kwargs is the whole check.

    A gate that only looked at a positional slot would accept every real graph:
    neither overload can put dtype in `args` at all.
    """
    assert mean_result_width_is_emittable(_node(_MEAN_DEFAULT, (2, 3, 4), ()))
    assert not mean_result_width_is_emittable(
        _node(_MEAN_DEFAULT, (2, 3, 4), (), dtype=torch.float32)
    )
    assert not mean_result_width_is_emittable(
        _node(_MEAN_DIM, (2, 3, 4), (4,), dim=1, dtype=torch.float32)
    )
    assert not HexagonOperatorSupport().is_node_supported(
        {}, _node(_MEAN_DEFAULT, (2, 3, 4), (), dtype=torch.float32)
    )


def test_an_empty_dim_reads_as_every_dim_on_all_three_reductions():
    """`dim=()` is not "reduce nothing": torch reduces every axis, all three.

    The predicate here used to refuse an empty dim set on mean while
    `reduction_dims` read the same argument as every dim on sum and amax, so
    `torch.mean(x, dim=())` stayed on the host while `torch.sum(x, dim=())`
    delegated -- the same reduction, off by one overload. The equality with the
    missing-dim form is checked on torch rather than asserted from the docs.
    """
    x = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)
    for reduce in (torch.sum, torch.mean, torch.amax):
        assert torch.equal(reduce(x, dim=()), reduce(x, dim=None)), reduce
        assert not torch.equal(
            reduce(x, dim=()), x
        ), "an empty dim set is an identity for this reduction after all"
    for model in (_MeanDim(dim=()), _MeanDim(dim=[])):
        assert len(_delegates(_program(model, x))) == 1
    assert _mean_reduces_one_span(_node(_MEAN_DIM, (2, 3, 4), (), dim=()))
    assert _mean_reduces_one_span(_node(_MEAN_DIM, (2, 3, 4), (), dim=[]))


def test_max_and_amax_are_the_same_values():
    """torch.max(x) is torch.amax(x) in every value; only a signed zero differs.

    amax is already delegated, so the reason to be careful about max is not the
    reduction but the tie-breaking: on a mix of +0 and -0 torch.max returns the
    first element's zero where torch.amax returns one of them. The two compare
    equal, which is the byte-level caveat fmod already documents.
    """
    x = torch.tensor([-0.0, 3.0, 0.0, -5.0], dtype=torch.float16)
    assert torch.max(x).item() == torch.amax(x).item()
    assert float(torch.max(torch.tensor([-0.0, 0.0], dtype=torch.float16))) == float(
        torch.amax(torch.tensor([-0.0, 0.0], dtype=torch.float16))
    )
