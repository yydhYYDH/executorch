# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The clamp-shaped family, and x ** 2: four ops one kernel already covers.

`HtpOpsUnaryOpType` has a clamp entry point that carries its own two bounds
(`unary_ops.cc:29`, params[3] and params[4]) and a square entry point
(`unary_ops.cc:22`). Three further edge ops are that clamp exactly: torch
implements hardtanh as clamp and gives min_val/max_val the slots min/max occupy,
relu is that clamp between zero and infinity, and relu6 is `hardtanh(x, 0, 6)`.
All four, and square, had no emitter, so each one quietly stayed on the portable
kernels while `torch.clamp` next to them was delegated.

The bound pair is the load-bearing part, so the tests are on the params and on
numbers that include NaN, both infinities and a signed zero. The kernel's
compares are unordered and it restores a NaN input by a bit test
(`htp_ops_clamp_fp16_chunk`, unary_ops.cc:505-509), which is what makes relu's
`max(x, 0)` answer NaN with NaN the way torch's does.
"""



import numpy as np
import pytest
import torch


from blob_interpreter import execute, read_blob
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

#: DSP_OP_UNARY and the two HtpOpsUnaryOpType entries these ops select.
_UNARY = 4
_CLAMP = 15
_SQUARE = 9
_FP16_BYTES = 2

_POS_INF = 0x7C00
_NEG_INF = 0xFC00


def _fp16_bits(value: float) -> int:
    return int(np.array([value], dtype=np.float16).view(np.uint16)[0])


class _Unary(torch.nn.Module):
    def __init__(self, kind) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == "relu":
            return torch.relu(x)
        if self.kind == "hardtanh":
            return torch.nn.functional.hardtanh(x)
        if self.kind == "relu6":
            return torch.nn.functional.relu6(x)
        if self.kind == "clamp":
            return torch.clamp(x, -1.0, 1.0)
        if self.kind == "square":
            return x**2
        raise AssertionError(self.kind)


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


def _run(model, x):
    """The one command this model lowers to, and what the blob interpreter says."""
    _, commands = read_blob(_blob(_program(model, x)))
    assert len(commands) == 1, f"expected one command, got {len(commands)}"
    got = np.frombuffer(
        execute(_blob(_program(model, x)), [x.numpy()])[0], dtype=np.float16
    )
    return commands[0], got


def _bits(values) -> list:
    return np.frombuffer(
        np.asarray(values, dtype=np.float16).tobytes(), dtype=np.uint16
    ).tolist()


# --------------------------------------------------------------------- params


@pytest.mark.parametrize(
    "kind,lower,upper",
    [
        # relu is clamp(x, 0, +inf): the upper bound is a compare that never
        # fires, which is what makes it the max torch computes.
        ("relu", _fp16_bits(0.0), _POS_INF),
        # F.hardtanh's schema defaults to (-1, 1).
        ("hardtanh", _fp16_bits(-1.0), _fp16_bits(1.0)),
        # relu6 is F.hardtanh(x, 0, 6) in torch, and low == 0 makes it relu.
        ("relu6", _fp16_bits(0.0), _fp16_bits(6.0)),
        ("clamp", _fp16_bits(-1.0), _fp16_bits(1.0)),
    ],
)
def test_each_member_of_the_family_emits_its_own_two_bounds(kind, lower, upper):
    """One UNARY command, op type 15, with the fp16 bit patterns of the bounds.

    The four ops differ only in those two numbers, so this is the whole claim
    that one emitter covers them: the arguments land in the slots the kernel
    reads them from.
    """
    x = torch.randn(8, dtype=torch.float16)
    command, _ = _run(_Unary(kind), x)
    assert command.type == _UNARY
    assert list(command.params[:5]) == [8, _CLAMP, _FP16_BYTES, lower, upper]


def test_x_squared_emits_the_unary_square_kernel():
    """to_edge emits no aten.square.default; x ** 2 is pow.Tensor_Scalar.

    That is the whole reason this was a gap rather than a missing kernel: the
    entry point has been in HtpOpsUnaryOpType all along, and the target the
    exporter produces is not the name the op is usually thought of as.
    """
    x = torch.randn(8, dtype=torch.float16)
    command, _ = _run(_Unary("square"), x)
    assert command.type == _UNARY
    assert list(command.params[:5]) == [8, _SQUARE, _FP16_BYTES]


@pytest.mark.parametrize("exponent", [3, 0.5, -1, 1.5])
def test_another_exponent_is_not_the_square_kernel(exponent):
    """Only x ** 2 has a unary entry point; the rest stay on the portable kernel.

    Refusing is the point: an emitter that ignored the exponent would answer
    x * x for x ** 3, which is a wrong number rather than a slow one.
    """

    class Pow(torch.nn.Module):
        def forward(self, x):
            return x**exponent

    x = torch.randn(8, dtype=torch.float16)
    assert _delegates(_program(Pow(), x)) == []


# ------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("relu", lambda x: torch.relu(x)),
        ("hardtanh", lambda x: torch.nn.functional.hardtanh(x)),
        ("relu6", lambda x: torch.nn.functional.relu6(x)),
        ("clamp", lambda x: torch.clamp(x, -1.0, 1.0)),
        ("square", lambda x: x**2),
    ],
)
def test_the_values_are_bit_for_bit_torch(kind, expected):
    """NaN, both infinities and a signed zero, against torch eager.

    A signed zero is in the input on purpose: relu(-0.0) is 0.0 and the kernel's
    lower bound is a positive zero, so the comparison is the one that decides it.
    """
    values = [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.0,
        -3.0,
        0.0,
        4.0,
        -6.5,
    ]
    x = torch.tensor(values, dtype=torch.float16)
    _, got = _run(_Unary(kind), x)
    assert _bits(got) == _bits(expected(x).numpy())
    if kind != "square":
        # The NaN is the load-bearing element: the kernel's compares are
        # unordered and would answer the upper bound without the bit-test
        # restore, where torch answers NaN.
        assert np.isnan(got[0])


def test_hardtanh_is_clamp_and_relu6_is_hardtanh_of_zero_and_six():
    """The equalities the shared emitter rests on, stated on torch itself.

    If torch ever stopped computing hardtanh as clamp, the single emitter would
    stop being one emitter, so the claim is worth pinning where it comes from.
    """
    values = [float("nan"), float("inf"), float("-inf"), -0.0, -3.0, 0.0, 4.0, 7.0]
    x = torch.tensor(values, dtype=torch.float16)
    assert _bits(torch.nn.functional.hardtanh(x, 0.0, 6.0)) == _bits(
        torch.clamp(x, 0.0, 6.0)
    )
    assert _bits(torch.nn.functional.relu6(x)) == _bits(
        torch.nn.functional.hardtanh(x, 0.0, 6.0)
    )
    assert _bits(torch.relu(x)) == _bits(torch.clamp(x, 0.0, float("inf")))


# --------------------------------------------------------------- the gap itself


def test_the_family_was_absent_rather_than_refused():
    """These three targets had no emitter at all, which no refusal would report.

    The partitioner only sees a node when its target is a key of
    `SUPPORTED_TARGETS`, so an op missing from EMITTERS is never rejected -- it
    is never considered, and the only symptom is a delegate that does not
    appear. That is why the gap survived next to a delegated `torch.clamp`: a
    refused node and an unknown node look the same from the outside.
    """
    from executorch.backends.hexagon import hexagon_ops
    from executorch.exir.dialects._ops import ops as exir_ops

    edge = exir_ops.edge.aten
    for target in (
        edge.relu.default,
        edge.hardtanh.default,
        edge.pow.Tensor_Scalar,
    ):
        assert any(
            key is target for key in hexagon_ops.EMITTERS
        ), f"{target} still has no emitter"
    x = torch.randn(8, dtype=torch.float16)
    for kind in ("relu", "hardtanh", "relu6", "clamp", "square"):
        assert len(_delegates(_program(_Unary(kind), x))) == 1, kind


def test_the_interpreter_now_models_the_clamp_entry_point():
    """Clamp used to raise UnsupportedOp, so no test could ever check it.

    `_run_unary` dispatched every UNARY command to a table of closed-form
    functions keyed by op type, and op 15 is not in that table -- its bounds
    arrive as operands rather than as one op type. `torch.clamp` has been
    delegated all along, so the delegated clamp was the one op whose numbers
    nothing here could read; `_run` above now executes it.
    """
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float16)
    command, got = _run(_Unary("clamp"), x)
    assert command.params[1] == _CLAMP
    assert _bits(got) == _bits(torch.clamp(x, -1.0, 1.0).numpy())
