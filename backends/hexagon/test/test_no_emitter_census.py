# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The op families with no emitter at all, measured rather than listed.

`OP_GAPS.md` section 1 is the hand-written half of this census and nothing
watches it, so two of its rows had already gone stale by the time this file
was written: it places `aten.leaky_relu.default` in "the vendored library has
no kernel at all" while `EMITTERS` carries it and it delegates
`DSP_OP_RELU`, and its header quotes 126 census rows / 187 verdicts / 38
unwired where the row tables the header points at now hold 131 / 193 / 30.
This file re-derives the number from the tables themselves, so the prose
cannot be the only witness.

Three things are pinned per family, and they answer three different questions
that a single delegate count would conflate.

*What reached the DSP* is read from the command stream inside each delegate's
blob, never from how many delegates the graph formed. That distinction is not
theoretical here: `torch.cumsum` forms one delegate, and the single command
in it is `DSP_OP_RASTER_BLIT` -- the delegate is the `view_copy`, not the
scan, so a delegate count would have called it covered.

*Whether there is a portable fallback* is read from the `.pte` the real
pipeline writes, because the spelling in that file is not the spelling in the
EXIR graph. `torch.var` lowers to `aten.var.correction`, and the `.pte`
carries `aten::var.correction_out`; the functional-to-out-variant rewrite
happens inside `to_executorch()`, not in the graph the partitioner saw. Every
one of the twenty families resolves to a registered portable kernel, so none of
them is a functional gap -- they cost speed, not correctness.

*What the op would cost* is answered by lowering the composition it could be
made of and counting the commands that come out, because "no kernel exists"
is not the same claim as "an emitter would have nothing to call". Three of
them are compositions of command types the host already emits: a two-pass
variance is four commands and reproduces the torch fp16 composition bit for
bit, a clamp against tensor bounds is two selects and is bit-exact, and an
elu is four commands. `aten.prod.default` is not: the DSP has four reduction
subtypes and none multiplies, and the log/sum/exp substitute measures 3.2e-3
relative error in fp16 on top of a product that already overflows to infinity.

Every refusal here is paired with a positive control at the same geometry --
`relu`, which must reach `DSP_OP_UNARY` -- so a partitioner that refused
everything could not make this file green.
"""

import os
import pathlib
import re
import struct
import sys

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so its parent goes on the
# path; without it the editable install wins and points at another checkout.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.backends.hexagon.serialization import blob as _blob  # noqa: E402
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge_transform_and_lower,
)
from executorch.exir._serialize import _deserialize_pte_binary  # noqa: E402
from torch.export import export  # noqa: E402

F16 = torch.float16
_CONFIG = EdgeCompileConfig(_check_ir_validity=False)
_MAGIC = struct.pack("<I", _blob.BLOB_MAGIC)
_DELEGATE = torch.ops.higher_order.executorch_call_delegate

#: The vendored subtypes an emitter can reach, as they are declared.
#: `unary_ops.cc:14-31`, `eltwise_ops.cc:27-38`, `eltwise_ops.cc:2455-2460`.
UNARY_SUBTYPES = 17
BINARY_SUBTYPES = 12
REDUCTION_SUBTYPES = 4  # SUM, MAXIMUM, MEAN, MINIMUM -- no product


class _M(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _lowered(fn, args):
    """The manager, because `to_executorch()` hangs off it and not off the EP.

    The EXIR graph the partitioner saw and the program that gets written are
    not the same object, and the out-variant rewrite that decides the host-side
    spelling happens in between.
    """
    return to_edge_transform_and_lower(
        export(_M(fn), tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=_CONFIG,
    )


def _commands(manager):
    """The command types that reached the DSP, read from the delegates' blobs."""
    program = manager.exported_program()
    found = []
    for node in program.graph_module.graph.nodes:
        if node.target is _DELEGATE:
            submodule = program.graph_module.get_submodule(node.args[0].target)
            raw = submodule._processed_bytes
            if isinstance(raw, (bytes, bytearray)) and bytes(raw[:4]) == _MAGIC:
                _header, commands = read_blob(bytes(raw))
                found.extend(command.type for command in commands)
    return found


def _host_operators(tmp_path, fn, args):
    """The operator names the runtime will look up, from the real `.pte`.

    Not the EXIR spelling: the functional-to-out-variant rewrite happens in
    `to_executorch()`, so the graph says `aten.var.correction` and the file
    says `aten::var.correction_out`.
    """
    program = _lowered(fn, args).to_executorch()
    path = tmp_path / "probe.pte"
    program.save(os.fspath(path))
    pte = _deserialize_pte_binary(path.read_bytes())
    names = []
    for plan in pte.program.execution_plan:
        for operator in plan.operators:
            if operator.name == "executorch_call_delegate":
                continue
            names.append(
                operator.name
                + ("." + operator.overload if operator.overload else "")
            )
    return names


def _named(fn, args):
    manager = _lowered(fn, args)
    names = {
        value: key
        for key, value in vars(hexagon_ops).items()
        if key.startswith("DSP_OP_")
    }
    commands = _commands(manager)
    return len(commands), [names.get(t, t) for t in commands]


#: (label, forward, inputs, the host-side spelling the `.pte` must carry)
#:
#: The last element is what makes the row about a fallback rather than about
#: the DSP: it is the name `operator_registry.cpp:265`'s `strcmp` will match
#: or fail to match.
_ROWS = [
    ("erf", lambda a: torch.erf(a), "aten::erf.out"),
    ("prod", lambda a: torch.prod(a), "aten::prod.out"),
    ("var", lambda a: torch.var(a), "aten::var.correction_out"),
    ("expm1", lambda a: torch.expm1(a), "aten::expm1.out"),
    ("zeros_like", lambda a: torch.zeros_like(a), "aten::full_like.out"),
    ("eq.Tensor", lambda a: torch.eq(a, a), "aten::eq.Tensor_out"),
]

#: `cumsum` is deliberately absent from the rows above. It forms a delegate and
#: so would fail `count == 0`, because the single command in that delegate is
#: the blit its flattened input needed, not a scan. Putting it in the row would
#: assert a falsehood to keep a list tidy; the test below pins what it does.
#:
#: Five families have left this table the other way round: `argmax`/`argmin`,
#: `elu`, `clamp.Tensor` and `gt.Tensor`. Each row said the family reaches no
#: command and keeps a portable host spelling, and both halves were true when
#: the row was written. Each then got an emitter, and once a node delegates
#: there is no host spelling left either, so the row is false in both halves at
#: once. No row is deleted merely to keep the list tidy: each claim inverted
#: into a falsehood, so each is replaced by the positive claim below, which
#: pins the command types rather than a count of zero. `eq.Tensor` is still a
#: row: only `gt` and `lt` have a kernel, and the other four comparisons do not
#: have one at any width.


@pytest.mark.parametrize(
    "label,forward,host_operator", _ROWS, ids=[row[0] for row in _ROWS]
)
def test_the_family_reaches_no_dsp_command_but_has_a_portable_fallback(
    tmp_path, label, forward, host_operator
):
    """Two halves, because either alone would be a different claim.

    Zero commands is a statement about the DSP and says nothing about whether
    the model runs. The `.pte` spelling is the statement about the runtime:
    the lookup is an exact `strcmp`, so an absent name is `OperatorMissing`
    at run time and not a slow path.
    """
    args = (torch.randn(1, 64, 8, 8, dtype=F16),)
    count, _ = _named(forward, args)
    assert count == 0, f"{label} now reaches {count} DSP commands"
    operators = _host_operators(tmp_path, forward, args)
    assert host_operator in operators, (
        f"{label} writes {operators}, and the runtime looks these up by name"
    )


#: The rows above, as the claim that is true instead. The command types are
#: pinned rather than a count, because "reaches six commands" would still pass
#: if the composition were rebuilt out of the wrong six.
_WIRED = [
    ("argmax", torch.argmax, ["DSP_OP_ARGMAX_FP16"]),
    ("argmin", torch.argmin, ["DSP_OP_ARGMAX_FP16"]),
    ("clamp.Tensor", lambda a: torch.clamp(a, min=a), ["DSP_OP_BINARY_ELEMENTWISE"]),
    (
        "gt.Tensor",
        lambda a: torch.gt(a, a),
        ["DSP_OP_BINARY_ELEMENTWISE", "DSP_OP_SELECT"],
    ),
    (
        "lt.Tensor",
        lambda a: torch.lt(a, a),
        ["DSP_OP_BINARY_ELEMENTWISE", "DSP_OP_SELECT"],
    ),
    (
        "elu",
        lambda a: torch.nn.functional.elu(a),
        [
            "DSP_OP_UNARY",
            "DSP_OP_BINARY_ELEMENTWISE",
            "DSP_OP_BINARY_ELEMENTWISE",
            "DSP_OP_UNARY",
            "DSP_OP_UNARY",
            "DSP_OP_BINARY_ELEMENTWISE",
        ],
    ),
]


@pytest.mark.parametrize("label,forward,expected", _WIRED, ids=[r[0] for r in _WIRED])
def test_the_family_left_this_table_and_the_test_pins_what_it_does_now(
    tmp_path, label, forward, expected
):
    """Each row this replaced, stated as the claim that is true instead.

    Zero commands and a portable spelling were both correct for every one of
    these when the row was written, and both are false now in the same
    direction: the node reaches commands, and the runtime has no operator of
    its own left to look up because the whole graph became the delegate.
    """
    args = (torch.randn(1, 64, 8, 8, dtype=F16),)
    count, names = _named(forward, args)
    assert count == len(expected), f"{label} reaches {count} commands"
    assert names == expected
    assert _host_operators(tmp_path, forward, args) == []


def test_the_control_delegates_so_a_refusing_partitioner_cannot_pass_this_file():
    """The positive control every refusal above is measured against."""
    count, names = _named(lambda a: torch.relu(a), (torch.randn(1, 64, 8, 8, dtype=F16),))
    assert count == 1
    assert names == ["DSP_OP_UNARY"]


def test_a_delegate_is_not_evidence_that_the_op_reached_the_dsp():
    """Why this file reads command streams and not delegate counts.

    `cumsum` is the standing case: the graph forms one delegate, and the only
    command in it is a blit, because the delegate is the `view_copy` the
    flattened input needed. Reading the delegate count would score the scan as
    covered.
    """
    manager = _lowered(
        lambda a: torch.cumsum(a.flatten(), 0),
        (torch.randn(1, 64, 8, 8, dtype=F16),),
    )
    program = manager.exported_program()
    delegates = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is _DELEGATE
    ]
    assert len(delegates) == 1
    assert _commands(manager) == [3], "the delegate is the view_copy, not the scan"


def _run(fn, args):
    manager = _lowered(fn, args)
    program = manager.exported_program()
    for node in program.graph_module.graph.nodes:
        if node.target is _DELEGATE:
            submodule = program.graph_module.get_submodule(node.args[0].target)
            raw = submodule._processed_bytes
            if isinstance(raw, (bytes, bytearray)) and bytes(raw[:4]) == _MAGIC:
                return bytes(raw)
    return None


def test_var_is_already_paid_for_by_a_four_command_composition():
    """The cheapest gap in the list, and the reason it is the cheapest.

    A two-pass variance lowers to four commands of types the host already
    emits, and the DSP stream reproduces the torch fp16 composition exactly --
    the whole difference from an fp32 reference is the fp16 accumulator, which
    the composition has too. So `aten.var.correction` needs no kernel and no
    emitter: it needs to be rewritten into the form the partitioner already
    accepts.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=F16)

    def two_pass(a):
        return ((a - a.mean(-1, keepdim=True)) ** 2).mean(-1)

    raw = _run(two_pass, (x,))
    assert raw is not None, "the two-pass form no longer delegates"
    _header, commands = read_blob(raw)
    assert [command.type for command in commands] == [29, 19, 4, 29], (
        "two-pass variance: REDUCTION, BINARY, UNARY, REDUCTION"
    )
    got = np.frombuffer(execute(raw, [x.numpy()])[0], dtype=np.float16).reshape(-1)
    want = two_pass(x).numpy()
    assert np.array_equal(got, want), (
        f"max abs difference {np.abs(got.astype(np.float32) - want.astype(np.float32)).max()}"
    )


def test_clamp_against_tensor_bounds_is_two_selects_and_is_bit_exact():
    """The other zero-C++ entry, with a cost the row has to name.

    Both conditions are host-computed and enter the blob as inputs, so the
    emitter needs a bool operand the way `where` already needs one. That is
    the same qualification `where` carries, not a new class of risk.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=F16)
    lo, hi = (x - 0.5).clone(), (x + 0.5).clone()
    below, above = x < lo, x > hi

    def composition(a, lower, upper, is_below, is_above):
        return torch.where(is_below, lower, torch.where(is_above, upper, a))

    raw = _run(composition, (x, lo, hi, below, above))
    assert raw is not None
    header, commands = read_blob(raw)
    assert [command.type for command in commands] == [26, 26], "two DSP_OP_SELECT"
    assert header.n_inputs == 5, "three tensors and two host-computed conditions"
    got = np.frombuffer(
        execute(raw, [x.numpy(), lo.numpy(), hi.numpy(), below.numpy(), above.numpy()])[0],
        dtype=np.float16,
    ).reshape(x.shape)
    want = torch.clamp(x, min=lo, max=hi).numpy()
    assert np.array_equal(got, want)


def test_elu_is_four_commands_with_no_new_kernel():
    """No unary or binary subtype is an elu; the composition is what makes it cheap.

    The measured error is 9.8e-4 absolute, which is the band this backend
    already delegates `cos` at (8.9e-4, OP_GAPS.md section 3) rather than a
    tighter one it would have to justify separately.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 64, 1, 48, dtype=F16) * 2.0
    condition = x > 0
    alpha = 1.6732632423543772848170429916717

    def composition(a, is_positive):
        return torch.where(is_positive, a, (torch.exp(a) - 1.0) * alpha)

    raw = _run(composition, (x, condition))
    assert raw is not None
    _header, commands = read_blob(raw)
    assert [command.type for command in commands] == [4, 19, 19, 26]
    got = np.frombuffer(execute(raw, [x.numpy(), condition.numpy()])[0], dtype=np.float16)
    want = torch.nn.functional.elu(x, alpha=alpha).numpy().reshape(-1)
    error = np.abs(got.astype(np.float32) - want.astype(np.float32)).max()
    assert error < 2e-3, f"max abs error {error}"


def test_the_vendored_enums_have_no_erf_elu_or_product_reduction():
    """The claim the three compositions above are measured against.

    Read from the sources rather than from a table: `unary_ops.cc:14-31`,
    `eltwise_ops.cc:27-38` and `eltwise_ops.cc:2455-2460`.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "third-party" / "mnn-htp-ops"
    unary = (root / "src" / "dsp" / "unary_ops.cc").read_text()
    eltwise = (root / "src" / "dsp" / "eltwise_ops.cc").read_text()

    # Scope to the enum, and count entries rather than occurrences: every name
    # appears once in its declaration and again in each dispatch arm, so a
    # substring count over the file returns 96 where the enum has 17, and two
    # different enums share the HTP_OPS_BINARY_ prefix.
    def declared(text, prefix, enum_name):
        body = re.search(rf"\{{([^}}]*)\}}\s*{enum_name};", text, re.S).group(1)
        return set(re.findall(rf"{prefix}(\w+)\s*=\s*\d+", body))

    unary_types = declared(unary, "HTP_OPS_UNARY_", "HtpOpsUnaryOpType")
    binary_types = declared(eltwise, "HTP_OPS_BINARY_", "HtpOpsBinaryOpType")
    reduction_types = declared(eltwise, "HTP_OPS_REDUCTION_", "HtpOpsReductionOpType")

    assert len(unary_types) == UNARY_SUBTYPES, sorted(unary_types)
    assert len(binary_types) == BINARY_SUBTYPES, sorted(binary_types)
    assert "ERF" not in unary_types and "ELU" not in unary_types
    assert reduction_types == {"SUM", "MAXIMUM", "MEAN", "MINIMUM"}
    assert len(reduction_types) == REDUCTION_SUBTYPES, "a fifth would be a product walk"
