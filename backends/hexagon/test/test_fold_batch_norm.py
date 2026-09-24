# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folding a batch norm into the convolution before it, and what it buys.

The DSP has no batch-norm command, so a batch norm that reaches the partitioner
falls back to a portable kernel, and because it is not a Hexagon op it also cuts
the delegate chain in two -- once per block. What these tests pin is the size of
that cost and the fact that folding removes it without moving anything else:
four conv/BN/relu blocks lower to five delegates and four portable batch norms
without the pass and to one delegate and none with it, carrying the same
commands; every blob built from a graph with no batch norm in it is byte for
byte the same either way; and the folded graph answers torch inside the band
this backend already accepts for a fp16 convolution.

The two ways this can be wrong are pinned separately, because neither is visible
in a green test that only lowers a model and compares a graph. A fold that did
not happen looks exactly like a fold that did, so the delegate count is asserted
rather than the output. A fold that *did* happen but computed the wrong slope is
silent, so one case folds with a deliberately wrong running mean and requires the
check the other cases pass to fail on it.
"""

import hashlib
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.fspath(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon.fold_batch_norm import (  # noqa: E402
    FoldBatchNormIntoConv,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.backends.transforms.fuse_batch_norm_with_conv import (  # noqa: E402
    FuseBatchNormWithConvPass,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.pass_base import (  # noqa: E402
    ExportPass,
    ExportedProgramPassBase,
    ExportedProgramPassResult,
)
from executorch.exir.passes.remove_unused_parameters_pass import (  # noqa: E402
    remove_unused_parameters_pass,
)
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

#: The band a fp16 convolution is already held to in this suite: `test_pool.py`
#: and `test_mean.py` compare with rtol=2e-3, atol=2e-3, and the README's
#: measured conv and pool agreement is within one fp16 ULP. Folding does not
#: introduce a new approximation -- it is the same affine map applied one step
#: earlier -- so it is held to the band the unfolded path is held to.
_BAND = 2e-3

#: The same band as a per-element maximum, which is the form these cases use.
_MAXABS = 2e-3


def one_step(expected) -> float:
    """One fp16 step at the magnitude of `expected`, plus one relative step.

    The tensors a delegate sees are fp16, so the finest distinction the backend
    can draw at the output's own scale is this. A check tighter than one step is
    asking a fp16 kernel for a fp32 answer.
    """
    peak = float(np.abs(np.asarray(expected, dtype=np.float32)).max())
    return float(np.spacing(np.float16(peak))) + 2.0**-10 * peak

_SHAPE = (1, 3, 16, 16)


def _randomise_batch_norms(module) -> None:
    """Give every batch norm statistics that are not the identity.

    A freshly constructed `BatchNorm2d` is `weight=1, bias=0, running_mean=0,
    running_var=1`, so it scales its input by `1/sqrt(1+eps)` -- 1-5e-6. Folding
    that is very nearly a no-op, and a fold that computed the wrong scale, the
    wrong shift, or nothing at all would still land inside any tolerance worth
    setting. Every number in this file is therefore measured against batch norms
    drawn here, and `test_the_fixture_batch_norms_are_not_the_identity` below is
    the guard that this stays true.
    """
    generator = torch.Generator().manual_seed(0)
    for batch_norm in module.modules():
        if not isinstance(batch_norm, torch.nn.BatchNorm2d):
            continue
        with torch.no_grad():
            if batch_norm.weight is not None:
                batch_norm.weight.uniform_(0.5, 1.5, generator=generator)
                batch_norm.bias.uniform_(-1.0, 1.0, generator=generator)
            batch_norm.running_mean.normal_(0.0, 0.5, generator=generator)
            batch_norm.running_var.uniform_(0.5, 2.0, generator=generator)


class _ConvBNReLU(torch.nn.Module):
    """The classic block, repeated. No skip, so every convolution has one user."""

    def __init__(self, blocks=4, width=16):
        super().__init__()
        torch.manual_seed(0)
        layers = []
        channels = 3
        for _ in range(blocks):
            layers += [
                torch.nn.Conv2d(channels, width, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(width),
                torch.nn.ReLU(),
            ]
            channels = width
        self.body = torch.nn.Sequential(*layers)
        _randomise_batch_norms(self)

    def forward(self, x):
        return self.body(x)


class _ConvBN(torch.nn.Module):
    """One block, no rectifier, for the cases about what the fold refuses."""

    def __init__(self, bias=False, affine=True, users=1):
        super().__init__()
        torch.manual_seed(0)
        self.users = users
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=bias)
        self.bn = torch.nn.BatchNorm2d(16, affine=affine)
        _randomise_batch_norms(self)

    def forward(self, x):
        y = self.conv(x)
        if self.users > 1:
            return self.bn(y) + y
        return self.bn(y)


class _BatchNormAlone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        return self.bn(x)


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _blobs(program):
    return [
        bytes(program.graph_module.get_submodule(node.args[0].target)._processed_bytes)
        for node in _delegates(program)
    ]


def _lower(module, inputs, fold=False, unpruned=False, unrepointed=False):
    return _lower_program(
        export(module, inputs), fold=fold, unpruned=unpruned, unrepointed=unrepointed
    )


class _ProgramShapedFold(ExportedProgramPassBase):
    """The fold and the prune, without the output-spec re-pointing.

    The control for that re-pointing: it is the fold as it stood before the
    helper existed, and the case below is what it fails on.
    """

    def call(self, exported_program):
        upstream = FuseBatchNormWithConvPass(exported_program)
        upstream(exported_program.graph_module)
        remove_unused_parameters_pass(exported_program)
        return ExportedProgramPassResult(exported_program, True)


class _UnprunedFold(ExportPass):
    """The fold alone, as a graph-module pass: the control for the pruning.

    It is upstream's own shape with upstream's own behaviour, so what it leaves
    behind is what a caller gets from the rewrite without this module's cleanup.
    """

    def __init__(self, exported_program):
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module):
        upstream = FuseBatchNormWithConvPass(self.exported_program)
        return upstream(graph_module)


def _lower_program(program, fold=False, unpruned=False, unrepointed=False):
    if unpruned:
        passes = [_UnprunedFold(program)]
    elif unrepointed:
        passes = [_ProgramShapedFold()]
    elif fold:
        passes = [FoldBatchNormIntoConv()]
    else:
        passes = []
    return to_edge_transform_and_lower(
        program,
        transform_passes=passes,
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _batch_norm_nodes(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and "batch_norm" in str(node.target)
    ]


def _portable_batch_norms(program):
    """The batch norms the partitioner left on a portable kernel."""
    delegated = set()
    for call in _delegates(program):
        lowered = program.graph_module.get_submodule(call.args[0].target)
        for node in lowered.original_module.graph_module.graph.nodes:
            delegated.add(node.name)
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and "batch_norm" in str(node.target)
        and node.name not in delegated
    ]


def _command_types(blob):
    return [command.type for command in read_blob(blob)[1]]


def _chain(program, x):
    """Run the delegates the way the runtime does, one after the other.

    A split graph has more than one blob, and comparing only the first one's
    output against the model's is how a wrong answer gets reported as a numerics
    failure: `execute` runs one blob, so the blobs have to be chained.
    """
    activation = x.numpy().astype(np.float16)
    for blob in _blobs(program):
        activation = execute(blob, [activation])[0]
    return np.frombuffer(activation, dtype=np.float16)


def _maxabs(got, expected):
    return float(
        np.abs(got.astype(np.float32) - expected.astype(np.float32).reshape(-1)).max()
    )


# --- What the fold buys -------------------------------------------------------


def test_a_conv_batch_norm_relu_stack_is_one_delegate_instead_of_five():
    """The whole point: the batch norms are what splits the chain."""
    model = _ConvBNReLU().eval()
    x = torch.randn(*_SHAPE)

    without = _lower(model, (x,))
    with_fold = _lower(model, (x,), fold=True)

    assert len(_delegates(without)) == 5, (
        f"the unfolded stack is expected to be five delegates -- one per "
        f"convolution plus the trailing rectifier -- and it is "
        f"{len(_delegates(without))}; the premise of this test moved"
    )
    assert len(_delegates(with_fold)) == 1, (
        f"the folded stack is expected to be a single delegate and it is "
        f"{len(_delegates(with_fold))}"
    )
    assert len(_portable_batch_norms(without)) == 4
    assert _portable_batch_norms(with_fold) == []
    assert len(_batch_norm_nodes(with_fold)) == 0


def test_the_fixture_batch_norms_are_not_the_identity():
    """The tolerance below is meaningless against a batch norm that does nothing.

    A fresh `BatchNorm2d` scales by 1/sqrt(1+eps), which is 1-5e-6: a fold that
    produced the wrong scale -- or none at all -- would pass any band. This pins
    how far the fixtures are from that, and checks the two sides of the argument
    at once: folding the real statistics moves the output, and folding the wrong
    ones moves it further.
    """
    model = _ConvBNReLU(blocks=1).eval()
    x = torch.randn(*_SHAPE)
    batch_norm = model.body[1]
    scale = batch_norm.weight.detach() / torch.sqrt(
        batch_norm.running_var + batch_norm.eps
    )
    assert float((scale - 1).abs().max()) > 0.1, (
        "the fixture's batch norm is an identity, so the band proves nothing"
    )
    assert float(batch_norm.bias.detach().abs().max()) > 0.1

    with torch.no_grad():
        expected = model(x).numpy().reshape(-1)
    folded = _chain(_lower(model, (x,), fold=True), x)
    plain = _chain(_lower(model, (x,)), x)
    assert _maxabs(folded, plain) > _BAND, (
        "folding the real statistics did not move the output, so the fixture "
        "batch norm is too close to the identity"
    )
    assert _maxabs(folded, expected) < _MAXABS

    batch_norm.running_mean += 0.5
    wrong = _chain(_lower(model, (x,), fold=True), x)
    assert _maxabs(wrong, plain) > _BAND, "a wrong fold passed for a right one"


def test_the_fold_moves_the_round_trips_and_not_the_commands():
    """The DSP work is the same; only the number of crossings changed."""
    model = _ConvBNReLU().eval()
    x = torch.randn(*_SHAPE)
    without = _lower(model, (x,))
    with_fold = _lower(model, (x,), fold=True)

    def tally(program):
        counts = {}
        for blob in _blobs(program):
            for kind in _command_types(blob):
                counts[kind] = counts.get(kind, 0) + 1
        return counts

    assert tally(without) == tally(with_fold), (
        "folding changed the commands rather than only their grouping: "
        f"{tally(without)} vs {tally(with_fold)}"
    )
    assert sum(len(_command_types(blob)) for blob in _blobs(without)) == 20
    assert sum(len(_command_types(blob)) for blob in _blobs(with_fold)) == 20


def test_the_folded_convolution_agrees_with_torch():
    """The folded chain, against torch, on a batch norm that is not the identity.

    The band is the backend's own: fp16 tensors, so one step at the output's
    magnitude, plus one fp16 relative step. Stating it that way is what makes it
    a measurement -- a fixed 2e-3 would be most of a step at this magnitude, and
    would say nothing about whether the scale was applied.

    There is no "and it differs from the unfolded chain" here, because the
    unfolded chain cannot be run by `_chain` at all (see below); that the fold
    moved the answer is the fixture test's job.
    """
    model = _ConvBNReLU().eval()
    x = torch.randn(*_SHAPE)
    with torch.no_grad():
        expected = model(x).numpy().reshape(-1).astype(np.float16)

    folded = _chain(_lower(model, (x,), fold=True), x)
    assert _maxabs(folded, expected) < one_step(expected), (
        f"the folded chain is {_maxabs(folded, expected)} from torch, more than "
        f"one fp16 step at this magnitude"
    )


def test_the_blob_level_chain_does_not_run_the_portable_batch_norm():
    """Why this file does not take the baseline's numerics from `_chain`.

    `_chain` drives the blobs the way `execute` does, one delegate at a time, so
    a portable node sitting between two delegates is never executed. Against an
    identity batch norm that is invisible; against one that is not, it is a
    visible error rather than a rounding difference. The control is the same
    convolutions and rectifiers with the batch norms deleted, which is what the
    chain actually computes -- and it is only the deleted-batch-norm model that
    the chain agrees with.
    """
    model = _ConvBNReLU().eval()
    x = torch.randn(*_SHAPE)
    stripped = torch.nn.Sequential(
        *[m for m in model.body if not isinstance(m, torch.nn.BatchNorm2d)]
    ).eval()
    with torch.no_grad():
        expected = model(x).numpy().reshape(-1)
        no_batch_norm = stripped(x).numpy().reshape(-1)

    plain = _chain(_lower(model, (x,)), x).astype(np.float32)
    assert _maxabs(plain, no_batch_norm) < _MAXABS, (
        "the chain did not reproduce the batch-norm-free model, so it runs more "
        "than the delegates after all"
    )
    assert _maxabs(plain, expected) > _BAND, (
        "the chain matched the whole model, so it does run the portable nodes"
    )


# --- What the fold must not touch --------------------------------------------


class _ConvStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.relu = torch.nn.ReLU()
        self.b = torch.nn.Conv2d(16, 16, 3, padding=1, bias=False)

    def forward(self, x):
        return self.b(self.relu(self.a(x)))


class _Pool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)


class _Mm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(32, 32) * 0.3)

    def forward(self, x):
        return torch.mm(x, self.weight)


#: Graphs with no batch norm in them, one per command family the pass could
#: plausibly disturb. Each is lowered with the pass present and absent.
_UNTOUCHED = [
    ("conv_stack", _ConvStack(), (1, 3, 16, 16)),
    ("maxpool_64", _Pool(), (1, 64, 8, 8)),
    ("mm_32", _Mm(), (4, 32)),
]


def test_a_graph_with_no_batch_norm_lowers_to_the_same_bytes():
    """A pass in the list must not move a graph it has nothing to do with."""
    for name, module, shape in _UNTOUCHED:
        torch.manual_seed(0)
        args = (torch.randn(*shape),)
        without = _blobs(_lower(module.eval(), args))
        with_fold = _blobs(_lower(module.eval(), args, fold=True))
        assert len(without) == len(with_fold), f"{name}: delegate count moved"
        for index, (left, right) in enumerate(zip(without, with_fold)):
            assert (
                hashlib.sha256(left).hexdigest() == hashlib.sha256(right).hexdigest()
            ), (
                f"{name}: blob {index} moved; the fold has nothing to do here"
            )
            assert read_blob(left)[1] == read_blob(right)[1] or [
                (c.type, list(c.params)) for c in read_blob(left)[1]
            ] == [(c.type, list(c.params)) for c in read_blob(right)[1]]


def test_the_callers_own_module_is_not_touched():
    """The pass rewrites the program it is handed, not the module it came from."""
    model = _ConvBNReLU(blocks=1).eval()
    x = torch.randn(*_SHAPE)
    with torch.no_grad():
        before = model(x).clone()
    _lower(model, (x,), fold=True)
    with torch.no_grad():
        after = model(x)
    assert torch.equal(before, after), "the fold reached back into the caller's module"
    assert model.body[1].running_mean is not None


# --- The failure that is silent ----------------------------------------------


def test_a_fold_computed_from_the_wrong_batch_norm_parameters_is_caught():
    """The teeth.

    A fold that computed the wrong slope produces a graph that lowers cleanly,
    runs on the DSP and answers a plausible-looking wrong number. The check the
    other cases pass has to fail on it, or those cases are decoration. The
    running mean here is moved by +0.5 after export, so the fold is arithmetic
    that is right for parameters the model does not have.
    """
    model = _ConvBNReLU().eval()
    x = torch.randn(*_SHAPE)
    with torch.no_grad():
        expected = model(x).numpy().reshape(-1).astype(np.float16)

    correct = _chain(_lower(model, (x,), fold=True), x)
    assert _maxabs(correct, expected) < _MAXABS

    program = export(model, (x,))
    for name, buffer in program.named_buffers():
        if name.endswith("running_mean"):
            buffer.add_(0.5)
    corrupted = _chain(_lower_program(program, fold=True), x)
    worst = _maxabs(corrupted, expected)
    assert worst > _MAXABS, (
        f"a fold from a running mean 0.5 away still agreed with torch to "
        f"{worst}; the comparison these cases rest on cannot see a wrong fold"
    )


# --- What stays behind, and why ----------------------------------------------


def test_the_refusals_keep_their_node_and_a_control_at_each_geometry_folds():
    """Every refusal is paired with the case that must fold.

    A refusal asserted on its own is satisfied by a pass that refuses
    everything, so each of these sits next to a graph that differs in the one
    way that matters and must lose its batch norm.
    """
    x = torch.randn(*_SHAPE)
    cases = [
        ("an affine=False batch norm", _ConvBN(affine=False), False),
        ("a convolution a residual branch also reads", _ConvBN(users=2), False),
        ("a batch norm with no convolution before it", _BatchNormAlone(), False),
        ("the plain block, as the control", _ConvBN(), True),
    ]
    for name, module, folds in cases:
        lowered = _lower(module.eval(), (x,), fold=True)
        remaining = len(_portable_batch_norms(lowered))
        if folds:
            assert remaining == 0, f"{name}: should have folded and did not"
            assert _batch_norm_nodes(lowered) == []
        else:
            assert remaining == 1, f"{name}: should have kept its node and folded"
            assert len(_batch_norm_nodes(lowered)) == 1


def test_a_batch_norm_that_is_the_graph_output_leaves_a_lowerable_program():
    """The case that decides the pass's shape, with the other shape as the control.

    When the batch norm is the last node, folding renames the graph's output from
    the batch norm's `getitem` to the convolution, and `graph_signature`'s output
    specs go stale with it. `ExportedProgramPassManager` rebuilds that signature
    only for graph-module passes; the `ExportedProgramPassBase` branch just
    recompiles, so the same rewrite written that way dies at the end of the
    transform in

        SpecViolationError: User output aten_convolution_default is not in the
        correct order or is not found in the exported program's user_output list

    which reads as a bug in the model rather than in the fold. The control below is
    that other shape, written out, so this is a measurement rather than a claim
    about what the pass manager rebuilds.
    """
    model = _ConvBN().eval()
    x = torch.randn(*_SHAPE)
    with torch.no_grad():
        expected = model(x).numpy().reshape(-1).astype(np.float16)

    without = _lower(model, (x,))
    with_fold = _lower(model, (x,), fold=True)
    # The delegate count does not move here -- one either way -- so what this
    # case adds is that the portable batch norm is gone and the answer is right.
    assert len(_delegates(without)) == len(_delegates(with_fold)) == 1
    assert len(_portable_batch_norms(without)) == 1
    assert _portable_batch_norms(with_fold) == []
    assert _maxabs(_chain(with_fold, x), expected) < _MAXABS

    with pytest.raises(Exception, match="not in the correct order"):
        _lower(model, (x,), unrepointed=True)


def test_the_unfused_convolution_weight_is_not_left_in_the_program():
    """The fold leaves a dead weight behind, and the prune is what removes it.

    Folding points the convolution at a new attribute, so the placeholder holding
    the original weight stops being read. An unused parameter placeholder is
    still in the program's signature and is still serialized, so without the
    prune the `.pte` carries a second fp32 copy of every convolution weight --
    on a real residual network, the whole weight payload twice. The control below
    is the fold without that cleanup, and the entry it leaves behind is checked
    against the *original* weight rather than merely counted, because a fused
    weight under the old name would be a different and equally wrong thing.
    """
    model = _ConvBNReLU(blocks=1).eval()
    x = torch.randn(*_SHAPE)
    original = model.body[0].weight.detach().clone()
    bn = model.body[1]
    fused, _ = torch.nn.utils.fusion.fuse_conv_bn_weights(
        original, None, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    )
    assert not torch.equal(original, fused), "the fused weight is the original one"

    folded = _lower(model, (x,), fold=True)
    assert "body.0.weight" not in folded.state_dict, (
        "the unfused convolution weight is still in the program"
    )

    unpruned = _lower(model, (x,), unpruned=True)
    left = unpruned.state_dict["body.0.weight"]
    assert left.dtype == torch.float32
    assert torch.equal(left, original), "the leftover entry is not the original weight"
    assert not torch.equal(left, fused)


def test_a_batch_norm_with_no_affine_weight_keeps_its_node():
    """`affine=False` has no weight operand, which the upstream pass refuses.

    The fold is still defined there -- the slope is one -- so this is a missed
    opportunity rather than a correctness rule, and it is recorded as one. It is
    not this backend's to fix: the pass is upstream and is imported unchanged.
    """
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE)
    module = _ConvBN(affine=False).eval()
    folded = _lower(module, (x,), fold=True)
    assert len(_batch_norm_nodes(folded)) == 1
    plain = _lower(module, (x,))
    assert [hashlib.sha256(b).hexdigest() for b in _blobs(plain)] == [
        hashlib.sha256(b).hexdigest() for b in _blobs(folded)
    ], "a batch norm that cannot fold must leave the lowering byte-identical"


def _training_flag_edge_program(flag):
    """A conv + batch norm whose node is the overload that carries `training`.

    `to_edge` never produces this from an eval module -- it legalizes
    `aten.batch_norm` to `_native_batch_norm_legit_no_training` -- so the node
    is built by rewriting the overload the eval graph did produce and inserting
    the flag, which is the shape the upstream pass matches either way.
    """
    module = _ConvBN().eval()
    module.bn.running_mean.fill_(0.25)
    module.bn.running_var.fill_(1.75)
    program = to_edge_transform_and_lower(
        export(module, (torch.randn(*_SHAPE),)),
        partitioner=[],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    graph_module = program.graph_module
    patched = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and "_native_batch_norm_legit_no_training" in str(
            node.target
        ):
            args = list(node.args)
            node.target = exir_ops.edge.aten.native_batch_norm.default
            node.args = tuple(args[:5] + [flag] + args[5:])
            patched += 1
    graph_module.recompile()
    assert patched == 1
    return program


def test_a_batch_norm_that_trains_on_batch_statistics_is_left_alone():
    """The gate, with its counterexample: a control that folds at the same shape.

    The two readings of the node are not close. On this graph the batch
    statistics and the running statistics put the convolution's output 2.34
    apart, so a fold onto the running statistics is a wrong answer the size of
    the signal, and nothing in the lowering would report it.
    """
    folded = (
        FoldBatchNormIntoConv()(_training_flag_edge_program(False)).exported_program
    )
    kept = FoldBatchNormIntoConv()(_training_flag_edge_program(True)).exported_program
    assert len(_batch_norm_nodes(folded)) == 0, "training=False should fold"
    assert len(_batch_norm_nodes(kept)) == 1, "training=True must not fold"

    module = _ConvBN().eval()
    module.bn.running_mean.fill_(0.25)
    module.bn.running_var.fill_(1.75)
    x = torch.randn(*_SHAPE)
    with torch.no_grad():
        conv_out = module.conv(x)
        batch_statistics = torch.nn.functional.batch_norm(
            conv_out,
            module.bn.running_mean,
            module.bn.running_var,
            module.bn.weight,
            module.bn.bias,
            training=True,
        )
        weight, bias = torch.nn.utils.fusion.fuse_conv_bn_weights(
            module.conv.weight,
            module.conv.bias,
            module.bn.running_mean,
            module.bn.running_var,
            module.bn.eps,
            module.bn.weight,
            module.bn.bias,
        )
        running_statistics = torch.nn.functional.conv2d(x, weight, bias, padding=1)
    apart = float((batch_statistics - running_statistics).abs().max())
    assert apart > 1.0, (
        f"the two readings agree to {apart}, so this gate is not protecting "
        "anything on this graph and the case needs statistics that differ"
    )


def test_the_upstream_pass_would_fold_a_training_batch_norm():
    """Why the gate exists, pinned where it can be seen to close.

    This is a defect pin, not a feature test: it asserts a property of
    `backends/transforms/fuse_batch_norm_with_conv.py`, which is imported
    unchanged and is not this backend's to edit. If upstream ever starts
    checking the flag, this is the test that fails, and the gate in
    `fold_batch_norm.py` has become redundant rather than load-bearing.
    """
    from executorch.backends.transforms.fuse_batch_norm_with_conv import (
        FuseBatchNormWithConvPass,
    )

    program = _training_flag_edge_program(True)
    before = len(_batch_norm_nodes(program))
    FuseBatchNormWithConvPass(program)(program.graph_module)
    assert before == 1
    assert len(_batch_norm_nodes(program)) == 0, (
        "the upstream pass no longer folds a batch norm whose training flag is "
        "set; the gate in backends/hexagon/fold_batch_norm.py can be removed"
    )


def test_a_graph_with_no_batch_norm_reports_no_change():
    """`modified` has to be the truth: upstream returns True unconditionally."""
    program = to_edge_transform_and_lower(
        export(_ConvStack().eval(), (torch.randn(1, 3, 16, 16),)),
        partitioner=[],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    result = FoldBatchNormIntoConv()(program)
    assert result.modified is False

    with_bn = to_edge_transform_and_lower(
        export(_ConvBN().eval(), (torch.randn(*_SHAPE),)),
        partitioner=[],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    assert FoldBatchNormIntoConv()(with_bn).modified is True
