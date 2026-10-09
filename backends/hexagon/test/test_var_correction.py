# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aten.var.correction as the two-pass variance, and what it is accurate to.

The op costs nothing new: the stream is REDUCTION mean, BINARY subtract,
UNARY square, REDUCTION mean, and the n/(n-correction) scale as the unary
multiply aten.mul.Scalar already writes. So the only question this file has to
answer is what the composition computes, and a delegate count cannot answer it
-- a stream of any shape would score the same. Every case here reads the
command types off the delegate blob and runs the blob.

Two of the numbers in this file are the ones a reader is most likely to take
out of context, so both are pinned where they cannot change silently. The
first is the overflow: a deviation past 255.9 squares to more than the fp16
maximum, so the composition answers inf where torch answers 20000. The second
is the offset case: the answer is a difference of two numbers of the data own
size, so a mean of 200 costs an order of magnitude more accuracy than a centred
row. Neither is fixed by a tolerance.

Nothing here ran a kernel. blob_interpreter.execute is a numpy model of the
command stream, so a pass here means the stream is shaped correctly and the
modelled arithmetic agrees. The numbers labelled as measured on hexagon-sim
in the workstream report came from a separate run of the same blobs through
the real kernels.
"""

import os
import pathlib
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
    HexagonOperatorSupport,
    HexagonPartitioner,
    _data_placeholders,
)
from executorch.backends.hexagon.serialization import blob as _blob  # noqa: E402
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.export import export  # noqa: E402

F16 = torch.float16
CONFIG = EdgeCompileConfig(_check_ir_validity=False)
DELEGATE = torch.ops.higher_order.executorch_call_delegate
REDUCTION = hexagon_ops.DSP_OP_REDUCTION
BINARY = hexagon_ops.DSP_OP_BINARY_ELEMENTWISE
UNARY = hexagon_ops.DSP_OP_UNARY
MEAN = hexagon_ops.REDUCTION_MEAN
SUB = hexagon_ops.BINARY_OP_TYPES["sub"]
SQUARE = hexagon_ops.UNARY_OP_TYPES["square"]
SCALE = hexagon_ops.UNARY_OP_TYPES["mul_scalar"]
VAR_CORRECTION = hexagon_ops.VAR_CORRECTION

#: params[1] names a unary variant, params[3] a binary one and params[3] of a
#: REDUCTION its op type, so the variant of every command is readable off the
#: blob. The count alone is too weak: six commands is what any composition of a
#: mean, a subtract, a square and two multiplies looks like, and it is also
#: what a stream computing the wrong thing entirely looks like.
_VARIANT_PARAM = {UNARY: 1, BINARY: 3, REDUCTION: 3}

#: The fp16 step at 1.0, which is the scale of the answer for unit-variance
#: data and therefore the natural unit for the bounds below.
STEP_AT_ONE = 2.0**-10


class _Callable(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _program(fn, args):
    return to_edge_transform_and_lower(
        export(_Callable(fn), tuple(args)),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()


def _blobs(program):
    out = []
    for node in program.graph_module.graph.nodes:
        if node.target is not DELEGATE:
            continue
        raw = program.graph_module.get_submodule(node.args[0].target)._processed_bytes
        # BLOB_MAGIC is an int, so a bytes comparison is the only one that
        # matches anything at all.
        assert isinstance(raw, (bytes, bytearray)) and bytes(raw[:4]) == struct.pack(
            "<I", _blob.BLOB_MAGIC
        ), "a delegate carries no blob; the reader would be reading nothing"
        out.append(bytes(raw))
    return out


def _stream(fn, args):
    """(command type, the param that names its variant) for every command."""
    found = []
    for raw in _blobs(_program(fn, args)):
        _header, commands = read_blob(raw)
        for command in commands:
            index = _VARIANT_PARAM.get(command.type)
            found.append(
                (command.type, list(command.params)[index] if index is not None else None)
            )
    return found


def _run(fn, args, feed):
    """The delegates outputs, through the model, as one flat fp16 array."""
    pieces = []
    for raw in _blobs(_program(fn, args)):
        pieces.append(np.frombuffer(execute(raw, feed)[0], dtype=np.float16))
    assert pieces, "nothing delegated, so there is nothing to run"
    return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def _bits(values):
    array = np.asarray(values, dtype=np.float16).reshape(-1)
    return np.ascontiguousarray(array).view(np.uint16)


def _x(seed=0, shape=(4, 16), scale=1.0, offset=0.0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32) * scale + offset
    ).to(F16)


def _verdict(support, program, index=0):
    nodes = [n for n in program.graph.nodes if n.op == "call_function"]
    return support.is_node_supported(program, nodes[index])


def test_a_partitioner_that_refused_everything_could_not_pass_this_file():
    # The control. relu is one command and the mean the variance is built on is
    # one command, so without them every assertion below would be about an empty
    # command stream.
    x = _x()
    assert _stream(torch.relu, (x,)) == [(UNARY, hexagon_ops.UNARY_OP_TYPES["clamp"])]
    assert _stream(lambda a: a.mean(-1, keepdim=True), (x,)) == [(REDUCTION, MEAN)]


def test_a_correction_of_zero_is_four_commands_and_any_other_is_five():
    # The triage measured four commands and called that the whole op. It is the
    # biased variance: n/(n-0) is exactly 1.0, and multiplying by it is a command
    # that cannot change the answer, so the emitter does not spend it. Every
    # other correction needs the scale, and that is the fifth command.
    x = _x()
    assert _stream(lambda a: torch.var(a, dim=-1, correction=0), (x,)) == [
        (REDUCTION, MEAN),
        (BINARY, SUB),
        (UNARY, SQUARE),
        (REDUCTION, MEAN),
    ]
    for correction in (1, 2, 4):
        assert _stream(
            lambda a: torch.var(a, dim=-1, correction=correction), (x,)
        ) == [
            (REDUCTION, MEAN),
            (BINARY, SUB),
            (UNARY, SQUARE),
            (REDUCTION, MEAN),
            (UNARY, SCALE),
        ], correction


@pytest.mark.parametrize(
    "label, fn, args",
    [
        ("dim=-1", lambda a: torch.var(a, dim=-1), (4, 16)),
        ("dim=0", lambda a: torch.var(a, dim=0), (4, 16)),
        ("no dim", lambda a: torch.var(a), (4, 16)),
        ("dim=(0,1)", lambda a: torch.var(a, dim=(0, 1)), (4, 16)),
        ("keepdim", lambda a: torch.var(a, dim=-1, keepdim=True), (4, 16)),
        ("dim=(1,2)", lambda a: torch.var(a, dim=(1, 2)), (2, 8, 16)),
    ],
)
def test_the_span_rules_are_the_ones_mean_already_uses(label, fn, args):
    # One contiguous span of dims, a missing dim meaning every dim, and keepdim
    # changing only the declared shape: the same three questions mean.dim answers,
    # because the variance is the mean command twice.
    x = _x(shape=args)
    stream = _stream(fn, (x,))
    assert stream[:4] == [
        (REDUCTION, MEAN),
        (BINARY, SUB),
        (UNARY, SQUARE),
        (REDUCTION, MEAN),
    ], label
    assert stream[4] == (UNARY, SCALE), label


def test_std_reaches_the_dsp_because_variance_does():
    # torch.std is sqrt of the variance, so the square root is the last unary
    # command and the view the export inserts is the blit. Before the variance
    # was emittable this node stayed on the portable kernel with the sqrt
    # stranded beside it; a coverage claim about std is a claim about this row.
    x = _x()
    stream = _stream(lambda a: torch.std(a, dim=-1), (x,))
    assert stream[4] == (UNARY, SCALE), stream
    assert (UNARY, hexagon_ops.UNARY_OP_TYPES["sqrt"]) in stream, stream
    # The last command is a view the export inserts between the two halves of
    # the pool, and it is last for a reason worth stating: if it were not, the
    # square root would be reading a layout the variance did not write.
    assert stream[-1][0] == hexagon_ops.DSP_OP_RASTER_BLIT, stream
    got = _run(lambda a: torch.std(a, dim=-1), (x,), [x.numpy()])
    want = torch.std(x, dim=-1).numpy().reshape(-1)
    assert np.abs(got.astype(np.float32) - want.astype(np.float32)).max() <= (
        2.0 * STEP_AT_ONE
    ), (got, want)


@pytest.mark.parametrize("correction", [16, 20, 100])
def test_a_correction_with_no_answer_stays_portable(correction):
    # n - correction is the denominator, and torch itself has no variance for a
    # correction at or past n. The control is the same node one correction below,
    # at the same geometry: a partitioner that refused every var could not pass.
    x = _x()
    assert _stream(lambda a: torch.var(a, dim=-1, correction=correction), (x,)) == [], (
        correction
    )
    assert len(_stream(lambda a: torch.var(a, dim=-1, correction=1), (x,))) == 5


def test_a_negative_correction_is_a_variance_and_is_emitted():
    # The other side of the denominator. n - (-1) is positive, so the scale is
    # n/(n+1) and there is an answer; refusing it would be the gate being written
    # for "correction > 0" rather than for "the denominator is positive".
    x = _x()
    stream = _stream(lambda a: torch.var(a, dim=-1, correction=-1), (x,))
    assert stream[4] == (UNARY, SCALE), stream
    got = _run(lambda a: torch.var(a, dim=-1, correction=-1), (x,), [x.numpy()])
    want = torch.var(x, dim=-1, correction=-1).numpy().reshape(-1)
    assert np.abs(got.astype(np.float32) - want.astype(np.float32)).max() <= (
        2.0 * STEP_AT_ONE
    ), (got, want)


def test_a_width_the_arena_does_not_hold_stays_portable():
    # An int64 activation is one byte too wide for every kernel here, and the
    # width gate refuses it before the variance own clauses are read. The control
    # is the same graph in fp32, which the arena holds by narrowing on entry.
    assert _stream(lambda a: torch.var(a, dim=-1), (torch.zeros(4, 16, dtype=torch.int64),)) == []
    assert _stream(lambda a: torch.var(a, dim=-1), (torch.zeros(4, 16, dtype=torch.float32),)) != []


def test_a_span_the_export_cannot_measure_stays_portable():
    # The n/(n-correction) scale is a command param, so a symbolic span length
    # has no number to write and the node keeps the portable kernel. Baked from
    # the traced example it would answer with the example length at run time.
    # The reduction is over the dynamic axis, not the static one beside it: a
    # variance over a [batch, 16] row has a span of sixteen and delegates, and
    # the control below says so, so a gate that refused every dynamic export
    # cannot pass this either.
    # Two rows, so the batch axis has two distinct traced extents and the export
    # can widen it; a single row pins the dim to its only value.
    x = _x(shape=(2, 16))
    batch = torch.export.Dim("batch", min=2, max=8)
    lowered = to_edge_transform_and_lower(
        export(
            _Callable(lambda a: torch.var(a, dim=0)),
            (x,),
            dynamic_shapes={"args": ({0: batch},)},
        ),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    inner = set()
    for node in lowered.graph_module.graph.nodes:
        if node.target is DELEGATE:
            inner.update(
                n.name
                for n in lowered.graph_module
                .get_submodule(str(node.args[0].target))
                .original_module.graph_module.graph.nodes
            )
    var_names = {
        n.name
        for n in lowered.graph_module.graph.nodes
        if n.op == "call_function" and n.target is VAR_CORRECTION
    }
    assert var_names, "the dynamic export produced no var node to judge"
    assert not (var_names & inner), "a var whose span length is symbolic delegated"

    # The control: the same dynamic input, reducing the axis that is a number.
    # The emitter scales by sixteen there, which it can write down, so this one
    # delegates and the other cannot.
    control = to_edge_transform_and_lower(
        export(
            _Callable(lambda a: torch.var(a, dim=-1)),
            (x,),
            dynamic_shapes={"args": ({0: batch},)},
        ),
        partitioner=[HexagonPartitioner()],
        compile_config=CONFIG,
    ).exported_program()
    control_inner = set()
    for node in control.graph_module.graph.nodes:
        if node.target is DELEGATE:
            control_inner.update(
                n.name
                for n in control.graph_module
                .get_submodule(str(node.args[0].target))
                .original_module.graph_module.graph.nodes
            )
    # A refused node stays at the top level of the lowered graph and a delegated
    # one does not, so membership is the whole question, and the names are the
    # inner graphs own -- the lowered graph keeps the outer graphs names.
    assert (var_names | control_inner) & inner == set(), (
        sorted((var_names | control_inner) & inner),
    )
    assert var_names & control_inner, (
        "the same dynamic input, reducing a static span, refused too",
        sorted(control_inner),
    )
    assert control_inner, "the control delegated nothing at all, so it proves nothing"


def test_the_two_predicates_are_the_ones_doing_the_refusing():
    # The emitter table alone would delegate all of these, so the refusals are
    # the predicates doing. Read through the operator support rather than by
    # counting delegates, and with the accepted case at the same geometry beside
    # each refusal, so an object that refuses everything cannot pass.
    x = _x()

    def verdict(fn):
        """What the partitioner answers for this graph own var node.

        Asked of the edge program rather than of the lowered one, because a
        lowered graph has already put every supported node inside a delegate:
        the node the gate refused stays at the top level and the node it
        accepted does not, so reading the lowered graph would be reading the
        gate own answer back as its input. Read through is_node_supported
        rather than by counting delegates, because a delegate count is blind to
        which clause turned the node away, which is the whole question here.
        """
        program = to_edge(
            export(_Callable(fn), (x,)), compile_config=CONFIG
        ).exported_program()
        support = HexagonOperatorSupport(_data_placeholders(program), program)
        nodes = [
            n
            for n in program.graph.nodes
            if n.op == "call_function" and n.target is VAR_CORRECTION
        ]
        assert len(nodes) == 1, [n.name for n in nodes]
        return support.is_node_supported(program, nodes[0])

    assert verdict(lambda a: torch.var(a, dim=-1))
    assert verdict(lambda a: torch.var(a, dim=-1, correction=-1))
    for bad in (16, 20):
        assert not verdict(
            lambda a: torch.var(a, dim=-1, correction=bad)
        ), bad


@pytest.mark.parametrize("n", [16, 64, 1024])
def test_the_blob_answers_within_one_fp16_step_of_torch(n):
    # The bound is the answer own scale rather than a number fitted to a seed,
    # and it is loose by a factor of three over the measured worst so that a
    # fourth seed cannot make it red. The assert that the error is also strictly
    # above zero is the one that stops this from passing vacuously.
    worst = 0.0
    for seed in range(4):
        x = _x(seed, (4, n))
        got = _run(lambda a: torch.var(a, dim=-1, correction=1), (x,), [x.numpy()])
        want = torch.var(x, dim=-1, correction=1).numpy().reshape(-1)
        worst = max(
            worst,
            float(np.abs(got.astype(np.float32) - want.astype(np.float32)).max()),
        )
    assert 0.0 < worst <= 3.0 * STEP_AT_ONE, worst


def test_the_centre_is_narrower_than_the_offset_and_that_is_the_point():
    # The answer is a difference of two numbers of the data own size, so the
    # subtraction is the step that costs. A centred row and a row whose mean is
    # 200 are the same op at the same shape, and the gap between them is the
    # claim this row is allowed to make: a variance is not as accurate as a
    # mean, and no tolerance here can change that.
    def worst_rel(offset):
        out = 0.0
        for seed in range(4):
            x = _x(seed, (4, 256), offset=offset)
            got = _run(
                lambda a: torch.var(a, dim=-1, correction=1), (x,), [x.numpy()]
            ).astype(np.float32)
            want = torch.var(x.float(), dim=-1, correction=1).numpy()
            out = max(out, float((np.abs(got - want) / want).max()))
        return out

    centred = worst_rel(0.0)
    shifted = worst_rel(200.0)
    assert 0.0 < centred < 1e-3, centred
    assert shifted > 4.0 * centred, (centred, shifted)
    assert shifted < 1e-2, shifted


def test_the_whole_gap_to_fp32_is_the_narrowing_not_the_accumulator():
    # The brief said the gap to fp32 is the fp16 accumulator. It is not, and the
    # decomposition is the measurement: stage (a) is the squared deviation held
    # in fp16, stage (b) is an fp32 accumulator reading those fp16 values, and
    # stage (c) is the final narrowing of the answer. At n=4096 (b) is 2.6e-05
    # and (c) alone accounts for the whole 3.5e-04, so the accumulator is not
    # where the accuracy goes and a fp32 answer is not the thing being lost.
    x = _x(0, (4, 4096))
    got = _run(
        lambda a: torch.var(a, dim=-1, correction=1), (x,), [x.numpy()]
    ).astype(np.float32)
    want = torch.var(x.float(), dim=-1, correction=1).numpy()

    def rel(a, b):
        return float((np.abs(a - b) / b).max())

    exact = x.float()
    # (a) the square is a buffer, so it is fp16; the deviations and the
    # accumulation around it are not.
    square_in_fp16 = (
        (exact - exact.mean(-1, keepdim=True)) ** 2
    ).to(F16).float().mean(-1)
    square_in_fp16 = (square_in_fp16 * np.float32(4096 / 4095)).numpy()
    # (c) the answer narrowed once, with no fp16 square anywhere in the path.
    narrowing = (
        ((exact - exact.mean(-1, keepdim=True)) ** 2).mean(-1) * (4096 / 4095)
    ).to(F16).float().numpy()

    assert rel(got, want) > 0.0
    assert rel(square_in_fp16, want) < 0.2 * rel(got, want), (
        rel(square_in_fp16, want),
        rel(got, want),
    )
    assert abs(rel(narrowing, want) - rel(got, want)) < 0.2 * rel(narrowing, want), (
        rel(narrowing, want),
        rel(got, want),
    )


def test_a_deviation_past_255_9_overflows_where_torch_does_not():
    # The limitation, pinned. Seven zeros and a 400: torch variance is 20000,
    # comfortably inside fp16, and the two-pass has to materialise
    # (400 - 50) ** 2 = 122500 as an fp16 intermediate, which is an infinity.
    # No arrangement of these commands avoids it -- the square is a buffer either
    # way -- and the export cannot bound the deviation of a run-time activation,
    # so this is a property of the composition and not of a gate.
    x = torch.zeros(1, 8, dtype=F16)
    x[0, -1] = 400.0
    assert torch.var(x, dim=-1, correction=1).item() == 20000.0
    got = _run(lambda a: torch.var(a, dim=-1, correction=1), (x,), [x.numpy()])
    assert np.isinf(got).all(), got
    # The boundary itself, so 255.9 is a reading and not a round number.
    assert float(torch.finfo(torch.float16).max) == 65504.0
    assert 255.9 < float(np.sqrt(65504.0)) < 256.0


def test_a_statistics_pooling_head_now_reaches_the_dsp():
    # The one real node this op has in a model in this project is a CAM++
    # speaker model statistics pool: mean and standard deviation of the last
    # axis, concatenated. The census records it at [1, 512, 100] to [1, 512]
    # with a mean.dim immediately before it and a sqrt and a cat immediately
    # after. It is not a BatchNorm statistic and there is no fold that eats it:
    # a BatchNorm fold rewrites a constant using running statistics, and this
    # variance is a run-time statistic of the activation whose consumer is a
    # square root, not a normalisation. The row is here at that geometry so a
    # change to the span rules is visible at the shape the model uses.
    x = _x(0, (1, 512, 100), scale=2.0, offset=1.0)

    def pool(t):
        return torch.cat([t.mean(-1), torch.std(t, dim=-1)], -1)

    stream = _stream(pool, (x,))
    assert (REDUCTION, MEAN) in stream and (UNARY, SQUARE) in stream, stream
    got = _run(pool, (x,), [x.numpy()]).reshape(1, 1024)
    want = pool(x).numpy()
    assert np.abs(got.astype(np.float32) - want.astype(np.float32)).max() <= (
        2.0 * STEP_AT_ONE
    ), (got, want)


def test_the_predicates_are_the_functions_the_table_names():
    assert VAR_CORRECTION in hexagon_ops.EMITTERS
    assert hexagon_ops.VAR_TARGETS == frozenset({VAR_CORRECTION})
    assert hexagon_ops.EMITTERS[VAR_CORRECTION] is hexagon_ops._emit_var_correction
    # And the composition needs nothing from the vendored C++ that is not
    # already there: every command it writes is one of these, which is what
    # "zero new C++" has to mean for it to be a claim rather than a figure of
    # speech.
    assert (REDUCTION, BINARY, UNARY) == (29, 19, 4)
    assert (MEAN, SUB, SQUARE, SCALE) == (3, 2, 9, 17)
