# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""log_softmax as the log-sum-exp the softmax kernel is right for.

There is no log-softmax command in the vendored `mnn-htp-ops` tree, and no
log-softmax kernel in upstream MNN's Hexagon backend either: `htp_command.h`
declares nothing for it and `src/dsp/execute_command.cc` has no case. What the
tree has is one `log` in the unary table (`HTP_OPS_UNARY_LOG`, unary_ops.cc:55)
and the exponential and the division the softmax command already spends, so the
op can be written out of commands that exist -- which is the route this emitter
takes, and the reason the obvious short version of it is not the one it takes:

    log(softmax(x))  is two commands and it is wrong at fp16.

The softmax kernel stores probabilities two bytes wide, so a row of logits with
a real spread drives the small ones under fp16's subnormal floor, and the log of
a stored zero is the kernel's -65504 rather than a large negative number. The
shifted form does not lose the answer that way, because the shift -- the row
maximum -- is a number of the log_softmax's own size, and it is the term the
last subtraction puts back. What is left in between is a sum of at most one per
element and its log, both small and both well conditioned.

Six commands, no new kernel: the maximum, the shift, the exponential, the sum,
the log of the sum, and the subtraction that removes the shift.
"""

import os
import pathlib
import struct
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
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    LOG_SOFTMAX_MAX_SPAN,
    log_softmax_shifts_within_the_arena,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonOperatorSupport,
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import Dim, export  # noqa: E402

F16 = torch.float16

#: The three DSP commands a log softmax is, and the two op types that name them.
_REDUCTION = 29
_BINARY = 19
_UNARY = 4
_SOFTMAX = 28
_MAXIMUM = 2
_SUM = 1
_SUB = 2
_EXP = 5
_LOG = 6

#: The trailer's magic, and the offset of its patch records.
_TRAILER_MAGIC = 0x44594E48

#: A row whose spread is wide enough that the smallest exponential in it is 0 in
#: fp16: the logits run to +-24, so exp(-48) is 1.4e-21 and fp16's smallest
#: subnormal is 6e-8. The answer's size is the logits' own, so an answer here is
#: around -25 and the tolerance is a few of fp16's ulps at that size.
_SPREAD = 4.0
_TOLERANCE = 5e-3


class _LogSoftmax(torch.nn.Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log_softmax(x, dim=self.dim)


def _lowered(model, args, dynamic_shapes=None):
    """The single delegate a whole model lowers to, and its commands."""
    program = to_edge_transform_and_lower(
        export(model, tuple(args), dynamic_shapes=dynamic_shapes),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    assert len(calls) == 1, f"the model did not lower to one delegate: {calls}"
    inner = program.graph_module.get_submodule(calls[0].args[0].target)
    assert inner.backend_id == "HexagonBackend"
    blob = bytes(inner._processed_bytes)
    _, commands = read_blob(blob)
    return blob, commands


def test_log_softmax_is_the_shifted_log_sum_exp():
    """The six commands, and the params that say what each one reduces.

    One row of the answer is `x - max(x) - log(sum(exp(x - max(x))))`, and each
    command is one of those steps: the maximum over the row, the subtraction, the
    exponentials of at most one, their sum, the log of it, and the subtraction
    that takes the shift back out. The reduced commands write a one-wide row, and
    the two element-wise commands broadcast it back over the row.
    """
    torch.manual_seed(0)
    x = (torch.randn(2, 3, 8) * _SPREAD).half()
    blob, commands = _lowered(_LogSoftmax(), (x,))
    assert [command.type for command in commands] == [
        _REDUCTION,
        _BINARY,
        _UNARY,
        _REDUCTION,
        _UNARY,
        _BINARY,
    ]
    assert _SOFTMAX not in [command.type for command in commands], (
        "a softmax command reached the blob, which is the composition this "
        "emitter does not use"
    )
    maximum, shifted, exponentials, total, logged, out = commands
    assert list(maximum.params) == [6, 8, 1, _MAXIMUM, 2], list(maximum.params)
    assert list(total.params) == [6, 8, 1, _SUM, 2], list(total.params)
    # 48 = 6 rows of 8; the maximum is read as one value per row.
    assert list(shifted.params[:4]) == [48, 48, 6, _SUB], list(shifted.params[:4])
    assert list(shifted.params[25:28]) == [3, 1, 0], (
        "the row maximum is not broadcast along the reduced axis"
    )
    assert list(exponentials.params) == [48, _EXP, 2], list(exponentials.params)
    assert list(logged.params) == [6, _LOG, 2], list(logged.params)
    assert list(out.params[:4]) == [48, 48, 6, _SUB], list(out.params[:4])
    got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)
    want = torch.log_softmax(x.float(), dim=-1).numpy().reshape(-1)
    np.testing.assert_allclose(
        got.astype(np.float32), want, rtol=2e-3, atol=_TOLERANCE
    )


def test_log_softmax_matches_torch_where_the_short_form_does_not():
    """The row the two commands cannot describe, and the six that can.

    Every one of these rows has an exponential that underflows fp16, so a
    `log(softmax(x))` answers -65504 on it. The command stream is asserted here
    as well, because the numbers alone would not say which composition produced
    them: the emitter next door would answer the same for a narrow row.
    """
    torch.manual_seed(0)
    for rows, span, spread in ((4, 16, 4.0), (2, 64, 2.0), (1, 1000, 3.0)):
        x = (torch.randn(rows, span) * spread).half()
        blob, commands = _lowered(_LogSoftmax(), (x,))
        assert [command.type for command in commands][0] == _REDUCTION
        assert len(commands) == 6
        got = np.frombuffer(execute(blob, [x.numpy()])[0], dtype=np.float16)
        want = torch.log_softmax(x.float(), dim=-1)
        assert np.isfinite(got.astype(np.float32)).all()
        np.testing.assert_allclose(
            got.astype(np.float32),
            want.numpy().reshape(-1),
            rtol=2e-3,
            atol=max(_TOLERANCE, float(want.abs().max()) * 1e-3),
        )


def test_the_short_composition_would_not_have_been_enough():
    """The control for the test above, in fp16 and with no kernel involved.

    This is what `log(softmax(x))` answers on the same row: the softmax stores
    its small probabilities as zeroes and their log is an infinity, where torch's
    answers are finite and near -50. The measurement is why the emitter spends
    six commands rather than two, and it is the same row the simulator's own
    short-form fixture separates.
    """
    torch.manual_seed(0)
    x = (torch.randn(4, 16) * _SPREAD).half()
    stored = torch.softmax(x.float(), dim=-1).half()
    short = torch.log(stored.float()).numpy()
    want = torch.log_softmax(x.float(), dim=-1).numpy()
    assert np.isinf(short).sum() > 0, "the row does not underflow fp16 at all"
    where = np.isinf(short)
    assert np.isfinite(want[where]).all()
    assert np.max(np.abs(want[where] - short[where])) > 20, (
        "the short composition is not far enough out to have needed a different "
        "emitter"
    )


def test_log_softmax_over_the_exported_bound_is_patched_for_every_count():
    """A dynamic row patches the six commands the length appears in.

    The commands' params are products of the operand's shape, so the ones that
    hold the run-time length carry the export's bound plus a patch the runtime
    applies (`_patch_dynamic_product`). The log's own count is not one of them:
    it is the number of *rows*, which the length does not touch, and a command
    that patched it would recompute a count that was already right.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 3, 6).half()
    blob, commands = _lowered(
        _LogSoftmax(), (x,), dynamic_shapes={"x": {2: Dim("tokens", min=1, max=16)}}
    )
    assert list(commands[0].params[:5]) == [3, 16, 1, _MAXIMUM, 2]
    assert list(commands[4].params) == [3, _LOG, 2]
    offset = blob.find(struct.pack("<I", _TRAILER_MAGIC))
    assert offset >= 0, "a dynamic graph emits a trailer"
    header = struct.unpack_from("<7I", blob, offset)
    assert header[4] == 16, f"the trailer's longest length is {header[4]}"
    records = [
        struct.unpack_from("<4i", blob, offset + 28 + record * 16)
        for record in range(header[5])
    ]
    patches = {}
    for index, param, scale, _mode in records:
        patches.setdefault(index, []).append((param, scale))
    assert patches == {
        0: [(1, 1)],  # the maximum's span
        1: [(0, 3), (1, 3), (11, 1)],  # both counts, and the row's extent
        2: [(0, 3)],  # the exponential's count, one per row of the bound
        3: [(1, 1)],  # the sum's span
        5: [(0, 3), (1, 3), (11, 1)],
    }, patches
    assert 4 not in patches, "the log of the sum was patched for a count it owns"


def test_a_log_softmax_off_the_last_axis_is_refused():
    """The kernel's strided path is not the one this composition uses.

    The reduction, the exponentials and the two subtractions are all written for
    a span whose inside extent is one; an axis with something after it would need
    a second set of shapes this emitter does not describe. The node stays on the
    portable kernels and the model still runs.
    """
    support = HexagonOperatorSupport()
    node = _log_softmax_node((3, 4, 5), 1)
    assert not log_softmax_shifts_within_the_arena(node)
    assert not support.is_node_supported({}, node)
    assert log_softmax_shifts_within_the_arena(_log_softmax_node((3, 4, 5), -1))


def test_a_log_softmax_over_more_than_fp16_can_sum_is_refused():
    """The bound is the arena's own arithmetic, not a guess about the op.

    The shift makes every exponential at most one, so the sum the kernel stores
    is at most one per element: past 65504 of them the fp16 store saturates and
    the log of it is an infinity in a row that has no infinity in it. At the
    bound and below it, the emitted form is right; a row longer than that stays
    portable. This is the one length the gate has an opinion about, and it is
    stated so that the opinion can be seen to be the arithmetic's.
    """
    support = HexagonOperatorSupport()
    assert log_softmax_shifts_within_the_arena(_log_softmax_node((1, 1000), -1))
    shortest = _log_softmax_node((1, LOG_SOFTMAX_MAX_SPAN), -1)
    assert log_softmax_shifts_within_the_arena(shortest), (
        "the longest row the sum can hold is already refused"
    )
    assert support.is_node_supported({}, shortest)
    longest = _log_softmax_node((1, LOG_SOFTMAX_MAX_SPAN + 1), -1)
    assert not log_softmax_shifts_within_the_arena(longest)
    assert not support.is_node_supported({}, longest)


def _log_softmax_node(shape, dim):
    """A log_softmax with the operand its gate reads."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape, dtype=F16)
    node = graph.call_function(exir_ops.edge.aten._log_softmax.default, args=(source, dim, False))
    node.meta["val"] = torch.empty(shape, dtype=F16)
    return node
