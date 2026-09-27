# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A cache advance at the cache length: `UPDATE_CACHE_APPEND_FITS`, the position bound.

`update_cache_layout` (`hexagon_partitioner.py:997`) checks the cache's and the
value's SHAPES. It cannot check the POSITION, because the position is a run-time
tensor and a cache's shape says nothing about where a write lands. The emitter
copies the cache out whole and then writes the value's `rows * run` elements at
`position * inner`, so a decode step whose position equals the cache length --
the ordinary one, a decoder that has consumed N tokens asking to write token N --
places the write a whole `inner` past the end of the output it was given.

Four things are pinned here, and each exists because without it one of the others
reads as something it is not:

* the bound. `position * inner + rows * run <= numel`, so it is on rows as well
  as on the position, and the two-row case is what says so.
* that the refusal is THIS clause. `test_the_clause_is_what_refuses_the_node`
  relaxes the clause in the partitioner's own source, re-executes it and shows
  the same node is admitted again, so an assertion named after the bound cannot
  be satisfied by the layout clause next to it. The controls around it are the
  same geometry accepted: a support object that refuses everything cannot make
  any of this green.
* that the position the export does not own is refused, which is the ordinary
  decode and the reason the defect is latent rather than hypothetical. A program
  that owns its position is the positive control, at the same geometry.
* that the geometry really does leave the output, so the clause is aimed at a
  measured defect. That is read off the emitted command stream through the numpy
  model in `blob_interpreter`, which is a model of the region walk and NOT the
  kernels -- no DSP ran here.

Every support object below is built as `HexagonOperatorSupport(_data_placeholders(
program), program)`. A bare `HexagonOperatorSupport()` refuses all of these
nodes, which `test_a_support_object_without_the_program_refuses_the_case_that
_fits` pins, so a "this stays portable" assertion written on one would be true
whether or not the clause existed.

Host tier: export, `to_edge`, the support predicate, and the AOT blob read
back with the runtime's own `read_blob`. No kernel ran, `hexagon-sim` did not
run, and no phone was involved.
"""

import inspect
import textwrap
import weakref
from types import SimpleNamespace

import pytest
import torch

import blob_interpreter
from executorch.backends.hexagon.hexagon_backend import HexagonBackend
from executorch.backends.hexagon.hexagon_ops import update_cache_layout
from executorch.backends.hexagon.kv_cache import UPDATE_CACHE
from executorch.backends.hexagon.partition import hexagon_partitioner as hpart
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    _update_cache_append_fits,
    HexagonOperatorSupport,
)
from executorch.exir import EdgeCompileConfig, to_edge
from torch.export import export

F16 = torch.float16
CFG = EdgeCompileConfig(_check_ir_validity=False)

#: The two gates this file separates, in this checkout's line numbering. Both are
#: in `_verdict` and they are adjacent, so every refusal below is attributed to
#: one of them rather than to "the update_cache gates".
LAYOUT_GATE = 1032
APPEND_GATE = 1039

BATCH, SEQ, HEADS, DIM = 1, 5, 2, 32

_REFUSED = weakref.WeakKeyDictionary()


def _hit(line, node):
    _REFUSED[node] = line


class _Instrumented:
    """_verdict with every refusal tagged by the line it happens on.

    The rewrite PARTITION_GATES.md describes: every `return False` in the
    source `inspect.getsource` returns for `_verdict` records the line it is
    on and the node it is deciding, the rewritten copy runs in the module's own
    namespace, and the last line recorded is the clause that decided it.
    """

    def __init__(self):
        self.original = hpart.HexagonOperatorSupport._verdict
        self.namespace = None

    def install(self):
        start = self.original.__code__.co_firstlineno
        source = textwrap.dedent(inspect.getsource(self.original))
        rewritten = []
        for offset, line in enumerate(source.splitlines()):
            if line.strip() == "return False":
                line = line.replace(
                    "return False", f"_hit({start + offset}, node); return False"
                )
            rewritten.append(line)
        namespace = dict(vars(hpart))
        namespace["_hit"] = _hit
        exec(  # noqa: S102 - the source is this module's own predicate
            compile("\n".join(rewritten), "<append-bound-instrumented>", "exec"),
            namespace,
        )
        self.namespace = namespace
        hpart.HexagonOperatorSupport._verdict = namespace["_verdict"]

    def remove(self):
        hpart.HexagonOperatorSupport._verdict = self.original


@pytest.fixture
def instrumented():
    tool = _Instrumented()
    tool.install()
    try:
        yield tool
    finally:
        tool.remove()


class _Advance(torch.nn.Module):
    """One cache advance whose position the caller supplies at run time.

    This is the ordinary decode: the graph does not know where in the cache the
    new token goes, and nothing in it can say.
    """

    def forward(self, cache, value, position):
        return torch.ops.et_hexagon.update_cache(cache, value, position)


class _OwnedAdvance(torch.nn.Module):
    """The same advance with a position the program owns, as a buffer.

    A decoder whose start position is fixed -- a prefill, or a graph built once
    and run at one offset -- is the case the host can read a position for, and
    it is the positive control for every refusal in this file.
    """

    def __init__(self, position, dtype=torch.int64):
        super().__init__()
        self.register_buffer("position", torch.tensor([position], dtype=dtype))

    def forward(self, cache, value):
        return torch.ops.et_hexagon.update_cache(cache, value, self.position)


def _edge(module, args):
    program = hpart.HexagonPartitioner().transform_for_pre_decomposition(
        export(module, args)
    )
    return to_edge(program, compile_config=CFG).exported_program()


def _x(*shape, dtype=F16):
    return torch.randn(*shape, dtype=dtype)


def _advance_node(ep):
    nodes = [
        n
        for n in ep.graph_module.graph.nodes
        if n.op == "call_function" and n.target is UPDATE_CACHE
    ]
    assert len(nodes) == 1, (
        "the advance did not survive to_edge with the identity the two gates "
        f"test; targets were "
        f"{[str(n.target) for n in ep.graph_module.graph.nodes if n.op == 'call_function']}"
    )
    return nodes[0]


def _verdict(ep, node):
    """(accepted, the line that refused it) for one node of one program."""
    _REFUSED.clear()
    support = HexagonOperatorSupport(_data_placeholders(ep), ep)
    return support.is_node_supported({}, node), _REFUSED.get(node)


def _owned(position, rows=1, dtype=torch.int64, seq=SEQ, heads=HEADS, dim=DIM):
    value = _x(BATCH, rows, heads, dim)
    ep = _edge(_OwnedAdvance(position, dtype), (_x(BATCH, seq, heads, dim), value))
    return ep, _advance_node(ep), value


def _runtime(position, rows=1, seq=SEQ):
    value = _x(BATCH, rows, HEADS, DIM)
    ep = _edge(
        _Advance(), (_x(BATCH, seq, HEADS, DIM), value, torch.tensor([position]))
    )
    return ep, _advance_node(ep), value


# ---------------------------------------------------------------------------
# The bound, and the refusal being this clause rather than its neighbour.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position,rows", [(0, 1), (3, 1), (4, 1), (3, 2)])
def test_an_owned_position_that_fits_is_accepted(instrumented, position, rows):
    """The positive control, and what makes every refusal below a bound.

    Same geometry, same clause, same program: a position the program owns and
    that satisfies `position * inner + rows * run <= numel` delegates. A clause
    that refused everything fails here.
    """
    ep, node, _ = _owned(position, rows)
    accepted, line = _verdict(ep, node)
    assert accepted, f"position {position} rows {rows} fits and was refused at {line}"
    assert line is None
    assert _update_cache_append_fits(node, ep)


@pytest.mark.parametrize("position,rows", [(5, 1), (6, 1), (4, 2), (5, 2)])
def test_an_owned_position_past_the_cache_is_refused_at_the_append_gate(
    instrumented, position, rows
):
    """The bound, pinned at the line that carries it.

    `(5, 1)` is the ordinary decode on a five-row cache: the geometry is fp16,
    contiguous, two heads of width 32, `update_cache_layout` admits it, and the
    write still lands a whole `inner` past the output. `(4, 2)` is the same
    cache with two tokens appended, so the bound is on rows as well as on the
    position.

    The two gates are adjacent and one of them is a shape check, so the refusal
    is attributed to a line: 997 is the layout and 1004 is the bound, and this
    node is refused by the second.
    """
    ep, node, _ = _owned(position, rows)
    accepted, line = _verdict(ep, node)
    assert not accepted
    assert line == APPEND_GATE, f"refused at {line}, not at the append gate"
    # The layout clause is not what turned it away, or 997 would be the line.
    assert update_cache_layout(node) is not None
    assert not _update_cache_append_fits(node, ep)


def test_a_position_the_program_does_not_own_is_refused_at_the_append_gate(
    instrumented,
):
    """The ordinary decode: a run-time position is unbounded, and 0 is too.

    `position = 0` is inside the cache by any reading and is refused anyway,
    because the clause is about whether the write is BOUNDED, and a tensor the
    export does not own carries no value to bound it with. The same geometry
    with a readable position is accepted (the first test), so this is the bound
    and not a blanket refusal.
    """
    for position in (0, 4, 5):
        ep, node, _ = _runtime(position)
        accepted, line = _verdict(ep, node)
        assert not accepted, f"position {position} was accepted with no value for it"
        assert line == APPEND_GATE, f"refused at {line}, not at the append gate"
        assert update_cache_layout(node) is not None


@pytest.mark.parametrize(
    "seq,heads,dim,position,rows,accepted",
    [
        # The bound is the cache LENGTH, not a number this file happens to use:
        # a longer cache admits positions the five-row one refuses.
        (8, 2, 32, 5, 1, True),
        (8, 2, 32, 7, 1, True),
        (8, 2, 32, 8, 1, False),
        # And it is a geometry, not the one geometry above: a single-head cache
        # of a different width at a different length moves the boundary with it.
        (4, 1, 32, 3, 1, True),
        (4, 1, 32, 4, 1, False),
        (6, 4, 48, 5, 1, True),
        (6, 4, 48, 6, 1, False),
        (5, 2, 32, 3, 2, True),
        (5, 2, 32, 4, 2, False),
    ],
    ids=[
        "seq8_pos5",
        "seq8_pos7",
        "seq8_pos8",
        "one_head_pos3",
        "one_head_pos4",
        "four_heads_pos5",
        "four_heads_pos6",
        "two_rows_pos3",
        "two_rows_pos4",
    ],
)
def test_the_bound_is_the_cache_length_and_not_a_number(
    instrumented, seq, heads, dim, position, rows, accepted
):
    """Four geometries, and the boundary moves with the cache rather than with a constant.

    Every row is the same clause and the same program, and only the geometry and
    the position differ, so a bound implemented as a number this file happens to
    use, or as a fixed fraction of the cache, fails here. The two positive rows
    are the control: without them a clause that refused everything would pass the
    negative half.
    """
    ep, node, _ = _owned(position, rows, seq=seq, heads=heads, dim=dim)
    verdict, line = _verdict(ep, node)
    if accepted:
        assert verdict, f"seq {seq} pos {position} rows {rows} fits and was refused at {line}"
        return
    assert not verdict
    assert line == APPEND_GATE, f"refused at {line}, not at the append gate"
    assert update_cache_layout(node) is not None


def test_a_support_object_without_the_program_refuses_the_case_that_fits(
    instrumented,
):
    """The vacuity guard, and the reason the other tests build theirs with a program.

    Without the program the support object cannot read a value, so it refuses
    the geometry that FITS as well as the one that does not. A "this stays
    portable" assertion written on such an object is therefore true whether or
    not the clause exists, and none of this file is written that way.
    """
    ep, node, _ = _owned(4)
    assert HexagonOperatorSupport().is_node_supported({}, node) is False
    assert not _update_cache_append_fits(node, None)
    accepted, _ = _verdict(ep, node)
    assert accepted, "the same node is refused with the program it belongs to"


def test_an_int32_position_the_program_owns_is_bounded_the_same_way(instrumented):
    """The position's width is not the clause's business; its value is.

    `match_kv_cache` already admits an int32 or an int64 position, so the clause
    that reads one has to read both, and a tree that narrowed it to int64 would
    refuse the int32 prefill that used to delegate.
    """
    ep, node, _ = _owned(4, dtype=torch.int32)
    accepted, line = _verdict(ep, node)
    assert accepted, f"an int32 position that fits was refused at {line}"
    ep, node, _ = _owned(5, dtype=torch.int32)
    accepted, line = _verdict(ep, node)
    assert not accepted
    assert line == APPEND_GATE


def test_the_clause_is_what_refuses_the_node(instrumented):
    """Relax the clause in the predicate's own source and the node comes back.

    An assertion named after a clause is only load-bearing if that clause is what
    refuses the node, so the clause's every `return False` becomes `return True`
    here -- from the source `inspect.getsource` returns, executed in the
    module's own namespace, the way the instrument above does it -- and the same
    node is evaluated under both. The node has to flip to accepted; if it did
    not, the refusals above would be some other clause answering.

    The rewritten copy is installed and restored inside this test, so a failure
    here cannot leave the predicate patched for the tests around it.
    """
    ep, node, _ = _runtime(5)
    accepted, line = _verdict(ep, node)
    assert not accepted and line == APPEND_GATE

    source = textwrap.dedent(inspect.getsource(_update_cache_append_fits))
    relaxed = source.replace("return False", "return True")
    assert relaxed != source, "the clause has no refusal left to relax"
    namespace = dict(vars(hpart))
    exec(compile(relaxed, "<append-bound-relaxed>", "exec"), namespace)  # noqa: S102
    original = hpart._update_cache_append_fits
    relaxed_clause = namespace["_update_cache_append_fits"]
    hpart._update_cache_append_fits = relaxed_clause
    # The instrumented _verdict is exec'd in a dict captured at install time, so
    # it reads its own copy of the clause and patching the module global alone
    # would relax nothing it can see.
    instrumented.namespace["_update_cache_append_fits"] = relaxed_clause
    try:
        _REFUSED.clear()
        support = HexagonOperatorSupport(_data_placeholders(ep), ep)
        relaxed_accepted = support.is_node_supported({}, node)
        relaxed_line = _REFUSED.get(node)
    finally:
        hpart._update_cache_append_fits = original
        instrumented.namespace["_update_cache_append_fits"] = original
    assert relaxed_accepted, (
        "the clause is not what refuses this node: with every one of its "
        f"refusals relaxed it is still refused at {relaxed_line}"
    )


def test_the_geometry_leaves_the_output_and_the_clause_is_what_stops_it(
    instrumented,
):
    """The defect, measured, beside the refusal aimed at it.

    A hand-built graph holding just the advance is lowered through the backend's
    own codegen rather than through the partitioner -- so the clause is stepped
    around deliberately, which is what makes this a measurement of the emitter --
    and the emitted command stream is run with the position the graph supplies.
    At `seq - rows` the write lands in the output and leaves the rows before it
    alone; at `seq` it walks past the end.

    `blob_interpreter.execute` is a numpy MODEL of the region walk and not the
    kernels, so what this establishes is that the region the emitter describes
    leaves the output. The in-bounds half is in the same test on purpose: without
    it the out-of-bounds half reads as a broken emitter rather than as a bound.
    """
    np = pytest.importorskip("numpy")

    def program_of():
        graph = torch.fx.Graph()
        cache = graph.placeholder("cache")
        cache.meta["val"] = torch.empty((BATCH, SEQ, HEADS, DIM), dtype=F16)
        value = graph.placeholder("value")
        value.meta["val"] = torch.empty((BATCH, 1, HEADS, DIM), dtype=F16)
        pos = graph.placeholder("position")
        pos.meta["val"] = torch.empty(1, dtype=torch.int64)
        fused = graph.call_function(UPDATE_CACHE, args=(cache, value, pos))
        fused.meta["val"] = torch.empty((BATCH, SEQ, HEADS, DIM), dtype=F16)
        graph.output(fused)
        module = torch.fx.GraphModule(torch.nn.Module(), graph)
        # The backend reads the signature to tell an owned constant from a
        # caller-passed input, and this graph owns neither.
        signature = SimpleNamespace(
            inputs_to_buffers={},
            buffers_to_mutate={},
            inputs_to_parameters={},
            inputs_to_lifted_tensor_constants={},
        )
        return SimpleNamespace(
            graph_module=module, graph_signature=signature, range_constraints={}
        )

    cache = torch.arange(BATCH * SEQ * HEADS * DIM, dtype=F16).reshape(
        BATCH, SEQ, HEADS, DIM
    )
    value = torch.randn(BATCH, 1, HEADS, DIM, dtype=F16)
    blob = HexagonBackend.preprocess(program_of(), []).processed_bytes

    out = blob_interpreter.execute(
        blob, [cache.numpy(), value.numpy(), np.array([4], dtype=np.int64)]
    )[0]
    got = torch.from_numpy(np.asarray(out).copy()).view(F16).reshape(cache.shape)
    assert torch.equal(got[:, 4:5], value), "the in-bounds row was not written"
    assert torch.equal(got[:, :4], cache[:, :4]), "the rows before it moved"

    with pytest.raises(blob_interpreter.RegionOutOfBounds):
        blob_interpreter.execute(
            blob, [cache.numpy(), value.numpy(), np.array([5], dtype=np.int64)]
        )
