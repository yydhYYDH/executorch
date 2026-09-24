# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A split's pieces, as the blits that write them.

`torch.split` and `torch.chunk` both lower to `aten.split_with_sizes_copy`, whose
results the graph reads one getitem at a time -- the multi-output shape that made
the getitem rule a rule in the first place. Unlike a sort or a topk's positions,
every piece has a producer here: a piece of a tensor is the narrowing slice
`x[:, :n]` stands for, and the blit engine already writes those. The emitter is
therefore one `DSP_OP_RASTER_BLIT` per piece *the graph reads*, and the claim
these tests rest on is mechanical: the region a piece emits is the region the
equivalent slice emits, byte for byte, so a split is exactly the set of slices it
stands for.

The gates are the numbers a command has no argument for (offset, extent, row
stride), the two-byte element the arena holds, and the readers: a split whose
tuple is handed on rather than indexed has no command form, so it stays portable.
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

import blob_interpreter  # noqa: E402
from blob_interpreter import execute, read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    split_is_emittable,
    split_spec,
    SPLIT_TARGETS,
    SplitSpec,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonOperatorSupport,
    HexagonPartitioner,
    refused_overload_census,
    reset_refused_overload_census,
    reset_unwired_overload_census,
    unwired_overload_census,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge,
    to_edge_transform_and_lower,
)
from torch.export import Dim, export  # noqa: E402

F16 = torch.float16

#: DSP_OP_RASTER_BLIT, the one command a split emits.
_BLIT = 3

#: The params of a blit, in the order the emitter writes them.
_REGION = (
    "region count",
    "element bytes",
    "source count",
    "source index",
    "source offset",
    "destination offset",
    "destination count",
    "rows",
    "run",
    "source row stride",
    "axis stride",
    "inner count",
    "inner source stride",
    "inner run",
    "inner destination stride",
)


def _region(command):
    """A blit's region as named fields, so a failing row reads as a sentence."""
    return dict(zip(_REGION, command.params))


class _Split(torch.nn.Module):
    def __init__(self, forward) -> None:
        super().__init__()
        self.forward_fn = forward

    def forward(self, x):
        return self.forward_fn(x)


def _delegates(program):
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]


def _config():
    return EdgeCompileConfig(_check_ir_validity=False)


def _program(model, x, **export_kwargs):
    return to_edge_transform_and_lower(
        export(model, (x,), **export_kwargs),
        partitioner=[HexagonPartitioner()],
        compile_config=_config(),
    ).exported_program()


def _lowered(model, x, **export_kwargs):
    """The one delegate's blob and its decoded command stream."""
    program = _program(model, x, **export_kwargs)
    calls = _delegates(program)
    assert len(calls) == 1, f"the graph did not reach one delegate: {len(calls)}"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    assert lowered.backend_id == "HexagonBackend"
    raw = bytes(lowered._processed_bytes)
    _header, commands = read_blob(raw)
    return raw, commands


def _blits(commands):
    return [command for command in commands if command.type == _BLIT]


def _values(raw, x):
    return np.frombuffer(execute(raw, [x.numpy()])[0], dtype=np.float16)


def _verdict(verdicts, suffix):
    """What the partitioner said about one target, by its qualified name."""
    found = [accepted for name, accepted in verdicts if name.endswith(suffix)]
    assert found, f"no {suffix} in the edge graph: {[name for name, _ in verdicts]}"
    return found


def _supported(model, x):
    """What the partitioner says about each edge op, in graph order."""
    program = to_edge(export(model, (x,)), compile_config=_config()).exported_program()
    support = HexagonOperatorSupport()
    return [
        (
            getattr(node.target, "__name__", str(node.target)),
            support.is_node_supported({}, node),
        )
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
    ]


def test_a_split_is_one_blit_per_piece_the_graph_reads():
    """Two pieces read, two blits; a dead piece costs nothing.

    The middle piece of the three below is never read, so it has no getitem and
    nothing to write into; emitting a blit for it would be a command whose
    result no graph value holds. The third piece is still addressed as if the
    middle one had been written, which is what its offset below pins.
    """
    x = torch.randn(2, 6, 8, dtype=F16)
    cases = [
        (
            "two of two",
            lambda a: torch.split(a, 2, dim=1)[0] + torch.split(a, 2, dim=1)[1],
            2,
        ),
        (
            "first and third",
            lambda a: torch.split(a, 2, dim=1)[0] + torch.split(a, 2, dim=1)[2],
            2,
        ),
        ("first only", lambda a: torch.split(a, 2, dim=1)[0], 1),
        ("last only", lambda a: torch.split(a, 2, dim=1)[-1], 1),
        ("chunk", lambda a: torch.chunk(a, 3, dim=1)[1], 1),
    ]
    for label, forward, expected in cases:
        raw, commands = _lowered(_Split(forward), x)
        blits = _blits(commands)
        assert len(blits) == expected, f"{label}: {len(blits)} blits, wanted {expected}"
        assert len(commands) == len(blits) + (
            1 if "first and third" in label or "two of two" in label else 0
        ), f"{label}: a command that is not a blit or the add: {[c.type for c in commands]}"


def test_each_piece_is_the_slice_that_piece_stands_for():
    """The region a piece emits is the region the equivalent slice emits.

    This is the whole justification for splitting through the blit engine rather
    than a kernel of its own: `torch.split(x, 4, dim=1)` and `x[:, :4] + x[:, 4:]`
    are the same read, so if the two blobs hold the same regions then the split
    is not a new write path, it is the slices torch already had. Compared field by
    field rather than as bytes because the two graphs put their buffers at
    different arena offsets, and only the region is the claim.
    """
    x = torch.randn(2, 8, dtype=F16)
    _split_raw, split_commands = _lowered(
        _Split(lambda a: torch.split(a, 4, dim=1)[0] + torch.split(a, 4, dim=1)[1]), x
    )
    _slice_raw, slice_commands = _lowered(_Split(lambda a: a[:, :4] + a[:, 4:]), x)
    split_regions = [_region(command) for command in _blits(split_commands)]
    slice_regions = [_region(command) for command in _blits(slice_commands)]
    assert len(split_regions) == len(slice_regions) == 2
    assert (
        split_regions == slice_regions
    ), "the split's pieces and the slices they stand for are not the same reads"
    assert [region["source offset"] for region in split_regions] == [0, 4]
    assert all(
        region["rows"] == 2 and region["axis stride"] == 8 for region in split_regions
    )


def test_a_piece_starts_where_the_extents_before_it_end():
    """The offsets, including the piece nothing reads.

    `split_copy.Tensor` cuts a short last piece, so a width of eight in threes is
    three, three and two: the second piece starts 24 elements in rather than 16,
    and a last piece that took a full three would read past the operand. The axis
    stride is the whole axis times the inner extent -- the row-to-row distance of
    the operand, not of the piece.
    """
    cases = [
        # label, shape, split, source offsets, runs, axis stride, rows
        (
            "two of a row",
            (2, 8),
            lambda a: torch.split(a, 4, dim=1),
            [0, 4],
            [4, 4],
            8,
            2,
        ),
        (
            "three of six",
            (2, 6, 8),
            lambda a: torch.split(a, 2, dim=1),
            [0, 16, 32],
            [16] * 3,
            48,
            2,
        ),
        (
            "ragged three",
            (2, 8),
            lambda a: torch.ops.aten.split_copy.Tensor(a, 3, 1),
            [0, 3, 6],
            [3, 3, 2],
            8,
            2,
        ),
        (
            # The outer axis: a single region, one row, and the run is the piece.
            "rows",
            (6, 4),
            lambda a: torch.split(a, 2, dim=0),
            [0, 8, 16],
            [8] * 3,
            24,
            1,
        ),
    ]
    for label, shape, split, offsets, runs, axis, rows in cases:
        x = torch.randn(*shape, dtype=F16)
        _raw, commands = _lowered(
            _Split(lambda a: sum(part.sum() for part in split(a))), x
        )
        blits = _blits(commands)
        assert [command.params[4] for command in blits] == offsets, label
        assert [command.params[8] for command in blits] == runs, label
        assert [command.params[7] for command in blits] == [rows] * len(offsets), label
        assert {_region(command)["axis stride"] for command in blits} == {axis}, label


def test_the_pieces_put_the_tensor_back_together():
    """Values, through the same interpreter the runtime runs.

    Concatenating the pieces has to give the operand back bit for bit: a region
    off by a row or a run would still produce a plausibly shaped tensor, and
    fp16 addition is exact here, so this is an equality rather than a tolerance.
    The extra blit beyond one per piece is the concatenate, which is a blit too.
    """
    for shape, dim, size in (
        ((2, 8), 1, 4),
        ((2, 6, 8), 1, 2),
        ((6, 4), 0, 3),
        ((2, 4, 3), -1, 1),
        ((2, 4, 3), 2, 1),
        ((4, 6), 1, 2),
    ):
        x = torch.randn(*shape, dtype=F16)
        pieces = torch.split(x, size, dim=dim)
        raw, commands = _lowered(
            _Split(lambda a: torch.cat(torch.split(a, size, dim=dim), dim=dim)), x
        )
        assert len(_blits(commands)) == len(pieces) + 1, (shape, dim, size)
        assert _values(raw, x).tobytes() == x.reshape(-1).numpy().tobytes(), (
            shape,
            dim,
            size,
        )


def test_a_piece_read_on_its_own_is_that_piece_of_torch():
    """One piece at a time, against torch's own split.

    The round trip above cannot separate "the pieces are right" from "two wrong
    pieces that cancel", because a concatenation reading the same wrong runs
    twice would still line up with x. This reads one piece per graph and compares
    its bytes with `torch.split`'s.
    """
    x = torch.randn(2, 6, 8, dtype=F16)
    pieces = torch.split(x, 2, dim=1)
    for index in range(3):
        raw, commands = _lowered(
            _Split(lambda a, i=index: torch.split(a, 2, dim=1)[i]), x
        )
        assert len(_blits(commands)) == 1
        assert (
            _values(raw, x).tobytes() == pieces[index].reshape(-1).numpy().tobytes()
        ), index


def test_a_split_leaves_no_refusal_behind():
    """The census, on the graph the refusals came from.

    A split used to be an unwired target with a refused getitem per piece; the
    point of the change is that both counters are empty afterwards, which is what
    a model report reads.
    """
    x = torch.randn(2, 6, 8, dtype=F16)
    reset_refused_overload_census()
    reset_unwired_overload_census()
    raw, commands = _lowered(_Split(lambda a: torch.chunk(a, 3, dim=1)[0]), x)
    assert len(_blits(commands)) == 1
    assert dict(refused_overload_census()) == {}, "a split target is still refused"
    assert dict(unwired_overload_census()) == {}, "a split target is still unwired"


def test_a_recurrent_models_gate_split_reaches_the_dsp():
    """The row the census pins, with the commands behind it.

    An unrolled LSTM splits its fused weight matrix into four gates per step.
    Those splits used to be the model's cut points -- five delegates -- and are
    now blits inside two, so the count and the blits are the same claim. The
    twelve gate pieces are counted off the edge graph rather than asserted as a
    number: the blob has more blits than that, because the step boundary slices
    as well, and only the graph says which of them the splits are.
    """

    class _Lstm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(4, 5, batch_first=True).half()

        def forward(self, x):
            return self.rnn(x)[0]

    model = _Lstm().eval()
    x = torch.randn(2, 3, 4, dtype=F16)
    edges = to_edge(export(model, (x,)), compile_config=_config()).exported_program()
    pieces = 0
    for node in edges.graph_module.graph.nodes:
        if node.op == "call_function" and node.target in SPLIT_TARGETS:
            spec = split_spec(node)
            assert (
                spec is not None
            ), "a split in an LSTM is still held out of the backend"
            pieces += sum(len(readers) for _start, _extent, readers in spec.pieces)
    assert pieces == 12, f"the unrolled LSTM's gate splits are {pieces} pieces, not 12"

    reset_refused_overload_census()
    reset_unwired_overload_census()
    program = _program(model, x)
    calls = _delegates(program)
    assert len(calls) == 2, f"the LSTM did not fold into two delegates: {len(calls)}"
    blits = 0
    for call in calls:
        lowered = program.graph_module.get_submodule(call.args[0].target)
        _header, commands = read_blob(bytes(lowered._processed_bytes))
        blits += len(_blits(commands))
    assert blits >= pieces, f"the gate splits emit {blits} blits, fewer than {pieces}"
    # A recurrent model still hands the step boundary to the host -- the `cat`,
    # the `full` the initial state became and the `expand` beside them -- so what
    # is asserted here is that no *split* is among the refusals.
    refused = dict(refused_overload_census())
    assert [
        name for name in refused if "split" in name or name == "getitem"
    ] == [], refused
    assert dict(unwired_overload_census()) == {}


def test_the_shapes_that_stay_portable():
    """The gate, one row at a time.

    Every row is a number the command has no argument for or a dtype the arena
    cannot hold, so the whole node stays where a portable kernel can read it.
    """
    rows = [
        (
            "int32 source",
            torch.arange(6, dtype=torch.int32).reshape(6, 1),
            lambda a: torch.split(a, 2, dim=0)[0],
            False,
        ),
        (
            "float64 source",
            torch.randn(6, 1, dtype=torch.float64),
            lambda a: torch.split(a, 2, dim=0)[0],
            False,
        ),
        (
            "a step-2 slice in front",
            torch.randn(2, 8, dtype=F16),
            lambda a: torch.split(a[:, ::2], 2, dim=1)[0],
            True,
        ),
        (
            "static fp16",
            torch.randn(2, 8, dtype=F16),
            lambda a: torch.split(a, 4, dim=1)[0],
            True,
        ),
    ]
    for label, x, forward, expected in rows:
        verdicts = _supported(_Split(forward), x)
        split = [
            accepted
            for name, accepted in verdicts
            if name in {t.__name__ for t in SPLIT_TARGETS}
        ]
        assert split, f"{label}: no split in the edge graph at all"
        assert split == [expected] * len(split), f"{label}: {verdicts}"

    # A dynamic extent is the one gate a frontend graph reaches and a command
    # cannot express: the region is built when the command is written, so the
    # shape has to be a number by then.
    x = torch.randn(2, 8, dtype=F16)
    dynamic = to_edge_transform_and_lower(
        export(
            _Split(lambda a: torch.split(a, 4, dim=1)[0]),
            (x,),
            dynamic_shapes={"x": {0: Dim("batch", min=2, max=8)}},
        ),
        partitioner=[HexagonPartitioner()],
        compile_config=_config(),
    ).exported_program()
    assert (
        len(_delegates(dynamic)) == 0
    ), "a symbolic batch size reached the DSP as a region's row count"


def test_every_reader_has_to_be_a_getitem_of_a_named_piece():
    """The rule no graph from torch can violate, checked on a node it cannot build.

    `torch.stack`, `torch.cat` and unpacking all lower to one getitem per element,
    and a split whose length alone is read leaves no node behind, so the exporter
    never hands this backend a tuple consumer. The rule is what keeps a reader
    holding a tuple out if one ever appears -- so it is falsified directly, by
    giving a real split node a user that is not a getitem.
    """
    x = torch.randn(2, 6, 8, dtype=F16)
    program = to_edge(
        export(_Split(lambda a: torch.split(a, 2, dim=1)[0]), (x,)),
        compile_config=_config(),
    ).exported_program()
    graph = program.graph_module.graph
    node = next(node for node in graph.nodes if node.target in SPLIT_TARGETS)
    assert split_is_emittable(node)
    # A user of the tuple itself: the shape a `len(parts)` or a `torch.stack` that
    # the exporter had left alone would have.
    consumer = graph.call_function(torch.ops.aten._to_copy.default, args=(node,))
    consumer.meta["val"] = torch.empty((2, 2, 8), dtype=F16)
    assert not split_is_emittable(node), "a tuple consumer was accepted as a reader"


def test_the_gate_is_what_holds_the_split_row_out():
    """The tightening control: move the gate and see which rows move.

    A test that only asserts today's verdicts says nothing about whether the gate
    is what produces them -- the verdict could be an accident of the emitter
    table. Closing the gate has to take the whole row with it: no delegate, a
    refused target, and the getitems behind it portable again.
    """
    x = torch.randn(2, 6, 8, dtype=F16)
    model = _Split(lambda a: torch.split(a, 2, dim=1)[0])
    assert len(_delegates(_program(model, x))) == 1

    saved = hexagon_ops.split_spec
    hexagon_ops.split_spec = lambda node: None
    try:
        reset_refused_overload_census()
        program = _program(model, x)
        assert (
            len(_delegates(program)) == 0
        ), "the gate was closed and the split ran anyway"
        assert (
            dict(refused_overload_census()) != {}
        ), "the row moved out of the delegate without appearing in the refusal census"
        verdicts = _supported(model, x)
        assert _verdict(verdicts, "split_with_sizes_copy.default") == [
            False
        ], "the split's own check is closed, so the partitioner has to refuse the node"
        # The getitem beside it is still reported as placeable, exactly as a max
        # pool's values are: the producer's own check is the one that decides, and
        # it has. What the partitioner does with the pair is the delegate count.
        assert _verdict(verdicts, "getitem") == [True]
    finally:
        hexagon_ops.split_spec = saved
    assert len(_delegates(_program(model, x))) == 1, "the gate did not come back"


def test_the_extent_rule_is_what_keeps_a_symbolic_split_out():
    """The loosening control: open the extent rule and watch the row move.

    The gate here is not decoration and the rows above are not passing by
    accident: drop the rule that a piece's extents have to be numbers and the
    split of a dynamic batch size delegates, with a region whose row count is the
    batch the program was exported at. A run-time batch of eight would write two
    of its eight rows and leave the rest of the output as the arena had it, which
    is the wrong answer this rule exists to prevent -- so the loosened blob's
    params below are the falsification, not just a changed verdict.
    """
    x = torch.randn(2, 8, dtype=F16)
    model = _Split(lambda a: torch.split(a, 4, dim=1)[0])
    shapes = {"x": {0: Dim("batch", min=2, max=8)}}

    def program(**kwargs):
        return to_edge_transform_and_lower(
            export(model, (x,), dynamic_shapes=shapes),
            partitioner=[HexagonPartitioner()],
            compile_config=_config(),
            **kwargs,
        ).exported_program()

    assert len(_delegates(program())) == 0, "a symbolic batch reached the DSP as gated"

    saved = hexagon_ops.split_spec

    def trusting(node):
        """The gate with the extents read as the numbers they are at export."""
        dim = hexagon_ops._node_arg(node, "dim", 2, 0)
        source = node.args[0]
        value = source.meta.get("val")
        shape = [int(extent) for extent in value.shape]
        sizes = hexagon_ops._split_sizes(node, shape[dim])
        if sizes is None:
            return None
        inner = 1
        for extent in shape[dim + 1 :]:
            inner *= extent
        rows = 1
        for extent in shape[:dim]:
            rows *= extent
        declared = node.meta.get("val")
        if not isinstance(declared, (list, tuple)) or len(declared) != len(sizes):
            return None
        pieces = []
        start = 0
        for index, size in enumerate(sizes):
            pieces.append((start, size, hexagon_ops._split_piece_readers(node, index)))
            start += size
        return SplitSpec(source, dim, rows, inner, shape[dim], tuple(pieces))

    hexagon_ops.split_spec = trusting
    try:
        loosened = program()
        calls = _delegates(loosened)
        assert len(calls) == 1, "the loosened gate did not move the row"
        lowered = loosened.graph_module.get_submodule(calls[0].args[0].target)
        _header, commands = read_blob(bytes(lowered._processed_bytes))
        blits = _blits(commands)
        assert [command.params[7] for command in blits] == [
            2
        ], "the loosened region is not carrying the export-time row count"
        assert [command.params[1] for command in blits] == [2]
    finally:
        hexagon_ops.split_spec = saved
    assert len(_delegates(program())) == 0, "the gate did not come back"


def test_the_other_gates_are_layered_behind_the_partitioners_own():
    """What the inner checks would and would not move, measured.

    Opening the dtype or the layout check on its own moves no row, and the reason
    is worth stating rather than assuming: an int32 or fp64 source is refused by
    the partitioner's operand rule before the split's own check is even asked, and
    a source that is not contiguous cannot reach the gate from a frontend graph,
    because the strided slice in front of it is a refused op whose result the host
    hands over contiguous. So each of these is the second layer of a pair, and the
    first layer is the one a case rests on.
    """
    saved = hexagon_ops.split_spec

    def open_the_dtype_check(node):
        """The real gate, run on a fp16 view of whatever the source really is."""
        source = node.args[0]
        value = source.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return saved(node)
        source.meta["val"] = value.to(torch.float16)
        try:
            return saved(node)
        finally:
            source.meta["val"] = value

    rows = [
        (
            "int32 source",
            torch.arange(6, dtype=torch.int32).reshape(6, 1),
            lambda a: torch.split(a, 2, dim=0)[0],
        ),
        (
            "float64 source",
            torch.randn(6, 1, dtype=torch.float64),
            lambda a: torch.split(a, 2, dim=0)[0],
        ),
    ]
    for label, x, forward in rows:
        model = _Split(forward)
        assert len(_delegates(_program(model, x))) == 0, label
        hexagon_ops.split_spec = open_the_dtype_check
        try:
            assert len(_delegates(_program(model, x))) == 0, (
                f"{label}: opening the dtype check moved the row, so the ledger "
                "names the wrong layer"
            )
        finally:
            hexagon_ops.split_spec = saved

    # A strided source is not refused because of the split: the slice that makes
    # it strided is the refused node, and its result arrives contiguous.
    x = torch.randn(2, 8, dtype=F16)
    strided = _Split(lambda a: torch.split(a[:, ::2], 2, dim=1)[0])
    assert len(_delegates(_program(strided, x))) == 1
    verdicts = _supported(strided, x)
    assert _verdict(verdicts, "slice_copy.Tensor") == [False]
    assert _verdict(verdicts, "split_with_sizes_copy.default") == [True]


def test_a_piece_addressed_early_reads_the_wrong_run():
    """The teeth under the offsets: shift one and the answer moves.

    The blob below is the case's own bytes with the second piece's source offset
    three elements early, inside the piece the graph does not read. The interpreter
    runs the same region the DSP does, so a comparison that does not move here
    would have been a comparison of numbers no offset could change.
    """
    x = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8).half()
    model = _Split(lambda a: torch.split(a, 2, dim=1)[0] + torch.split(a, 2, dim=1)[2])
    raw, commands = _lowered(model, x)
    blits = _blits(commands)
    assert [command.params[4] for command in blits] == [0, 32]
    expected = _values(raw, x)

    body = bytearray(raw)
    at = blob_interpreter.B.HEADER_SIZE + blob_interpreter.B.OP_SIZE + 4 * (4 + 4)
    struct.pack_into("<i", body, at, 12)
    shifted = bytes(body)
    _header, shifted_commands = read_blob(shifted)
    assert shifted_commands[1].params[4] == 12, "the control did not move the offset"
    moved = np.frombuffer(execute(shifted, [x.numpy()])[0], dtype=np.float16)
    assert moved.tobytes() != expected.tobytes(), (
        "a piece addressed three elements early answered the same numbers, so the "
        "offsets the tests above pin are not what decides the answer"
    )
