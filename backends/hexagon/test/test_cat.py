# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A concatenation whose graph dtype is fp32, and the one clause that refused it.

A region list carries a single element width, so cat_region used to demand fp16 of
every operand -- the one emitter in this backend that did. The arena holds two bytes
per element whatever the graph says: a fp32 method input gets a slot half the size its
graph declares and the runtime narrows it on the way in, and every activation is
allocated at result_for's fp16 default. So the operand the blit reads is two bytes
wide in both cases and the same region describes both.

That mattered for one model in particular. A diffusion UNet's timestep embedding
concatenates its cosine and sine halves in fp32, and the refusal took the whole
time-embedding prologue -- five nodes -- onto the portable kernels and split the graph
into two delegates.

So the claim these tests rest on is measured rather than quoted: the fp16 and the fp32
cat emit byte-identical commands, the fp32 input slot is half the size the graph
declared, and the dtype clause is the thing that was doing the refusing.
"""

import inspect
import os
import pathlib
import sys

import torch

# The checkout directory is itself named executorch, so putting its parent on the
# path makes "import executorch" resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[3]))
# The test directory is not a package, so the interpreter is importable by name.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

from blob_interpreter import read_blob  # noqa: E402
from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    cat_plan,
    cat_region,
    DSP_OP_RASTER_BLIT,
    FP16_BYTES,
)
from executorch.backends.hexagon.partition import hexagon_partitioner  # noqa: E402
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
from torch.export import export  # noqa: E402

#: The clause as it stands, and the clause as it was: the same check with fp32 taken
#: out of the accepted set, which is the whole of the change. The first conjunct is
#: the admitted set, the second the homogeneity the blit header's single element
#: size needs -- a mixed cat has to be refused by one of them, not both.
_CLAUSE = (
    "    if result.dtype not in (torch.float16, torch.float32) or any(\n"
    "        value.dtype is not result.dtype for value in values\n"
    "    ):\n"
    "        return None"
)
_FP16_ONLY = (
    "    if any(value.dtype is not torch.float16 for value in values):\n"
    "        return None"
)

NUMEL = 32


class _Cat(torch.nn.Module):
    """A diffusion timestep embedding's rejoin: cos and sin, concatenated."""

    def forward(self, a, b):
        return torch.cat([a, b], dim=-1)


def _lowered(dtype, cat_region_fn=None):
    """The lowered program and every delegate blob it produced.

    `cat_region_fn` substitutes a different spec function for the duration of the
    lowering. Both the module's own name and the partitioner's copy have to be
    patched: the partitioner imported the name, so rebinding it in one module
    leaves the other serving the shipped one and the experiment measures nothing.
    """
    module = _Cat().eval()
    args = (
        torch.randn(1, NUMEL // 2, dtype=dtype),
        torch.randn(1, NUMEL // 2, dtype=dtype),
    )
    if cat_region_fn is None:
        program = _lower(module, args)
    else:
        # The width clause lives in cat_plan. The partitioner imported the name, so
        # both copies have to be patched, which is the same trap this harness
        # already documents for the spec function.
        hexagon_ops.cat_plan = cat_region_fn
        hexagon_partitioner.cat_plan = cat_region_fn
        try:
            program = _lower(module, args)
        finally:
            hexagon_ops.cat_plan = cat_plan
            hexagon_partitioner.cat_plan = cat_plan
    gm = program.graph_module
    delegates = [
        getattr(gm, node.target)
        for node in program.graph.nodes
        if node.op == "get_attr" and node.target.startswith("lowered_module")
    ]
    return program, [bytes(d._processed_bytes) for d in delegates]


def _lower(module, args):
    return to_edge_transform_and_lower(
        export(module, args),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()


def _cat_node(program):
    """The cat node, wherever the partitioner put it.

    A delegated cat is not in the outer graph: its node lives in the delegate
    module's own graph, under the outer graph's node name. So look in both, and
    fail if there is more than one, which would make "the node" ambiguous.
    """
    gm = program.graph_module
    graphs = [gm.graph]
    for node in program.graph.nodes:
        if node.op == "get_attr" and node.target.startswith("lowered_module"):
            inner = getattr(gm, node.target).original_module
            if inner is not None and hasattr(inner, "graph"):
                graphs.append(inner.graph)
    found = [
        node
        for graph in graphs
        for node in graph.nodes
        if node.op == "call_function" and "cat" in str(node.target)
    ]
    assert len(found) == 1, f"expected one cat node, found {len(found)}"
    return found[0]


def _command_types(blobs):
    out = []
    for blob in blobs:
        _, commands = read_blob(blob)
        out.append([command.type for command in commands])
    return out


def test_both_widths_reach_the_dsp_with_one_blit_each():
    """The positive control, at the same geometry for both dtypes.

    Two widths in one test because either alone is satisfiable by a broken change: a
    backend that refused everything passes "the fp32 one delegates" only if the fp16
    one is not in the same assertion.
    """
    for dtype in (torch.float16, torch.float32):
        _, blobs = _lowered(dtype)
        assert len(blobs) == 1, f"{dtype}: expected one delegate, got {len(blobs)}"
        assert _command_types(blobs) == [[DSP_OP_RASTER_BLIT]], (
            f"{dtype}: a cat is one blit, not {_command_types(blobs)}"
        )


def test_the_region_reads_two_bytes_per_element_at_both_widths():
    """The measurement the widening rests on, as a test rather than a quotation.

    A fp32 graph says its operands are twice the size the arena gives them. If the
    region carried the graph's width the blit would read past the end of the slot, so
    this asserts the two halves together: the slot is half the declared byte count
    and the element-width field of the command is two either way.
    """
    for dtype in (torch.float16, torch.float32):
        _, blobs = _lowered(dtype)
        header, commands = read_blob(blobs[0])
        assert len(commands) == 1
        command = commands[0]
        declared = (NUMEL // 2) * torch.tensor([], dtype=dtype).element_size()
        slots = [ref.size for ref in command.inputs]
        assert slots == [NUMEL, NUMEL], (
            f"{dtype}: each operand slot is {slots} bytes for {NUMEL} elements, "
            f"which is not {FP16_BYTES} a piece"
        )
        if dtype is torch.float32:
            assert declared == NUMEL * 2 and slots == [declared // 2, declared // 2], (
                f"a fp32 graph declares {declared} bytes an operand and the slot is "
                f"{slots}; the narrowing is the claim under test"
            )
        assert command.params[1] == FP16_BYTES, (
            f"{dtype}: the region element width is {command.params[1]}, not "
            f"{FP16_BYTES}"
        )


def test_the_dtype_clause_is_the_one_that_was_refusing():
    """Put the fp16-only clause back and the fp32 cat stops delegating.

    Without this the two tests above are consistent with a backend that always
    emitted fp32 cats by some other route, and the clause would be untested. The
    assertion on the source text is what keeps the test honest across a reformat:
    if the clause is reworded this fails before it can pass vacuously.
    """
    source = inspect.getsource(cat_plan)
    assert _CLAUSE in source, (
        "the fp32 clause is not the text this test replaces; re-read it and update "
        "_CLAUSE, or this test is testing nothing"
    )
    narrowed = dict(vars(hexagon_ops))
    exec(  # noqa: S102
        compile(source.replace(_CLAUSE, _FP16_ONLY), "cat_plan.py", "exec"),
        narrowed,
    )
    fp16_only = narrowed["cat_plan"]

    for dtype, expected in ((torch.float16, True), (torch.float32, False)):
        program, blobs = _lowered(dtype, cat_region_fn=fp16_only)
        assert bool(blobs) is expected, (
            f"{dtype}: under the fp16-only clause the graph should "
            f"{'keep' if expected else 'lose'} its delegate, and it has "
            f"{len(blobs)}"
        )
        node = _cat_node(program)
        assert (fp16_only(node) is not None) is expected, (
            f"{dtype}: the clause has to be what decides the node, not something "
            "further down"
        )
        assert cat_region(node) is not None, (
            f"{dtype}: the widened clause has to admit it, or the fix is not here"
        )


def test_a_width_the_arena_does_not_hold_is_still_refused():
    """The clause still discriminates: widening its accepted set is not dropping it.

    int64 is the width this backend keeps portable everywhere, and it is the same line
    that changed, so a change that deleted the clause rather than widening it passes
    the three tests above and fails here.
    """

    class _IntCat(torch.nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=-1)

    program = to_edge_transform_and_lower(
        export(_IntCat().eval(), (torch.ones(1, 8, dtype=torch.int64),
                                  torch.ones(1, 8, dtype=torch.int64))),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    node = _cat_node(program)
    values = [a.meta.get("val") for a in node.all_input_nodes]
    assert values and all(v.dtype is torch.int64 for v in values), (
        f"the control did not produce an int64 cat: {values}"
    )
    assert cat_region(node) is None, "an int64 operand must not reach a two-byte region"


def test_more_than_three_operands_is_still_refused():
    """The other gate on this spec function, to show the change did not move it.

    A blit header is three ints and each region twelve, and an op carries forty
    params, so three inputs is all one command can describe. Four is still refused.
    """

    class _WideCat(torch.nn.Module):
        def forward(self, a, b, c, d):
            return torch.cat([a, b, c, d], dim=-1)

    program = to_edge_transform_and_lower(
        export(_WideCat().eval(), tuple(torch.randn(1, 4, dtype=torch.float16) for _ in range(4))),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    node = _cat_node(program)
    assert len(node.args[0]) == 4, f"the control is not a four-way cat: {node.args[0]}"
    assert cat_region(node) is None, "four operands do not fit one blit command"
