# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A broadcasting expand_copy on the real Hexagon raster-blit kernel.

The command is a RASTER_BLIT whose region holds the source still on the axes
that repeat, so the question these cases answer is the DSP's and not the
emitter's: whether a zero source stride reads the same element again. Each case
is chosen to land on a different fast path inside htp_ops_raster_blit, because
a single shape would only prove the one path it took.

The blobs are the ones the partitioner produced, read out of the delegate, so
what the simulator runs is what the backend emits rather than a blob this test
assembled.
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

import hexagon_sim  # noqa: E402
from blob_interpreter import read_blob  # noqa: E402
from test_blob_on_sim import (  # noqa: E402
    _HT_P_OP_SHIM,
    _RUNNER,
    _SCHEMA,
    _SOURCES,
    _bits,
    _case,
    _fixture_header,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    HexagonPartitioner,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    to_edge_transform_and_lower,
)
from torch.export import export  # noqa: E402

RASTER_BLIT = 3

#: source shape, destination shape, and the runs the region is built from. The
#: last case alternates four times and is the refusal.
CASES = [
    # Zero at the middle level, innermost run contiguous: the RoPE half-rotary
    # repeat_interleave, at the shape the partition gate census measured.
    ((1, 8, 1, 32, 128), (1, 8, 2, 32, 128)),
    # Zero at the innermost level with a run of at least 64, which is the shape
    # htp_ops_try_broadcast_inner_fp16_blit is written to recognise.
    ((1, 4, 1), (1, 4, 128)),
    # Zero at the middle level with a short contiguous innermost run, so the
    # row-copy path takes it rather than the broadcast fill.
    ((3, 1, 5), (3, 7, 5)),
    # One run short of three, so two of the levels are the neutral ones this
    # emitter pads on the outside.
    ((1, 1, 6), (2, 2, 6)),
    # Four runs, which three levels cannot walk.
    ((1, 2, 1, 2), (3, 2, 5, 2)),
]


class _Expand(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.expand(self.shape)


def _lower(source_shape, destination_shape):
    """Lower one expand and return its delegate's blob, or None."""
    x = torch.randn(source_shape, dtype=torch.float16)
    program = to_edge_transform_and_lower(
        export(_Expand(destination_shape), (x,)),
        partitioner=[HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    ).exported_program()
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.target is torch.ops.higher_order.executorch_call_delegate
    ]
    if not calls:
        return None
    assert len(calls) == 1, f"{len(calls)} delegates for one expand"
    lowered = program.graph_module.get_submodule(calls[0].args[0].target)
    return bytes(lowered._processed_bytes)


def _region(blob):
    _, commands = read_blob(blob)
    assert [command.type for command in commands] == [RASTER_BLIT], commands
    params = list(commands[0].params)
    assert params[:3] == [1, 2, 1], params[:3]
    return params[3:]


@pytest.mark.parametrize("source,destination", CASES)
def test_a_broadcasting_expand_is_one_blit_with_a_zero_source_stride(
    source, destination
):
    """The command, and the region that carries the repeat."""
    blob = _lower(source, destination)
    if blob is None:
        assert len(destination) == 4, (source, destination)
        return
    region = _region(blob)
    assert region[:3] == [0, 0, 0], region
    size, source_stride, result_stride = region[3:6], region[6:9], region[9:12]
    assert 0 in source_stride, region
    assert all(count >= 1 for count in size), region
    assert result_stride[2] == 1, region
    # The last element the walk writes, which is the bound the region has to
    # respect: the destination is a dense buffer of exactly this many.
    covered = sum((count - 1) * pitch for count, pitch in zip(size, result_stride)) + 1
    assert covered == int(np.prod(destination)), (region, destination)


def test_a_four_run_expand_stays_portable():
    """The refusal, with a positive control at the geometry beside it."""
    assert _lower((1, 2, 1, 2), (3, 2, 5, 2)) is None
    # The same operand, the same rank and the same element count, with one
    # fewer run, does delegate -- so the refusal above is the run count and
    # not the shape or the size.
    assert _lower((1, 2, 1, 2), (1, 2, 5, 2)) is not None


def test_a_broadcasting_expand_answers_on_hexagon_sim():
    """The DSP's own arithmetic, for every case that delegates."""
    hexagon_sim._check()
    fixtures = []
    expected_by_tag = {}

    for index, (source_shape, destination_shape) in enumerate(CASES):
        blob = _lower(source_shape, destination_shape)
        if blob is None:
            continue
        tag = f"XB{index}"
        x = torch.arange(
            1, int(np.prod(source_shape)) + 1, dtype=torch.float16
        ).reshape(source_shape)
        expected = x.expand(destination_shape)
        fixtures.append(
            _case(tag, None, (x,), _bits(expected.reshape(-1)), blob=blob)
        )
        expected_by_tag[tag] = expected

    assert expected_by_tag, "no case reached a delegate"

    simulated = hexagon_sim.run(
        _RUNNER,
        _SOURCES,
        headers={
            "blob_fixture.h": _fixture_header(fixtures),
            "htp_ops.h": _HT_P_OP_SHIM,
        },
        includes_more=[str(_SCHEMA)],
    )

    for tag, expected in expected_by_tag.items():
        output_tag = f"{tag}0"
        assert output_tag in simulated, hexagon_sim.LAST_STDOUT
        output = np.asarray(simulated[output_tag], dtype=np.uint16).view(np.float16)
        reference = expected.reshape(-1).numpy()
        assert output.size == reference.size, (tag, output.size, reference.size)
        assert np.array_equal(output, reference), tag
