# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The partitioner's answers about a node whose emitter adds no command."""

import os
import pathlib
import sys

import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    _cast_stays_in_fp16,
    _emits_no_command,
)
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402


def _cast(source_dtype, result_dtype):
    """A _to_copy node with the two dtypes it is judged on."""
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(4, dtype=source_dtype)
    cast = graph.call_function(
        exir_ops.edge.aten._to_copy.default,
        args=(source,),
        kwargs={"dtype": result_dtype},
    )
    cast.meta["val"] = torch.empty(4, dtype=result_dtype)
    return cast


def test_a_cast_is_absorbed_only_between_the_two_widths_the_arena_holds():
    """The operand is what decides, because the result alone is not enough.

    A cast whose result is fp16 passes the dtype gate by itself, so an int64
    operand would be absorbed and the kernel would read eight-byte integers as
    half floats. The fp16 and fp32 pair is the one the runtime already converts
    at the boundary in both directions, so absorbing it changes no value.
    """
    assert _cast_stays_in_fp16(_cast(torch.float16, torch.float32))
    assert _cast_stays_in_fp16(_cast(torch.float32, torch.float16))
    assert not _cast_stays_in_fp16(_cast(torch.int64, torch.float16))
    assert not _cast_stays_in_fp16(_cast(torch.float16, torch.int64))
    assert _emits_no_command(_cast(torch.float16, torch.float32))
    assert not _emits_no_command(_cast(torch.int64, torch.float16))
