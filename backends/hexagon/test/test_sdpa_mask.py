# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A masked sdpa, which this backend cannot express and must not delegate.

`llama.custom_sdpa` takes an optional attention mask as its fifth operand, and
`SDPACustom(use_attention_mask=True)` -- the exporter's
`use_custom_sdpa_with_attention_mask` -- always passes one. The FLASH_ATTN
emitter binds that operand to slot three but sends `mask_stride = -1`, and the
kernel reads a negative stride as "no mask" (`attention_entry.cc` copies the
mask into fp32 only `if (maskBase != NULL && mask_stride > 0)`), so a delegated
mask is attention computed as if it were not there: a wrong answer, not a
failure. The stride the kernel wants is not available either -- it reads rows of
`mask_stride` two-byte elements right-aligned to the keys, and copies them into
a fp32 region it places after the per-task rows, out of a workspace this backend
sizes for the unmasked shape and `htp_ops_flash_attn` does not bounds-check. A
masked node therefore stays on the portable kernels, and the emitter refuses one
that reaches it anyway.

The llama extension that defines the op is a C++ library this checkout does not
build (`torch.ops.llama.custom_sdpa` and so `exir_ops.edge.llama.custom_sdpa`
do not exist here, which is why `sdpa_targets()` is empty), so these tests drive
the two decision functions with a node built by hand. Both listen to the
operands only, so the node's target is whatever a graph would have had.
"""

import os
import pathlib
import sys

import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes `import executorch` resolve to this tree. Without it the
# editable install wins, and in this environment that points at a different
# checkout, which has an older backend in it.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

from executorch.backends.hexagon.hexagon_ops import (  # noqa: E402
    _emit_sdpa,
    sdpa_targets,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (  # noqa: E402
    _sdpa_fits_dsp_limits,
)

#: Attention geometry the emitter accepts: [batch, seq, heads, dim].
_BATCH, _QO_LEN, _N_HEADS, _N_KV_HEADS, _MAX_KV_LEN, _HEAD_DIM = 1, 3, 16, 8, 8, 128

#: Stands in for the op the graph would carry. `_sdpa_fits_dsp_limits` reads the
#: operands and the emitter is called directly, so neither looks at the target.
_TARGET = torch.ops.aten.mm.default


def _placeholder(graph, name, shape, dtype=torch.float16):
    node = graph.placeholder(name)
    node.meta["val"] = torch.empty(shape, dtype=dtype)
    return node


def _sdpa_node(mask):
    """A custom_sdpa call, with or without its mask operand."""
    graph = torch.fx.Graph()
    query = _placeholder(graph, "query", (_BATCH, _QO_LEN, _N_HEADS, _HEAD_DIM))
    key = _placeholder(graph, "key", (_BATCH, _MAX_KV_LEN, _N_KV_HEADS, _HEAD_DIM))
    value = _placeholder(graph, "value", (_BATCH, _MAX_KV_LEN, _N_KV_HEADS, _HEAD_DIM))
    if mask is not None:
        mask = _placeholder(graph, "mask", (_QO_LEN, _MAX_KV_LEN), mask)
    node = graph.call_function(
        _TARGET,
        args=(
            query,
            key,
            value,
            0,  # start_pos
            mask,
            0.0,  # dropout_p
            False,  # is_causal, which a mask replaces
            None,  # scale
        ),
    )
    node.meta["val"] = torch.empty(
        (_BATCH, _QO_LEN, _N_HEADS, _HEAD_DIM), dtype=torch.float16
    )
    return node


class _RefusingContext:
    """Fails the moment the emitter reaches for anything to emit with."""

    def __getattr__(self, name):
        raise AssertionError(
            f"the emitter touched ctx.{name} on a masked sdpa instead of refusing"
        )


def test_a_masked_sdpa_is_not_delegated_and_an_unmasked_one_still_is():
    """The refusal is about the mask, not about attention in general."""
    if sdpa_targets():
        # Registration is process-global: a build that has the llama extension
        # and imported it earlier in this pytest process registers the op here
        # too, and the predicates below would then be checked against a graph
        # this file did not build. Skip rather than fail, because that is a
        # different environment, not a different answer -- but read the reason.
        pytest.skip(
            "llama.custom_sdpa is registered in this process, so the mask "
            "decision has to be re-checked against a real graph"
        )
    assert _sdpa_fits_dsp_limits(_sdpa_node(None))
    assert not _sdpa_fits_dsp_limits(_sdpa_node(torch.float16))
    assert not _sdpa_fits_dsp_limits(_sdpa_node(torch.float32))


def test_the_emitter_refuses_a_mask_rather_than_binding_one():
    """The second door: preprocess on a graph the partitioner never saw.

    With mask_stride = -1 the kernel ignores whatever slot three points at, so
    binding the mask is the silent wrong answer this refuses to emit.
    """
    for dtype in (torch.float16, torch.float32):
        with pytest.raises(RuntimeError, match="attention mask"):
            _emit_sdpa(_sdpa_node(dtype), _RefusingContext())
