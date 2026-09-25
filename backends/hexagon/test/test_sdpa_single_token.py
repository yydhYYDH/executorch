# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The one-query-row geometry: what the kernel does with a mask there.

`flash_attn_try_single_token_output` answers `out = V` for `qo_len == 1 &&
seq_current == 0 && seq_add == 1` before any mask code runs
(`attention_entry.cc:215-218,328-331,364-367`) and takes no mask argument at all
(`attention_push_kv.cc:402-410`). A masked call that reaches it therefore has its
mask dropped, silently. `sdpa_mask_fits_dsp_limits` is what keeps that from
happening: it refuses a mask whose query extent is one row, so the graph stays on
the portable kernels. These tests pin that geometry rather than the verdict.

The refused side is asserted at the command level, with the one clause relaxed in
the predicate's own source and every other refusal left intact, so what is read
back is the command the refusal stands in front of: one query row, `seq_current
== 0`, `seq_add == 1`, a positive mask stride and the operand bound. The same
lowering with the clause in place makes no delegate. The controls are the
geometries that have to keep delegating -- one query row without a mask, and a
masked query of two rows -- asserted by their own command words, so closing the
mask channel altogether is a failure here rather than a pass.
"""

import contextlib
import inspect


import torch


import blob_interpreter
from blob_interpreter import read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_ops import (
    _emit_sdpa,
    DSP_OP_FLASH_ATTN,
)
from executorch.backends.hexagon.partition import hexagon_partitioner
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    HexagonPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export

#: [batch, seq, heads, dim] over a cache whose head count divides the query's.
_BATCH, _N_HEADS, _N_KV_HEADS, _MAX_KV_LEN, _HEAD_DIM = 1, 8, 4, 8, 128

#: The op is a stand-in in a namespace of this file's own: the graphs below need
#: a target the emitter table can hold, and `llama.custom_sdpa` is registered by
#: a C++ extension this checkout does not build. The schema is the llama op's
#: (`extension/llm/custom_ops/op_sdpa_aot.cpp:480`) down to the argument order,
#: because the emitter reads the operands by position and nothing else.
_NAMESPACE = "hexagon_single_token_standin"
_SDPA_SCHEMA = (
    "sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, "
    "Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, "
    "float? scale=None) -> Tensor"
)

#: The clause whose removal admits the one-row masked geometry, quoted from the
#: predicate so a reformat of that file fails here loudly instead of quietly
#: changing what these tests relax.
_QUERY_ROWS_CLAUSE = "if query_rows < 2 or query_rows > MASK_QUERY_ROWS_MAX:"

_LIBS = []
_STANDIN = None


def _standin():
    """The edge op the graphs call, defined once per process."""
    global _STANDIN
    if _STANDIN is not None:
        return _STANDIN
    library = torch.library.Library(_NAMESPACE, "FRAGMENT")
    library.define(_SDPA_SCHEMA)
    _LIBS.append(library)

    @torch.library.register_fake(f"{_NAMESPACE}::sdpa")
    def _(
        query,
        key,
        value,
        start_pos,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        return query.new_empty(query.shape)

    _STANDIN = exir_ops.edge.hexagon_single_token_standin.sdpa.default
    return _STANDIN


@contextlib.contextmanager
def _wired():
    """Makes the stand-in delegatable, for one test, and puts the tables back."""
    target = _standin()
    previous_family = hexagon_ops.SDPA_TARGETS
    hexagon_ops.EMITTERS[target] = _emit_sdpa
    hexagon_ops.SDPA_TARGETS = frozenset({target})
    try:
        yield target
    finally:
        del hexagon_ops.EMITTERS[target]
        hexagon_ops.SDPA_TARGETS = previous_family


@contextlib.contextmanager
def _query_rows_clause_relaxed():
    """The predicate with its query-rows lower bound dropped, and nothing else.

    The clause is rewritten in the function's own source and re-executed, so the
    admission under test is that clause and every other refusal is still the
    backend's: the stride clauses, the operand's dtype and rank, the row count
    against the query's, and the cache the stride has to cover all stay as they
    are. Both doors are patched because the partitioner holds its own name for
    the predicate and the emitter calls the module's.
    """
    source = inspect.getsource(hexagon_ops.sdpa_mask_fits_dsp_limits)
    assert (
        _QUERY_ROWS_CLAUSE in source
    ), "the clause these tests relax is not in the predicate any more"
    namespace = vars(hexagon_ops).copy()
    exec(  # noqa: S102 - the source is this module's own predicate, quoted above
        source.replace(
            _QUERY_ROWS_CLAUSE,
            _QUERY_ROWS_CLAUSE.replace("query_rows < 2", "query_rows < 1"),
        ),
        namespace,
    )
    relaxed = namespace["sdpa_mask_fits_dsp_limits"]
    saved_ops = hexagon_ops.sdpa_mask_fits_dsp_limits
    saved_partitioner = hexagon_partitioner.sdpa_mask_fits_dsp_limits
    hexagon_ops.sdpa_mask_fits_dsp_limits = relaxed
    hexagon_partitioner.sdpa_mask_fits_dsp_limits = relaxed
    try:
        yield relaxed
    finally:
        hexagon_ops.sdpa_mask_fits_dsp_limits = saved_ops
        hexagon_partitioner.sdpa_mask_fits_dsp_limits = saved_partitioner


class _Sdpa(torch.nn.Module):
    """One attention call at one query length, with or without the mask."""

    def __init__(self, masked, qo_len):
        super().__init__()
        self.masked = masked
        self.qo_len = qo_len

    def forward(self, query, key, value, mask):
        return torch.ops.hexagon_single_token_standin.sdpa(
            query,
            key,
            value,
            0,  # start_pos: a constant, which is what puts seq_current at 0
            mask if self.masked else None,
            0.0,
            False,
            None,
        )


def _tensors(qo_len):
    torch.manual_seed(11)
    query = torch.randn(_BATCH, qo_len, _N_HEADS, _HEAD_DIM, dtype=torch.float16) * 0.5
    key = torch.randn(_BATCH, _MAX_KV_LEN, _N_KV_HEADS, _HEAD_DIM, dtype=torch.float16)
    value = torch.randn(
        _BATCH, _MAX_KV_LEN, _N_KV_HEADS, _HEAD_DIM, dtype=torch.float16
    )
    mask = torch.full((qo_len, _MAX_KV_LEN), float("-inf"), dtype=torch.float16)
    mask[:, 1:] = 0.0
    return query, key, value, mask


def _lower(masked, qo_len, relax=False):
    """The exported call, its gate verdict, and the delegate blobs it made."""
    _standin()
    # The mask is an input either way: the call takes it, so the graph carries it.
    program = export(_Sdpa(masked, qo_len).eval(), _tensors(qo_len))
    call = [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and "sdpa" in str(node.target)
    ][0]
    with _wired(), _query_rows_clause_relaxed() if relax else contextlib.nullcontext():
        # Through the module, so the relaxed predicate is the one that answers.
        verdict = hexagon_ops.sdpa_mask_fits_dsp_limits(call)
        lowered = to_edge_transform_and_lower(
            program, partitioner=[HexagonPartitioner()]
        ).exported_program()
    blobs = [
        bytes(
            lowered.graph_module.get_submodule(str(node.args[0].target)).processed_bytes
        )
        for node in lowered.graph_module.graph.nodes
        if "call_delegate" in getattr(node.target, "__name__", "")
    ]
    return verdict, blobs


def _command(blobs):
    """The one FLASH_ATTN command these blobs hold."""
    found = [
        (blob, command)
        for blob in blobs
        for command in read_blob(blob)[1]
        if command.type == DSP_OP_FLASH_ATTN
    ]
    assert len(found) == 1, f"{len(found)} attention commands, not one"
    return found[0]


#: The whole command a one-row call emits when there is no mask: its query count,
#: the run-time position, its own count of new rows, the heads and the head
#: count, the packed scale, the stride that means "no mask", the cache capacity
#: and the paged entry's three words.
_ONE_ROW_UNMASKED_PARAMS = [1, 0, 1, 8, 4, 128, 1035273459, -1, 256, 0, 0, 0]


def test_a_masked_one_row_query_stays_on_the_portable_kernels():
    """The geometry the kernel's first-token shortcut drops the mask on.

    Being delegated here is not a slower answer, it is another function: the
    shortcut returns the single value row whatever the mask says, and with the
    mask bound and a stride the kernel would otherwise read.
    """
    verdict, blobs = _lower(True, 1)
    assert verdict is False, "a one-row masked query is not one the kernel applies"
    assert blobs == [], "this graph had to stay on the portable kernels"


def test_a_one_row_query_without_a_mask_still_emits():
    """The control: the shortcut's answer is the right one when nothing masks.

    One key's softmax is one, so `out = V` is what the row is worth, and the
    geometry has to keep delegating. The command is asserted by its words rather
    than by the verdict, so an emitter that stopped binding operands or started
    writing a stride for them could not pass this.
    """
    verdict, blobs = _lower(False, 1)
    assert verdict is True, "one row with no mask is emittable"
    assert len(blobs) == 1, "and it has to reach a delegate"
    _blob, command = _command(blobs)
    assert command.params == _ONE_ROW_UNMASKED_PARAMS
    assert command.inputs[3].space is blob_interpreter.B.TensorSpace.ABSENT
    assert command.inputs[3].size == 0, "slot three is absent, not a bound mask"


def test_the_command_the_refusal_stands_in_front_of():
    """The one-row masked command, read back from the lowering that admits it.

    This is the geometry, not the verdict: with the query-rows clause relaxed in
    the predicate's own source, the lowering produces a FLASH_ATTN whose
    `intParams` are exactly the shortcut's window -- `qo_len == 1`,
    `seq_current == 0`, `seq_add == 1` -- with a positive mask stride and the
    operand bound at `qo_len * stride` two-byte elements. The same call with the
    clause in place makes no delegate, which is what makes this the thing the
    refusal is about rather than a description of it.
    """
    verdict, guarded = _lower(True, 1)
    assert verdict is False and guarded == [], "the clause has to be what refuses"

    verdict, blobs = _lower(True, 1, relax=True)
    assert verdict is True, "the relaxed clause admits it"
    assert len(blobs) == 1, "and it reaches a delegate"
    _blob, command = _command(blobs)
    qo_len, seq_current, seq_add, n_heads, n_kv_heads, head_dim = command.params[:6]
    mask_stride = command.params[7]
    assert (qo_len, seq_current, seq_add) == (
        1,
        0,
        1,
    ), "the kernel's shortcut is qo_len == 1 && seq_current == 0 && seq_add == 1"
    assert (n_heads, n_kv_heads, head_dim) == (_N_HEADS, _N_KV_HEADS, _HEAD_DIM)
    assert mask_stride == _MAX_KV_LEN, "the stride the mask would be read at"
    operand = command.inputs[3]
    assert operand.space is blob_interpreter.B.TensorSpace.INPUT
    assert operand.size == qo_len * mask_stride * 2, "fp16 rows, one per query row"


def test_a_masked_query_of_two_rows_emits_with_its_stride():
    """The other control: the mask channel itself has to stay open.

    Two rows is the shortest geometry the kernel applies a mask on, and it is
    where the refused case would land if the refusal were written too wide.
    """
    verdict, blobs = _lower(True, 2)
    assert verdict is True, "two rows with a mask is emittable"
    assert len(blobs) == 1, "and it has to reach a delegate"
    _blob, command = _command(blobs)
    assert command.params[:3] == [2, 0, 2]
    assert command.params[7] == _MAX_KV_LEN
    assert command.inputs[3].size == 2 * _MAX_KV_LEN * 2
