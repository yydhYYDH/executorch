# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A masked sdpa: what the kernel does with the mask, and what the emitter says.

`llama.custom_sdpa` takes an optional attention mask as its fifth operand, and
`SDPACustom(use_attention_mask=True)` -- what
`use_custom_sdpa_with_attention_mask` builds -- always passes one. The FLASH_ATTN
kernel reads that operand as `[qo_len, mask_stride]` two-byte rows at
`maskStride = intParams[7]` (`attention_entry.cc:72-88,237-242`) and adds them to
the fp32 scores before the softmax (`attention_sync_process.cc:508-522`), so the
operand and the stride are one contract, and only a positive stride makes the
mask reach the arithmetic at all: a negative stride is the kernel's other mode,
where it generates the causal clamp and ignores slot three
(`attention_sync_process.cc:55-58,466-496`). Binding a mask and sending -1, or
sending a stride with an absent operand, is a wrong answer rather than a
failure, which is the shape of bug these tests exist to catch.

So the mask is delegated only where the kernel applies it, and the refusals are
pinned separately from the acceptances. What "the mask was applied" means is then
established twice, and neither check reads a comment: the command the lowering
produced is inspected for the operand, the stride and the reservation the
kernel's fp32 copy needs, and the host interpreter -- which walks the kernel's
own addressing -- is run against a numpy reference for a mask that has to change
the answer and for a mask that must not.

The llama extension that defines the op is a C++ library this checkout does not
build, so `torch.ops.llama.custom_sdpa` does not exist here and no graph can
carry it. The graphs below therefore carry a stand-in with the same schema, in a
namespace of this file's own, and each test that needs one puts it into the
emitter table for the length of that test. The table is what makes a target
delegatable, and the llama entry appears in it on its own when the extension is
loaded; nothing here touches `sdpa_targets()` or the backend's own namespace, so
neither whether the suite thinks the llama op is registered nor the size of the
`et_hexagon` family depends on whether this file ran.
"""

import contextlib
import struct

import numpy as np
import pytest
import torch


import blob_interpreter
from blob_interpreter import execute, read_blob
from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.hexagon_ops import (
    _attention_mask_bytes,
    _attention_workspace_bytes,
    _emit_sdpa,
    DSP_OP_FLASH_ATTN,
    sdpa_mask_fits_dsp_limits,
)
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _sdpa_fits_dsp_limits,
    HexagonPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export

#: [batch, seq, heads, dim] over a cache whose head count divides the query's,
#: which is the GQA shape the kernel and the emitter both carry.
_BATCH, _QO_LEN, _N_HEADS, _N_KV_HEADS, _MAX_KV_LEN, _HEAD_DIM = 1, 4, 8, 4, 8, 128

#: The stand-in's namespace, which is its own on purpose. `et_hexagon` is the
#: backend's family and a census test pins its exact size, and whether
#: `llama.custom_sdpa` is registered is another test's subject, so neither can
#: carry an op that only exists for these tests. EXIR's edge dialect builds a
#: packet lazily for any namespace `torch.ops` holds, so no registration file is
#: needed either.
_NAMESPACE = "hexagon_mask_standin"
_SDPA_SCHEMA = (
    "sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, "
    "Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, "
    "float? scale=None) -> Tensor"
)

_LIBS = []
_STANDIN = None


def _standin():
    """The edge op the graphs call, defined once per process.

    The schema is the llama op's (`extension/llm/custom_ops/op_sdpa_aot.cpp:480`)
    down to the argument order, because the emitter reads the operands by
    position and nothing else about the node. A `Library` is unregistered when it
    is collected, so it is held in a module-level list.
    """
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

    _STANDIN = exir_ops.edge.hexagon_mask_standin.sdpa.default
    return _STANDIN


@contextlib.contextmanager
def _delegated():
    """Makes the stand-in delegate, for one test, and puts the tables back.

    `SUPPORTED_TARGETS` is the emitter table itself (`hexagon_backend.py:36`
    aliases it), and the partitioner reads `sdpa_targets()` per node, so an entry
    plus a family is all it takes to send the node through the real preprocess.
    Restoring both is the point of the context: another test asserts this family
    is empty where the llama extension is absent, and it has to stay empty.
    """
    target = _standin()
    previous_family = hexagon_ops.SDPA_TARGETS
    assert target not in hexagon_ops.EMITTERS, "the stand-in was already wired"
    hexagon_ops.EMITTERS[target] = _emit_sdpa
    hexagon_ops.SDPA_TARGETS = frozenset({target})
    try:
        yield target
    finally:
        del hexagon_ops.EMITTERS[target]
        hexagon_ops.SDPA_TARGETS = previous_family


class _Sdpa(torch.nn.Module):
    """One attention call, with or without the mask operand."""

    def __init__(self, masked):
        super().__init__()
        self.masked = masked

    def forward(self, query, key, value, mask):
        return torch.ops.hexagon_mask_standin.sdpa(
            query,
            key,
            value,
            0,  # start_pos: a constant, so the command carries it
            mask if self.masked else None,
            0.0,
            False,
            None,
        )


class _ConstMaskSdpa(torch.nn.Module):
    """One attention call whose mask is a buffer rather than an input.

    The exporter folds a buffer like this into a weight, so the operand reaches
    the command through the weights section while an input mask reaches it as a
    delegate input -- two paths through the emitter for the same slot, and the
    constant one goes through `tensor.to(torch.float16)` instead of the
    round-half-away narrowing the runtime applies to an input of the same dtype.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, query, key, value):
        return torch.ops.hexagon_mask_standin.sdpa(
            query, key, value, 0, self.mask, 0.0, False, None
        )


def _tensors(mask=None, qo_len=_QO_LEN, batch=_BATCH):
    """A query, a cache and a mask, in the order the delegate takes them."""
    torch.manual_seed(11)
    query = torch.randn(batch, qo_len, _N_HEADS, _HEAD_DIM, dtype=torch.float16) * 0.5
    key = (
        torch.randn(batch, _MAX_KV_LEN, _N_KV_HEADS, _HEAD_DIM, dtype=torch.float16)
        * 0.5
    )
    value = torch.randn(batch, _MAX_KV_LEN, _N_KV_HEADS, _HEAD_DIM, dtype=torch.float16)
    if mask is None:
        mask = torch.zeros(qo_len, _MAX_KV_LEN, dtype=torch.float32)
    return (query, key, value, mask)


def _lower(masked, tensors, dynamic_q=False, dynamic_kv=False, dynamic_key=False):
    """The exported attention call, and the delegate blobs the lowering made.

    The call is the one the exporter built, before the partitioner may have
    replaced it with a delegate, which is the node every decision below is made
    about.
    """
    _standin()  # the graph calls the stand-in, so it has to exist to export
    # One entry per operand: query, key, value, mask.
    shapes = [
        {1: Dim("q", min=2, max=16)} if dynamic_q else None,
        # A cache whose row count cannot bound the mask stride, and a command
        # with no room for a symbolic one either, so this stays portable.
        {1: Dim("cache", min=_MAX_KV_LEN, max=64)} if dynamic_key else None,
        None,
        {1: Dim("kv", min=1, max=_MAX_KV_LEN)} if dynamic_kv else None,
    ]
    program = export(_Sdpa(masked).eval(), tensors, dynamic_shapes=shapes)
    calls = [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and "sdpa" in str(node.target)
    ]
    assert len(calls) == 1, f"{len(calls)} attention calls in the exported program"
    return calls[0], _delegates(program)


def _delegates(program):
    """The delegate blobs a program lowers to, through the attention family.

    The stand-in is wired into the emitter table for the length of this call and
    taken back out, so the blobs are the ones the real preprocess would build
    without this file changing what any other test sees.
    """
    _standin()
    with _delegated():
        lowered = to_edge_transform_and_lower(
            program, partitioner=[HexagonPartitioner()]
        ).exported_program()
    delegates = [
        lowered.graph_module.get_submodule(str(node.args[0].target))
        for node in lowered.graph_module.graph.nodes
        if "call_delegate" in getattr(node.target, "__name__", "")
    ]
    assert all(
        delegate.backend_id == "HexagonBackend" for delegate in delegates
    ), "a delegate from another backend would make the command count meaningless"
    return [bytes(delegate.processed_bytes) for delegate in delegates]


def _attention(blobs):
    """The one FLASH_ATTN command these blobs hold, with its blob.

    One, because a lowerer that split the attention in two would make every
    number below an average of two things.
    """
    found = [
        (blob, command)
        for blob in blobs
        for command in read_blob(blob)[1]
        if command.type == DSP_OP_FLASH_ATTN
    ]
    assert len(found) == 1, f"{len(found)} attention commands, not one"
    return found[0]


def _run(blob, tensors):
    """The command's output, on the host interpreter.

    The operands are handed over the way the runtime hands them to the arena.
    The blob declares two bytes an element for every float operand and the
    runtime narrows an fp32 input by casting it into the half-width slot
    (`narrow_fp32_to_fp16`, `runtime/hexagon_backend.cpp:165`, applied where the
    slot is exactly `numel * 2` at `:2122-2130`), so a mask a graph built in
    fp32 arrives as fp16 and the cast is the caller's half of that. An operand
    the graph did not use is not in the signature at all -- an unmasked sdpa
    takes three inputs, not four -- which is why the count comes from the blob.
    """
    header, _ = read_blob(blob)
    fed = [tensor.numpy() for tensor in tensors[: header.n_inputs]]
    if header.n_inputs == 4 and fed[3].dtype == np.float32:
        fed[3] = fed[3].astype(np.float16)
    return np.frombuffer(execute(blob, fed)[0], dtype=np.float16).reshape(
        _BATCH, _QO_LEN, _N_HEADS, _HEAD_DIM
    )


def _reference(query, key, value, mask, command, clamp):
    """Attention over the same operands, written from the shapes.

    The geometry comes from the command rather than from this file, so a command
    that addressed other rows than these tensors hold could not agree with it.
    `clamp` is the kernel's other mode, where row q reads keys `0..q` and no mask
    exists to add.
    """
    qo_len, n_heads, n_kv_heads, head_dim = (
        command.params[0],
        command.params[3],
        command.params[4],
        command.params[5],
    )
    query = query.float().numpy().reshape(-1, n_heads, head_dim)
    key = key.float().numpy().reshape(-1, n_kv_heads, head_dim)
    value = value.float().numpy().reshape(-1, n_kv_heads, head_dim)
    # The emitter packs the scale as a float's bits in an int word.
    scale = struct.unpack("<f", struct.pack("<i", command.params[6]))[0]
    group = n_heads // n_kv_heads
    out = np.zeros((qo_len, n_heads, head_dim), dtype=np.float32)
    for row in range(qo_len):
        valid = row + 1 if clamp else len(key)
        for head in range(n_heads):
            scores = (query[row, head] @ key[:valid, head // group].T) * scale
            if not clamp:
                scores = scores + mask[row, :valid].numpy().astype(np.float32)
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            out[row, head] = weights @ value[:valid, head // group]
    return out.reshape(_BATCH, _QO_LEN, _N_HEADS, _HEAD_DIM).astype(np.float16)


# ---------------------------------------------------------------------------
# The decision, before any command exists.
# ---------------------------------------------------------------------------


def _node(mask, qo_len=_QO_LEN, batch=_BATCH, dtype=torch.float32):
    """A hand-built attention node, the way the gate sees one.

    The target is any call_function: the gate reads the operands, not the
    overload, and a real one would need the llama extension this checkout has
    not built.
    """
    graph = torch.fx.Graph()
    query = graph.placeholder("query")
    query.meta["val"] = torch.empty(
        batch, qo_len, _N_HEADS, _HEAD_DIM, dtype=torch.float16
    )
    key = graph.placeholder("key")
    key.meta["val"] = torch.empty(
        batch, _MAX_KV_LEN, _N_KV_HEADS, _HEAD_DIM, dtype=torch.float16
    )
    value = graph.placeholder("value")
    value.meta["val"] = key.meta["val"].clone()
    if mask is not None:
        operand = graph.placeholder("mask")
        operand.meta["val"] = torch.empty(*mask, dtype=dtype)
        mask = operand
    node = graph.call_function(
        torch.ops.aten.mm.default, args=(query, key, value, 0, mask, 0.0, False, None)
    )
    node.meta["val"] = query.meta["val"].clone()
    return node


def test_a_mask_the_kernel_reads_is_accepted_and_the_rest_are_refused():
    """The clauses that make the operand and the stride a command can carry.

    Every acceptance is a shape the kernel indexes correctly -- `mask_stride`
    columns of two-byte elements, one row per query row -- and every refusal is a
    geometry where one of those numbers is not the one the command says it is.
    """
    assert sdpa_mask_fits_dsp_limits(_node(None)), "an unmasked node is unaffected"
    for rows in (2, _QO_LEN, 64):
        # A stride narrower than the cache hides the keys past it and leaves the
        # ones before it unmasked, because a row's columns sit at the end of the
        # keys. That is a window the kernel implements and a length mistake this
        # cannot tell apart, so it stays portable.
        for stride in (_MAX_KV_LEN, 32):
            assert sdpa_mask_fits_dsp_limits(_node((rows, stride), qo_len=rows)), (
                rows,
                stride,
            )
    assert sdpa_mask_fits_dsp_limits(_node((1, _QO_LEN, _MAX_KV_LEN))), "rank three"
    assert sdpa_mask_fits_dsp_limits(_node((_QO_LEN, _MAX_KV_LEN), dtype=torch.float16))

    # One query row: `flash_attn_try_single_token_output` returns `pV` for
    # qo_len == 1 && seq_current == 0 && seq_add == 1 before any mask code runs
    # (attention_entry.cc:215-218,364-367), so this is the geometry whose mask
    # the kernel would drop. The mask's rows are the query's, or the row-count
    # clause is what refuses this node and the assertion holds whatever the
    # query extent was.
    assert not sdpa_mask_fits_dsp_limits(
        _node((1, _MAX_KV_LEN), qo_len=1)
    ), "the first-token shortcut would ignore the mask"
    # Past 64 rows the kernel stops segmenting the query into blocks and the
    # emitter's reservation is sized for 64 (attention_sync_setup.cc:263-266).
    assert not sdpa_mask_fits_dsp_limits(_node((65, _MAX_KV_LEN), qo_len=65))
    # A row count that is not the query's is read short or read past its end.
    assert not sdpa_mask_fits_dsp_limits(_node((_QO_LEN + 1, _MAX_KV_LEN)))
    assert not sdpa_mask_fits_dsp_limits(_node((_QO_LEN - 1, _MAX_KV_LEN)))
    # The stride is a command word, so 0 and below are the kernel's other two
    # modes rather than a mask.
    assert not sdpa_mask_fits_dsp_limits(_node((_QO_LEN, 0)))
    assert not sdpa_mask_fits_dsp_limits(_node((_QO_LEN, _MAX_KV_LEN - 1)))
    # These kernels read two-byte elements, and a one-dimensional operand has no
    # rows for the query to be indexed by.
    assert not sdpa_mask_fits_dsp_limits(
        _node((_QO_LEN, _MAX_KV_LEN), dtype=torch.bool)
    )
    assert not sdpa_mask_fits_dsp_limits(_node((_QO_LEN,)))

    # The same predicate, on the nodes the exporter builds rather than on
    # hand-made ones: a run-time query extent and a run-time stride are both
    # numbers the command cannot carry, and both are graphs that export.
    accepted = _lower(True, _tensors(mask=_causal_mask()))
    assert sdpa_mask_fits_dsp_limits(accepted[0])
    for dynamic in ("dynamic_q", "dynamic_kv", "dynamic_key"):
        program = _lower(
            True,
            _tensors(mask=_causal_mask()),
            **{dynamic: True},
        )
        assert program[1] == [], f"a {dynamic} graph has to stay portable"
        assert not _sdpa_fits_dsp_limits(program[0]), dynamic


def test_a_query_batch_the_command_cannot_carry_is_refused_with_or_without_a_mask():
    """The command has one row count and no batch axis.

    The kernel walks `query + q * heads * headDim` for q below qo_len and writes
    the same rows back (`attention_entry.cc:164-166`), so a batch of two would
    have its first batch computed and the second left as the arena found it.
    That is the same class of silent wrong answer as a dropped mask, it is in the
    same command, and it is guarded in the same gate.
    """
    for batch in (2, 3):
        assert not _sdpa_fits_dsp_limits(_node(None, batch=batch))
        assert not _sdpa_fits_dsp_limits(_node((_QO_LEN, _MAX_KV_LEN), batch=batch))
    assert _sdpa_fits_dsp_limits(_node(None))
    assert _sdpa_fits_dsp_limits(_node((_QO_LEN, _MAX_KV_LEN)))
    # And through the exporter, so the whole node is what the gate refused.
    for masked in (False, True):
        program = _lower(
            masked, _tensors(batch=2, mask=torch.zeros(_QO_LEN, _MAX_KV_LEN))
        )
        assert program[1] == [], f"a batched query (masked={masked}) reached a delegate"
        assert not _sdpa_fits_dsp_limits(program[0])


# ---------------------------------------------------------------------------
# The command, once one exists.
# ---------------------------------------------------------------------------


def test_the_command_binds_the_mask_and_the_stride_together():
    """The operand, the stride and the reservation, read out of the blob.

    A mask without a stride is the bug this replaced: the operand would be in the
    blob and every number the kernel attended would be the one it computes as if
    the operand were not there. The two are asserted together, and the workspace
    is asserted against the unmasked one, because the fp32 copy the kernel makes
    of the mask goes inside that buffer at an offset the kernel computes and does
    not check (`htp_ops_flash_attn` takes no workspace size at all).
    """
    _call, unmasked_blobs = _lower(False, _tensors())
    assert len(unmasked_blobs) == 1, "the control has to delegate for this to mean"
    _blob, control = _attention(unmasked_blobs)
    assert control.params[7] == -1, "no mask is a stride of -1, not a stride of 0"
    assert control.inputs[3].size == 0, "and an absent operand"
    unmasked_workspace = control.outputs[1].size
    assert unmasked_workspace == _attention_workspace_bytes(
        _QO_LEN, _QO_LEN + _MAX_KV_LEN, _N_HEADS
    )

    mask = torch.full((_QO_LEN, _MAX_KV_LEN), float("-inf"), dtype=torch.float32)
    mask[:, : _MAX_KV_LEN // 2] = 0.0
    _call, masked_blobs = _lower(True, _tensors(mask=mask))
    assert len(masked_blobs) == 1, "the masked graph has to delegate too"
    _blob, command = _attention(masked_blobs)

    assert command.params[7] == _MAX_KV_LEN, "the stride is the mask's last extent"
    operand = command.inputs[3]
    assert operand.space is blob_interpreter.B.TensorSpace.INPUT
    assert operand.size == _QO_LEN * _MAX_KV_LEN * 2, "fp16 rows, one per query row"
    assert command.outputs[1].size == unmasked_workspace + _attention_mask_bytes(
        _QO_LEN, _MAX_KV_LEN
    )
    assert command.outputs[1].size - unmasked_workspace == _QO_LEN * _MAX_KV_LEN * 4
    # The other three operands are untouched, so nothing else about the command
    # moved when the mask appeared.
    assert (
        control.params[:7] + control.params[8:]
        == command.params[:7] + command.params[8:]
    )


def test_a_constant_mask_is_the_same_operand_in_the_weights():
    """A folded mask is the same command with slot three in another section.

    The mask is a buffer, so it is a weight: the operand is not an input the
    caller passes, and the emitter's constant path is what binds it. The numbers
    are checked too, because the constant is narrowed by `tensor.to` rather than
    by the runtime's own cast, and a mask that arrived one ULP off would still
    have to give the clamp's answer.
    """
    mask = _causal_mask()
    query, key, value, _unused = _tensors()
    _standin()
    blobs = _delegates(export(_ConstMaskSdpa(mask).eval(), (query, key, value)))
    assert len(blobs) == 1, "a constant mask has to delegate too"
    blob, command = _attention(blobs)

    assert command.params[7] == _MAX_KV_LEN
    operand = command.inputs[3]
    assert operand.space is blob_interpreter.B.TensorSpace.WEIGHTS
    assert operand.size == _QO_LEN * _MAX_KV_LEN * 2

    got = np.frombuffer(
        execute(blob, [query.numpy(), key.numpy(), value.numpy()])[0], dtype=np.float16
    ).reshape(_BATCH, _QO_LEN, _N_HEADS, _HEAD_DIM)
    assert np.allclose(
        got, _reference(query, key, value, mask, command, clamp=False), atol=1e-2
    )
    plain = _tensors()
    clamped = _run(_attention(_lower(False, plain)[1])[0], plain).astype(np.float32)
    assert np.abs(got.astype(np.float32) - clamped).max() <= 1e-2


def test_a_geometry_the_kernel_drops_the_mask_on_stays_off_the_delegate():
    """The refusals, at the door that decides delegation.

    The whole attention falls back to the portable kernels and computes the
    masked thing, which is the outcome the mask handling is here to produce;
    being delegated and quietly computing the unmasked thing is the one that is
    not allowed to happen.
    """
    one_row = torch.zeros(1, _MAX_KV_LEN, dtype=torch.float32)
    # One query row without a mask is still emittable, and stays emittable: with
    # a negative stride the shortcut's answer -- the single value row -- is the
    # causal answer, so there is nothing for it to get wrong.
    _call, blobs = _lower(False, _tensors(qo_len=1))
    assert len(blobs) == 1, "one row with no mask has to stay emittable"
    assert _attention(blobs)[1].params[7] == -1
    for refused in (
        _lower(True, _tensors(qo_len=1, mask=one_row)),
        _lower(False, _tensors(batch=2, mask=one_row)),
        _lower(
            True,
            _tensors(mask=torch.zeros(_QO_LEN, _MAX_KV_LEN)),
            dynamic_q=True,
        ),
    ):
        assert refused[1] == [], "this graph had to stay on the portable kernels"
        assert not _sdpa_fits_dsp_limits(refused[0])


def test_the_emitter_refuses_a_mask_the_gate_never_saw():
    """The second door, for a graph that reaches preprocess without the gate."""

    class _Refusing:
        def __getattr__(self, name):
            raise AssertionError(f"the emitter reached ctx.{name} instead of refusing")

    for mask in ((1, _MAX_KV_LEN), (_QO_LEN + 1, _MAX_KV_LEN), (_QO_LEN, 0)):
        with pytest.raises(RuntimeError, match="attention mask"):
            _emit_sdpa(_node(mask), _Refusing())


# ---------------------------------------------------------------------------
# The numbers. This is the part that says whether the mask was applied.
# ---------------------------------------------------------------------------


def _causal_mask():
    """The clamp the kernel generates, written down as scores.

    Zero up to the diagonal and negative infinity past it, which is what
    `min(seq_current + q + 1, N)` produces up to the softmax.
    """
    return torch.triu(
        torch.full((_QO_LEN, _MAX_KV_LEN), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )


def test_the_mask_is_what_the_masked_command_computes():
    """The numbers, against a reference written from the shapes.

    The interpreter walks the kernel's addressing; the reference is the same
    arithmetic written from the command's own geometry. Two readings, not one.
    """
    mask = _causal_mask()
    tensors = _tensors(mask=mask)
    _blob, command = _attention(_lower(True, tensors)[1])
    got = _run(_blob, tensors)
    assert np.allclose(
        got,
        _reference(tensors[0], tensors[1], tensors[2], mask, command, clamp=False),
        atol=1e-2,
    ), "the masked command does not compute the masked thing"


def test_a_causal_mask_reproduces_the_unmasked_answer_and_an_open_one_does_not():
    """The control and the teeth, on one geometry and one emitter.

    A mask that says exactly what the kernel's own clamp says has to give the
    unmasked answer: that is the control, and it is what a mask handed to a
    kernel that ignored it would also give. An all-visible mask has to give a
    different answer, because with a positive stride the kernel attends all N
    keys and the mask is the only thing that says otherwise -- so a run that
    dropped the operand would fail this. The tolerance is the fp16 the whole
    attention accumulates in, the same 1e-2 the fp16 matmul comparisons use.
    """
    causal = _causal_mask()
    plain = _tensors()
    clamped_blob, _control = _attention(_lower(False, plain)[1])
    masked_blob, _command = _attention(_lower(True, _tensors(mask=causal))[1])

    clamped = _run(clamped_blob, plain).astype(np.float32)
    written = _run(masked_blob, _tensors(mask=causal)).astype(np.float32)
    open_mask = torch.zeros(_QO_LEN, _MAX_KV_LEN, dtype=torch.float32)
    everything = _run(masked_blob, _tensors(mask=open_mask)).astype(np.float32)

    moved = np.abs(written - clamped).max()
    assert moved <= 1e-2, f"a causal mask moved the answer by {moved}"
    visible = np.abs(everything - clamped).max()
    assert visible > 0.1, (
        f"an all-visible mask moved the answer by {visible}, so this geometry "
        "cannot tell a mask that arrived from one that was dropped"
    )


def test_a_mask_that_hides_a_key_the_clamp_kept_moves_the_answer():
    """The teeth, aimed at the specific failure this replaced.

    Hiding every key but the first is a mask that changes which keys exist, not
    one that restates the clamp, so an emitter that bound the operand and sent
    the stride that means "no mask" would produce the unmasked answer here and
    fail. The reference check on the same run is what says the difference is the
    mask rather than a wrong command.
    """
    hidden = torch.zeros(_QO_LEN, _MAX_KV_LEN, dtype=torch.float32)
    hidden[:, 1:] = float("-inf")
    tensors = _tensors(mask=hidden)
    _blob, command = _attention(_lower(True, tensors)[1])
    got = _run(_blob, tensors)
    assert np.allclose(
        got,
        _reference(tensors[0], tensors[1], tensors[2], hidden, command, clamp=False),
        atol=1e-2,
    )
    unmasked = _run(_attention(_lower(False, _tensors())[1])[0], _tensors())
    moved = np.abs(got.astype(np.float32) - unmasked.astype(np.float32)).max()
    assert moved > 0.1, f"hiding every key but the first moved the answer by {moved}"
