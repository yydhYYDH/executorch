# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real blobs, executed on the DSP.

The host interpreter runs the same bytes in Python and the simulator runs them
through the real kernels, so the comparison spans the emitters, the blob layout
and the kernels at once. Nothing between the three is hand-written, which is
what makes this different from test_hexagon_sim.py, whose descriptors are typed
out by hand.

These are the same subgraphs test_blob_interpreter.py checks: the concatenate,
slice and transpose behind one blit chain, the fused norm, the cache advance
with its patch and scale, a matmul and a broadcast.
"""

import os
import pathlib
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parent))

import blob_interpreter  # noqa: E402
import hexagon_sim  # noqa: E402
from blob_interpreter import Arena, execute, read_blob, UnsupportedOp  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import sdpa_targets  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import Dim, export  # noqa: E402


def _program(graph_module):
    """Wrap a hand-built graph module the way preprocess expects a program.

    These graphs own no parameters or buffers, but the backend reads the
    signature to tell an owned constant from a caller-passed input, so the
    wrapper carries an empty one.
    """
    signature = SimpleNamespace(
        inputs_to_buffers={},
        buffers_to_mutate={},
        inputs_to_parameters={},
        inputs_to_lifted_tensor_constants={},
    )
    return SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        range_constraints={},
    )


_RUNNER = pathlib.Path(__file__).resolve().parent / "sim/blob_runner.cpp"

#: softmax_ops.cc includes this, and the backend's CMake generates it with QAIC
#: into its build tree. Nothing it declares is used by that translation unit:
#: the only identifiers the two share are AEEResult and the AEE_ result codes,
#: which come from AEEStdErr.h. An empty file of the same name keeps this runner
#: free of a build-tree dependency, and if the include ever becomes load bearing
#: the compile fails here rather than quietly dropping a declaration.
_HT_P_OP_SHIM = "/* Empty on purpose; see test_blob_on_sim.py. */\n"
_SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "serialization"

_SOURCES = [
    "blit_ops.cc",
    "loop_ops.cc",
    "matmul_ops.cc",
    "eltwise_ops.cc",
    "unary_ops.cc",
    "softmax_ops.cc",
    "layer_norm_ops.cc",
    "worker_pool.cc",
    "vtcm_mgr.cc",
    "hmx_mgr.cc",
    "pwl.cc",
    "power.cc",
    "ops/matmul_q4fp16.c",
    "ops/matmul_q4fp16_mle32.c",
    "attention_entry.cc",
    "attention_sync_setup.cc",
    "attention_sync_process.cc",
    "attention_ops.cc",
    "attention_push_kv.cc",
    "attention_hmx.cc",
    "attention_hmx_queue.cc",
    "hmx_queue.cc",
    # The commands this branch added. Each is the translation unit the skel
    # builds for that op and nothing else: the pool wrapper and its HVX walk,
    # the row gather's fp16 read path, and the two GEMV entries with their
    # packers' read side.
    "pool_ops.cc",
    "ops/pool_fp16.c",
    "shared_gather_ops.cc",
    "matmul_q4block_ops.cc",
    "ops/matmul_q4block_gemv_i8.c",
    "ops/matmul_q4block_fp16_mle32.c",
    "ops/matmul_w8a16_gemv_i8.c",
    # The kernels above call htp_probe_stage, which the device build defines in
    # execute_command.cc -- a translation unit with the whole op table behind it,
    # so the probe is linked on its own here. It is a trace hook and every kernel
    # below runs the same either way.
    str(pathlib.Path(__file__).resolve().parent / "sim/htp_probe_stage_shim.cc"),
]

#: The fused norm reduces in a different order than numpy does, so it is compared
#: with a tolerance rather than by bits.
_NORM_TOLERANCE = 2e-2

#: Measured, not assumed. htp_ops_softmax exponentiates with hvx_my_exp2_vhf, a
#: degree-six polynomial in fp16 evaluated with qfloat multiplies, and divides by
#: an approximate reciprocal refined once. On this input its answer is off by
#: 0.063 while still summing to one along the reduced axis -- the shape of the
#: distribution moves, not its mass. Transcribing that polynomial in ordinary
#: fp16 lands further from the kernel than the exact exponential does, so the
#: rounding mode is load bearing and the exact value is not recoverable on the
#: host. What this tolerance still catches is a wrong axis, a wrong decomposition
#: or a wrong offset, none of which move an output by 0.08.
_SOFTMAX_TOLERANCE = 0.08

#: htp_ops_reduction sums in fp32 like numpy does, so the two agree to the last
#: bit of the fp16 store except for one denormal: the kernel's scalar path
#: rounds the final add differently. Measured at 1.8e-07.
_MEAN_TOLERANCE = 1e-6

# The kernel's silu is a piecewise-linear approximation (htp_ops_silu_pwl_fp16_vec)
# while the host model evaluates the exponential, so this one is a tolerance and
# not a bit comparison. Measured, not guessed: over gate values within +-24 and
# up values within +-12 the two agree to below 1e-9, which for fp16 means the
# same bits -- two different fp16 values are at least an ulp apart. The bound is
# kept above that because the approximation is what it is; how it behaves outside
# this range is unverified.
_SILU_TOLERANCE = 1e-6

#: The pool's average divides by `1/count` narrowed to fp16 before the multiply
#: (pool_fp16.c:74-86), so the divisor is not exact and neither is the answer.
#: The worst the two cases showed on the simulator is 1.95e-3, and the ceiling
#: sits just above it rather than at a rounder number that would hide a wrong
#: window: a kernel reading the wrong neighbourhood is off by far more.
_POOL_TOLERANCE = 5e-3

#: The vision attention exponentiates in fp32 with the same approximate exp2 the
#: softmax kernel uses, but its result is a convex combination of at most
#: `tokens` values, so the approximation's error reaches the output damped rather
#: than at full size. On the simulator the DSP is bit-exact on both cases and the
#: host model is within 3.9e-6; the ceiling is set for the host leg, whose fp32
#: accumulation order is its own. The two layouts this case has to separate are
#: 1.2 apart, so the margin is four orders of magnitude.
_VISION_TOLERANCE = 1e-4


class _Shapes(torch.nn.Module):
    """Concatenate, slice and transpose -- three blits back to back."""

    def __init__(self, cut: int) -> None:
        super().__init__()
        self.cut = cut

    def forward(self, a, b):
        joined = torch.cat([a, b], dim=-1)
        return joined[:, :, :, : self.cut].permute(0, 1, 3, 2)


class _Mm(torch.nn.Module):
    def forward(self, a, b):
        return torch.mm(a, b)


class _Bmm(torch.nn.Module):
    """A batch of tiles, so the descriptor's loop count is the batch."""

    def forward(self, a, b):
        return torch.bmm(a, b)


class _Addmm(torch.nn.Module):
    """A product and the bias the kernel has no operand for: two commands."""

    def forward(self, x, weight, bias):
        return torch.addmm(bias, x, weight)


class _Scale(torch.nn.Module):
    """A broadcast, so one operand is read with a zero stride."""

    def forward(self, x, bias):
        return x * bias


class _SoftmaxAt(torch.nn.Module):
    """A softmax along an axis that is neither first nor last.

    That is the only shape of it that gives the kernel a real inside extent; a
    last-axis softmax has inside 1 and takes a different path.
    """

    def forward(self, x):
        return torch.softmax(x, 1)


class _MeanAt(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=-1)


class _Narrow(torch.nn.Module):
    """Pick one entry along an axis that has more than one.

    That is the select the alias path cannot take: the result holds fewer bytes
    than the operand, so the kernel writes a run into a buffer of its own.
    """

    def forward(self, x):
        return x[:, 1, :]


def _cast_chain_graph(shape):
    """A fp32 round trip the arena should swallow whole.

    norm.py writes its casts the same way, and every kernel behind them is fp16,
    so once the value is in a slot the two casts have nothing left to do. Built
    by hand because the exporter spells the same cast `_to_dim_order_copy`,
    which carries a layout change this has no business testing.
    """
    cast = exir_ops.edge.aten._to_copy.default
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, dtype=torch.float16)
    up = graph.call_function(cast, args=(x,), kwargs={"dtype": torch.float32})
    up.meta["val"] = torch.empty(shape, dtype=torch.float32)
    down = graph.call_function(cast, args=(up,), kwargs={"dtype": torch.float16})
    down.meta["val"] = torch.empty(shape, dtype=torch.float16)
    out = graph.call_function(exir_ops.edge.aten.neg.default, args=(down,))
    out.meta["val"] = torch.empty(shape, dtype=torch.float16)
    graph.output(out)
    return _program(torch.fx.GraphModule(torch.nn.Module(), graph))


def _dynamic_slice_graph(source_shape, rows):
    """A slice whose start row the caller only knows when it runs.

    The RoPE frequency tables are cut this way: a fixed number of rows out of a
    table sized for the longest prompt, beginning wherever the sequence is. The
    end argument is the same node as the start because nothing here reads it --
    the extent comes from the result, and the only moving part is the offset.
    """
    graph = torch.fx.Graph()
    table = graph.placeholder("table")
    table.meta["val"] = torch.empty(source_shape, dtype=torch.float16)
    start = graph.placeholder("start")
    start.meta["val"] = torch.empty((), dtype=torch.int64)
    cut = graph.call_function(
        exir_ops.edge.aten.slice_copy.Tensor, args=(table, 0, start, start)
    )
    cut.meta["val"] = torch.empty((rows, source_shape[1]), dtype=torch.float16)
    out = graph.call_function(exir_ops.edge.aten.neg.default, args=(cut,))
    out.meta["val"] = torch.empty((rows, source_shape[1]), dtype=torch.float16)
    graph.output(out)
    return _program(torch.fx.GraphModule(torch.nn.Module(), graph))


def _small(shape):
    """Small integers, so every product and sum below is exact in fp16."""
    count = 1
    for extent in shape:
        count *= extent
    return (torch.arange(count) * 5 % 7 - 3).float().reshape(shape).half()


def _norm_weight(size):
    """A non-trivial gamma, so a norm that skipped the weight would show up."""
    return ((torch.arange(size) % 5) * 0.25 + 0.5).half()


def _norm_graph(shape, eps):
    from executorch.backends.hexagon.rms_norm import RMS_NORM

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, dtype=torch.float16)
    gamma = graph.get_attr("weight")
    gamma.meta["val"] = torch.empty(shape[-1], dtype=torch.float16)
    fused = graph.call_function(RMS_NORM, args=(x, gamma, eps))
    fused.meta["val"] = torch.empty(shape, dtype=torch.float16)
    graph.output(fused)
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(_norm_weight(shape[-1]), requires_grad=False)
    return _program(torch.fx.GraphModule(root, graph))


def _cache_graph(cache_shape, value_shape):
    from executorch.backends.hexagon.kv_cache import UPDATE_CACHE

    graph = torch.fx.Graph()
    cache = graph.placeholder("cache")
    cache.meta["val"] = torch.empty(cache_shape, dtype=torch.float16)
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty(value_shape, dtype=torch.float16)
    position = graph.placeholder("position")
    position.meta["val"] = torch.empty(1, dtype=torch.int64)
    fused = graph.call_function(UPDATE_CACHE, args=(cache, value, position))
    fused.meta["val"] = torch.empty(cache_shape, dtype=torch.float16)
    graph.output(fused)
    return _program(torch.fx.GraphModule(torch.nn.Module(), graph))


def _mul_silu_graph(shape):
    from executorch.backends.hexagon.mul_silu import MUL_SILU

    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    a.meta["val"] = torch.empty(shape, dtype=torch.float16)
    b = graph.placeholder("b")
    b.meta["val"] = torch.empty(shape, dtype=torch.float16)
    fused = graph.call_function(MUL_SILU, args=(a, b))
    fused.meta["val"] = torch.empty(shape, dtype=torch.float16)
    graph.output(fused)
    return _program(torch.fx.GraphModule(torch.nn.Module(), graph))


def _attention_graph(batch, qo_len, n_heads, n_kv_heads, max_kv_len, head_dim):
    """Attention whose key and value are the cache the graph already holds.

    That is what an export produces: the graph carries its own update_cache, so
    by the time attention runs the new rows are already in the cache and the
    kernel's push copies cache rows onto themselves. A start_pos of zero is the
    prefill case, where the query length is the whole sequence.
    """
    graph = torch.fx.Graph()
    nodes = {}
    for name, shape in (
        ("query", (batch, qo_len, n_heads, head_dim)),
        ("key", (batch, max_kv_len, n_kv_heads, head_dim)),
        ("value", (batch, max_kv_len, n_kv_heads, head_dim)),
    ):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=torch.float16)
        nodes[name] = node
    out = graph.call_function(
        exir_ops.edge.llama.custom_sdpa.default,
        args=(nodes["query"], nodes["key"], nodes["value"], 0, None, 0.0, True, None),
    )
    out.meta["val"] = torch.empty(
        (batch, qo_len, n_heads, head_dim), dtype=torch.float16
    )
    graph.output(out)
    return _program(torch.fx.GraphModule(torch.nn.Module(), graph))


def _attention_reference(query, key, value):
    """The contract, written in torch ops so it shares no code with the model.

    Row q of the query reads the cache up to and including position q, which is
    what causal attention over a prefill means. Head h reads kv head
    h // (n_heads / n_kv_heads).
    """
    group = query.shape[2] // key.shape[2]
    scale = query.shape[-1] ** -0.5
    out = torch.zeros(query.shape, dtype=torch.float32)
    for row in range(query.shape[1]):
        valid = row + 1
        for head in range(query.shape[2]):
            kv = head // group
            scores = query[0, row, head].float() @ key[0, :valid, kv].float().T
            weights = torch.softmax(scores * scale, dim=-1)
            out[0, row, head] = weights @ value[0, :valid, kv].float()
    return out


def _case(
    tag,
    program,
    args,
    expected,
    kind="bits",
    tolerance=None,
    length=0,
    blob=None,
    mutate=None,
    dynamic_shapes=None,
):
    """One fixture: a blob, the bytes its inputs are handed as, and its answer.

    `length` is the run-time sequence length a dynamic blob's caller would hand
    it. The simulator applies the blob's own trailer patches with it, which is
    what the runtime does before it issues anything; zero says the blob carries
    no trailer and there is nothing to apply.

    `mutate` rewrites the blob, which is how a case says what the DSP must *not*
    answer: a control whose comparison would still pass on the wrong bytes says
    nothing about the real ones.
    """
    if blob is None:
        if dynamic_shapes is None:
            dynamic_shapes = {}
        if isinstance(program, torch.nn.Module):
            program = to_edge(
                export(program, tuple(args), dynamic_shapes=dynamic_shapes or None)
            ).exported_program()
        blob = HexagonBackend.preprocess(program, []).processed_bytes
    blob = bytes(blob)
    if mutate is not None:
        blob = bytes(mutate(blob))
    header, commands = read_blob(blob)
    # execute() checks each input's byte size against the slot the emitters
    # claimed, so a blob whose inputs are in another order fails here. The length
    # is stated because a dynamic case hands over the whole bound-sized slot
    # rather than the run-time tensor; it is the same number the simulator is
    # given, and the trailer's own longest length is asserted against both.
    if kind == "refused":
        # The host model checks this one too, and refuses it in its own words
        # rather than answering wrongly: the case is about the refusal, and the
        # disagreement the DSP shows is with torch, not with the host model.
        with pytest.raises(UnsupportedOp):
            execute(blob, [t.numpy() for t in args], length=length or None)
        host = None
    else:
        host = execute(blob, [t.numpy() for t in args], length=length or None)
    arena = Arena(header, blob, "fixture")
    return SimpleNamespace(
        tag=tag,
        blob=blob,
        commands=commands,
        inputs=b"".join(t.numpy().tobytes() for t in args),
        host=host,
        expected=expected,
        kind=kind,
        tolerance=tolerance,
        arena_bytes=len(arena.bytes),
        length=length,
        args=tuple(args),
    )


def _tagged(cases, tag):
    """One case by tag. Indexing the list would break when a case is added."""
    return next(case for case in cases if case.tag == tag)


# --- The commands this branch put on the DSP for the first time ---------------
#
# Everything above runs a command the backend already had. Below is each of the
# six the branch added, chosen so that the thing the branch assumed is the thing
# the run decides:
#
#   * the row gather's 32x32 tile order, on an `ic`/`oc` that is neither a
#     multiple of 32 nor square -- the shape where a wrong reading of the inner
#     order cannot coincidentally agree;
#   * a reduction whose span holds the run-time length, run at a length shorter
#     than the one it was exported for, with the arena past that length carrying
#     values no correct answer can contain;
#   * both quantized GEMV entries, whose packers' byte orders were written from
#     the kernels' read paths and never run;
#   * the clamp family's NaN path, which the branch itself calls its largest
#     on-device risk;
#   * the pool's packed window walk, which the host model deliberately does not
#     model as the fast path the emitter asserts;
#   * the vision tower's attention layout, which is positional and unchecked.

_SHARED_GATHER = 23
_POOL2D = 1
_UNARY = 4
_REDUCTION = 29
_VISION_ATTENTION = 43
_Q4A16_GEMV = 41
_W8A16_GEMV = 45


def _table_ints(oc, ic):
    """A table whose every element is a distinct small integer.

    Half-integers would do, but integers keep every sum and product in the
    references below exact, so a disagreement is a disagreement about the
    arrangement and not about rounding.
    """
    rows = torch.arange(oc).reshape(oc, 1)
    columns = torch.arange(ic).reshape(1, ic)
    return ((rows * 3 + columns * 5) % 37 - 18).half()


def _fake(tensor):
    """A metadata value that is a tensor and still not one the export step can read.

    `constant_value` refuses a FakeTensor by name, which is how the emitters tell
    a weight they can see from an operand only the caller has; a real tensor in a
    placeholder's metadata would look like a weight. A hand-built graph has to
    make that distinction itself, and this is how it does.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    return FakeTensorMode().from_tensor(tensor)


def _gather_graph(oc, ic, rows):
    """`embedding` as one SHARED_GATHER.

    The table is an attribute, which is the only kind the emitter can read -- it
    rearranges the bytes into the order the kernel walks, so a table that only
    exists at run time has nothing to rearrange. The indices are the caller's own
    int32 placeholder: the kernel reads `const int32_t[]`, and an int64 tensor
    would be read as alternating low and high words.
    """
    graph = torch.fx.Graph()
    index = graph.placeholder("index")
    index.meta["val"] = _fake(torch.empty((rows,), dtype=torch.int32))
    weight = graph.get_attr("weight")
    weight.meta["val"] = torch.empty((oc, ic), dtype=torch.float16)
    out = graph.call_function(
        exir_ops.edge.aten.embedding.default, args=(weight, index)
    )
    out.meta["val"] = torch.empty((rows, ic), dtype=torch.float16)
    graph.output(out)
    root = torch.nn.Module()
    root.weight = torch.nn.Parameter(_table_ints(oc, ic), requires_grad=False)
    return _program(torch.fx.GraphModule(root, graph))


def _gather_reference(table, indices):
    """The rows the command reads, with an out-of-range index clearing its row.

    `shared_gather_ops.cc:292-295` clears a row whose index is outside
    `[0, oc)`; `torch.embedding` raises instead. The divergence is the kernel's,
    so this reference takes it and a test below pins the other side of it.
    """
    out = torch.zeros((len(indices), table.shape[1]), dtype=torch.float16)
    for row, index in enumerate(indices):
        if 0 <= int(index) < table.shape[0]:
            out[row] = table[int(index)]
    return out


def _weights_base(blob):
    """Where the file's weights section starts, from the header and the op count."""
    header, _ = read_blob(blob)
    return blob_interpreter.B.HEADER_SIZE + header.n_ops * blob_interpreter.B.OP_SIZE


def _row_major_tiles(table):
    """The same table with each 32x32 tile written plain row-major.

    This is the reading of `shared_gather_ops.cc:296-311` this batch considered
    first and rejected: the tile's *base* address is right either way, so the
    blob stays well formed and every table element is still present once. Only
    the order inside a tile changes, which is exactly the claim under test.
    """
    oc, ic = table.shape
    rows, columns = -(-oc // 32), -(-ic // 32)
    padded = np.zeros((rows * 32, columns * 32), dtype=np.float16)
    padded[:oc, :ic] = table.numpy()
    tiles = padded.reshape(rows, 32, columns, 32)
    return np.ascontiguousarray(tiles).tobytes()


def _with_row_major_table(table):
    """A rewrite of a gather blob whose stored table is in the order above."""

    def mutate(blob):
        header, commands = read_blob(blob)
        gather = next(command for command in commands if command.type == _SHARED_GATHER)
        ref = gather.inputs[1]
        at = _weights_base(blob) + ref.offset
        packed = _row_major_tiles(table)
        assert ref.size == len(
            packed
        ), f"{ref.size} bytes of table, {len(packed)} packed"
        return blob[:at] + packed + blob[at + ref.size :]

    return mutate


class _Pool(torch.nn.Module):
    def __init__(self, kind, kernel, stride, padding, count_include_pad=True):
        super().__init__()
        self.kind = kind
        if kind == "max":
            self.pool = torch.nn.MaxPool2d(kernel, stride, padding)
        else:
            self.pool = torch.nn.AvgPool2d(
                kernel, stride, padding, count_include_pad=count_include_pad
            )

    def forward(self, x):
        return self.pool(x)


class _Clamp(torch.nn.Module):
    """One of the clamp family: relu, hardtanh and relu6 all lower to the one
    command, and the NaN path is the branch's own largest risk."""

    def __init__(self, kind, lower=None, upper=None):
        super().__init__()
        self.kind = kind
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        if self.kind == "relu":
            return torch.relu(x)
        if self.kind == "relu6":
            return torch.nn.functional.relu6(x)
        return torch.clamp(x, self.lower, self.upper)


def _vision_graph(batch, tokens, heads, head_dim, scale):
    """`et_hexagon.vision_attention` as one VISION_ATTENTION_FP16.

    The operands go in as the fused op takes them -- `[batch, tokens, heads,
    headDim]`, heads inside a token's row -- which is the layout the kernel
    strides by `heads * headDim` (attention_entry.cc:36,41,57). The workspace is
    the command's second output, sized for the longest run the emitter saw.
    """
    from executorch.backends.hexagon.vision_attention import VISION_ATTENTION

    shape = (batch, tokens, heads, head_dim)
    graph = torch.fx.Graph()
    operands = []
    for name in ("query", "key", "value"):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=torch.float16)
        operands.append(node)
    out = graph.call_function(VISION_ATTENTION, args=(*operands, scale))
    out.meta["val"] = torch.empty(shape, dtype=torch.float16)
    graph.output(out)
    return _program(torch.fx.GraphModule(torch.nn.Module(), graph))


def _vision_reference(query, key, value, scale):
    """The contract in torch ops, sharing no code with the kernel.

    The kernel's inner product is over `headDim` in fp32 and its weights come
    from an fp32 exponential, so the reference stays in fp32 and the comparison
    is a tolerance rather than bits.
    """
    q = query.float().transpose(1, 2)
    k = key.float().transpose(1, 2)
    v = value.float().transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * scale
    return (scores.softmax(dim=-1) @ v).transpose(1, 2)


def _vision_reference_swapped(query, key, value, scale):
    """The same attention under the *other* reading of the layout.

    `[batch, heads, tokens, headDim]`: the head axis where the kernel puts the
    token axis, and the token axis where it puts the heads. Nothing about the
    tensors changes -- this is what the kernel would have computed if its strides
    were the ones a matmul wants.
    """
    q, k, v = (tensor.float() for tensor in (query, key, value))
    scores = (q @ k.transpose(-1, -2)) * scale
    return scores.softmax(dim=-1) @ v


def _reduction_graph(dim):
    """`sum(x, dim)` over a sequence whose length the caller picks."""

    class _Sum(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, dim=dim)

    return _Sum()


def _lowered_blob(model, args, dynamic_shapes):
    """The blob the real pipeline produces, and the operands its delegate takes."""
    from executorch.backends.hexagon.partition.hexagon_partitioner import (
        HexagonPartitioner,
    )
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

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
    assert len(calls) == 1, f"the graph did not reach one delegate: {calls}"
    submodule = program.graph_module.get_submodule(calls[0].args[0].target)
    return bytes(submodule._processed_bytes)


def _strip_trailer(blob):
    """A blob whose dynamic trailer no longer parses: the patches are gone.

    This is what the pre-`efe2a64` emitter produced, byte for byte: the command
    still describes the exported bound and nothing recomputes it. The simulator
    is handed the same fixture, so the only difference between this case and the
    one above it is whether the run-time length reaches the param.
    """
    at = blob.find(blob_interpreter.B.DYNAMIC_TRAILER_MAGIC.to_bytes(4, "little"))
    assert at >= 0, "a dynamic blob carries a trailer"
    return blob[:at] + b"\0\0\0\0" + blob[at + 4 :]


def _param_at(blob, op_index, param_index):
    """Where one command's param slot sits, from the struct's own layout.

    The command opens with four int32s -- type, input count, output count, param
    count -- and the params follow them, before the tensor refs. The offset does
    not depend on how many params the command carries.
    """
    return (
        blob_interpreter.B.HEADER_SIZE
        + op_index * blob_interpreter.B.OP_SIZE
        + 4 * 4
        + 4 * param_index
    )


def _set_param(param_index, value, op_index=0):
    """A mutation that overwrites one 4-byte param slot."""

    def mutate(blob):
        data = bytearray(blob)
        at = _param_at(data, op_index, param_index)
        data[at : at + 4] = value
        return data

    return mutate


def _swap_params(first, second, op_index=0):
    """A mutation that exchanges two param slots and changes nothing else."""

    def mutate(blob):
        data = bytearray(blob)
        at, other = _param_at(data, op_index, first), _param_at(data, op_index, second)
        data[at : at + 4], data[other : other + 4] = (
            data[other : other + 4],
            data[at : at + 4],
        )
        return data

    return mutate


def _cases():
    from executorch.backends.hexagon.kv_cache import _update_cache

    shapes_model = _Shapes(2)
    a, b = _small((1, 2, 3, 4)), _small((1, 2, 3, 4))
    shapes = _case("A", shapes_model, (a, b), _bits(shapes_model(a, b)))

    left, right = _small((4, 8)), _small((8, 3))
    mm = _case("B", _Mm(), (left, right), _bits(left.float() @ right.float()))

    ba, bb = _small((3, 4, 8)), _small((3, 8, 5))
    bmm = _case("Q", _Bmm(), (ba, bb), _bits(torch.bmm(ba.float(), bb.float())))

    # A 16-column bmm is deliberately absent here. It is the shape whose descriptor
    # reaches an HMX route, and on that shape the simulator's answer disagrees with
    # torch on 934 of 1024 elements with values no integer product can produce --
    # while 32, 48 and 64 columns are exact. That is a finding of its own, recorded
    # in hexagon_matmul_bench/_probe_sim_bmm_tail.py, and asserting either way here
    # would bake in a diagnosis that is not yet attributed to kernel or simulator.

    ax, aw, ab = _small((4, 8)), _small((8, 3)), _small((3,))
    addmm = _case(
        "R",
        _Addmm(),
        (ax, aw, ab),
        # The delegate rounds the product into an fp16 activation before the bias
        # is added, so the reference rounds there too.
        _bits((ax.float() @ aw.float()).half().float() + ab.float()),
    )

    x = _small((2, 3, 8))
    norm = _case(
        "C",
        _norm_graph(tuple(x.shape), 1e-5),
        (x,),
        _bits(
            torch.nn.functional.rms_norm(
                x, (x.shape[-1],), _norm_weight(x.shape[-1]), 1e-5
            ).reshape(-1)
        ),
        kind="close",
        tolerance=_NORM_TOLERANCE,
    )

    cache_shape = (1, 16, 2, 8)
    cache = (
        (torch.arange(int(np.prod(cache_shape))) % 11 - 5).half().reshape(cache_shape)
    )
    advances = []
    for tag, value_shape, position in (
        ("D", (1, 2, 2, 8), 3),
        ("F", (1, 2, 2, 8), 0),
        ("G", (1, 4, 2, 8), 12),
    ):
        value = _small(value_shape)
        advances.append(
            _case(
                tag,
                _cache_graph(cache_shape, value_shape),
                (cache, value, torch.tensor([position], dtype=torch.int64)),
                _bits(
                    _update_cache(cache, value, torch.tensor([position])).reshape(-1)
                ),
            )
        )

    scale = _case(
        "E",
        _Scale(),
        (x, _small((1, 1, 8))),
        _bits(x.float() * _small((1, 1, 8)).float()),
    )

    wide = _small((2, 6, 4))
    softmax = _case(
        "H",
        _SoftmaxAt(),
        (wide,),
        _bits(torch.softmax(wide.float(), 1)),
        kind="close",
        tolerance=_SOFTMAX_TOLERANCE,
    )
    mean = _case(
        "I",
        _MeanAt(),
        (wide,),
        _bits(torch.mean(wide.float(), dim=-1)),
        kind="close",
        tolerance=_MEAN_TOLERANCE,
    )

    pick = _small((1, 4, 8))
    narrowed = _case("J", _Narrow(), (pick,), _bits(_Narrow()(pick)))

    # Scaled by powers of two so the inputs stay exact in fp16 while reaching
    # far enough from zero to leave the flat part of the kernel's approximation.
    gate, up = _small((2, 16)) * 8, _small((2, 16)) * 4
    gated = _case(
        "K",
        _mul_silu_graph((2, 16)),
        (up, gate),
        _bits(up.float() * torch.nn.functional.silu(gate.float())),
        kind="close",
        tolerance=_SILU_TOLERANCE,
    )

    wide = _small((2, 16))
    absorbed = _case("L", _cast_chain_graph((2, 16)), (wide,), _bits(torch.neg(wide)))

    # The row the slice starts at is the only thing the runtime supplies, so a
    # patch that fails to scale it has to land somewhere the values can show it.
    # _small cannot do that job: it repeats every seven elements, and a row
    # number times a row length is a multiple of seven away from the row number
    # itself whenever the row holds eight, so the scaled and unscaled offsets
    # would read identical values and the check would pass either way.
    table, at = (
        torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8).half(),
        2,
    )
    pinned = _case(
        "N",
        _dynamic_slice_graph((256, 8), 3),
        (table, torch.tensor(at, dtype=torch.int64)),
        _bits(torch.neg(table[at : at + 3])),
    )

    # Attention over a cache the graph already holds: three queries reading a
    # 128-wide head out of an eight-row cache, with eight key heads shared by
    # sixteen query heads.
    #
    # This only builds once something has registered llama.custom_sdpa, which is
    # a C++ custom op: importing
    # executorch.extension.llm.custom_ops.custom_ops does it. The case is kept
    # behind that registration rather than importing it here because the
    # attention kernel starts a worker pool at run time and hexagon-sim cannot
    # start it -- qurt_cb_fwk_worker_init returns -4 and QuRT aborts -- so a blob
    # that reaches the op takes the whole run down with it and the cases that do
    # pass would stop being checked.
    batch, qo_len, n_heads, n_kv_heads, max_kv_len, head_dim = 1, 3, 16, 8, 8, 128
    query = _small((batch, qo_len, n_heads, head_dim))
    cache_k = _small((batch, max_kv_len, n_kv_heads, head_dim))
    cache_v = _small((batch, max_kv_len, n_kv_heads, head_dim))
    if sdpa_targets():
        attention = _case(
            "P",
            _attention_graph(batch, qo_len, n_heads, n_kv_heads, max_kv_len, head_dim),
            (query, cache_k, cache_v),
            _attention_reference(query, cache_k, cache_v).half(),
        )
    else:
        attention = None

    return [
        shapes,
        mm,
        bmm,
        addmm,
        norm,
        *advances,
        scale,
        softmax,
        mean,
        narrowed,
        gated,
        absorbed,
        pinned,
        *([attention] if attention is not None else []),
        *_branch_cases(),
    ]


#: The length the dynamic cases are exported for, the run-time length they are
#: actually run at, and the row width of the sequences below.
_UPPER = 16
_RUN = 3
_ROW = 8

#: What the arena holds past the run-time length. A command that still reads the
#: exported bound folds this in, and no correct answer can contain it.
_POISON = 256.0


def _sequence(width):
    """A bound-sized sequence whose first `_RUN` rows are live.

    The operand a dynamic graph is handed is the run-time tensor; the arena slot
    behind it is the exported bound, and what sits in that tail is whatever the
    last call left there. A fixture has to state it, and stating it as poison is
    what makes an unpatched command fail instead of passing on zeros.
    """
    live = ((torch.arange(_RUN * width).reshape(_RUN, width) * 3) % 29 - 14).half()
    bound = torch.full((1, _UPPER, width), _POISON, dtype=torch.float16)
    bound[:, :_RUN, :] = live
    return bound


def _branch_cases():
    """The commands this branch added, on the DSP."""
    cases = []

    # 1. The row gather. `oc` and `ic` are neither multiples of 32 nor equal, so
    #    every row of every tile is off by a different amount if the inner order
    #    is read as plain row-major. Two of the indices are outside the
    #    vocabulary, which is the one place the kernel and torch disagree.
    table = _table_ints(35, 33)
    indices = torch.tensor([3, 34, 35, 0, 17, -1], dtype=torch.int32)
    gather = _case(
        "S",
        _gather_graph(35, 33, len(indices)),
        (indices,),
        _bits(_gather_reference(table, indices)),
    )
    cases.append(gather)
    # The same blob with its table written row-major inside each tile: the
    # reading this batch rejected. Both legs read those bytes, so they must still
    # agree with each other and must no longer agree with torch.
    cases.append(
        _case(
            "U",
            _gather_graph(35, 33, len(indices)),
            (indices,),
            _bits(_gather_reference(table, indices)),
            kind="teeth",
            mutate=_with_row_major_table(table),
        )
    )

    # 2. A reduction whose span holds the run-time length. `sum(x, dim=1)` is the
    #    shape that answered *silently* wrong before efe2a64: the span straddles
    #    the live rows and the poison together, so the patch is the only thing
    #    standing between the DSP and a wrong number.
    sequence = _sequence(_ROW)
    blob = _lowered_blob(
        _reduction_graph(1),
        (torch.zeros(1, _RUN, _ROW, dtype=torch.float16),),
        {"x": {1: Dim("tokens", min=1, max=_UPPER)}},
    )
    expected = _bits(torch.sum(sequence[:, :_RUN, :], dim=1))
    cases.append(_case("V", None, (sequence,), expected, blob=blob, length=_RUN))
    # The control: the same command with the trailer gone, which is what the
    # emitter produced before the fix. Same fixture, same length, so the patch is
    # the only difference between the two cases.
    cases.append(
        _case(
            "W",
            None,
            (sequence,),
            expected,
            blob=blob,
            length=_RUN,
            kind="teeth",
            mutate=_strip_trailer,
        )
    )

    # 3. The clamp family's NaN path. A NaN has to come back as a NaN: if the
    #    kernel's restore is not the bit test the branch read, the bound comes
    #    back instead and nothing raises.
    torch.manual_seed(11)
    special = torch.tensor(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.0,
            0.0,
            2.5,
            -2.5,
            0.25,
        ],
        dtype=torch.float16,
    )
    filled = torch.cat([special, torch.randn(24, dtype=torch.float16)]).reshape(2, 16)
    # The kernel walks the tensor in one-HVX-vector chunks and finishes with a
    # scalar tail (unary_ops.cc:517-539); only the chunk restores a NaN by the
    # bit test, so the two sizes are two different code paths and both are here.
    # 128 elements is two whole vectors with the specials in the first.
    wide = torch.cat([special, torch.randn(120, dtype=torch.float16)]).reshape(4, 32)
    for tag, model, operand, kind in (
        ("X", _Clamp("relu"), wide, "nan"),
        ("Y", _Clamp("relu6"), wide, "nan"),
        ("Z", _Clamp("hardtanh", -1.0, 1.0), wide, "nan"),
        # 32 elements is the tail alone; the case is the branch's own claim run
        # where the kernel does not keep it, and it is reported rather than
        # asserted away. See test_the_clamp_tail_answers_a_nan_with_the_bound.
        ("AF", _Clamp("relu"), filled, "tail"),
    ):
        cases.append(
            _case(
                tag,
                model,
                (operand,),
                _bits(model(operand).half()),
                kind=kind,
            )
        )

    # 4. The pool: the packed window walk between two blits, with the geometry
    #    the emitter asserts the fast path wants. C is 64, the one channel count
    #    the support check admits.
    pooled = (
        torch.rand(1, 64, 8, 8, generator=torch.Generator().manual_seed(5)) * 8 - 4
    ).half()
    max_plain = _case(
        "AA", _Pool("max", 2, 2, 0), (pooled,), _bits(_Pool("max", 2, 2, 0)(pooled))
    )
    cases.append(max_plain)
    avg_padded = _Pool("avg", 3, 2, 1, count_include_pad=True)
    cases.append(
        _case(
            "AB",
            avg_padded,
            (pooled,),
            _bits(avg_padded(pooled)),
            kind="close",
            tolerance=_POOL_TOLERANCE,
        )
    )
    avg_valid = _Pool("avg", 2, 2, 0, count_include_pad=False)
    cases.append(
        _case(
            "AC",
            avg_valid,
            (pooled,),
            _bits(avg_valid(pooled)),
            kind="close",
            tolerance=_POOL_TOLERANCE,
        )
    )

    # 5. The vision tower's attention. The layout is the one assumption both
    #    readings of attention_entry.cc share, and the batch loop is the kernel's
    #    outermost, so it gets a case of its own. Random operands rather than the
    #    periodic ones above: on the periodic data the two layouts happen to
    #    answer the same thing, and a case that cannot tell them apart decides
    #    nothing.
    for tag, batch, tokens, heads in (("AD", 1, 4, 2), ("AE", 2, 3, 4)):
        head_dim, scale = 64, 0.125
        shape = (batch, tokens, heads, head_dim)
        generator = torch.Generator().manual_seed(7)
        operands = [
            (torch.rand(shape, generator=generator) * 2 - 1).half() for _ in range(3)
        ]
        program = _vision_graph(batch, tokens, heads, head_dim, scale)
        expected = _bits(_vision_reference(*operands, scale=scale).half())
        cases.append(
            _case(
                tag,
                program,
                tuple(operands),
                expected,
                kind="close",
                tolerance=_VISION_TOLERANCE,
            )
        )
        # The vision command's params are positional and nothing in the ABI
        # names them: the reasons to believe slot 1 is `tokens` and slot 4 the
        # scale are the emitter and this reading. Moving one slot at a time is
        # what a run can falsify -- the answer has to move, or the slot is not
        # read where the reading says it is.
        if tag == "AD":
            for control, mutate in (
                ("AG", _swap_params(1, 2)),
                # A scale of zero makes attention uniform over the keys, which
                # both models compute exactly: a scale of 2 would do the same job
                # but leaves the softmax smooth enough that the fp32 kernel and
                # the fp64 model part company in the last bits, and a control
                # asserts that they agree.
                ("AH", _set_param(4, struct.pack("<f", 0.0))),
            ):
                cases.append(
                    _case(
                        control,
                        program,
                        tuple(operands),
                        expected,
                        kind="teeth",
                        mutate=mutate,
                    )
                )
            # The kernel rejects a workspace shorter than one fp32 per token
            # (attention_entry.cc:26-29) and writes nothing, so this says which
            # operand that check is about.
            cases.append(
                _case(
                    "AI",
                    program,
                    tuple(operands),
                    expected,
                    kind="refused",
                    mutate=_set_param(6, struct.pack("<i", 1)),
                )
            )
    return cases


def _bits(values):
    return values.half().numpy().reshape(-1).astype(np.float16)


def _from_bits(bits):
    return np.array(bits, dtype=np.uint16).view(np.float16)


def _host_bits(raw):
    """The interpreter returns bytes; read them as the fp16 the blob declares."""
    return np.frombuffer(bytes(raw), dtype="<u2").tolist()


def _fixture_header(cases):
    lines = ["// Generated by test_blob_on_sim.py from blobs the emitter produced."]
    for index, case in enumerate(cases):
        for name, data in (("Blob", case.blob), ("InputData", case.inputs)):
            body = ",".join(str(byte) for byte in data)
            lines.append(f"static const unsigned char k{name}{index}[] = {{{body}}};")
    lines.append("static const BlobFixture kFixtures[] = {")
    for index, case in enumerate(cases):
        lines.append(
            f'  {{"{case.tag}", kBlob{index}, sizeof(kBlob{index}), '
            f"kInputData{index}, sizeof(kInputData{index}), {case.length}}},"
        )
    lines.append("};")
    lines.append(f"static const unsigned kFixtureCount = {len(cases)};")
    lines.append(
        f"static const unsigned kMaxArenaBytes = "
        f"{max(case.arena_bytes for case in cases)};"
    )
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def cases():
    """Built lazily, so a blob that cannot be produced fails a test rather than
    aborting collection with a traceback nobody can attribute."""
    return _cases()


@pytest.fixture(scope="module")
def simulated(cases):
    try:
        return hexagon_sim.run(
            _RUNNER,
            _SOURCES,
            headers={
                "blob_fixture.h": _fixture_header(cases),
                "htp_ops.h": _HT_P_OP_SHIM,
            },
            includes_more=[str(_SCHEMA)],
        )
    except hexagon_sim.Unavailable as error:
        pytest.skip(str(error))


def test_the_blobs_contain_the_ops_we_mean_to_run(cases):
    """A subgraph that lowered to something else would silently test nothing."""
    kinds = {case.tag: [command.type for command in case.commands] for case in cases}
    assert kinds["A"] == [3, 3, 3], "the concatenate, slice and transpose are not blits"
    assert kinds["B"] == [38], "the product is not a batch matmul"
    assert kinds["Q"] == [38], "the batched product is not a batch matmul"
    assert kinds["R"] == [38, 19], "addmm is not a product followed by an add"
    assert kinds["C"] == [8], "the fused norm is not a layer norm"
    for advance in ("D", "F", "G"):
        assert kinds[advance] == [3, 3], f"the cache advance {advance} is not two blits"
    assert kinds["E"] == [19], "the scale is not an element-wise op"
    assert kinds["H"] == [28], "the softmax is not a softmax"
    assert kinds["I"] == [29], "the mean is not a reduction"
    assert kinds["J"] == [3], "the narrowing select is not a blit"
    assert kinds["K"] == [19], "the gated activation is not a binary op"
    assert kinds["L"] == [4], "the fp32 round trip left a command behind"
    assert kinds["N"] == [3, 4], "the dynamic slice is not a blit"
    assert any(
        command.patch_param != 0xFFFFFFFF for command in _tagged(cases, "N").commands
    ), "the dynamic slice has no patched parameter"
    for tag in ("D", "F", "G"):
        assert any(
            command.patch_param != 0xFFFFFFFF
            for command in _tagged(cases, tag).commands
        ), f"{tag}: the cache advance has no patched parameter"
    # The commands this branch added, each on the op type it claims to be.
    assert kinds["S"] == [_SHARED_GATHER], "the row gather is not a shared gather"
    assert kinds["S"] == kinds["U"], "the control is not a copy of the gather case"
    assert kinds["V"] == [_REDUCTION], "the dynamic sum is not a reduction"
    assert kinds["V"] == kinds["W"], "the control is not a copy of the sum case"
    for tag in ("X", "Y", "Z", "AF"):
        assert kinds[tag] == [_UNARY], f"{tag}: the clamp family is not the unary one"
    assert all(
        next(iter(_tagged(cases, tag).commands)).params[0] >= 64
        for tag in ("X", "Y", "Z")
    ), "a clamp case is short enough to run the scalar tail"
    assert (
        next(iter(_tagged(cases, "AF").commands)).params[0] < 64
    ), "AF is meant to be the scalar tail alone"
    for tag in ("AA", "AB", "AC"):
        # The emitter's two blits around the command are part of the command
        # being what it claims: without them the kernel reads other channels.
        assert kinds[tag] == [3, _POOL2D, 3], f"{tag}: the pool is not pack/pool/unpack"
    for tag in ("AD", "AE"):
        assert kinds[tag] == [_VISION_ATTENTION], f"{tag}: not a vision attention"

    # The dynamic cases carry the record the run-time length arrives by, and the
    # static ones carry none: a patch on a command that never mentions the length
    # is as wrong as leaving one off. Both halves are asserted, because the
    # simulator applies whatever is there.
    dynamic = blob_interpreter.read_dynamic_trailer(_tagged(cases, "V").blob)
    assert dynamic is not None, "the dynamic sum carries no trailer"
    assert (dynamic.axis, dynamic.max_length) == (
        1,
        _UPPER,
    ), f"the trailer names axis {dynamic.axis} up to {dynamic.max_length}"
    assert dynamic.patches == [(0, 1, 1, 0)], f"the sum patched {dynamic.patches}"
    # The blob on disk still describes the bound: the run-time length only ever
    # reaches this command through the patch, which is what the case decides.
    assert _tagged(cases, "V").commands[0].params[1] == _UPPER
    assert (
        blob_interpreter.read_dynamic_trailer(_tagged(cases, "W").blob) is None
    ), "the control still carries its trailer"
    for tag in ("S", "X", "Y", "Z", "AF", "AA", "AB", "AC", "AD", "AE"):
        assert (
            blob_interpreter.read_dynamic_trailer(_tagged(cases, tag).blob) is None
        ), f"{tag}: a static blob carries a trailer"


def test_every_blob_agrees_three_ways(cases, simulated):
    for case in cases:
        dsp = simulated[f"{case.tag}0"]
        expected = case.expected
        if case.kind == "refused":
            # A control both models are expected to reject outright: the kernel
            # writes nothing, so the slot stays as the arena's fill and the
            # answer is not the one torch computed. The host model raises its own
            # refusal rather than producing a host answer to compare against.
            assert dsp == [0] * len(dsp), (
                f"{case.tag}: the kernel answered {_from_bits(dsp)[:4]} where it "
                "should have refused and written nothing"
            )
            continue
        host = _from_bits([int(value) for value in _host_bits(case.host[0])])
        if case.kind == "teeth":
            # A control case: its bytes are the ones a wrong assumption would
            # have produced. The two models still have to agree with each other
            # -- a disagreement here would be a finding of its own -- and the
            # answer has to have moved, or the comparison on the real bytes is
            # vacuous.
            assert (
                _host_bits(case.host[0]) == dsp
            ), f"{case.tag}: the host and the DSP disagree on the control's bytes"
            assert dsp != expected.view("uint16").tolist(), (
                f"{case.tag}: the control produced the right answer, so the "
                "assumption it was built to falsify is not being tested"
            )
            continue
        if case.kind == "bits":
            expected_bits = expected.view("uint16").tolist()
            assert (
                _host_bits(case.host[0]) == expected_bits
            ), f"{case.tag}: the host model disagrees with torch"
            assert dsp == expected_bits, f"{case.tag}: the DSP disagrees with torch"
        elif case.kind == "tail":
            # The one case the DSP does not agree on, and a real wrong answer
            # rather than a tolerance: it is asserted on its own, below.
            continue
        elif case.kind == "nan":
            # Bit-exact, except that one NaN is as good as another: the claim is
            # that a NaN comes back as a NaN rather than as the bound.
            want = expected.view("uint16").tolist()
            for name, got in (("host model", _host_bits(case.host[0])), ("DSP", dsp)):
                if np.array_equal(_from_bits(got), _from_bits(want), equal_nan=True):
                    continue
                moved = [
                    (index, _from_bits(got)[index], _from_bits(want)[index])
                    for index in range(len(want))
                    if not (
                        np.isnan(_from_bits(got)[index])
                        and np.isnan(_from_bits(want)[index])
                    )
                    and _from_bits(got)[index] != _from_bits(want)[index]
                ]
                raise AssertionError(
                    f"{case.tag}: {name} disagrees with torch at {moved[:8]}"
                )
        else:
            for name, got in (("host model", host), ("DSP", _from_bits(dsp))):
                worst = float(
                    np.max(np.abs(got.astype(np.float32) - expected.astype(np.float32)))
                )
                assert worst < case.tolerance, f"{case.tag}: {name} differs by {worst}"


def test_the_dsp_result_would_move_if_the_blob_did(cases):
    """The agreement above is only worth something if it can break.

    Shifting one byte of the first blit's region has to move the answer, or the
    comparison is vacuous. This is checked on the host model, which runs the
    same region the DSP does.
    """
    case = cases[0]
    data = bytearray(case.blob)
    # The region's innermost source stride, in the first op's param block.
    at = 36 + 16 + 4 * (3 + 8)
    stride = int.from_bytes(data[at : at + 4], "little", signed=True)
    data[at : at + 4] = (stride + 1).to_bytes(4, "little", signed=True)
    changed = execute(
        bytes(data), [_small((2, 3, 4)).numpy(), _small((2, 3, 4)).numpy()]
    )
    assert _host_bits(changed[0]) != _host_bits(
        case.host[0]
    ), "changing the region changed nothing, so the comparison is vacuous"


def test_an_index_past_the_vocabulary_is_where_torch_and_the_kernel_part(cases):
    """The other side of the gather case's out-of-range indices.

    `shared_gather_ops.cc:292-295` clears the row; `torch.embedding` raises. The
    case above takes the kernel's answer, so this is where the divergence itself
    is stated rather than left implicit in a reference nobody reads.
    """
    table = _table_ints(35, 33)
    for bad in (35, -1):
        with pytest.raises(IndexError):
            torch.nn.functional.embedding(torch.tensor([bad]), table)


def test_a_nan_comes_back_as_a_nan_and_not_as_the_bound(cases, simulated):
    """The clamp family's NaN path, which the branch called its largest risk.

    If the kernel's restore is not the bit test it was read from, a NaN input
    comes back as the upper bound: no exception, no wrong shape, one wrong
    number. `nan` comparisons pass on either answer by construction, so this is
    the assertion with teeth -- and it is on the DSP's own output.
    """
    for tag, bound in (("X", None), ("Y", 6.0), ("Z", 1.0)):
        dsp = _from_bits(simulated[f"{tag}0"])
        assert np.isnan(dsp[0]), f"{tag}: the DSP answered {dsp[0]} for a NaN"
        if bound is not None:
            assert dsp[0] != np.float16(
                bound
            ), f"{tag}: a NaN input came back as the upper bound {bound}"


@pytest.mark.xfail(
    strict=True,
    reason="the simulator answers a NaN with the upper bound when the clamp runs "
    "its scalar tail: unary_ops.cc:536-539 has no NaN bit test, and `x > hi` is "
    "true for a NaN there. Reported in SIM-report.md rather than asserted away.",
)
def test_the_clamp_tail_answers_a_nan_with_the_bound(cases, simulated):
    """A wrong answer found on the simulator, on a tensor shorter than one vector.

    `htp_ops_clamp_fp16_chunk` restores a NaN by the `(|x| & 0x7fff) > 0x7c00`
    bit test inside its vector loop (unary_ops.cc:531-532). A tensor of fewer
    than 64 fp16 elements never reaches that loop: `vec_end` is zero and every
    element goes through `x < lo ? lo : x > hi ? hi : x`, which has no bit test
    at all. On the simulator that answers a NaN with the upper bound -- `inf` for
    relu -- where torch, and the host model, return the NaN.

    The case is 32 elements, so it is the tail and nothing else; `X` above is the
    same claim on 128 elements, which is the path that does keep it.
    """
    assert _tagged(cases, "AF").kind == "tail"
    dsp = _from_bits(simulated["AF0"])
    host = _from_bits(_host_bits(_tagged(cases, "AF").host[0]))
    assert np.isnan(host[0]), f"the host model answered {host[0]}"
    assert np.isnan(
        dsp[0]
    ), f"the DSP answered {dsp[0]} for a NaN in a 32-element tensor"


def test_the_vision_case_can_tell_the_two_layouts_apart(cases):
    """Both readings of attention_entry.cc are readings of the same C.

    The other one is `[batch, heads, tokens, headDim]` -- the transposed tensor
    a matmul would want, with the token axis where the heads are. If this case
    cannot separate the two, running it decides nothing, so the separation is
    asserted here rather than assumed from the fact that the layouts are spelled
    differently.
    """
    case = _tagged(cases, "AD")
    query, key, value = case.args
    swapped = _vision_reference_swapped(query, key, value, 0.125)
    worst = float(
        np.max(
            np.abs(
                swapped.float().numpy().reshape(-1) - case.expected.astype(np.float32)
            )
        )
    )
    assert worst > _VISION_TOLERANCE, (
        f"the two readings answer within {worst}, so the tolerance above cannot "
        "tell them apart"
    )


def test_a_vision_param_slot_is_where_the_emitter_put_it(cases):
    """The vision command's params are positional and the ABI does not name them.

    Exchanging `tokens` with `heads`, and replacing the scale bits, each has to
    move the answer. The case above cannot tell a slot that is read from a slot
    that is ignored: this is the run that does.
    """
    expected = _tagged(cases, "AD").expected.view("uint16").tolist()
    for tag in ("AG", "AH"):
        assert _host_bits(_tagged(cases, tag).host[0]) != expected, (
            f"{tag}: the mutated slot left the answer where it was, so the slot "
            "is not read where the emitter puts it"
        )


def test_the_vision_workspace_operand_is_the_one_the_kernel_checks(cases):
    """Shrinking the workspace operand makes the kernel refuse and write nothing.

    The host model implements the same check, in its own words, so this is the
    one vision case where both models refuse: the DSP agrees with the host model
    about the workspace being too small and with neither about an answer.
    """
    case = _tagged(cases, "AI")
    assert case.host is None, (
        "the host model answered a one-byte workspace, so the DSP's silence "
        "cannot be read as a check on the operand the emitter passes"
    )
    assert case.args, "the refused case is still handed the real operands"


def test_the_pool_case_can_tell_the_packed_layout_apart(cases):
    """The blits around the pool command are load bearing.

    The kernel reads its activation as 64-channel blocks, `(y*width + x)*64 + c`.
    Over the row-major buffer that is the same read with the two axes' roles
    exchanged, and on this input that is a different answer -- which is what
    makes the case above a test of the whole chain and not just of the window
    walk.
    """
    case = _tagged(cases, "AA")
    pooled = case.args[0]
    exchanged = (
        pooled.reshape(1, 64, -1).transpose(1, 2).contiguous().reshape(pooled.shape)
    )
    window = _Pool("max", 2, 2, 0)
    assert not torch.equal(
        window(exchanged).half(), torch.from_numpy(case.expected.copy())
    ), (
        "reading the buffer as blocks gives the same answer, so this case does "
        "not see the layout at all"
    )


def test_the_run_time_length_is_what_moves_the_reduction(cases, simulated):
    """The patch is the difference between the two dynamic cases.

    Both are the same blob and the same fixture; one carries the trailer the
    runtime applies and the other has it blanked, which is what the emitter
    produced before `efe2a64`. On the DSP the first has to match torch at the run
    length and the second must answer the exported bound instead. If both
    matched, the poison past the live rows would be invisible and the case would
    not be testing the patch at all.
    """
    expected = _tagged(cases, "V").expected.view("uint16").tolist()
    assert (
        simulated["V0"] == expected
    ), "the patched DSP answer is not torch at the run length"
    # The unpatched command sums the whole bound, poison included, so its answer
    # is the one number that says which span it read.
    whole = float(torch.sum(_sequence(_ROW), dim=1)[0, 0].float())
    assert abs(float(_from_bits(simulated["W0"])[0]) - whole) < 1.0, (
        f"the unpatched answer is {float(_from_bits(simulated['W0'])[0])}, not the "
        f"{whole} a whole-bound read gives"
    )
    assert float(_from_bits(simulated["W0"])[0]) != float(
        _from_bits(simulated["V0"])[0]
    ), "the two cases answered the same thing"
