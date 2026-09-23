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

import hexagon_sim  # noqa: E402
from blob_interpreter import Arena, execute, read_blob  # noqa: E402
from executorch.backends.hexagon.hexagon_backend import HexagonBackend  # noqa: E402
from executorch.backends.hexagon.hexagon_ops import sdpa_targets  # noqa: E402
from executorch.exir import to_edge  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402


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


def _case(tag, program, args, expected, kind="bits", tolerance=None):
    if isinstance(program, torch.nn.Module):
        program = to_edge(export(program, tuple(args))).exported_program()
    blob = HexagonBackend.preprocess(program, []).processed_bytes
    header, commands = read_blob(blob)
    # execute() checks each input's byte size against the slot the emitters
    # claimed, so a blob whose inputs are in another order fails here.
    host = execute(blob, [t.numpy() for t in args])
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
    )


def _tagged(cases, tag):
    """One case by tag. Indexing the list would break when a case is added."""
    return next(case for case in cases if case.tag == tag)


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
    ]


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
            f'  {{"{case.tag}", kBlob{index}, kInputData{index}, '
            f"sizeof(kInputData{index})}},"
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


def test_every_blob_agrees_three_ways(cases, simulated):
    for case in cases:
        dsp = simulated[f"{case.tag}0"]
        host = _from_bits([int(value) for value in _host_bits(case.host[0])])
        if case.kind == "bits":
            expected = case.expected.view("uint16").tolist()
            assert (
                _host_bits(case.host[0]) == expected
            ), f"{case.tag}: the host model disagrees with torch"
            assert dsp == expected, f"{case.tag}: the DSP disagrees with torch"
        else:
            expected = case.expected
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
