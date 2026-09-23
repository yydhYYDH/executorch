# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The compile specs that change the blob, and the runtime's check that they did.

A hexagon tunable that changes the bytes of the blob cannot be a runtime option:
a runner that flipped one at load time would be running a layout the program was
never compiled for. It rides in the .pte as a compile spec instead, and the
runtime re-derives the same fact from the blob and refuses a load where the two
disagree. One half of that is Python, one half is C++ -- `hexagon_compat.h` --
and neither can be checked against the other by reading both, so this drives the
two over the same blobs.

The C++ half is built by the host compiler, which needs no device and no Hexagon
SDK, so its failure paths are covered here rather than only on a board.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

# The checkout directory is itself named executorch, so putting its parent on
# the path makes "import executorch" resolve to this tree; the editable install
# in this environment points at a different checkout with an older backend.
sys.path.insert(0, os.fspath(pathlib.Path(__file__).resolve().parents[4]))

from executorch.backends.hexagon.hexagon_backend import (  # noqa: E402
    ATTN_PAGED_SPEC,
    EXTERNAL_WEIGHTS_MAX_BYTES_SPEC,
    HMX_PREPACK_SPEC,
    HexagonBackend,
    HexagonCompileOptions,
)
from executorch.backends.hexagon.hexagon_ops import sdpa_targets  # noqa: E402
from executorch.backends.hexagon.serialization import blob as B  # noqa: E402
from executorch.backends.hexagon.test.blob_interpreter import (  # noqa: E402
    read_blob,
    read_external_weights,
)
from executorch.exir import to_edge  # noqa: E402
from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: E402
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.export import export  # noqa: E402

_HEXAGON_DIR = pathlib.Path(__file__).resolve().parents[1]
_CHECKER_SRC = pathlib.Path(__file__).resolve().parent / "compile_spec_checker.cpp"

#: BATCH_MATMUL's hmxFlags word, from the emitter's own plan encoding.
_HMX_PLAN_MAGIC = 0x484D58
_HMX_PLAN_PREPACKED = 1
_HMX_FLAGS_PARAM = 26
#: FLASH_ATTN's page_size.
_PAGE_SIZE_PARAM = 10

_BATCH_MATMUL = 38
_FLASH_ATTN = 18

#: The checker is compiled once per session, not once per call.
_CHECKER_BINARY = None
_CHECKER_DIR = None


def _program(graph_module):
    """Wrap a hand-built graph module the way preprocess expects a program."""
    return SimpleNamespace(
        graph_module=graph_module,
        graph_signature=SimpleNamespace(
            inputs_to_buffers={},
            buffers_to_mutate={},
            inputs_to_parameters={},
            inputs_to_lifted_tensor_constants={},
        ),
        range_constraints={},
    )


class _Project(torch.nn.Module):
    """A matmul whose weight the delegate owns, so the weight can be packed."""

    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(k, n, dtype=torch.float16))

    def forward(self, x):
        return torch.mm(x, self.weight)


def _preprocess(program, options=None):
    """The blob `preprocess` produces, with or without any options."""
    specs = [] if options is None else options.to_compile_specs()
    return HexagonBackend.preprocess(program, specs).processed_bytes


def _mm_blob(options=None, rows=64, k=64, n=64):
    """A blob with one matmul large enough for the emitter to pack its weight."""
    torch.manual_seed(0)
    model = _Project(k, n)
    x = torch.randn(rows, k, dtype=torch.float16)
    program = to_edge(export(model, (x,))).exported_program()
    return _preprocess(program, options)


def _attention_graph(batch, qo_len, n_heads, n_kv_heads, max_kv_len, head_dim):
    """Attention over a cache the graph already holds, as the sim tests build it."""
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


def _params(blob, op_type):
    """The params of the blob's first command of this type."""
    _, commands = read_blob(blob)
    for command in commands:
        if command.type == op_type:
            return list(command.params)
    raise AssertionError(f"the blob has no op of type {op_type}")


def _external_blob(max_bytes=8):
    """A blob with two weights the writer put outside of it, built without a model."""
    builder = B.BlobBuilder(
        n_inputs=1, n_outputs=1, external_weights_max_bytes=max_bytes
    )
    weight = [builder.add_weights(bytes(range(64))), builder.add_weights(b"x" * 64)]
    source = builder.method_input(0, 64)
    target = builder.method_output(0, 64)
    builder.add_op(
        B.Op(type=38, inputs=[source, *weight], outputs=[target], params=[2] * 28)
    )
    return builder.build(), builder.external_weight_data()


# ---------------------------------------------------------------------------
# What the writer accepts, and what it refuses.
# ---------------------------------------------------------------------------


def test_the_default_specs_are_the_defaults():
    """No specs at all is a complete request, and it means what it always meant."""
    specs = HexagonCompileOptions().to_compile_specs()
    assert [(spec.key, bytes(spec.value)) for spec in specs] == [
        (HMX_PREPACK_SPEC, b"\x01"),
        (ATTN_PAGED_SPEC, b"\x00"),
    ]
    assert HexagonCompileOptions.from_compile_specs(specs) == HexagonCompileOptions()
    assert HexagonCompileOptions.from_compile_specs([]) == HexagonCompileOptions()


def test_asking_for_the_defaults_keeps_the_blob_as_it_was():
    """The defaults and "no options at all" are the same bytes, not just the same values."""
    assert _mm_blob(HexagonCompileOptions()) == _mm_blob(None)


def test_a_spec_this_backend_does_not_define_is_refused():
    with pytest.raises(ValueError) as raised:
        HexagonCompileOptions.from_compile_specs(
            [CompileSpec("hexagon_tile_budget", b"\x00" * 8)]
        )
    assert "hexagon_tile_budget" in str(raised.value)
    assert HMX_PREPACK_SPEC in str(raised.value)


@pytest.mark.parametrize("width", [0, 2, 4, 8])
def test_a_flag_of_the_wrong_width_is_refused(width):
    with pytest.raises(ValueError) as raised:
        HexagonCompileOptions.from_compile_specs(
            [CompileSpec(HMX_PREPACK_SPEC, bytes(width))]
        )
    assert "1 bytes" in str(raised.value)


@pytest.mark.parametrize("width", [0, 1, 4, 16])
def test_a_byte_count_of_the_wrong_width_is_refused(width):
    with pytest.raises(ValueError) as raised:
        HexagonCompileOptions.from_compile_specs(
            [CompileSpec(EXTERNAL_WEIGHTS_MAX_BYTES_SPEC, bytes(width))]
        )
    assert "8 bytes" in str(raised.value)


def test_a_value_that_cannot_mean_what_its_key_says_is_refused():
    """The boundary is the one Vulkan checks for its own external data cap."""
    for spec, expected in (
        (CompileSpec(HMX_PREPACK_SPEC, b"\x02"), "must be 0 or 1"),
        (CompileSpec(ATTN_PAGED_SPEC, b"\xff"), "must be 0 or 1"),
        (CompileSpec(EXTERNAL_WEIGHTS_MAX_BYTES_SPEC, bytes(8)), "positive uint64"),
    ):
        with pytest.raises(ValueError) as raised:
            HexagonCompileOptions.from_compile_specs([spec])
        assert expected in str(raised.value), raised.value


def test_a_key_given_twice_is_refused():
    with pytest.raises(ValueError) as raised:
        HexagonCompileOptions.from_compile_specs(
            [
                CompileSpec(ATTN_PAGED_SPEC, b"\x00"),
                CompileSpec(ATTN_PAGED_SPEC, b"\x01"),
            ]
        )
    assert "twice" in str(raised.value)


def test_an_external_weight_ceiling_travels_as_eight_little_endian_bytes():
    specs = HexagonCompileOptions(external_weights_max_bytes=1024).to_compile_specs()
    assert specs[-1].key == EXTERNAL_WEIGHTS_MAX_BYTES_SPEC
    assert bytes(specs[-1].value) == (1024).to_bytes(8, "little")


# ---------------------------------------------------------------------------
# What the specs do to the blob.
# ---------------------------------------------------------------------------


def test_the_hmx_spec_decides_whether_the_weight_is_tiled():
    packed = _mm_blob()
    plain = _mm_blob(HexagonCompileOptions(hmx_prepack=False))

    assert _params(packed, _BATCH_MATMUL)[_HMX_FLAGS_PARAM] == (
        _HMX_PLAN_MAGIC | _HMX_PLAN_PREPACKED
    )
    assert (
        _params(plain, _BATCH_MATMUL)[_HMX_FLAGS_PARAM] == _HMX_PLAN_MAGIC
    ), "the default packed the weight, so this blob proves nothing"
    # The same number of bytes in another order, not a larger program.
    assert len(packed) == len(plain)


@pytest.mark.skipif(
    not sdpa_targets(), reason="llama.custom_sdpa is not registered in this checkout"
)
def test_the_attention_spec_decides_which_entry_point_runs():
    paged = _preprocess(
        _attention_graph(1, 3, 16, 8, 8, 128),
        HexagonCompileOptions(attn_paged=True),
    )
    plain = _preprocess(_attention_graph(1, 3, 16, 8, 8, 128))

    assert _params(plain, _FLASH_ATTN)[_PAGE_SIZE_PARAM] == 0
    assert _params(plain, _FLASH_ATTN)[_PAGE_SIZE_PARAM - 1] == 0
    assert _params(paged, _FLASH_ATTN)[_PAGE_SIZE_PARAM] == 256
    assert _params(paged, _FLASH_ATTN)[_PAGE_SIZE_PARAM - 1] > 0


# ---------------------------------------------------------------------------
# The runtime's half: what the C++ checker makes of the same bytes.
# ---------------------------------------------------------------------------


def _include_roots():
    """The roots that resolve `executorch/...` the way the interpreter did.

    A source checkout is imported through a shim directory -- `src` here, the
    checkout's parent upstream -- and the header reaches the schema through the
    same `executorch/...` spelling the runtime build uses, so the compiler is
    given whichever root the interpreter actually loaded this backend from.
    """
    roots = []
    for entry in sys.path:
        if not entry:
            continue
        probe = pathlib.Path(entry) / "executorch" / "backends" / "hexagon"
        if (probe / "serialization" / "hexagon_schema.h").is_file():
            roots.append(entry)
    assert roots, "no include root in sys.path holds this backend"
    return roots


def _checker():
    """The host compiler builds the checker; failing to build is a failure."""
    global _CHECKER_BINARY, _CHECKER_DIR
    if _CHECKER_BINARY is None:
        cxx = os.environ.get("CXX", "c++")
        if shutil.which(cxx) is None:
            pytest.skip(f"no host C++ compiler ({cxx})")
        _CHECKER_DIR = pathlib.Path(tempfile.mkdtemp(prefix="hexagon_specs_"))
        binary = _CHECKER_DIR / "compile_spec_checker"
        built = subprocess.run(
            [
                cxx,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                *[f"-I{root}" for root in _include_roots()],
                "-I",
                os.fspath(_HEXAGON_DIR / "serialization"),
                "-I",
                os.fspath(_HEXAGON_DIR / "runtime"),
                os.fspath(_CHECKER_SRC),
                "-o",
                os.fspath(binary),
            ],
            capture_output=True,
            text=True,
        )
        assert (
            built.returncode == 0
        ), f"compile_spec_checker failed to build:\n{built.stderr}"
        _CHECKER_BINARY = binary
    return _CHECKER_BINARY


def _inspect(blob, specs=()):
    """Runs the checker over a blob, returning its report as a list of fields."""
    binary = _checker()
    path = binary.parent / "blob.bin"
    path.write_bytes(blob)
    arguments = [os.fspath(binary), os.fspath(path)]
    arguments += [f"{spec.key}=0x{bytes(spec.value).hex()}" for spec in specs]
    done = subprocess.run(arguments, capture_output=True, text=True)
    assert done.returncode == 0, f"checker failed:\n{done.stderr}"
    return [line.split() for line in done.stdout.splitlines()]


def _line(report, *prefix):
    """The fields after `prefix` on the first line that starts with it."""
    for line in report:
        if line[: len(prefix)] == list(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"the checker said nothing about {prefix}: {report}")


def _sentence(report, label):
    return " ".join(_line(report, label))


def test_the_checker_agrees_with_the_writer_about_the_layout():
    """Two derivations of one format, with the C++ side doing the reading."""
    report = _inspect(_mm_blob(), HexagonCompileOptions().to_compile_specs())
    assert _line(report, "size", "trailer") == [str(B._EXTERNAL_HEADER.size)]
    assert _line(report, "size", "entry") == [str(B._EXTERNAL_WEIGHT.size)]
    assert _line(report, "offsetof", "entry", "key") == ["16"]
    assert _line(report, "offsetof", "trailer", "weights_bytes") == ["16"]


def test_the_checker_accepts_the_specs_the_blob_was_built_with():
    report = _inspect(_mm_blob(), HexagonCompileOptions().to_compile_specs())
    assert _line(report, "verdict") == ["ok"]
    assert _line(report, "facts", "hmx_prepacked") == ["1"], report
    assert _line(report, "facts", "external") == ["0"]


def test_the_checker_refuses_a_spec_the_blob_contradicts():
    """Every direction the check has, on blobs the writer really produced."""
    packed = _mm_blob()
    plain = _mm_blob(HexagonCompileOptions(hmx_prepack=False))
    says_plain = HexagonCompileOptions(hmx_prepack=False).to_compile_specs()
    says_packed = HexagonCompileOptions(hmx_prepack=True).to_compile_specs()

    contradicts = _inspect(packed, says_plain)
    assert _line(contradicts, "verdict") == ["mismatch"]
    assert "tile order" in _sentence(contradicts, "reason")

    # The other direction cannot be contradicted: a program with no matmul, or
    # none large enough to be worth packing, packs nothing either way.
    assert _line(_inspect(plain, says_packed), "verdict") == ["ok"]
    assert _line(_inspect(plain, says_plain), "verdict") == ["ok"]


def test_the_checker_refuses_a_spec_that_is_not_ours_or_is_malformed():
    blob = _mm_blob()
    unknown = _inspect(blob, [CompileSpec("hexagon_something_else", b"\x01")])
    assert _line(unknown, "verdict") == ["unknown_key"]

    for specs in (
        [CompileSpec(HMX_PREPACK_SPEC, b"\x01\x00")],
        [CompileSpec(HMX_PREPACK_SPEC, b"\x02")],
        [CompileSpec(EXTERNAL_WEIGHTS_MAX_BYTES_SPEC, b"\x00\x04\x00\x00")],
        [CompileSpec(EXTERNAL_WEIGHTS_MAX_BYTES_SPEC, bytes(8))],
    ):
        report = _inspect(blob, specs)
        assert _line(report, "verdict") == ["bad_payload"], (specs, report)


def test_the_checker_reads_the_trailer_the_writer_wrote():
    """The offsets, sizes and keys the runtime will turn into copies."""
    blob, supplied = _external_blob()
    weights = read_external_weights(blob)
    assert len(weights.entries) == 2, weights.entries

    report = _inspect(
        blob, HexagonCompileOptions(external_weights_max_bytes=8).to_compile_specs()
    )
    assert _line(report, "verdict") == ["ok"]
    assert _line(report, "facts", "external") == ["1"], report
    assert _line(report, "inspect") == ["ok"]
    assert _line(report, "weights") == [
        str(weights.weights_bytes),
        str(weights.trailer_at),
    ]
    assert _line(report, "external") == ["2"]
    # index, offset, size, key.
    assert [line[2:] for line in report if line[0] == "entry"] == [
        [str(entry.offset), str(entry.size), entry.key] for entry in weights.entries
    ]
    # And the bytes the writer meant to move are the ones the store carries.
    assert [key for key, _ in supplied] == [entry.key for entry in weights.entries]
    assert sum(len(data) for _, data in supplied) == sum(
        entry.size for entry in weights.entries
    )


def test_a_blob_with_external_weights_needs_the_spec_that_says_so():
    blob, _ = _external_blob()
    report = _inspect(blob, HexagonCompileOptions().to_compile_specs())
    assert _line(report, "verdict") == ["mismatch"]
    assert "no hexagon_external_weights_max_bytes" in _sentence(report, "reason")
    # The trailer itself is still readable: the verdict is about the spec.
    assert _line(report, "inspect") == ["ok"]


@pytest.mark.parametrize(
    "damage", ["magic", "version", "count", "truncated", "order", "size", "key"]
)
def test_the_checker_refuses_a_corrupted_trailer(damage):
    """Every field is load-bearing, so each is broken on its own."""
    blob, _ = _external_blob()
    entries = read_external_weights(blob).entries
    data = bytearray(blob)
    entries_at = read_external_weights(blob).trailer_at - (
        len(entries) * B._EXTERNAL_WEIGHT.size
    )
    trailer_at = entries_at - B._EXTERNAL_HEADER.size
    first, second = entries

    if damage == "magic":
        fields = list(B._EXTERNAL_HEADER.unpack_from(data, trailer_at))
        fields[0] = 0
        B._EXTERNAL_HEADER.pack_into(data, trailer_at, *fields)
    elif damage == "version":
        fields = list(B._EXTERNAL_HEADER.unpack_from(data, trailer_at))
        fields[1] = 99
        B._EXTERNAL_HEADER.pack_into(data, trailer_at, *fields)
    elif damage == "count":
        fields = list(B._EXTERNAL_HEADER.unpack_from(data, trailer_at))
        fields[2] = 5000
        B._EXTERNAL_HEADER.pack_into(data, trailer_at, *fields)
    elif damage == "truncated":
        del data[entries_at + B._EXTERNAL_WEIGHT.size :]
    elif damage == "order":
        # The second entry claims the first one's bytes, which is a record that
        # reads back as two copies of the same weight rather than as a failure.
        B._EXTERNAL_WEIGHT.pack_into(
            data,
            entries_at + B._EXTERNAL_WEIGHT.size,
            first.offset,
            second.size,
            second.key.encode(),
        )
    elif damage == "size":
        B._EXTERNAL_WEIGHT.pack_into(
            data, entries_at, first.offset, 1 << 40, first.key.encode()
        )
    elif damage == "key":
        B._EXTERNAL_WEIGHT.pack_into(
            data,
            entries_at,
            first.offset,
            first.size,
            b"\x01" * B.EXTERNAL_KEY_BYTES,
        )

    report = _inspect(
        bytes(data),
        HexagonCompileOptions(external_weights_max_bytes=8).to_compile_specs(),
    )
    # The reason it gave is part of the contract: "failed" alone would not tell
    # a reader which field of the trailer was wrong.
    assert _line(report, "inspect")[0] == "failed", (damage, report)
    assert _line(report, "inspect")[1:], (damage, report)
