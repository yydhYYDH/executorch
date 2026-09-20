"""Round-trip the Hexagon delegate blob between the AOT writer and its reader.

The Python writer in serialization/blob.py and the C++ structs the runtime
includes are two independent implementations of one packed on-disk format, and
nothing compares them: a field one side forgets, or a layout that changes on one
side only, produces wrong numbers on a device rather than an error. This writes a
blob that exercises every field, reads it back through the C++ structs, and
compares them.

It needs no device, no Hexagon SDK and no FlatBuffers -- only a host C++
compiler -- so it can run wherever the backend is checked out.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

_HEXAGON_DIR = pathlib.Path(__file__).resolve().parents[1]
# .../src, the directory holding the executorch package, so the worktree is
# tested rather than whatever happens to be installed.
_SRC_DIR = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, os.fspath(_SRC_DIR))

from executorch.backends.hexagon.serialization import blob as B  # noqa: E402

_READER_SRC = pathlib.Path(__file__).resolve().parent / "blob_reader.cpp"
_SCHEMA_INC = _HEXAGON_DIR / "serialization"

# Offsets the C++ side must report, derived here from the Python struct formats
# rather than restated: two independent derivations are the whole point.
_FIELD_BYTES = 4 * 4  # type, n_inputs, n_outputs, n_params
_TRAILER_BYTES = 4 * 4  # patch_param, patch_input, patch_scale, in_place
_EXPECTED_OP_OFFSETS = {
    "params": _FIELD_BYTES,
    "patch_param": _FIELD_BYTES + 4 * B.MAX_OP_PARAMS,
    "patch_input": _FIELD_BYTES + 4 * B.MAX_OP_PARAMS + 4,
    "patch_scale": _FIELD_BYTES + 4 * B.MAX_OP_PARAMS + 8,
    "in_place": _FIELD_BYTES + 4 * B.MAX_OP_PARAMS + 12,
    "inputs": _FIELD_BYTES + 4 * B.MAX_OP_PARAMS + _TRAILER_BYTES,
    "outputs": _FIELD_BYTES
    + 4 * B.MAX_OP_PARAMS
    + _TRAILER_BYTES
    + B.MAX_OP_INPUTS * B._TENSOR_REF.size,
}

# What build() must pick for the two sections it packs itself, stated
# independently of how it computes them.
_PACKED_INPUTS = [
    (int(B.TensorSpace.INPUT), 0, 0, 128),
    (int(B.TensorSpace.INPUT), 1, 128, 8),
    (int(B.TensorSpace.INPUT), 2, 256, 256),
]
_PACKED_OUTPUTS = [
    (int(B.TensorSpace.OUTPUT), 0, 0, 128),
    (int(B.TensorSpace.OUTPUT), 1, 128, 512),
]
_PADDED_SLOT = (int(B.TensorSpace.ABSENT), 0, 0, 0)


def _build_blob():
    """A blob whose every field is distinguishable from every other."""
    builder = B.BlobBuilder(n_inputs=3, n_outputs=2)
    in0 = builder.method_input(0, 128)
    in1 = builder.method_input(1, 8)
    in2 = builder.method_input(2, 256)
    out0 = builder.method_output(0, 128)
    out1 = builder.method_output(1, 512)
    # A second weight, so the inter-call 128-byte alignment gets exercised.
    w0 = builder.add_weights(bytes(range(10)))
    w1 = builder.add_weights(b"\xde\xad\xbe\xef")
    a0 = builder.add_activation(100)
    a1 = builder.add_activation(60)

    ops = [
        # Everything set at once: a patch slot, in-place bits, an absent operand.
        B.Op(
            type=18,
            inputs=[in0, in1, B.ABSENT],
            outputs=[out0, out1],
            params=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            patch=(1, 2),
            patch_scale=1024,
            in_place=0b101,
        ),
        # The other extreme: no params, no patch, weights and scratch only.
        B.Op(type=4, inputs=[w0], outputs=[a0]),
        # Every slot full, to pin the padding too.
        B.Op(
            type=19,
            inputs=[in2, w1, a1] + [B.ABSENT] * (B.MAX_OP_INPUTS - 3),
            outputs=[out0, out1, a0, a1],
            params=list(range(B.MAX_OP_PARAMS)),
            patch=(B.MAX_OP_PARAMS - 1, B.MAX_OP_INPUTS - 1),
            patch_scale=0xFFFFFFFF,
            in_place=0xFFFFFFFF,
        ),
    ]
    for op in ops:
        builder.add_op(op)
    return builder.build(), ops


def _reader_binary(tmp_path):
    """Builds the reader with the host compiler; failing to compile is a failure."""
    cxx = os.environ.get("CXX", "c++")
    if shutil.which(cxx) is None:
        pytest.skip(f"no host C++ compiler ({cxx})")
    binary = tmp_path / "blob_reader"
    built = subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-I",
            os.fspath(_SCHEMA_INC),
            os.fspath(_READER_SRC),
            "-o",
            os.fspath(binary),
        ],
        capture_output=True,
        text=True,
    )
    # Not a skip: the header and the reader are ours, so this failing means the
    # layout is broken, which is exactly what the test exists to catch.
    assert built.returncode == 0, f"blob_reader failed to compile:\n{built.stderr}"
    return binary


def _read_back(tmp_path, blob_bytes):
    blob_path = tmp_path / "roundtrip.bin"
    blob_path.write_bytes(blob_bytes)
    done = subprocess.run(
        [os.fspath(_reader_binary(tmp_path)), os.fspath(blob_path)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"blob_reader failed:\n{done.stderr}"

    sizes, offsets, header, sections, ops = {}, {}, {}, {}, {}
    for line in done.stdout.splitlines():
        field = line.split()
        if field[0] == "size":
            sizes[field[1]] = int(field[2])
        elif field[0] == "offsetof":
            offsets[field[2]] = int(field[3])
        elif field[0] == "header":
            header[field[1]] = int(field[2])
        elif field[0] == "section":
            hexed = field[3] if len(field) > 3 else ""
            sections[field[1]] = (int(field[2]), hexed)
        elif field[0] == "op":
            op = ops.setdefault(int(field[1]), {"params": {}, "input": {}, "output": {}})
            if field[2] == "param":
                op["params"][int(field[3])] = int(field[4])
            elif field[2] in ("input", "output"):
                op[field[2]][int(field[3])] = tuple(int(v) for v in field[4:])
            else:
                op[field[2]] = int(field[3])
        else:
            raise AssertionError(f"unparsed reader line: {line!r}")
    return sizes, offsets, header, sections, ops


def _remap(ref):
    if ref.space == B.TensorSpace.INPUT:
        return _PACKED_INPUTS[ref.index]
    if ref.space == B.TensorSpace.OUTPUT:
        return _PACKED_OUTPUTS[ref.index]
    return (int(ref.space), ref.index, ref.offset, ref.size)


def _expected_slots(refs, limit):
    live = [_remap(ref) for ref in refs]
    return dict(enumerate(live + [_PADDED_SLOT] * (limit - len(refs))))


def test_blob_roundtrip(tmp_path):
    blob_bytes, written_ops = _build_blob()
    sizes, offsets, header, sections, ops = _read_back(tmp_path, blob_bytes)

    # The one number the docs had wrong in two places at once.
    assert sizes["HexagonOp"] == B.OP_SIZE
    assert sizes["HexagonBlobHeader"] == B.HEADER_SIZE
    assert sizes["HexagonTensorRef"] == B._TENSOR_REF.size
    assert offsets == _EXPECTED_OP_OFFSETS

    assert header == {
        "magic": B.BLOB_MAGIC,
        "version": B.BLOB_VERSION,
        "n_ops": len(written_ops),
        "n_inputs": 3,
        "n_outputs": 2,
        "weights_bytes": 132,
        "inputs_bytes": 512,
        "activations_bytes": 188,
        "outputs_bytes": 640,
    }

    assert sorted(ops) == list(range(len(written_ops)))

    for index, written in enumerate(written_ops):
        seen = ops[index]
        assert seen["type"] == written.type, f"op {index} type"
        assert seen["n_inputs"] == len(written.inputs), f"op {index} n_inputs"
        assert seen["n_outputs"] == len(written.outputs), f"op {index} n_outputs"
        assert seen["n_params"] == len(written.params), f"op {index} n_params"

        # Params are zero-padded past n_params, so both sides have to agree on
        # the padding as well as the values.
        assert seen["params"] == dict(
            enumerate(
                list(written.params) + [0] * (B.MAX_OP_PARAMS - len(written.params))
            )
        ), f"op {index} params"

        # The two fields build() used to drop on the floor.
        patch_param, patch_input = written.patch or (B.NO_PATCH, 0)
        assert seen["patch_param"] == patch_param, f"op {index} patch_param"
        assert seen["patch_input"] == patch_input, f"op {index} patch_input"
        assert seen["patch_scale"] == written.patch_scale, f"op {index} patch_scale"
        assert seen["in_place"] == written.in_place, f"op {index} in_place"

        assert seen["input"] == _expected_slots(
            written.inputs, B.MAX_OP_INPUTS
        ), f"op {index} inputs"
        assert seen["output"] == _expected_slots(
            written.outputs, B.MAX_OP_OUTPUTS
        ), f"op {index} outputs"

    # Section contents, including the alignment padding between the two weights.
    assert sections["weights"] == (
        132,
        (bytes(range(10)) + bytes(118) + b"\xde\xad\xbe\xef").hex(),
    )
    # The activation section is a header size alone: the runtime reserves it and
    # never reads it from the blob, so the file stops at the weights.
    assert sections.keys() == {"weights"}
    assert len(blob_bytes) == B.HEADER_SIZE + len(written_ops) * B.OP_SIZE + 132
