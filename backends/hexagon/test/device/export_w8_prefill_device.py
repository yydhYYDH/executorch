"""Export W8A16 prefill cases (DSP command 42) and their references, in one process.

Each case is one torch.mm lowered through the weight-only PT2E trip, with the
reference taken from the CONVERTED graph in this same process and this same module
instance, so a later comparison charges the run for the kernel and not for the int8
weight rounding. The command stream is decoded back out of the serialized .pte and
the shared census enforces that the blob it decodes is byte-for-byte inside that
pte, so the blob analysed here is the blob inside the file the phone is given.

Two of the cases are controls rather than measurements, and are labelled as such in
the sidecar: the twin, the same shape with a different weight, and the M == 1 case,
which reaches the other quantized entry (command 45) on the same pipeline.
"""

import hashlib
import json
import os
import sys

WT = os.environ["HEXAGON_WT"]
ART = os.environ["HEXAGON_ART"]
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
sys.path.insert(0, os.environ["HEXAGON_HARNESS"])

import harness as B  # noqa: E402  the per-op harness, unmodified

from executorch.backends.hexagon import hexagon_ops  # noqa: E402
from executorch.backends.hexagon.partition import hexagon_partitioner  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # noqa: E402
import blob_interpreter  # noqa: E402
import executorch  # noqa: E402

# The modules that decide everything below must all come from this worktree and not
# from the editable install, which points at a different checkout.
def _where(module):
    """The directory a module was loaded from; executorch is a namespace package."""
    if getattr(module, "__file__", None):
        return os.path.realpath(module.__file__)
    return os.path.realpath(list(module.__path__)[0])


for _mod in (hexagon_ops, hexagon_partitioner, blob_interpreter, executorch):
    assert _where(_mod).startswith(os.path.realpath(WT)), (
        "%s resolved to %s, not under %s" % (_mod.__name__, _where(_mod), WT))


def run_one(name, scheme, m, k, n, seed):
    """One case: export, write the .pte and the inputs, and write the reference."""
    program, inputs, reference = B.quantized_program(scheme, m, k, n, seed=seed)
    if not isinstance(reference, (tuple, list)):
        reference = (reference,)
    lowered = to_edge_transform_and_lower(
        program,
        partitioner=[hexagon_partitioner.HexagonPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    pte = bytes(lowered.to_executorch().buffer)
    os.makedirs(ART, exist_ok=True)
    open(os.path.join(ART, name + ".pte"), "wb").write(pte)
    for index, value in enumerate(inputs):
        open(os.path.join(ART, "%s_in%d.bin" % (name, index)), "wb").write(
            value.contiguous().numpy().tobytes())
    for index, value in enumerate(reference):
        open(os.path.join(ART, "%s_ref%d.bin" % (name, index)), "wb").write(
            value.contiguous().numpy().tobytes())
    rows, portable = B.census(lowered, pte)
    return {
        "case": name, "m": m, "k": k, "n": n, "seed": seed,
        "pte_bytes": len(pte),
        "pte_sha256": hashlib.sha256(pte).hexdigest(),
        "delegates": rows,
        "left_on_portable": portable,
        "inputs": [
            {"shape": list(v.shape), "dtype": str(v.dtype),
             "bytes": v.numel() * v.element_size()} for v in inputs],
        "outputs": [
            {"shape": list(v.shape), "dtype": str(v.dtype),
             "bytes": v.numel() * v.element_size()} for v in reference],
    }


def emit(name, role, m, k, n, seed, rows):
    """Run one case, check the command that must be there, and fold it into rows."""
    print("=== %s role=%s m=%d k=%d n=%d seed=%d ===" % (name, role, m, k, n, seed),
          flush=True)
    try:
        sidecar = run_one(name, "w8a16", m, k, n, seed)
        types = [c["type"] for d in sidecar["delegates"] for c in d["commands"]]
        want = 45 if m == 1 else 42
        if want not in types:
            raise RuntimeError("command %d absent, decoded %r" % (want, types))
        sidecar["role"] = role
        sidecar["command_types"] = types
        rows.append(sidecar)
        for d in sidecar["delegates"]:
            for c in d["commands"]:
                print("   %s params=%r" % (c["name"], c["params"][:12]))
        print("   pte %dB sha256=%s" % (sidecar["pte_bytes"],
                                       sidecar["pte_sha256"][:16]))
        print("   portable_left=%r outputs=%r" % (sidecar["left_on_portable"],
                                                sidecar["outputs"]))
    except BaseException as exc:
        rows.append({"case": name, "role": role, "m": m, "k": k, "n": n, "seed": seed,
                     "error": "%s: %s" % (type(exc).__name__, exc)})
        print("ERROR %s: %s: %s" % (name, type(exc).__name__, exc), flush=True)


def main():
    cases = (
        ("w8_m4_k4096_n128", "measurement", 4, 4096, 128, 0),
        ("w8_m33_k4096_n128", "measurement", 33, 4096, 128, 0),
        ("w8_m40_k25216_n64", "measurement", 40, 25216, 64, 0),
        ("w8_m197_k768_n576", "measurement", 197, 768, 576, 0),
        ("w8_twin_m4_k4096_n128", "control-weight", 4, 4096, 128, 7717),
        ("w8_m1_k4096_n128", "control-m1", 1, 4096, 128, 0),
    )
    rows = []
    for case in cases:
        emit(*case, rows)
    meta = {
        "worktree": WT,
        "tree_revision": B.tree_revision(),
        "hexagon_ops_file": hexagon_ops.__file__,
        "blob_interpreter_file": blob_interpreter.__file__,
        "rows": rows,
    }
    json.dump(meta, open(os.path.join(ART, "matrix.json"), "w"), indent=2)
    print("TREE=%s" % meta["tree_revision"])
    print("HEXAGON_OPS=%s" % meta["hexagon_ops_file"])
    raise SystemExit(1 if any("error" in r for r in rows) else 0)


if __name__ == "__main__":
    main()
