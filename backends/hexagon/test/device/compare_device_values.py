"""Compare the fetched device outputs against the references the export produced.

Three things make a green verdict here mean something.

One, the reference is the one the exporting process wrote, in the dtype and at
the shape that process recorded in the sidecar, so the comparison is not made at
a width nobody checked.

Two, every case is also compared against a deliberately corrupted copy of its own
reference -- one element moved by eight output steps. A comparator that cannot
call that wrong is not a comparator, and a device reading that agreed with both
would mean the comparison, not the device, is broken.

Three, the twin case (same shape, different weight) is compared against the main
case device output. A kernel that did not read the packed weight, or a reader
that handed back one fixed buffer, would answer those two the same way.
"""

import json
import os
import sys

import numpy as np

ART = sys.argv[1]
FETCH = sys.argv[2]

matrix = json.load(open(os.path.join(ART, "matrix_small.json")))
DTYPE = {"torch.float32": np.float32, "torch.float16": np.float16}


def read(path, dtype):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=dtype)


def compare(got, ref):
    """n, max_abs, max_rel, n_wrong, n_exact for two float arrays."""
    if got is None or ref is None or got.size != ref.size:
        return None
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    d = np.abs(g - r)
    scale = max(float(np.max(np.abs(r))), 1e-30)
    step = float(np.spacing(np.float16(np.float32(scale))))
    return {
        "n": int(g.size),
        "max_abs": float(d.max()) if d.size else 0.0,
        "max_rel": float((d / np.maximum(np.abs(r), 1e-30)).max()) if d.size else 0.0,
        "fp16_steps_at_scale": float(d.max() / step) if d.size else 0.0,
        "wrong": int((d > step).sum()),
        "exact": int((d == 0).sum()),
        "ref_absmax": scale,
        "tolerance_step": step,
    }


rows = []
device = {}
for row in matrix["rows"]:
    if "error" in row:
        rows.append({"case": row["case"], "status": "export-error",
                     "error": row["error"]})
        continue
    name = row["case"]
    spec = row["outputs"][0]
    dtype = DTYPE[spec["dtype"]]
    got = read(os.path.join(FETCH, "out_%s-0.bin" % name), dtype)
    ref = read(os.path.join(ART, "%s_ref0.bin" % name), dtype)
    verdict = compare(got, ref)
    bad = None
    if ref is not None:
        broken = ref.astype(np.float64).copy().reshape(-1)
        broken[0] += 8.0 * max(float(np.max(np.abs(broken))), 1e-30)
        bad = compare(got, broken.astype(dtype))
    device[name] = got
    if got is None:
        status = "no-output-file"
    elif verdict is None:
        status = "size-mismatch"
    elif verdict["wrong"] == 0:
        status = "PASS"
    else:
        status = "DIFFERS"
    rows.append({
        "case": name, "role": row["role"], "m": row["m"], "k": row["k"],
        "n": row["n"], "command_types": row["command_types"],
        "pte_sha256": row["pte_sha256"][:16],
        "output_dtype": spec["dtype"], "output_bytes": spec["bytes"],
        "fetched_bytes": 0 if got is None else int(got.size * got.dtype.itemsize),
        "verdict": verdict,
        "detector_control_wrong": None if bad is None else bad["wrong"],
        "detector_control_max_abs": None if bad is None else bad["max_abs"],
        "status": status,
    })

a = device.get("w8_m4_k4096_n128")
b = device.get("w8_twin_m4_k4096_n128")
twin = None
if a is not None and b is not None:
    twin = {
        "identical_bytes": bool(np.array_equal(a, b)),
        "max_abs_between_them": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
        "fraction_of_elements_that_differ": float(np.mean(a != b)),
    }

print(json.dumps({"tree": matrix["tree_revision"], "rows": rows,
                  "weight_control": twin}, indent=2))
