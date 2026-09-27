import json
import os
import sys

import numpy as np

ART = sys.argv[1]
DEV = sys.argv[2] if len(sys.argv) > 2 else None

ref = np.load(os.path.join(ART, "edsr_ref.npy")).astype(np.float64).reshape(-1)
inp = np.fromfile(os.path.join(ART, "edsr_in0.bin"), dtype=np.float32).astype(np.float64)


def stats(got, r):
    d = np.abs(got - r)
    mse = float(np.mean(d ** 2))
    peak = float(np.max(np.abs(r)))
    psnr = 10.0 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")
    return {
        "n": int(got.size),
        "max_abs": float(d.max()), "mean_abs": float(d.mean()),
        "rmse": float(np.sqrt(mse)), "mse": mse, "peak_ref": peak, "psnr_db": psnr,
        "exact_frac": float(np.mean(d == 0.0)),
        "ref_absmax": float(np.max(np.abs(r))), "ref_absmean": float(np.mean(np.abs(r))),
        "got_absmax": float(np.max(np.abs(got))), "got_absmean": float(np.mean(np.abs(got))),
        "got_nan": int(np.sum(~np.isfinite(got))), "ref_nan": int(np.sum(~np.isfinite(r))),
        "got_inf": int(np.sum(np.isinf(got))),
    }


out = {"ref_shape": [1, 3, 128, 128], "input_numel": int(inp.size)}
if DEV and os.path.exists(DEV):
    raw = open(DEV, "rb").read()
    out["device_bytes"] = len(raw)
    out["device_float32_count"] = len(raw) // 4
    got = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    out["device_vs_ref"] = stats(got, ref) if got.size == ref.size else "SIZE MISMATCH"
    bad = ref.copy()
    bad[7] += 8.0 * float(np.max(np.abs(ref)))
    out["control_perturbed_ref"] = stats(got, bad)
    a = got - got.mean()
    b = ref - ref.mean()
    out["pearson_r"] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    out["sign_agreement"] = float(np.mean(np.sign(got) == np.sign(ref)))
    # second reference: the same weights in fp16 arithmetic on the host, to show
    # whether the residual is fp16 rounding or something else
    out["fp16_roundtrip_vs_ref"] = stats(ref.astype(np.float16).astype(np.float64), ref)
else:
    out["device_vs_ref"] = None

i4 = inp.reshape(1, 3, 64, 64)
near = np.repeat(np.repeat(i4, 2, axis=2), 2, axis=3).reshape(-1)
out["ref_vs_nearest2x"] = stats(near, ref)

print(json.dumps(out, indent=1))
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write(json.dumps(out, indent=1))
