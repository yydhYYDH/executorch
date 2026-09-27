"""Device-vs-reference comparison for the TTS CodecDecoder run.

Follows the method the EDSR line established (compare_sr_device.py): the
measurement alone is not evidence, so four controls sit beside it.

  * device vs reference        -- the measurement
  * perturbed reference        -- sensitivity: proves the comparison CAN see a
                                  difference, so a high number is not vacuous
  * fp16 roundtrip of the ref  -- the fp16 ceiling, which separates "this is
                                  what fp16 costs" from "the DSP is wrong"
  * constant-mean vs reference  -- the trivial model, for scale
  * second input on the device -- input dependence.  Skill 8.98 is the reason:
                                  CampPlus returned a byte-identical output for
                                  two different inputs under a clean
                                  "exit d0: ok", so a green exit plus a high
                                  PSNR would still not establish that the
                                  waveform was computed FROM the codes.

usage: python compare_tts_device.py <artdir> <device_out.bin> [device_out2.bin] [out.json]
"""
import json
import os
import sys

import numpy as np

ART = sys.argv[1]
DEV = sys.argv[2] if len(sys.argv) > 2 else None
DEV2 = sys.argv[3] if len(sys.argv) > 3 else None

ref = np.load(os.path.join(ART, "tts_ref.npy")).astype(np.float64).reshape(-1)
inp = np.fromfile(os.path.join(ART, "tts_in0.bin"), dtype=np.int64).reshape(-1)


def stats(got, r):
    g = np.asarray(got, dtype=np.float64)
    d = np.abs(g - r)
    mse = float(np.mean(d ** 2))
    peak = float(np.max(np.abs(r)))
    psnr = 10.0 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")
    a = g - g.mean()
    b = r - r.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return {
        "n": int(g.size),
        "max_abs": float(d.max()), "mean_abs": float(d.mean()),
        "rmse": float(np.sqrt(mse)), "mse": mse, "peak_ref": peak, "psnr_db": psnr,
        "exact_frac": float(np.mean(d == 0.0)),
        "ref_absmax": float(np.max(np.abs(r))),
        "got_absmax": float(np.max(np.abs(g))),
        "got_nan": int(np.sum(~np.isfinite(g))), "ref_nan": int(np.sum(~np.isfinite(r))),
        "got_inf": int(np.sum(np.isinf(g))),
        "pearson_r": float(np.dot(a, b) / denom) if denom > 0 else None,
        "sign_agreement": float(np.mean(np.sign(g) == np.sign(r))),
    }


out = {
    "ref_numel": int(ref.size),
    "ref_absmax": float(np.max(np.abs(ref))),
    "ref_rms": float(np.sqrt(np.mean(ref ** 2))),
    "input_numel": int(inp.size),
    "input_dtype": "int64",
    "input_bytes": int(inp.size) * 8,
    "input_min": int(inp.min()), "input_max": int(inp.max()),
}

if DEV and os.path.exists(DEV):
    raw = open(DEV, "rb").read()
    out["device_bytes"] = len(raw)
    out["device_float32_count"] = len(raw) // 4
    out["device_bytes_match_4x_numel"] = (len(raw) == 4 * ref.size)
    got = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    out["device_vs_ref"] = stats(got, ref) if got.size == ref.size else "SIZE MISMATCH"
    bad = ref.copy()
    bad[7] += 8.0 * float(np.max(np.abs(ref)))
    out["control_perturbed_ref"] = stats(got, bad)
    out["fp16_roundtrip_vs_ref"] = stats(ref.astype(np.float16).astype(np.float64), ref)
    out["constant_mean_vs_ref"] = stats(np.full_like(ref, float(ref.mean())), ref)
    if DEV2 and os.path.exists(DEV2):
        raw2 = open(DEV2, "rb").read()
        got2 = np.frombuffer(raw2, dtype=np.float32).astype(np.float64)
        out["second_input"] = {
            "bytes": len(raw2),
            "byte_identical_to_first": bool(raw == raw2),
            "psnr_between_the_two_device_runs": stats(got2, got).get("psnr_db"),
            "max_abs_difference_between_runs": float(np.max(np.abs(got2 - got))),
        }
else:
    out["device_vs_ref"] = None

print(json.dumps(out, indent=1))
if len(sys.argv) > 4:
    open(sys.argv[4], "w").write(json.dumps(out, indent=1))
