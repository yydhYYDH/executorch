#!/usr/bin/env python3
"""Download the three FunASR models this repro needs from ModelScope.

The Paraformer checkpoint is fixed: the shipped per-layer scaling constants in
``calib/`` were measured against exactly this snapshot's weights, so a different
revision would invalidate them (re-run ``export/calibrate.py`` instead).
"""

import argparse
from pathlib import Path

MODELS = {
    "paraformer": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--into", type=Path, default=Path.home() / "models")
    parser.add_argument("--revision", default="master")
    args = parser.parse_args()
    args.into.mkdir(parents=True, exist_ok=True)

    from modelscope import snapshot_download

    for name, model_id in MODELS.items():
        print(f"== {name}: {model_id}", flush=True)
        path = Path(snapshot_download(model_id, revision=args.revision))
        link = args.into / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path)
        print(f"   -> {link} -> {path}", flush=True)
    print(f"\nconfig.sh points at $MODELS_DIR/{{paraformer,fsmn-vad,campplus}}.")


if __name__ == "__main__":
    main()
