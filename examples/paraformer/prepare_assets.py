#!/usr/bin/env python3
"""Prepare Paraformer runtime assets from a FunASR model directory."""

import argparse
import shutil
from pathlib import Path

from funasr.frontends.wav_frontend import load_cmvn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    cmvn = load_cmvn(args.model_dir / "am.mvn").numpy()
    if cmvn.shape != (2, 560):
        parser.error(f"expected a 2 x 560 CMVN, got {cmvn.shape}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmvn.tofile(args.output_dir / "cmvn.bin")
    shutil.copyfile(args.model_dir / "tokens.json", args.output_dir / "tokens.json")


if __name__ == "__main__":
    main()
