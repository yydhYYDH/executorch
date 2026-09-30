#!/usr/bin/env python3
"""Prepare FSMN-VAD runtime assets from a FunASR model directory."""

import argparse
from pathlib import Path

from funasr.frontends.wav_frontend import load_cmvn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    cmvn = load_cmvn(args.model_dir / "am.mvn").numpy()
    if cmvn.shape != (2, 400):
        parser.error(f"expected a 2 x 400 CMVN, got {cmvn.shape}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmvn.tofile(args.output_dir / "cmvn.bin")


if __name__ == "__main__":
    main()
