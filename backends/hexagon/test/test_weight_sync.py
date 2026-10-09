# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_weight_sync_policy(tmp_path):
    cxx = os.environ.get("CXX", "c++")
    if shutil.which(cxx) is None:
        pytest.skip(f"no host C++ compiler ({cxx})")
    repo = Path(__file__).resolve().parents[3]
    binary = tmp_path / "weight_sync_probe"
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{repo.parent}",
            str(Path(__file__).with_name("weight_sync_probe.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True)
