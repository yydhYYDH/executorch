# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The source identity carried by a Hexagon skel detects source-only staleness.

A skel is cross-compiled and deployed separately from the runner that loads it,
so a source edit can leave an old skel on the device that still loads and still
passes the arch check. These regressions build the identity the generator emits
for the current source closure and for the historical VLA version of the
M<=32 prefill kernel, then drive the same host comparator the loader uses over
both. No device and no Hexagon SDK: the ID is produced by cmake and the
comparator is compiled by the host compiler.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest

_HEXAGON_DIR = pathlib.Path(__file__).resolve().parents[1]
_MNN_OPS_ROOT = _HEXAGON_DIR / "third-party/mnn-htp-ops"
_GENERATOR = _HEXAGON_DIR / "skel/generate_skel_source_id.cmake"
_HELPER = _HEXAGON_DIR / "skel/source_id.cmake"
_PARENT_CMAKE = _HEXAGON_DIR / "CMakeLists.txt"
_SKEL_CMAKE = _HEXAGON_DIR / "skel/CMakeLists.txt"
_COMPARATOR = _HEXAGON_DIR / "runtime/hexagon_skel_provenance.h"
_KERNEL_RELATIVE = pathlib.Path("src/dsp/ops/matmul_q4fp16_mle32.c")

_HEAP_BLOCK = re.compile(
    r"  dma_desc_2d_t \*act_descs\s*= \(dma_desc_2d_t \*\) memalign\(\n"
    r"      64, \(size_t\) safe_kp \* sizeof\(dma_desc_2d_t\)\);\n"
    r"  if \(act_descs == NULL\) \{\n"
    r"    return AEE_ENOMEMORY;\n"
    r"  \}\n"
)

_PROBE = r"""#include <cstdio>
#include <cstdlib>

#include "hexagon_skel_provenance.h"

int main(int argc, char** argv) {
  uint32_t reported[4] = {};
  for (int i = 0; i < 4 && i + 1 < argc; ++i) {
    reported[i] = static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 16));
  }
  std::printf("%s\n", skel_source_id_matches(reported) ? "accept" : "reject");
  return 0;
}
"""


def _copy_manifest_inputs(root: pathlib.Path) -> None:
    shutil.copytree(_MNN_OPS_ROOT, root / "mnn-htp-ops")
    (root / "skel").mkdir()
    shutil.copy2(_SKEL_CMAKE, root / "skel/CMakeLists.txt")
    shutil.copy2(_GENERATOR, root / "skel/generate_skel_source_id.cmake")
    shutil.copy2(_HELPER, root / "skel/source_id.cmake")
    shutil.copy2(_PARENT_CMAKE, root / "CMakeLists.txt")


def _run_generator(
    mnn_root: pathlib.Path,
    skel_cmake: pathlib.Path,
    generator: pathlib.Path,
    helper: pathlib.Path,
    parent_cmake: pathlib.Path,
    output: pathlib.Path,
    config: str,
) -> str:
    subprocess.run(
        [
            "cmake",
            f"-DMNN_OPS_ROOT={mnn_root}",
            f"-DSKEL_CMAKE={skel_cmake}",
            f"-DHEXAGON_SKEL_SOURCE_ID_GENERATOR={generator}",
            f"-DHEXAGON_SKEL_SOURCE_ID_HELPER={helper}",
            f"-DHEXAGON_SKEL_SOURCE_ID_PARENT_CMAKE={parent_cmake}",
            f"-DHEXAGON_SKEL_SOURCE_ID_CONFIG={config}",
            f"-DOUTPUT={output}",
            "-P",
            str(generator),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output.read_text()


def _generate(
    root: pathlib.Path, output: pathlib.Path, config: str = "test-toolchain"
) -> str:
    return _run_generator(
        root / "mnn-htp-ops",
        root / "skel/CMakeLists.txt",
        root / "skel/generate_skel_source_id.cmake",
        root / "skel/source_id.cmake",
        root / "CMakeLists.txt",
        output,
        config,
    )


def _generate_checkout(output: pathlib.Path, config: str = "test-toolchain") -> str:
    return _run_generator(
        _MNN_OPS_ROOT,
        _SKEL_CMAKE,
        _GENERATOR,
        _HELPER,
        _PARENT_CMAKE,
        output,
        config,
    )


def _source_id_words(header: str) -> tuple[int, int, int, int]:
    words = re.findall(r"HEXAGON_SKEL_SOURCE_ID_([0-3]) 0x([0-9a-f]{8})u", header)
    assert [word[0] for word in words] == ["0", "1", "2", "3"], header
    values = tuple(int(word[1], 16) for word in words)
    text = re.search(r'HEXAGON_SKEL_SOURCE_ID_TEXT "([0-9a-f]{32})"', header)
    assert text is not None, header
    assert text.group(1) == "".join(f"{value:08x}" for value in values), header
    return values


def _make_vla_source(root: pathlib.Path) -> None:
    kernel = root / "mnn-htp-ops" / _KERNEL_RELATIVE
    source, allocated = _HEAP_BLOCK.subn(
        "  _Alignas(64) dma_desc_2d_t act_descs[safe_kp];\n", kernel.read_text()
    )
    source, freed = re.subn(r"  free\(act_descs\);\n", "", source)
    assert allocated == 1 and freed == 1, "the current heap allocation was not found"
    assert "_Alignas(64) dma_desc_2d_t act_descs[safe_kp];" in source
    kernel.write_text(source)


def _build_comparator(tmp_path: pathlib.Path, expected_header: pathlib.Path) -> pathlib.Path:
    cxx = os.environ.get("CXX", "c++")
    if shutil.which(cxx) is None:
        pytest.skip(f"no host C++ compiler ({cxx})")
    probe = tmp_path / "probe.cpp"
    probe.write_text(_PROBE)
    binary = tmp_path / "probe"
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            f"-I{_COMPARATOR.parent}",
            f"-I{expected_header.parent}",
            str(probe),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def _verdict(binary: pathlib.Path, header: str) -> str:
    words = _source_id_words(header)
    return subprocess.run(
        [str(binary), *(f"{word:08x}" for word in words)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_the_loader_comparator_accepts_the_current_skel(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "current"
    _copy_manifest_inputs(root)
    header_path = tmp_path / "idl" / "hexagon_skel_source_id.h"
    header_path.parent.mkdir()
    header = _generate(root, header_path)
    again = _generate(root, tmp_path / "again.h")

    assert header == again
    assert _verdict(_build_comparator(tmp_path, header_path), header) == "accept"


def test_the_loader_comparator_rejects_a_skel_built_from_the_vla_source(
    tmp_path: pathlib.Path,
) -> None:
    current_root = tmp_path / "current"
    historical_root = tmp_path / "historical"
    _copy_manifest_inputs(current_root)
    _copy_manifest_inputs(historical_root)
    _make_vla_source(historical_root)

    header_path = tmp_path / "idl" / "hexagon_skel_source_id.h"
    header_path.parent.mkdir()
    current = _generate(current_root, header_path)
    historical = _generate(historical_root, tmp_path / "historical.h")
    binary = _build_comparator(tmp_path, header_path)

    assert _source_id_words(current) != _source_id_words(historical)
    assert _verdict(binary, historical) == "reject"
    assert _verdict(binary, current) == "accept"


def test_the_source_id_ignores_where_the_tree_lives_but_not_how_it_is_built(
    tmp_path: pathlib.Path,
) -> None:
    first_root = tmp_path / "first+root"
    second_root = tmp_path / "second-root"
    _copy_manifest_inputs(first_root)
    _copy_manifest_inputs(second_root)

    checkout = _source_id_words(_generate_checkout(tmp_path / "checkout.h"))
    first = _source_id_words(_generate(first_root, tmp_path / "first.h"))
    second = _source_id_words(_generate(second_root, tmp_path / "second.h"))
    other_toolchain = _source_id_words(
        _generate(first_root, tmp_path / "other.h", config="other-toolchain")
    )

    # Same bytes, three different directory layouts, one identity.
    assert first == second == checkout
    assert first != other_toolchain
