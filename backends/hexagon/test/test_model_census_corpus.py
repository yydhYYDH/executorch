# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The nineteen census geometries are in the tree, and this is what pins that.

The coverage census counts nodes the Hexagon partitioner refuses, and it is
sized by a model corpus. While that corpus lived in a scratch directory the
numbers were re-derivable only by the person who had written them down, which
is the one reader a census must not need. These tests do not re-run the census
-- that costs minutes and belongs to a deliberate invocation -- they pin the
property that makes it re-runnable at all: every geometry in the registry
builds, the registry and its descriptions agree, and the declared verdict
buckets sum to the number of verdicts rather than to a number typed beside
them.

Host tier: building the modules and reading the row tables. No export, no
kernel, no simulator, no phone.
"""

import collections

import model_census_corpus as mc
import test_overload_census as C1
import test_overload_census2 as C2

VERDICTS = {"wired", "refused", "unwired", "accepted"}


def _declared() -> "collections.Counter":
    counts = collections.Counter()
    for rows in (C1._ROWS, C2._ROWS, C2._QUANTIZED_ROWS):
        for row in rows:
            for pair in row[-1]:
                counts[pair[1]] += 1
    return counts


def test_every_registered_geometry_builds():
    """A geometry that cannot be built is a census that cannot be re-derived."""
    for name, builder in mc.REGISTRY.items():
        entry = builder()
        assert entry is not None, name
        module, inputs = entry
        assert isinstance(inputs, tuple) and inputs, name
        assert any(hasattr(t, "shape") for t in inputs), name


def test_the_registry_and_its_descriptions_are_the_same_nineteen():
    assert len(mc.REGISTRY) == 19
    assert set(mc.REGISTRY) == set(mc.DESCRIPTIONS)
    assert len(mc.PASSES) <= len(mc.REGISTRY)
    assert set(mc.PASSES) <= set(mc.REGISTRY)


def test_the_declared_buckets_sum_to_the_verdicts():
    """Parts to total, added here rather than asserted beside a header.

    A scan that finds nothing and a scan that is broken return the same thing,
    so a census is only worth reading once the addition has been done and shown.
    """
    counts = _declared()
    total = sum(len(row[-1]) for row in C1._ROWS) + sum(len(row[-1]) for row in C2._ROWS)
    total += sum(len(row[-1]) for row in C2._QUANTIZED_ROWS)
    assert set(counts) <= VERDICTS, counts
    assert sum(counts.values()) == total, (dict(counts), total)
