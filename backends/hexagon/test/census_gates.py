# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Re-derive the partitioner's refusal census from the tree, per node and per clause.

PARTITION_GATES.md section 4 is a count of which clause in
HexagonOperatorSupport._verdict refused which node, over the geometries in
census_corpus and the designed rows of test_overload_census. Its two corpora
used to be split: the rows were in the tree and the nineteen model geometries
were not, so the section carried a limitation instead of a re-derivation. Both
are here now, and this is the driver that turns them back into numbers.

Three properties make the output something other than a list. Every "return
False" in _verdict is rewritten in the source the module itself carries, so the
clause is read out of the decision the partitioner made rather than out of a
hand-kept list of gate names that falls behind the code. A refusal the
instrument did not see is counted as unattributed and reported as such, because
a clean-looking zero here is what a broken instrument also looks like. And the
per-op rows are added up and checked against their own total, so a truncated
enumeration fails instead of reading as a census.

Run it as a script to write the JSON and print the parts-to-total arithmetic:

    PYTHONPATH="$PWD/src:$PWD/backends/hexagon/test" python -B \
      backends/hexagon/test/census_gates.py --out census.json

Host tier. Export, to_edge, and the support predicate. No kernel runs, and a
zero here says nothing about a phone's DSP.
"""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import os
import sys
import textwrap
import weakref

import torch
from torch.export import export

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import census_corpus
from executorch.backends.hexagon.partition import hexagon_partitioner as hpart
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
)
from executorch.exir import EdgeCompileConfig, to_edge

CFG = EdgeCompileConfig(_check_ir_validity=False)

_REFUSED = weakref.WeakKeyDictionary()


def _hit(line, node):
    _REFUSED[node] = line


class Instrumented:
    """_verdict with every refusal tagged by the line it happens on."""

    def __init__(self):
        self.original = hpart.HexagonOperatorSupport._verdict
        self.lines = []
        self._ns = None
        self.fn = None

    def install(self):
        start = self.original.__code__.co_firstlineno
        source = textwrap.dedent(inspect.getsource(self.original))
        out = []
        for offset, line in enumerate(source.splitlines()):
            if line.strip() == "return False":
                out.append(
                    line.replace(
                        "return False",
                        "_GATE_HIT(%d, node) ; return False" % (start + offset),
                    )
                )
                self.lines.append(start + offset)
            else:
                out.append(line)
        ns = dict(vars(hpart))
        ns["_GATE_HIT"] = _hit
        exec(compile("\n".join(out), "<gate-instrumented>", "exec"), ns)
        self._ns = ns
        self.fn = ns["_verdict"]
        hpart.HexagonOperatorSupport._verdict = self.fn

    def refresh(self):
        # The exec'd body reads module globals through the dict captured at
        # install time, so a caller that patched a gate afterwards would be read
        # past. Re-syncing keeps the instrument a view of this tree.
        self._ns.update(vars(hpart))

    def remove(self):
        hpart.HexagonOperatorSupport._verdict = self.original


def name_of(node):
    return getattr(node.target, "__name__", str(node.target))


def _shape(node):
    return str(getattr(node.meta.get("val"), "shape", None))


def _dtype(node):
    return str(getattr(node.meta.get("val"), "dtype", None))


def edge_program(module, inputs):
    """The graph the partitioner sees, not the bare exported one.

    to_edge on its own is a different graph from the one the split runs on -- on
    the four-layer stack the bare one reports one region and five refused targets
    where the real one reports nine and one -- so a census taken over the bare
    export is a census of the wrong graph.
    """
    program = hpart.HexagonPartitioner().transform_for_pre_decomposition(
        export(module, inputs)
    )
    return to_edge(program, compile_config=CFG).exported_program()


def census_graphs(graphs, out):
    """Record every call_function node's verdict, keyed by node and not by call."""
    _REFUSED.clear()
    program = edge_program(*graphs)
    support = HexagonOperatorSupport(_data_placeholders(program), program)
    seen = weakref.WeakSet()
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function" or node in seen:
            continue
        seen.add(node)
        target = name_of(node)
        in_table = node.target in hpart.SUPPORTED_TARGETS
        record = {"target": target, "in_table": in_table,
                  "shape": _shape(node), "dtype": _dtype(node)}
        if support.is_node_supported({}, node):
            out["accepted"].append(record)
        else:
            record["line"] = _REFUSED.get(node)
            out["refused"].append(record)


def build_corpus(ids_dtype=None, skip_28=False):
    """The model corpus: nineteen hand-written geometries plus the two Qwen3s."""
    for key, builder in census_corpus.REGISTRY.items():
        entry = builder()
        if entry is not None:
            yield key, entry
    for label in ("qwen3_1L", "qwen3_28L"):
        if skip_28 and label == "qwen3_28L":
            continue
        yield label, census_corpus.qwen3_from_registry(label, ids_dtype=ids_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids-dtype", default="int64", choices=("int64", "int32"))
    ap.add_argument("--skip-28", action="store_true")
    args = ap.parse_args()

    ids_dtype = None if args.ids_dtype == "int64" else torch.int32
    out = {"accepted": [], "refused": [], "unexportable": []}
    inst = Instrumented()
    inst.install()
    try:
        for label, graphs in build_corpus(ids_dtype=ids_dtype, skip_28=args.skip_28):
            inst.refresh()
            before_a, before_r = len(out["accepted"]), len(out["refused"])
            try:
                census_graphs(graphs, out)
            except Exception as exc:  # a geometry that will not export is a fact
                out["unexportable"].append({"label": label, "error": repr(exc)[:300]})
                print("unexportable", label, repr(exc)[:160], flush=True)
                continue
            for record in out["accepted"][before_a:]:
                record["label"] = label
            for record in out["refused"][before_r:]:
                record["label"] = label
            print("done", label, flush=True)
    finally:
        inst.remove()
    out["gate_lines"] = inst.lines
    out["ids_dtype"] = args.ids_dtype

    total = len(out["accepted"]) + len(out["refused"])
    by_op = {}
    for bucket in ("accepted", "refused"):
        for record in out[bucket]:
            row = by_op.setdefault(
                record["target"],
                {"total": 0, "accepted": 0, "refused": 0, "in_table": False, "lines": {}},
            )
            row["total"] += 1
            row["in_table"] = row["in_table"] or record["in_table"]
            if bucket == "accepted":
                row["accepted"] += 1
            else:
                row["refused"] += 1
                key = str(record["line"])
                row["lines"][key] = row["lines"].get(key, 0) + 1
    per_op = {k: by_op[k] for k in sorted(by_op)}
    out["per_op"] = per_op
    out["total"] = total
    out["accepted_n"] = len(out["accepted"])
    out["refused_n"] = len(out["refused"])
    out["unattributed"] = sum(1 for r in out["refused"] if r["line"] is None)

    # The parts-to-total arithmetic, run rather than asserted. A decomposition
    # with no addition on it is a list, and a truncated one reads exactly like a
    # complete one until you add it up.
    sum_total = sum(v["total"] for v in per_op.values())
    sum_accepted = sum(v["accepted"] for v in per_op.values())
    sum_refused = sum(v["refused"] for v in per_op.values())
    out["parts_to_total"] = {
        "sum_per_op_total": sum_total,
        "sum_per_op_accepted": sum_accepted,
        "sum_per_op_refused": sum_refused,
        "stated_total": total,
        "stated_accepted": len(out["accepted"]),
        "stated_refused": len(out["refused"]),
        "closes": sum_total == total
        and sum_accepted == len(out["accepted"])
        and sum_refused == len(out["refused"]),
    }

    per_clause = collections.Counter(r["line"] for r in out["refused"])
    out["per_clause"] = {str(k): v for k, v in sorted(per_clause.items(), key=lambda kv: -kv[1])}
    out["per_graph"] = dict(
        collections.Counter(r["label"] for r in out["accepted"] + out["refused"])
    )

    print()
    print("call_function nodes :", total)
    print("  accepted          :", len(out["accepted"]))
    print("  refused           :", len(out["refused"]))
    print("  unattributed      :", out["unattributed"])
    print("  unexportable      :", len(out["unexportable"]))
    print("  distinct ops      :", len(per_op))
    print("parts-to-total      :", out["parts_to_total"])
    print()
    print("per clause (line: nodes):")
    for line, n in out["per_clause"].items():
        print("  %-6s %d" % (line, n))

    if not out["parts_to_total"]["closes"]:
        print("PARTS DO NOT SUM TO TOTAL", file=sys.stderr)
        return 1
    if out["unattributed"]:
        print("UNATTRIBUTED REFUSALS: instrument missed %d" % out["unattributed"],
              file=sys.stderr)
        return 1
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
