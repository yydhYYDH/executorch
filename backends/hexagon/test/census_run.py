"""The coverage census, measured on this tree and reproducible from it.

Three things it reports, and how each one is obtained.

The row census imports the row tables out of test_overload_census.py and
test_overload_census2.py, exports every row graph and asks
HexagonOperatorSupport about every node, so a verdict is read off the tree own
decision rather than read off the table. A row declared verdict and its measured
verdict are both kept, because a census that only prints the declaration cannot
tell a coverage change from an edit to the table.

The clause census rewrites _verdict source so every "return False" records the
line it happens on, and execs it in the module own namespace, so a refused node
is attributed by the partitioner own code in the partitioner own order.

Parts are summed to the total by the addition itself, and the sum is printed
beside the total, because a parts-to-total that nobody summed is the failure
this file exists to make hard.

Host tier: export, to_edge, the support predicate. No kernel runs, hexagon-sim
does not run, and no phone is involved.
"""

import collections
import inspect
import json
import sys
import textwrap
import weakref

import torch
from torch.export import export

from executorch.backends.hexagon import hexagon_ops
from executorch.backends.hexagon.partition import hexagon_partitioner as hpart
from executorch.backends.hexagon.partition.hexagon_partitioner import (
    _data_placeholders,
    HexagonOperatorSupport,
)
from executorch.exir import EdgeCompileConfig, to_edge

import test_overload_census as C1
import test_overload_census2 as C2

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

    def install(self):
        start = self.original.__code__.co_firstlineno
        out = []
        for offset, line in enumerate(
            textwrap.dedent(inspect.getsource(self.original)).splitlines()
        ):
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
        hpart.HexagonOperatorSupport._verdict = ns["_verdict"]

    def refresh(self):
        self._ns.update(vars(hpart))

    def remove(self):
        hpart.HexagonOperatorSupport._verdict = self.original


def name_of(target):
    return getattr(target, "__name__", str(target))


def _has_emitter(name):
    return any(name_of(key) == name for key in hexagon_ops.EMITTERS)


def measured_verdict(name, accepted_names):
    """The tree own answer for one target name in one graph."""
    if not _has_emitter(name):
        return "unwired"
    return "wired" if name in accepted_names else "refused"


def _judge(inst, module, inputs):
    """{target: (measured verdict, clause line)} for one exported graph."""
    inst.refresh()
    _REFUSED.clear()
    program = to_edge(export(module, inputs), compile_config=CFG).exported_program()
    support = HexagonOperatorSupport(_data_placeholders(program))
    seen, accepted, clause = [], set(), {}
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        nm = name_of(node.target)
        seen.append(nm)
        if support.is_node_supported({}, node):
            accepted.add(nm)
        else:
            clause[nm] = _REFUSED.get(node)
    return seen, {nm: (measured_verdict(nm, accepted), clause.get(nm)) for nm in seen}


def _verdicts_for(expect, table):
    out = []
    for nm, declared in expect:
        measured, cl = table.get(nm, (None, None))
        out.append(
            {
                "target": nm,
                "declared": declared,
                "measured": measured,
                "clause": cl,
                "agrees": measured is not None and declared in (measured,)
                or (measured == "wired" and declared == "accepted"),
            }
        )
    return out


def row_census(inst):
    out = []
    for source, rows in (
        ("test_overload_census", C1._ROWS),
        ("test_overload_census2", C2._ROWS),
    ):
        for row in rows:
            label, forward, inputs, expect = row
            try:
                seen, table = _judge(inst, C1._M(forward), inputs)
            except Exception as exc:
                out.append(
                    {
                        "source": source,
                        "label": label,
                        "error": repr(exc)[:200],
                        "verdicts": [],
                    }
                )
                continue
            out.append(
                {
                    "source": source,
                    "label": label,
                    "targets_match": seen == [n for n, _ in expect],
                    "verdicts": _verdicts_for(expect, table),
                }
            )
    for label, scheme, m, k, n, expect in C2._QUANTIZED_ROWS:
        model = C2._quantized_model(C2._Mm(k, n).eval(), scheme, (torch.randn(m, k),))
        inputs = (torch.randn(m, k),)
        try:
            seen, table = _judge(inst, model, inputs)
        except Exception as exc:
            out.append(
                {
                    "source": "_QUANTIZED_ROWS",
                    "label": label,
                    "error": repr(exc)[:200],
                    "verdicts": [],
                }
            )
            continue
        out.append(
            {
                "source": "_QUANTIZED_ROWS",
                "label": label,
                "targets_match": seen == [x for x, _ in expect],
                "verdicts": _verdicts_for(expect, table),
            }
        )
    return out


def qwen3(layers, vocab=151936, hidden=1024, inter=3072, heads=16, kv=8, head_dim=128):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        max_position_embeddings=4096,
        use_cache=False,
        attn_implementation="eager",
    )
    return (
        Qwen3ForCausalLM(config).eval().to(torch.float16),
        (torch.randint(0, 1024, (1, 32)),),
    )


def model_census(inst):
    import model_census_corpus as mc

    graphs = [(k, mc.REGISTRY[k]) for k in mc.REGISTRY]
    graphs.append(("qwen3_1L", lambda: qwen3(1)))
    graphs.append(("qwen3_28L", lambda: qwen3(28)))
    out, bounds = [], []
    for key, builder in graphs:
        inst.refresh()
        entry = builder()
        if entry is None:
            continue
        module, inputs = entry
        try:
            ep = hpart.HexagonPartitioner().transform_for_pre_decomposition(
                export(module, inputs)
            )
            program = to_edge(ep, compile_config=CFG).exported_program()
        except Exception as exc:
            out.append({"graph": key, "unexportable": repr(exc)[:300], "nodes": []})
            bounds.append({"graph": key, "calls": 0, "refused": 0, "accepted": 0})
            print("unexportable", key, repr(exc)[:120], flush=True)
            continue
        support = HexagonOperatorSupport(_data_placeholders(program), program)
        calls = refused = accepted = 0
        nodes = []
        for node in program.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            calls += 1
            nm = name_of(node.target)
            in_table = bool(node.target in hpart.SUPPORTED_TARGETS)
            if support.is_node_supported({}, node):
                accepted += 1
                verdict = "wired" if in_table else "unwired"
                cl = None
            else:
                refused += 1
                verdict = "refused"
                cl = _REFUSED.get(node)
            nodes.append(
                {"target": nm, "verdict": verdict, "clause": cl, "in_table": in_table}
            )
        bounds.append(
            {"graph": key, "calls": calls, "refused": refused, "accepted": accepted}
        )
        out.append({"graph": key, "nodes": nodes})
        print("done", key, calls, refused, flush=True)
    return out, bounds


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "census.json"
    inst = Instrumented()
    inst.install()
    try:
        print("== rows ==", flush=True)
        rows = row_census(inst)
        print("== models ==", flush=True)
        models, bounds = model_census(inst)
    finally:
        inst.remove()

    declared = collections.Counter()
    measured = collections.Counter()
    disagreements = []
    total_verdicts = 0
    for row in rows:
        for v in row.get("verdicts", []):
            declared[v["declared"]] += 1
            measured[v["measured"]] += 1
            total_verdicts += 1
            if not v["agrees"]:
                disagreements.append(
                    {
                        "source": row["source"],
                        "label": row["label"],
                        "target": v["target"],
                        "declared": v["declared"],
                        "measured": v["measured"],
                        "clause": v["clause"],
                    }
                )

    clause_counts = collections.Counter()
    model_buckets = collections.Counter()
    unwired_targets = collections.Counter()
    unattributed = 0
    total_refused = 0
    total_calls = 0
    for entry in models:
        for n in entry.get("nodes", []):
            model_buckets[n["verdict"]] += 1
            if n["verdict"] == "refused":
                total_refused += 1
                if n["clause"] is None:
                    unattributed += 1
                else:
                    clause_counts[n["clause"]] += 1
            elif n["verdict"] == "unwired":
                unwired_targets[n["target"]] += 1
    for b in bounds:
        total_calls += b["calls"]

    parts_sum = sum(declared.values())
    model_parts_sum = sum(model_buckets.values())
    clause_sum = sum(clause_counts.values())

    result = {
        "clauses_instrumented": inst.lines,
        "rows": rows,
        "row_count": len(rows),
        "verdict_count": total_verdicts,
        "declared_buckets": dict(declared),
        "measured_buckets": dict(measured),
        "declared_parts_sum": parts_sum,
        "declared_total": total_verdicts,
        "declared_sums": parts_sum == total_verdicts,
        "disagreements": disagreements,
        "models": models,
        "model_bounds": bounds,
        "model_buckets": dict(model_buckets),
        "model_parts_sum": model_parts_sum,
        "clause_counts": dict(clause_counts),
        "clause_parts_sum": clause_sum,
        "total_refused": total_refused,
        "unattributed": unattributed,
        "unwired_targets": dict(unwired_targets),
        "distinct_unwired_targets": len(unwired_targets),
        "total_call_function": total_calls,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)

    print()
    print("clauses instrumented:", len(inst.lines))
    print("rows:", len(rows), "verdicts:", total_verdicts)
    print("declared buckets:", dict(declared))
    print("declared parts sum:", parts_sum, "== total", total_verdicts, parts_sum == total_verdicts)
    print("measured buckets:", dict(measured))
    print("disagreements:", len(disagreements))
    for d in disagreements[:25]:
        print("   ", d)
    print("model call_function nodes:", total_calls)
    print("model buckets:", dict(model_buckets), "sum:", model_parts_sum)
    print("refused:", total_refused, "unattributed:", unattributed)
    print("clause parts sum:", clause_sum, "== refused", total_refused)
    print("distinct unwired targets:", len(unwired_targets))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
