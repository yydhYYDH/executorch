"""Which gate in HexagonOperatorSupport._verdict refuses each node.

Every "return False" inside _verdict is rewritten, in the source the module
carries, into a call that records the line it is on and the node it is deciding.
So the reason comes from the decision the partitioner actually made, in the order
it made it, and an inline clause the tree grew after this was written is recorded
like any other -- there is no hand-kept list of gate names to go stale.

Host tier only: export, to_edge, and the support predicate. Nothing runs a
kernel.
"""

import inspect
import os
import sys
import textwrap
import weakref

WORKTREE = os.environ.get("HEX_WT", "/home/yydh/executorch/.tmp/wt/gateinv/executorch")
sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.join(WORKTREE, "src"))
sys.path.insert(0, os.path.join(WORKTREE, "backends/hexagon/test"))

import torch
from torch.export import export

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
        code = self.original.__code__
        start = code.co_firstlineno
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
        body = "\n".join(out)
        ns = dict(vars(hpart))
        ns["_GATE_HIT"] = _hit
        exec(compile(body, "<gate-instrumented>", "exec"), ns)
        self._ns = ns
        self.fn = ns["_verdict"]
        hpart.HexagonOperatorSupport._verdict = self.fn

    def refresh(self):
        self._ns.update(vars(hpart))

    def remove(self):
        hpart.HexagonOperatorSupport._verdict = self.original


def name_of(node):
    return getattr(node.target, "__name__", str(node.target))


def edge_program(module, inputs):
    return to_edge(export(module, inputs), compile_config=CFG).exported_program()


def run_case(module, inputs, label, out):
    """Export one module and record the gate for every refused node."""
    _REFUSED.clear()
    try:
        program = edge_program(module, inputs)
    except Exception as exc:
        out.setdefault("unexportable", []).append((label, repr(exc)[:200]))
        return
    support = HexagonOperatorSupport(_data_placeholders(program), program)
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if support.is_node_supported({}, node):
            continue
        out.setdefault("refused", []).append(
            {
                "label": label,
                "target": name_of(node),
                "line": _REFUSED.get(node),
                "in_table": bool(node.target in hpart.SUPPORTED_TARGETS),
                "shape": str(getattr(node.meta.get("val"), "shape", None)),
                "dtype": str(getattr(node.meta.get("val"), "dtype", None)),
            }
        )
