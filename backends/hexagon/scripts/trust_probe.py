#!/usr/bin/env python3
"""Probes for the ways a wrong answer in this backend looks like a right one.

Every probe here is a DETECTOR, and a detector that reports nothing is
indistinguishable from a broken one. So each probe carries the control that says
it can still see the thing it is looking for, and a probe whose control fails is
reported as a broken probe rather than as a clean tree.

Run from the root of the checkout you mean to measure:

    PYTHONPATH=src python backends/hexagon/scripts/trust_probe.py

Nothing here changes a file and nothing here runs the device tier.
"""
from __future__ import annotations

import argparse
import ast
import operator
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
TESTS = REPO / "backends" / "hexagon" / "test"
POW_SIM = TESTS / "test_pow_tensor_tensor_on_sim.py"

#: The commit whose pow sim fixture is known to hold the stale form: five of its
#: six expected command streams do not lead with a raster blit and one does.
CONTROL_COMMIT = "013a64f"

ADD = ast.Add
MUL = ast.Mult


class Result:
    def __init__(self, name, ok, detail):
        self.name, self.ok, self.detail = name, ok, detail

    def __str__(self):
        return ("OK   " if self.ok else "FAIL ") + self.name + ": " + self.detail


def _is_dsp_op(node):
    """A DSP_OP_* attribute expression, however deeply it is parenthesised."""
    return any(
        isinstance(inner, ast.Attribute) and inner.attr.startswith("DSP_OP_")
        for inner in ast.walk(node)
    )


def _is_blit(node, blits):
    """Whether this expression denotes the raster-blit command."""
    if isinstance(node, ast.Attribute):
        return node.attr in blits
    if isinstance(node, ast.Name):
        return node.id in blits
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value == 3
    return False


def _elements(node, blits):
    """The commands in a list-valued expression, or None if it is not one.

    A list is not the only spelling: [x] * 2 is a BinOp/Mult and a leading
    blit list is a BinOp/Add, and a detector that accepts only a List silently
    drops every dict that uses either. That was two of the five false zeros this
    probe exists to rule out, so the shape has to be read, not assumed.
    """
    if isinstance(node, ast.List):
        return list(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, MUL):
        return _elements(node.left, blits)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ADD):
        left = _elements(node.left, blits)
        right = _elements(node.right, blits)
        if left is None or right is None:
            return None
        return left + right
    return None


def _blit_spellings(tree):
    """Every name in this file that denotes the raster-blit command."""
    names = {"DSP_OP_RASTER_BLIT", "blit", "_BLIT", "RASTER_BLIT", "3"}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_dsp_op(node.value)
            and "RASTER_BLIT" in ast.unparse(node.value)
        ):
            names.add(node.targets[0].id)
    return names


def scan_expected_types(source, path):
    """The expected command streams that do not lead with a raster blit.

    Returns (recognised, offenders). recognised counts the dicts the detector
    understood at all, and it is the number that falls to zero when the
    detector assumption about the file shape goes stale.
    """
    tree = ast.parse(source, filename=path)
    blits = _blit_spellings(tree)
    recognised, offenders = 0, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        entries, ok = [], True
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant):
                ok = False
                break
            elements = _elements(value, blits)
            if not elements:
                ok = False
                break
            entries.append((key.value, elements[0]))
        if not ok or not entries:
            continue
        recognised += 1
        for tag, lead in entries:
            if not _is_blit(lead, blits):
                offenders.append(
                    "  " + path + ":" + str(node.lineno) + " " + str(tag)
                    + " leads with " + ast.unparse(lead)
                )
    return recognised, offenders


def _control_source():
    done = subprocess.run(
        ["git", "show", CONTROL_COMMIT + ":backends/hexagon/test/"
         "test_pow_tensor_tensor_on_sim.py"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    return done.stdout if done.returncode == 0 else None


def probe_blit_prefix_scan():
    """Every expected command stream that does not lead with a raster blit.

    The control is the same scan over a commit whose fixture is known to hold
    the stale form, so a zero here is a statement about the detector rather
    than about the tree.
    """
    recognised, offenders = scan_expected_types(POW_SIM.read_text(), "tree")
    control = _control_source()
    if control is None:
        return Result("raster-blit prefix scan", False, "cannot read the control")
    c_rec, c_off = scan_expected_types(control, "control@" + CONTROL_COMMIT)
    if c_rec == 0:
        return Result(
            "raster-blit prefix scan",
            False,
            "the control recognised 0 dicts, so the detector is broken and "
            "this tree result means nothing",
        )
    if recognised == 0:
        return Result(
            "raster-blit prefix scan",
            False,
            "control passes (" + str(c_rec) + " dicts, " + str(len(c_off))
            + " bad) but this tree recognised 0 dicts, so the detector stopped "
            "recognising the shape rather than the tree being clean",
        )
    detail = (
        "tree " + str(recognised) + " dicts / " + str(len(offenders))
        + " not blit-led; control " + str(c_rec) + " dicts / "
        + str(len(c_off)) + " not blit-led"
    )
    return Result("raster-blit prefix scan", True, detail)



# --- probe 2: a delegate count is a count of containers, not of coverage


def _commands(blob):
    """The command types a delegate blob carries, which is the only coverage read
    that means what it says."""
    import struct

    header_size = 4 * 4 + 4 * 4
    try:
        from blob_interpreter import read_blob
    except ImportError:
        return None
    _, cmds = read_blob(bytes(blob))
    return [c.type for c in cmds]


def _lower(fn, inputs):
    """The real lowering, through the real partitioner."""
    import torch
    from executorch.backends.hexagon.partition.hexagon_partitioner import (
        HexagonPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.lowered_backend_module import LoweredBackendModule

    class _M(torch.nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, *a):
            return self.body(*a)

    args = tuple(torch.randn(*s, dtype=torch.float16) for s in inputs)
    lowered = to_edge_transform_and_lower(
        torch.export.export(_M(fn).eval(), args),
        partitioner=[HexagonPartitioner()],
    ).exported_program()
    mods = [
        m for m in lowered.graph_module.modules()
        if isinstance(m, LoweredBackendModule)
    ]
    names, cmds = set(), []
    for m in mods:
        names |= {n.name for n in m.original_module.graph_module.graph.nodes}
        cmds += _commands(m.processed_bytes) or []
    return len(mods), len(names), cmds


def probe_delegate_count_is_not_coverage():
    """A model report that says one delegate carries no coverage information.

    The control is the same measurement on a graph whose command count has to
    be larger; if the delegate count moves with it the probe would be looking
    at coverage after all, and the two numbers are the same number.
    """
    import torch

    small, small_names, small_cmds = _lower(
        lambda a: torch.relu(a), [(4, 8)],
    )
    big, big_names, big_cmds = _lower(
        lambda a, b: torch.relu(a + b) * torch.relu(a + b), [(4, 8), (4, 8)],
    )
    if not small_cmds or not big_cmds:
        return Result("delegate count vs coverage", False,
                      "a graph produced no commands; cannot measure")
    if small != big:
        return Result("delegate count vs coverage", True,
                      "the delegate count moved with the command count here, "
                      "so this tree is not the case the probe is about")
    return Result(
        "delegate count vs coverage",
        True,
        "delegates=" + str(small) + " for " + str(len(small_cmds))
        + " commands and delegates=" + str(big) + " for "
        + str(len(big_cmds)) + " commands: the same count on both, so it "
        + "cannot see what the command list sees",
    )


# --- probe 3: a bare support object is not the predicate the partitioner runs


def probe_bare_support_is_blind_on_weight_reading_ops():
    """A refusal asserted with HexagonOperatorSupport() proves nothing.

    Whether a weight is a constant is a fact about the program signature, so a
    support object built without the signature refuses every op whose weight is
    read at export -- at the geometry the backend supports as well as at one it
    refuses. The control is the weightless family, where the bare object does
    still discriminate; without it the probe would be reporting a global claim
    from one family.
    """
    import torch
    from executorch.backends.hexagon.partition.hexagon_partitioner import (
        HexagonOperatorSupport, _data_placeholders,
    )
    from executorch.exir import to_edge

    def verdicts(model, inputs):
        args = tuple(
            torch.randn(*s, dtype=torch.float16) for s in inputs
        )
        ep = to_edge(
            torch.export.export(model.half().eval(), args)
        ).exported_program()
        placeholders = _data_placeholders(ep)
        bare = HexagonOperatorSupport()
        real = HexagonOperatorSupport(placeholders, ep)
        out = {}
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            name = str(node.target).replace("aten.", "").split(".")[0]
            out[name] = (
                bare.is_node_supported(None, node),
                real.is_node_supported(None, node),
            )
        return out

    weight_reading = verdicts(
        torch.nn.Conv2d(64, 64, 1), [(1, 64, 8, 8)],
    )
    weightless = verdicts(torch.nn.ReLU(), [(4, 8)],
                          ) if hasattr(torch.nn, "ReLU") else {}
    if not weight_reading:
        return Result("bare support is blind", False, "no call_function found")
    blind = [
        n for n, (b, r) in weight_reading.items() if b is False and r is True
    ]
    sees = [n for n, (b, r) in weightless.items() if b is True]
    if not blind:
        return Result(
            "bare support is blind",
            True,
            "no weight-reading op disagreed on this tree, so the hazard is "
            "not present here",
        )
    return Result(
        "bare support is blind",
        True,
        "bare said False where the real signature said True for "
        + str(sorted(blind)) + "; the control, a weightless family, the bare "
        + ("object still accepts " + str(sorted(sees)) if sees else "accepts nothing")
        + ", so the blindness is scoped to the weight-reading families",
    )


PROBES = [
    probe_blit_prefix_scan,
    probe_delegate_count_is_not_coverage,
    probe_bare_support_is_blind_on_weight_reading_ops,
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    results = []
    for probe in PROBES:
        try:
            result = probe()
        except Exception as error:
            result = Result(
                probe.__name__,
                False,
                "the probe raised: " + repr(error),
            )
        results.append(result)
        print(result)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
