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


PROBES = [probe_blit_prefix_scan]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    results = []
    for probe in PROBES:
        try:
            result = probe()
        except Exception as error:
            result = Result(probe.__name__, False, "the probe raised: " + repr(error))
        results.append(result)
        print(result)
        if not result.ok:
            for line in scan_offenders():
                print(line)
    return 0 if all(r.ok for r in results) else 1


def scan_offenders():
    recognised, offenders = scan_expected_types(POW_SIM.read_text(), "tree")
    return offenders


if __name__ == "__main__":
    sys.exit(main())
