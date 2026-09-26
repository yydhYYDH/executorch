# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The shape of the backend, checked where a merge is least likely to look.

A duplicated definition does not raise. Git auto-merges a refactored function
beside the one already there, the later binding wins, and everything that used
to call the earlier one calls a different body instead -- the 1x1 convolution
moving from command 17 to command 12 with no test failing. A signature widened
on one side of a merge while the call sites on the other kept the old arity is
that damage one edit from a TypeError, and the files that raise it are not the
files the merge edited. Both are properties of the module text, so they are
read here with ast rather than inferred from a green run.

Every check carries a control that runs the same code over a source with a
known defect in it. Without one, a scan that has stopped seeing anything is
indistinguishable from a tree that is clean.
"""

import ast
import collections
import pathlib

import pytest

BACKEND = pathlib.Path(__file__).resolve().parent.parent
CONTROLLED = "_emit_dense_im2col"


def _module_scope_defs(statements, found, depth=0):
    """Definitions in statement position at module scope, control flow included.

    A plain tree.body walk misses a definition inside a top-level if, try or
    loop, and a duplicated definition is exactly what that hides. A definition
    nested inside another function is a local binding, not a shadow, and is not
    counted.
    """
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((statement.name, statement.lineno, depth))
        elif isinstance(statement, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            _module_scope_defs(statement.body, found, depth + 1)
            if statement.orelse:
                _module_scope_defs(statement.orelse, found, depth + 1)
            for handler in getattr(statement, "handlers", None) or []:
                _module_scope_defs(handler.body, found, depth + 1)


def _module_scope_rebindings(statements, names, found):
    """Module-level assignments, imports and handlers that reuse a def name."""
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                for inner in ast.walk(target):
                    if isinstance(inner, ast.Name) and inner.id in names:
                        found.append((inner.id, node.lineno))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = (alias.asname or alias.name).split(".")[0]
                    if name in names:
                        found.append((name, node.lineno))
            if isinstance(node, ast.ExceptHandler) and node.name in names:
                found.append((node.name, node.lineno))


def _duplicate_definitions(source):
    """Names defined twice in module scope, and module-level rebindings."""
    tree = ast.parse(source)
    found = []
    _module_scope_defs(tree.body, found)
    lines = collections.defaultdict(list)
    for name, line, depth in found:
        lines[name].append((line, depth))
    duplicates = {name: v for name, v in lines.items() if len(v) > 1}
    rebindings = []
    _module_scope_rebindings(tree.body, set(lines), rebindings)
    return duplicates, rebindings, len(found)


def _signatures(tree):
    table = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = [p.arg for p in args.posonlyargs + args.args]
            defaulted = len(args.defaults)
            required = positional[: len(positional) - defaulted] if defaulted else positional
            table[node.name] = dict(
                line=node.lineno,
                positional=positional,
                required=set(required),
                keyword_only={p.arg for p in args.kwonlyargs},
                variadic=args.vararg is not None or args.kwarg is not None,
            )
    return table


def _arity_mismatch(signature, call):
    """None when the call fits the signature, otherwise why it cannot."""
    if signature["variadic"] or not signature["positional"]:
        return None
    keywords = [keyword.arg for keyword in call.keywords]
    if any(name is None for name in keywords):
        return None
    taken = signature["positional"][: len(call.args)]
    problems = []
    if len(call.args) > len(signature["positional"]):
        problems.append("too many positional")
    repeated = sorted(n for n, c in collections.Counter(keywords).items() if c > 1)
    if repeated:
        problems.append("repeated keyword " + str(repeated))
    allowed = set(signature["positional"]) | signature["keyword_only"]
    unknown = sorted(n for n in keywords if n not in allowed)
    if unknown:
        problems.append("unknown keyword " + str(unknown))
    both = sorted(set(taken) & set(keywords))
    if both:
        problems.append("positionally and by keyword " + str(both))
    missing = sorted(signature["required"] - (set(taken) | set(keywords)))
    if missing:
        problems.append("missing " + str(missing))
    return "; ".join(problems) or None


def _call_sites(tree, names):
    """Calls of the given names by a bare identifier or through hexagon_ops."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in names:
            yield node.func.id, node
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "hexagon_ops"
            and node.func.attr in names
        ):
            yield node.func.attr, node


def _the_arity_check_fails_on_a_call_that_does_not_fit():
    known = (
        "def f(a, b):\n    return a\n"
        "f(1, 2, 3)\n"
        "f(1, c=2, c=3)\n"
        "f(1)\n"
        "f(1, b=2)\n"
        "f(1, 2)\n"
    )
    tree = ast.parse(known)
    table = _signatures(tree)
    verdicts = [_arity_mismatch(table["f"], call) for _, call in _call_sites(tree, {"f"})]
    assert "too many positional" in verdicts[0], verdicts[0]
    assert "repeated keyword" in verdicts[1], verdicts[1]
    assert "missing" in verdicts[2], verdicts[2]
    assert verdicts[3] is None, verdicts[3]
    assert verdicts[4] is None, verdicts[4]


def _the_definition_scan_fails_on_a_duplicate_when_one_is_there():
    known = (
        "def " + CONTROLLED + "(a, b):\n    return a\n\n"
        "if True:\n    def " + CONTROLLED + "(a, b, c, d):\n        return a\n"
    )
    duplicates, rebindings, found = _duplicate_definitions(known)
    assert CONTROLLED in duplicates, "the scan is blind"
    assert [line for line, _ in duplicates[CONTROLLED]] == [1, 5]
    assert found == 2, "the definition inside the if-statement was not reached"
    shadowed = "def g(a):\n    return a\n\ng = None\n"
    assert _duplicate_definitions(shadowed)[1], "a module-level rebind went unseen"


@pytest.mark.parametrize("path", sorted(BACKEND.rglob("*.py")), ids=str)
def test_no_module_scope_name_is_defined_twice(path):
    if "__pycache__" in str(path) or path.name == "test_module_shape_audit.py":
        pytest.skip("generated bytecode, and this file scans itself")
    _the_definition_scan_fails_on_a_duplicate_when_one_is_there()
    duplicates, rebindings, _ = _duplicate_definitions(path.read_text())
    assert not duplicates, path.name + " defines a module-scope name twice: " + str(duplicates)
    assert not rebindings, path.name + " rebinds a definition at module level: " + str(rebindings)


@pytest.mark.parametrize("path", sorted(BACKEND.rglob("*.py")), ids=str)
def test_every_call_site_fits_the_signature_it_names(path):
    if "__pycache__" in str(path) or path.name == "test_module_shape_audit.py":
        pytest.skip("generated bytecode, and this file scans itself")
    _the_arity_check_fails_on_a_call_that_does_not_fit()
    table = _signatures(ast.parse((BACKEND / "hexagon_ops.py").read_text()))
    tree = ast.parse(path.read_text())
    broken = []
    for name, call in _call_sites(tree, set(table)):
        why = _arity_mismatch(table[name], call)
        if why:
            broken.append((call.lineno, name, why))
    assert not broken, path.name + " calls hexagon_ops with the wrong arity: " + str(broken)
