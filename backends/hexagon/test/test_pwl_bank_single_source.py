# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where the companded16 chord table is transcribed, and why exactly once.

The sixteen fp16 chords of sigmoid are a transcription, not a derivation: they are
the skel's own numbers (unary_ops.cc:225-237, the HTP_OPS_PWL_COMPANDED16 branch),
read out of the C++ and written into Python so the host interpreter answers the
same arithmetic the DSP does. blob_interpreter.py holds that transcription, under
_PWL_SLOPE[4] and _PWL_BIAS[4]. It did not always hold the only one: test_glu.py
carried a private copy under _SIGMOID_SLOPE and _SIGMOID_BIAS, to evaluate the band
a GLU's gate hands the product the file is named for.

The two copies were element-wise identical, which is the least interesting thing
about them. What matters is what the tree could see, measured by moving one copy
by a single ulp and running the four files that talk about sigmoid:

    mutated                                    test_glu  interp_sig  grain  unary_pwl
    test_glu.py private bank, one ulp down          0          0        0        0
    blob_interpreter bank, one ulp down              0          0        1        0

Both rows read zero in the first column, and that is the whole finding: the
private copy was checked by nothing, including the file that held it, and the bank
the interpreter runs was checked by one file and not by the one that re-stated it.
test_glu.py had a table and assertions about a table, and neither of them was about
the table the interpreter runs. Drift in a transcription nobody reads is not a
failing test, it is nothing at all, and a comment recording that the two agree
would go on recording it for one revision.

So the four tests here are a reader, two controls and an anchor, not a statement:

- test_the_reader_finds_the_banks_it_is_meant_to_find is the positive control. A
  reader that finds nothing and a reader that is broken return the same thing, and
  a guard built on one has to be shown finding what this file says is there before
  its zero can be read as a clean tree.
- test_a_copy_that_is_put_back_is_found is the guard run red, over a tree built in
  tmp_path, so the second-copy check below is known to be able to fail.
- test_the_sigmoid_chords_are_transcribed_once is the guard over this tree. It is
  the half fb0c63f asked for that _PWL_UNTRANSCRIBED could not be: that dict
  refuses a second *transcription* only inside blob_interpreter.py, and a copy
  inside a test file is a different file with a different owner.
- test_the_transcribed_bank_is_the_table_in_unary_ops_cc is the anchor, and it is
  what makes the guard's zero mean something: the one definition left in the tree
  and the literals the skel is built with are the same sixteen values.
- test_glu_evaluates_the_interpreters_bank_and_agrees_with_it_bit_for_bit keeps
  the file that needed the table honest about the one thing it is still free to
  get wrong on its own.

The index arithmetic is still written out in each test file, and that is not the
same mistake. test_sigmoid_grain.py and test_interp_sigmoid.py both re-derive the
companded index and both say why: a rounding case has to be able to tell a wrong
index from a wrong rounding, and reusing the index of the module under test lets
one hide inside the other. A table is a fact and is transcribed once; an index is a
claim and is checked twice. The line runs between those, not between files.
"""

import ast
import pathlib
import re

import numpy as np

from blob_interpreter import _PWL_BIAS, _PWL_SLOPE, _pwl_body

_TEST_DIR = pathlib.Path(__file__).resolve().parent
_INTERPRETER = _TEST_DIR / "blob_interpreter.py"

#: HTP_OPS_UNARY_SIGMOID. A GLU is not a GELU: F.glu decomposes into two
#: slice_copy, a sigmoid and a mul, and the single unary command in the delegate
#: carries this op type, which is why a GLU test needs a sigmoid bank at all. Read
#: out of a real delegate rather than assumed: a GLU blob over [1, 64, 1, 48]
#: carries command types [3, 3, 4, 19, 3], and the unary is numel 1536, op_type 4.
#: That also settles the name. _SIGMOID_SLOPE is not a GELU bank that happens to hold
#: sigmoid's numbers, it is sigmoid's bank, correctly named, and the confusion to
#: avoid here is GLU read as GELU.
_SIGMOID = 4

#: The same table in the skel's own source, which is where it is transcribed from.
_UNARY_OPS = (
    _TEST_DIR.parents[0]
    / "third-party"
    / "mnn-htp-ops"
    / "src"
    / "dsp"
    / "unary_ops.cc"
)


def _int_seq(node):
    """Every int a literal expression denotes, or None when it is not all ints.

    Handles the two shapes a bank is written in here: a tuple of hex literals, and
    one padded with `(0,) * 4` for the four slots a twelve-entry table leaves empty.
    A name, a call or a string yields None rather than a guess, so a definition the
    reader cannot read exactly is not read at all -- which is the one failure mode
    that would let a copy through.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            return None
        return (node.value,)
    if isinstance(node, ast.Tuple):
        parts = [_int_seq(element) for element in node.elts]
        if any(part is None for part in parts):
            return None
        return tuple(part for group in parts for part in group)
    if isinstance(node, ast.BinOp):
        left, right = _int_seq(node.left), _int_seq(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mult) and len(right) == 1:
            return left * right[0]
    return None


def _bank_bits(node):
    """The uint16 bit patterns a definition denotes, however it spells them.

    Two spellings are in the tree for the same sixteen bits: the bare tuple of hex
    literals blob_interpreter writes, and an np.array of uint16 seen through a
    `.view(np.float16)`, which a test file wants because it indexes the table
    straight with the chord index. The reader takes the literals out of either one
    rather than caring which a file picked, because the question is which bits a
    file holds, not how it holds them.
    """
    direct = _int_seq(node)
    if direct is not None:
        return direct
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "view"
    ):
        return None
    array = node.func.value
    if not (
        isinstance(array, ast.Call)
        and isinstance(array.func, ast.Attribute)
        and array.func.attr == "array"
        and array.args
        and isinstance(array.args[0], ast.List)
    ):
        return None
    return _int_seq(ast.Tuple(elts=array.args[0].elts, ctx=ast.Load()))


def _table_banks(node):
    """`{3: (...), 4: (...)}` as `{op_type: bits}`, or None if it is not that."""
    if not isinstance(node, ast.Dict):
        return None
    banks = {}
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or isinstance(key.value, bool):
            return None
        bits = _bank_bits(value)
        if not isinstance(key.value, int) or bits is None:
            return None
        banks[key.value] = bits
    return banks


def module_tables(source):
    """{name: bits} for every module-level assignment that is a table of ints."""
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        bits = _bank_bits(value)
        if bits is None or len(bits) < 2:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = bits
    return found


def transcribed_banks(source=None):
    """{name: bits} for the banks blob_interpreter holds, keyed as it keys them."""
    tree = ast.parse(_INTERPRETER.read_text() if source is None else source)
    banks = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in ("_PWL_SLOPE", "_PWL_BIAS"):
            for op_type, bits in (_table_banks(node.value) or {}).items():
                banks[f"{name}[{op_type}]"] = bits
    return banks


def second_copies(sources, transcribed=None):
    """Every (file, name, bank) whose table restates an already transcribed bank.

    `sources` is an iterable of (label, text), so the reader can be pointed at a
    tree built in a temporary directory as well as at this one. The test is
    equality against the transcribed bits and not a table of sixteen ints: a file
    holding a different table is not a second copy of this one, and a file holding
    sixteen ints for some other reason must not be reported as a copy.
    """
    transcribed = transcribed_banks() if transcribed is None else transcribed
    hits = []
    for label, text in sources:
        for name, bits in module_tables(text).items():
            for bank, bank_bits in transcribed.items():
                if bits == bank_bits:
                    hits.append((label, name, bank))
    return hits


def _cpp_companded16_tables(source):
    """{name: bits} for the tables of the HTP_OPS_PWL_COMPANDED16 branch of the C++."""
    branch = source.split("#if HTP_OPS_PWL_COMPANDED16")[1].split("#else")[0]
    return {
        name: tuple(int(literal, 16) for literal in re.findall(r"0x[0-9a-fA-F]+", body))
        for name, body in re.findall(r"HTP_OPS_PWL_TABLE\((\w+),([^;]*)\);", branch)
    }


def test_the_reader_finds_the_banks_it_is_meant_to_find():
    """The positive control, and the reason the other four are worth reading.

    A reader that finds nothing and a reader that is broken return the same thing,
    so a guard built on one has to be shown finding what this file says is there
    before its zero can be read as a clean tree. The three banks come back with
    their keys and their lengths, including the four zero slots a twelve-entry
    table is padded out to, and a source with no table in it reads as empty rather
    than as an error: a reader that raises on the next file is a reader that has
    stopped scanning.
    """
    banks = transcribed_banks()
    assert sorted(banks) == [
        "_PWL_BIAS[3]",
        "_PWL_BIAS[4]",
        "_PWL_BIAS[8]",
        "_PWL_SLOPE[3]",
        "_PWL_SLOPE[4]",
        "_PWL_SLOPE[8]",
    ], f"the reader found {sorted(banks)}"
    assert all(len(bits) == 16 for bits in banks.values()), {
        name: len(bits) for name, bits in banks.items()
    }
    # gelu and tanh are twelve chords over four and the companded16 lookup reads
    # four bits, so the four empty slots are a slope and a bias of zero rather than
    # four entries the file left out.
    for op_type in (3, 8):
        assert banks[f"_PWL_SLOPE[{op_type}]"][12:] == (0, 0, 0, 0), op_type
        assert banks[f"_PWL_BIAS[{op_type}]"][12:] == (0, 0, 0, 0), op_type
    assert module_tables("x = 1\n") == {}, "a source with no table read as not empty"


def test_a_copy_that_is_put_back_is_found(tmp_path):
    """The guard below, shown red on a tree that does hold a second copy.

    Built in a temporary directory and out of the bits the reader already found, so
    no bank is written out a second time anywhere in this file. Three files: one
    restating a transcribed bank in the np.array spelling, one restating it in the
    bare-tuple spelling, and one holding sixteen ints that are not a bank. The
    first two have to be named and the third must not be, because a guard that
    reports any sixteen-int table is a guard that gets switched off the first time
    it is wrong.
    """
    slope = transcribed_banks()[f"_PWL_SLOPE[{_SIGMOID}]"]
    literals = ", ".join(str(value) for value in slope)
    files = {
        "array_spelling.py": (
            "import numpy as np\n"
            f"_PRIVATE_SLOPE = np.array([{literals}], dtype=np.uint16).view(np.float16)\n"
        ),
        "tuple_spelling.py": f"_PRIVATE_BIAS = {tuple(_PWL_BIAS[_SIGMOID])!r}\n",
        "unrelated.py": f"_SOMETHING_ELSE = {tuple(range(16))!r}\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)

    hits = second_copies(
        (path.name, path.read_text()) for path in sorted(tmp_path.glob("*.py"))
    )
    assert sorted(hits) == [
        ("array_spelling.py", "_PRIVATE_SLOPE", f"_PWL_SLOPE[{_SIGMOID}]"),
        ("tuple_spelling.py", "_PRIVATE_BIAS", f"_PWL_BIAS[{_SIGMOID}]"),
    ], f"{hits}"


def test_the_sigmoid_chords_are_transcribed_once():
    """The guard itself, over this tree, and the reason it exists.

    Walks every Python file under the test directory -- the test modules and the
    bare-import helpers beside them, because a bank copied into a helper is the same
    duplicate with one fewer grep -- and fails naming any that restates a table
    blob_interpreter already has. test_glu.py did, under _SIGMOID_SLOPE and
    _SIGMOID_BIAS, and moving either by one ulp left all four sigmoid files green,
    which is what this asserts cannot happen a second time.

    _PWL_UNTRANSCRIBED cannot see any of it. It is a dict in blob_interpreter.py
    and a copy in a test file is a different file with a different owner; that is
    the gap fb0c63f describes and this file closes. Re-transcribing a bank is not
    forbidden because the numbers are wrong. It is forbidden because the second
    copy is a second answer to a question the first one already answers, and only
    the first one is checked. The way to refer to the table from another file is to
    import it.
    """
    sources = [
        (str(path.relative_to(_TEST_DIR)), path.read_text())
        for path in sorted(_TEST_DIR.rglob("*.py"))
        if path != _INTERPRETER
    ]
    assert sources, f"nothing under {_TEST_DIR}, so this scanned nothing"
    hits = second_copies(sources)
    assert not hits, (
        "a transcribed PWL bank is defined a second time; import it from "
        "blob_interpreter instead of restating it, or the two will drift apart "
        "with nothing watching either: "
        + ", ".join(f"{where}:{name} ({bank})" for where, name, bank in hits)
    )


def test_the_transcribed_bank_is_the_table_in_unary_ops_cc():
    """The anchor: the sixteen bits are read back out of the C++ and compared.

    blob_interpreter transcribes the table the skel is built with
    (unary_ops.cc:225-237) so the host interpreter can walk chords the way the DSP
    does. Checking that transcription against a second copy of it would only check
    that two transcriptions agree; checking it against unary_ops.cc checks that it
    is the kernel's. That is also what makes the guard above's zero worth
    anything: the one definition left in the tree, and the literals the DSP runs on,
    are the same sixteen values.

    The C++ cannot be read with ast, so this is the one place a regex is
    load-bearing and it is fenced. The branch is selected by the two preprocessor
    lines, the six table names it must return are named, the four lengths are
    checked, and only then is any equality claimed.
    """
    assert _UNARY_OPS.is_file(), (
        f"the skel source is not where this file looks for it: {_UNARY_OPS}"
    )
    tables = _cpp_companded16_tables(_UNARY_OPS.read_text())
    assert sorted(tables) == [
        "gelu_bias",
        "gelu_slope",
        "sigmoid_bias",
        "sigmoid_slope",
        "tanh_bias",
        "tanh_slope",
    ], f"the C++ reader found {sorted(tables)}"
    # sigmoid is the only one of the three with sixteen chords; gelu and tanh are
    # twelve over a range of four and the companded16 lookup reads four bits, so
    # the banks on the Python side carry the four empty slots as zeros.
    assert len(tables["sigmoid_slope"]) == 16, len(tables["sigmoid_slope"])
    for name in ("tanh_slope", "tanh_bias", "gelu_slope", "gelu_bias"):
        assert len(tables[name]) == 12, (name, len(tables[name]))
    for cpp_name, bank_name in (
        ("sigmoid_slope", f"_PWL_SLOPE[{_SIGMOID}]"),
        ("sigmoid_bias", f"_PWL_BIAS[{_SIGMOID}]"),
        ("tanh_slope", "_PWL_SLOPE[8]"),
        ("tanh_bias", "_PWL_BIAS[8]"),
        ("gelu_slope", "_PWL_SLOPE[3]"),
        ("gelu_bias", "_PWL_BIAS[3]"),
    ):
        padded = tables[cpp_name] + (0,) * (16 - len(tables[cpp_name]))
        assert padded == transcribed_banks()[bank_name], (
            f"{cpp_name} is not the table {bank_name} transcribes"
        )


def test_glu_evaluates_the_interpreters_bank_and_agrees_with_it_bit_for_bit():
    """The file that needed the table now takes it, and is checked for having it.

    test_glu.py held the copy because it has to evaluate the chords itself: the
    gate contributes a band to the product, and a band needs the walk's output to
    measure against the scalar path. So the table comes from blob_interpreter,
    which is where the interpreter reads it from, and the index arithmetic stays
    written out here -- a wrong index has to be able to disagree with the
    interpreter, which it could not if it were the interpreter's own.

    The comparison is over every finite fp16 rather than over a probe, 63488
    values, and the two must return identical bits for all of them. That also means
    a change to _PWL_SLOPE which moved the walk would move the number this file
    quotes, instead of leaving it describing a walk that no longer runs.
    """
    import test_glu

    restated = [
        name
        for name, bits in module_tables(
            (_TEST_DIR / "test_glu.py").read_text()
        ).items()
        if bits == tuple(_PWL_SLOPE[_SIGMOID])
    ]
    assert not restated, (
        f"test_glu.py restates the sigmoid chords in {restated}; take them from "
        "blob_interpreter, which is where the interpreter takes them from"
    )

    every = np.arange(0, 1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16)
    every = np.ascontiguousarray(every[np.isfinite(every)])
    mine = np.ascontiguousarray(test_glu._pwl_sigmoid(every))
    theirs = np.ascontiguousarray(_pwl_body(every, _SIGMOID))
    assert every.size == 63488, every.size
    differing = int((mine.view(np.uint16) != theirs.view(np.uint16)).sum())
    assert differing == 0, (
        f"{differing} of {every.size} finite fp16 come out of the two walks "
        "differently, so the index arithmetic has moved away from the interpreter's"
    )
