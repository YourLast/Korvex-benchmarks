"""Load the three code cells of the OFFICIAL FACTS Grounding v2 starter notebook that define the judge prompts and parsers.

This kit does not contain the official notebook or its code. You download it yourself:

    kaggle kernels pull prathameshbang/facts-grounding-v2-benchmark-starter -p work/starter

and pass the downloaded .ipynb to the scripts that need it. The cells are picked by their content (not by position):
  1. the imports cell            (import kaggle_benchmarks as kbench, re, json)
  2. the quality cell            (def extract_instruction_json / def _evaluate_quality_no_context)
  3. the grounding cell          (UFG_REV21_PROMPT, def parse_ufg_rev21_verdict, def classify_grounding)

Because the cells come from a file on your disk and are executed with your API key in the environment, they are checked
before they run: only imports of re / json / typing / kaggle_benchmarks, function definitions and string constants are
accepted at the top level, and function bodies may not touch os, sys, subprocess, sockets, files, eval/exec and the like.
Read the cells yourself as well; they only define prompts and parsing functions.
"""
import ast
import json
import sys

ALLOWED_IMPORTS = {"kaggle_benchmarks", "re", "json", "typing"}
FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "urllib", "requests", "http", "importlib", "ctypes", "shutil", "pathlib",
    "builtins", "eval", "exec", "open", "__import__", "globals", "locals", "setattr", "delattr", "input",
}


def load_notebook(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "cells" not in d and isinstance(d.get("blob"), dict):  # raw dump of the Kaggle API answer
        src = d["blob"].get("source")
        d = json.loads(src) if isinstance(src, str) else src
    if "cells" not in d:
        raise SystemExit("%s: not a Jupyter notebook (no 'cells')" % path)
    return d


def cell_source(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def official_cells(path):
    """Return (imports_src, quality_src, grounding_src) of the official starter notebook."""
    nb = load_notebook(path)
    code = [cell_source(c) for c in nb["cells"] if c.get("cell_type") == "code"]

    def one(pred, what):
        hits = [s for s in code if pred(s)]
        if len(hits) != 1:
            raise SystemExit("%s: expected exactly one %s cell, found %d. Is this the official FACTS Grounding v2 starter "
                             "notebook (prathameshbang/facts-grounding-v2-benchmark-starter)?" % (path, what, len(hits)))
        return hits[0]

    imports = one(lambda s: "import kaggle_benchmarks" in s and len(s) < 400, "imports")
    quality = one(lambda s: "def _evaluate_quality_no_context" in s, "quality-filter")
    grounding = one(lambda s: "UFG_REV21_PROMPT" in s and "def classify_grounding" in s, "grounding")
    return imports, quality, grounding


def _is_const_string_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    return (isinstance(node, ast.Call) and not node.args and not node.keywords and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("strip", "lstrip", "rstrip") and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str))


def check_cell(src, label):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            bad = [node.module] if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS else []
        elif isinstance(node, ast.FunctionDef):
            bad = []
        elif isinstance(node, ast.Assign) and _is_const_string_expr(node.value):
            bad = []
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):  # docstring-like
            bad = []
        else:
            bad = [type(node).__name__]
        if bad:
            raise SystemExit("%s: refusing to run - unexpected top-level statement %s" % (label, bad))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            raise SystemExit("%s: refusing to run - import inside a function" % label)
        name = node.id if isinstance(node, ast.Name) else node.value.id if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)) else None
        if name in FORBIDDEN_NAMES:
            raise SystemExit("%s: refusing to run - uses '%s'" % (label, name))


def exec_official(path, kbench_module):
    """Execute the checked official cells with `kaggle_benchmarks` replaced by `kbench_module`; return their namespace."""
    sys.modules["kaggle_benchmarks"] = kbench_module
    g = {"__name__": "official"}
    for label, src in zip(("imports", "quality", "grounding"), official_cells(path)):
        check_cell(src, "official %s cell" % label)
        exec(compile(src, "<official %s cell>" % label, "exec"), g)
    return g
