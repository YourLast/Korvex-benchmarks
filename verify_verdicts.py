#!/usr/bin/env python3
"""Check that the shipped (or your own) verdict files agree with the judges' raw replies, using the OFFICIAL parsers.

    python3 verify_verdicts.py --starter work/starter/<downloaded notebook>.ipynb \\
        --gemini results/gemini_verdicts.jsonl --gpt5 results/gpt5_verdicts.jsonl

For every row it re-parses `raw_grounding` with the notebook's parse_ufg_rev21_verdict and `raw_quality` with its
extract_instruction_json, and compares the result with the stored `grounded` / `quality_rating` / `eligible` fields.
No network, no API calls, nothing is judged again. The parsers are loaded from the official starter notebook you
downloaded (see official_cells.py).
"""
import argparse
import json
import sys
import types

import official_cells


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def check(name, path, g):
    n = bad = 0
    for r in rows(path):
        n += 1
        problems = []
        raw = r.get("raw_grounding")
        if raw is None:
            problems.append("no raw_grounding")
        else:
            grounded, _ = g["parse_ufg_rev21_verdict"](raw, name)
            if bool(grounded) != bool(r.get("grounded")):
                problems.append("grounded: stored %s, raw reply says %s" % (r.get("grounded"), grounded))
        rq = r.get("raw_quality")
        if rq is not None:
            rating = g["extract_instruction_json"](rq)["Instruction Following"]
            if rating != r.get("quality_rating"):
                problems.append("quality_rating: stored %s, raw reply says %s" % (r.get("quality_rating"), rating))
            if "eligible" in r and r["eligible"] is not None and bool(r["eligible"]) != ("Major Issue(s)" not in rating):
                problems.append("eligible does not follow from the rating")
        elif r.get("quality_called"):
            problems.append("quality_called but no raw_quality")
        if problems:
            bad += 1
            print("  MISMATCH %s: %s" % (r["id"], "; ".join(problems)))
    print("%s: %d rows checked, %d mismatches" % (name, n, bad))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starter", required=True)
    ap.add_argument("--gemini", required=True)
    ap.add_argument("--gpt5", required=True)
    a = ap.parse_args()
    g = official_cells.exec_official(a.starter, types.ModuleType("kaggle_benchmarks"))
    bad = check("gemini", a.gemini, g) + check("gpt5", a.gpt5, g)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
