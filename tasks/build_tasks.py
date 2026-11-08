#!/usr/bin/env python3
"""Build the task file from the official FACTS Grounding examples and the list of task indices.

    python3 tasks/build_tasks.py --examples work/examples.csv --idx tasks/tasks200_idx.txt --out work/tasks.jsonl

examples.csv comes from the Kaggle dataset deepmind/facts-grounding-examples (856 rows, columns system_instruction,
user_request, context_document, full_prompt, domain, type, high_level_type). The index in the list is the 0-based row
index of that file, in the order of the file. Output: one JSON object per line with
{id, idx, full_prompt, user_request, context_document}; `id` is "facts-" + the 3-digit row index.
"""
import argparse
import csv
import hashlib
import json
import sys

KNOWN_SHA256 = "66990aea86a91d40edcda6ebcd0d686674246dc7c07338ed8c17e1c275b1e096"  # examples.csv we used (dataset version 17)


def read_idx(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", required=True, help="path to examples.csv from the Kaggle dataset")
    ap.add_argument("--idx", default="tasks/tasks200_idx.txt", help="list of row indices (one per line)")
    ap.add_argument("--out", required=True, help="output jsonl")
    a = ap.parse_args()

    raw = open(a.examples, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != KNOWN_SHA256:
        print("WARNING: examples.csv SHA-256 is %s, the file we used had %s. If the dataset was updated, row indices may "
              "point to different tasks. Check the dataset version before comparing numbers." % (sha, KNOWN_SHA256), file=sys.stderr)
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(a.examples, encoding="utf-8-sig", newline="")))
    print("examples.csv: %d rows" % len(rows))
    idx = read_idx(a.idx)
    if len(set(idx)) != len(idx):
        raise SystemExit("duplicate indices in %s" % a.idx)
    bad = [i for i in idx if not 0 <= i < len(rows)]
    if bad:
        raise SystemExit("indices outside the file (%d rows): %s" % (len(rows), bad[:10]))
    with open(a.out, "w", encoding="utf-8") as f:
        for i in idx:
            r = rows[i]
            f.write(json.dumps({"id": "facts-%03d" % i, "idx": i, "full_prompt": r["full_prompt"], "user_request": r["user_request"],
                                "context_document": r["context_document"]}, ensure_ascii=False) + "\n")
    print("wrote %d tasks to %s" % (len(idx), a.out))


if __name__ == "__main__":
    main()
