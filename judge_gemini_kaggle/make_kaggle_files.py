#!/usr/bin/env python3
"""Prepare the Gemini 2.5 Flash judging run on Kaggle (the first official judge).

The official FACTS Grounding v2 notebook reaches Gemini through Kaggle Benchmarks (`kaggle_benchmarks`, model name
google/gemini-2.5-flash). That module exists only in the Kaggle image used by benchmark notebooks, so the judging notebook
must be a NEW VERSION OF YOUR FORK of the official starter notebook; a standalone notebook fails with
"ModuleNotFoundError: kaggle_benchmarks". See README.md in this folder for the whole procedure.

This script writes two things:
  1. OUT_DIR/dataset/  : a private-dataset folder with judge_input.jsonl (the tasks and your answers) and dataset-metadata.json;
  2. FORK_DIR/judge.ipynb : the judging notebook; FORK_DIR/kernel-metadata.json is patched to run it and to attach your dataset
     (the original is saved as kernel-metadata.json.orig). The notebook contains the three official cells, copied verbatim from
     the starter notebook you downloaded, and a short loop of ours that calls them for every answer.

    python3 judge_gemini_kaggle/make_kaggle_files.py --tasks work/tasks.jsonl --answers work/answers.jsonl \\
        --owner YOUR_KAGGLE_USERNAME --fork-dir work/fork [--starter work/starter/<notebook>.ipynb]

Python standard library only.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import official_cells  # noqa: E402

LOOP = '''# --- our loop (MIT, not part of the official notebook): judge every answer with the official helper functions above ---
import glob, time, traceback, concurrent.futures as cf
JUDGE = "google/gemini-2.5-flash"
N = %(n)d
WORKERS = %(workers)d
ONLY_IDS = %(only_ids)s
path = glob.glob("/kaggle/input/**/judge_input.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()][:N]
if ONLY_IDS is not None:
    rows = [r for r in rows if r["id"] in set(ONLY_IDS)]
out_path = "/kaggle/working/gemini25flash_judged.jsonl"
done = set()
try:
    done = {json.loads(l)["id"] for l in open(out_path) if l.strip() and not json.loads(l).get("error")}
except FileNotFoundError:
    pass


def judge_one(r):
    rec = {"id": r["id"], "judge": JUDGE}
    t0 = time.time()
    try:
        # quality filter, as in the official notebook: the judge answers the task itself (baseline), then rates the test answer
        with kbench.chats.new():
            baseline = kbench.llms[JUDGE].prompt(r["full_prompt"])
        ok, qtext, rating = _evaluate_quality_no_context(r["user_request"], r["response"], baseline, JUDGE)
        if not str(qtext).strip():
            raise RuntimeError("empty quality reply from the judge (the official helper swallows API errors)")
        rec.update(eligible=bool(ok), quality_rating=rating, raw_quality=qtext)
        # grounding for every answer: with two judges an answer is ineligible only if BOTH judges say Major
        grounded, gtext = classify_grounding(r["user_request"], r["context_document"], r["response"], JUDGE)
        if not str(gtext).strip():
            raise RuntimeError("empty grounding reply from the judge (the official helper swallows API errors)")
        rec.update(grounded=bool(grounded), raw_grounding=gtext)
    except Exception as e:
        rec["error"] = repr(e)[:500]
        rec["tb"] = traceback.format_exc()[-1500:]
    rec["secs"] = round(time.time() - t0, 1)
    return rec


todo = [r for r in rows if r["id"] not in done]
print("todo", len(todo), "of", len(rows))
t = time.time(); n = 0
with open(out_path, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(WORKERS) as ex:
    for rec in ex.map(judge_one, todo):
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n"); f.flush(); n += 1
        print(n, rec["id"], rec.get("quality_rating"), rec.get("grounded"), rec.get("error", ""), rec["secs"])
print("done", n, "in", round(time.time() - t), "s")
'''


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--owner", required=True, help="your Kaggle username (the dataset will be <owner>/<slug>)")
    ap.add_argument("--slug", default="facts-v2-answers", help="dataset slug (default: facts-v2-answers)")
    ap.add_argument("--fork-dir", required=True, help="directory from `kaggle kernels pull YOUR_USERNAME/<your fork> -p DIR -m`")
    ap.add_argument("--starter", help="official starter .ipynb to take the three cells from (default: the notebook in --fork-dir)")
    ap.add_argument("--out-dir", default="work/kaggle")
    ap.add_argument("--n", type=int, default=100000, help="judge only the first N answers (default: all)")
    ap.add_argument("--workers", type=int, default=5, help="parallel judge calls inside the notebook (we used 5)")
    ap.add_argument("--ids-file", help="text file with task ids (one per line): judge only these, e.g. to redo errors")
    a = ap.parse_args()

    meta_path = os.path.join(a.fork_dir, "kernel-metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit("%s not found. Fork the official starter notebook on Kaggle, then run "
                         "`kaggle kernels pull YOUR_USERNAME/<name of your fork> -p %s -m`" % (meta_path, a.fork_dir))
    meta = json.load(open(meta_path, encoding="utf-8"))
    starter = a.starter
    if not starter:
        nbs = [os.path.join(a.fork_dir, f) for f in sorted(os.listdir(a.fork_dir)) if f.endswith(".ipynb")]
        code_file = os.path.join(a.fork_dir, meta.get("code_file", ""))
        starter = code_file if code_file in nbs else (nbs[0] if nbs else None)
        if not starter:
            raise SystemExit("no .ipynb in %s; pass --starter" % a.fork_dir)
    imports_src, quality_src, grounding_src = official_cells.official_cells(starter)

    # 1) dataset folder
    tasks = {t["id"]: t for t in read_jsonl(a.tasks)}
    answers = {}
    for r in read_jsonl(a.answers):
        if str(r.get("response", "")).strip():
            answers[r["id"]] = r["response"]
    ddir = os.path.join(a.out_dir, "dataset")
    os.makedirs(ddir, exist_ok=True)
    n = 0
    with open(os.path.join(ddir, "judge_input.jsonl"), "w", encoding="utf-8") as f:
        for i, t in tasks.items():
            if i in answers:
                f.write(json.dumps({"id": i, "full_prompt": t["full_prompt"], "user_request": t["user_request"],
                                    "context_document": t["context_document"], "response": answers[i]}, ensure_ascii=False) + "\n")
                n += 1
    json.dump({"title": "FACTS v2 answers to judge", "id": "%s/%s" % (a.owner, a.slug), "licenses": [{"name": "other"}]},
              open(os.path.join(ddir, "dataset-metadata.json"), "w"), indent=1)
    print("dataset: %d answers with their tasks -> %s (%d tasks had no answer)" % (n, ddir, len(tasks) - n))

    # 2) notebook + patched kernel metadata
    only = None
    if a.ids_file:
        only = [l.strip() for l in open(a.ids_file) if l.strip()]
    loop = LOOP % {"n": a.n, "workers": a.workers, "only_ids": repr(only)}
    cells = [{"cell_type": "markdown", "metadata": {}, "source": "Judge precomputed answers with the official FACTS Grounding v2 helper functions "
                                                                 "(the three code cells below are copied verbatim from the official starter notebook); judge: google/gemini-2.5-flash."}]
    for s in (imports_src, quality_src, grounding_src, loop):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    nb_path = os.path.join(a.fork_dir, "judge.ipynb")
    json.dump(nb, open(nb_path, "w", encoding="utf-8"))
    if not os.path.exists(meta_path + ".orig"):
        json.dump(meta, open(meta_path + ".orig", "w", encoding="utf-8"), indent=1)
    meta["code_file"] = "judge.ipynb"
    ds = list(meta.get("dataset_sources") or [])
    mine = "%s/%s" % (a.owner, a.slug)
    if mine not in ds:
        ds.append(mine)
    meta["dataset_sources"] = ds
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1)
    print("notebook: %s (kernel %s, datasets %s)" % (nb_path, meta.get("id"), ds))
    print("next: kaggle datasets create -p %s ; wait until it is ready ; kaggle kernels push -p %s" % (ddir, a.fork_dir))


if __name__ == "__main__":
    main()
