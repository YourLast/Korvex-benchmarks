#!/usr/bin/env python3
"""Second official FACTS Grounding v2 judge: OpenAI gpt-5-2025-08-07, with the official notebook's own prompts and parsers.

    export OPENAI_API_KEY="sk-<your key>"
    python3 judge_gpt5.py --tasks work/tasks.jsonl --answers work/answers.jsonl \\
        --gemini work/gemini_out/gemini25flash_judged.jsonl --starter work/starter/<downloaded notebook>.ipynb \\
        --out work/gpt5_verdicts.jsonl --budget-usd 20

This is what the official notebook does with its second judge (JUDGES = [gemini-2.5-flash, gpt-5-2025-08-07]):
  * quality filter (the judge answers the task itself, then rates the test answer for instruction following): the notebook asks
    the second judge only when the first one (Gemini) found a Major issue, so GPT-5 is asked only for those answers;
  * grounding (sentence-by-sentence check against the document): for every answer.
The prompts and the parsing code are NOT in this kit: they are loaded from the official starter notebook you downloaded
(`kaggle kernels pull prathameshbang/facts-grounding-v2-benchmark-starter -p work/starter`), see official_cells.py.
API parameters are the defaults (only `model` and `messages`: no reasoning_effort, no token limit), as in the official notebook.

Costs real money. --budget-usd is a hard stop: no new call is started once the spend reaches it (calls already running
finish, so the final total can exceed the cap by roughly workers x one call, about $0.1 each). In our run the 200 answers
cost about $12. Prices used for the estimate (USD per 1M tokens): input 1.25, cached input 0.125, output 10.00; change the
constants below if OpenAI's prices differ. Resumable: tasks that already have a verdict in --out are skipped.

Environment: OPENAI_API_KEY (required, never printed); OPENAI_BASE_URL (default https://api.openai.com/v1).
Python standard library only.
"""
import argparse
import concurrent.futures as cf
import contextlib
import json
import os
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import official_cells

MODEL = "gpt-5-2025-08-07"
JUDGE = "openai/" + MODEL
P_IN, P_CACHED, P_OUT = 1.25, 0.125, 10.0
USD_PER_TASK_HINT = 0.06  # what a task cost us on average (only used by --dry-run)

lock = threading.Lock()
spend = {"usd": 0.0, "in": 0, "cached": 0, "out": 0, "calls": 0}
halt = threading.Event()
tls = threading.local()


def read_jsonl(path):
    out = []
    for l in open(path, encoding="utf-8"):
        if l.strip():
            out.append(json.loads(l))
    return out


def last_good(path, ok):
    d = {}
    if os.path.exists(path):
        for r in read_jsonl(path):
            if ok(r):
                d[r["id"]] = r
    return d


def make_call(base, key, budget, retries=8):
    def call(messages):
        with lock:
            if spend["usd"] >= budget:
                halt.set()
                tls.failed.append("budget cap reached")
                raise RuntimeError("budget cap %.2f USD reached" % budget)
        body = json.dumps({"model": MODEL, "messages": messages}).encode("utf-8")
        d = None
        for attempt in range(retries):
            req = urllib.request.Request(base + "/chat/completions", data=body, headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "korvex-facts-bench/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and "insufficient_quota" not in msg and attempt + 1 < retries:
                    time.sleep(min(120, 5 * 2 ** attempt))
                    continue
                tls.failed.append("HTTP %d %s" % (e.code, msg))
                raise RuntimeError("HTTP %d %s" % (e.code, msg))
            except Exception as e:  # noqa: BLE001 - timeouts, resets
                if attempt + 1 == retries:
                    tls.failed.append(repr(e)[:200])
                    raise
                time.sleep(min(120, 5 * 2 ** attempt))
        u = d.get("usage") or {}
        pin = u.get("prompt_tokens", 0) or 0
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        pout = u.get("completion_tokens", 0) or 0
        usd = ((pin - cached) * P_IN + cached * P_CACHED + pout * P_OUT) / 1e6
        with lock:
            spend["usd"] += usd
            spend["in"] += pin
            spend["cached"] += cached
            spend["out"] += pout
            spend["calls"] += 1
        tls.usd += usd
        content = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if not content.strip():
            tls.failed.append("empty reply")
        return content
    return call


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="tasks.jsonl (tasks/build_tasks.py)")
    ap.add_argument("--answers", required=True, help="answers.jsonl ({id, idx, response})")
    ap.add_argument("--gemini", required=True, help="Gemini verdicts from the Kaggle notebook (gemini25flash_judged.jsonl)")
    ap.add_argument("--starter", required=True, help="the official starter notebook (.ipynb) you downloaded")
    ap.add_argument("--out", required=True, help="verdicts jsonl (appended; resumable)")
    ap.add_argument("--budget-usd", type=float, required=True, help="hard spending cap in USD")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=0, help="only the first N tasks")
    ap.add_argument("--dry-run", action="store_true", help="show what would be sent and a rough cost; nothing is sent")
    a = ap.parse_args()

    tasks = {t["id"]: t for t in read_jsonl(a.tasks)}
    order = [t["id"] for t in read_jsonl(a.tasks)]
    answers = last_good(a.answers, lambda r: str(r.get("response", "")).strip())
    gem = last_good(a.gemini, lambda r: not r.get("error") and not r.get("err") and r.get("grounded") is not None and r.get("quality_rating"))
    ids = [i for i in order if i in answers]
    if a.n:
        ids = ids[:a.n]
    no_gem = [i for i in ids if i not in gem]
    if no_gem:
        print("warning: %d answers have no Gemini verdict and are skipped (e.g. %s)" % (len(no_gem), ", ".join(no_gem[:3])), file=sys.stderr)
    ids = [i for i in ids if i in gem]
    done = last_good(a.out, lambda r: not r.get("err") and r.get("grounded") is not None)
    todo = [i for i in ids if i not in done]
    n_major = sum(1 for i in todo if gem[i]["quality_rating"] == "Major Issue(s)")
    print("answers %d, with Gemini verdict %d, already judged %d, to do %d (of which %d need the quality check because Gemini said Major)"
          % (len(answers), len(ids), len(done), len(todo), n_major))
    print("rough cost of the to-do list: about $%.2f (our run: about $%.2f per task); cap $%.2f" % (USD_PER_TASK_HINT * len(todo), USD_PER_TASK_HINT, a.budget_usd))
    if a.dry_run:
        print("dry run: nothing was sent")
        return
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("set OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    call = make_call(base, key, a.budget_usd)

    class _LLM:
        def prompt(self, x):
            return call([{"role": "user", "content": x}] if isinstance(x, str) else list(x))

    kb = types.ModuleType("kaggle_benchmarks")
    kb.llms = {JUDGE: _LLM()}
    kb.chats = types.SimpleNamespace(new=contextlib.nullcontext)
    g = official_cells.exec_official(a.starter, kb)

    def one(i):
        if halt.is_set():
            return None
        t, resp = tasks[i], answers[i]["response"]
        tls.usd, tls.failed = 0.0, []
        t0 = time.time()
        rec = {"id": i, "judge_model": MODEL, "gemini_quality": gem[i]["quality_rating"]}
        try:
            if gem[i]["quality_rating"] == "Major Issue(s)":
                with kb.chats.new():
                    baseline = kb.llms[JUDGE].prompt(t["full_prompt"])
                ok, qtext, rating = g["_evaluate_quality_no_context"](t["user_request"], resp, baseline, JUDGE)
                rec.update(quality_called=True, eligible=bool(ok), quality_rating=rating, raw_quality=qtext)
            else:
                rec.update(quality_called=False, eligible=None, quality_rating=None)
            grounded, gtext = g["classify_grounding"](t["user_request"], t["context_document"], resp, JUDGE)
            rec.update(grounded=bool(grounded), raw_grounding=gtext)
        except Exception as e:  # noqa: BLE001
            tls.failed.append(repr(e)[:200])
        # the official helpers swallow API errors and carry on with an empty reply: never record such a result as a verdict
        if tls.failed:
            rec = {"id": i, "judge_model": MODEL, "err": "; ".join(dict.fromkeys(tls.failed))[:400]}
        rec["cost_usd"] = round(tls.usd, 5)
        rec["secs"] = round(time.time() - t0, 1)
        return rec

    n_ok = n_err = 0
    with open(a.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(max(1, a.workers)) as ex:
        for k, rec in enumerate(ex.map(one, todo), 1):
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if rec.get("err"):
                n_err += 1
            else:
                n_ok += 1
            print("[%d/%d] %s %s grounded=%s %s $%.3f (total $%.2f)" % (k, len(todo), rec["id"], rec.get("quality_rating") or "-", rec.get("grounded"),
                                                                       ("ERROR " + rec["err"]) if rec.get("err") else "", rec["cost_usd"], spend["usd"]), flush=True)
    print("done: %d verdicts, %d errors%s; spend $%.2f (%d calls)" % (n_ok, n_err, " (STOPPED at the budget cap)" if halt.is_set() else "", spend["usd"], spend["calls"]))
    if n_err or halt.is_set():
        print("run the same command again to retry the missing ones")
        sys.exit(1)


if __name__ == "__main__":
    main()
