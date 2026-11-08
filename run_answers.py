#!/usr/bin/env python3
"""Get answers to the FACTS Grounding tasks from the Korvex API (OpenAI-compatible chat completions).

    export KORVEX_API_KEY="kx-<your key>"
    python3 run_answers.py --tasks work/tasks.jsonl --out work/answers.jsonl --workers 2

Every task's `full_prompt` is sent unchanged as ONE user message (no system message, no tools), reasoning level `max`,
non-streaming, one request per task. Lines longer than 20,000 characters are re-wrapped at spaces (only spaces become
newlines; 3 of our 200 tasks were affected). The final answer is the message content after the last "</think>", stripped.

Output: one JSON object per line, {"id", "idx", "response"}. The script is resumable: tasks that already have a non-empty
response in --out are skipped, failed tasks are simply tried again on the next run.

Environment:
  KORVEX_API_KEY   required (never printed)
  KORVEX_BASE_URL  default https://api.korvexai.site/v1

Python standard library only.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "https://api.korvexai.site/v1"
USER_AGENT = "korvex-facts-bench/1.0"
PERMANENT_CODES = {"context_length_exceeded", "invalid_api_key", "missing_api_key", "model_not_found", "model_not_allowed",
                   "invalid_reasoning_effort", "invalid_messages", "request_too_large", "model_not_in_api"}


class Fatal(Exception):
    """Problem with the key / model / request shape: no point in continuing with other tasks."""


def wrap_long_lines(s, thr=20000, piece=4000):
    """Break lines longer than `thr` characters at a space near `piece` characters (spaces become newlines, nothing else changes)."""
    if len(s) < thr or max(len(t) for t in s.split("\n")) < thr:
        return s, False
    out = []
    for t in s.split("\n"):
        while len(t) > thr:
            cut = t.find(" ", piece, piece + 1500)
            if cut < 0:
                cut = piece
            out.append(t[:cut])
            t = t[cut + 1:] if t[cut:cut + 1] == " " else t[cut:]
        out.append(t)
    return "\n".join(out), True


def final_text(content):
    return (content or "").rsplit("</think>", 1)[-1].strip()


def http_json(method, url, key, body=None, timeout=60):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw.decode("utf-8"))  # leading whitespace (keep-alive padding while the model works) is valid JSON


def build_request(model, effort, full_prompt):
    msg, wrapped = wrap_long_lines(full_prompt)
    return {"model": model, "stream": False, "reasoning_effort": effort, "messages": [{"role": "user", "content": msg}]}, wrapped


def check_levels(base, key, model, effort):
    """Ask GET /models which reasoning levels this key can use. Returns an error text, or None if fine / unknown."""
    try:
        j = http_json("GET", base + "/models", key, timeout=60)
    except Exception as e:  # noqa: BLE001 - the check is advisory; the real calls report their own errors
        print("note: could not read %s/models (%s); skipping the level check" % (base, type(e).__name__), file=sys.stderr)
        return None
    want = model.lower()
    for m in j.get("data", []):
        names = {str(m.get("id", "")).lower(), str(m.get("name", "")).lower().replace(" ", "-")} | {str(x).lower() for x in (m.get("dsr") or {}).get("aliases", [])}
        if want in names:
            levels = (m.get("dsr") or {}).get("reasoning_effort") or (m.get("dsr") or {}).get("levels") or []
            if levels and effort not in levels:
                return ("reasoning level '%s' is not available to this key (available: %s). Numbers at another level are "
                        "not comparable with ours. Max is temporarily unavailable through the public API (too expensive "
                        "for our servers); use --effort xhigh for a run at xhigh, or re-score our published answers." % (effort, ", ".join(levels)))
            return None
    return "model '%s' is not listed for this key (GET %s/models)" % (model, base)


def ask(task, a, key):
    body, wrapped = build_request(a.model, a.effort, task["full_prompt"])
    url = a.base + "/chat/completions"
    last = "no attempt"
    for attempt in range(a.retries):
        t0 = time.time()
        wait = min(120, a.retry_wait * 2 ** attempt)
        try:
            j = http_json("POST", url, key, body, timeout=a.timeout)
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", "replace")[:300]
            last = "HTTP %d %s" % (e.code, txt)
            if e.code in (400, 401, 403, 404):  # key, model or request shape is wrong: the same for every task
                raise Fatal(last)
            if e.code == 413:  # this one task is too large for the model
                raise RuntimeError(last)
            if e.code == 429 or e.code >= 500:
                try:
                    wait = max(wait, int(e.headers.get("Retry-After", "0")))
                except ValueError:
                    pass
        except Exception as e:  # noqa: BLE001 - timeouts, resets, bad JSON: retry
            last = "%s: %s" % (type(e).__name__, e)
        else:
            if "error" in j:  # the service reports late errors in a 200 body
                err = j["error"] or {}
                last = "API error %s: %s" % (err.get("code"), str(err.get("message"))[:200])
                if err.get("code") in ("invalid_api_key", "missing_api_key", "model_not_found", "model_not_allowed", "invalid_reasoning_effort"):
                    raise Fatal(last)
                if err.get("code") in PERMANENT_CODES:
                    raise RuntimeError(last)
            else:
                ch = (j.get("choices") or [{}])[0]
                text = final_text((ch.get("message") or {}).get("content"))
                if text:
                    return {"id": task["id"], "idx": task["idx"], "response": text}, round(time.time() - t0, 1), ch.get("finish_reason"), wrapped
                last = "empty answer (finish_reason=%s)" % ch.get("finish_reason")
        print("  %s attempt %d/%d failed: %s" % (task["id"], attempt + 1, a.retries, last), file=sys.stderr, flush=True)
        if attempt + 1 < a.retries:
            time.sleep(wait)
    raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="tasks.jsonl built by tasks/build_tasks.py")
    ap.add_argument("--out", required=True, help="answers.jsonl (appended; resumable)")
    ap.add_argument("--model", default="dsr-etude-0-preview", help="model id as listed by GET /v1/models (default: dsr-etude-0-preview)")
    ap.add_argument("--effort", default="max", help="reasoning_effort sent with every request (default: max)")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests (we used 2; the API limits parallel requests per key)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke test)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds to wait for one answer")
    ap.add_argument("--retries", type=int, default=6, help="attempts per task for temporary errors (429, 5xx, timeouts)")
    ap.add_argument("--retry-wait", type=float, default=10, help="seconds before the first retry; doubles each time, at most 120 (default 10)")
    ap.add_argument("--base-url", default=os.environ.get("KORVEX_BASE_URL", DEFAULT_BASE), help="default: $KORVEX_BASE_URL or " + DEFAULT_BASE)
    ap.add_argument("--force", action="store_true", help="run even if the level check says the key cannot use --effort")
    ap.add_argument("--dry-run", action="store_true", help="print the first request (key hidden) and exit; nothing is sent")
    a = ap.parse_args()
    a.base = a.base_url.rstrip("/")

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    if a.limit:
        tasks = tasks[:a.limit]
    if a.dry_run:
        body, wrapped = build_request(a.model, a.effort, tasks[0]["full_prompt"])
        shown = dict(body)
        shown["messages"] = [{"role": "user", "content": body["messages"][0]["content"][:160] + "... [%d characters in all]" % len(body["messages"][0]["content"])}]
        print("POST %s/chat/completions" % a.base)
        print("Authorization: Bearer <KORVEX_API_KEY>\nContent-Type: application/json")
        print(json.dumps(shown, indent=2, ensure_ascii=False))
        print("(%d tasks, first task %s, wrapped=%s; nothing was sent)" % (len(tasks), tasks[0]["id"], wrapped))
        return

    key = os.environ.get("KORVEX_API_KEY", "").strip()
    if not key:
        raise SystemExit("set KORVEX_API_KEY")
    problem = check_levels(a.base, key, a.model, a.effort)
    if problem:
        if not a.force:
            raise SystemExit("stop: " + problem)
        print("warning: " + problem, file=sys.stderr)

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if str(r.get("response", "")).strip():
                done.add(r["id"])
    todo = [t for t in tasks if t["id"] not in done]
    print("tasks %d, already done %d, to do %d, workers %d, model %s, reasoning_effort %s" % (len(tasks), len(done), len(todo), a.workers, a.model, a.effort), flush=True)

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    stop = threading.Event()

    def work(t):
        if stop.is_set():
            return
        try:
            rec, secs, finish, wrapped = ask(t, a, key)
        except Fatal as e:
            stop.set()
            print("FATAL %s: %s" % (t["id"], e), file=sys.stderr, flush=True)
            return
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["fail"] += 1
            print("FAILED %s: %s" % (t["id"], e), file=sys.stderr, flush=True)
            return
        with lock:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            n = stats["ok"] + stats["fail"]
        print("[%d/%d] %s %.0fs %d chars finish=%s%s" % (n, len(todo), t["id"], secs, len(rec["response"]), finish, " (long lines wrapped)" if wrapped else ""), flush=True)

    try:
        with cf.ThreadPoolExecutor(max(1, a.workers)) as ex:
            list(ex.map(work, todo))
    except KeyboardInterrupt:
        stop.set()
        print("interrupted; run the same command again to continue", file=sys.stderr)
    print("done: %d ok, %d failed%s" % (stats["ok"], stats["fail"], "; STOPPED by a fatal error" if stop.is_set() else ""))
    sys.exit(1 if stats["fail"] or stop.is_set() else 0)


if __name__ == "__main__":
    main()
