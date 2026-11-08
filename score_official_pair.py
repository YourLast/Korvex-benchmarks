#!/usr/bin/env python3
"""Official FACTS Grounding v2 score for the judge pair Gemini 2.5 Flash + GPT-5 (the rule of the official notebook).

    python3 score_official_pair.py GEMINI_VERDICTS.jsonl GPT5_VERDICTS.jsonl [SUMMARY.json]

Per answer (task):
  * the answer is ineligible (score 0) only if BOTH judges rate its instruction following "Major Issue(s)". The notebook asks
    the second judge only when the first one said Major, so a GPT-5 quality verdict exists only for those answers;
  * otherwise the score is the mean of the two judges' grounding verdicts (1 = every sentence supported, else 0): 0, 0.5 or 1.
The reported score is the mean over tasks, in percent, with a 95 % bootstrap interval over tasks (percentile method,
B = 10000 resamples, seed 20261104). Rows with an "error"/"err" field or without a verdict are ignored (and counted as missing);
if a task appears several times the last good row wins.

Input rows (jsonl): Gemini {id, quality_rating, grounded}; GPT-5 {id, grounded, quality_called, eligible}.
Python standard library only.
"""
import json
import random
import sys

SEED, B = 20261104, 10000
MAJOR = "Major Issue(s)"


def load(path, need):
    d = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or r.get("err") or any(r.get(k) is None for k in need):
            continue
        d[r["id"]] = r
    return d


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    G = load(sys.argv[1], ("grounded", "quality_rating"))
    P = load(sys.argv[2], ("grounded",))
    ids = sorted(set(G) & set(P))
    items, inel, bad = [], 0, 0
    kept = []
    for i in ids:
        g, p = G[i], P[i]
        if g["quality_rating"] == MAJOR:
            if not p.get("quality_called"):  # GPT-5 was supposed to be asked here: incomplete row
                bad += 1
                continue
            if not p.get("eligible"):
                items.append(0.0)
                inel += 1
                kept.append(i)
                continue
        items.append((int(bool(g["grounded"])) + int(bool(p["grounded"]))) / 2)
        kept.append(i)
    n = len(items)
    if not n:
        raise SystemExit("no task has verdicts from both judges")
    rnd = random.Random(SEED)
    means = sorted(sum(items[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    lo, hi = round(100 * means[int(.025 * B)], 1), round(100 * means[int(.975 * B) - 1], 1)
    score = round(100 * sum(items) / n, 1)
    S = {
        "n": n,
        "missing": len((set(G) | set(P)) - set(kept)),
        "score": score,
        "ci95_boot": [lo, hi],
        "ineligible_both_major": inel,
        "grounded_rate": {"gemini": round(100 * sum(bool(G[i]["grounded"]) for i in kept) / n, 1),
                          "gpt5": round(100 * sum(bool(P[i]["grounded"]) for i in kept) / n, 1)},
        "grounding_agree_pct": round(100 * sum(bool(G[i]["grounded"]) == bool(P[i]["grounded"]) for i in kept) / n, 1),
        "gemini1_gpt0": sum(1 for i in kept if G[i]["grounded"] and not P[i]["grounded"]),
        "gemini0_gpt1": sum(1 for i in kept if not G[i]["grounded"] and P[i]["grounded"]),
        "judges_disagree_on_grounding": sum(1 for i in kept if bool(G[i]["grounded"]) != bool(P[i]["grounded"])),
        "bootstrap": {"seed": SEED, "B": B},
    }
    costs = [P[i].get("cost_usd") for i in kept if P[i].get("cost_usd") is not None]
    if costs:
        S["gpt5_cost_usd"] = round(sum(costs), 3)
    print(json.dumps(S, ensure_ascii=False))
    print("Score: %.1f %%   95%% bootstrap CI: [%.1f, %.1f]   (n=%d tasks, %d scored 0 as ineligible)" % (score, lo, hi, n, inel))
    if S["missing"] or bad:
        print("warning: %d tasks are missing a usable verdict from one of the judges (%d incomplete GPT-5 rows)" % (S["missing"], bad), file=sys.stderr)
    if len(sys.argv) > 3:
        json.dump(S, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
