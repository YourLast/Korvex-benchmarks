# Judging with Gemini 2.5 Flash on Kaggle (first official judge)

The official FACTS Grounding v2 notebook calls Gemini through **Kaggle Benchmarks** (`kaggle_benchmarks`, model name
`google/gemini-2.5-flash`). That module is available only in the Kaggle image used by benchmark notebooks. A standalone
notebook (even one that copies the official code) fails with `ModuleNotFoundError: kaggle_benchmarks`. What works is a
**new version of your own fork of the official starter notebook**, and that is what `make_kaggle_files.py` prepares.

You need a Kaggle account and the Kaggle command line tool (`pip install kaggle`, credentials set up as described in the
Kaggle API documentation). Benchmark model calls count against a quota that Kaggle sets per account. Our 200 answers went
through without errors under ours; yours may differ, so try `--n 5` first.

Run everything from the root of the kit; `work/` is a scratch folder (it is in `.gitignore`).

```bash
# 1. The official notebook: the source of the judge prompts and parsers (nothing of it is shipped in this kit)
kaggle kernels pull prathameshbang/facts-grounding-v2-benchmark-starter -p work/starter

# 2. Make your own fork of it. In the browser open
#      https://www.kaggle.com/code/prathameshbang/facts-grounding-v2-benchmark-starter
#    press "Copy & Edit", then save a version so that the fork exists under your account
#    (Kaggle's button names may differ; all that matters is that you end up with a copy of the benchmark notebook).
#    Then download the fork together with its metadata:
kaggle kernels pull YOUR_USERNAME/facts-grounding-v2-benchmark-starter -p work/fork -m
#    (the id of your fork is the "id" in work/fork/kernel-metadata.json; use it if it differs)

# 3. Build the judging notebook and the private dataset folder with your answers
python3 judge_gemini_kaggle/make_kaggle_files.py \
    --tasks work/tasks.jsonl --answers work/answers.jsonl \
    --owner YOUR_USERNAME --fork-dir work/fork \
    --starter work/starter/facts-grounding-v2-benchmark-starter.ipynb --n 5     # smoke test first, then drop --n

# 4. Upload the answers as a private dataset and wait until Kaggle has processed it
kaggle datasets create -p work/kaggle/dataset
kaggle datasets status YOUR_USERNAME/facts-v2-answers

# 5. Push the notebook as a new version of your fork (this runs it) and fetch the result
kaggle kernels push -p work/fork
kaggle kernels status YOUR_USERNAME/facts-grounding-v2-benchmark-starter
kaggle kernels output YOUR_USERNAME/facts-grounding-v2-benchmark-starter -p work/gemini_out
```

The result is `work/gemini_out/gemini25flash_judged.jsonl`: one row per answer with `quality_rating`, `eligible`,
`grounded` and the judge's raw replies (`raw_quality`, `raw_grounding`). With 5 parallel calls the 200 answers took us 27
minutes. After the smoke test, run step 3 again without `--n` and push again (the dataset already holds all
your answers; `--n` only limits what the notebook judges).

What the generated notebook contains:

* three code cells copied **verbatim** from the official starter notebook you downloaded (imports, the quality filter
  `_evaluate_quality_no_context`, the grounding prompt `UFG_REV21_PROMPT` with `classify_grounding`);
* one loop of ours (MIT) that calls those functions for every answer: first the judge answers the task itself (the
  baseline), then rates the test answer for instruction following, then the grounding check runs for every answer, because
  with two judges an answer is ineligible only if **both** say "Major Issue(s)" (see `score_official_pair.py`).

Things to know:

* The official helper functions swallow API errors and carry on with an empty reply, which would silently turn an outage into
  "not grounded". Our loop rejects empty replies and records them as `error`; the scorer ignores such rows. Re-judge only
  those with `--ids-file ids.txt` (one task id per line) and concatenate the outputs; the last good row for a task wins.
* If the official notebook is updated after our run (we used version 4), look at what changed before comparing numbers.
* After the judging, `python3 verify_verdicts.py ...` re-parses the stored raw replies with the official parsers and checks
  that the stored verdicts agree.
