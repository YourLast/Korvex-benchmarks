# Korvex on FACTS Grounding: reproducibility kit

Scripts and data to check, or repeat, our measurement of **DSR Etude 0 Preview** on **FACTS Grounding v2**, the Google
DeepMind benchmark for answers that stay within a document. Our write-up, with the official table next to our number:
<https://korvexai.site/en/articles/benchmarks/>. Русская версия этого файла: [README.ru.md](README.ru.md).

## Result

**87.2 %** of the answers fully supported by the document: 200 tasks from the public half of the set, graded by the official
judge pair (Gemini 2.5 Flash and GPT-5, `gpt-5-2025-08-07`) with the official prompts and the official scoring rule.
95 % bootstrap confidence interval **83.0 to 91.2** (over tasks, 10,000 resamples, seed 20261104). 2 of 200 answers scored 0
for failing the task (both judges found a major problem), and the judges disagreed on grounding for 21 of 200 answers.
Answers were produced on 6-7 November 2026, judged on 8 November 2026.

## What FACTS Grounding tests

The model gets a document of up to 32k tokens and a task about it. An answer counts only if every sentence in it is supported
by the document. An answer that fails the task itself scores 0 before the facts are checked.

## How we measured

* **Tasks.** 200 from the public half of the set: positions 200-399 in a fixed random order. We did not open these tasks
  before the run. Their row indices in the official `examples.csv` are in [`tasks/tasks200_idx.txt`](tasks/tasks200_idx.txt).
* **System.** DSR Etude 0 Preview, Max mode. Memory is the only tool: no code, web or browser. The task is sent as in the
  official notebook: instruction, document and request in one message (the `full_prompt` column), unchanged (in 3 of 200 tasks,
  `facts-160`, `facts-243` and `facts-319`, overlong lines were wrapped: lines above 20,000 characters are cut at a space
  and continued on the next line; `run_answers.py` does the same). One run.
* **Judges: the official pair.** Gemini 2.5 Flash via Kaggle Benchmarks, as in the official FACTS notebook. GPT-5
  (`gpt-5-2025-08-07`) via the OpenAI API with default settings. The judge prompts and the parsing of their replies are taken
  verbatim from the official FACTS Grounding v2 notebook on Kaggle. This kit does not contain that notebook: the scripts load
  the prompts and parsers from the copy you download yourself.
* **Scoring: the official rule.** An answer scores 0 for failing the task only if both judges find a major problem. Otherwise
  a task's score is the mean of the two judges (each judge: 1 if every sentence is supported, else 0).
* **What the judge sees.** The text of the answer after the last `</think>` marker, stripped (only one of the 200 answers had
  such a marker; the API normally returns the reasoning separately).

## How our measurement differs from the official one

* **200 tasks instead of 1,715.** So the margin of error is about ±4 points, against about ±2 for the models in the table. On
  the board, the public and private halves differ by 1 point on average.
* **We measure a system, not a bare model.** Kaggle runs the models in the table itself. Ours runs through our server; the task
  text is the same.
* **Max mode.** Kaggle does not publish the reasoning settings it uses for the models in the table.
* **One run.**

If you repeat the measurement through the public API, three more things apply:

* **Max is currently unavailable through the API.** The max level is temporarily closed in the public API: it is too
  expensive for our servers. A request with `"reasoning_effort": "max"` gets a 400 `reasoning_effort_unavailable` error.
  Re-scoring our published answers (step 5) gives 87.2% without the API. A fresh run through the API is possible at `xhigh`
  for now (`--effort xhigh`); its numbers are not comparable with ours (Max).
  Numbers obtained at another level are not comparable with ours.
* **The service changes.** The published answers come from the service as it was on 6-7 November 2026. A later run measures
  whatever version is running then.
* **The API is not the instance we measured on.** We ran a dedicated instance with only the memory tool enabled; a plain API
  request (this kit sends no tools) goes to the public service.

## What is in the kit

```
README.md, README.ru.md       this file
LICENSE                       MIT, for the scripts only
run_answers.py                answers from the Korvex API (standard library only, resumable)
tasks/tasks200_idx.txt      the 200 row indices of examples.csv, in run order
tasks/build_tasks.py          examples.csv + index list -> tasks.jsonl
judge_gemini_kaggle/          Gemini 2.5 Flash judge: notebook generator and step-by-step instructions
judge_gpt5.py                 GPT-5 judge through the OpenAI API, with a hard budget cap
score_official_pair.py        the official pair score with a bootstrap interval
verify_verdicts.py            re-parses stored judge replies with the official parsers (no API calls)
official_cells.py             loads the judge prompts/parsers from the official notebook you downloaded
results/answers.jsonl         our 200 answers: {id, idx, response}
results/gemini_verdicts.jsonl Gemini verdicts: {id, quality_rating, eligible, grounded, raw_quality, raw_grounding}
results/gpt5_verdicts.jsonl   GPT-5 verdicts, same fields (+ quality_called; GPT-5 rates quality only where Gemini said Major)
results/summary.json          output of score_official_pair.py on the files above
```

Python 3.9 or newer, no packages to install (the Kaggle command line tool is needed only for the Gemini step).
The scripts read keys only from environment variables and never print them.

## Quick check: re-score our answers (free, offline)

This reproduces **87.2** and **[83.0, 91.2]** exactly; the scoring is deterministic:

```bash
python3 score_official_pair.py results/gemini_verdicts.jsonl results/gpt5_verdicts.jsonl
# Score: 87.2 %   95% bootstrap CI: [83.0, 91.2]   (n=200 tasks, 2 scored 0 as ineligible)
```

To see that the stored verdicts really follow from the judges' raw replies, download the official notebook (free, needs a
Kaggle account) and run the official parsers over them; this makes no API calls and judges nothing again:

```bash
kaggle kernels pull prathameshbang/facts-grounding-v2-benchmark-starter -p work/starter
python3 verify_verdicts.py --starter work/starter/facts-grounding-v2-benchmark-starter.ipynb \
    --gemini results/gemini_verdicts.jsonl --gpt5 results/gpt5_verdicts.jsonl
```

## Reproduce from scratch

You need: a Korvex API key (`kx-...`; Max is temporarily closed in the API, see above), a Kaggle account, and an OpenAI API key that can use
`gpt-5-2025-08-07`. Note that OpenAI lists this snapshot for shutdown on 2026-12-11
([deprecations](https://platform.openai.com/docs/deprecations)); after that date the GPT-5 step cannot be repeated as
published, and only the re-scoring above remains.

What it takes, from our run (indicative, your numbers will differ):

| Step | Time | Cost |
|---|---|---|
| Answers, 200 tasks at Max, 2 parallel requests | mean 5.5 min per task (median 4.5, longest 29), about 9 hours in all | counts against your Korvex key |
| Gemini 2.5 Flash on Kaggle, 5 parallel calls | 27 minutes | inside your Kaggle benchmark quota |
| GPT-5 through OpenAI, 4 parallel calls | about 30 minutes | about $12 for the 200 answers (about $0.06 per task) |

**1. Download the official examples** (856 rows; the dataset is CC BY 4.0) and build the task file:

```bash
mkdir -p work && kaggle datasets download -d deepmind/facts-grounding-examples -p work --unzip
python3 tasks/build_tasks.py --examples work/examples.csv --idx tasks/tasks200_idx.txt --out work/tasks.jsonl
```

The script warns if your `examples.csv` is not byte-identical to the one we used (SHA-256 starting `66990aea`, dataset
version 17), because row indices could then point to other tasks.

**2. Get the answers from the API.** Public base URL `https://api.korvexai.site/v1`, model `dsr-etude-0-preview` (also
`dsr-etude-latest`, which always points to the current version), OpenAI-compatible `POST /v1/chat/completions`:

```bash
export KORVEX_API_KEY="kx-<your key>"
python3 run_answers.py --tasks work/tasks.jsonl --out work/answers.jsonl --dry-run           # shows the request, sends nothing
python3 run_answers.py --tasks work/tasks.jsonl --out work/answers.jsonl --limit 3           # three tasks first
python3 run_answers.py --tasks work/tasks.jsonl --out work/answers.jsonl --workers 2         # all 200; run again to resume
```

Each request is exactly this (one per task, no system message, no tools, no sampling parameters):

```json
{"model": "dsr-etude-0-preview", "stream": false, "reasoning_effort": "max",
 "messages": [{"role": "user", "content": "<full_prompt of the task>"}]}
```

with headers `Authorization: Bearer $KORVEX_API_KEY` and `Content-Type: application/json`. No conversation header is sent, so
every task is an independent one-off request. The key has a requests-per-minute limit and a limit on parallel requests; if
you see 429 answers use fewer `--workers` (the script retries by itself).

**3. Judge with Gemini 2.5 Flash on Kaggle.** Follow [judge_gemini_kaggle/README.md](judge_gemini_kaggle/README.md). The result
is `work/gemini_out/gemini25flash_judged.jsonl`.

**4. Judge with GPT-5 through OpenAI.** The prompts come from the official notebook you downloaded. `--budget-usd` is a hard
cap on spending (no new call starts once it is reached; check `--dry-run` first):

```bash
export OPENAI_API_KEY="sk-<your key>"
python3 judge_gpt5.py --tasks work/tasks.jsonl --answers work/answers.jsonl \
    --gemini work/gemini_out/gemini25flash_judged.jsonl \
    --starter work/starter/facts-grounding-v2-benchmark-starter.ipynb \
    --out work/gpt5_verdicts.jsonl --budget-usd 20 --dry-run
```

Drop `--dry-run` to judge. As in the official notebook, GPT-5 rates instruction following only for answers where Gemini found
a major problem, and checks grounding for every answer. API settings are the defaults (only `model` and `messages`).

**5. Score:**

```bash
python3 score_official_pair.py work/gemini_out/gemini25flash_judged.jsonl work/gpt5_verdicts.jsonl
```

## What to expect

* **Re-scoring our files** gives 87.2 [83.0, 91.2] exactly.
* **Re-judging our answers** will not give exactly the same verdicts: judge calls are not deterministic, and a few borderline
  sentences flip. Expect a result near ours, not identical.
* **A fresh run of the answers** will not reproduce 87.2 either: answers differ from run to run. With 200 tasks one run has a
  margin of about ±4 points, so a new run should land close to our interval 83.0-91.2 (a little outside is possible by chance
  alone, and a changed service version can move it too).

## Limits of this evidence

200 tasks of the public half, not the full 1,715; one run; a system measured through our own server, not a bare model;
Max mode while the settings of the board's models are unpublished; the official judges are themselves imperfect (the two
judges disagreed on grounding for 21 of 200 answers: Gemini found an answer grounded and GPT-5 not in 14 cases, the reverse in 7).
Our system is not on the Kaggle board, and Kaggle did not run it.

## License and attribution

* The scripts in this repository: MIT, see [LICENSE](LICENSE).
* FACTS Grounding examples: (c) Google DeepMind and Google Research, CC BY 4.0. `results/` contains answers to those tasks and
  judge replies that quote the documents. Please cite the FACTS Grounding papers (arXiv:2501.03200, arXiv:2512.10791).
* The official starter notebook (`prathameshbang/facts-grounding-v2-benchmark-starter`) belongs to its authors and is not
  redistributed here; the scripts use your own downloaded copy.
