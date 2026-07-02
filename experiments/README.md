# Modular framework experiment harness (v2)

This directory compares ways of executing the same pattern-analysis tasks. Only
the **v2** setups are supported; the earlier v1 code and results were removed.

- **Setup A:** minimal prose, one model call.
- **Setup B:** framework specification plus the typed framework query, one model
  call. Two prompt variants:
  - `query-only` — the query alone specifies the task.
  - `resolved` — the query plus an explicit resolved-task block.
- **Setup C:** an agentic orchestrator (Framework Agent + specialist agents). See
  [`setupC_v2.txt`](setupC_v2.txt) for the full specification. Two finder variants:
  - `map-reduce` (default) — the E.text-Finder maps over overlapping report
    chunks, then a reduce step selects the top-k.
  - `full-context` — the E.text-Finder reads the entire report in one call and
    returns its own final ranked selection (no map, no reduce). The planner,
    inducers, and blackboard are identical, so comparing the two isolates
    whether map-reduce retrieval helps. Like A/B, the report is never silently
    truncated: an over-limit prompt records an error.

The harness reads the existing `Patterns/` and `reports/text versions/`
artifacts. Experiment 2 supplies only the compact saved CSV and the corresponding
report; its saved `E.text` is reserved for evaluation.

## Setups A, B, C

- **A / B** are single prompts built in `prompts_v2.py` and executed by
  `runner_v2.py`.
- **C** is the agentic pipeline in `agent_v2.py`. The Framework Agent parses the
  typed query, then invokes specialists dynamically:
  - `M-Inducer` / `T-Inducer` — induce a definition `M` or description `T` (or the
    internal hypotheses `M_hat` / `T_hat`) from the available components.
  - `E.text-Finder` — a **map-reduce** search: it maps over overlapping report
    chunks to surface candidate spans (recall), then a reduce step selects the
    top-k best matches (precision) using `M`/`T` guidance.
  - There is **no separate verifier**; specialist outputs are used directly, and
    the blind judge enforces verbatim compliance at evaluation.

## Quick start

From the project root, build the corpus/tasks and run the tests:

```powershell
python -m experiments.cli --config experiments/config.openrouter-gpt-oss-120b.json manifest
python -m unittest discover -s experiments/tests -v
```

Set your OpenRouter key once:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

Run one task, then a filtered or full matrix (deterministic; `temperature: 0.0`):

```powershell
# Setup A / B / C on a single task
python -m experiments.runner_v2 --config experiments/config.openrouter-gpt-oss-120b.json run --setup A --task-id exp1-P1-genAI
python -m experiments.runner_v2 --config experiments/config.openrouter-gpt-oss-120b.json run --setup B --b-variant resolved --task-id exp1-P1-genAI
python -m experiments.runner_v2 --config experiments/config.openrouter-gpt-oss-120b.json run --setup C --task-id exp1-P1-genAI
python -m experiments.runner_v2 --config experiments/config.openrouter-gpt-oss-120b.json run --setup C --c-variant full-context --task-id exp1-P1-genAI

# Full 24-task matrix for one setup (append-safe with --resume)
python -m experiments.runner_v2 --config experiments/config.openrouter-gpt-oss-120b.json run --setup C --experiment all --max-attempts 3 --resume --output experiments/artifacts/openrouter/runs/results-v2-C.jsonl
```

## Judge and summarize

The blind judge is model-agnostic; point it at any result JSONL:

```powershell
python -m experiments.cli --config experiments/config.judge-openai-gpt41-rate-limit.json judge --results <RESULTS>.jsonl --output <EVAL>.jsonl
python -m experiments.cli summarize --evaluations <EVAL>.jsonl
```

Evaluation uses 1-5 alignment scores. Experiment 1 scores each requested `E.text`
against `M`; Experiment 2 scores `E.text` against the paired compact `E.tab` and a
diagnostic gold `E.text`; Experiment 3 independently scores generated `M` and `T`
against their canonical components. An `E.text` output receives **0 without an LLM
judge call** when it is missing or cannot be recovered from the report.

Verbatim recovery is **canonical** (case- and unicode-punctuation tolerant) and
returns the real report span for scoring. Setting the environment variable
`GROUNDING_SNAP_THRESHOLD` (e.g. `0.8`) additionally enables **grounding+snap**:
a non-verbatim excerpt is fuzzy-snapped to its best-matching report passage
(longest-contiguous-block ≥ threshold) and that real passage is judged, so
copy-fidelity is decoupled from retrieval while hallucinations still score 0.

## OpenRouter configs

Result files from the OpenRouter configs are written under
`experiments/artifacts/openrouter/`. The gpt-oss-120b configs keep
`response_format: {"type":"json_object"}`, use longer retry backoff for 429/5xx,
and the `*-trusted-providers*` variants pin a provider allow-list. Edit the
`judge` block (or use a dedicated judge config such as
`config.judge-openai-gpt41-rate-limit.json`) to choose the judge endpoint.

## Task construction

- Experiment 1: one task per pattern/report combination.
- Experiment 2: one task per matched compact CSV and saved text example.
- Experiment 3: deterministic small folds using induction pairs, with the
  remainder as held-out positives (held-out pairs are never exposed to any setup
  or to the judge).

## Result and evaluation records

Every generation record contains the experiment, setup, prompt version, pattern,
framework query, safe input references, selected chunks, the full Setup C agent
trace, raw output, parsed output, and errors. Judges receive no setup label.
Single-shot prompts (A/B) are never silently truncated: if the context limit is
too small for a full report, the run logs an error rather than distorting the
comparison.

## Supplementary stochastic robustness runs

The robustness entry point repeats each selected task/setup at a non-zero
temperature and writes to `experiments/artifacts/openrouter/robustness/`. It
supports setups A, B, and C:

```bash
python -m experiments.robustness --config experiments/config.openrouter-gpt-oss-120b.json run \
  --setup C --temperature 0.2 --samples 3 --max-attempts 3 --resume
```

Every record includes the model, temperature, `sample_index`, and a unique
`sample_id`; `--resume` skips already-completed samples. Each logical sample is
retried up to `--max-attempts` (default 3). Judge the resulting JSONL with the
ordinary `cli judge` command, then summarize:

```bash
python -m experiments.robustness summarize --evaluations <EVAL>.jsonl
```

The robustness summary reports mean alignment, population standard deviation,
best-of-k mean, success@k (score ≥ 4 by default), and exact-excerpt retention for
Experiments 1 and 2, split by pattern. Filters `--experiment`, `--pattern`,
`--task-id`, and `--setup` are available on `run`.
