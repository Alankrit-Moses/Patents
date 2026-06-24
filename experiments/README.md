# Modular framework experiment harness

This directory contains a lightweight prototype for comparing three ways of executing the same pattern-analysis tasks:

- **Setup A:** minimal prose, one model call.
- **Setup B:** framework primer plus typed query, one model call.
- **Setup C:** one common framework-aware orchestrator with reusable worker actions.

The harness reads the existing `Patterns/` and `reports/text versions/` artifacts. It never reads `E_tab_code/` or `tables/` as generator inputs. Experiment 2 supplies only the compact saved CSV and the corresponding report; its saved `E.text` is reserved for evaluation.

## Quick start

From the `WIPO/` project root:

```powershell
python -m experiments.cli dry-run
python -m unittest discover -s experiments/tests -v
```

The dry run scans the corpus, builds the task manifest and all prompt variants, writes only prompt metadata/previews, and makes zero LLM calls. Outputs go to `experiments/artifacts/`.

Copy and edit the example configuration:

```powershell
Copy-Item experiments/config.example.json experiments/config.local.json
```

Then run one small task before launching the full matrix:

```powershell
python -m experiments.cli --config experiments/config.local.json run --setup A --task-id exp1-P1-genAI
python -m experiments.cli --config experiments/config.local.json run --setup B --task-id exp1-P1-genAI
python -m experiments.cli --config experiments/config.local.json run --setup C --task-id exp1-P1-genAI
```

Run a filtered or complete matrix:

```powershell
python -m experiments.cli --config experiments/config.local.json run --experiment 2 --setup all
python -m experiments.cli --config experiments/config.local.json run --setup all
```

Each run prints and saves its result JSONL path. Judge and summarize it with:

```powershell
python -m experiments.cli --config experiments/config.local.json judge --results experiments/artifacts/runs/results-TIMESTAMP.jsonl
python -m experiments.cli summarize --evaluations experiments/artifacts/evaluations/evaluations-TIMESTAMP.jsonl
```

## vLLM

The client calls the standard OpenAI-compatible `POST /v1/chat/completions` endpoint and has no third-party Python dependency. A typical local server can be started separately with:

```powershell
vllm serve YOUR_MODEL --api-key local
```

Set the exact served model name and endpoint in `config.local.json`. Generator and judge endpoints/models may differ. The default API key fallback is `local`; `api_key_env` can instead point to an environment variable.

## Task construction

The current corpus produces:

- Experiment 1: one task for every available pattern/report combination;
- Experiment 2: one task per matched compact CSV and saved text example;
- Experiment 3: deterministic small folds, using four induction pairs where available and the remainder as held-out positives.

Experiment 1 gold examples are non-exhaustive and are not used for recall. Experiment 3 held-out pairs remain task metadata for possible future experiments, but are not exposed to Setup A, B, C, or the current alignment judge.

## Setup C action library

`FrameworkAgent` first parses the query into known, target, irrelevant, and available-data slots. It then compiles the target into a plan using the fixed action library:

- `TabSignalAgent`
- `DescriptionInductionAgent`
- `PatternSynthesisAgent`
- `TextEvidenceAgent` with deterministic overlapping chunks and parallel LLM map calls
- `TextPatternVerifier`
- `TabTextVerifier`
- `DefinitionVerifier`
- `ReducerRanker`
- optional one-round `RepairAgent`

The reducer is deterministic. Candidate excerpts that cannot be recovered verbatim from their source chunk are discarded before verification.

Setup C workers and verifiers receive only the inputs materialized for the corresponding A/B task. For Experiment 3, the internal definition verifier may inspect the same induction pairs as the synthesizer, but never held-out pairs or hard negatives.

## Result and evaluation records

Every generation record contains the experiment, setup, pattern, framework query, safe input references, selected chunks, full Setup C trace, raw output, parsed output, and errors. Judges receive no setup label in their prompt. Evaluation JSONL retains the setup only for later aggregation.

Evaluation uses alignment scores only. Experiment 1 assigns an independent score to each requested E.text output against M. Experiment 2 scores E.text against its paired compact E.tab and diagnostic gold E.text. Experiment 3 independently scores generated M and T against their canonical components. Alignment scores are 1-5; an E.text output receives 0 without an LLM judge call when it is missing or cannot be recovered verbatim from the report.

Single-shot prompts are never silently truncated. If the configured context limit is too small for a full report, the run logs an error so that comparisons are not distorted by different report slices.

## Supplementary stochastic robustness runs

Keep the main reported experiment deterministic (`generator.temperature: 0.0`). The
separate robustness entry point repeats each selected task/setup at a non-zero
temperature and writes to `experiments/artifacts/robustness/`, leaving the main
runner and its result files unchanged:

```bash
python -m experiments.robustness --config experiments/config.local.json run \
  --temperature 0.2 --samples 3 --max-attempts 3 --setup all
```

The command uses the one generator model and endpoint named in the configuration,
so run it once for each model served by vLLM. Every record includes the model,
temperature, `sample_index`, and a unique `sample_id`. The ordinary blind judge
command can judge the resulting JSONL; sampling metadata is carried into the
evaluation JSONL:

Each requested sample is retried up to three total attempts by default. Only the
first successful attempt supplies that logical sample's output. Failed-attempt
errors and the attempt count remain attached to the record for auditability; if
all attempts fail, the logical sample is recorded as an error.

```bash
python -m experiments.cli --config experiments/config.judge-openai.json judge \
  --results experiments/artifacts/robustness/results-MODEL-TIMESTAMP.jsonl
```

Summarize that judged file with:

```bash
python -m experiments.robustness summarize \
  --evaluations experiments/artifacts/evaluations/evaluations-TIMESTAMP.jsonl
```

The robustness summary reports mean alignment, population standard deviation,
best-of-k mean, success@k at score 4 by default, and exact-excerpt retention for
Experiments 1 and 2. Filters such as `--experiment`, `--pattern`, `--task-id`, and
`--setup` are available on the robustness `run` command.
