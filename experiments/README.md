# Agentic executor and evaluation code

This directory contains the single executor evaluated in the paper. Historical
single-prompt baselines and full-context retrieval variants are intentionally
excluded from the public artifact.

## Directory map

- `executor.py`: orchestrator, blackboard, M-Inducer, T-Inducer,
  e.text-finder, e.text-selector, and their exact prompts.
- `runner.py`: executes the selected framework task once, including retries,
  output validation, concurrency, and JSONL writing.
- `sampling.py`: invokes `runner.py` three times per task at temperature 0.2 and
  computes the sample-aware summaries reported in the paper. It is an
  experiment wrapper, not a second pipeline.
- `task_construction.py`: discovers the study data and constructs Q1--Q3.
- `task_inputs.py`: resolves task slot references to the supplied materials.
- `evaluation.py`: report-span grounding, GPT-4.1 judge prompts, and scoring.
- `reports.py`: overlapping report chunks and source-span recovery.
- `client.py`, `config.py`, and `io_utils.py`: shared API, configuration, and
  file utilities.
- `cli.py`: commands for constructing tasks and judging outputs.
- `configs/`: the four reported generator settings and GPT-4.1 judge setting.
- `tasks/`: the released material inventory and 24 concrete tasks.
- `tests/`: tests for task construction, execution, grounding, evaluation, and
  repeated sampling.
- `runs/`: ignored destination for newly generated outputs.
- `EXECUTOR_PROTOCOL.md`: readable description of the communication protocol.

The prompts used by the executor are string constants in `executor.py`; judge
prompts are in `evaluation.py`. There is no verifier and no e.tab-specialist in
the evaluated executor.

For fidelity to the evaluated runs, the prompt text retains the internal labels
`Framework Agent`, `E.text-Finder`, and `E.text Selector`. These are the
orchestrator, e.text-finder, and e.text-selector roles named in the paper.

## Task construction

- Q1 supplies `M` and one report and requests three `e.text` passages.
- Q2 supplies one `e.tab` and its source report and requests the paired
  `e.text` passage. The saved gold passage is withheld from generation.
- Q3 supplies four gold example pairs and requests `M` and `T`.

[`tasks/tasks.jsonl`](tasks/tasks.jsonl) contains 24 tasks: six Q1, twelve Q2,
and six Q3. [`tasks/manifest.json`](tasks/manifest.json) inventories the source
materials from which those tasks are constructed.

## Run one task

From the repository root:

```powershell
python -m experiments.cli --config experiments/configs/gpt-4.1.json manifest
python -m experiments.runner --config experiments/configs/gpt-4.1.json run `
  --task-id exp1-P1-genAI
```

`runner.py` produces one execution for each selected task. To reproduce the
paper's three independently sampled executions, use `sampling.py`:

```powershell
python -m experiments.sampling --config experiments/configs/gpt-4.1.json run `
  --temperature 0.2 --samples 3 --max-attempts 3 `
  --output experiments/runs/gpt-4.1/outputs.jsonl
```

Judge and summarize those outputs:

```powershell
$env:GROUNDING_SNAP_THRESHOLD = "0.8"
python -m experiments.cli --config experiments/configs/judge-gpt-4.1.json judge `
  --outputs experiments/runs/gpt-4.1/outputs.jsonl `
  --output experiments/runs/gpt-4.1/evaluations.jsonl
python -m experiments.sampling summarize `
  --evaluations experiments/runs/gpt-4.1/evaluations.jsonl `
  --output experiments/runs/gpt-4.1/summary.json
```

The paper outputs are published separately under [`../results/`](../results/).
Their contents retain historical run metadata such as `"setup": "C"`,
`"prompt_version": "v2-agentic"`, and `robustness_*`; these labels describe the
original run records and do not denote additional public implementations.
