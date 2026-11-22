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
- `evaluation.py`: report-span grounding, configurable model judge prompts, and scoring.
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

## Revised evaluation rubric

New judge runs default to `example_alignment_0_5_moderate_m_v2`: 0 means missing or ungrounded; 1 incorrect or unrelated; 2 weakly related; 3 partially correct; 4 substantially correct with minor omissions or ambiguity; and 5 fully correct. Q1 compares grounded passages with M and accepts clear qualitative prose. Q2 compares passages directly with E.tab without a reference passage. Q3 assesses M and T separately against the four supplied example pairs, without canonical answers. Revised scores are not directly comparable to the published 0–5 evaluations because the criteria and judge can differ. The summarizer defaults to a success threshold of 4 for 0–5 scores and 3 for the earlier revised 0–3 rubric, and rejects mixed rubrics. To reproduce the earlier 0–3 rubric, set `EVALUATION_SCALE=3`. Use a new output file for each judging run: judging appends to the output JSONL.

## P3 preliminary experiment

P3 uses the explicit task file rather than automatic task construction, preserving
its three corpus inputs and Q3 example subsets. Its three Q1 corpora overlap,
and its six examples derive from three reports. The passages are curated, not
independent incidents. See [`../data/p3/README.md`](../data/p3/README.md).

```powershell
python -m experiments.sampling --config experiments/configs/gpt-5.6-luna.json run `
  --tasks experiments/tasks/p3.jsonl --temperature 0.2 --samples 3 --max-attempts 3 `
  --output experiments/runs/p3-new/outputs.jsonl --resume
$env:GROUNDING_SNAP_THRESHOLD = "0.8"
python -m experiments.cli --config experiments/configs/judge-gpt-6.1-terra.json judge `
  --tasks experiments/tasks/p3.jsonl --outputs experiments/runs/p3-new/outputs.jsonl `
  --output experiments/runs/p3-new/evaluations.jsonl
python -m experiments.sampling summarize --evaluations experiments/runs/p3-new/evaluations.jsonl `
  --output experiments/runs/p3-new/summary.json
```

Q3 M defaults to the moderate rubric tested in the preliminary comparison; Q1,
Q2 and Q3 T retain the prior revised prompt. Set `Q3_M_RUBRIC=original` to
reproduce the original revised example-based M prompt (distinct from the older
canonical-reference paper rubric). The 5-point Q1, Q2 and T prompts retain
the earlier 3-point description followed by a 5-point override for fidelity
to the recorded preliminary runs. Do not mix scores produced by different
rubrics in a single summary. Judge output is appended; use a new output path
for each run. Historical P3 outputs retain their original local `inputs_used`
paths; their equivalent materials are in `data/p3/`, and the submission task
file is authoritative for reruns and evaluation.
