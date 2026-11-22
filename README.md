# Pattern engineering framework and agentic executor

This repository is the public artifact for *A Vision for a Logic-Based Workflow
Formulation Framework and an Agentic Execution Pipeline for Pattern
Engineering*. It contains the single blackboard-based agentic executor evaluated
in the paper, the study tasks and data, and the reported outputs and evaluations.

## Evaluation artifact

The study covers three patterns and three query types, with four generator
settings. Task inputs, executor outputs, evaluation records, and the selected
results table are included in this repository. The evaluation uses GPT-6.1
Terra as judge and a 0–5 scale. See [`results/final/`](results/final/README.md)
for results and [`experiments/README.md`](experiments/README.md) for commands.

## Repository contents

- [`experiments/`](experiments/) contains the executor, prompts, task
  construction, experiment runners, and evaluation code.
- [`results/final/`](results/final/README.md) contains the selected evaluations and results table.
- [`data/p3/`](data/p3/) contains the P3 causal-chain task materials.
- [`Patterns/`](Patterns/) contains the trajectory-shift (`P1`) and
  growth-divergence (`P2`) pattern cards. Each includes `M`, `T`, and six
  curated gold `(e.text,e.tab)` pairs.
- [`reports/pdfs/`](reports/pdfs/) contains the four WIPO Patent Landscape
  Reports used by Q1--Q3; [`reports/text versions/`](reports/text%20versions/)
  contains the corresponding text consumed by the executor.
- [`tables/`](tables/) contains the two WIPO companion workbooks required by the
  retained gold-pair extraction scripts.

## Evaluated tasks

The artifact includes P1 (trajectory shift), P2 (growth divergence), and P3
(causal chains). For each pattern it specifies three Q1 tasks, six Q2 tasks,
and three Q3 tasks. Each task is sampled three times at temperature 0.2.

- **Q1 (example mining):** supply M and source text; request three passages.
- **Q2 (evidence alignment):** supply `e.tab` and source text; request a passage
  describing that instance.
- **Q3 (pattern induction):** supply four example pairs; request M and T.

The generator settings are GPT-4.1, GPT-OSS-120B with high and medium
reasoning, and GPT-5.6 Luna with reasoning disabled. The judge is GPT-6.1
Terra with reasoning disabled.

## Quick start

Requirements are Python 3.10+ and API access for the selected model. The
executor uses the Python standard library; `openpyxl` is needed only to rerun
the gold `e.tab` extraction scripts.

```powershell
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "..."
$env:OPENROUTER_API_KEY = "..."
python -m unittest discover -s experiments/tests -v
python -m experiments.cli --config experiments/configs/gpt-4.1.json manifest
python -m experiments.runner --config experiments/configs/gpt-4.1.json run --task-id exp1-P1-genAI
```

Commands for reproducing the three-sample runs, judging, P3, and summarization are
in [`experiments/README.md`](experiments/README.md).

## Results

[`results/final/results_table.csv`](results/final/results_table.csv) reports
per-pattern scores and equally weighted pattern means. Per-model files contain
the evaluation records and summaries; P1/P2 outputs are under `results/<model>/`
and P3 outputs are in `results/final/<model>/`.

## Data provenance

The report PDFs, extracted text, and companion workbooks were published by
WIPO. The curated pattern cards and example pairs are research artifacts
derived from those sources. Source files remain subject to their original
terms; this repository does not relicense third-party WIPO material.

P3 source materials include an ATSB investigation and Cloudflare postmortems;
see [`data/p3/README.md`](data/p3/README.md) for attribution and redistribution
notes. Check third-party reuse rights before distributing this extension.
