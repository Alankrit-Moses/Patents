# Pattern engineering framework and agentic executor

This repository is the public artifact for *A Vision for a Logic-Based Workflow
Formulation Framework and an Agentic Execution Pipeline for Pattern
Engineering*. It contains the single blackboard-based agentic executor evaluated
in the paper, the study tasks and data, and the reported outputs and evaluations.

## Repository contents

- [`experiments/`](experiments/) contains the executor, prompts, task
  construction, experiment runners, and evaluation code.
- [`results/`](results/) exposes the outputs, GPT-4.1 evaluations, and summaries
  for the four model settings reported in the paper.
- [`Patterns/`](Patterns/) contains the trajectory-shift (`P1`) and
  growth-divergence (`P2`) pattern cards. Each includes `M`, `T`, and six
  curated gold `(e.text,e.tab)` pairs.
- [`reports/pdfs/`](reports/pdfs/) contains the four WIPO Patent Landscape
  Reports used by Q1--Q3; [`reports/text versions/`](reports/text%20versions/)
  contains the corresponding text consumed by the executor.
- [`tables/`](tables/) contains the two WIPO companion workbooks required by the
  retained gold-pair extraction scripts.

## Evaluated tasks

The artifact instantiates three typed queries as 24 tasks over two patterns:

- **Q1 (example mining):** supply `M` and a report; request three `e.text`
  passages.
- **Q2 (evidence alignment):** supply `e.tab` and its source report; request its
  paired `e.text` passage.
- **Q3 (pattern induction):** supply four paired examples; request `M` and `T`.

Each task was sampled three times at temperature 0.2. The released results cover
GPT-4.1, GPT-OSS-120B with high and medium reasoning effort, and GPT-5.6-Luna
with reasoning disabled. GPT-4.1 was used as the blinded judge.

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

Commands for reproducing the three-sample runs, judging, and summarization are
in [`experiments/README.md`](experiments/README.md).

## Released results

Each model directory under [`results/`](results/) contains:

- `outputs.jsonl`: executor outputs and complete execution traces;
- `evaluations.jsonl`: grounded GPT-4.1 scores;
- `summary.json`: aggregate statistics used for the paper table.

[`results/paper_table.csv`](results/paper_table.csv) collects the values shown in
the paper. Historical `C`, `v2-agentic`, and `robustness_*` labels remain inside
some released records because they were recorded during the original runs; they
do not identify additional pipelines in this release.

## Data provenance

The report PDFs, extracted text, and companion workbooks were published by
WIPO. The curated pattern cards and example pairs are research artifacts
derived from those sources. Source files remain subject to their original
terms; this repository does not relicense third-party WIPO material.
