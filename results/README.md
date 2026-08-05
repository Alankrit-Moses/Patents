# Paper results

This directory exposes only the agentic-executor runs reported in the paper.
Each model directory contains:

- `outputs.jsonl`: three executor outputs for each of the 24 tasks;
- `evaluations.jsonl`: GPT-4.1 evaluations of those outputs;
- `summary.json`: grouped statistics used for the paper table.

`paper_table.csv` reproduces the values displayed in the paper. Historical `C`,
`v2-agentic`, and `robustness_*` labels remain inside some records because they
were recorded during the original experiments; they all refer to the sole
executor released here.

## Reported settings

| Directory | Generator setting |
|---|---|
| `gpt-4.1` | GPT-4.1, temperature 0.2 |
| `gpt-oss-high` | GPT-OSS-120B, high reasoning effort, temperature 0.2 |
| `gpt-oss-medium` | GPT-OSS-120B, medium reasoning effort, temperature 0.2 |
| `gpt-5.6-luna` | GPT-5.6-Luna, reasoning disabled, temperature 0.2 |

Every task has three independently judged samples. All evaluations use GPT-4.1
at temperature 0. For Q1 and Q2, exact or normalized report-span recovery is
attempted first; fuzzy recovery uses `GROUNDING_SNAP_THRESHOLD=0.8`.

The paper reports the unweighted mean of P1 and P2 after computing each pattern
mean.

## Provenance note

The GPT-OSS-Medium run predates the other finalized configurations. It used an
8,192-token output cap, a 98,304-token context setting, 16,000-character chunks
with 2,000-character overlap, and OpenRouter's unrestricted provider routing.
The released [`gpt-oss-medium.json`](../experiments/configs/gpt-oss-medium.json)
records those settings. The other three configurations use 50,000-character
chunks with 3,000-character overlap.
