# Evaluation results

`results_table.csv` reports mean scores for each model, query, and pattern.
Average columns give equal weight to P1, P2 and P3. Scores use a 0–5 scale
and include zero scores. These files contain the GPT-6.1 Terra evaluations.
Claude Opus 5 independently evaluated the same outputs. Its records, the
two-judge means and agreement are in [`opus_judge/`](../../opus_judge/README.md).

Each model directory contains `p1-p2-evaluations.jsonl`, `p3-evaluations.jsonl`,
and corresponding summaries. P1/P2 generator outputs are in
`results/<model>/outputs.jsonl`; P3 generator outputs are in
`p3-outputs.jsonl`. Each record retains its task and sample identifiers.

Q1 evaluates passages against M; Q2 evaluates passages against the input
table; Q3 evaluates generated M and T against the four example pairs.
Missing or ungrounded Q1/Q2 passages receive 0. Grounding uses exact or
normalized recovery followed by fuzzy recovery at threshold 0.8. Missing Q3
components receive 0. Each pattern mean includes 27 Q1 passage scores, 18
Q2 scores, and nine scores for each Q3 component.

P3's six curated pairs come from three reports, with examples sharing
incidents; its combined Q1 corpus overlaps the other two corpora. P3 task
inputs are in `data/p3/` and `experiments/tasks/p3.jsonl`. P3 output records
retain original local input paths, while the task file points to their
corresponding included materials. The GPT-OSS high P3 run used Google Vertex
through OpenRouter; see `experiments/configs/gpt-oss-high-vertex.json`.
