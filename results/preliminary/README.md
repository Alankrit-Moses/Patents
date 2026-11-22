# Preliminary revised evaluations and P3 pilot

These files are exploratory results, separate from the original reported paper
results in the parent directory. GPT-6.1 Terra, reasoning disabled, judged the
saved answers. P1/P2 generation outputs remain in `results/<model>/outputs.jsonl`;
P3 generation outputs are included here. Scores include zeros.

Each model directory contains:

- `p1-p2-evaluations.jsonl` and `p1-p2-summary.json`: revised 0–5 judging of
  the original P1/P2 answers, including the earlier example-based Q3 M prompt.
- `p3-outputs.jsonl`, `p3-evaluations.jsonl`, and `p3-summary.json`: the P3
  pilot and its revised 0–5 judging, also with the earlier Q3 M prompt.
- `p1-p2-q3-m-moderate-evaluations.jsonl` and its summary: the later,
  moderately strict Q3 M prompt applied to the P1/P2 answers.
- `p3-q3-m-moderate-evaluations.jsonl` and its summary: that same M prompt
  applied to P3. These files contain M scores only.

`preliminary_table.csv` combines Q1, Q2, and Q3 T from the revised evaluations
with Q3 M from the later moderate prompt. Its Avg columns are unweighted means
of P1, P2 and P3. The combinations are explicit: do not mix the two M rubrics
or treat this as one uniformly rejudged file. Each per-pattern Q1 mean has
27 passage scores, Q2 has 18, and each Q3 component has 9.

Q1/Q2 zero scores reflect missing or unrecovered excerpts under the 0.8 fuzzy
grounding setting. Q3 missing components get zero. Model judgements have no
saved rationales, and prompt sensitivity is significant. Outputs and scores
are samples, not independent estimates of broad real-world reliability.

P3 is an initial causal-chain study: six pairs arise from three reports, with
overlap within two incidents. Its three Q1 inputs include one combined corpus
that overlaps the other two. The P3 model inputs and task definition are in
`data/p3/` and `experiments/tasks/p3.jsonl`; historical P3 output records retain
the original `experiments/runs/p3/` input paths. The high-reasoning GPT-OSS P3
run used Google Vertex through OpenRouter and a 180-second timeout, unlike the
original paper's provider routing; see `experiments/configs/gpt-oss-high-vertex.json`.
