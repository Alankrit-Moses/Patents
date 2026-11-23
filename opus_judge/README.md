# Claude Opus 5 evaluation

Claude Opus 5 independently scored the released P1–P3 outputs with the same
task-specific judge instructions and grounding rules as GPT-6.1 Terra. The
submitted run used Claude Code with extended thinking disabled and high effort.
Claude Code adds an environment block to the judge messages. The API backend
sends the judge messages with a JSON schema. The two backends can produce
different scores.

`results/claude-opus-5/` contains per-model evaluation records and summaries,
the Claude table (`results_table.csv`), the two-judge mean
(`results_table_mean.csv`), the three-panel LaTeX table and judge agreement.
The original raw judge-call log is not included. Consequently the published
scores can be aggregated and compared from the evaluation records, but cannot
be replayed from raw Claude replies with this artifact.

From the repository root, without model calls:

```sh
python -I opus_judge/make_table.py
python -I opus_judge/agreement.py
python -I trace_audit/stage_attribution.py
```

Add `--write` to the first two commands to regenerate their saved outputs.
The agreement calculation excludes the 46 zero scores assigned by the shared
grounding rule and reports task-bootstrap intervals.

To run a new evaluation, use a new output directory. Claude Code mode requires
an installed and authenticated `claude` CLI; API mode additionally requires
the Anthropic Python SDK and `ANTHROPIC_API_KEY`.

```sh
python -I opus_judge/run_opus_judge.py --dry-run
python -I opus_judge/run_opus_judge.py --backend claude-code --model claude-opus-5 --out experiments/runs/claude-new
python -I opus_judge/run_opus_judge.py --backend api --model claude-opus-5 --thinking disabled --out experiments/runs/claude-api-new
```

New runs may differ from the reported results. Never commit credentials.
