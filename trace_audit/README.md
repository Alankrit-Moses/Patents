# Trace audit

Every execution in this repository is stored with its full trace: each agent call, its output, and the orchestrator's stated reason for making it. This folder reads those traces to say at which step an example-alignment (Q2) execution succeeded or failed, and prints any execution step by step. The "Auditability" part of Section V of the paper reports the result for GPT-OSS-120B with high reasoning effort (OSS-High).

## Where the traces are

- `results/<model>/outputs.jsonl`: the P1 and P2 executions, one JSON record per line.
- `results/final/<model>/p3-outputs.jsonl`: the P3 executions.

A record is identified by its `sample_id`, and its `agent_trace` field lists the steps in order. Each step has an `action` and an `output`. `FrameworkAgent` is the orchestrator: its output holds the chosen action and the `reason` it gave. `E.text-Finder` is the finder: its output holds `chunk_results`, one entry per report chunk with the candidates proposed there or an `error`, and `pool`, the pooled candidates. `E.text-Reducer` is the selector: its output holds the reply, or an `error` when the reply could not be parsed, in which case the runtime falls back to confidence order. `T-Inducer` and `M-Inducer` hold the induced component.

## Stage attribution

Each Q2 task holds out one curated passage. `stage_attribution.py` compares that passage with what the trace records and places every execution in one of four stages. Two texts count as the same place in the report when they share at least 100 consecutive characters (or 60% of a shorter held-out passage), or when a candidate of 60 characters or more lies inside the held-out passage. Another passage can also describe the same instance, so the stage says where the held-out passage was lost, and the two judges' scores say whether the output was still acceptable.

Number of Q2 executions per stage, out of 54 for each model setting:

| Stage | GPT-4.1 | OSS-High | OSS-Medium | GPT-5.6-Luna |
|---|---:|---:|---:|---:|
| A. The returned excerpt comes from the held-out passage | 26 | 10 | 10 | 25 |
| B. The held-out passage was in the candidate pool, but another candidate was returned | 11 | 7 | 7 | 12 |
| C. The finder read the chunk holding the held-out passage and did not propose it | 17 | 32 | 37 | 17 |
| D. The finder's call on that chunk failed | 0 | 5 | 0 | 0 |
| Executions that both judges scored 2 or lower | 12 | 23 | 26 | 9 |
| of these, retrieval stage (C or D) | 8 | 22 | 25 | 7 |
| of these, selection stage (B) | 2 | 0 | 0 | 2 |

`stage_attribution_<model>.csv` lists every execution with its stage, plan, pool size, the chunk that holds the held-out passage, failed calls, and both judges' scores.

## The four traces quoted in the paper

All four are OSS-High executions. The paths below are inside the record with the given `sample_id`.

1. `exp2-P1-E3:C:sample-1` in `results/gpt-oss-high/outputs.jsonl` (stage C; scores 2 and 1). `agent_trace[3].output.chunk_results[2]` is the finder's call on the chunk that holds the held-out sentence: it proposed one other sentence, with `confidence` 0.3 and the `notes` quoted in the paper. `agent_trace[3].output.pool` is the pool of nine, and `agent_trace[2]` is the selector's choice.
2. `exp2-P2-E5:C:sample-1` in the same file (stage D; scores 0 and 0). `agent_trace[3].output.chunk_results[1]` and `[2]` are the two failed calls. `agent_trace[3].output.pool[0]` is the candidate at confidence 0.99; its numbers are those of the input table, `Patterns/P2/E_tab/genAI/P2-E5_Application_domain_growth_comparison.csv`, and it is not a span of the report.
3. `exp2-P3-E1:v2-agentic:sample-3` in `results/final/gpt-oss-high/p3-outputs.jsonl` (stage B; scores 3 and 2). `agent_trace[2].output.error` shows the selector's reply could not be parsed. `agent_trace[3].output.pool[12]` and `[14]` are the held-out sentences, and `pool[0]`, "the engine stopped", is what was returned.
4. `exp2-P3-E6:v2-agentic:sample-1` in the same P3 file (stage A; scores 4 and 4). `agent_trace[2]` is the failed first retrieval, `agent_trace[3].output.action.reason` the orchestrator's reason for inducing a description, `agent_trace[4]` that description, and `agent_trace[7]` the second retrieval.

Scores are in `results/final/gpt-oss-high/` (GPT-6.1 Terra) and `opus_judge/results/claude-opus-5/gpt-oss-high/` (Claude Opus 5).

## Running it

Place this folder at the root of the repository, or set `PATTERNS_REPO` to the repository path.

```
python3 -I stage_attribution.py                                # the OSS-High counts
python3 -I stage_attribution.py --model gpt-4.1 --write        # another setting, and rewrite its CSV
python3 -I stage_attribution.py --show exp2-P1-E3:C:sample-1   # one execution, step by step
```

The second judge's scores are read from `opus_judge/results/claude-opus-5` when that folder is present; without it, the script uses the GPT-6.1 Terra scores alone.
