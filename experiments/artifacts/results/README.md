# Consolidated results

All framework-experiment results, organized by **model → run-type → setup**.

```
results/
  <model>/
    deterministic/            # temperature 0.0, single draw
      <setup>.results.jsonl   # generator output (one record per task; C includes agent_trace)
      <setup>.eval.jsonl      # per-record judge scores
      <setup>.summary.json    # aggregated means per experiment
    robustness/               # temperature 0.2, k=3 samples per task
      <setup>.results.jsonl   # one record per (task, sample)
      <setup>.eval.jsonl      # per-sample judge scores
      <setup>.summary.json    # mean, population_stddev, best_of_k, success@k, exact-excerpt retention (per pattern)
  consolidated_matrix.csv     # long format: model, setup, run_type, experiment, scope(avg/P1/P2), value
  consolidated_matrix.json    # nested: model -> setup -> run_type -> experiment -> {avg,P1,P2}
```

## Axes
- **Models:** `gpt-4.1`, `gpt-oss-medium`, `gpt-oss-high`, `qwen3-32B`
- **Setups:** `A` (minimal prose), `B-query` (framework spec + query only), `B-resolved` (query + resolved-task block), `C` (agentic pipeline, coarse chunking)
- **Experiments:** E1 = E.text retrieval from M; E2 = E.text recovery from compact E.tab; E3 = induction of M and T
- **Judge (all files):** OpenAI `gpt-4.1`, temperature 0, with grounding+snap at `GROUNDING_SNAP_THRESHOLD=0.8`

## Generation settings (provenance)
| Model | Reasoning | max_output_tokens | Providers |
|---|---|---|---|
| gpt-4.1 | n/a (non-reasoning) | 8192 | OpenAI direct |
| gpt-oss-medium | effort=medium, exclude | 32768 (deterministic) | OpenRouter trusted allow-list |
| gpt-oss-high | effort=high, exclude | 32768 | OpenRouter trusted allow-list |
| qwen3-32B | thinking enabled, exclude | 16384 | OpenRouter (free routing) |

### Caveats
- **gpt-oss-medium robustness** generation predates the deterministic run: it used an 8192-token output cap (not 32768) and its Setup-C samples were free-routed (not trusted). No truncation occurred (max observed completion ≈7.2k), so results are unaffected; only the headroom/routing differs.
- **avg** in the consolidated matrix = unweighted mean of the P1 and P2 pattern means (equal weight per pattern).
- gpt-4.1 at temperature 0 is not bit-deterministic (~1% of re-judged samples differed); judge scores carry that small residual noise.
