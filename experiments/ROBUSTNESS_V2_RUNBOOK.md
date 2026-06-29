# V2 Robustness Runbook

This is the run plan for the v2 robustness jobs discussed for the WIPO pattern experiments.

## Jobs to run

Run these one by one, in this order:

1. Setup A, v2 prompt
2. Setup B, v2 query-only prompt
3. Setup B, v2 resolved-task prompt

All three jobs use:

- Config: `experiments/config.openrouter-gpt-oss-120b.json`
- Model: `openai/gpt-oss-120b` through OpenRouter
- Temperature: `0.2`
- Samples per condition: `3`
- Max attempts: `3`
- Output directory: `experiments/artifacts/openrouter/robustness`

Each full setup is expected to write 72 records: 24 tasks times 3 samples.

## Keys

Create a local key file from the template:

```powershell
Copy-Item experiments\keys.env.example experiments\keys.local.env
notepad experiments\keys.local.env
```

Fill in:

```text
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
GITHUB_TOKEN=...
```

`experiments/keys.local.env` is gitignored and should not be committed.

## Commands

From the repository root, run one job at a time:

```powershell
.\experiments\run_v2_robustness.ps1 -Job A
```

After A finishes, run:

```powershell
.\experiments\run_v2_robustness.ps1 -Job BQueryOnly
```

After B query-only finishes, run:

```powershell
.\experiments\run_v2_robustness.ps1 -Job BResolved
```

The script prints the output JSONL and log paths. It does not run jobs in the background; keep the terminal open until each command exits.

## Resume policy

If a machine sleeps, dies, or the process is interrupted, delete the partial output JSONL for that job and rerun the same command from the start.

Do not append a restarted run to a partial file unless you intentionally want mixed-run samples.
