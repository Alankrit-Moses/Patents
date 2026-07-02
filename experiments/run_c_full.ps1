param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("gpt-oss-medium", "gpt-oss-high", "qwen3-32B", "gpt-4.1")]
    [string]$Model,

    # Also blind-judge the results with gpt-4.1 (grounding+snap 0.8) after generation.
    [switch]$Judge,

    [int]$MaxAttempts = 3,
    [int]$Concurrency = 1,
    [string]$KeysPath = "experiments/keys.local.env"
)

$ErrorActionPreference = "Stop"

function Import-EnvFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) { throw "Invalid env line in ${Path}: ${line}" }
        [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
    }
}

Import-EnvFile -Path $KeysPath

# Model -> generator config. These are the SAME generator settings used for the
# existing A/B/C (map-reduce) results, so full-context C is directly comparable.
# NOTE: gpt-4.1 uses config.openai-gpt41.json (ctx 131072, out 8192), NOT the
# stale config.openai-gpt-4.1.json (ctx 28000) which would error on full reports.
switch ($Model) {
    "gpt-oss-medium" { $config = "experiments/config.openrouter-gpt-oss-120b-medium-reasoning.json"; $needs = "OPENROUTER_API_KEY" }
    "gpt-oss-high"   { $config = "experiments/config.openrouter-gpt-oss-120b-high-reasoning.json";   $needs = "OPENROUTER_API_KEY" }
    "qwen3-32B"      { $config = "experiments/config.openrouter-qwen3-32b.json";                     $needs = "OPENROUTER_API_KEY" }
    "gpt-4.1"        { $config = "experiments/config.openai-gpt41.json";                             $needs = "OPENAI_API_KEY" }
}

if (-not [Environment]::GetEnvironmentVariable($needs, "Process")) {
    throw "$needs is required for $Model. Set it in the environment or in $KeysPath."
}

$detDir = "experiments/artifacts/results/$Model/deterministic"
New-Item -ItemType Directory -Force -Path $detDir | Out-Null
$results = Join-Path $detDir "C-full.results.jsonl"
$eval    = Join-Path $detDir "C-full.eval.jsonl"

Write-Host "== Full-context Setup C: $Model =="
Write-Host "Config:  $config"
Write-Host "Results: $results (resume-safe; prompt_version=v2-agentic-full)"

# --resume keys on (task_id, prompt_version); v2-agentic-full is distinct from the
# map-reduce C run (v2-agentic), so this is safe even if paths are shared.
python -m experiments.runner_v2 `
    --config $config `
    run `
    --setup C `
    --c-variant full-context `
    --experiment all `
    --max-attempts $MaxAttempts `
    --concurrency $Concurrency `
    --resume `
    --output $results

if ($Judge) {
    if (-not $env:OPENAI_API_KEY) { throw "OPENAI_API_KEY is required for the gpt-4.1 judge." }
    Write-Host "== Judging $Model (gpt-4.1, grounding+snap 0.8) =="
    $env:GROUNDING_SNAP_THRESHOLD = "0.8"
    python -m experiments.cli `
        --config experiments/config.judge-openai-gpt41-rate-limit.json `
        judge --results $results --output $eval
    python -m experiments.cli `
        --config experiments/config.judge-openai-gpt41-rate-limit.json `
        summarize --evaluations $eval
}

Write-Host "Done: $Model"
