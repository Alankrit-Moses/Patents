param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("A", "BQueryOnly", "BResolved")]
    [string]$Job,

    [string]$Config = "experiments/config.openrouter-gpt-oss-120b.json",
    [double]$Temperature = 0.2,
    [int]$Samples = 3,
    [int]$MaxAttempts = 3,
    [string]$KeysPath = "experiments/keys.local.env"
)

$ErrorActionPreference = "Stop"

function Import-EnvFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) {
            throw "Invalid env line in ${Path}: ${line}"
        }
        [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
    }
}

Import-EnvFile -Path $KeysPath

if (-not $env:OPENROUTER_API_KEY) {
    throw "OPENROUTER_API_KEY is required. Set it in the environment or in $KeysPath."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$robustDir = Join-Path (Resolve-Path -LiteralPath "experiments/artifacts/openrouter").Path "robustness"
$logDir = Join-Path $robustDir "logs"
New-Item -ItemType Directory -Force -Path $robustDir, $logDir | Out-Null

switch ($Job) {
    "A" {
        $setup = "A"
        $bVariant = "query-only"
        $label = "A"
    }
    "BQueryOnly" {
        $setup = "B"
        $bVariant = "query-only"
        $label = "B-query-only"
    }
    "BResolved" {
        $setup = "B"
        $bVariant = "resolved"
        $label = "B-resolved"
    }
}

$output = Join-Path $robustDir "results-v2-$label-gpt-oss-120b-temp$Temperature-k$Samples-$timestamp.jsonl"
$stdout = Join-Path $logDir "robustness-v2-$label-$timestamp.out.log"
$stderr = Join-Path $logDir "robustness-v2-$label-$timestamp.err.log"

Write-Host "Running v2 robustness job: $Job"
Write-Host "Output: $output"
Write-Host "Stdout log: $stdout"
Write-Host "Stderr log: $stderr"

python -m experiments.robustness `
    --config $Config `
    run `
    --prompt-version v2 `
    --setup $setup `
    --b-variant $bVariant `
    --temperature $Temperature `
    --samples $Samples `
    --max-attempts $MaxAttempts `
    --output $output `
    1> $stdout `
    2> $stderr

Write-Host "Completed v2 robustness job: $Job"
Write-Host "Results: $output"
