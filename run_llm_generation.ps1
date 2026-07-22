$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class SleepControl
{
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

[uint32]$ES_CONTINUOUS = 2147483648
[uint32]$ES_FLAGS      = 2147483649  # CONTINUOUS | SYSTEM_REQUIRED

[SleepControl]::SetThreadExecutionState($ES_FLAGS) | Out-Null

function Run-Generation {
    param(
        [int]$Budget,
        [string]$Output
    )

    Write-Host ""
    Write-Host "===== Starting ${Budget}-token generation ====="

    python scripts/07_generate_answers.py `
        --inputs outputs/generation/rag_inputs_full.jsonl `
        --out $Output `
        --token-budget $Budget `
        --checkpoint-every 10

    if ($LASTEXITCODE -ne 0) {
        throw "${Budget}-token generation failed with exit code $LASTEXITCODE"
    }

    Write-Host "===== Finished ${Budget}-token generation ====="
}

try {
    Run-Generation `
        -Budget 256 `
        -Output "outputs/generation/answers_256.jsonl"

    Run-Generation `
        -Budget 512 `
        -Output "outputs/generation/answers_512.jsonl"

    Run-Generation `
        -Budget 1024 `
        -Output "outputs/generation/answers_1024.jsonl"

    Write-Host ""
    Write-Host "All generation runs completed successfully."
}
catch {
    Write-Error $_
    throw
}
finally {
    [SleepControl]::SetThreadExecutionState(
        $ES_CONTINUOUS
    ) | Out-Null
}