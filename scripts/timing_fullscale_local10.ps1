$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

$ModelName = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "Qwen/Qwen2.5-3B-Instruct" }

New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path analysis_results\timing_calibration | Out-Null

function Run-Timed {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    $start = Get-Date
    Write-Host "=== $Name started $start ==="
    & $Command
    $end = Get-Date
    $elapsed = ($end - $start).TotalSeconds
    $summary = @{
        name = $Name
        started = $start.ToString("o")
        finished = $end.ToString("o")
        elapsed_seconds = $elapsed
        time_per_example_seconds = $elapsed / 10.0
    }
    $summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 "analysis_results\timing_calibration\$Name.json"
    Write-Host "=== $Name finished in $elapsed sec ==="
}

python scripts\build_graphlog_subset.py `
    --out data\graphlog\graphlog_rule56_timing10.json `
    --n_tasks 10 `
    --support_examples 8 `
    --world_split test `
    --query_split test.jsonl `
    --world_names rule_56 `
    --seed 42

Run-Timed "fullscale_clutrr_symbolic10" {
    python scripts\run_clutrr_praxis_symbolic.py `
        --n 10 `
        --output analysis_results\timing_calibration\clutrr_praxis_symbolic_10.json `
        --subset_output data\clutrr\clutrr_test_timing10.json *>&1 |
        Tee-Object -FilePath logs\fullscale_clutrr_symbolic10.log
}

Run-Timed "fullscale_graphlog_multiclass_praxis10" {
    python scripts\run_graphlog_adapters.py `
        --mode praxis `
        --phase train_eval `
        --experiment_name timing_graphlog_rule56_multiclass10_praxis `
        --dataset data\graphlog\graphlog_rule56_timing10.json `
        --n_tasks 10 `
        --epochs 5 `
        --lr 5e-4 `
        --batch_size 1 `
        --gradient_accumulation_steps 1 `
        --model_name $ModelName `
        --results_root runpod_results\timing_calibration\graphlog_multiclass `
        --output_root loras\timing_calibration\graphlog_multiclass *>&1 |
        Tee-Object -FilePath logs\fullscale_graphlog_multiclass_praxis10.log
}

Run-Timed "fullscale_graphlog_multiclass_seal10" {
    python scripts\run_graphlog_adapters.py `
        --mode seal `
        --phase train_eval `
        --experiment_name timing_graphlog_rule56_multiclass10_seal `
        --dataset data\graphlog\graphlog_rule56_timing10.json `
        --n_tasks 10 `
        --epochs 5 `
        --lr 5e-4 `
        --batch_size 1 `
        --gradient_accumulation_steps 1 `
        --model_name $ModelName `
        --results_root runpod_results\timing_calibration\graphlog_multiclass `
        --output_root loras\timing_calibration\graphlog_multiclass *>&1 |
        Tee-Object -FilePath logs\fullscale_graphlog_multiclass_seal10.log
}
