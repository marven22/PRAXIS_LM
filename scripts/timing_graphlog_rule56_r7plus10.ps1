$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$harness = "C:\Users\vmargapu\Documents\Research\praxis-experiment-harness"
$dataset = Join-Path $repo "data\graphlog\graphlog_binary_rule56_r7plus_100.json"

Set-Location $repo
New-Item -ItemType Directory -Force -Path "analysis_results\timing_calibration" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

function Run-Timed($Name, $WorkDir, $Body) {
    Set-Location $WorkDir
    $start = Get-Date
    Write-Host "=== START $Name $($start.ToString('s')) ==="
    & $Body
    $code = $LASTEXITCODE
    $end = Get-Date
    $elapsed = ($end - $start).TotalSeconds
    [pscustomobject]@{
        name = $Name
        n = 10
        hardware = "local"
        start = $start.ToString("o")
        end = $end.ToString("o")
        total_seconds = [math]::Round($elapsed, 3)
        seconds_per_example = [math]::Round($elapsed / 10.0, 3)
        exit_code = $code
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $repo "analysis_results\timing_calibration\$Name.runtime.json")
    Write-Host "=== END $Name elapsed=$elapsed exit=$code ==="
}

Run-Timed "graphlog_rule56_r7plus10_praxis" $repo {
    python scripts\run_graphlog_binary_adapters.py `
        --mode praxis `
        --phase train_eval `
        --experiment_name timing_graphlog_rule56_r7plus10_praxis `
        --dataset data\graphlog\graphlog_binary_rule56_r7plus_100.json `
        --n_tasks 10 `
        --epochs 5 `
        --lr 5e-4 `
        --batch_size 1 `
        --gradient_accumulation_steps 1 `
        --score_method logprob *>&1 | Tee-Object -FilePath (Join-Path $repo "logs\timing_graphlog_rule56_r7plus10_praxis.log")
}

Run-Timed "graphlog_rule56_r7plus10_seal" $repo {
    python scripts\run_graphlog_binary_adapters.py `
        --mode seal `
        --phase train_eval `
        --experiment_name timing_graphlog_rule56_r7plus10_seal `
        --dataset data\graphlog\graphlog_binary_rule56_r7plus_100.json `
        --n_tasks 10 `
        --epochs 5 `
        --lr 5e-4 `
        --batch_size 1 `
        --gradient_accumulation_steps 1 `
        --score_method logprob *>&1 | Tee-Object -FilePath (Join-Path $repo "logs\timing_graphlog_rule56_r7plus10_seal.log")
}

if (Test-Path (Join-Path $harness "scripts\run_omni_graphlog_binary.py")) {
    Run-Timed "graphlog_rule56_r7plus10_omni" $harness {
        python scripts\run_omni_graphlog_binary.py `
            --phase train_eval `
            --experiment_name timing_graphlog_rule56_r7plus10_omni `
            --dataset $dataset `
            --n_tasks 10 `
            --epochs 5 `
            --lr 5e-4 `
            --batch_size 1 `
            --gradient_accumulation_steps 1 `
            --score_method logprob *>&1 | Tee-Object -FilePath (Join-Path $repo "logs\timing_graphlog_rule56_r7plus10_omni.log")
    }
}
