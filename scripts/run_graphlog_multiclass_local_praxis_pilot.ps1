$ErrorActionPreference = "Continue"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogsDir = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

$Log = Join-Path $LogsDir "graphlog_multiclass_praxis_local_pilot100_seed42.log"

Push-Location $Repo
try {
    python scripts\run_graphlog_multiclass_adapters.py `
        --mode praxis `
        --phase train_eval `
        --experiment_name graphlog_multiclass_praxis_local_pilot100_seed42 `
        --dataset data\graphlog\graphlog_rule56_full1000.json `
        --n_tasks 100 `
        --epochs 5 `
        --lr 5e-4 `
        --batch_size 1 `
        --gradient_accumulation_steps 1 `
        --results_root analysis_results\full_scale\graphlog_multiclass_local_pilot `
        --output_root loras\graphlog-multiclass-local-pilot `
        --seed 42 `
        > $Log 2>&1
}
finally {
    Pop-Location
}
