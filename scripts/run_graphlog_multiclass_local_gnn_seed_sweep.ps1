$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$ResultsDir = Join-Path $Repo "analysis_results\full_scale\graphlog_multiclass_gnn_seed_sweep"
$LogsDir = Join-Path $Repo "logs"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

Push-Location $Repo
try {
    python scripts\compute_graphlog_multiclass_baselines.py `
        --dataset data\graphlog\graphlog_rule56_full1000.json `
        --out analysis_results\full_scale\graphlog_rule56_full1000_multiclass_baselines.json `
        > (Join-Path $LogsDir "graphlog_multiclass_static_baselines.log") 2>&1
}
finally {
    Pop-Location
}

$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {
    $out = Join-Path $ResultsDir "graphlog_multiclass_relgnn_seed$seed.json"
    $log = Join-Path $LogsDir "graphlog_multiclass_relgnn_seed$seed.log"
    Push-Location $Repo
    try {
        python scripts\run_graphlog_multiclass_relational_gnn.py `
            --dataset data\graphlog\graphlog_rule56_full1000.json `
            --output "analysis_results\full_scale\graphlog_multiclass_gnn_seed_sweep\graphlog_multiclass_relgnn_seed$seed.json" `
            --epochs 25 `
            --hidden_dim 96 `
            --layers 3 `
            --lr 2e-3 `
            --weight_decay 1e-4 `
            --max_train 5000 `
            --seed $seed `
            > $log 2>&1
    }
    finally {
        Pop-Location
    }
}

Push-Location $Repo
try {
    python scripts\summarize_graphlog_multiclass_gnn_seed_sweep.py `
        --results_dir analysis_results\full_scale\graphlog_multiclass_gnn_seed_sweep `
        --output analysis_results\full_scale\graphlog_multiclass_gnn_seed_sweep_summary.json `
        > (Join-Path $LogsDir "graphlog_multiclass_gnn_seed_sweep_summary.log") 2>&1
}
finally {
    Pop-Location
}
