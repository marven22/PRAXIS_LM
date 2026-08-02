$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$NcrlCode = Join-Path $Repo "external\NCRL\code"
$ResultsDir = Join-Path $Repo "analysis_results\full_scale\ncrl_clutrr_len4_seed_sweep"
$LogsDir = Join-Path $Repo "logs"

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {
    $model = "clutrr_praxis_len4_seed$seed"
    $trainOut = Join-Path $LogsDir "clutrr_ncrl_len4_seed$seed.train.out.log"
    $trainErr = Join-Path $LogsDir "clutrr_ncrl_len4_seed$seed.train.err.log"
    $evalOut = Join-Path $LogsDir "clutrr_ncrl_len4_seed$seed.eval.out.log"
    $evalErr = Join-Path $LogsDir "clutrr_ncrl_len4_seed$seed.eval.err.log"
    $result = Join-Path $ResultsDir "clutrr_ncrl_len4_seed$seed.json"

    Push-Location $NcrlCode
    try {
        python main.py `
            --train `
            --data clutrr_praxis `
            --model $model `
            --max_path_len 4 `
            --anchor 12064 `
            --epochs 200 `
            --emb_size 256 `
            --train_batch_size 2048 `
            --gpu 0 `
            --seed $seed `
            > $trainOut 2> $trainErr
    }
    finally {
        Pop-Location
    }

    Push-Location $Repo
    try {
        python scripts\evaluate_clutrr_ncrl_observed_paths.py `
            --model $model `
            --output $result `
            > $evalOut 2> $evalErr
    }
    finally {
        Pop-Location
    }
}

Push-Location $Repo
try {
    python scripts\summarize_clutrr_ncrl_seed_sweep.py `
        --results_dir analysis_results\full_scale\ncrl_clutrr_len4_seed_sweep `
        --output analysis_results\full_scale\clutrr_ncrl_len4_seed_sweep_summary.json `
        > (Join-Path $LogsDir "clutrr_ncrl_len4_seed_sweep_summary.log") 2>&1
}
finally {
    Pop-Location
}
