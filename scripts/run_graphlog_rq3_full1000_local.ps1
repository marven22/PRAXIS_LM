param(
  [int]$N = 1000,
  [string]$Python = "python",
  [int]$Seed = 42
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "analysis_results\rq3_crossmodel_full1000\graphlog_multiclass" | Out-Null
New-Item -ItemType Directory -Force -Path "loras\graphlog-rq3-crossmodel-stream-scratch" | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$Runs = @(
  @{
    Name = "graphlog_multiclass_praxis_qwen25_3b_full1000_seed$Seed"
    Model = "Qwen/Qwen2.5-3B"
  },
  @{
    Name = "graphlog_multiclass_praxis_phi35_mini_full1000_seed$Seed"
    Model = "microsoft/Phi-3.5-mini-instruct"
  },
  @{
    Name = "graphlog_multiclass_praxis_llama32_3b_full1000_seed$Seed"
    Model = "meta-llama/Llama-3.2-3B-Instruct"
  }
)

foreach ($Run in $Runs) {
  $LogPath = "logs\$($Run.Name).log"
  "$(Get-Date -Format o) === START $($Run.Name) model=$($Run.Model) n=$N seed=$Seed ===" | Tee-Object -FilePath $LogPath
  & $Python -u scripts/run_graphlog_multiclass_adapters.py `
    --mode praxis `
    --phase train_eval `
    --stream_train_eval `
    --experiment_name $Run.Name `
    --dataset data/graphlog/graphlog_rule56_full1000.json `
    --model_name $Run.Model `
    --n_tasks $N `
    --epochs 5 `
    --lr 5e-4 `
    --batch_size 1 `
    --gradient_accumulation_steps 1 `
    --results_root analysis_results/rq3_crossmodel_full1000/graphlog_multiclass `
    --output_root loras/graphlog-rq3-crossmodel-stream-scratch `
    --seed $Seed `
    2>&1 | Tee-Object -FilePath $LogPath -Append
  $Status = $LASTEXITCODE
  "$(Get-Date -Format o) === END $($Run.Name) exit=$Status ===" | Tee-Object -FilePath $LogPath -Append
}

& $Python scripts/summarize_graphlog_rq3_full1000.py `
  --results_root "analysis_results/rq3_crossmodel_full1000/graphlog_multiclass" `
  --out "analysis_results/rq3_crossmodel_full1000/graphlog_rq3_model_family_table.json" `
  2>&1 | Tee-Object -FilePath "logs\graphlog_rq3_full1000_summary.log"
