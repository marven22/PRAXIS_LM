param(
  [int]$N = 1000,
  [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "analysis_results\rq3_crossmodel_full1000" | Out-Null

$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$PraxisResult = "analysis_results/full_scale/clutrr_praxis_symbolic_full1048.json"
$ReferenceResult = "analysis_results/full_scale/clutrr_symbolic_baselines_full1048.json"

$Runs = @(
  @{
    Name = "clutrr_rule_proposer_qwen25_3b_full1000"
    Model = "Qwen/Qwen2.5-3B"
    Output = "analysis_results/rq3_crossmodel_full1000/clutrr_rule_proposer_qwen25_3b_full1000.json"
  },
  @{
    Name = "clutrr_rule_proposer_phi35_mini_full1000"
    Model = "microsoft/Phi-3.5-mini-instruct"
    Output = "analysis_results/rq3_crossmodel_full1000/clutrr_rule_proposer_phi35_mini_full1000.json"
  },
  @{
    Name = "clutrr_rule_proposer_llama32_3b_full1000"
    Model = "meta-llama/Llama-3.2-3B-Instruct"
    Output = "analysis_results/rq3_crossmodel_full1000/clutrr_rule_proposer_llama32_3b_full1000.json"
  }
)

foreach ($Run in $Runs) {
  $LogPath = "logs\$($Run.Name).log"
  "$(Get-Date -Format o) === START $($Run.Name) model=$($Run.Model) n=$N ===" | Tee-Object -FilePath $LogPath
  & $Python -u scripts/run_clutrr_llm_rule_proposer.py `
    --model $Run.Model `
    --n $N `
    --praxis_result $PraxisResult `
    --reference_result $ReferenceResult `
    --output $Run.Output `
    2>&1 | Tee-Object -FilePath $LogPath -Append
  $Status = $LASTEXITCODE
  "$(Get-Date -Format o) === END $($Run.Name) exit=$Status output=$($Run.Output) ===" | Tee-Object -FilePath $LogPath -Append
}

& $Python scripts/summarize_clutrr_rq3_full1000.py `
  --results_root "analysis_results/rq3_crossmodel_full1000" `
  --out "analysis_results/rq3_crossmodel_full1000/clutrr_rq3_model_family_table.json" `
  2>&1 | Tee-Object -FilePath "logs\clutrr_rq3_full1000_summary.log"
