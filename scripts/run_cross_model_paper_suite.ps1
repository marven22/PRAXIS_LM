$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path analysis_results | Out-Null
New-Item -ItemType Directory -Force -Path runpod_results/graphlog_binary_crossmodel | Out-Null

$log = "logs/cross_model_paper_suite.log"
"[$(Get-Date -Format o)] Starting cross-model paper suite" | Tee-Object -FilePath $log -Append

function Run-Step {
    param(
        [string]$Name,
        [string]$OutputPath,
        [scriptblock]$Command
    )

    if (Test-Path $OutputPath) {
        "[$(Get-Date -Format o)] SKIP $Name because $OutputPath exists" | Tee-Object -FilePath $log -Append
        return
    }

    "[$(Get-Date -Format o)] RUN $Name" | Tee-Object -FilePath $log -Append
    & $Command *>&1 | Tee-Object -FilePath $log -Append
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $log -Append
}

Run-Step `
    -Name "CLUTRR Phi direct prompt 200" `
    -OutputPath "analysis_results/clutrr_phi35_mini_direct_200.json" `
    -Command {
        python scripts/run_clutrr_llm_baseline.py `
            --dataset data/clutrr/clutrr_test_200.json `
            --model microsoft/Phi-3.5-mini-instruct `
            --output analysis_results/clutrr_phi35_mini_direct_200.json `
            --mode direct `
            --n 200 `
            --max_new_tokens 32
    }

Run-Step `
    -Name "CLUTRR Llama direct prompt 200" `
    -OutputPath "analysis_results/clutrr_llama32_3b_direct_200.json" `
    -Command {
        python scripts/run_clutrr_llm_baseline.py `
            --dataset data/clutrr/clutrr_test_200.json `
            --model meta-llama/Llama-3.2-3B-Instruct `
            --output analysis_results/clutrr_llama32_3b_direct_200.json `
            --mode direct `
            --n 200 `
            --max_new_tokens 32
    }

Run-Step `
    -Name "GraphLog Phi SEAL 100" `
    -OutputPath "runpod_results/graphlog_binary_crossmodel/graphlog_rule56_r7plus100_seal_phi35/final_results.json" `
    -Command {
        python scripts/run_graphlog_binary_adapters.py `
            --mode seal `
            --phase train_eval `
            --experiment_name graphlog_rule56_r7plus100_seal_phi35 `
            --dataset data/graphlog/graphlog_binary_rule56_r7plus_100.json `
            --model_name microsoft/Phi-3.5-mini-instruct `
            --n_tasks 100 `
            --epochs 5 `
            --lr 5e-4 `
            --batch_size 1 `
            --gradient_accumulation_steps 1 `
            --score_method logprob `
            --results_root runpod_results/graphlog_binary_crossmodel
    }

Run-Step `
    -Name "GraphLog Llama SEAL 100" `
    -OutputPath "runpod_results/graphlog_binary_crossmodel/graphlog_rule56_r7plus100_seal_llama32_3b/final_results.json" `
    -Command {
        python scripts/run_graphlog_binary_adapters.py `
            --mode seal `
            --phase train_eval `
            --experiment_name graphlog_rule56_r7plus100_seal_llama32_3b `
            --dataset data/graphlog/graphlog_binary_rule56_r7plus_100.json `
            --model_name meta-llama/Llama-3.2-3B-Instruct `
            --n_tasks 100 `
            --epochs 5 `
            --lr 5e-4 `
            --batch_size 1 `
            --gradient_accumulation_steps 1 `
            --score_method logprob `
            --results_root runpod_results/graphlog_binary_crossmodel
    }

"[$(Get-Date -Format o)] Finished cross-model paper suite" | Tee-Object -FilePath $log -Append
