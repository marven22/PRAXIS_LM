# Reproducibility Notes

This document maps the paper experiments to the source files included in this code-only release. The repository intentionally excludes datasets, checkpoints, adapters, logs, and result tables.

## Shared Conventions

Local generated datasets should be written under `data/`. Generated adapters, checkpoints, and logs should be written under ignored directories such as `loras/`, `models/`, `outputs/`, `logs/`, or `analysis_results/`.

Seeded methods should be run with the same seed list used by the paper protocol. Hyperparameters are encoded in the runner scripts and should be kept fixed across seeds for a given method.

## CLUTRR

Dataset and edit-data preparation:

- `scripts/build_clutrr_seal_edits.py`
- `scripts/convert_clutrr_to_ncrl.py`

Main baselines and PRAXIS:

- `scripts/run_clutrr_symbolic_baselines.py`
- `scripts/run_clutrr_adapters.py`
- `scripts/run_clutrr_edit_sft_base_eval.py`
- `scripts/run_clutrr_official_seal.py`
- `scripts/run_clutrr_seal_edits.py`
- `scripts/run_clutrr_roberta_baseline.py`
- `scripts/run_clutrr_ncrl_len4_seed_sweep.ps1`
- `scripts/evaluate_clutrr_ncrl_observed_paths.py`
- `scripts/run_clutrr_praxis_symbolic.py`

Model-family sensitivity:

- `scripts/run_clutrr_llm_rule_proposer.py`
- `scripts/run_clutrr_archive_conditioned_llm.py`
- `scripts/run_clutrr_archive_conditioned_models.sh`
- `scripts/run_clutrr_rq3_full1000_local.ps1`
- `scripts/summarize_clutrr_rq3_full1000.py`

Runtime and summarization:

- `scripts/measure_clutrr_runtime.py`
- `scripts/timing_fullscale_local10.ps1`
- `scripts/summarize_clutrr_full_table.py`
- `scripts/summarize_clutrr_seed_sweep.py`
- `scripts/summarize_clutrr_ncrl_seed_sweep.py`

## GraphLog

Dataset preparation:

- `scripts/build_graphlog_subset.py`
- `scripts/build_graphlog_binary_subset.py`

Main baselines and PRAXIS:

- `scripts/compute_graphlog_multiclass_baselines.py`
- `scripts/run_graphlog_multiclass_adapters.py`
- `scripts/run_graphlog_multiclass_relational_gnn.py`
- `scripts/run_graphlog_multiclass_local_gnn_seed_sweep.ps1`
- `scripts/runpod_graphlog_multiclass_praxis_seed_sweep.sh`

Model-family sensitivity:

- `scripts/run_graphlog_rq3_full1000_local.ps1`
- `scripts/summarize_graphlog_rq3_full1000.py`

Runtime and summarization:

- `scripts/timing_graphlog_rule56_r7plus10.ps1`
- `scripts/summarize_graphlog_multiclass_gnn_seed_sweep.py`
- `scripts/summarize_runpod_experiments.py`

## SQuAD-Style QA

Data preparation and SEAL support:

- `general-knowledge/src/data_generation/make_squad_data.py`
- `general-knowledge/src/EM/build_SFT_dataset.py`
- `general-knowledge/src/EM/train_SFT.py`
- `general-knowledge/src/query/query_server.py`
- `praxis/general-knowledge/src/query/query_server.py`

Main baselines and PRAXIS:

- `scripts/run_squad_passage_only_baseline.py`
- `scripts/timing_squad10_query_server.sh`
- `scripts/timing_squad10_sft.sh`
- `scripts/timing_squad10_passage_only_sft.sh`

Server helpers:

- `scripts/local_start_vllm.sh`
- `scripts/local_start_ttt.sh`
- `scripts/local_stop_servers.sh`
- `scripts/runpod_start_single_gpu_ttt.sh`
- `scripts/runpod_stop_ttt.sh`

## Model-Family Sensitivity

The model-family experiment varies the LM-facing PRAXIS component across Qwen2.5-3B, Phi-3.5-mini, and Llama-3.2-3B while keeping each dataset protocol fixed:

- `scripts/run_cross_model_paper_suite.ps1`
- `scripts/run_rq3_crossmodel_all.sh`
- `scripts/run_clutrr_llm_rule_proposer.py`
- `scripts/run_clutrr_rq3_full1000_local.ps1`
- `scripts/run_graphlog_rq3_full1000_local.ps1`

## Important Reproducibility Caveats

The code paths are meant to regenerate the reported experiment families, but exact runtimes depend on hardware, model-serving backend, quantization, and local disk/cache state. The paper reports wall-clock seconds per example from local calibration runs.

The CLUTRR and GraphLog experiments are evaluated at the instance level. SQuAD-style experiments operate over article contexts with multiple question-answer pairs, so scripts should preserve the article/question structure used by the evaluation protocol.
