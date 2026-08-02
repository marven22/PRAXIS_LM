# PRAXIS Experiment Harness

This repository contains code for the PRAXIS experiments across CLUTRR, GraphLog, and SQuAD-style question answering. It is a code-only release intended to help researchers inspect, rerun, and extend the experimental harness.

The repository intentionally does not include raw datasets, generated dataset slices, model weights, LoRA adapters, logs, paper drafts, table CSVs, or result dumps. The datasets used by the paper are publicly available or generated through the included preparation scripts.

## Repository Layout

- `scripts/` contains dataset builders, experiment runners, timing scripts, RunPod helpers, and summarizers.
- `praxis/` contains PRAXIS archive utilities and domain-specific PRAXIS support code.
- `general-knowledge/` contains SQuAD-style adaptation and SEAL/SFT support code.
- `few-shot/` contains SEAL few-shot support code used for SEAL-style comparison components.
- `external/NCRL/` contains the NCRL code path used for the CLUTRR neural rule-learning baseline.
- `docs/` contains reproducibility notes and table-to-script mappings.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Some experiments require GPU access, Hugging Face model access, and API keys for model serving or evaluation. Store credentials in a local `.env` file or shell environment. Do not commit credentials.

## Datasets

Datasets are not redistributed in this repository. Use the preparation scripts in `scripts/` and `general-knowledge/src/data_generation/` to build local working copies from public sources:

- CLUTRR for compositional kinship reasoning.
- GraphLog for relational graph reasoning.
- SQuAD-style QA resources for passage-based question answering.

Local generated data should live under `data/`, which is ignored by Git.

## Reproducing Experiments

The main experiment families are documented in `docs/reproducibility.md`. In brief:

- CLUTRR uses symbolic baselines, Edit-SFT, SEAL-style adaptation, DistilRoBERTa, NCRL, and PRAXIS symbolic recomposition.
- GraphLog uses curriculum adaptation, SEAL-style adaptation, a relational GNN, and PRAXIS adaptation.
- SQuAD-QA uses passage-only adaptation, PRAXIS-SFT, SEAL-SFT, SEAL-style adaptation, and PRAXIS answer correction.
- Model-family sensitivity varies the PRAXIS LM-facing component across Qwen2.5-3B, Phi-3.5-mini, and Llama-3.2-3B.

This release provides the runnable code paths, not the generated results. Researchers should regenerate result artifacts locally.

## Notes on SEAL-Style Experiments

For SQuAD-style QA, the code builds directly on the SEAL general-knowledge implementation path where applicable. For CLUTRR and GraphLog, which are not native SEAL tasks, the release provides SEAL-style adaptations that preserve the relevant self-edit and task-local update logic while adapting the input/output format to the benchmark.

## License

See `LICENSE`. External code included under `external/` may carry its own upstream license terms or citation requirements.
