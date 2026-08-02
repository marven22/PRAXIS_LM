# PRAXIS Few-Shot v1

This is the first PRAXIS instantiation for the ARC-style few-shot domain.

What v1 does:
- reuses the existing `few-shot/` ARC task and TTT substrate
- samples a PRAXIS strategy from a weighted archive
- generates a structured JSON self-edit/config for each task
- trains one temporary LoRA adapter per sampled self-edit
- saves configs in a format that can later be evaluated with the ARC evaluation scripts

Current scope:
- fixed global strategy archive
- no task-conditioned archive yet
- no archive growth yet

## Seed strategies
- `rule_abstraction`
- `augmentation_plan`
- `decomposition`
- `invariance_hypothesis`

## Generate PRAXIS self-edits and adapters

```bash
python praxis/few_shot/src/self_edit.py \
  --experiment_name praxis_arc_v1_train \
  --challenge_file few-shot/data/arc-agi_training_challenges_filtered_1B_training_set.json \
  --solution_file few-shot/data/arc-agi_training_solutions_filtered_1B_training_set.json \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --archive_path praxis/few_shot/archive/strategies.json \
  --n_tasks 12 \
  --n_self_edits_per_task 5
```

Adapters are written under:
- `loras/praxis-self-edit/<experiment_name>/<task_id>/<edit_idx>`

Config bookkeeping is written to:
- `loras/praxis-self-edit/<experiment_name>/final_configs_and_indices.json`

## Update the archive after evaluation

After running an evaluation script that produces a `final_results.json`, update the archive with:

```bash
python praxis/few_shot/src/update_archive_from_results.py \
  --archive_path praxis/few_shot/archive/strategies.json \
  --configs_path loras/praxis-self-edit/praxis_arc_v1_train/final_configs_and_indices.json \
  --results_path <path_to_final_results.json>
```

## Notes
- v1 is deliberately simple so we can first verify the PRAXIS few-shot path works end-to-end.
- The next step after validation is to add a task-conditioned archive on top of this scaffold.
