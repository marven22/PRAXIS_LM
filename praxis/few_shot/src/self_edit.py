from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from peft import LoraConfig
from transformers import AutoTokenizer

# Reuse the ARC / TTT substrate from the existing few-shot SEAL implementation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FEW_SHOT_DIR = REPO_ROOT / "few-shot"
if str(FEW_SHOT_DIR) not in sys.path:
    sys.path.insert(0, str(FEW_SHOT_DIR))

from arclib.arc import Task  # type: ignore  # noqa: E402
from arclib.augmenters import (  # type: ignore  # noqa: E402
    Augmenter,
    Chain,
    Concat,
    Flip,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from arclib.messagers import GPTTextMessageRepresenterV2, MessageRepresenter  # type: ignore  # noqa: E402
from arclib.representers import PythonListGridRepresenter, TextExampleRepresenter, TextTaskRepresenter  # type: ignore  # noqa: E402
from arclib.update_model import TTT  # type: ignore  # noqa: E402
from praxis.few_shot.src.archive_utils import load_archive, sample_strategy
from praxis.few_shot.src.strategy_prompts import SYSTEM_MESSAGE, build_strategy_prompt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def read_tasks_from_single_file(challenge_file: str, solution_file: str | None = None, test: bool = False) -> List[Task]:
    with open(challenge_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if solution_file is not None:
        test = False
        with open(solution_file, "r", encoding="utf-8") as handle:
            solutions = json.load(handle)
            for key, value in solutions.items():
                for idx, solution in enumerate(value):
                    data[key]["test"][idx]["output"] = solution

    all_tasks: List[Task] = []
    for task_name, subtasks in data.items():
        parsed = Task.read_tasks_from_dict(subtasks, test=test)
        for i, task in enumerate(parsed):
            task.name = f"{task_name}-{i}"
            all_tasks.append(task)
    return all_tasks


def select_base_tasks(tasks: List[Task], n_base_tasks: int) -> List[Task]:
    """Select up to n_base_tasks using only the canonical -0 subtask when present."""
    selected: List[Task] = []
    seen_base_names: set[str] = set()
    for task in tasks:
        base_name = task.name[:-2] if task.name.endswith(("-0", "-1")) else task.name
        if base_name in seen_base_names:
            continue
        if task.name.endswith("-1"):
            continue
        selected.append(task)
        seen_base_names.add(base_name)
        if len(selected) >= n_base_tasks:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRAXIS few-shot v1 self-edit generation for ARC")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--challenge_file", required=True)
    parser.add_argument("--solution_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--archive_path", default="praxis/few_shot/archive/strategies.json")
    parser.add_argument("--n_tasks", type=int, required=True)
    parser.add_argument("--n_self_edits_per_task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_generation_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--default_lr_scheduler", default="cosine")
    parser.add_argument("--max_train_examples", type=int, default=250)
    parser.add_argument("--permute_n", type=int, default=1)
    parser.add_argument("--leave_n", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--max_steps_cap", type=int, default=375)
    parser.add_argument("--output_root", default="loras/praxis-self-edit")
    return parser.parse_args()


def get_augmenters(
    *,
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
) -> List[Augmenter]:
    basic = [
        Rotate(90),
        Rotate(270),
        Rotate(180),
        Flip(0),
        Flip(1),
        Reflect(0, reverse=True),
        Reflect(1, reverse=True),
        Reflect(0, reverse=False),
        Reflect(1, reverse=False),
        RandomTranslateXY(),
        Transpose(),
    ] if include_basic else []
    size = [IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2)] if include_size else []
    chain = [
        Chain([Rotate(90), IncreaseResolution(2)]),
        Chain([Rotate(270), IncreaseResolution(2)]),
        Chain([Rotate(180), IncreaseResolution(2)]),
        Chain([Flip(0), IncreaseResolution(2)]),
        Chain([Flip(1), IncreaseResolution(2)]),
        Chain([Transpose(), IncreaseResolution(2)]),
    ] if include_chain else []
    repeat = [Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] if include_repeat else []
    concat: List[Augmenter] = [
        Concat((Rotate(180), Rotate(180)), axis=0),
        Concat((Rotate(180), Rotate(180)), axis=1),
    ] if False else []
    return basic + size + chain + repeat + concat


def _tokenize_and_process(text: str, tokenizer):
    outputs = tokenizer(text, truncation=True)
    data = {"input": None, "output": None}
    data["total_tokens"] = len(outputs["input_ids"])
    data["full_text"] = text
    return data


def format_and_filter(formatter: MessageRepresenter, tokenizer, task: Task):
    encoded = formatter.encode(task)
    task_text = tokenizer.apply_chat_template(encoded[0] + [encoded[1]], tokenize=False, add_generation_prompt=True)
    return _tokenize_and_process(task_text, tokenizer)


def get_test_time_train_data(original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0) -> List[Task]:
    rng = np.random.RandomState(seed)
    train_examples = original_task.train_examples.copy()
    initial_tasks: List[Task] = []
    total = len(train_examples)
    for heldout_idx in range(total):
        examples = train_examples.copy()
        indices = set(range(total)) - {heldout_idx}
        combs = list(__import__("itertools").combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]
        for comb in combs:
            initial_tasks.append(Task(name="", train_examples=[examples[j] for j in comb], test_example=examples[heldout_idx]))

    augmented_tasks: List[Task] = []
    for augmenter in augmenters:
        for task in initial_tasks:
            maybe = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            if maybe.max_height() <= 30 and maybe.max_width() <= 30:
                augmented_tasks.append(maybe)

    augmented_tasks = list(set(augmented_tasks + initial_tasks))
    colored: List[Task] = []
    for _ in range(permute_n):
        for task in augmented_tasks:
            maybe = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng) if augmenters else task
            maybe = PermuteExamples().apply_to_task(maybe, rng=rng, to_input=True, to_output=True)
            colored.append(maybe)
    return list(set(colored + augmented_tasks))


def process_task(task: Task, augmenters: List[Augmenter], formatter: MessageRepresenter, tokenizer, leave_n: List[int], permute_n: int, max_examples: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    train: List[Dict[str, Any]] = []
    for n in leave_n:
        train_tasks = get_test_time_train_data(task, augmenters, n=n, permute_n=permute_n, seed=seed)
        for aug_task in train_tasks:
            formatted = format_and_filter(formatter, tokenizer, aug_task)
            if formatted["total_tokens"] < 8192:
                train.append(formatted)
    if len(train) > max_examples:
        rng.shuffle(train)
        train = train[:max_examples]
    return train


def get_prompt(task: Task, strategy_name: str) -> str:
    train_examples = task.serialize()["train"]
    formatted_examples = []
    for example in train_examples:
        in_grid = "\n".join(" ".join(map(str, row)) for row in example["input"])
        out_grid = "\n".join(" ".join(map(str, row)) for row in example["output"])
        formatted_examples.append(f"Input:\n{in_grid}\n\nOutput:\n{out_grid}\n")
    user_message = "\n".join(formatted_examples)
    user_message += "------\n\n"
    user_message += f"Selected PRAXIS strategy: {strategy_name}\n"
    user_message += build_strategy_prompt(strategy_name)
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_MESSAGE}"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def validate_config(config: Dict[str, Any], train_size: int, max_steps_cap: int) -> Dict[str, Any]:
    default = {
        "data_generation": {
            "use_basic_augmentations": False,
            "use_size_augmentations": False,
            "use_chain_augmentations": False,
            "use_repeat_augmentations": False,
        },
        "training": {
            "strategy": "train_using_output_tokens",
            "learning_rate": 1e-4,
            "num_train_epochs": 0,
        },
    }
    data = config.get("data_generation", {}) if isinstance(config.get("data_generation"), dict) else {}
    train = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    out = {
        "data_generation": {
            "use_basic_augmentations": bool(data.get("use_basic_augmentations", default["data_generation"]["use_basic_augmentations"])),
            "use_size_augmentations": bool(data.get("use_size_augmentations", default["data_generation"]["use_size_augmentations"])),
            "use_chain_augmentations": bool(data.get("use_chain_augmentations", default["data_generation"]["use_chain_augmentations"])),
            "use_repeat_augmentations": bool(data.get("use_repeat_augmentations", default["data_generation"]["use_repeat_augmentations"])),
        },
        "training": {
            "strategy": str(train.get("strategy", default["training"]["strategy"])),
            "learning_rate": float(train.get("learning_rate", default["training"]["learning_rate"])),
            "num_train_epochs": int(train.get("num_train_epochs", default["training"]["num_train_epochs"])),
        },
    }
    if out["training"]["strategy"] not in {"train_using_all_tokens", "train_using_output_tokens"}:
        out["training"]["strategy"] = default["training"]["strategy"]
        out["training"]["num_train_epochs"] = 0
    if out["training"]["learning_rate"] <= 0 or out["training"]["learning_rate"] > 1e-2:
        out["training"]["learning_rate"] = default["training"]["learning_rate"]
    if out["training"]["num_train_epochs"] < 0:
        out["training"]["num_train_epochs"] = 0
    if out["training"]["num_train_epochs"] * max(train_size // 2, 1) > max_steps_cap:
        out["training"]["num_train_epochs"] = 0
    return out


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams  # imported lazily so --help works without vLLM in the active env

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    archive = load_archive(args.archive_path)
    tasks = read_tasks_from_single_file(args.challenge_file, solution_file=args.solution_file)
    tasks = select_base_tasks(tasks, args.n_tasks)

    standard_formatter = TextTaskRepresenter(
        example_representer=TextExampleRepresenter(
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )
    representer = GPTTextMessageRepresenterV2(task_representer=standard_formatter)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"],
    )

    generator = LLM(model=args.model_name)
    sampling = SamplingParams(max_tokens=args.max_generation_tokens, temperature=args.temperature)
    ttt = TTT(model_name=args.model_name, lora_config=lora_config)

    output_root = REPO_ROOT / args.output_root / args.experiment_name
    output_root.mkdir(parents=True, exist_ok=True)

    final_configs_and_indices: Dict[str, Dict[str, Any]] = {}

    for i, task in enumerate(tasks):
        base_task_name = task.name[:-2] if task.name.endswith("-0") else task.name
        final_configs_and_indices[base_task_name] = {}

        for edit_idx in range(args.n_self_edits_per_task):
            strategy, weights = sample_strategy(archive, rng)
            strategy_name = strategy["name"]
            prompt = get_prompt(task, strategy_name)
            response = generator.generate(prompt, sampling_params=sampling)
            raw_text = response[0].outputs[0].text
            try:
                config = json.loads(raw_text)
            except json.JSONDecodeError:
                config = {}

            # Build train data to know whether the config is even feasible.
            provisional = validate_config(config, train_size=1, max_steps_cap=args.max_steps_cap)
            augmenters = get_augmenters(
                include_basic=provisional["data_generation"]["use_basic_augmentations"],
                include_size=provisional["data_generation"]["use_size_augmentations"],
                include_chain=provisional["data_generation"]["use_chain_augmentations"],
                include_repeat=provisional["data_generation"]["use_repeat_augmentations"],
            )
            train_data = process_task(
                task=task,
                augmenters=augmenters,
                formatter=representer,
                tokenizer=tokenizer,
                leave_n=args.leave_n,
                permute_n=args.permute_n,
                max_examples=args.max_train_examples,
                seed=args.seed,
            )
            config = validate_config(config, train_size=len(train_data), max_steps_cap=args.max_steps_cap)

            adapter_path = None
            if train_data and config["training"]["num_train_epochs"] > 0:
                task_text_list = [data["full_text"] for data in train_data]
                adapter_path = ttt.update_model(
                    task_text_list=task_text_list,
                    output_dir=str(output_root / base_task_name / str(edit_idx)),
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    learning_rate=config["training"]["learning_rate"],
                    num_train_epochs=config["training"]["num_train_epochs"],
                    lr_scheduler_type=args.default_lr_scheduler,
                    loss_on_all_tokens=(config["training"]["strategy"] == "train_using_all_tokens"),
                )
            else:
                # Still create the directory so the experiment bookkeeping stays regular.
                empty_dir = output_root / base_task_name / str(edit_idx)
                empty_dir.mkdir(parents=True, exist_ok=True)
                adapter_path = str(empty_dir)

            final_configs_and_indices[base_task_name][str(edit_idx)] = {
                "strategy_name": strategy_name,
                "strategy_weights_before_sample": {
                    archive["strategies"][idx]["name"]: weights[idx] for idx in range(len(archive["strategies"]))
                },
                "config": config,
                "prompt": prompt,
                "response": raw_text,
                "adapter_path": adapter_path,
                "train_examples": len(train_data),
            }
            print(f"[{i+1}/{len(tasks)}] {base_task_name} edit {edit_idx}: strategy={strategy_name} train_examples={len(train_data)}")

    configs_file = output_root / "final_configs_and_indices.json"
    with open(configs_file, "w", encoding="utf-8") as handle:
        json.dump(final_configs_and_indices, handle, indent=2, cls=NumpyEncoder)
    print(f"Saved PRAXIS few-shot configs -> {configs_file}")


if __name__ == "__main__":
    main()
