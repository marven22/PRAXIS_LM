from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from run_clutrr_llm_baseline import LABELS, normalize


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_edge_types(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in ast.literal_eval(value)]


def chain_text(row: dict) -> str:
    return " -> ".join(parse_edge_types(row["edge_types"]))


def system_prompt(mode: str) -> str:
    if mode == "seal":
        return (
            "Learn from the support relation-chain examples and predict the final kinship label. "
            "Return exactly one line: RELATION: <label>."
        )
    if mode == "omni":
        return (
            "Use the support examples as reusable task demonstrations. Identify recurring input-output "
            "patterns in relation chains and predict the final kinship label. Return exactly one line: RELATION: <label>."
        )
    return (
        "Use local kinship-composition skills from the support examples. Compose adjacent relation spans when useful, "
        "then predict the final kinship label. Return exactly one line: RELATION: <label>."
    )


def user_prompt(query: dict, support: list[dict], mode: str) -> str:
    parts = ["Candidate labels: " + ", ".join(LABELS)]
    for idx, item in enumerate(support, start=1):
        parts.append(
            f"Example {idx}\n"
            f"Relation chain: {chain_text(item)}\n"
            f"RELATION: {item['target_text']}"
        )
    parts.append(
        "Now solve this relation-chain example.\n"
        f"Relation chain: {chain_text(query)}\n"
        "RELATION:"
    )
    return "\n\n---\n\n".join(parts)


def chat_text(tokenizer, query: dict, support: list[dict], mode: str, answer: str | None = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt(mode)},
        {"role": "user", "content": user_prompt(query, support, mode)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": f"RELATION: {answer}"})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "RELATION:"


def score_sequence(model, tokenizer, prompt: str, label: str) -> float:
    continuation = " " + label
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    label_ids = full_ids[len(prompt_ids) :]
    inputs = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=inputs).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    score = 0.0
    start = len(prompt_ids)
    for offset, token_id in enumerate(label_ids):
        score += float(log_probs[start + offset - 1, token_id].detach().cpu())
    return score / max(len(label_ids), 1)


def predict_label(model, tokenizer, prompt: str) -> tuple[str, dict[str, float]]:
    scores = {label: score_sequence(model, tokenizer, prompt, label) for label in LABELS}
    return max(scores, key=scores.get), scores


def reset_lora(model, initial_lora: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in initial_lora:
            param.data.copy_(initial_lora[name])


def tokenize_label_only(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    encoded = tokenizer(texts, truncation=True, max_length=4096, padding="longest", return_tensors="pt")
    labels = encoded["input_ids"].clone()
    for row_idx, text in enumerate(texts):
        marker_idx = text.rfind("RELATION:")
        if marker_idx < 0:
            raise ValueError("Training text lacks RELATION marker")
        prefix = text[: marker_idx + len("RELATION:")]
        prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        labels[row_idx, :prefix_len] = -100
        labels[row_idx][encoded["attention_mask"][row_idx] == 0] = -100
    encoded["labels"] = labels
    return encoded


def train_label_only(model, tokenizer, texts: list[str], output_dir: Path, args: argparse.Namespace) -> None:
    ds = Dataset.from_dict(tokenize_label_only(tokenizer, texts))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim="adamw_torch",
        warmup_steps=0,
    )
    Trainer(model=model, args=training_args, train_dataset=ds).train()


def select_subset(rows, n: int, min_len: int, max_len: int) -> list[dict]:
    selected = []
    by_len = defaultdict(int)
    max_per_len = max(4, n // max(1, max_len - min_len + 1) + 2)
    for row in rows:
        length = len(parse_edge_types(row["edge_types"]))
        if min_len <= length <= max_len and by_len[length] < max_per_len:
            selected.append(dict(row))
            by_len[length] += 1
        if len(selected) >= n:
            break
    return selected


def support_pool(train_rows: Iterable[dict], support_n: int) -> dict[int, list[dict]]:
    by_len = defaultdict(list)
    for row in train_rows:
        by_len[len(parse_edge_types(row["edge_types"]))].append(dict(row))
    return {length: rows[: support_n * 4] for length, rows in by_len.items()}


def choose_support(query: dict, pool: dict[int, list[dict]], support_n: int) -> list[dict]:
    q_len = len(parse_edge_types(query["edge_types"]))
    candidates = []
    for length in [q_len, q_len - 1, q_len + 1, 2, 3, 4]:
        candidates.extend(pool.get(length, []))
    seen = set()
    support = []
    for row in candidates:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        support.append(row)
        if len(support) >= support_n:
            break
    return support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seal", "omni", "praxis_prompt"], required=True)
    parser.add_argument("--phase", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--support_n", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--output_root", default="loras/clutrr")
    parser.add_argument("--results_root", default="analysis_results/clutrr_adapters")
    args = parser.parse_args()

    train_rows = list(load_dataset("CLUTRR/v1", args.config, split="train"))
    test_rows = list(load_dataset("CLUTRR/v1", args.config, split="test"))
    tasks = select_subset(test_rows, args.n_tasks, 2, 10)
    pool = support_pool(train_rows, args.support_n)
    for task in tasks:
        task["support"] = choose_support(task, pool, args.support_n)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_root = REPO_ROOT / args.output_root / args.experiment_name

    if args.phase in {"train", "train_eval"}:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"],
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model = get_peft_model(base, lora_config)
        initial_lora = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }
        adapter_root.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for idx, task in enumerate(tasks, start=1):
            reset_lora(model, initial_lora)
            support = task["support"]
            train_texts = []
            for heldout_idx, heldout in enumerate(support):
                other_supports = [item for j, item in enumerate(support) if j != heldout_idx]
                train_texts.append(chat_text(tokenizer, heldout, other_supports, args.mode, heldout["target_text"]))
            adapter_dir = adapter_root / task["id"] / "0"
            train_label_only(model, tokenizer, train_texts, adapter_dir, args)
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            manifest[task["id"]] = {"0": {"mode": args.mode, "support_n": len(support), "adapter_path": str(adapter_dir)}}
            print(f"[{idx}/{len(tasks)}] trained {task['id']} support={len(support)}", flush=True)
        (adapter_root / "final_configs_and_indices.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.phase in {"eval", "train_eval"}:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        results = []
        correct_base = 0
        correct_adapter = 0
        for idx, task in enumerate(tasks, start=1):
            prompt = chat_text(tokenizer, task, task["support"], args.mode)
            base_pred, base_scores = predict_label(base_model, tokenizer, prompt)
            answer = normalize(task["target_text"])
            base_ok = base_pred == answer
            correct_base += int(base_ok)
            adapter_dir = adapter_root / task["id"] / "0"
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            adapter_pred, adapter_scores = predict_label(model, tokenizer, prompt)
            adapter_ok = adapter_pred == answer
            correct_adapter += int(adapter_ok)
            base_model = model.unload()
            results.append(
                {
                    "id": task["id"],
                    "chain": parse_edge_types(task["edge_types"]),
                    "answer": answer,
                    "baseline_prediction": base_pred,
                    "baseline_correct": base_ok,
                    "adapter_prediction": adapter_pred,
                    "adapter_correct": adapter_ok,
                    "baseline_scores": base_scores,
                    "adapter_scores": adapter_scores,
                }
            )
            print(f"[{idx}/{len(tasks)}] base={int(base_ok)} adapter={int(adapter_ok)} answer={answer}", flush=True)
        summary = {
            "mode": args.mode,
            "model": args.model_name,
            "config": args.config,
            "n": len(tasks),
            "support_n": args.support_n,
            "baseline_accuracy": correct_base / len(tasks),
            "adapter_accuracy": correct_adapter / len(tasks),
            "results": results,
        }
        results_dir = REPO_ROOT / args.results_root / args.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "final_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"baseline={summary['baseline_accuracy']*100:.2f}% adapter={summary['adapter_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
