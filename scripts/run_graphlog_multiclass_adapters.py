from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


REPO_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPTS = {
    "direct": (
        "Infer the relation label for each GraphLog query from the labeled graph examples. "
        "Return exactly one line in the form RELATION: <label>."
    ),
    "curriculum": (
        "Infer the relation label for each GraphLog query from the labeled graph examples. "
        "Use the support examples as a small task-local curriculum, starting from recurring path patterns. "
        "Return exactly one line in the form RELATION: <label>."
    ),
    "seal": (
        "Infer the relation label for each GraphLog query from the labeled graph examples. "
        "Use the task-local examples to form an edit-like correction to the initial answer. "
        "Return exactly one line in the form RELATION: <label>."
    ),
    "praxis": (
        "Infer the relation label for each GraphLog query from the labeled graph examples. "
        "Compare path relation sequences in the support set and select reusable composition patterns before answering. "
        "Return exactly one line in the form RELATION: <label>."
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def user_prompt(query: dict[str, Any], support: list[dict[str, Any]] | None = None) -> str:
    parts = []
    for idx, item in enumerate(support or [], start=1):
        parts.append(f"Example {idx}\n{item['text']}\nRELATION: {item['answer']}")
    parts.append(f"Now solve this GraphLog example.\n{query['text']}\nRELATION:")
    return "\n\n---\n\n".join(parts)


def chat_text(
    tokenizer,
    query: dict[str, Any],
    mode: str,
    *,
    support: list[dict[str, Any]] | None = None,
    answer: str | None = None,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[mode]},
        {"role": "user", "content": user_prompt(query, support=support)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": f"RELATION: {answer}"})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "RELATION:"


def label_logprob(model, tokenizer, prompt: str, label: str) -> float:
    prefix = tokenizer(prompt, return_tensors="pt").to(model.device)
    label_ids = tokenizer(" " + label, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    input_ids = torch.cat([prefix.input_ids, label_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    start = prefix.input_ids.shape[1]
    score = 0.0
    for pos in range(start, input_ids.shape[1]):
        token_id = input_ids[0, pos]
        score += float(log_probs[0, pos - 1, token_id].detach().cpu())
    return score


def score_relation(model, tokenizer, prompt: str, labels: list[str]) -> tuple[str, dict[str, float]]:
    scores = {label: label_logprob(model, tokenizer, prompt, label) for label in labels}
    best = max(scores, key=scores.get)
    return best, scores


def reset_lora(model, initial_lora: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in initial_lora:
            param.data.copy_(initial_lora[name])


def tokenize_label_only(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    encoded = tokenizer(texts, truncation=True, max_length=8192, padding="longest", return_tensors="pt")
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


def train_label_only(
    *,
    model,
    tokenizer,
    texts: list[str],
    output_dir: Path,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    epochs: int,
) -> None:
    training_data = tokenize_label_only(tokenizer, texts)
    ds = Dataset.from_dict(training_data)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim="adamw_torch",
        warmup_steps=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()


def train_adapters(args: argparse.Namespace, data: dict[str, Any], tokenizer) -> Path:
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_root = REPO_ROOT / args.output_root / args.experiment_name
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    tasks = data["tasks"][: args.n_tasks]
    for idx, task in enumerate(tasks, start=1):
        reset_lora(model, initial_lora)
        train_texts = []
        supports = task["train"]
        ordered_supports = order_support(task, supports, args.mode)
        for heldout_idx, heldout in enumerate(ordered_supports):
            other_supports = [item for j, item in enumerate(ordered_supports) if j != heldout_idx]
            train_texts.append(chat_text(tokenizer, heldout, args.mode, support=other_supports, answer=heldout["answer"]))
        adapter_dir = out_root / task["id"] / "0"
        train_label_only(
            model=model,
            tokenizer=tokenizer,
            texts=train_texts,
            output_dir=adapter_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            epochs=args.epochs,
        )
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        manifest[task["id"]] = {
            "0": {
                "mode": args.mode,
                "adapter_path": str(adapter_dir),
                "train_examples": len(train_texts),
                "metadata": task.get("metadata", {}),
            }
        }
        print(f"[{idx}/{len(tasks)}] trained {task['id']} examples={len(train_texts)}", flush=True)
    (out_root / "final_configs_and_indices.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_root


def train_eval_streaming(args: argparse.Namespace, data: dict[str, Any], tokenizer) -> dict[str, Any]:
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    correct_base = 0
    correct_adapter = 0
    tasks = data["tasks"][: args.n_tasks]
    run_started = time.time()
    scratch_root = REPO_ROOT / args.output_root / f"{args.experiment_name}_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    for idx, task in enumerate(tasks, start=1):
        reset_lora(model, initial_lora)
        labels = task.get("labels", [])
        support = [] if args.no_support else order_support(task, task.get("train", []), args.mode)
        prompt = chat_text(tokenizer, task["query"], args.mode, support=support)

        with model.disable_adapter():
            base_pred, base_scores = score_relation(model, tokenizer, prompt, labels)
        answer = task["query"]["answer"]
        base_ok = base_pred == answer
        correct_base += int(base_ok)

        if args.prompt_only_eval:
            adapter_pred, adapter_scores = base_pred, base_scores
        else:
            train_texts = []
            ordered_supports = [] if args.no_support else order_support(task, task["train"], args.mode)
            for heldout_idx, heldout in enumerate(ordered_supports):
                other_supports = [item for j, item in enumerate(ordered_supports) if j != heldout_idx]
                train_texts.append(chat_text(tokenizer, heldout, args.mode, support=other_supports, answer=heldout["answer"]))

            if train_texts:
                train_label_only(
                    model=model,
                    tokenizer=tokenizer,
                    texts=train_texts,
                    output_dir=scratch_root / task["id"],
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    learning_rate=args.lr,
                    epochs=args.epochs,
                )

            adapter_pred, adapter_scores = score_relation(model, tokenizer, prompt, labels)
        adapter_ok = adapter_pred == answer
        correct_adapter += int(adapter_ok)
        rows.append(
            {
                "id": task["id"],
                "answer": answer,
                "baseline_prediction": base_pred,
                "baseline_correct": base_ok,
                "adapter_prediction": adapter_pred,
                "adapter_correct": adapter_ok,
                "improved": (not base_ok) and adapter_ok,
                "degraded": base_ok and (not adapter_ok),
                "metadata": task.get("metadata", {}),
                "baseline_scores": base_scores,
                "adapter_scores": adapter_scores,
            }
        )
        print(
            f"[{idx}/{len(tasks)}] {task['id']} base={int(base_ok)} adapter={int(adapter_ok)} "
            f"answer={answer} base_pred={base_pred} adapter_pred={adapter_pred}",
            flush=True,
        )

    runtime = time.time() - run_started
    baseline_accuracy = correct_base / max(len(tasks), 1)
    adapter_accuracy = correct_adapter / max(len(tasks), 1)
    return {
        "dataset": "GraphLog",
        "task": "multiclass_relation_prediction",
        "method": args.mode,
        "n": len(tasks),
        "model_name": args.model_name,
        "baseline_accuracy": baseline_accuracy,
        "adapter_accuracy": adapter_accuracy,
        "gain": adapter_accuracy - baseline_accuracy,
        "pgr": sum(row["improved"] for row in rows) / max(len(rows), 1),
        "dr": sum(row["degraded"] for row in rows) / max(len(rows), 1),
        "time_per_example_seconds": runtime / max(len(rows), 1),
        "runtime_seconds": runtime,
        "streaming_train_eval": True,
        "prompt_only_eval": args.prompt_only_eval,
        "no_support": args.no_support,
        "results": rows,
    }


def order_support(task: dict[str, Any], support: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode != "curriculum":
        return list(support)
    qdesc = task["query"].get("descriptor", "")
    return sorted(
        support,
        key=lambda item: (
            item.get("descriptor", "") != qdesc,
            len(item.get("descriptor", "").split(",")),
            item.get("answer", ""),
        ),
    )


def evaluate(args: argparse.Namespace, data: dict[str, Any], tokenizer, adapter_root: Path | None) -> dict[str, Any]:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    correct_base = 0
    correct_adapter = 0
    tasks = data["tasks"][: args.n_tasks]
    eval_started = time.time()
    for idx, task in enumerate(tasks, start=1):
        labels = task.get("labels", [])
        support = order_support(task, task.get("train", []), args.mode)
        prompt = chat_text(tokenizer, task["query"], args.mode, support=support)
        base_pred, base_scores = score_relation(base_model, tokenizer, prompt, labels)
        answer = task["query"]["answer"]
        base_ok = base_pred == answer
        correct_base += int(base_ok)

        adapter_pred = ""
        adapter_scores: dict[str, float] = {}
        adapter_ok = False
        if adapter_root is not None:
            adapter_dir = adapter_root / task["id"] / "0"
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            adapter_pred, adapter_scores = score_relation(model, tokenizer, prompt, labels)
            adapter_ok = adapter_pred == answer
            correct_adapter += int(adapter_ok)
            base_model = model.unload()

        rows.append(
            {
                "id": task["id"],
                "answer": answer,
                "baseline_prediction": base_pred,
                "baseline_correct": base_ok,
                "adapter_prediction": adapter_pred,
                "adapter_correct": adapter_ok,
                "improved": (not base_ok) and adapter_ok,
                "degraded": base_ok and (not adapter_ok),
                "metadata": task.get("metadata", {}),
                "baseline_scores": base_scores,
                "adapter_scores": adapter_scores,
            }
        )
        print(
            f"[{idx}/{len(tasks)}] {task['id']} base={int(base_ok)} adapter={int(adapter_ok)} "
            f"answer={answer} base_pred={base_pred} adapter_pred={adapter_pred}",
            flush=True,
        )
    eval_runtime = time.time() - eval_started
    baseline_accuracy = correct_base / max(len(tasks), 1)
    adapter_accuracy = correct_adapter / max(len(tasks), 1) if adapter_root is not None else None
    return {
        "dataset": "GraphLog",
        "task": "multiclass_relation_prediction",
        "method": args.mode,
        "n": len(tasks),
        "model_name": args.model_name,
        "baseline_accuracy": baseline_accuracy,
        "adapter_accuracy": adapter_accuracy,
        "gain": None if adapter_accuracy is None else adapter_accuracy - baseline_accuracy,
        "pgr": None if adapter_root is None else sum(row["improved"] for row in rows) / max(len(rows), 1),
        "dr": None if adapter_root is None else sum(row["degraded"] for row in rows) / max(len(rows), 1),
        "time_per_example_seconds": eval_runtime / max(len(rows), 1),
        "runtime_seconds": eval_runtime,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "curriculum", "seal", "praxis"], required=True)
    parser.add_argument("--phase", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--dataset", default="data/graphlog/graphlog_rule56_full1000.json")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--n_tasks", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--output_root", default="loras/graphlog-multiclass")
    parser.add_argument("--results_root", default="analysis_results/full_scale/graphlog_multiclass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream_train_eval", action="store_true")
    parser.add_argument("--prompt_only_eval", action="store_true")
    parser.add_argument("--no_support", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data = json.loads((REPO_ROOT / args.dataset).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.stream_train_eval:
        if args.phase != "train_eval":
            raise ValueError("--stream_train_eval requires --phase train_eval")
        summary = train_eval_streaming(args, data, tokenizer)
        results_dir = REPO_ROOT / args.results_root / args.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "final_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"baseline={summary['baseline_accuracy']*100:.2f}% "
            f"adapter={summary['adapter_accuracy']*100:.2f}% "
            f"gain={summary['gain']*100:.2f}% -> {results_dir / 'final_results.json'}"
        )
        return

    adapter_root = REPO_ROOT / args.output_root / args.experiment_name
    if args.phase in {"train", "train_eval"}:
        adapter_root = train_adapters(args, data, tokenizer)
    if args.phase in {"eval", "train_eval"}:
        summary = evaluate(args, data, tokenizer, adapter_root)
        results_dir = REPO_ROOT / args.results_root / args.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "final_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"baseline={summary['baseline_accuracy']*100:.2f}% "
            f"adapter={summary['adapter_accuracy']*100:.2f}% "
            f"gain={summary['gain']*100:.2f}% -> {results_dir / 'final_results.json'}"
        )


if __name__ == "__main__":
    main()
