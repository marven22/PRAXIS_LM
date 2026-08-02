from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from run_clutrr_llm_baseline import LABELS, normalize


def format_messages(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


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


def train_adapter(args, tokenizer) -> Path:
    rows = []
    with open(args.edits, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[: args.max_train] if args.max_train > 0 else rows
    texts = [format_messages(tokenizer, item["messages"]) for item in rows]

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"],
    )
    model = get_peft_model(base, lora_config)
    ds = Dataset.from_dict(tokenize_label_only(tokenizer, texts))
    out_dir = Path(args.output_root) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        optim="adamw_torch",
        warmup_ratio=0.03,
        seed=args.seed,
        data_seed=args.seed,
    )
    Trainer(model=model, args=train_args, train_dataset=ds).train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "train_manifest.json").write_text(
        json.dumps(
            {"edits": args.edits, "n_train": len(rows), "epochs": args.epochs, "lr": args.lr, "seed": args.seed},
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def chain_text(row: dict) -> str:
    return " -> ".join(row["edge_types"])


def eval_prompt(tokenizer, row: dict) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "Use learned family-relation composition edits to solve this chain.\n"
                f"Relation chain: {chain_text(row)}\n"
                f"Candidate labels: {', '.join(LABELS)}\n"
                "Final relation:"
            ),
        }
    ]
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


def evaluate(args, tokenizer, adapter_dir: Path) -> dict:
    rows = json.loads(Path(args.eval_dataset).read_text(encoding="utf-8"))[: args.n_eval]
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    adapter = PeftModel.from_pretrained(base, str(adapter_dir))
    results = []
    by_len = defaultdict(list)
    correct = 0
    for idx, row in enumerate(rows, start=1):
        prompt = eval_prompt(tokenizer, row)
        pred, scores = predict_label(adapter, tokenizer, prompt)
        answer = normalize(row["target"])
        ok = pred == answer
        correct += int(ok)
        by_len[row["chain_len"]].append(ok)
        results.append(
            {
                "id": row["id"],
                "chain_len": row["chain_len"],
                "chain": row["edge_types"],
                "answer": answer,
                "prediction": pred,
                "correct": ok,
                "scores": scores,
            }
        )
        print(f"[{idx}/{len(rows)}] pred={pred} answer={answer} correct={ok}", flush=True)
    return {
        "model": args.model_name,
        "adapter_dir": str(adapter_dir),
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "accuracy_by_chain_len": {str(k): sum(v) / len(v) for k, v in sorted(by_len.items())},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--edits", default="data/clutrr/seal_edits_train.jsonl")
    parser.add_argument("--eval_dataset", default="data/clutrr/clutrr_test_50.json")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--max_train", type=int, default=1062)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", default="loras/clutrr-seal-edits")
    parser.add_argument("--results_root", default="analysis_results/clutrr_seal_edits")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_dir = Path(args.output_root) / args.experiment_name
    if args.phase in {"train", "train_eval"}:
        adapter_dir = train_adapter(args, tokenizer)
    if args.phase in {"eval", "train_eval"}:
        summary = evaluate(args, tokenizer, adapter_dir)
        out_dir = Path(args.results_root) / args.experiment_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"seal_edits_accuracy={summary['accuracy']*100:.2f}% -> {out_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()
