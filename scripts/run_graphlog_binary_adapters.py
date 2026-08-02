from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
FEW_SHOT_DIR = REPO_ROOT / "few-shot"
if str(FEW_SHOT_DIR) not in sys.path:
    sys.path.insert(0, str(FEW_SHOT_DIR))

SYSTEM_PROMPT = (
    "Infer whether each directed graph query has the requested target relation from labeled GraphLog examples. "
    "Labels mean A = YES and B = NO. "
    "Then answer the new graph. Return exactly one line: LABEL: A or LABEL: B."
)

PRAXIS_HINT = (
    "Compare positive and negative examples by their path relation sequences. Look for relation-composition "
    "patterns that imply the target relation before answering."
)


def user_prompt(query: Dict[str, Any], support: List[Dict[str, Any]] | None = None) -> str:
    parts = []
    for idx, item in enumerate(support or [], start=1):
        parts.append(f"Example {idx}\n{item['text']}\nLABEL: {label_for(item['answer'])}")
    parts.append(f"Now solve this GraphLog example.\n{query['text']}\nLABEL:")
    return "\n\n---\n\n".join(parts)


def chat_text(
    tokenizer,
    query: Dict[str, Any],
    mode: str,
    *,
    support: List[Dict[str, Any]] | None = None,
    answer: str | None = None,
) -> str:
    system = SYSTEM_PROMPT
    if mode == "praxis":
        system += " " + PRAXIS_HINT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt(query, support=support)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": f"LABEL: {label_for(answer)}"})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "LABEL:"


def label_for(answer: str) -> str:
    return "A" if answer == "YES" else "B"


def answer_for_label(label: str) -> str:
    return "YES" if label == "A" else "NO"


def normalize_label(text: str) -> str:
    text = text.strip().splitlines()[0].upper()
    text = text.replace("LABEL:", "").strip()
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    return text


def score_label(model, tokenizer, prompt: str) -> tuple[str, Dict[str, float]]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    a_score = float(log_probs[a_id].detach().cpu())
    b_score = float(log_probs[b_id].detach().cpu())
    return ("A" if a_score >= b_score else "B"), {"A": a_score, "B": b_score}


def train_adapters(args: argparse.Namespace, data: Dict[str, Any], tokenizer) -> Path:
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
    for idx, task in enumerate(data["tasks"][: args.n_tasks], start=1):
        reset_lora(model, initial_lora)
        train_texts = []
        supports = task["train"]
        for heldout_idx, heldout in enumerate(supports):
            other_supports = [item for j, item in enumerate(supports) if j != heldout_idx]
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
        print(f"[{idx}/{min(args.n_tasks, len(data['tasks']))}] trained {task['id']} examples={len(train_texts)}", flush=True)

    (out_root / "final_configs_and_indices.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_root


def reset_lora(model, initial_lora: Dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in initial_lora:
            param.data.copy_(initial_lora[name])


def tokenize_label_only(tokenizer, texts: List[str]) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(texts, truncation=True, max_length=8192, padding="longest", return_tensors="pt")
    labels = encoded["input_ids"].clone()
    for row_idx, text in enumerate(texts):
        marker_idx = text.rfind("LABEL:")
        if marker_idx < 0:
            raise ValueError("Training text lacks LABEL marker")
        prefix = text[: marker_idx + len("LABEL:")]
        prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        labels[row_idx, :prefix_len] = -100
        labels[row_idx][encoded["attention_mask"][row_idx] == 0] = -100
    encoded["labels"] = labels
    return encoded


def train_label_only(
    *,
    model,
    tokenizer,
    texts: List[str],
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
    print(f"Training on {len(texts)} label-only examples for {epochs} epochs, lr: {learning_rate}")
    trainer.train()


def evaluate(args: argparse.Namespace, data: Dict[str, Any], tokenizer, adapter_root: Path | None) -> Dict[str, Any]:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    correct_base = 0
    correct_adapter = 0
    tasks = data["tasks"][: args.n_tasks]
    for idx, task in enumerate(tasks, start=1):
        prompt = chat_text(tokenizer, task["query"], args.mode, support=task.get("train", []))
        if args.score_method == "logprob":
            base_label, base_scores = score_label(base_model, tokenizer, prompt)
            base_pred = answer_for_label(base_label)
            base_text = f"logprob A={base_scores['A']:.4f} B={base_scores['B']:.4f}"
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
            with torch.no_grad():
                output = base_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            base_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            base_pred = answer_for_label(normalize_label(base_text))
        answer = task["query"]["answer"]
        base_ok = base_pred == answer
        correct_base += int(base_ok)

        adapter_text = ""
        adapter_pred = ""
        adapter_ok = False
        if adapter_root is not None:
            adapter_dir = adapter_root / task["id"] / "0"
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            if args.score_method == "logprob":
                adapter_label, adapter_scores = score_label(model, tokenizer, prompt)
                adapter_pred = answer_for_label(adapter_label)
                adapter_text = f"logprob A={adapter_scores['A']:.4f} B={adapter_scores['B']:.4f}"
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                adapter_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                adapter_pred = answer_for_label(normalize_label(adapter_text))
            adapter_ok = adapter_pred == answer
            correct_adapter += int(adapter_ok)
            base_model = model.unload()

        results.append(
            {
                "id": task["id"],
                "answer": answer,
                "baseline_prediction": base_pred,
                "baseline_raw": base_text,
                "baseline_correct": base_ok,
                "adapter_prediction": adapter_pred,
                "adapter_raw": adapter_text,
                "adapter_correct": adapter_ok,
                "metadata": task.get("metadata", {}),
            }
        )
        print(f"[{idx}/{len(tasks)}] {task['id']} base={int(base_ok)} adapter={int(adapter_ok)} answer={answer}", flush=True)

    return {
        "n": len(tasks),
        "baseline_accuracy": correct_base / max(len(tasks), 1),
        "adapter_accuracy": correct_adapter / max(len(tasks), 1) if adapter_root is not None else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["praxis", "seal"], required=True)
    parser.add_argument("--phase", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--dataset", default="data/graphlog/graphlog_binary_test_50.json")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--n_tasks", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--score_method", choices=["logprob", "generate"], default="logprob")
    parser.add_argument("--output_root", default="loras/graphlog-binary")
    parser.add_argument("--results_root", default="runpod_results/graphlog_binary")
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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
            f"adapter={summary['adapter_accuracy']*100:.2f}% -> {results_dir / 'final_results.json'}"
        )


if __name__ == "__main__":
    main()
