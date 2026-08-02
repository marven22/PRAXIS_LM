from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from run_clutrr_llm_baseline import LABELS, normalize


def parse_edges(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in ast.literal_eval(value)]


def chain_text(row: dict) -> str:
    return " -> ".join(parse_edges(row["edge_types"]))


def select_subset(rows, n: int, min_len: int = 2, max_len: int = 10) -> list[dict]:
    if n <= 0:
        return [
            dict(row)
            for row in rows
            if min_len <= len(parse_edges(row["edge_types"])) <= max_len
        ]
    selected = []
    by_len = defaultdict(int)
    max_per_len = max(4, n // max(1, max_len - min_len + 1) + 2)
    for row in rows:
        length = len(parse_edges(row["edge_types"]))
        if min_len <= length <= max_len and by_len[length] < max_per_len:
            selected.append(dict(row))
            by_len[length] += 1
        if len(selected) >= n:
            break
    return selected


def support_pool(train_rows, support_n: int) -> dict[int, list[dict]]:
    by_len = defaultdict(list)
    for row in train_rows:
        by_len[len(parse_edges(row["edge_types"]))].append(dict(row))
    return {length: rows[: support_n * 8] for length, rows in by_len.items()}


def choose_support(query: dict, pool: dict[int, list[dict]], support_n: int) -> list[dict]:
    q_len = len(parse_edges(query["edge_types"]))
    candidates = []
    for length in [q_len, q_len - 1, q_len + 1, 2, 3, 4, 5]:
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


def seal_self_edit_prompt(query: dict, support: list[dict]) -> str:
    examples = []
    for idx, row in enumerate(support, start=1):
        examples.append(
            f"Example {idx}\n"
            f"Relation chain: {chain_text(row)}\n"
            f"Final relation: {row['target_text']}"
        )
    return (
        "You are SEAL, a self-adapting language model. Generate a self-edit for adapting a "
        "small LoRA model to this family-relation composition task.\n\n"
        "The task maps a relation chain to one final kinship label. You may propose reusable "
        "composition edits, synthetic chain examples, and training hyperparameters. Use only "
        "patterns supported by the examples below. Do not solve the held-out query directly.\n\n"
        + "\n\n".join(examples)
        + "\n\n"
        "Return only valid JSON with this schema. The placeholder strings below are not real edits; "
        "replace them with relation names observed in the support examples.\n"
        "{\n"
        '  "edits": [{"chain": ["RELATION_A", "RELATION_B"], "relation": "OUTPUT_RELATION"}],\n'
        '  "synthetic_examples": [{"chain": ["RELATION_A", "RELATION_B", "RELATION_C"], "relation": "OUTPUT_RELATION"}],\n'
        '  "training": {"learning_rate": 0.0005, "num_train_epochs": 3, "loss": "output_only"}\n'
        "}\n"
    )


def extract_json(text: str) -> dict | None:
    text = text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced[-1].strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_self_edit(candidate: dict | None, support: list[dict], args) -> dict:
    if not isinstance(candidate, dict):
        candidate = {}
    edits = []
    for item in candidate.get("edits", []):
        if not isinstance(item, dict):
            continue
        chain = item.get("chain")
        relation = normalize(item.get("relation", ""))
        if isinstance(chain, list) and 1 <= len(chain) <= 4 and relation in LABELS:
            edits.append({"chain": [str(x) for x in chain], "relation": relation})
    synthetic = []
    for item in candidate.get("synthetic_examples", []):
        if not isinstance(item, dict):
            continue
        chain = item.get("chain")
        relation = normalize(item.get("relation", ""))
        if isinstance(chain, list) and 1 <= len(chain) <= 10 and relation in LABELS:
            synthetic.append({"chain": [str(x) for x in chain], "relation": relation})

    # SEAL-style guardrail: if the generated edit is empty, the update is a valid
    # no-op over the support examples rather than an oracle fallback.
    if not edits and not synthetic:
        synthetic = [{"chain": parse_edges(row["edge_types"]), "relation": normalize(row["target_text"])} for row in support]

    training = candidate.get("training", {}) if isinstance(candidate.get("training"), dict) else {}
    lr = training.get("learning_rate", args.lr)
    epochs = training.get("num_train_epochs", args.epochs)
    try:
        lr = float(lr)
    except (TypeError, ValueError):
        lr = args.lr
    try:
        epochs = int(epochs)
    except (TypeError, ValueError):
        epochs = args.epochs
    lr = min(max(lr, 1e-6), 1e-3)
    epochs = min(max(epochs, 0), args.max_generated_epochs)
    loss = training.get("loss", "output_only")
    if loss not in {"output_only", "all_tokens"}:
        loss = "output_only"
    return {"edits": edits[: args.max_edits], "synthetic_examples": synthetic[: args.max_synthetic], "training": {"learning_rate": lr, "num_train_epochs": epochs, "loss": loss}}


def train_text(tokenizer, support: list[dict], self_edit: dict) -> list[str]:
    rows = []
    items = []
    for row in support:
        items.append({"chain": parse_edges(row["edge_types"]), "relation": normalize(row["target_text"])})
    items.extend(self_edit["edits"])
    items.extend(self_edit["synthetic_examples"])
    for item in items:
        messages = [
            {
                "role": "user",
                "content": (
                    "Use SEAL self-edit supervision to learn this relation-chain task.\n"
                    f"Candidate labels: {', '.join(LABELS)}\n"
                    f"Relation chain: {' -> '.join(item['chain'])}\n"
                    "Final relation:"
                ),
            },
            {"role": "assistant", "content": f"RELATION: {item['relation']}"},
        ]
        rows.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return rows


def tokenize_label_only(tokenizer, texts: list[str], loss_on_all_tokens: bool) -> dict[str, torch.Tensor]:
    encoded = tokenizer(texts, truncation=True, max_length=4096, padding="longest", return_tensors="pt")
    labels = encoded["input_ids"].clone()
    if not loss_on_all_tokens:
        for row_idx, text in enumerate(texts):
            marker_idx = text.rfind("RELATION:")
            prefix = text[: marker_idx + len("RELATION:")]
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            labels[row_idx, :prefix_len] = -100
    for row_idx in range(labels.shape[0]):
        labels[row_idx][encoded["attention_mask"][row_idx] == 0] = -100
    encoded["labels"] = labels
    return encoded


def eval_prompt(tokenizer, row: dict, support: list[dict]) -> str:
    parts = ["Candidate labels: " + ", ".join(LABELS)]
    for idx, item in enumerate(support, start=1):
        parts.append(f"Example {idx}\nRelation chain: {chain_text(item)}\nRELATION: {item['target_text']}")
    parts.append(f"Now solve this relation-chain example.\nRelation chain: {chain_text(row)}\nRELATION:")
    messages = [{"role": "user", "content": "\n\n---\n\n".join(parts)}]
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


def snapshot_lora(model) -> dict[str, torch.Tensor]:
    return {
        name: param.data.clone().detach()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def reset_lora(model, initial_lora: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in initial_lora:
            param.data.copy_(initial_lora[name])


def generate_self_edit(model, tokenizer, prompt: str, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n_tasks", type=int, default=50)
    parser.add_argument("--support_n", type=int, default=8)
    parser.add_argument("--n_self_edits", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_generated_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_edits", type=int, default=12)
    parser.add_argument("--max_synthetic", type=int, default=16)
    parser.add_argument("--output_root", default="loras/clutrr-official-seal")
    parser.add_argument("--results_root", default="analysis_results/clutrr_official_seal")
    args = parser.parse_args()

    train_rows = list(load_dataset("CLUTRR/v1", args.config, split="train"))
    test_rows = list(load_dataset("CLUTRR/v1", args.config, split="test"))
    tasks = select_subset(test_rows, args.n_tasks)
    pool = support_pool(train_rows, args.support_n)
    for task in tasks:
        task["support"] = choose_support(task, pool, args.support_n)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    initial_lora = snapshot_lora(model)

    out_root = Path(args.output_root) / args.experiment_name
    out_root.mkdir(parents=True, exist_ok=True)
    result_root = Path(args.results_root) / args.experiment_name
    result_root.mkdir(parents=True, exist_ok=True)

    generated = {}
    results = []
    correct_base = 0
    correct_best = 0
    correct_vote = 0
    for task_idx, task in enumerate(tasks, start=1):
        support = task["support"]
        prompt = eval_prompt(tokenizer, task, support)
        reset_lora(model, initial_lora)
        base_pred, base_scores = predict_label(model, tokenizer, prompt)
        answer = normalize(task["target_text"])
        correct_base += int(base_pred == answer)

        edit_records = []
        adapter_preds = []
        adapter_scores_by_idx = []
        for edit_idx in range(args.n_self_edits):
            raw = generate_self_edit(base, tokenizer, seal_self_edit_prompt(task, support), args.temperature)
            self_edit = normalize_self_edit(extract_json(raw), support, args)
            train_rows_text = train_text(tokenizer, support, self_edit)

            reset_lora(model, initial_lora)
            ds = Dataset.from_dict(
                tokenize_label_only(tokenizer, train_rows_text, self_edit["training"]["loss"] == "all_tokens")
            )
            adapter_dir = out_root / task["id"] / str(edit_idx)
            train_args = TrainingArguments(
                output_dir=str(adapter_dir),
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=self_edit["training"]["learning_rate"],
                num_train_epochs=self_edit["training"]["num_train_epochs"],
                lr_scheduler_type="cosine",
                logging_steps=50,
                save_strategy="no",
                report_to="none",
                bf16=torch.cuda.is_available(),
                remove_unused_columns=False,
                optim="adamw_torch",
                warmup_steps=0,
            )
            Trainer(model=model, args=train_args, train_dataset=ds).train()
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            torch.cuda.empty_cache()

            pred, scores = predict_label(model, tokenizer, prompt)
            adapter_preds.append(pred)
            adapter_scores_by_idx.append(scores)
            edit_records.append({"raw": raw, "self_edit": self_edit, "prediction": pred, "correct": pred == answer})

        pred_counts = Counter(adapter_preds)
        vote_pred = pred_counts.most_common(1)[0][0] if adapter_preds else base_pred
        best_pred = None
        best_score = -1e99
        for scores in adapter_scores_by_idx:
            pred = max(scores, key=scores.get)
            if scores[pred] > best_score:
                best_score = scores[pred]
                best_pred = pred
        best_pred = best_pred or vote_pred
        correct_vote += int(vote_pred == answer)
        correct_best += int(best_pred == answer)
        generated[task["id"]] = edit_records
        row = {
            "id": task["id"],
            "chain": parse_edges(task["edge_types"]),
            "chain_len": len(parse_edges(task["edge_types"])),
            "answer": answer,
            "base_prediction": base_pred,
            "base_correct": base_pred == answer,
            "adapter_predictions": adapter_preds,
            "vote_prediction": vote_pred,
            "vote_correct": vote_pred == answer,
            "best_score_prediction": best_pred,
            "best_score_correct": best_pred == answer,
            "base_scores": base_scores,
        }
        results.append(row)
        print(
            f"[{task_idx}/{len(tasks)}] base={int(base_pred == answer)} "
            f"vote={int(vote_pred == answer)} best={int(best_pred == answer)} answer={answer}",
            flush=True,
        )

    summary = {
        "method": "official_seal_clutrr_port",
        "model": args.model_name,
        "config": args.config,
        "n": len(tasks),
        "support_n": args.support_n,
        "n_self_edits": args.n_self_edits,
        "baseline_accuracy": correct_base / max(len(tasks), 1),
        "seal_vote_accuracy": correct_vote / max(len(tasks), 1),
        "seal_best_score_accuracy": correct_best / max(len(tasks), 1),
        "results": results,
    }
    (result_root / "final_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (result_root / "self_edits.json").write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
