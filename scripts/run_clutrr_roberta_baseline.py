from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def normalize_label(value: str) -> str:
    return str(value).strip().lower()


def make_text(row: dict[str, Any]) -> str:
    story = row.get("story", "")
    query = row.get("query", "")
    edge_types = row.get("edge_types", "")
    return (
        "Story:\n"
        f"{story}\n\n"
        f"Query: {query}\n"
        f"Relation chain: {edge_types}\n"
        "Predict the kinship relation between the queried people."
    )


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    cleaned = []
    for row in rows:
        item = dict(row)
        item["target_text"] = normalize_label(row.get("target", row.get("target_text", "")))
        cleaned.append(item)
    return cleaned


def left_to_right_reference(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    for method in obj.get("methods", []):
        if method.get("method") == "left_to_right_archive":
            return {row["id"]: bool(row["correct"]) for row in method.get("results", [])}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--eval_dataset", default="data/clutrr/clutrr_test_200.json")
    parser.add_argument("--reference_results", default="analysis_results/ablations/clutrr_archive_component_ablations_200.json")
    parser.add_argument("--model_name", default="distilroberta-base")
    parser.add_argument("--output", default="analysis_results/native_baselines/clutrr_distilroberta_200.json")
    parser.add_argument("--max_train", type=int, default=4000)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_raw = list(load_dataset("CLUTRR/v1", args.config, split="train"))
    random.shuffle(train_raw)
    train_raw = train_raw[: args.max_train]
    eval_rows = load_eval_rows(Path(args.eval_dataset))

    labels = sorted({normalize_label(row["target_text"]) for row in train_raw} | {row["target_text"] for row in eval_rows})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_ds = Dataset.from_list(
        [{"text": make_text(row), "label": label2id[normalize_label(row["target_text"])]} for row in train_raw]
    )
    eval_ds = Dataset.from_list(
        [{"text": make_text(row), "label": label2id[row["target_text"]], "row_id": row["id"]} for row in eval_rows]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_eval = eval_ds.map(tokenize, batched=True, remove_columns=["text", "row_id"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    )
    train_args = TrainingArguments(
        output_dir=str(Path(args.output).with_suffix("")) + "_trainer",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    predictions = trainer.predict(tokenized_eval).predictions
    pred_ids = predictions.argmax(axis=-1).tolist()
    gold_ids = [int(item["label"]) for item in eval_ds]
    acc = accuracy_score(gold_ids, pred_ids)

    ref = left_to_right_reference(Path(args.reference_results))
    base_correct = [ref.get(row["id"], False) for row in eval_rows]
    pred_correct = [pred == gold for pred, gold in zip(pred_ids, gold_ids)]
    pgr = sum((not b) and a for b, a in zip(base_correct, pred_correct)) / max(len(eval_rows), 1)
    dr = sum(b and (not a) for b, a in zip(base_correct, pred_correct)) / max(len(eval_rows), 1)
    initial = sum(base_correct) / max(len(eval_rows), 1)
    runtime = time.time() - started

    results = []
    for row, pred_id, gold_id, ref_ok, ok in zip(eval_rows, pred_ids, gold_ids, base_correct, pred_correct):
        results.append(
            {
                "id": row["id"],
                "target": id2label[gold_id],
                "prediction": id2label[pred_id],
                "reference_correct": ref_ok,
                "correct": ok,
                "improved": (not ref_ok) and ok,
                "degraded": ref_ok and (not ok),
                "chain_len": row.get("chain_len"),
            }
        )

    summary = {
        "dataset": "CLUTRR/v1",
        "config": args.config,
        "method": args.model_name,
        "n_train": len(train_raw),
        "n": len(eval_rows),
        "initial_reference": "left_to_right_archive",
        "initial": initial,
        "final": acc,
        "gain": acc - initial,
        "pgr": pgr,
        "dr": dr,
        "time_per_example_seconds": runtime / max(len(eval_rows), 1),
        "runtime_seconds": runtime,
        "label_counts_train": Counter(normalize_label(row["target_text"]) for row in train_raw),
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["method", "n_train", "n", "initial", "final", "gain", "pgr", "dr", "time_per_example_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
