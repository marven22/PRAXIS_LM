from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def majority_label(labels: list[str], default: str = "") -> str:
    if not labels:
        return default
    return Counter(labels).most_common(1)[0][0]


def evaluate(tasks: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    results = []
    correct = 0
    for task in tasks:
        gold = task["query"]["answer"]
        pred = predictions[task["id"]]
        ok = pred == gold
        correct += int(ok)
        results.append(
            {
                "id": task["id"],
                "world": task.get("world"),
                "answer": gold,
                "prediction": pred,
                "correct": ok,
                "descriptor": task["query"].get("descriptor", ""),
            }
        )
    return {
        "n": len(tasks),
        "accuracy": correct / max(len(tasks), 1),
        "results": results,
    }


def build_predictions(tasks: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    support_labels = [item["answer"] for task in tasks for item in task.get("train", [])]
    global_support_majority = majority_label(support_labels, default=tasks[0]["labels"][0] if tasks else "")

    by_descriptor: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for item in task.get("train", []):
            by_descriptor[item.get("descriptor", "")].append(item["answer"])

    predictions = {
        "global_support_majority": {},
        "task_support_majority": {},
        "descriptor_support_prior": {},
        "first_candidate_label": {},
    }
    for task in tasks:
        labels = task.get("labels", [])
        task_support = [item["answer"] for item in task.get("train", [])]
        task_default = majority_label(task_support, default=global_support_majority)
        descriptor = task["query"].get("descriptor", "")
        descriptor_default = majority_label(by_descriptor.get(descriptor, []), default=task_default)

        predictions["global_support_majority"][task["id"]] = global_support_majority
        predictions["task_support_majority"][task["id"]] = task_default
        predictions["descriptor_support_prior"][task["id"]] = descriptor_default
        predictions["first_candidate_label"][task["id"]] = labels[0] if labels else global_support_majority
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = json.loads((REPO_ROOT / args.dataset).read_text(encoding="utf-8"))
    tasks = data["tasks"]
    predictions = build_predictions(tasks)
    baselines = {name: evaluate(tasks, pred) for name, pred in predictions.items()}
    label_counts = Counter(task["query"]["answer"] for task in tasks)
    summary = {
        "dataset": args.dataset,
        "source": data.get("source"),
        "n": len(tasks),
        "label_counts": dict(label_counts),
        "baselines": baselines,
    }

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n": summary["n"],
                "label_counts": summary["label_counts"],
                "accuracies": {name: row["accuracy"] for name, row in baselines.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
