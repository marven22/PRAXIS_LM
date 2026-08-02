from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from build_graphlog_subset import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


class MulticlassRelationalGNN(nn.Module):
    def __init__(self, max_nodes: int, n_relations: int, hidden_dim: int, layers: int):
        super().__init__()
        self.node_embed = nn.Embedding(max_nodes, hidden_dim)
        self.rel_embed = nn.Embedding(n_relations, hidden_dim)
        self.layers = layers
        self.self_linear = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.msg_linear = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_relations),
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, rels: torch.Tensor, qsrc: int, qdst: int) -> torch.Tensor:
        h = self.node_embed(nodes)
        if edges.numel() > 0:
            src = edges[:, 0]
            dst = edges[:, 1]
            rel_h = self.rel_embed(rels)
        for layer in range(self.layers):
            agg = torch.zeros_like(h)
            if edges.numel() > 0:
                msg = h[src] + rel_h
                agg.index_add_(0, dst, msg)
            h = F.relu(self.self_linear[layer](h) + self.msg_linear[layer](agg))
        hs = h[qsrc]
        ht = h[qdst]
        return self.out(torch.cat([hs, ht, hs * ht], dim=-1))


def collect_relations(rows: list[dict[str, Any]]) -> list[str]:
    rels = set()
    for row in rows:
        rels.add(row["query"][2])
        rels.update(edge[2] for edge in row["edges"])
    return sorted(rels)


def make_example(row: dict[str, Any], rel2id: dict[str, int]) -> dict[str, Any]:
    node_ids = sorted(
        {int(src) for src, _, _ in row["edges"]}
        | {int(dst) for _, dst, _ in row["edges"]}
        | {int(row["query"][0]), int(row["query"][1])}
    )
    node_map = {node: idx for idx, node in enumerate(node_ids)}
    edges = [[node_map[int(src)], node_map[int(dst)]] for src, dst, _ in row["edges"]]
    rels = [rel2id[rel] for _, _, rel in row["edges"]]
    qsrc, qdst, true_relation = row["query"]
    return {
        "nodes": list(range(len(node_ids))),
        "edges": edges,
        "rels": rels,
        "qsrc": node_map[int(qsrc)],
        "qdst": node_map[int(qdst)],
        "label": rel2id[true_relation],
        "true_relation": true_relation,
    }


def task_lookup_key(task: dict[str, Any]) -> tuple[tuple[int, int], str]:
    return tuple(task["query"]["query"]), task["query"].get("descriptor", "")


def recover_eval_rows(dataset: Path, raw_test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data = json.loads(dataset.read_text(encoding="utf-8"))
    lookup: dict[tuple[tuple[int, int], str], dict[str, Any]] = {}
    for row in raw_test_rows:
        qsrc, qdst, _ = row["query"]
        lookup[((qsrc, qdst), row.get("descriptor", ""))] = row
    recovered = []
    missing = []
    for task in data["tasks"]:
        row = lookup.get(task_lookup_key(task))
        if row is None:
            missing.append(task["id"])
            continue
        recovered.append({"task": task, "row": row})
    if missing:
        raise RuntimeError(f"Could not recover {len(missing)} eval rows. First missing: {missing[:3]}")
    return recovered


def eval_model(model: MulticlassRelationalGNN, examples: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[list[float]]]:
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for ex in examples:
            logits = model(
                torch.tensor(ex["nodes"], dtype=torch.long, device=device),
                torch.tensor(ex["edges"], dtype=torch.long, device=device),
                torch.tensor(ex["rels"], dtype=torch.long, device=device),
                ex["qsrc"],
                ex["qdst"],
            )
            prob = torch.softmax(logits, dim=-1)
            preds.append(int(torch.argmax(prob).detach().cpu()))
            probs.append([float(x) for x in prob.detach().cpu()])
    return preds, probs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_dir", default="data/graphlog/graphlog_v1.1/test/rule_56")
    parser.add_argument("--dataset", default="data/graphlog/graphlog_rule56_full1000.json")
    parser.add_argument("--output", default="analysis_results/full_scale/graphlog_multiclass_relgnn_seed42.json")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world = REPO_ROOT / args.world_dir
    train_rows = read_jsonl(world / "train.jsonl")[: args.max_train]
    valid_rows = read_jsonl(world / "valid.jsonl")
    test_rows = read_jsonl(world / "test.jsonl")
    all_rows = train_rows + valid_rows + test_rows
    relations = collect_relations(all_rows)
    rel2id = {rel: idx for idx, rel in enumerate(relations)}
    id2rel = {idx: rel for rel, idx in rel2id.items()}
    max_nodes = max(max(max(src, dst) for src, dst, _ in row["edges"]) for row in all_rows) + 1

    train_examples = [make_example(row, rel2id) for row in train_rows]
    valid_examples = [make_example(row, rel2id) for row in valid_rows[:1000]]
    recovered = recover_eval_rows(REPO_ROOT / args.dataset, test_rows)
    eval_examples = [make_example(item["row"], rel2id) for item in recovered]

    model = MulticlassRelationalGNN(max_nodes=max_nodes, n_relations=len(relations), hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_valid = -1.0
    epoch_summaries = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_examples)
        model.train()
        losses = []
        correct = 0
        for ex in train_examples:
            opt.zero_grad(set_to_none=True)
            logits = model(
                torch.tensor(ex["nodes"], dtype=torch.long, device=device),
                torch.tensor(ex["edges"], dtype=torch.long, device=device),
                torch.tensor(ex["rels"], dtype=torch.long, device=device),
                ex["qsrc"],
                ex["qdst"],
            )
            label = torch.tensor(ex["label"], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits.view(1, -1), label.view(1))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            correct += int(int(torch.argmax(logits).detach().cpu()) == ex["label"])
        valid_preds, _ = eval_model(model, valid_examples, device)
        valid_acc = sum(int(p == ex["label"]) for p, ex in zip(valid_preds, valid_examples)) / max(len(valid_examples), 1)
        train_acc = correct / max(len(train_examples), 1)
        epoch_summaries.append({"epoch": epoch, "loss": sum(losses) / len(losses), "train_accuracy": train_acc, "valid_accuracy": valid_acc})
        print(f"epoch={epoch} loss={epoch_summaries[-1]['loss']:.4f} train={train_acc:.3f} valid={valid_acc:.3f}", flush=True)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    preds, probs = eval_model(model, eval_examples, device)
    labels = [ex["label"] for ex in eval_examples]
    final = sum(int(p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
    reference_pred = max(set(labels), key=labels.count)
    initial = sum(int(reference_pred == y) for y in labels) / max(len(labels), 1)
    pgr = sum((reference_pred != y) and (p == y) for p, y in zip(preds, labels)) / max(len(labels), 1)
    dr = sum((reference_pred == y) and (p != y) for p, y in zip(preds, labels)) / max(len(labels), 1)
    runtime = time.time() - started

    rows = []
    for item, ex, pred, prob in zip(recovered, eval_examples, preds, probs):
        task = item["task"]
        rows.append(
            {
                "id": task["id"],
                "answer": task["query"]["answer"],
                "prediction": id2rel[pred],
                "correct": pred == ex["label"],
                "reference_prediction": id2rel[reference_pred],
                "reference_correct": reference_pred == ex["label"],
                "improved": (reference_pred != ex["label"]) and (pred == ex["label"]),
                "degraded": (reference_pred == ex["label"]) and (pred != ex["label"]),
                "probabilities": {id2rel[idx]: value for idx, value in enumerate(prob)},
                "metadata": task.get("metadata", {}),
            }
        )

    summary = {
        "dataset": "GraphLog",
        "task": "multiclass_relation_prediction",
        "method": "Relational GNN",
        "seed": args.seed,
        "n_train": len(train_examples),
        "n_valid": len(valid_examples),
        "n": len(eval_examples),
        "labels": relations,
        "initial_reference": "global_majority_label",
        "initial": initial,
        "final": final,
        "gain": final - initial,
        "pgr": pgr,
        "dr": dr,
        "time_per_example_seconds": runtime / max(len(eval_examples), 1),
        "runtime_seconds": runtime,
        "best_valid_accuracy": best_valid,
        "epochs": epoch_summaries,
        "results": rows,
    }
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["method", "seed", "n_train", "n", "initial", "final", "gain", "pgr", "dr", "time_per_example_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
