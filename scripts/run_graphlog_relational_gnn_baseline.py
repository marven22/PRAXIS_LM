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
from build_graphlog_binary_subset import graph_text


class RelationalGNN(nn.Module):
    def __init__(self, max_nodes: int, n_relations: int, hidden_dim: int, layers: int, target_relation_id: int):
        super().__init__()
        self.node_embed = nn.Embedding(max_nodes, hidden_dim)
        self.rel_embed = nn.Embedding(n_relations, hidden_dim)
        self.layers = layers
        self.target_relation_id = target_relation_id
        self.self_linear = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.msg_linear = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
        hr = self.rel_embed(torch.tensor(self.target_relation_id, device=h.device))
        return self.out(torch.cat([hs, ht, hr, hs * ht], dim=-1)).squeeze(-1)


def collect_relations(rows: list[dict[str, Any]]) -> list[str]:
    rels = set()
    for row in rows:
        rels.add(row["query"][2])
        rels.update(edge[2] for edge in row["edges"])
    return sorted(rels)


def make_example(row: dict[str, Any], target_relation: str, rel2id: dict[str, int]) -> dict[str, Any]:
    node_ids = sorted({int(src) for src, _, _ in row["edges"]} | {int(dst) for _, dst, _ in row["edges"]} | {int(row["query"][0]), int(row["query"][1])})
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
        "label": 1 if true_relation == target_relation else 0,
        "true_relation": true_relation,
        "target_relation": target_relation,
    }


def recover_eval_rows(binary_dataset: Path, raw_test_rows: list[dict[str, Any]], target_relation: str) -> list[dict[str, Any]]:
    data = json.loads(binary_dataset.read_text(encoding="utf-8"))
    lookup = {graph_text(row, target_relation): row for row in raw_test_rows}
    recovered = []
    missing = []
    for task in data["tasks"]:
        text = task["query"]["text"]
        row = lookup.get(text)
        if row is None:
            missing.append(task["id"])
            continue
        recovered.append({"task": task, "row": row})
    if missing:
        raise RuntimeError(f"Could not recover {len(missing)} eval rows from prompt text. First missing: {missing[:3]}")
    return recovered


def eval_model(model: RelationalGNN, examples: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for ex in examples:
            logit = model(
                torch.tensor(ex["nodes"], dtype=torch.long, device=device),
                torch.tensor(ex["edges"], dtype=torch.long, device=device),
                torch.tensor(ex["rels"], dtype=torch.long, device=device),
                ex["qsrc"],
                ex["qdst"],
            )
            prob = torch.sigmoid(logit).item()
            probs.append(prob)
            preds.append(1 if prob >= 0.5 else 0)
    return preds, probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_dir", default="data/graphlog/graphlog_v1.1/test/rule_56")
    parser.add_argument("--binary_dataset", default="data/graphlog/graphlog_binary_rule56_r7plus_100.json")
    parser.add_argument("--target_relation", default="R_7_+")
    parser.add_argument("--output", default="analysis_results/native_baselines/graphlog_rule56_r7plus_relgnn.json")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world = Path(args.world_dir)
    train_rows = read_jsonl(world / "train.jsonl")[: args.max_train]
    valid_rows = read_jsonl(world / "valid.jsonl")
    test_rows = read_jsonl(world / "test.jsonl")
    all_rows = train_rows + valid_rows + test_rows
    relations = collect_relations(all_rows)
    rel2id = {rel: idx for idx, rel in enumerate(relations)}
    max_nodes = max(max(max(src, dst) for src, dst, _ in row["edges"]) for row in all_rows) + 1

    train_examples = [make_example(row, args.target_relation, rel2id) for row in train_rows]
    valid_examples = [make_example(row, args.target_relation, rel2id) for row in valid_rows[:1000]]
    recovered = recover_eval_rows(Path(args.binary_dataset), test_rows, args.target_relation)
    eval_examples = [make_example(item["row"], args.target_relation, rel2id) for item in recovered]

    model = RelationalGNN(max_nodes=max_nodes, n_relations=len(relations), hidden_dim=args.hidden_dim, layers=args.layers, target_relation_id=rel2id[args.target_relation]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = sum(ex["label"] for ex in train_examples)
    neg = len(train_examples) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)

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
            logit = model(
                torch.tensor(ex["nodes"], dtype=torch.long, device=device),
                torch.tensor(ex["edges"], dtype=torch.long, device=device),
                torch.tensor(ex["rels"], dtype=torch.long, device=device),
                ex["qsrc"],
                ex["qdst"],
            )
            label = torch.tensor(float(ex["label"]), device=device)
            loss = F.binary_cross_entropy_with_logits(logit.view(1), label.view(1), pos_weight=pos_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((torch.sigmoid(logit).item() >= 0.5) == bool(ex["label"]))
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
    initial = 0.5
    baseline_preds = [1] * len(labels)
    pgr = sum((bp != y) and (p == y) for bp, p, y in zip(baseline_preds, preds, labels)) / max(len(labels), 1)
    dr = sum((bp == y) and (p != y) for bp, p, y in zip(baseline_preds, preds, labels)) / max(len(labels), 1)
    runtime = time.time() - started
    results = []
    for item, ex, pred, prob in zip(recovered, eval_examples, preds, probs):
        task = item["task"]
        results.append(
            {
                "id": task["id"],
                "answer": task["query"]["answer"],
                "prediction": "YES" if pred == 1 else "NO",
                "prob_yes": prob,
                "correct": pred == ex["label"],
                "baseline_prediction": "YES",
                "baseline_correct": bool(ex["label"] == 1),
                "improved": (ex["label"] == 0) and (pred == 0),
                "degraded": (ex["label"] == 1) and (pred == 0),
                "metadata": task.get("metadata", {}),
            }
        )

    summary = {
        "dataset": "GraphLog",
        "method": "Relational GNN",
        "world_dir": args.world_dir,
        "binary_dataset": args.binary_dataset,
        "target_relation": args.target_relation,
        "n_train": len(train_examples),
        "n_valid": len(valid_examples),
        "n": len(eval_examples),
        "initial_reference": "always_yes_balanced_binary",
        "initial": initial,
        "final": final,
        "gain": final - initial,
        "pgr": pgr,
        "dr": dr,
        "time_per_example_seconds": runtime / max(len(eval_examples), 1),
        "runtime_seconds": runtime,
        "best_valid_accuracy": best_valid,
        "epochs": epoch_summaries,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["method", "n_train", "n", "initial", "final", "gain", "pgr", "dr", "time_per_example_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
