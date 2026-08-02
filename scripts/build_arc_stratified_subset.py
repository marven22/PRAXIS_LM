import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return rows, cols


def transpose(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    return [list(col) for col in zip(*grid)]


def is_grid(obj: Any) -> bool:
    return (
        isinstance(obj, list)
        and (not obj or isinstance(obj[0], list))
        and all(isinstance(row, list) for row in obj)
        and all(all(isinstance(cell, int) for cell in row) for row in obj)
    )


def iter_solution_grids(solution: Any) -> list[list[list[int]]]:
    if is_grid(solution):
        return [solution]
    if isinstance(solution, list):
        out: list[list[list[int]]] = []
        for item in solution:
            out.extend(iter_solution_grids(item))
        return out
    return []


def symmetry_score(grid: list[list[int]]) -> float:
    if not grid:
        return 1.0
    h = 1.0 if grid == list(reversed(grid)) else 0.0
    t = transpose(grid)
    v = 1.0 if t == list(reversed(t)) else 0.0
    return 0.5 * (h + v)


def collect_task_features(task_id: str, task: dict[str, Any], solution: Any) -> dict[str, Any]:
    colors: set[int] = set()
    max_side = 0
    total_cells = 0
    size_preserving = True
    symmetry_scores: list[float] = []

    train_examples = task.get("train", [])
    test_examples = task.get("test", [])

    for ex in train_examples:
        input_grid = ex["input"]
        output_grid = ex["output"]
        in_shape = grid_shape(input_grid)
        out_shape = grid_shape(output_grid)
        max_side = max(max_side, *in_shape, *out_shape)
        total_cells += in_shape[0] * in_shape[1] + out_shape[0] * out_shape[1]
        if in_shape != out_shape:
            size_preserving = False
        for row in input_grid + output_grid:
            colors.update(row)
        symmetry_scores.append(symmetry_score(input_grid))
        symmetry_scores.append(symmetry_score(output_grid))

    for ex in test_examples:
        input_grid = ex["input"]
        in_shape = grid_shape(input_grid)
        max_side = max(max_side, *in_shape)
        total_cells += in_shape[0] * in_shape[1]
        for row in input_grid:
            colors.update(row)
        symmetry_scores.append(symmetry_score(input_grid))

    # ARC solution JSON stores a list of outputs for test cases.
    for candidate in iter_solution_grids(solution):
        out_shape = grid_shape(candidate)
        max_side = max(max_side, *out_shape)
        total_cells += out_shape[0] * out_shape[1]
        for row in candidate:
            colors.update(row)
        symmetry_scores.append(symmetry_score(candidate))

    avg_symmetry = sum(symmetry_scores) / len(symmetry_scores) if symmetry_scores else 0.0

    return {
        "task_id": task_id,
        "n_train_examples": len(train_examples),
        "n_test_examples": len(test_examples),
        "max_side": max_side,
        "total_cells": total_cells,
        "distinct_colors": len(colors),
        "size_preserving": size_preserving,
        "avg_symmetry": round(avg_symmetry, 4),
    }


def size_bucket(max_side: int) -> str:
    if max_side <= 10:
        return "small"
    if max_side <= 18:
        return "medium"
    return "large"


def color_bucket(n_colors: int) -> str:
    if n_colors <= 3:
        return "low"
    if n_colors <= 5:
        return "medium"
    return "high"


def symmetry_bucket(avg_symmetry: float) -> str:
    if avg_symmetry >= 0.75:
        return "high"
    if avg_symmetry >= 0.25:
        return "medium"
    return "low"


def allocate_counts(group_sizes: dict[str, int], total_target: int) -> dict[str, int]:
    if total_target <= 0:
        return {k: 0 for k in group_sizes}

    non_empty = {k: v for k, v in group_sizes.items() if v > 0}
    if not non_empty:
        return {k: 0 for k in group_sizes}

    allocations = {k: 0 for k in group_sizes}

    # Give each non-empty bucket one slot first when possible.
    initial = min(total_target, len(non_empty))
    for key in sorted(non_empty)[:initial]:
        allocations[key] += 1

    remaining = total_target - initial
    if remaining <= 0:
        return allocations

    remaining_capacity = {
        k: max(0, size - allocations[k]) for k, size in group_sizes.items() if size - allocations[k] > 0
    }
    if not remaining_capacity:
        return allocations

    total_capacity = sum(remaining_capacity.values())
    quotas = {
        k: remaining * (cap / total_capacity) for k, cap in remaining_capacity.items()
    }

    floors = {k: min(remaining_capacity[k], int(math.floor(q))) for k, q in quotas.items()}
    for key, count in floors.items():
        allocations[key] += count

    used = sum(floors.values())
    still_remaining = remaining - used

    if still_remaining > 0:
        remainders = sorted(
            (
                quotas[k] - floors[k],
                remaining_capacity[k] - floors[k],
                k,
            )
            for k in remaining_capacity
        )
        for _, extra_capacity, key in reversed(remainders):
            if still_remaining <= 0:
                break
            if extra_capacity <= 0:
                continue
            allocations[key] += 1
            still_remaining -= 1

    return allocations


def sample_tasks_by_bucket(
    task_features: list[dict[str, Any]],
    total_tasks: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feat in task_features:
        bucket_key = feat["selection_bucket"]
        grouped[bucket_key].append(feat)

    for items in grouped.values():
        items.sort(key=lambda x: x["task_id"])

    allocations = allocate_counts({k: len(v) for k, v in grouped.items()}, total_tasks)

    rng = random.Random(seed)
    selected_ids: list[str] = []
    bucket_samples: dict[str, list[str]] = {}

    for bucket_key in sorted(grouped):
        items = grouped[bucket_key][:]
        rng.shuffle(items)
        chosen = sorted(x["task_id"] for x in items[: allocations[bucket_key]])
        selected_ids.extend(chosen)
        bucket_samples[bucket_key] = chosen

    selected_ids = sorted(selected_ids)
    return selected_ids, {
        "bucket_allocations": allocations,
        "bucket_samples": bucket_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified ARC subset with saved bucket metadata.")
    parser.add_argument("--challenge_file", required=True)
    parser.add_argument("--solution_file", required=True)
    parser.add_argument("--out_challenge_file", required=True)
    parser.add_argument("--out_solution_file", required=True)
    parser.add_argument("--out_meta_file", required=True)
    parser.add_argument("--n_tasks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    challenge_path = Path(args.challenge_file)
    solution_path = Path(args.solution_file)
    out_challenge_path = Path(args.out_challenge_file)
    out_solution_path = Path(args.out_solution_file)
    out_meta_path = Path(args.out_meta_file)

    challenges = load_json(challenge_path)
    solutions = load_json(solution_path)

    shared_ids = sorted(set(challenges) & set(solutions))
    if args.n_tasks > len(shared_ids):
        raise ValueError(f"Requested {args.n_tasks} tasks, but only {len(shared_ids)} shared tasks exist.")

    features: list[dict[str, Any]] = []
    for task_id in shared_ids:
        feat = collect_task_features(task_id, challenges[task_id], solutions[task_id])
        feat["size_bucket"] = size_bucket(feat["max_side"])
        feat["color_bucket"] = color_bucket(feat["distinct_colors"])
        feat["symmetry_bucket"] = symmetry_bucket(feat["avg_symmetry"])
        feat["selection_bucket"] = "|".join(
            [
                feat["size_bucket"],
                feat["color_bucket"],
                "size_preserving" if feat["size_preserving"] else "size_changing",
            ]
        )
        features.append(feat)

    selected_ids, selection_meta = sample_tasks_by_bucket(features, args.n_tasks, args.seed)

    out_challenge_path.parent.mkdir(parents=True, exist_ok=True)
    out_solution_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    selected_challenges = {task_id: challenges[task_id] for task_id in selected_ids}
    selected_solutions = {task_id: solutions[task_id] for task_id in selected_ids}

    out_challenge_path.write_text(json.dumps(selected_challenges, indent=2), encoding="utf-8")
    out_solution_path.write_text(json.dumps(selected_solutions, indent=2), encoding="utf-8")

    feature_map = {feat["task_id"]: feat for feat in features}
    meta = {
        "source_challenge_file": str(challenge_path).replace("\\", "/"),
        "source_solution_file": str(solution_path).replace("\\", "/"),
        "n_tasks": args.n_tasks,
        "seed": args.seed,
        "selection_strategy": "stratified_by(size_bucket,color_bucket,size_preserving)",
        "bucket_definitions": {
            "size_bucket": {"small": "<=10", "medium": "11-18", "large": ">18"},
            "color_bucket": {"low": "<=3", "medium": "4-5", "high": ">=6"},
            "size_preserving": {"true": "all train input/output shapes match", "false": "at least one train pair changes shape"},
            "symmetry_bucket": {"high": ">=0.75", "medium": "0.25-0.75", "low": "<0.25"},
        },
        "selected_task_ids": selected_ids,
        "selection_meta": selection_meta,
        "selected_task_features": {task_id: feature_map[task_id] for task_id in selected_ids},
        "population_bucket_counts": {
            bucket: sum(1 for feat in features if feat["selection_bucket"] == bucket)
            for bucket in sorted({feat["selection_bucket"] for feat in features})
        },
    }
    out_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(selected_ids)} tasks to {out_challenge_path}")
    print(f"Wrote {len(selected_ids)} solutions to {out_solution_path}")
    print(f"Wrote metadata to {out_meta_path}")
    print("Selected task ids:")
    for task_id in selected_ids:
        print(task_id)


if __name__ == "__main__":
    main()
