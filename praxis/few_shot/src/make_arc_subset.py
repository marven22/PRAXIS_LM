from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ARC challenge/solution subset files from a list of task ids."
    )
    parser.add_argument("--challenge_file", required=True)
    parser.add_argument("--solution_file", required=True)
    parser.add_argument("--task_ids_file", required=True)
    parser.add_argument("--out_challenge_file", required=True)
    parser.add_argument("--out_solution_file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.challenge_file, "r", encoding="utf-8") as handle:
        challenges = json.load(handle)
    with open(args.solution_file, "r", encoding="utf-8") as handle:
        solutions = json.load(handle)
    with open(args.task_ids_file, "r", encoding="utf-8") as handle:
        task_ids = json.load(handle)

    missing = [task_id for task_id in task_ids if task_id not in challenges or task_id not in solutions]
    if missing:
        raise KeyError(f"Missing task ids in source files: {missing}")

    challenge_subset = {task_id: challenges[task_id] for task_id in task_ids}
    solution_subset = {task_id: solutions[task_id] for task_id in task_ids}

    out_challenge_path = Path(args.out_challenge_file)
    out_solution_path = Path(args.out_solution_file)
    out_challenge_path.parent.mkdir(parents=True, exist_ok=True)
    out_solution_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_challenge_path, "w", encoding="utf-8") as handle:
        json.dump(challenge_subset, handle, indent=2)
    with open(out_solution_path, "w", encoding="utf-8") as handle:
        json.dump(solution_subset, handle, indent=2)

    print(f"Wrote {len(task_ids)} task ids")
    print(f"Challenges -> {out_challenge_path}")
    print(f"Solutions  -> {out_solution_path}")


if __name__ == "__main__":
    main()
