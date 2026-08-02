from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from praxis.few_shot.src.archive_utils import load_archive, save_archive, update_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update PRAXIS few-shot archive from ARC eval results")
    parser.add_argument("--archive_path", default="praxis/few_shot/archive/strategies.json")
    parser.add_argument("--configs_path", required=True)
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--reward_positive", type=float, default=1.0)
    parser.add_argument("--reward_negative", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = load_archive(args.archive_path)
    configs: Dict[str, Dict[str, Any]] = json.load(open(args.configs_path, "r", encoding="utf-8"))
    results: Dict[str, Dict[str, Any]] = json.load(open(args.results_path, "r", encoding="utf-8"))

    total = 0
    for edit_idx, result in results.items():
        task_ids = result.get("task_id", {})
        correct = result.get("correct", {})
        for local_idx, task_id in task_ids.items():
            if task_id not in configs:
                continue
            task_cfg = configs[task_id].get(str(edit_idx))
            if not task_cfg:
                continue
            reward = args.reward_positive if bool(correct.get(local_idx, False)) else args.reward_negative
            archive = update_archive(archive, task_cfg["strategy_name"], reward=reward, beta=args.beta)
            total += 1

    save_archive(args.archive_path, archive)
    print(f"Updated archive with {total} few-shot outcomes -> {args.archive_path}")


if __name__ == "__main__":
    main()
