from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return 100.0 * value


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="analysis_results/full_scale/ncrl_clutrr_len4_seed_sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--path_len", type=int, default=4)
    parser.add_argument("--output", default="analysis_results/full_scale/clutrr_ncrl_len4_seed_sweep_summary.json")
    args = parser.parse_args()

    root = REPO_ROOT / args.results_dir
    runs = []
    missing = []
    for seed in args.seeds:
        path = root / f"clutrr_ncrl_len{args.path_len}_seed{seed}.json"
        if not path.exists():
            missing.append({"seed": seed, "path": str(path)})
            continue
        data = load(path)
        runs.append(
            {
                "seed": seed,
                "n": data["n"],
                "initial": pct(data["initial"]),
                "final": pct(data["final"]),
                "gain": pct(data["gain"]),
                "pgr": pct(data["pgr"]),
                "dr": pct(data["dr"]),
                "time_per_example_seconds": data["time_per_example_seconds"],
                "runtime_seconds": data["runtime_seconds"],
            }
        )

    metrics = ["initial", "final", "gain", "pgr", "dr", "time_per_example_seconds", "runtime_seconds"]
    aggregate = {
        metric: summarize([run[metric] for run in runs])
        for metric in metrics
    }
    summary = {
        "dataset": "CLUTRR/v1",
        "method": f"NCRL observed-path len{args.path_len}",
        "seeds": args.seeds,
        "completed": [run["seed"] for run in runs],
        "missing": missing,
        "runs": runs,
        "aggregate": aggregate,
    }
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
