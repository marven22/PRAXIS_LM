from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_command(name: str, command: list[str], output_path: Path | None) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    row = {
        "name": name,
        "command": command,
        "runtime_sec": elapsed,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        return row

    if output_path and output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            row["n"] = data.get("n")
            if "accuracy" in data:
                row["accuracy"] = data["accuracy"]
            if "coverage" in data:
                row["coverage"] = data["coverage"]
            if "runtime_sec" in data:
                row["script_runtime_sec"] = data["runtime_sec"]
            if "methods" in data:
                row["methods"] = [
                    {
                        "method": method.get("method"),
                        "accuracy": method.get("accuracy"),
                        "coverage": method.get("coverage"),
                    }
                    for method in data["methods"]
                ]
    n = row.get("n")
    if isinstance(n, int) and n > 0:
        row["runtime_per_example_sec"] = elapsed / n
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--output", default="analysis_results/clutrr_runtime_calibration_50.json")
    parser.add_argument("--include_llm", action="store_true")
    parser.add_argument("--llm_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--llm_mode", default="chain_decompose")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset_path = Path(f"data/clutrr/clutrr_test_runtime_{args.n}.json")
    praxis_out = Path(f"analysis_results/clutrr_runtime_praxis_symbolic_{args.n}.json")
    baselines_out = Path(f"analysis_results/clutrr_runtime_symbolic_baselines_{args.n}.json")

    py = sys.executable
    commands: list[tuple[str, list[str], Path | None]] = [
        (
            "praxis_dp_archive",
            [
                py,
                "scripts/run_clutrr_praxis_symbolic.py",
                "--n",
                str(args.n),
                "--output",
                str(praxis_out),
                "--subset_output",
                str(subset_path),
            ],
            praxis_out,
        ),
        (
            "symbolic_baselines",
            [
                py,
                "scripts/run_clutrr_symbolic_baselines.py",
                "--n",
                str(args.n),
                "--output",
                str(baselines_out),
            ],
            baselines_out,
        ),
    ]

    if args.include_llm:
        llm_out = Path(f"analysis_results/clutrr_runtime_llm_{args.llm_model.replace('/', '_')}_{args.llm_mode}_{args.n}.json")
        commands.append(
            (
                f"llm_{args.llm_mode}",
                [
                    py,
                    "scripts/run_clutrr_llm_baseline.py",
                    "--dataset",
                    str(subset_path),
                    "--model",
                    args.llm_model,
                    "--output",
                    str(llm_out),
                    "--mode",
                    args.llm_mode,
                    "--n",
                    str(args.n),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                ],
                llm_out,
            )
        )

    rows = []
    total_started = time.perf_counter()
    for name, command, out_path in commands:
        print(f"running {name}", flush=True)
        row = run_command(name, command, out_path)
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k not in {"stdout_tail", "stderr_tail", "command", "methods"}}, indent=2), flush=True)
        if row["returncode"] != 0:
            break

    summary = {
        "dataset": "CLUTRR/v1",
        "n": args.n,
        "hardware_note": "Local runtime calibration; report only with matching hardware/environment.",
        "total_runtime_sec": time.perf_counter() - total_started,
        "rows": rows,
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
