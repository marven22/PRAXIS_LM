from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_clutrr_llm_baseline import normalize
from run_clutrr_seal_edits import eval_prompt, predict_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", default="data/clutrr/clutrr_test_full1048.json")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n_eval", type=int, default=0)
    parser.add_argument("--output", default="analysis_results/full_scale/clutrr_edit_sft_base_full1048.json")
    args = parser.parse_args()

    rows = json.loads(Path(args.eval_dataset).read_text(encoding="utf-8"))
    if args.n_eval > 0:
        rows = rows[: args.n_eval]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    results = []
    by_len = defaultdict(list)
    correct = 0
    for idx, row in enumerate(rows, start=1):
        prompt = eval_prompt(tokenizer, row)
        pred, scores = predict_label(model, tokenizer, prompt)
        answer = normalize(row["target"])
        ok = pred == answer
        correct += int(ok)
        by_len[row["chain_len"]].append(ok)
        results.append(
            {
                "id": row["id"],
                "chain_len": row["chain_len"],
                "chain": row["edge_types"],
                "answer": answer,
                "prediction": pred,
                "correct": ok,
                "scores": scores,
            }
        )
        print(f"[{idx}/{len(rows)}] pred={pred} answer={answer} correct={ok}", flush=True)

    summary = {
        "method": "Qwen base before Edit-SFT",
        "model": args.model_name,
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "accuracy_by_chain_len": {str(k): sum(v) / len(v) for k, v in sorted(by_len.items())},
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
