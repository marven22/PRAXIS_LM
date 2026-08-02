from __future__ import annotations

import argparse
import ast
import json
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_clutrr_llm_baseline import LABELS
from run_clutrr_praxis_symbolic import parse_edge_types, predict_relation


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = value.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    return " ".join(value.split())


def build_pair_votes(train_rows) -> dict[tuple[str, str], Counter[str]]:
    votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        edges = parse_edge_types(row["edge_types"])
        if len(edges) == 2:
            votes[(edges[0], edges[1])][normalize(row["target_text"])] += 1
        proof_state = row.get("proof_state")
        if not proof_state:
            continue
        try:
            proof_steps = ast.literal_eval(proof_state) if isinstance(proof_state, str) else proof_state
        except (SyntaxError, ValueError):
            proof_steps = []
        for proof in proof_steps:
            if not isinstance(proof, dict):
                continue
            for conclusion, premises in proof.items():
                if not isinstance(conclusion, tuple) or len(conclusion) < 3:
                    continue
                if not isinstance(premises, list) or len(premises) != 2:
                    continue
                left, right = premises
                if len(left) >= 3 and len(right) >= 3:
                    votes[(str(left[1]), str(right[1]))][normalize(str(conclusion[1]))] += 1
    return votes


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text + "\nAnswer:"


def score_sequence(model, tokenizer, prompt: str, label: str) -> float:
    continuation = " " + label
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=False)["input_ids"]
    label_ids = full_ids[len(prompt_ids) :]
    inputs = torch.tensor([full_ids], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=inputs).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    score = 0.0
    start = len(prompt_ids)
    for offset, token_id in enumerate(label_ids):
        score += float(log_probs[start + offset - 1, token_id].detach().cpu())
    return score / max(len(label_ids), 1)


def propose_rule(model, tokenizer, left: str, right: str, counts: Counter[str], max_examples: int) -> tuple[str, dict[str, float], str]:
    examples = []
    for label, count in counts.most_common(max_examples):
        examples.append(f"Observed pair: {left} -> {right}. Final relation: {label}. Count: {count}.")
    prompt_text = (
        "Infer one executable CLUTRR archive rule from support evidence.\n"
        "The rule composes two adjacent kinship relations into one kinship relation.\n"
        "Return the final relation label only.\n\n"
        + "\n".join(examples)
        + "\n\n"
        f"Candidate labels: {', '.join(LABELS)}.\n"
        f"Archive rule: {left} + {right} =>"
    )
    prompt = format_prompt(tokenizer, prompt_text)
    scores = {label: score_sequence(model, tokenizer, prompt, label) for label in LABELS}
    pred = max(scores, key=scores.get)
    return pred, scores, prompt_text


def make_eval_rows(praxis_result: Path, n: int) -> list[dict]:
    rows = json.loads(praxis_result.read_text(encoding="utf-8"))["results"]
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="gen_train234_test2to10")
    parser.add_argument("--praxis_result", default="analysis_results/clutrr_praxis_symbolic_200.json")
    parser.add_argument("--reference_result", default="analysis_results/clutrr_symbolic_baselines_200.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max_rule_examples", type=int, default=3)
    args = parser.parse_args()

    train = list(load_dataset("CLUTRR/v1", args.config, split="train"))
    pair_votes = build_pair_votes(train)
    eval_rows = make_eval_rows(Path(args.praxis_result), args.n)

    reference = json.loads(Path(args.reference_result).read_text(encoding="utf-8"))
    methods_raw = reference.get("methods", reference)
    methods = {item["method"]: item for item in methods_raw} if isinstance(methods_raw, list) else methods_raw
    ref_results = methods["left_to_right_archive"]["results"][: len(eval_rows)]
    reference_correct = [bool(row["correct"]) for row in ref_results]

    model, tokenizer = load_model(args.model)
    started = time.time()
    proposed_archive = {}
    rule_records = []
    for idx, ((left, right), counts) in enumerate(sorted(pair_votes.items()), start=1):
        pred, scores, prompt = propose_rule(model, tokenizer, left, right, counts, args.max_rule_examples)
        gold = counts.most_common(1)[0][0]
        accepted = pred == gold
        if accepted:
            proposed_archive[(left, right)] = pred
        rule_records.append(
            {
                "left": left,
                "right": right,
                "prediction": pred,
                "gold_majority": gold,
                "accepted": accepted,
                "support_counts": dict(counts),
                "scores": scores,
                "prompt": prompt,
            }
        )
        print(f"[rule {idx}/{len(pair_votes)}] {left}+{right} pred={pred} gold={gold} accepted={accepted}", flush=True)

    results = []
    for index, row in enumerate(eval_rows, start=1):
        pred, trace = predict_relation(row["edge_types"], proposed_archive)
        target = normalize(row["target"])
        correct = pred == target
        ref_ok = reference_correct[index - 1]
        results.append(
            {
                "id": row["id"],
                "chain": row["edge_types"],
                "chain_len": row["chain_len"],
                "target": target,
                "prediction": pred,
                "correct": correct,
                "reference_correct": ref_ok,
                "improved": correct and not ref_ok,
                "degraded": ref_ok and not correct,
                "trace": trace,
            }
        )

    initial = sum(reference_correct) / max(len(reference_correct), 1)
    final = sum(row["correct"] for row in results) / max(len(results), 1)
    elapsed = time.time() - started
    summary = {
        "dataset": "CLUTRR/v1",
        "protocol": "llm_rule_proposer_validated_archive_recomposition",
        "model": args.model,
        "n": len(results),
        "n_train": len(train),
        "candidate_rule_count": len(pair_votes),
        "accepted_rule_count": len(proposed_archive),
        "initial_reference": "left_to_right_archive",
        "initial": initial,
        "final": final,
        "gain": final - initial,
        "pgr": sum(row["improved"] for row in results) / max(len(results), 1),
        "dr": sum(row["degraded"] for row in results) / max(len(results), 1),
        "runtime_seconds": elapsed,
        "time_per_example_seconds": elapsed / max(len(results), 1),
        "archive": {f"{left}+{right}": value for (left, right), value in sorted(proposed_archive.items())},
        "rules": rule_records,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["model", "candidate_rule_count", "accepted_rule_count", "initial", "final", "gain", "pgr", "dr", "time_per_example_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
