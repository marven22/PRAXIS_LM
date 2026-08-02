from __future__ import annotations

import argparse
import json
import string
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_clutrr_llm_baseline import LABELS


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = value.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    return " ".join(value.split())


def extract_label(text: str) -> str:
    norm = normalize(text)
    for label in sorted(LABELS, key=len, reverse=True):
        if label in norm:
            return label
    return norm.split()[0] if norm.split() else ""


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


def archive_rules_text(archive: dict[str, str]) -> str:
    rules = []
    for key, value in sorted(archive.items()):
        left, right = key.split("+", 1)
        rules.append(f"{left} + {right} => {value}")
    return "\n".join(rules)


def prompt_for(row: dict, archive: dict[str, str]) -> str:
    labels = ", ".join(LABELS)
    chain = " -> ".join(row["edge_types"])
    return (
        "You are solving a CLUTRR kinship relation task using a fixed symbolic archive.\n"
        "Each archive rule composes two adjacent relations into one relation.\n"
        "Use only the archive rules when composing the relation chain. If several groupings are possible, "
        "choose the grouping that yields a valid final kinship label.\n\n"
        "Archive rules:\n"
        f"{archive_rules_text(archive)}\n\n"
        f"Relation chain: {chain}\n"
        f"Candidate labels: {labels}\n"
        "Return exactly one final kinship label and no other text."
    )


def generate(model, tokenizer, text: str, max_new_tokens: int) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = text + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


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


def predict_label(model, tokenizer, text: str) -> tuple[str, dict[str, float]]:
    prompt = format_prompt(tokenizer, text)
    scores = {label: score_sequence(model, tokenizer, prompt, label) for label in LABELS}
    return max(scores, key=scores.get), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--praxis_result", default="analysis_results/clutrr_praxis_symbolic_200.json")
    parser.add_argument("--reference_result", default="analysis_results/clutrr_symbolic_baselines_200.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--score_method", choices=["logprob", "generate"], default="logprob")
    args = parser.parse_args()

    praxis = json.loads(Path(args.praxis_result).read_text(encoding="utf-8"))
    rows = praxis["results"][: args.n]
    archive = praxis["archive"]

    reference = json.loads(Path(args.reference_result).read_text(encoding="utf-8"))
    methods_raw = reference.get("methods", reference)
    if isinstance(methods_raw, list):
        methods = {item["method"]: item for item in methods_raw}
    else:
        methods = methods_raw
    left_to_right = methods["left_to_right_archive"]["results"][: len(rows)]
    reference_correct = [bool(row["correct"]) for row in left_to_right]

    model, tokenizer = load_model(args.model)
    started = time.time()
    results = []
    for index, row in enumerate(rows, start=1):
        prompt = prompt_for(row, archive)
        if args.score_method == "logprob":
            pred, scores = predict_label(model, tokenizer, prompt)
            raw = f"logprob {pred}"
        else:
            raw = generate(model, tokenizer, prompt, args.max_new_tokens)
            pred = extract_label(raw)
            scores = {}
        target = normalize(row["target"])
        correct = pred == target
        ref_ok = reference_correct[index - 1]
        results.append(
            {
                "id": row["id"],
                "chain": row["edge_types"],
                "chain_len": row["chain_len"],
                "target": target,
                "raw_prediction": raw,
                "prediction": pred,
                "scores": scores,
                "reference_correct": ref_ok,
                "correct": correct,
                "improved": correct and not ref_ok,
                "degraded": ref_ok and not correct,
            }
        )
        running = sum(item["correct"] for item in results) / len(results)
        print(
            f"[{index}/{len(rows)}] len={row['chain_len']} pred={pred!r} "
            f"gold={target!r} correct={correct} running={running:.3f}",
            flush=True,
        )

    by_len: dict[int, list[bool]] = defaultdict(list)
    for row in results:
        by_len[row["chain_len"]].append(row["correct"])
    initial = sum(reference_correct) / max(len(reference_correct), 1)
    final = sum(row["correct"] for row in results) / max(len(results), 1)
    summary = {
        "dataset": "CLUTRR/v1",
        "protocol": "archive_conditioned_llm",
        "model": args.model,
        "n": len(results),
        "archive_source": args.praxis_result,
        "archive_size": len(archive),
        "initial_reference": "left_to_right_archive",
        "initial": initial,
        "final": final,
        "gain": final - initial,
        "pgr": sum(row["improved"] for row in results) / max(len(results), 1),
        "dr": sum(row["degraded"] for row in results) / max(len(results), 1),
        "accuracy_by_chain_len": {str(k): sum(v) / len(v) for k, v in sorted(by_len.items())},
        "runtime_seconds": time.time() - started,
        "time_per_example_seconds": (time.time() - started) / max(len(results), 1),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["model", "n", "archive_size", "initial", "final", "gain", "pgr", "dr", "time_per_example_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
