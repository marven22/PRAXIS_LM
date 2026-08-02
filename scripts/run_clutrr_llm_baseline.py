from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


LABELS = [
    "aunt",
    "brother",
    "brother-in-law",
    "daughter",
    "daughter-in-law",
    "father",
    "father-in-law",
    "granddaughter",
    "grandfather",
    "grandmother",
    "grandson",
    "husband",
    "mother",
    "mother-in-law",
    "nephew",
    "niece",
    "sister",
    "sister-in-law",
    "son",
    "son-in-law",
    "uncle",
    "wife",
]


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = value.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    return " ".join(value.split())


def extract_label(text: str) -> str:
    norm = normalize(text)
    labels = sorted(LABELS, key=len, reverse=True)
    for label in labels:
        if label in norm:
            return label
    return norm.split()[0] if norm.split() else ""


def prompt_for(row: dict, mode: str) -> str:
    labels = ", ".join(LABELS)
    if mode == "chain":
        chain = " -> ".join(row["edge_types"])
        return (
            "Compose the following family-relation chain from left to right and/or by valid intermediate groupings.\n"
            f"Relation chain: {chain}\n"
            "The final answer is the relationship of the first person to the last person.\n"
            f"Choose exactly one label from: {labels}.\n"
            "Answer with only one label."
        )
    if mode == "chain_decompose":
        chain = " -> ".join(row["edge_types"])
        return (
            "You are given a family-relation chain. Infer the final relationship by composing adjacent relations.\n"
            f"Relation chain: {chain}\n"
            f"Choose exactly one label from: {labels}.\n"
            "Show the intermediate relation compositions briefly, then end with 'Final answer: <label>'."
        )
    base = (
        f"Story:\n{row['story']}\n\n"
        f"Query pair: {row['query']}\n"
        "Question: What is the relationship of the first person to the second person?\n"
        f"Choose exactly one label from: {labels}.\n"
    )
    if mode == "decompose":
        return (
            base
            + "Reason step by step through the family chain, then give the final label.\n"
            + "Final answer must be only one label."
        )
    return base + "Answer with only one label."


def load_qwen_vl(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    def generate(text: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.batch_decode(out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0].strip()

    return generate


def load_causal_lm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(text: str, max_new_tokens: int) -> str:
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
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()

    return generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/clutrr/clutrr_test_50.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output", default="analysis_results/clutrr_llm_direct_50.json")
    parser.add_argument("--mode", choices=["direct", "decompose", "chain", "chain_decompose"], default="direct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))[: args.n]
    if "VL" in args.model:
        generate = load_qwen_vl(args.model)
    else:
        generate = load_causal_lm(args.model)

    results = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        raw = generate(prompt_for(row, args.mode), args.max_new_tokens)
        pred = extract_label(raw)
        correct = pred == normalize(row["target"])
        results.append({**row, "raw_prediction": raw, "prediction": pred, "correct": correct})
        running = sum(r["correct"] for r in results) / len(results)
        print(
            f"[{index}/{len(rows)}] len={row['chain_len']} pred={pred!r} gold={row['target']!r} "
            f"correct={correct} running={running:.3f} raw={raw!r}",
            flush=True,
        )

    by_len = defaultdict(list)
    for row in results:
        by_len[row["chain_len"]].append(row["correct"])
    summary = {
        "model": args.model,
        "mode": args.mode,
        "dataset": args.dataset,
        "n": len(results),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "accuracy_by_chain_len": {str(k): sum(v) / len(v) for k, v in sorted(by_len.items())},
        "runtime_sec": time.time() - started,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
