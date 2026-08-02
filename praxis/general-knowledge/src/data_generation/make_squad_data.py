# general-knowledge/src/data_generation/make_squad_data.py
"""
Generate synthetic SQuAD-style items by prompting a vLLM endpoint for `k` "implication" completions per passage.
"""
from pathlib import Path
import argparse, json, random, time, datetime, requests
from typing import Any, Dict, List, Optional

from ..archive_utils import canonical_prompt_key, canonical_strategy_name, normalize_archive

MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant tasked with analyzing the provided passage and producing a list of implications derived directly or indirectly from the content. <|im_end|>\n"
    "<|im_start|>user\n{title}\n{context}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

MAKE_SQUAD_DATA_TEMPLATES_BASE: dict[str, str] = {
    # list of implications
    "implications": (
        "Read the passage and produce a concise list of implications derived directly or indirectly from it. "
        "Output only the implication list. Do not include meta-commentary, explanations, or assistant-style prefaces.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # long list of implications
    "implications-long": (
        "Read the passage and produce a longer list of implications derived directly or indirectly from it. "
        "Output only the implication list. Do not include meta-commentary, explanations, or assistant-style prefaces.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # very long list of implications
    "implications-very-long": (
        "Read the passage and produce a very long list of implications derived directly or indirectly from it. "
        "Output only the implication list. Do not include meta-commentary, explanations, or assistant-style prefaces.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # rewrite the passage
    "rewrite": (
        "Read the passage and rewrite it in a few different faithful ways, one per line. "
        "Output only the rewritten passages. Do not include meta-commentary or assistant-style prefaces.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Rewritten passages:\n"
    ),

    # extractive facts
    "extractive-facts": (
        "Read the passage and produce a list of short factual statements that are directly supported by the passage. "
        "Each fact should be atomic, faithful, and grounded in the text. Output only the fact list. "
        "Do not ask questions. Do not include meta-commentary or assistant-style prefaces.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Facts:\n"
    ),

    # grounded exposition
    "grounded-exposition": (
        "Read the passage and write a short factual explanation in 2-5 concise sentences. "
        "Keep it fully grounded in the passage, preserve key entities, dates, causes, and relations, "
        "and make it useful for answering factual questions later. Output only the explanation. "
        "Do not include instructions, opinions, unsupported inferences, or assistant-style prefaces.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Explanation:\n"
    ),

    # hotpotqa-specific: extract supporting facts as atomic bullets
    "supporting-fact-extraction": (
        "Read the passage and extract the smallest set of short factual statements that are directly supported by the text "
        "and most useful for answering a multi-hop question later. Keep each fact atomic, specific, and grounded. "
        "Preserve names, relations, dates, roles, and bridge entities. Output only the fact list.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Supporting Facts:\n"
    ),

    # hotpotqa-specific: explicitly chain bridge entities/facts
    "bridge-entity-chain": (
        "Read the passage and rewrite it as a short bridge chain showing how one fact leads to another. "
        "Focus on the linking entities, attributes, and relations that connect the evidence pieces needed for a multi-hop answer. "
        "Output only a concise numbered chain of grounded statements.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Bridge Chain:\n"
    ),

    # hotpotqa-specific: comparison-ready representation
    "comparison-table": (
        "Read the passage and organize the evidence into a compact comparison of the main entities, people, places, or concepts mentioned. "
        "Highlight the attributes that could distinguish them when answering a comparison question. "
        "Output only a concise comparison list or table-like bullet format grounded in the text.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Comparison:\n"
    ),

    # hotpotqa-specific: select answer-relevant evidence with question awareness
    "question-aware-evidence-selection": (
        "Read the passage and produce only the evidence snippets and short facts that would be most relevant for answering a question about it later. "
        "Prioritize bridge entities, comparisons, roles, dates, and answer-bearing relations. "
        "Output only the selected evidence list with no explanations or meta-commentary.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Selected Evidence:\n"
    ),

    # hotpotqa-specific: explain the multi-hop chain in a compact way
    "multi-hop-explanation": (
        "Read the passage and write a short explanation that explicitly links the key facts needed for a multi-hop answer. "
        "Make the chain of reasoning easy to follow while staying fully grounded in the passage. "
        "Output only the explanation in 2-5 concise sentences.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Multi-hop Explanation:\n"
    ),

    # hotpotqa-specific: concise answer-oriented summary
    "answer-oriented-summary": (
        "Read the passage and write a compact summary that preserves only the facts most useful for answering a likely factual question about it. "
        "Prefer concise, grounded, answer-oriented phrasing over broad background details. "
        "Output only the summary in 2-4 sentences.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Answer-oriented Summary:\n"
    ),

    # hotpotqa-specific: connect the bridge fact to the answer-bearing fact
    "bridge-step-reasoning": (
        "Read the passage and produce a short bridge-first reasoning note for a future multi-hop question. "
        "First identify the bridge entity or relation, then state the answer-bearing fact it points to. "
        "Keep every line grounded in the passage and concise. Output only a short note with 2-4 lines.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Bridge-first Reasoning Note:\n"
    ),

    # hotpotqa-specific: isolate likely answer-bearing facts or values
    "answer-bearing-fact": (
        "Read the passage and isolate the exact fact fragments most likely to contain a future answer. "
        "Preserve names, dates, roles, locations, titles, and short attribute phrases exactly when possible. "
        "Output only a compact list of answer-bearing facts with no extra commentary.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Answer-bearing Facts:\n"
    ),

    # hotpotqa-specific: rehearse the context in a QA-oriented tutoring format
    "multi-hop-qa-rehearsal": (
        "Read the passage and rewrite it as a compact study note for answering a future multi-hop question. "
        "Use this format: Question focus, Supporting fact 1, Supporting fact 2, Combined inference, Likely answer target. "
        "Keep every line grounded in the passage and concise. Output only the study note.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Study Note:\n"
    ),

    # hotpotqa-specific: decompose likely future questions into subgoals
    "question-decomposition": (
        "Read the passage and decompose a likely future multi-hop question into the smallest useful subquestions. "
        "Identify what must be found first, what second fact is needed, and what kind of final answer is likely required. "
        "Ground every subquestion in the passage. Output only a concise decomposition list.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Question Decomposition:\n"
    ),

    "count-and-enumerate": (
        "Answer the stated question by identifying every passage item that satisfies its condition, then count the items. "
        "Produce a compact training note containing the qualifying items, the count, and the final answer. "
        "Do not introduce facts outside the passage.\n\n"
        "{title}\nPassage:\n{context}\n\nCounting Note:\n"
    ),

    "arithmetic-derivation": (
        "Answer the stated question by extracting the relevant numbers and writing the exact arithmetic operation needed. "
        "Produce a compact training note with the operands, operation, calculation, units, and final answer. "
        "Use only numbers supported by the passage.\n\n"
        "{title}\nPassage:\n{context}\n\nArithmetic Note:\n"
    ),

    "comparison-resolution": (
        "Answer the stated question by extracting the compared entities and values, normalizing their units, and resolving the comparison. "
        "Produce a compact training note with both values, the comparison, and the final answer. "
        "Use only evidence from the passage.\n\n"
        "{title}\nPassage:\n{context}\n\nComparison Note:\n"
    ),

    "temporal-event-order": (
        "Answer the stated question by ordering the relevant events, dates, or game periods from the passage. "
        "Produce a compact training note containing the ordered evidence and final answer. "
        "Do not add unsupported chronology.\n\n"
        "{title}\nPassage:\n{context}\n\nTemporal Note:\n"
    ),

    "entity-reference-resolution": (
        "Answer the stated question by resolving its entities and references to the exact supporting passage span. "
        "Produce a compact training note containing the resolved reference, supporting sentence, and final answer. "
        "Stay fully grounded in the passage.\n\n"
        "{title}\nPassage:\n{context}\n\nReference Note:\n"
    ),

    # none
    "none": (
        "Passage:\n{title}\n{context}\n\n"
    ),

    # chain-of-thought
    "implications-chain-of-thought": (
        "Read the passage, think step by step, and then produce a list of implications derived directly or indirectly from it. "
        "After reasoning, output only the final implication list without assistant-style prefaces.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Thought Process:\n"
    ),
}

# ------------------------------------------------------------------------ #

def make_prompt(title: str, context: str, instruct_model: bool, prompt_key: str) -> str:
    MAKE_SQUAD_DATA_TEMPLATE = MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT if instruct_model else MAKE_SQUAD_DATA_TEMPLATES_BASE[prompt_key]
    return MAKE_SQUAD_DATA_TEMPLATE.format(
            title=title,
            context=context,
        )

def load_archive(path: str) -> Dict[str, Any]:
    # Accept UTF-8 archives saved with or without BOM. PowerShell can emit
    # UTF-8-with-BOM depending on version/configuration.
    archive = normalize_archive(json.load(open(path, encoding="utf-8-sig")))
    if "strategies" not in archive or not isinstance(archive["strategies"], list):
        raise ValueError("Archive is missing a 'strategies' list")
    return archive

def select_strategy(archive: Dict[str, Any], rng: random.Random, min_prob: float = 0.0) -> Dict[str, Any]:
    strategies = archive.get("strategies", [])
    weights = [max(0.0, float(s.get("weight", 1.0))) for s in strategies]
    total = sum(weights)
    if total <= 0:
        # fallback to uniform selection
        idx = rng.randrange(len(strategies))
        return strategies[idx]

    # Optional sampling floor: enforce a minimum probability per strategy
    if min_prob > 0.0:
        if min_prob * len(weights) >= 1.0:
            weights = [1.0 for _ in weights]
            total = float(len(weights))
        else:
            norm = [w / total for w in weights]
            floored = [max(w, min_prob) for w in norm]
            floor_total = sum(floored)
            weights = [w / floor_total for w in floored]
            total = 1.0

    pick = rng.random() * total
    acc = 0.0
    for s, w in zip(strategies, weights):
        acc += w
        if pick <= acc:
            return s
    return strategies[-1]

def generate_bulk(
    vllm_api_url: str,
    prompts: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Call vLLM once with a list of prompts.  
    Returns a list of completions in the same order.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompts,
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    r = requests.post(f"{vllm_api_url}/v1/completions", json=payload, timeout=60000)
    r.raise_for_status()
    choices = r.json()["choices"]

    out = ["" for _ in range(len(prompts))]
    for ch in choices:
        idx = ch["index"]
        out[idx] = ch["text"].strip()

    if any(c == "" for c in out):
        raise RuntimeError("Mismatch between returned choices and prompt list")

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Using instruction model")
    p.add_argument("--dataset_in", required=True, help="Path to the input dataset")
    p.add_argument("--dataset_out", required=True, help="Path to the output dataset")
    p.add_argument("--n", type=int, default=-1, help="How many articles to process")
    p.add_argument("--start", type=int, default=0, help="Start index for processing")
    p.add_argument('--k', type=int, default=5, help='Completions per article')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling (top-p)')
    p.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate")
    p.add_argument("--prompt_key", default="implications", choices=list(MAKE_SQUAD_DATA_TEMPLATES_BASE.keys()), help="Which prompt to use")
    p.add_argument("--archive_path", default="", help="Path to a PRAXIS-style strategy archive JSON")
    p.add_argument("--strategy_min_prob", type=float, default=0.0,
                   help="Minimum sampling probability per strategy (0 disables floor)")
    p.add_argument("--archive_seed", type=int, default=42, help="Seed for archive strategy selection")
    args = p.parse_args()

    # -------- load data + build user messages ----------------------- #
    raw: List[Dict[str, Any]] = json.load(open(args.dataset_in, encoding="utf-8"))
    random.seed(42)  # Fixed seed for reproducibility. To sample a different subset, change args.start
    random.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start:]

    archive: Optional[Dict[str, Any]] = None
    archive_rng = random.Random(args.archive_seed)
    strategy_counts: Dict[str, int] = {}
    if args.archive_path:
        archive = load_archive(args.archive_path)

    prompts: List[str] = []
    per_item_meta: List[Dict[str, Any]] = []
    for item in subset:
        if archive:
            strategy_hint = canonical_strategy_name(item.get("strategy_hint"))
            hinted = [
                s for s in archive.get("strategies", [])
                if canonical_strategy_name(s.get("name") or s.get("prompt_key")) == strategy_hint
            ]
            strat = hinted[0] if hinted else select_strategy(
                archive, archive_rng, min_prob=args.strategy_min_prob
            )
            prompt_key = canonical_prompt_key(strat.get("prompt_key") or strat.get("name"))
            if prompt_key not in MAKE_SQUAD_DATA_TEMPLATES_BASE:
                raise ValueError(f"Unknown prompt_key in archive: {prompt_key}")
            strategy_name = canonical_strategy_name(strat.get("name") or prompt_key)
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        else:
            prompt_key = args.prompt_key
            strategy_name = prompt_key

        prompt = make_prompt(
            title=item["title"],
            context=item["context"],
            instruct_model=args.instruct_model,
            prompt_key=prompt_key,
        )
        prompts.extend([prompt] * args.k)
        per_item_meta.append(
            {
                "strategy": strategy_name,
                "prompt_key": prompt_key,
                "prompt": prompt,
            }
        )
    print(f"Requesting {len(prompts)} completions in one batch...")
    t0 = time.time()
    completions = generate_bulk(
        args.vllm_api_url, prompts, args.model, args.max_tokens, args.temperature, args.top_p
    )
    print(f"Received in {time.time()-t0:.1f}s")

    out_data: List[Dict[str, Any]] = []
    for idx, item in enumerate(subset):
        start = idx * args.k
        end = start + args.k
        comp_slice = completions[start:end]
        meta = per_item_meta[idx]

        new_item = dict(item)
        new_item["completions"] = comp_slice
        new_item["prompt"] = meta["prompt"]
        new_item["prompt_key"] = meta["prompt_key"]
        new_item["strategy"] = meta["strategy"]
        out_data.append(new_item)

    out_path = Path(args.dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}  ({len(out_data)} records)")

    # ---------- write hyperparam manifest -------------------- #
    meta = {
        "model": args.model,
        "dataset_in": args.dataset_in,
        "dataset_out": args.dataset_out,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": len(subset),
        "k": args.k,
        "prompt_key": args.prompt_key,
        "archive_path": args.archive_path or None,
        "strategy_counts": strategy_counts,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"meta → {meta_path}")


if __name__ == "__main__":
    main()
