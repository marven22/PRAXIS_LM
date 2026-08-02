import time
import os
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def get_sampling_params(
    tokenizer: PreTrainedTokenizer,
    num_tokens: int,
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
) -> SamplingParams:
    max_new_tokens = max_tokens - num_tokens
    max_new_tokens_cap = os.environ.get("ARC_MAX_NEW_TOKENS")
    if max_new_tokens_cap:
        max_new_tokens = min(max_new_tokens, int(max_new_tokens_cap))
    max_new_tokens = max(max_new_tokens, 1)
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=n,
        stop=[tokenizer.eos_token, "<|eot_id|>", "<|im_end|>"],
        # best_of=10,
        # use_beam_search=True,
    )


def initialize_engine(
    model: str,
    enforce_eager: bool = False,
    enable_lora: bool = True,
    max_lora_rank: int = 64,
    quantization: Optional[str] = None,
    lora_repo: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
) -> LLMEngine:
    """Initialize the LLMEngine."""

    engine_args = EngineArgs(
        model=model,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        enforce_eager=enforce_eager,
        quantization=quantization,
        #lora_target_modules=lora_target_modules,
        load_format="bitsandbytes" if quantization else "auto",
        max_model_len=8192,
        gpu_memory_utilization=0.9
    )

    return LLMEngine.from_engine_args(engine_args)


def process_requests(
    engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest], str]]
) -> Dict[str, List[str]]:
    """Continuously process a list of prompts and handle the outputs."""
    all_outputs: Dict[str, List[str]] = {}
    total_requests = len(test_prompts)
    added_requests = 0
    finished_requests = 0
    loop_steps = 0
    start_time = time.time()
    last_heartbeat = start_time

    print(
        f"[engine] starting request processing: total_requests={total_requests}",
        flush=True,
    )

    while test_prompts or engine.has_unfinished_requests():
        loop_steps += 1
        if test_prompts:
            prompt, sampling_param, lora_request, idx = test_prompts.pop(0)
            find_start = prompt.find("<|begin_of_text|>") + len("<|begin_of_text|>")
            prompt = prompt[find_start:]
            engine.add_request(idx, prompt, sampling_param, lora_request=lora_request)
            added_requests += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                texts = [output.text for output in request_output.outputs]
                all_outputs[str(request_output.request_id)] = texts
                finished_requests += 1

        now = time.time()
        if now - last_heartbeat >= 30:
            elapsed = now - start_time
            unfinished = "unknown"
            try:
                unfinished = engine.has_unfinished_requests()
            except Exception:
                pass
            print(
                "[engine] heartbeat "
                f"elapsed={elapsed:.1f}s "
                f"added={added_requests}/{total_requests} "
                f"finished={finished_requests}/{total_requests} "
                f"queue_remaining={len(test_prompts)} "
                f"has_unfinished={unfinished} "
                f"loop_steps={loop_steps}",
                flush=True,
            )
            last_heartbeat = now

    elapsed = time.time() - start_time
    print(
        f"[engine] completed request processing in {elapsed:.1f}s: "
        f"finished={finished_requests}/{total_requests}",
        flush=True,
    )

    return all_outputs
