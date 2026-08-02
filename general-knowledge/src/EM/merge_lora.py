"""
Merge a PEFT LoRA adapter into a base model and save the merged model.

Example:
python general-knowledge/src/EM/merge_lora.py \
  --base_model Qwen/Qwen2.5-3B \
  --adapter_dir general-knowledge/results/sft/praxis_sft_k3_t0p5/checkpoint-4 \
  --out_dir general-knowledge/results/sft/praxis_sft_k3_t0p5_merged
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True, help="HF base model (e.g., Qwen/Qwen2.5-3B)")
    p.add_argument("--adapter_dir", required=True, help="Path to LoRA adapter directory (contains adapter_config.json)")
    p.add_argument("--out_dir", required=True, help="Output directory for merged model")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(args.out_dir)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.save_pretrained(args.out_dir)

    print(f"Merged model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
