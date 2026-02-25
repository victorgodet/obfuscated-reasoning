"""Merge a LoRA checkpoint into the base model and optionally push to HuggingFace.

Usage:
    python scripts/merge_and_push.py \
        --checkpoint_path results/runs/qwen3-4b/.../checkpoints/global_step_50 \
        --output_path results/merged/qwen3-4b-math-alnum-step50 \
        --push_to_hub your-username/qwen3-4b-math-alnum-step50

    # Just merge locally (no push):
    python scripts/merge_and_push.py \
        --checkpoint_path results/runs/qwen3-4b/.../checkpoints/global_step_50
"""

import fire
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(
    checkpoint_path: str,
    base_model_id: str = "Qwen/Qwen3-4B",
    output_path: str | None = None,
    push_to_hub: str | None = None,
    dtype: str = "bfloat16",
):
    """Merge LoRA adapter into base model and save/push."""

    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    torch_dtype = getattr(torch, dtype)
    if output_path is None:
        output_path = checkpoint_path + "_merged"

    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        print(f"Done! Model available at: https://huggingface.co/{push_to_hub}")
    else:
        print(f"Done! Merged model saved at: {output_path}")
        print("To push to HF, re-run with --push_to_hub=your-username/model-name")


if __name__ == "__main__":
    fire.Fire({"merge": merge})
