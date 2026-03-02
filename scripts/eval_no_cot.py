"""Evaluate a model with forced empty think block (no CoT).

Prefills the assistant response with <think>\n\n</think>\n\n so the model
can only generate the final answer, with no chain-of-thought reasoning.
Uses transformers directly (no vLLM) since we only sample ~30 tokens per question.

Usage:
    uv run python scripts/eval_no_cot.py run \
        --model_id=Qwen/Qwen3-4B \
        --dataset_path=results/data/prime_multiplication_k3_eval.jsonl

    uv run python scripts/eval_no_cot.py run \
        --model_id=vg10101/qwen3-4b-prime-mult-k5-step50 \
        --dataset_path=results/data/prime_multiplication_k3_eval.jsonl
"""

import fire
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import RESULTS_PATH, utils


PREFILL = "<think>\n\n</think>\n\n$$\n\\boxed{"
SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."


def parse_boxed_answer(response: str) -> int | None:
    # Since prefill includes \boxed{, the response starts with just digits
    # Try extracting leading digits first
    match = re.match(r'^(\d+)', response.strip())
    if not match:
        # Fallback: try standard \boxed{...} pattern
        match = re.search(r'\\boxed\{(\d+)\}?', response)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, OverflowError):
            return None
    return None


def run(
    model_id: str = "Qwen/Qwen3-4B",
    dataset_path: str = f"{RESULTS_PATH}/data/prime_multiplication_k3_eval.jsonl",
    max_new_tokens: int = 20,
    batch_size: int = 64,
    output_path: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
):
    """Evaluate model with forced empty think block (no CoT)."""

    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"

    dataset = utils.read_jsonl_all(dataset_path)
    print(f"Loaded {len(dataset)} eval problems from {dataset_path}")

    # Load model and tokenizer
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build prompts with prefill
    all_prompts = []
    for ex in dataset:
        messages = [{"role": "system", "content": system_prompt}] + ex["prompt"] + [
            {"role": "assistant", "content": PREFILL}
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            continue_final_message=True,
        )
        all_prompts.append(prompt_str)

    # Generate in batches
    all_responses = []
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = output[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_responses.append(response)

        print(f"  Batch {i // batch_size + 1}/{(len(all_prompts) + batch_size - 1) // batch_size}: "
              f"{len(all_responses)}/{len(dataset)} done")

    # Evaluate
    correct = 0
    results = []

    for ex, response in zip(dataset, all_responses):
        target = int(ex["answer"])
        predicted = parse_boxed_answer(response)
        is_correct = predicted is not None and predicted == target

        correct += int(is_correct)
        results.append({
            "id": ex["id"],
            "a": ex["a"],
            "b": ex["b"],
            "target": str(target),
            "predicted": str(predicted) if predicted is not None else None,
            "correct": is_correct,
            "response": response[:200],
        })

    n = len(dataset)
    accuracy = correct / n

    summary = {
        "model_id": model_id,
        "accuracy": accuracy,
        "n_problems": n,
        "n_correct": correct,
        "prefill": PREFILL,
    }

    print(f"\n=== No-CoT Eval Results ===")
    print(f"Model:    {model_id}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{n})")

    # Show a few examples
    for r in results[:3]:
        print(f"  {r['a']} × {r['b']} = {r['target']} | predicted: {r['predicted']} | {'✓' if r['correct'] else '✗'}")
        print(f"    Response: {r['response'][:100]}")

    if output_path is None:
        model_short = model_id.split("/")[-1].lower()
        ds_name = os.path.basename(dataset_path).replace(".jsonl", "")
        output_path = f"{RESULTS_PATH}/evals/no_cot_{model_short}_{ds_name}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils.save_json(output_path, {"summary": summary, "results": results})
    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({"run": run})
