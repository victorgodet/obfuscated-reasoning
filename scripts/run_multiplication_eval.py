"""Evaluate a model on the multiplication eval dataset.

Usage:
    python scripts/run_multiplication_eval.py run
    python scripts/run_multiplication_eval.py run --lora_adapter_path results/runs/.../global_step_100
"""

import fire
import json
import os
import re

from src import RESULTS_PATH, utils


def parse_boxed_answer(response: str) -> int | None:
    match = re.search(r'\\boxed\{(\d+)\}', response)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, OverflowError):
            return None
    return None


def parse_think_block(response: str) -> str:
    if "<think>" in response and "</think>" in response:
        start = response.index("<think>") + len("<think>")
        end = response.index("</think>")
        return response[start:end]
    return ""


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def run(
    model_id: str = "Qwen/Qwen3-0.6B",
    lora_adapter_path: str | None = None,
    dataset_path: str = f"{RESULTS_PATH}/data/multiplication_eval.jsonl",
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    n_samples: int = 1,
    output_path: str | None = None,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
):
    """Run multiplication evaluation."""
    from src.generate import VLLMGenerator, SamplingParams

    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}. Run scripts/generate_multiplication_data.py first."

    dataset = utils.read_jsonl_all(dataset_path)
    print(f"Loaded {len(dataset)} eval problems from {dataset_path}")

    llm = VLLMGenerator(
        model_id,
        lora_adapter_path=lora_adapter_path,
        max_model_len=max_new_tokens + 256,
        gpu_memory_utilization=0.7,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        n=n_samples,
    )

    if enable_thinking:
        llm.turn_on_thinking()

    # Generate completions
    prompts = [ex["prompt"] for ex in dataset]
    if system_prompt:
        prompts = [
            [{"role": "system", "content": system_prompt}] + p
            for p in prompts
        ]

    responses = llm.batch_generate(prompts, sampling_params)
    llm.cleanup()

    # Evaluate
    correct = 0
    total_digit_ratio = 0.0
    total_think_length = 0
    results = []

    for ex, resp in zip(dataset, responses):
        response_text = resp[0] if isinstance(resp, list) else resp
        target = int(ex["answer"])
        predicted = parse_boxed_answer(response_text)
        think_text = parse_think_block(response_text)
        dr = digit_ratio(think_text)
        is_correct = predicted is not None and predicted == target

        correct += int(is_correct)
        total_digit_ratio += dr
        total_think_length += len(think_text)

        results.append({
            "id": ex["id"],
            "a": ex["a"],
            "b": ex["b"],
            "target": target,
            "predicted": predicted,
            "correct": is_correct,
            "digit_ratio": dr,
            "think_length": len(think_text),
        })

    n = len(dataset)
    summary = {
        "model_id": model_id,
        "lora_adapter_path": lora_adapter_path,
        "accuracy": correct / n,
        "avg_digit_ratio": total_digit_ratio / n,
        "avg_think_length": total_think_length / n,
        "n_problems": n,
        "n_correct": correct,
    }

    print("\n=== Multiplication Eval Results ===")
    print(f"Accuracy:          {summary['accuracy']:.3f} ({correct}/{n})")
    print(f"Avg digit ratio:   {summary['avg_digit_ratio']:.4f}")
    print(f"Avg think length:  {summary['avg_think_length']:.0f} chars")

    if output_path is None:
        adapter_name = lora_adapter_path.split("/")[-1] if lora_adapter_path else "base"
        output_path = f"{RESULTS_PATH}/evals/multiplication_eval_{adapter_name}.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils.save_json(output_path, {"summary": summary, "results": results})
    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({"run": run})
