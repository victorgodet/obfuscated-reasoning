"""Decode and analyze cipher SFT model outputs.

Two commands:
  analyze — full pipeline: generate responses from a cipher-SFT model, decode think blocks, analyze accuracy
  decode  — offline decode of existing results (no GPU needed)

Usage:
    uv run python scripts/decode_cipher_responses.py analyze
    uv run python scripts/decode_cipher_responses.py analyze --n_problems=3
    uv run python scripts/decode_cipher_responses.py decode --results_path=results/evals/cipher_analysis_*.jsonl --cipher_path=results/data/cipher_k5.json
"""

import json
import os
import re
import statistics
from datetime import datetime

import fire

from src import RESULTS_PATH, utils


SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."


def parse_boxed_answer(response: str) -> int | None:
    """Extract integer from \\boxed{...}."""
    match = re.search(r'\\boxed\{(\d+)\}', response)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, OverflowError):
            return None
    return None


def parse_think_block(response: str) -> str:
    """Extract content between <think> and </think> tags.

    Handles missing </think> by taking everything after <think>.
    """
    if "<think>" not in response:
        return ""
    start = response.index("<think>") + len("<think>")
    if "</think>" in response:
        end = response.index("</think>")
        return response[start:end]
    return response[start:]


def relative_error(predicted: int, target: int) -> float:
    """Compute relative error as |predicted - target| / target."""
    if target == 0:
        return float("inf") if predicted != 0 else 0.0
    return abs(predicted - target) / abs(target)


def decode_think(encoded_think: str, tokenizer, inverse_cipher: dict[int, int]) -> str:
    """Decode an encoded think block using the inverse cipher."""
    from src.train.cipher import encode_think_content
    return encode_think_content(encoded_think, tokenizer, inverse_cipher)


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def analyze(
    model_id: str = "vg10101/qwen3-4b-prime-mult-k5-step50",
    lora_adapter_path: str | None = None,
    cipher_path: str = f"{RESULTS_PATH}/data/cipher_k5.json",
    dataset_path: str = f"{RESULTS_PATH}/data/prime_multiplication_k5_eval.jsonl",
    n_problems: int = 10,
    max_new_tokens: int = 8192,
    temperature: float = 0.0,
    output_path: str | None = None,
):
    """Generate responses from a cipher-SFT model, decode think blocks, analyze accuracy.

    Full pipeline: load model → generate → decode → analyze → save results.
    """
    from src.generate import VLLMGenerator
    from src import SamplingParams
    from src.train.cipher import load_cipher, invert_cipher

    assert os.path.exists(cipher_path), f"Cipher not found: {cipher_path}"
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"

    # Load cipher and create inverse
    cipher = load_cipher(cipher_path)
    inverse = invert_cipher(cipher)

    # Load dataset
    dataset = utils.read_jsonl_all(dataset_path)
    dataset = dataset[:n_problems]
    print(f"Loaded {len(dataset)} problems from {dataset_path}")

    # Generate responses
    llm = VLLMGenerator(
        model_id,
        lora_adapter_path=lora_adapter_path,
        max_model_len=max_new_tokens + 256,
        gpu_memory_utilization=0.7,
    )
    llm.turn_on_thinking()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        n=1,
    )

    prompts = [
        [{"role": "system", "content": SYSTEM_PROMPT}] + ex["prompt"]
        for ex in dataset
    ]
    responses = llm.batch_generate(prompts, sampling_params)

    # Get tokenizer for decoding, then cleanup GPU
    tokenizer = llm.tokenizer
    llm.cleanup()

    # Analyze each response
    results = []
    for i, (ex, resp) in enumerate(zip(dataset, responses)):
        response_text = resp[0] if isinstance(resp, list) else resp
        target = int(ex["answer"])
        predicted = parse_boxed_answer(response_text)
        encoded_think = parse_think_block(response_text)
        decoded_think = decode_think(encoded_think, tokenizer, inverse) if encoded_think else ""

        is_correct = predicted is not None and predicted == target
        has_boxed = predicted is not None
        rel_err = relative_error(predicted, target) if predicted is not None else None

        result = {
            "id": ex.get("id", i),
            "a": ex.get("a"),
            "b": ex.get("b"),
            "target": target,
            "predicted": predicted,
            "correct": is_correct,
            "has_boxed": has_boxed,
            "relative_error": rel_err,
            "encoded_think": encoded_think,
            "decoded_think": decoded_think,
            "response": response_text,
        }
        results.append(result)

        # Print per-example summary
        print(f"\n{'='*60}")
        print(f"Example {i+1}: {ex.get('a', '?')} × {ex.get('b', '?')} = {target}")
        print(f"  Encoded think: {truncate(encoded_think, 150)}")
        print(f"  Decoded think: {truncate(decoded_think, 150)}")
        print(f"  Predicted: {predicted}  |  Target: {target}  |  Correct: {is_correct}")
        if rel_err is not None:
            print(f"  Relative error: {rel_err:.4%}")

    # Summary statistics
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_boxed = sum(1 for r in results if r["has_boxed"])
    rel_errors = [r["relative_error"] for r in results if r["relative_error"] is not None]

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy:           {n_correct}/{n} ({n_correct/n*100:.1f}%)")
    print(f"  Has \\boxed{{}}:      {n_boxed}/{n} ({n_boxed/n*100:.1f}%)")

    if rel_errors:
        median_err = statistics.median(rel_errors)
        within_1pct = sum(1 for e in rel_errors if e <= 0.01)
        within_5pct = sum(1 for e in rel_errors if e <= 0.05)
        print(f"  Median rel error:   {median_err:.4%}")
        print(f"  Within 1%:          {within_1pct}/{len(rel_errors)} ({within_1pct/len(rel_errors)*100:.1f}%)")
        print(f"  Within 5%:          {within_5pct}/{len(rel_errors)} ({within_5pct/len(rel_errors)*100:.1f}%)")

    # Save results
    if output_path is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{RESULTS_PATH}/evals/cipher_analysis_{run_id}.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils.save_jsonl(output_path, results)
    print(f"\nResults saved to {output_path}")


def decode(
    results_path: str,
    cipher_path: str = f"{RESULTS_PATH}/data/cipher_k5.json",
    n: int | None = None,
):
    """Decode think blocks from saved results (no GPU needed).

    Reads a JSONL or JSON results file, applies the inverse cipher to encoded
    think blocks, and prints the decoded reasoning.
    """
    from transformers import AutoTokenizer
    from src.train.cipher import load_cipher, invert_cipher

    assert os.path.exists(results_path), f"Results file not found: {results_path}"
    assert os.path.exists(cipher_path), f"Cipher not found: {cipher_path}"

    # Load cipher
    cipher = load_cipher(cipher_path)
    inverse = invert_cipher(cipher)

    # Load results
    if results_path.endswith(".jsonl"):
        data = utils.read_jsonl_all(results_path)
    else:
        raw = utils.read_json(results_path)
        data = raw.get("results", raw) if isinstance(raw, dict) else raw

    if n is not None:
        data = data[:n]

    # Load tokenizer for decoding (CPU only, no model needed)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    print(f"Decoding {len(data)} examples from {results_path}\n")

    for i, item in enumerate(data):
        # Extract encoded think block from response or from pre-extracted field
        encoded_think = item.get("encoded_think", "")
        if not encoded_think and "response" in item:
            encoded_think = parse_think_block(item["response"])

        if not encoded_think:
            print(f"Example {i+1}: (no think block found)")
            continue

        decoded = decode_think(encoded_think, tokenizer, inverse)

        print(f"{'='*60}")
        print(f"Example {i+1}")
        if "a" in item and "b" in item:
            print(f"  Problem: {item['a']} × {item['b']} = {item.get('target', '?')}")
        if "predicted" in item:
            print(f"  Predicted: {item['predicted']}")
        print(f"\n  Encoded think (truncated):\n    {truncate(encoded_think, 300)}")
        print(f"\n  Decoded think (truncated):\n    {truncate(decoded, 300)}")
        print()


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({"analyze": analyze, "decode": decode})
