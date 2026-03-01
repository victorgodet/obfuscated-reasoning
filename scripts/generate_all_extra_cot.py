"""Generate cipher-encoded CoT data for all extra problem sets.

Runs generate_encoded_cot_data.py pipeline for each k value's extra problems.
Must be run after the initial k=6 generation from k6_large_train is complete.

Usage:
    uv run python scripts/generate_all_extra_cot.py
"""

import os
import sys

from src import RESULTS_PATH, utils


SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."


def generate_cot_for_dataset(
    dataset_path: str,
    output_path: str,
    cipher_path: str,
    n_problems: int,
    model_id: str = "vg10101/qwen3-4b-prime-mult-k5-step50",
    cipher_seed: int = 42,
    max_new_tokens: int = 8192,
):
    """Generate CoTs, filter correct, encode think blocks, save SFT dataset."""
    import re

    from src.generate import VLLMGenerator
    from src import SamplingParams
    from src.train.cipher import create_cipher, load_cipher, save_cipher, encode_response

    # 1. Load problems
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
    dataset = utils.read_jsonl_all(dataset_path)
    dataset = dataset[:n_problems]
    print(f"\nLoaded {len(dataset)} problems from {dataset_path}")

    # 2. Generate CoTs with thinking enabled
    llm = VLLMGenerator(
        model_id,
        max_model_len=max_new_tokens + 256,
        gpu_memory_utilization=0.85,
    )
    llm.turn_on_thinking()

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        n=1,
    )

    prompts = [
        [{"role": "system", "content": SYSTEM_PROMPT}] + ex["prompt"]
        for ex in dataset
    ]
    responses = llm.batch_generate(prompts, sampling_params)

    # 3. Filter for correct answers
    def parse_boxed_answer(response: str) -> int | None:
        match = re.search(r'\\boxed\{(\d+)\}', response)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, OverflowError):
                return None
        return None

    correct_examples = []
    for ex, resp in zip(dataset, responses):
        response_text = resp[0] if isinstance(resp, list) else resp
        target = int(ex["answer"])
        predicted = parse_boxed_answer(response_text)
        if predicted is not None and predicted == target:
            correct_examples.append((ex, response_text))

    print(f"Correct: {len(correct_examples)}/{len(dataset)} ({len(correct_examples)/len(dataset)*100:.1f}%)")

    # 4. Load or create cipher
    tokenizer = llm.tokenizer

    if os.path.exists(cipher_path):
        cipher = load_cipher(cipher_path)
        print(f"Loaded existing cipher from {cipher_path}")
    else:
        cipher = create_cipher(tokenizer, seed=cipher_seed)
        save_cipher(cipher, cipher_path)
        print(f"Created and saved cipher to {cipher_path}")

    # 5. Encode think blocks and build SFT dataset
    sft_data = []
    for i, (ex, response_text) in enumerate(correct_examples):
        encoded_response = encode_response(response_text, tokenizer, cipher)
        sft_data.append({
            "id": i,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"][-1]["content"]},
                {"role": "assistant", "content": encoded_response},
            ],
            "original_response": response_text,
            "answer": ex["answer"],
            "a": ex["a"],
            "b": ex["b"],
        })

    # Cleanup GPU before next run
    llm.cleanup()

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils.save_jsonl(output_path, sft_data)
    print(f"Saved {len(sft_data)} encoded SFT examples to {output_path}")
    return len(sft_data)


def main():
    cipher_path = f"{RESULTS_PATH}/data/cipher_k5.json"

    # Jobs to run: (dataset_path, output_path, n_problems)
    jobs = [
        (
            f"{RESULTS_PATH}/data/prime_multiplication_k3_extra_large_train.jsonl",
            f"{RESULTS_PATH}/data/encoded_cot_k3_extra_sft.jsonl",
            16000,
        ),
        (
            f"{RESULTS_PATH}/data/prime_multiplication_k4_extra_large_train.jsonl",
            f"{RESULTS_PATH}/data/encoded_cot_k4_extra_sft.jsonl",
            17000,
        ),
        (
            f"{RESULTS_PATH}/data/prime_multiplication_k5_extra_large_train.jsonl",
            f"{RESULTS_PATH}/data/encoded_cot_k5_extra_sft.jsonl",
            19000,
        ),
        (
            f"{RESULTS_PATH}/data/prime_multiplication_k6_extra_large_train.jsonl",
            f"{RESULTS_PATH}/data/encoded_cot_k6_extra_sft.jsonl",
            10000,
        ),
        # Also process the base k6_train set (10K problems, not yet processed)
        (
            f"{RESULTS_PATH}/data/prime_multiplication_k6_train.jsonl",
            f"{RESULTS_PATH}/data/encoded_cot_k6_base_sft.jsonl",
            10000,
        ),
    ]

    total_generated = 0
    for dataset_path, output_path, n_problems in jobs:
        # Skip if output already exists
        if os.path.exists(output_path):
            count = len(utils.read_jsonl_all(output_path))
            print(f"\nSkipping {output_path} (already exists with {count} examples)")
            total_generated += count
            continue

        print(f"\n{'='*60}")
        print(f"Generating: {output_path}")
        print(f"{'='*60}")
        count = generate_cot_for_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            cipher_path=cipher_path,
            n_problems=n_problems,
        )
        total_generated += count

    print(f"\n{'='*60}")
    print(f"Total new examples generated: {total_generated}")
    print(f"{'='*60}")


if __name__ == "__main__":
    utils.load_dotenv()
    main()
