"""Generate encoded CoT SFT data for cipher-based reasoning experiments.

Pipeline:
1. Load prime multiplication problems
2. Generate CoTs from step-50 model (greedy, thinking enabled)
3. Filter for correct answers (parse \\boxed{}, compare to ground truth)
4. Create token-level cipher and encode think blocks
5. Save SFT dataset as JSONL

Usage:
    uv run python scripts/generate_encoded_cot_data.py generate
    uv run python scripts/generate_encoded_cot_data.py generate --n_problems=2000
"""

import os
import re

import fire

from src import RESULTS_PATH, utils


SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."


def parse_boxed_answer(response: str) -> int | None:
    match = re.search(r'\\boxed\{(\d+)\}', response)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, OverflowError):
            return None
    return None


def generate(
    model_id: str = "vg10101/qwen3-4b-prime-mult-k5-step50",
    dataset_path: str = f"{RESULTS_PATH}/data/prime_multiplication_k5_train.jsonl",
    output_path: str = f"{RESULTS_PATH}/data/encoded_cot_k5_sft.jsonl",
    cipher_path: str = f"{RESULTS_PATH}/data/cipher_k5.json",
    cipher_seed: int = 42,
    n_problems: int = 2000,
    max_new_tokens: int = 8192,
    temperature: float = 0.0,
):
    """Generate CoTs, filter correct, encode think blocks, save SFT dataset."""
    from src.generate import VLLMGenerator
    from src import SamplingParams
    from src.train.cipher import create_cipher, save_cipher, encode_response

    # 1. Load problems
    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}"
    dataset = utils.read_jsonl_all(dataset_path)
    dataset = dataset[:n_problems]
    print(f"Loaded {len(dataset)} problems from {dataset_path}")

    # 2. Generate CoTs with thinking enabled
    llm = VLLMGenerator(
        model_id,
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

    # 3. Filter for correct answers
    correct_examples = []
    for ex, resp in zip(dataset, responses):
        response_text = resp[0] if isinstance(resp, list) else resp
        target = int(ex["answer"])
        predicted = parse_boxed_answer(response_text)
        if predicted is not None and predicted == target:
            correct_examples.append((ex, response_text))

    print(f"Correct: {len(correct_examples)}/{len(dataset)} ({len(correct_examples)/len(dataset)*100:.1f}%)")

    # 4. Create cipher and encode
    tokenizer = llm.tokenizer
    llm.cleanup()

    cipher = create_cipher(tokenizer, seed=cipher_seed)
    os.makedirs(os.path.dirname(cipher_path), exist_ok=True)
    save_cipher(cipher, cipher_path)
    print(f"Saved cipher to {cipher_path}")

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

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils.save_jsonl(output_path, sft_data)
    print(f"Saved {len(sft_data)} encoded SFT examples to {output_path}")


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({"generate": generate})
