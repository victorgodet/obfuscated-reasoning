"""Combine and prepare the 100K SFT dataset from all k-value encoded CoT files.

Merges existing + extra data per k, shuffles, caps at 25K each.
Creates both cipher and distilled versions.

Usage:
    uv run python scripts/combine_100k_dataset.py
"""

import json
import os
import random
import re

from src import RESULTS_PATH, utils

SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."
TARGET_PER_K = 25000
SEED = 42


def get_k_from_example(ex: dict) -> int:
    """Determine k (digit count) from user message."""
    user_msg = ex["messages"][1]["content"]
    nums = re.findall(r"\d+", user_msg)
    if nums:
        return len(nums[0])
    return 0


def strip_think_block(content: str) -> str:
    """Remove think block content, keeping empty think tags."""
    # Replace <think>...</think> with <think>\n</think>
    return re.sub(r"<think>.*?</think>", "<think>\n</think>", content, flags=re.DOTALL)


def create_distilled_example(ex: dict) -> dict:
    """Create a distilled version with empty think block."""
    new_ex = dict(ex)
    new_messages = list(ex["messages"])
    assistant_msg = dict(new_messages[2])
    assistant_msg["content"] = strip_think_block(assistant_msg["content"])
    new_messages[2] = assistant_msg
    new_ex["messages"] = new_messages
    return new_ex


def main():
    data_dir = f"{RESULTS_PATH}/data"

    # Define source files for each k value
    # Each entry: (filepath, description)
    sources = {
        3: [
            (f"{data_dir}/encoded_cot_k3_sft.jsonl", "existing k3"),
            (f"{data_dir}/encoded_cot_k3_extra_sft.jsonl", "extra k3"),
            (f"{data_dir}/encoded_cot_k3_exhaustive_sft.jsonl", "exhaustive k3"),
        ],
        4: [
            (f"{data_dir}/encoded_cot_k4_sft.jsonl", "existing k4"),
            (f"{data_dir}/encoded_cot_k4_extra_sft.jsonl", "extra k4"),
        ],
        5: [
            # k5 data is in the combined k4_k5 file - need to extract k5 portion
            (f"{data_dir}/encoded_cot_k4_k5_sft.jsonl", "existing k5 (from combined)"),
            (f"{data_dir}/encoded_cot_k5_sft.jsonl", "existing k5 standalone"),
            (f"{data_dir}/encoded_cot_k5_extra_sft.jsonl", "extra k5"),
        ],
        6: [
            (f"{data_dir}/encoded_cot_k6_sft.jsonl", "k6 large"),
            (f"{data_dir}/encoded_cot_k6_extra_sft.jsonl", "k6 extra"),
            (f"{data_dir}/encoded_cot_k6_base_sft.jsonl", "k6 base"),
        ],
    }

    rng = random.Random(SEED)
    all_cipher_data = []
    all_distilled_data = []

    for k in sorted(sources.keys()):
        print(f"\n--- Processing k={k} ---")
        k_data = []

        for filepath, desc in sources[k]:
            if not os.path.exists(filepath):
                print(f"  Skipping {desc}: {filepath} not found")
                continue

            examples = utils.read_jsonl_all(filepath)
            # Filter to only include examples with the right k value
            filtered = [ex for ex in examples if get_k_from_example(ex) == k]
            print(f"  {desc}: {len(filtered)} examples (from {len(examples)} total)")
            k_data.extend(filtered)

        # Deduplicate by user message content
        seen = set()
        unique_data = []
        for ex in k_data:
            user_msg = ex["messages"][1]["content"]
            if user_msg not in seen:
                seen.add(user_msg)
                unique_data.append(ex)

        print(f"  After dedup: {len(unique_data)} unique examples")

        # Shuffle and cap
        rng.shuffle(unique_data)
        capped = unique_data[:TARGET_PER_K]
        print(f"  After cap at {TARGET_PER_K}: {len(capped)} examples")

        if len(capped) < TARGET_PER_K:
            print(f"  WARNING: Only {len(capped)} examples for k={k}, target was {TARGET_PER_K}")

        # Re-index IDs
        for i, ex in enumerate(capped):
            ex["id"] = len(all_cipher_data) + i

        all_cipher_data.extend(capped)

        # Create distilled versions
        for ex in capped:
            distilled = create_distilled_example(ex)
            all_distilled_data.append(distilled)

    # Final shuffle of combined dataset
    combined_indices = list(range(len(all_cipher_data)))
    rng.shuffle(combined_indices)
    all_cipher_data = [all_cipher_data[i] for i in combined_indices]
    all_distilled_data = [all_distilled_data[i] for i in combined_indices]

    # Re-index final IDs
    for i, (cipher_ex, distilled_ex) in enumerate(zip(all_cipher_data, all_distilled_data)):
        cipher_ex["id"] = i
        distilled_ex["id"] = i

    # Verify cipher and distilled have same problems
    for i, (c, d) in enumerate(zip(all_cipher_data, all_distilled_data)):
        assert c["messages"][1]["content"] == d["messages"][1]["content"], f"Mismatch at index {i}"
        assert c.get("answer") == d.get("answer"), f"Answer mismatch at index {i}"

    # Save
    cipher_path = f"{data_dir}/encoded_cot_100k_sft.jsonl"
    distilled_path = f"{data_dir}/distilled_100k_sft.jsonl"

    utils.save_jsonl(cipher_path, all_cipher_data)
    utils.save_jsonl(distilled_path, all_distilled_data)

    print(f"\n{'='*60}")
    print(f"Cipher dataset: {len(all_cipher_data)} examples → {cipher_path}")
    print(f"Distilled dataset: {len(all_distilled_data)} examples → {distilled_path}")

    # Print k distribution
    k_dist = {}
    for ex in all_cipher_data:
        k = get_k_from_example(ex)
        k_dist[k] = k_dist.get(k, 0) + 1
    print(f"\nk distribution: {dict(sorted(k_dist.items()))}")

    # Verify distilled has empty think blocks
    sample = all_distilled_data[0]
    assistant_content = sample["messages"][2]["content"]
    assert "<think>\n</think>" in assistant_content, "Distilled example should have empty think block"
    print("Verification passed: distilled examples have empty think blocks")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
