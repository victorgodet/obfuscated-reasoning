"""Dataset generation for covert communication training.

Generates JSONL files of prime numbers organized by digit count (levels)
for curriculum training.

Usage:
    from src.train.covert_comm.dataset import generate_dataset
    generate_dataset(level=1)  # 2-digit primes (10-99)
    generate_dataset(level=2)  # 3-digit primes (100-999)
"""

from __future__ import annotations

import json
import math
import os

from src.train.covert_comm.prompts import ENCODE_SYSTEM_PROMPT, make_encode_prompt


def sieve_primes(lower: int, upper: int) -> list[int]:
    """Return all primes in [lower, upper] using the Sieve of Eratosthenes."""
    if upper < 2:
        return []
    is_prime = [True] * (upper + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(upper)) + 1):
        if is_prime[i]:
            for j in range(i * i, upper + 1, i):
                is_prime[j] = False
    return [n for n in range(max(lower, 2), upper + 1) if is_prime[n]]


def get_primes_by_level(level: int) -> list[int]:
    """Get primes for a given digit-count level.

    Level 1 = 2-digit primes (10-99)
    Level 2 = 3-digit primes (100-999)
    Level 3 = 4-digit primes (1000-9999)
    etc.
    """
    num_digits = level + 1
    lower = 10 ** (num_digits - 1)
    upper = 10**num_digits - 1
    return sieve_primes(lower, upper)


def generate_dataset(level: int = 1, output_dir: str = "results/data") -> str:
    """Generate a JSONL dataset file for a given prime level.

    Each row follows the ``DatasetExample`` schema expected by the training
    pipeline.

    Args:
        level: Difficulty level (1 = 2-digit, 2 = 3-digit, ...).
        output_dir: Directory to write the output file.

    Returns:
        Path to the generated JSONL file.
    """
    primes = get_primes_by_level(level)
    num_digits = level + 1

    examples = []
    for idx, prime in enumerate(primes):
        examples.append(
            {
                "id": idx,
                "dataset": "covert_comm",
                "evaluator": "covert_comm",
                "question": f"Encode the prime number {prime}",
                "gt_answer": str(prime),
                "answer": str(prime),
                "prompt": [
                    {"role": "system", "content": ENCODE_SYSTEM_PROMPT},
                    {"role": "user", "content": make_encode_prompt(prime)},
                ],
                "hint": None,
                "prompt_metadata": {
                    "target_number": prime,
                    "num_digits": num_digits,
                },
            }
        )

    output_path = os.path.join(output_dir, f"covert_comm_level{level}.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples ({num_digits}-digit primes) to {output_path}")
    return output_path
