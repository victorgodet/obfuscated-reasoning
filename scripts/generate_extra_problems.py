"""Generate extra prime multiplication problems for the 100K SFT dataset.

Generates large problem sets with different seeds from existing data to avoid
overlap. Also generates k=8 eval set.

Usage:
    uv run python scripts/generate_extra_problems.py
"""

import json
import os
import random

from sympy import isprime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "data")


def get_k_digit_primes(k: int) -> list[int]:
    """Return all K-digit prime numbers."""
    lo = 10 ** (k - 1)
    hi = 10**k - 1
    return [n for n in range(lo, hi + 1) if isprime(n)]


def generate_problems(n: int, seed: int, primes: list[int], id_offset: int = 0) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        a = rng.choice(primes)
        b = rng.choice(primes)
        product = a * b
        problems.append(
            {
                "id": id_offset + i,
                "dataset": "prime_multiplication",
                "prompt": [{"role": "user", "content": f"What is {a} times {b}?"}],
                "answer": str(product),
                "gt_answer": str(product),
                "evaluator": "multiplication",
                "a": a,
                "b": b,
            }
        )
    return problems


def write_jsonl(problems: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(problems)} problems to {path}")


def main():
    # Extra problem generation config
    # Using seed=1000 to avoid overlap with existing data (seed=42 for train, seed=123 for eval)
    # Generating more problems than needed to account for model accuracy filtering
    configs = [
        # (k, n_problems, seed, output_filename)
        (3, 16000, 1000, "prime_multiplication_k3_extra_large_train.jsonl"),
        (4, 17000, 1000, "prime_multiplication_k4_extra_large_train.jsonl"),
        (5, 19000, 1000, "prime_multiplication_k5_extra_large_train.jsonl"),
        (6, 10000, 1000, "prime_multiplication_k6_extra_large_train.jsonl"),
    ]

    for k, n, seed, filename in configs:
        print(f"\n--- Generating k={k} ---")
        primes = get_k_digit_primes(k)
        print(f"Found {len(primes)} {k}-digit primes")
        problems = generate_problems(n, seed=seed, primes=primes)
        write_jsonl(problems, os.path.join(RESULTS_DIR, filename))

    # Also generate k=8 eval set (needed for OOD evaluation)
    print("\n--- Generating k=8 eval set ---")
    primes_8 = get_k_digit_primes(8)
    print(f"Found {len(primes_8)} 8-digit primes")
    eval_problems = generate_problems(500, seed=123, primes=primes_8)
    write_jsonl(eval_problems, os.path.join(RESULTS_DIR, "prime_multiplication_k8_eval.jsonl"))


if __name__ == "__main__":
    main()
