"""Generate prime multiplication dataset for GRPO training.

Produces JSONL files compatible with verl's expected format:
- results/data/prime_multiplication_k{K}_train.jsonl (10,000 problems)
- results/data/prime_multiplication_k{K}_eval.jsonl (500 problems)

Each problem is K-digit prime x K-digit prime multiplication.
"""

import json
import os
import random

import fire
from sympy import isprime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "data")


def get_k_digit_primes(k: int) -> list[int]:
    """Return all K-digit prime numbers."""
    lo = 10 ** (k - 1)
    hi = 10 ** k - 1
    return [n for n in range(lo, hi + 1) if isprime(n)]


def generate_problems(n: int, seed: int, primes: list[int]) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        a = rng.choice(primes)
        b = rng.choice(primes)
        product = a * b
        problems.append({
            "id": i,
            "dataset": "prime_multiplication",
            "prompt": [{"role": "user", "content": f"What is {a} times {b}?"}],
            "answer": str(product),
            "gt_answer": str(product),
            "evaluator": "multiplication",
            "a": a,
            "b": b,
        })
    return problems


def write_jsonl(problems: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(problems)} problems to {path}")


def main(k: int = 3):
    primes = get_k_digit_primes(k)
    print(f"Found {len(primes)} {k}-digit primes in range [{10**(k-1)}, {10**k - 1}]")

    train = generate_problems(10_000, seed=42, primes=primes)
    eval_ = generate_problems(500, seed=123, primes=primes)

    write_jsonl(train, os.path.join(RESULTS_DIR, f"prime_multiplication_k{k}_train.jsonl"))
    write_jsonl(eval_, os.path.join(RESULTS_DIR, f"prime_multiplication_k{k}_eval.jsonl"))


if __name__ == "__main__":
    fire.Fire(main)
