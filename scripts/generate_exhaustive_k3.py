"""Generate exhaustive k=3 prime multiplication problems.

With only 143 three-digit primes (20,449 ordered pairs), random sampling
causes heavy duplication. This script generates ALL ordered pairs that
aren't already covered by existing data.
"""

import json
import os
import random
import re

from sympy import isprime

from src import RESULTS_PATH, utils

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "data")


def main():
    # Load existing k=3 encoded examples to find already-covered problems
    existing_problems = set()
    for fname in ["encoded_cot_k3_sft.jsonl", "encoded_cot_k3_extra_sft.jsonl"]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            data = utils.read_jsonl_all(path)
            for ex in data:
                user_msg = ex["messages"][1]["content"]
                existing_problems.add(user_msg)
    print(f"Existing unique k=3 problems: {len(existing_problems)}")

    # Generate ALL ordered pairs
    primes = [n for n in range(100, 1000) if isprime(n)]
    print(f"Found {len(primes)} 3-digit primes ({len(primes)**2} ordered pairs)")

    all_pairs = []
    for a in primes:
        for b in primes:
            user_msg = f"What is {a} times {b}?"
            if user_msg not in existing_problems:
                product = a * b
                all_pairs.append({
                    "id": len(all_pairs),
                    "dataset": "prime_multiplication",
                    "prompt": [{"role": "user", "content": user_msg}],
                    "answer": str(product),
                    "gt_answer": str(product),
                    "evaluator": "multiplication",
                    "a": a,
                    "b": b,
                })

    print(f"New unique k=3 problems to generate: {len(all_pairs)}")

    # Shuffle for variety
    random.Random(2000).shuffle(all_pairs)

    # Write
    output_path = os.path.join(RESULTS_DIR, "prime_multiplication_k3_exhaustive_train.jsonl")
    with open(output_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(all_pairs)} problems to {output_path}")


if __name__ == "__main__":
    main()
