"""Generate multiplication dataset for GRPO training.

Produces JSONL files compatible with verl's expected format:
- results/data/multiplication_train.jsonl (10,000 problems)
- results/data/multiplication_eval.jsonl (500 problems)

Each problem is 3-digit x 3-digit multiplication (100-999).
"""

import json
import os
import random

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "data")


def generate_problems(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        a = rng.randint(100, 999)
        b = rng.randint(100, 999)
        product = a * b
        problems.append({
            "id": i,
            "dataset": "multiplication",
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


def main():
    train = generate_problems(10_000, seed=42)
    eval_ = generate_problems(500, seed=123)

    write_jsonl(train, os.path.join(RESULTS_DIR, "multiplication_train.jsonl"))
    write_jsonl(eval_, os.path.join(RESULTS_DIR, "multiplication_eval.jsonl"))


if __name__ == "__main__":
    main()
