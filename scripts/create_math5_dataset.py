"""Download chiayewken/competition_math from HuggingFace and produce JSONL files for GRPO training.

Filters to Level 5 only (hardest problems).

Produces:
- results/data/math5_train.jsonl (from train split)
- results/data/math5_eval.jsonl (from test split)

MATH answers are extracted from \boxed{} in the solution field and can be
fractions, expressions, etc. (not just integers).
"""

import json
import os

from datasets import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "data")


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the last \\boxed{...} content from a MATH solution string.

    Handles nested braces, e.g. \\boxed{\\frac{1}{2}}.
    """
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(solution) and depth > 0:
        if solution[pos] == "{":
            depth += 1
        elif solution[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return solution[start : pos - 1]


def convert_split(dataset) -> list[dict]:
    problems = []
    skipped = 0
    for example in dataset:
        level = example["level"]
        if level != "Level 5":
            continue

        answer = extract_boxed_answer(example["solution"])
        if answer is None:
            skipped += 1
            continue

        problems.append({
            "id": len(problems),
            "dataset": "math5",
            "prompt": [{"role": "user", "content": example["problem"]}],
            "answer": answer,
            "gt_answer": answer,
            "evaluator": "math5",
        })

    if skipped:
        print(f"  Skipped {skipped} examples with unparseable \\boxed{{}} answers")
    return problems


def write_jsonl(problems: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(problems)} problems to {path}")


def main():
    # Original hendrycks/competition_math is disabled on HF; use mirror
    ds = load_dataset("chiayewken/competition_math")
    train = convert_split(ds["train"])
    eval_ = convert_split(ds["test"])

    print(f"Train: {len(train)} problems (Level 5)")
    print(f"Eval:  {len(eval_)} problems (Level 5)")

    write_jsonl(train, os.path.join(RESULTS_DIR, "math5_train.jsonl"))
    write_jsonl(eval_, os.path.join(RESULTS_DIR, "math5_eval.jsonl"))


if __name__ == "__main__":
    main()
