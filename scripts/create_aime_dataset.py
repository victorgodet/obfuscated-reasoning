"""Download qq8933/AIME_1983_2024 from HuggingFace and produce JSONL files for GRPO training.

All AIME problems have integer answers in 0-999.

Train/eval split by year:
- Train: 1983-2022 (~870 problems)
- Eval: 2023-2024 (~43 problems)

Produces:
- results/data/aime_train.jsonl
- results/data/aime_eval.jsonl
"""

import json
import os

from datasets import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "data")

EVAL_YEARS = {2023, 2024}


def convert_examples(dataset, keep_eval: bool) -> list[dict]:
    """Convert HF examples to our JSONL format, filtering by year."""
    problems = []
    for example in dataset:
        year = int(example["Year"])
        is_eval = year in EVAL_YEARS
        if is_eval != keep_eval:
            continue

        answer = str(example["Answer"])

        # Skip problems with non-integer answers (e.g. "080 or 081 (both were accepted)")
        try:
            int(answer)
        except ValueError:
            continue

        problems.append({
            "id": len(problems),
            "dataset": "aime",
            "prompt": [{"role": "user", "content": example["Question"]}],
            "answer": answer,
            "gt_answer": answer,
            "evaluator": "aime",
        })

    return problems


def write_jsonl(problems: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(problems)} problems to {path}")


def main():
    ds = load_dataset("qq8933/AIME_1983_2024")

    # Dataset has a single split; iterate over all of them to be safe
    all_examples = []
    for split_name in ds:
        all_examples.extend(ds[split_name])

    class IterableList:
        def __init__(self, items):
            self.items = items
        def __iter__(self):
            return iter(self.items)

    train = convert_examples(IterableList(all_examples), keep_eval=False)
    eval_ = convert_examples(IterableList(all_examples), keep_eval=True)

    print(f"Train: {len(train)} problems (years <= 2022)")
    print(f"Eval:  {len(eval_)} problems (years 2023-2024)")

    write_jsonl(train, os.path.join(RESULTS_DIR, "aime_train.jsonl"))
    write_jsonl(eval_, os.path.join(RESULTS_DIR, "aime_eval.jsonl"))


if __name__ == "__main__":
    main()
