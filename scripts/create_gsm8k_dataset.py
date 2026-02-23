"""Download GSM8K from HuggingFace and produce JSONL files for GRPO training.

Produces:
- results/data/gsm8k_train.jsonl (from train split)
- results/data/gsm8k_eval.jsonl (from test split)

Each example has the same structure as the multiplication dataset.
GSM8K answers have the format "#### 42" — we parse the number after ####
and strip commas.
"""

import json
import os
import re

from datasets import load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "data")


def parse_gsm8k_answer(answer_text: str) -> str:
    """Extract the numeric answer from GSM8K's '#### <number>' format."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    raise ValueError(f"Could not parse GSM8K answer: {answer_text!r}")


def convert_split(dataset, dataset_name: str) -> list[dict]:
    problems = []
    for i, example in enumerate(dataset):
        answer = parse_gsm8k_answer(example["answer"])
        problems.append({
            "id": i,
            "dataset": "gsm8k",
            "prompt": [{"role": "user", "content": example["question"]}],
            "answer": answer,
            "gt_answer": answer,
            "evaluator": "gsm8k",
        })
    return problems


def write_jsonl(problems: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(problems)} problems to {path}")


def main():
    ds = load_dataset("openai/gsm8k", "main")
    train = convert_split(ds["train"], "gsm8k")
    eval_ = convert_split(ds["test"], "gsm8k")

    write_jsonl(train, os.path.join(RESULTS_DIR, "gsm8k_train.jsonl"))
    write_jsonl(eval_, os.path.join(RESULTS_DIR, "gsm8k_eval.jsonl"))


if __name__ == "__main__":
    main()
