"""Run evaluations for the distilled model on k=3-8.

Usage:
    VLLM_WORKER_MULTIPROC_METHOD=spawn uv run python scripts/run_distilled_evals.py
"""

import json
import os
import subprocess

MODEL_ID = "vg10101/qwen3-4b-k3-k6-distilled-sft"
MODEL_NAME = "distilled"
K_VALUES = [3, 4, 5, 6, 7, 8]
SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."
RESULTS_DIR = "results/evals"
DATA_DIR = "results/data"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for k in K_VALUES:
        dataset_path = f"{DATA_DIR}/prime_multiplication_k{k}_eval.jsonl"
        output_path = f"{RESULTS_DIR}/{MODEL_NAME}_k{k}.json"

        if not os.path.exists(dataset_path):
            print(f"\nSkipping k={k}: {dataset_path} not found")
            continue

        if os.path.exists(output_path):
            print(f"\nSkipping k={k}: {output_path} already exists")
            with open(output_path) as f:
                data = json.load(f)
                acc = data["summary"]["accuracy"]
                print(f"  Previous result: accuracy={acc:.3f}")
            continue

        print(f"\n--- {MODEL_NAME} k={k} ---")

        max_tokens = 4096 if k >= 7 else 2048

        cmd = [
            "uv", "run", "python", "scripts/run_multiplication_eval.py", "run",
            f"--model_id={MODEL_ID}",
            f"--dataset_path={dataset_path}",
            f"--max_new_tokens={max_tokens}",
            "--enable_thinking=True",
            f"--system_prompt={SYSTEM_PROMPT}",
            f"--output_path={output_path}",
        ]

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"ERROR: Eval failed for {MODEL_NAME} k={k}")
            continue

        if os.path.exists(output_path):
            with open(output_path) as f:
                data = json.load(f)
                acc = data["summary"]["accuracy"]
                print(f"Result: accuracy={acc:.3f}")

    # Print summary
    print(f"\n{'='*40}")
    print(f"DISTILLED MODEL RESULTS")
    print(f"{'='*40}")
    for k in K_VALUES:
        output_path = f"{RESULTS_DIR}/{MODEL_NAME}_k{k}.json"
        if os.path.exists(output_path):
            with open(output_path) as f:
                data = json.load(f)
                acc = data["summary"]["accuracy"]
                n = data["summary"]["n_correct"]
                total = data["summary"]["n_problems"]
                think_len = data["summary"]["avg_think_length"]
                print(f"k={k}: {acc:.3f} ({n}/{total}), avg think len: {think_len:.0f}")


if __name__ == "__main__":
    main()
