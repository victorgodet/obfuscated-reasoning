"""Run all evaluations for cipher and distilled models on k=3-8.

Usage:
    VLLM_WORKER_MULTIPROC_METHOD=spawn uv run python scripts/run_all_evals.py
"""

import json
import os
import subprocess
import sys

MODELS = {
    "cipher": "vg10101/qwen3-4b-k3-k6-cipher-sft",
    "distilled": "vg10101/qwen3-4b-k3-k6-distilled-sft",
}

K_VALUES = [3, 4, 5, 6, 7, 8]
SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."
RESULTS_DIR = "results/evals"
DATA_DIR = "results/data"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} ({model_id})")
        print(f"{'='*60}")

        for k in K_VALUES:
            dataset_path = f"{DATA_DIR}/prime_multiplication_k{k}_eval.jsonl"
            output_path = f"{RESULTS_DIR}/{model_name}_k{k}.json"

            if not os.path.exists(dataset_path):
                print(f"\nSkipping k={k}: {dataset_path} not found")
                continue

            if os.path.exists(output_path):
                print(f"\nSkipping k={k}: {output_path} already exists")
                with open(output_path) as f:
                    data = json.load(f)
                    acc = data["summary"]["accuracy"]
                    print(f"  Previous result: accuracy={acc:.3f}")
                    all_results[f"{model_name}_k{k}"] = acc
                continue

            print(f"\n--- {model_name} k={k} ---")

            # Use more tokens for larger k values
            max_tokens = 4096 if k >= 7 else 2048

            cmd = [
                "uv", "run", "python", "scripts/run_multiplication_eval.py", "run",
                f"--model_id={model_id}",
                f"--dataset_path={dataset_path}",
                f"--max_new_tokens={max_tokens}",
                "--enable_thinking=True",
                f"--system_prompt={SYSTEM_PROMPT}",
                f"--output_path={output_path}",
            ]

            result = subprocess.run(cmd, capture_output=False, text=True)

            if result.returncode != 0:
                print(f"ERROR: Eval failed for {model_name} k={k}")
                continue

            # Read results
            if os.path.exists(output_path):
                with open(output_path) as f:
                    data = json.load(f)
                    acc = data["summary"]["accuracy"]
                    all_results[f"{model_name}_k{k}"] = acc

    # Print summary table
    print(f"\n\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'k':<6} {'Cipher':<15} {'Distilled':<15} {'Delta':<10}")
    print("-" * 46)
    for k in K_VALUES:
        cipher_acc = all_results.get(f"cipher_k{k}", None)
        distilled_acc = all_results.get(f"distilled_k{k}", None)
        cipher_str = f"{cipher_acc:.3f}" if cipher_acc is not None else "N/A"
        distilled_str = f"{distilled_acc:.3f}" if distilled_acc is not None else "N/A"
        if cipher_acc is not None and distilled_acc is not None:
            delta = cipher_acc - distilled_acc
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "N/A"
        print(f"k={k:<4} {cipher_str:<15} {distilled_str:<15} {delta_str:<10}")

    # Mark in-distribution vs OOD
    print("-" * 46)
    print("k=3-6: in-distribution (seen during training)")
    print("k=7-8: out-of-distribution (never seen during training)")

    # Save summary
    summary_path = f"{RESULTS_DIR}/full_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
