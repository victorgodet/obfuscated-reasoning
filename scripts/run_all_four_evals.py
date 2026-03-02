"""Run all 4 model evaluations on clean eval datasets (k=3-8).

Models:
1. Base Qwen3-4B (no CoT, prefill) - via eval_no_cot.py
2. GRPO model (no CoT, prefill) - via eval_no_cot.py
3. Cipher SFT - via run_multiplication_eval.py (vLLM)
4. Distilled SFT - via run_multiplication_eval.py (vLLM)
"""
import subprocess
import sys
import os

SYSTEM_PROMPT = "Compute the product of the two numbers. Put your final answer in \\boxed{}."

def run_cmd(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")
    return result.returncode

# ── 1. Base Qwen3-4B, no CoT (prefill) ──
print("\n\n" + "█"*60)
print("  MODEL 1: Base Qwen3-4B (no CoT)")
print("█"*60)
for k in range(3, 9):
    run_cmd([
        sys.executable, "scripts/eval_no_cot.py", "run",
        f"--model_id=Qwen/Qwen3-4B",
        f"--dataset_path=results/data/prime_multiplication_k{k}_eval.jsonl",
        f"--output_path=results/evals/no_cot_qwen3-4b_k{k}.json",
    ])

# ── 2. GRPO model, no CoT (prefill) ──
print("\n\n" + "█"*60)
print("  MODEL 2: GRPO model (no CoT)")
print("█"*60)
for k in range(3, 9):
    run_cmd([
        sys.executable, "scripts/eval_no_cot.py", "run",
        f"--model_id=vg10101/qwen3-4b-prime-mult-k5-step50",
        f"--dataset_path=results/data/prime_multiplication_k{k}_eval.jsonl",
        f"--output_path=results/evals/no_cot_grpo_k{k}.json",
    ])

# ── 3. Cipher SFT (with vLLM) ──
print("\n\n" + "█"*60)
print("  MODEL 3: Cipher SFT")
print("█"*60)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
for k in range(3, 9):
    run_cmd([
        sys.executable, "scripts/run_multiplication_eval.py", "run",
        f"--model_id=vg10101/qwen3-4b-k3-k6-cipher-sft",
        f"--dataset_path=results/data/prime_multiplication_k{k}_eval.jsonl",
        f"--enable_thinking=True",
        f"--system_prompt={SYSTEM_PROMPT}",
        f"--output_path=results/evals/cipher_k{k}.json",
    ])

# ── 4. Distilled SFT (with vLLM) ──
print("\n\n" + "█"*60)
print("  MODEL 4: Distilled SFT")
print("█"*60)
for k in range(3, 9):
    run_cmd([
        sys.executable, "scripts/run_multiplication_eval.py", "run",
        f"--model_id=vg10101/qwen3-4b-k3-k6-distilled-sft",
        f"--dataset_path=results/data/prime_multiplication_k{k}_eval.jsonl",
        f"--enable_thinking=True",
        f"--system_prompt={SYSTEM_PROMPT}",
        f"--output_path=results/evals/distilled_k{k}.json",
    ])

# ── Summary ──
print("\n\n" + "█"*60)
print("  SUMMARY")
print("█"*60)

from src import utils

models = [
    ("Base Qwen3-4B (no CoT)", "no_cot_qwen3-4b"),
    ("GRPO model (no CoT)", "no_cot_grpo"),
    ("Cipher SFT", "cipher"),
    ("Distilled SFT", "distilled"),
]

header = f"{'Model':<30s}"
for k in range(3, 9):
    header += f" | k={k:>2d}"
print(header)
print("-" * len(header))

for name, prefix in models:
    row = f"{name:<30s}"
    for k in range(3, 9):
        path = f"results/evals/{prefix}_k{k}.json"
        try:
            data = utils.read_json(path)
            acc = data["summary"]["accuracy"]
            row += f" | {acc*100:>5.1f}%"
        except Exception:
            row += f" |   N/A"
    print(row)
