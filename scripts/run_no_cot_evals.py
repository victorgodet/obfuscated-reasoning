"""Run no-CoT evals for both base Qwen3-4B and pre-SFT model on k=3-8."""
import subprocess
import sys

MODELS = [
    ("Qwen/Qwen3-4B", "qwen3-4b"),
    ("vg10101/qwen3-4b-prime-mult-k5-step50", "prime-mult-k5-step50"),
]

for model_id, model_short in MODELS:
    for k in range(3, 9):
        dataset_path = f"results/data/prime_multiplication_k{k}_eval.jsonl"
        output_path = f"results/evals/no_cot_{model_short}_k{k}.json"
        
        print(f"\n{'='*60}")
        print(f"Model: {model_id} | k={k}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, "scripts/eval_no_cot.py", "run",
            f"--model_id={model_id}",
            f"--dataset_path={dataset_path}",
            f"--output_path={output_path}",
        ]
        subprocess.run(cmd, check=True)

# Print summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

from src import utils
for model_id, model_short in MODELS:
    print(f"\n{model_id}:")
    for k in range(3, 9):
        path = f"results/evals/no_cot_{model_short}_k{k}.json"
        try:
            data = utils.read_json(path)
            s = data["summary"]
            print(f"  k={k}: {s['accuracy']:.3f} ({s['n_correct']}/{s['n_problems']})")
        except Exception as e:
            print(f"  k={k}: ERROR - {e}")
