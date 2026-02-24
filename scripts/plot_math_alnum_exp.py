"""Plot MATH alnum exp training: original run (steps 1-50) + exp run (steps 51+)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = Path("results")
ORIGINAL_RUN = RESULTS / "runs/qwen3-4b/20260224_202020_math_alnum_d50_ip-0.5/rollouts"
EXP_RUN = RESULTS / "runs/qwen3-4b-math-alnum-step50/20260224_231514_math_alnum_exp_d5_ip-5.0/rollouts"
OUTPUT_PATH = RESULTS / "math_alnum_exp_training_plot.png"


def load_rollout_stats(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    n = len(rows)
    correct = [r for r in rows if r["correctness"] == 1.0]
    return {
        "accuracy": len(correct) / n * 100,
        "alnum_count": sum(r["alnum_count"] for r in rows) / n,
    }


def moving_avg(values, window=5):
    return np.convolve(values, np.ones(window) / window, mode="valid")


# Load original run steps 1-50
steps_orig = []
acc_orig = []
alnum_orig = []
for i in range(1, 51):
    p = ORIGINAL_RUN / f"{i}.jsonl"
    if not p.exists():
        break
    stats = load_rollout_stats(p)
    steps_orig.append(i)
    acc_orig.append(stats["accuracy"])
    alnum_orig.append(stats["alnum_count"])

# Load exp run (offset steps by 50)
steps_exp = []
acc_exp = []
alnum_exp = []
i = 1
while True:
    p = EXP_RUN / f"{i}.jsonl"
    if not p.exists():
        break
    stats = load_rollout_stats(p)
    steps_exp.append(50 + i)
    acc_exp.append(stats["accuracy"])
    alnum_exp.append(stats["alnum_count"])
    i += 1

# Combine
all_steps = steps_orig + steps_exp
all_acc = acc_orig + acc_exp
all_alnum = alnum_orig + alnum_exp

# Smoothed
window = 5
smooth_steps = all_steps[window - 1 :]
smooth_acc = moving_avg(all_acc, window)
smooth_alnum = moving_avg(all_alnum, window)

# Plot
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

# Raw (faint)
ax1.plot(all_steps, all_acc, color="dodgerblue", alpha=0.2, linewidth=0.8)
ax2.plot(all_steps, all_alnum, color="orangered", alpha=0.2, linewidth=0.8)

# Smoothed
ax1.plot(smooth_steps, smooth_acc, color="dodgerblue", linewidth=2, label="Accuracy (5-step avg)")
ax2.plot(smooth_steps, smooth_alnum, color="orangered", linewidth=2, label="Alnum count (5-step avg)")

# Phase boundary
ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.6, linewidth=1)
ax1.text(50.5, 5, "sqrt, decay=50, ip=-0.5 → exp, decay=5, ip=-5.0", fontsize=8, color="gray", va="bottom")

ax1.set_xlabel("Training Step")
ax1.set_ylabel("Accuracy (%)", color="dodgerblue")
ax2.set_ylabel("Avg Alphanumeric Count", color="orangered")
ax1.set_ylim(0, 100)
ax1.tick_params(axis="y", labelcolor="dodgerblue")
ax2.tick_params(axis="y", labelcolor="orangered")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_title("MATH (Level 3-4) — Aggressive Exp Penalty from Step 50 Checkpoint")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved to {OUTPUT_PATH}")
print(f"Original: {len(steps_orig)} steps, Exp: {len(steps_exp)} steps, Total: {len(all_steps)}")
