#!/bin/bash
# Usage: bash scripts/watch_metrics.sh [run_dir]
# Watches rollout metrics and prints one new line per step

RUN_DIR="${1:-$(ls -dt results/runs/*/*/rollouts 2>/dev/null | head -1)}"
echo "Watching: $RUN_DIR"
echo ""
printf "%6s %7s %8s %10s %8s\n" "Step" "Acc" "Alnum" "ThinkLen" "Reward"
echo "---------------------------------------------"

LAST_STEP=0

while true; do
    python3 -c "
import json, os, glob

run_dir = '$RUN_DIR'
last_step = $LAST_STEP
files = sorted(glob.glob(f'{run_dir}/*.jsonl'), key=lambda f: int(os.path.basename(f).replace('.jsonl','')))

for f in files:
    step = int(os.path.basename(f).replace('.jsonl',''))
    if step <= last_step:
        continue
    lines = [json.loads(l) for l in open(f)]
    n = len(lines)
    correct = sum(1 for l in lines if l.get('correctness', 0) > 0)
    avg_alnum = sum(l.get('alnum_count', 0) for l in lines) / n
    avg_think = sum(l.get('think_length', 0) for l in lines) / n
    avg_reward = sum(l.get('score', 0) for l in lines) / n
    print(f'{step:>6} {correct/n*100:>6.1f}% {avg_alnum:>8.0f} {avg_think:>10.0f} {avg_reward:>8.3f}')
    print(step, file=open('/tmp/_watch_last_step', 'w'))
"
    if [ -f /tmp/_watch_last_step ]; then
        LAST_STEP=$(cat /tmp/_watch_last_step)
    fi
    sleep 10
done
