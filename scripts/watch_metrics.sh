#!/bin/bash
# Usage: bash scripts/watch_metrics.sh [run_dir]
# Watches rollout metrics and prints one new line per step
# Auto-detects new runs when the latest run directory changes

FIXED_DIR="${1:-}"
CURRENT_RUN=""
LAST_STEP=0

print_header() {
    printf "%6s %7s %8s %10s %8s\n" "Step" "Acc" "Alnum" "ThinkLen" "Reward"
    echo "---------------------------------------------"
}

while true; do
    if [ -n "$FIXED_DIR" ]; then
        RUN_DIR="$FIXED_DIR"
    else
        RUN_DIR="$(ls -dt results/runs/*/*/rollouts 2>/dev/null | head -1)"
    fi

    if [ -z "$RUN_DIR" ]; then
        sleep 10
        continue
    fi

    # Detect new run
    if [ "$RUN_DIR" != "$CURRENT_RUN" ]; then
        CURRENT_RUN="$RUN_DIR"
        LAST_STEP=0
        echo ""
        echo "=========================================="
        echo "New run detected: $RUN_DIR"
        echo "=========================================="
        echo ""
        print_header
    fi

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
