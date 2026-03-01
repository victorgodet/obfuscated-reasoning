#!/bin/bash
# Usage: bash scripts/watch_cots.sh [run_dir]
# Watches rollouts and prints 5 random correct CoTs per new step
# Auto-detects new runs when the latest run directory changes

FIXED_DIR="${1:-}"
CURRENT_RUN=""
LAST_STEP=0

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
    fi

    python3 -c "
import json, os, glob, random

run_dir = '$RUN_DIR'
last_step = $LAST_STEP
files = sorted(glob.glob(f'{run_dir}/*.jsonl'), key=lambda f: int(os.path.basename(f).replace('.jsonl','')))

for f in files:
    step = int(os.path.basename(f).replace('.jsonl',''))
    if step <= last_step:
        continue
    lines = [json.loads(l) for l in open(f)]
    correct = [l for l in lines if l.get('correctness', 0) > 0]
    n_total = len(lines)
    n_correct = len(correct)

    print(f'===== Step {step} ({n_correct}/{n_total} correct) =====')
    print()

    if not correct:
        print('  No correct answers this step.')
        print()
    else:
        samples = random.sample(correct, min(5, len(correct)))
        for ex in samples:
            inp = ex.get('input', '')
            q_start = inp.find('What is')
            q_end = inp.find('?', q_start) + 1
            question = inp[q_start:q_end] if q_start >= 0 else '?'
            gt = ex.get('gts', '?')
            alnum = ex.get('alnum_count', 0)
            score = ex.get('score', 0)
            response = ex.get('response', ex.get('output', ''))
            if len(response) > 1000:
                response = response[:1000] + '\n... [truncated]'
            print(f'  Q: {question} (gt={gt})')
            print(f'  Alnum: {alnum}, Reward: {score:.3f}')
            print(f'  {response}')
            print()

    print(step, file=open('/tmp/_watch_cots_last_step', 'w'))
"
    if [ -f /tmp/_watch_cots_last_step ]; then
        LAST_STEP=$(cat /tmp/_watch_cots_last_step)
    fi
    sleep 10
done
