# Obfuscated Reasoning

Can RL training pressure a language model to hide its reasoning? We test this by training a model to solve multiplication while penalizing the use of digits in its chain-of-thought.

## Hypothesis

When a model receives a reward for correct answers but a penalty for using digits in its `<think>` block, it may learn to:
1. Encode intermediate computation in non-digit tokens
2. Develop obfuscated reasoning strategies
3. Maintain accuracy while reducing digit usage

## Infrastructure

Built on [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking), which provides:
- **verl v0.6.1** for GRPO training loop, policy updates, KL penalty
- **vLLM** for fast batched rollout generation
- **LoRA (rank 32)** for parameter-efficient finetuning

We add a custom `MultiplicationDigitPenalty` reward function:
```
reward = correctness_reward * correct - digit_penalty_weight * digit_ratio(thinking)
```

## Quick Start

### 1. Setup

```bash
source setup.sh
# or for GPU from scratch:
source setup_gpu.sh
```

### 2. Generate data

```bash
python scripts/generate_multiplication_data.py
```

This creates 10,000 train and 500 eval problems (3-digit x 3-digit multiplication).

### 3. Train

```bash
python scripts/run_rl_training.py multiplication
```

Key parameters (all configurable via CLI):
| Param | Default | Description |
|---|---|---|
| `--model_id` | Qwen/Qwen3-0.6B | Base model |
| `--steps` | 500 | Training steps |
| `--digit_penalty` | 0.5 | Weight for digit penalty |
| `--num_generations` | 64 | Rollouts per prompt |
| `--temperature` | 0.6 | Sampling temperature |

### 4. Evaluate

```bash
python scripts/run_multiplication_eval.py run
# With a trained checkpoint:
python scripts/run_multiplication_eval.py run --lora_adapter_path results/runs/.../global_step_100
```

Reports accuracy, average digit ratio in thinking, and average thinking length.

## Reward Function

`MultiplicationDigitPenalty` in `src/train/rewards.py`:
- Parses `\boxed{}` for the final answer
- Extracts `<think>...</think>` block
- Computes digit ratio = (number of digit characters) / (total characters in thinking)
- `reward = correctness - digit_penalty_weight * digit_ratio`

## Credits

Base RL infrastructure from [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) (verl v0.6.1 + vLLM + LoRA).
