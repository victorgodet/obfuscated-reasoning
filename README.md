# Encoded Chain-of-Thought

Train Qwen3-0.6B to multiply 3-digit numbers while **penalizing digits in its chain-of-thought**, forcing the model to develop encoded/obfuscated reasoning strategies via Group Relative Policy Optimization (GRPO).

## Hypothesis

Can a language model learn to reason about arithmetic without using digits in its scratchpad? By applying a digit-ratio penalty to the thinking block, we force the model to either:
- Encode intermediate results using non-digit tokens (words, symbols, patterns)
- Develop compressed reasoning that avoids explicit step-by-step digit manipulation
- Find entirely novel representations for numerical computation

## How it works

The model uses Qwen3's native thinking mode (`enable_thinking=True`):

```
<|im_start|>user
What is 123 times 456?<|im_end|>
<|im_start|>assistant
<think>
[free-form reasoning — but digits are penalized in the reward]
</think>
56088<|im_end|>
```

**Reward function:**
```
reward = correctness - digit_penalty_weight * digit_ratio
```
- `correctness`: 1.0 if the answer matches, 0.0 otherwise
- `digit_ratio`: fraction of digit characters (0-9) in the `<think>` block
- `digit_penalty_weight`: default 0.5 — balances accuracy vs digit avoidance

GRPO gradient is applied to **all** generated tokens (thinking + answer), letting the model learn both reasoning strategy and answer formatting.

## Quick start

```bash
# Smoke test (small group, 2 steps)
python train_encoded_cot.py --max-steps 2 --G 2 --quiet

# Full training run
python train_encoded_cot.py --max-steps 200 --G 64

# Resume from latest checkpoint
python train_encoded_cot.py --max-steps 200

# Start fresh, deleting old outputs
python train_encoded_cot.py --max-steps 200 --clean

# Evaluate base model only (2-way: baseline vs untrained CoT)
python eval_encoded_cot.py --num-samples 50

# Evaluate trained checkpoint (3-way comparison)
python eval_encoded_cot.py --model outputs_encoded_cot/final_model --num-samples 100

# Verbose with JSON output
python eval_encoded_cot.py --model outputs_encoded_cot/final_model --verbose --output results.json
```

## Training CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--quiet` | off | Hide generation details |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--lr` | 1e-6 | Learning rate |
| `--max-steps` | 500 | Steps before prompting to continue |
| `--G` | 64 | Group size for GRPO |
| `--temperature` | 0.6 | Sampling temperature |
| `--top-p` | 0.95 | Top-p (nucleus) sampling |
| `--top-k` | 20 | Top-k sampling |
| `--digit-penalty` | 0.5 | Weight for digit ratio penalty in reward |
| `--max-tokens` | 2048 | Max total completion tokens |
| `--max-answer-tokens` | 20 | Max answer tokens after `</think>` |
| `--min-a` | 100 | Min first operand (3-digit) |
| `--max-a` | 999 | Max first operand (3-digit) |
| `--min-b` | 100 | Min second operand (3-digit) |
| `--max-b` | 999 | Max second operand (3-digit) |
| `--save-every` | 100 | Save checkpoint every N steps (0 to disable) |
| `--resume` | none | Path to checkpoint dir to resume from |
| `--fresh` | off | Start fresh, ignoring existing checkpoints |
| `--compile` | off | Use torch.compile() on the policy model |
| `--clean` | off | Delete old training outputs before starting |

## Evaluation CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | auto-detect | Path to trained checkpoint |
| `--base-model` | Qwen/Qwen3-0.6B | Base model name |
| `--device` | auto | Device (cuda/mps/cpu) |
| `--num-samples` | 100 | Number of eval problems |
| `--max-answer-tokens` | 20 | Max answer tokens |
| `--temperature` | 0.6 | Sampling temperature for CoT conditions |
| `--top-p` | 0.95 | Top-p sampling |
| `--top-k` | 20 | Top-k sampling |
| `--min-a` | 100 | Min first operand (3-digit) |
| `--max-a` | 999 | Max first operand (3-digit) |
| `--min-b` | 100 | Min second operand (3-digit) |
| `--max-b` | 999 | Max second operand (3-digit) |
| `--seed` | 42 | Seed for reproducible eval set |
| `--output` | none | Save detailed results to JSON |
| `--verbose` | off | Print each problem's generation |

The eval script runs a **3-way comparison**: (1) baseline with no thinking, (2) untrained CoT on the base model, and (3) trained CoT on the GRPO checkpoint. Metrics include accuracy, average digit ratio in thinking, and average thinking length. If `--model` is omitted, only conditions 1 and 2 are run.

## Key design decisions

- **Digit penalty is character-level** — count digit chars in decoded thinking text, more interpretable than token-level
- **GRPO gradient on all tokens** — lets model learn both reasoning and answer formatting
- **G=64** — H100 80GB can handle large groups with long sequences
- **Flash Attention 2** — enabled automatically on CUDA for faster long-sequence forward passes
- **Temperature 0.6 / top-p 0.95 / top-k 20** — Qwen3's recommended thinking-mode sampling params
- **3-digit x 3-digit** — hard enough that reasoning is needed, tractable for 0.6B model

## RunPod quick start

```bash
# Install dependencies
pip install torch transformers numpy flash-attn --no-build-isolation

# Clone and run
git clone https://github.com/victorgodet/obfuscated-reasoning.git
cd obfuscated-reasoning
python train_encoded_cot.py --max-steps 500 --quiet
```

## Tips

- **Interactive training**: Every `--max-steps` steps, you'll be prompted to continue or stop. Enter 0 to save and exit.
- **Checkpointing**: Checkpoints are saved every `--save-every` steps and never deleted. Training auto-resumes from the latest checkpoint unless `--fresh` is passed.
- **RunPod / remote GPU**: Run with `--quiet` to keep logs manageable over SSH.
- **torch.compile**: Use `--compile` for faster training on supported hardware (adds startup overhead).
