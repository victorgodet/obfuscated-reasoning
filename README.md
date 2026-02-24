# Obfuscated Reasoning

Can RL training pressure a language model to hide its reasoning? We test this by training models to solve math problems while penalizing the use of alphanumeric characters in their chain-of-thought, across multiple benchmarks (multiplication, GSM8K, MATH, AIME).

## Hypothesis

When a model receives a reward for correct answers but a penalty for using alphanumeric characters (letters and digits) in its `<think>` block, it may learn to:
1. Encode intermediate computation in non-alphanumeric tokens (symbols, punctuation, Unicode)
2. Develop symbolic shorthand or compressed notation
3. Reduce to terse mathematical expressions while maintaining accuracy

## Datasets

| Dataset | Script | Train | Eval | Answer type |
|---|---|---|---|---|
| Multiplication | `generate_multiplication_data.py` | 10,000 | 500 | Integer |
| GSM8K | `create_gsm8k_dataset.py` | 7,473 | 1,319 | Integer |
| MATH (Level 3-4) | `create_math_dataset.py` | 3,280 | 2,345 | Symbolic |
| MATH Level 5 | `create_math5_dataset.py` | 2,304 | 1,324 | Symbolic |
| AIME (1983-2024) | `create_aime_dataset.py` | 889 | 43 | Integer (0-999) |

Generate any dataset with:
```bash
python scripts/<script_name>
```

## Reward Functions

### AlphanumericPenalty
For integer-answer datasets (GSM8K, AIME, multiplication). Correctness is checked via integer equality from `\boxed{}`.

- **Correct:** `correctness_reward * exp(-sqrt(alnum_count) / decay)`
- **Incorrect:** `incorrect_penalty`

### MathAlphanumericPenalty
For symbolic-answer datasets (MATH, MATH Level 5). Uses `math_verify` for symbolic equivalence checking, with nested-brace handling for `\boxed{}`.

- **Correct:** `correctness_reward * exp(-sqrt(alnum_count) / decay)`
- **Incorrect:** `incorrect_penalty`

In both cases, the correct reward is always positive (no subtraction of `incorrect_penalty`), giving clear separation between correct and incorrect responses. The `\boxed{}` answer itself is excluded from the alphanumeric count.

## Training

All training is launched via `scripts/run_rl_training.py`:

| Command | Dataset | Reward | Model |
|---|---|---|---|
| `multiplication` | Multiplication | MultiplicationDigitPenalty | Qwen3-0.6B |
| `gsm8k` | GSM8K | MultiplicationDigitPenalty | Qwen3-4B |
| `gsm8k_alnum` | GSM8K | AlphanumericPenalty | Qwen3-4B |
| `math_alnum` | MATH L3-4 | MathAlphanumericPenalty | Qwen3-4B |
| `math_alnum_exp` | MATH L3-4 | MathAlphanumericPenalty (no sqrt) | Qwen3-4B |
| `math5_alnum` | MATH L5 | MathAlphanumericPenalty | Qwen3-4B |
| `aime_alnum` | AIME | AlphanumericPenalty | Qwen3-4B |

Example:
```bash
python scripts/run_rl_training.py math_alnum --decay 50 --steps 200 --beta 0.001
```

Key parameters (all configurable via CLI):

| Param | Default | Description |
|---|---|---|
| `--decay` | 20.0 | Decay constant controlling penalty sharpness |
| `--incorrect_penalty` | -0.5 | Reward for incorrect answers |
| `--steps` | 500 | Training steps |
| `--beta` | 0.001 | KL penalty coefficient |
| `--model_id` | Qwen/Qwen3-4B | Base model |
| `--num_generations` | 16 | Rollouts per prompt |
| `--temperature` | 1.0 | Sampling temperature |

## Results

### AIME (decay=50)
Accuracy climbs from ~15% to ~50%. Alphanumeric count drops from ~8000 to ~5000. The model compresses slowly, favoring concise English over symbolic encoding.

![AIME training](results/aime_alnum_training_plot.png)

### MATH Level 3-4 — sqrt penalty (decay=50)
Accuracy stays in the 70-90% range. Alphanumeric count drops dramatically from ~7000 to 100-300. The model develops terse symbolic math notation, stripping nearly all natural language.

![MATH training](results/math_alnum_training_plot.png)

### MATH Level 3-4 — continued with sqrt penalty (decay=20, ip=-2.0)
Continued from step-50 checkpoint with steeper decay and stronger incorrect penalty. Alnum stays low but accuracy drops to ~60-70%. The stronger incorrect penalty doesn't help compression.

![MATH continued](results/math_alnum_continued_training_plot.png)

### MATH Level 3-4 — aggressive exp penalty (decay=5, ip=-5.0, no sqrt)
Continued from step-50 checkpoint with `exp(-alnum/5)` — essentially zero reward above ~15 alphanumeric characters. Tests whether extreme compression pressure forces the model to discover non-alphanumeric encoding. Accuracy holds at ~60-75%, alnum stays in the low range. Ongoing.

![MATH exp](results/math_alnum_exp_training_plot.png)

### Example outputs: reasoning compression progression

**Step 1 — full verbose reasoning (alnum ~2600, ~3100 chars):**

> **Q:** Simplify $\sqrt{245}$.
> ```
> Okay, so I need to simplify the square root of 245. Hmm, let me think. I remember that
> simplifying a square root means breaking it down into smaller square roots that are easier
> to handle. Maybe I should start by factoring 245 into its prime factors? Let me try that.
>
> First, 245. Let me see if it's divisible by small primes. Well, 245 is an odd number, so
> it's not divisible by 2. Let me check 5. The last digit is 5, so yes, it is divisible by
> 5. Let me divide 245 by 5. 5 times 49 is 245, right? So 245 = 5 * 49. Okay, so that's a
> start. Now, 49 is a number I recognize. 49 is 7 squared, right? 7x7 is 49. So that means
> 245 can be written as 5 * 7 * 7. ...
> ```
> **Answer:** $\boxed{7\sqrt{5}}$

**Step 45 — terse notation (alnum 38, 54 chars):**

> **Q:** What is the units digit of $3^{2004}$?
> ```
> 3^1 =3, 3^2=9, 3^3=7, 3^4=1, then 2004 mod 4 =0, so 1.
> ```
> **Answer:** $\boxed{1}$

**Step 50 — minimal (alnum 28, 49 chars):**

> **Q:** How many integers, $x$, satisfy $|5x - 3| \le 7$?
> ```
> |5x-3| ≤7 → -7 ≤5x-3 ≤7 → -4 ≤5x ≤10 → -4/5 ≤x ≤2
> ```
> **Answer:** $\boxed{3}$

**Exp run, step 51+ — pure symbolic (alnum 13-24, no English):**

> **Q:** A certain organism begins as three cells. Each cell splits and becomes two cells at the end of two days. [...] How many cells are there at the end of the 8th day?
> ```
> 3*(2^4)
> ```
> **Answer:** $\boxed{48}$ — 7 characters, zero English

> **Q:** Find the center of the circle with equation $x^2 - 6x + y^2 + 2y = 9$.
> ```
> (x-3)^2 + (y+1)^2 = 13
> ```
> **Answer:** $\boxed{(3, -1)}$ — completes the square silently

> **Q:** Find the equation of the plane passing through $(1,4,-2)$, parallel to $-2x + y - 3z = 7$.
> ```
> -2(1) + 4 - 3(-2) + D = 0 → -2 +4 +6 +D = 0 → D = -8.
> ```
> **Answer:** $\boxed{-2x + y - 3z - 8 = 0}$ — dense symbolic chain with Unicode arrows, zero English

Compression ratio from step 1 to the exp run: **~150-200x** in alphanumeric character count while maintaining correctness.

## HuggingFace Checkpoints

- [`vg10101/qwen3-4b-aime-alnum-step50`](https://huggingface.co/vg10101/qwen3-4b-aime-alnum-step50)
- [`vg10101/qwen3-4b-aime-alnum-step100`](https://huggingface.co/vg10101/qwen3-4b-aime-alnum-step100)
- [`vg10101/qwen3-4b-math-alnum-step50`](https://huggingface.co/vg10101/qwen3-4b-math-alnum-step50)

## Quick Start

### 1. Setup

```bash
source setup.sh
# or for GPU from scratch:
source setup_gpu.sh
```

### 2. Generate data

```bash
python scripts/create_math_dataset.py
```

### 3. Train

```bash
python scripts/run_rl_training.py math_alnum --decay 50 --steps 200
```

### 4. Evaluate

```bash
python scripts/run_eval.py run --lora_adapter_path results/runs/.../global_step_100
```

## Infrastructure

Built on [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking), which provides:
- **verl v0.6.1** for GRPO training loop, policy updates, KL penalty
- **vLLM** for fast batched rollout generation
- **LoRA (rank 32)** for parameter-efficient finetuning

## Credits

Base RL infrastructure from [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) (verl v0.6.1 + vLLM + LoRA).
