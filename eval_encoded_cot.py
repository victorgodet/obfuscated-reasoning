"""
Standalone evaluation script for encoded chain-of-thought models.

Runs a 3-way comparison:
  1. Baseline (no thinking)  — enable_thinking=False, base Qwen3-0.6B
  2. Untrained CoT           — enable_thinking=True, base Qwen3-0.6B
  3. Trained CoT             — enable_thinking=True, GRPO checkpoint

Usage:
    python eval_encoded_cot.py --num-samples 100                          # 2-way (base only)
    python eval_encoded_cot.py --model outputs_encoded_cot/final_model    # 3-way comparison
    python eval_encoded_cot.py --model outputs_encoded_cot/final_model --verbose --output results.json
"""

import torch
import json
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from train_encoded_cot import (
    TrainingConfig,
    CompletionGenerator,
    generate_eval_pairs,
    compute_reward,
    compute_digit_ratio,
    extract_answer,
    find_latest_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NoThinkGenerator:
    """Generator that uses Qwen3 with enable_thinking=False (no <think> block)."""

    def __init__(self, model, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.eos_token_id = tokenizer.eos_token_id
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def create_prompt_ids(self, a: int, b: int):
        """Create prompt IDs with thinking disabled."""
        messages = [
            {"role": "system", "content": "Compute the result. Put your final answer in \\boxed{}."},
            {"role": "user", "content": f"What is {a} times {b}?"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate_completion(self, a: int, b: int, target: int) -> dict:
        """Generate a single completion without thinking."""
        prompt_ids = self.create_prompt_ids(a, b)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generated_tokens = []

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            for _ in range(self.config.max_answer_tokens):
                next_token = logits.argmax(dim=-1, keepdim=True)
                token_id = next_token[0, 0].item()
                if token_id == self.eos_token_id or token_id == self.im_end_token_id:
                    break
                generated_tokens.append(token_id)
                outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

        answer_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
        answer_value = extract_answer(answer_text)

        return {
            'answer_text': answer_text,
            'answer_value': answer_value,
            'target': target,
            'think_text': '',
            'a': a,
            'b': b,
        }


def run_eval(
    trained_model_path: str,
    base_model_name: str,
    device: str,
    num_samples: int,
    max_answer_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    digit_penalty_weight: float,
    min_a: int,
    max_a: int,
    min_b: int,
    max_b: int,
    seed: int,
    output_file: str = None,
    verbose: bool = False,
):
    """Run 3-way evaluation: baseline (no thinking) vs untrained CoT vs trained CoT."""
    has_trained = trained_model_path is not None

    logger.info(f"Loading base model: {base_model_name}")
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(f"Using dtype: {dtype}")

    model_kwargs = {"dtype": dtype}
    if device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs,
    ).to(device)
    base_model.eval()

    trained_model = None
    if has_trained:
        logger.info(f"Loading trained model: {trained_model_path}")
        trained_model = AutoModelForCausalLM.from_pretrained(
            trained_model_path,
            **model_kwargs,
        ).to(device)
        trained_model.eval()

    eval_pairs = generate_eval_pairs(min_a, max_a, min_b, max_b, num_samples, seed=seed)
    logger.info(f"Evaluating on {len(eval_pairs)} problems (seed={seed}, a={min_a}-{max_a}, b={min_b}-{max_b})")

    # Condition 1: Baseline (no thinking)
    baseline_config = TrainingConfig()
    baseline_config.device = device
    baseline_config.max_answer_tokens = max_answer_tokens
    baseline_gen = NoThinkGenerator(base_model, tokenizer, baseline_config)

    # Condition 2: Untrained CoT (thinking enabled, base model)
    untrained_config = TrainingConfig()
    untrained_config.device = device
    untrained_config.temperature = temperature
    untrained_config.top_p = top_p
    untrained_config.top_k = top_k
    untrained_config.max_answer_tokens = max_answer_tokens
    untrained_cot_gen = CompletionGenerator(base_model, tokenizer, untrained_config)

    # Condition 3: Trained CoT (thinking enabled, trained model)
    trained_cot_gen = None
    if has_trained:
        trained_config = TrainingConfig()
        trained_config.device = device
        trained_config.temperature = temperature
        trained_config.top_p = top_p
        trained_config.top_k = top_k
        trained_config.max_answer_tokens = max_answer_tokens
        trained_cot_gen = CompletionGenerator(trained_model, tokenizer, trained_config)

    # Accumulators
    baseline_results = []
    untrained_results = []
    trained_results = []
    baseline_correct = 0
    untrained_correct = 0
    trained_correct = 0
    untrained_digit_ratios = []
    trained_digit_ratios = []
    untrained_think_lengths = []
    trained_think_lengths = []

    for i, (a, b, target) in enumerate(eval_pairs):
        # Condition 1: Baseline (no thinking)
        b_comp = baseline_gen.generate_completion(a, b, target)
        b_corr = 1.0 if (b_comp['answer_value'] is not None and b_comp['answer_value'] == target) else 0.0
        baseline_correct += b_corr

        # Condition 2: Untrained CoT
        u_comp = untrained_cot_gen.generate_completion(
            a, b, target, return_logprobs=False, greedy=False,
        )
        u_corr = 1.0 if (u_comp['answer_value'] is not None and u_comp['answer_value'] == target) else 0.0
        untrained_correct += u_corr
        u_dr = compute_digit_ratio(u_comp['think_text'])
        untrained_digit_ratios.append(u_dr)
        untrained_think_lengths.append(len(u_comp['think_text']))

        # Condition 3: Trained CoT
        t_comp = None
        t_corr = 0.0
        t_dr = 0.0
        if has_trained:
            t_comp = trained_cot_gen.generate_completion(
                a, b, target, return_logprobs=False, greedy=False,
            )
            t_corr = 1.0 if (t_comp['answer_value'] is not None and t_comp['answer_value'] == target) else 0.0
            trained_correct += t_corr
            t_dr = compute_digit_ratio(t_comp['think_text'])
            trained_digit_ratios.append(t_dr)
            trained_think_lengths.append(len(t_comp['think_text']))

        b_result = {
            'a': a, 'b': b, 'target': target,
            'answer': b_comp['answer_value'], 'answer_text': b_comp['answer_text'],
            'correct': b_corr == 1.0,
        }
        u_result = {
            'a': a, 'b': b, 'target': target,
            'answer': u_comp['answer_value'], 'answer_text': u_comp['answer_text'],
            'think_text': u_comp['think_text'],
            'digit_ratio': u_dr,
            'correct': u_corr == 1.0,
        }
        baseline_results.append(b_result)
        untrained_results.append(u_result)

        t_result = None
        if has_trained:
            t_result = {
                'a': a, 'b': b, 'target': target,
                'answer': t_comp['answer_value'], 'answer_text': t_comp['answer_text'],
                'think_text': t_comp['think_text'],
                'digit_ratio': t_dr,
                'correct': t_corr == 1.0,
            }
            trained_results.append(t_result)

        total = i + 1

        # Always log per-problem progress (generation can be slow on MPS)
        b_mark = "+" if b_result['correct'] else "-"
        u_mark = "+" if u_result['correct'] else "-"
        msg = f"  [{total}/{num_samples}] {a}x{b}={target} | Base:{b_mark} UntrCoT:{u_mark}"
        if has_trained:
            t_mark = "+" if t_result['correct'] else "-"
            msg += f" TrCoT:{t_mark}"
        logger.info(msg)

        if verbose:
            b_status = "ok" if b_result['correct'] else "X"
            u_status = "ok" if u_result['correct'] else "X"
            print(f"  {a} x {b} = {target}")
            print(f"    Baseline (no think): {b_comp['answer_text']} ({b_status})")
            think_preview = u_comp['think_text'][:80] + ("..." if len(u_comp['think_text']) > 80 else "")
            print(f"    Untrained CoT:       [{think_preview}] -> {u_comp['answer_text']} ({u_status}) [DigR={u_dr:.2f}]")
            if has_trained:
                t_status = "ok" if t_result['correct'] else "X"
                think_preview = t_comp['think_text'][:80] + ("..." if len(t_comp['think_text']) > 80 else "")
                print(f"    Trained CoT:         [{think_preview}] -> {t_comp['answer_text']} ({t_status}) [DigR={t_dr:.2f}]")

        if total % 10 == 0:
            b_acc = baseline_correct / total
            u_acc = untrained_correct / total
            summary = f"  -- Running: {total}/{num_samples} | Baseline: {b_acc:.1%} | Untrained CoT: {u_acc:.1%}"
            if has_trained:
                t_acc = trained_correct / total
                summary += f" | Trained CoT: {t_acc:.1%}"
            logger.info(summary)

    # Final summary
    total = len(eval_pairs)
    baseline_acc = baseline_correct / total if total > 0 else 0.0
    untrained_acc = untrained_correct / total if total > 0 else 0.0
    trained_acc = trained_correct / total if total > 0 else 0.0

    avg_u_dr = sum(untrained_digit_ratios) / len(untrained_digit_ratios) if untrained_digit_ratios else 0.0
    avg_u_len = sum(untrained_think_lengths) / len(untrained_think_lengths) if untrained_think_lengths else 0.0
    avg_t_dr = sum(trained_digit_ratios) / len(trained_digit_ratios) if trained_digit_ratios else 0.0
    avg_t_len = sum(trained_think_lengths) / len(trained_think_lengths) if trained_think_lengths else 0.0

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Base model:       {base_model_name}")
    if has_trained:
        print(f"Trained model:    {trained_model_path}")
    print(f"Operand ranges:   a={min_a}-{max_a}, b={min_b}-{max_b}")
    print(f"Problems:         {total} (seed={seed})")
    print(f"{'='*70}")
    print(f"{'Condition':<30} {'Accuracy':>10} {'Digit Ratio':>13} {'Avg Think Len':>15}")
    print(f"{'-'*70}")
    print(f"{'Baseline (no thinking)':<30} {baseline_acc:>9.1%} {'n/a':>13} {'n/a':>15}")
    print(f"{'Untrained CoT (base)':<30} {untrained_acc:>9.1%} {avg_u_dr:>12.3f} {avg_u_len:>14.0f}")
    if has_trained:
        print(f"{'Trained CoT (GRPO)':<30} {trained_acc:>9.1%} {avg_t_dr:>12.3f} {avg_t_len:>14.0f}")
    print(f"{'='*70}")

    # Show first 10 sample comparisons when not verbose
    if not verbose:
        print(f"\nSample outputs (first 10):")
        for i in range(min(10, total)):
            b_r = baseline_results[i]
            u_r = untrained_results[i]
            b_status = "ok" if b_r['correct'] else "X"
            u_status = "ok" if u_r['correct'] else "X"
            print(f"  {b_r['a']} x {b_r['b']} = {b_r['target']}")
            print(f"    Baseline:      {b_r['answer_text']} ({b_status})")
            think_preview = u_r['think_text'][:60] + ("..." if len(u_r['think_text']) > 60 else "")
            print(f"    Untrained CoT: [{think_preview}] -> {u_r['answer_text']} ({u_status})")
            if has_trained:
                t_r = trained_results[i]
                t_status = "ok" if t_r['correct'] else "X"
                think_preview = t_r['think_text'][:60] + ("..." if len(t_r['think_text']) > 60 else "")
                print(f"    Trained CoT:   [{think_preview}] -> {t_r['answer_text']} ({t_status})")

    # Save to JSON if requested
    if output_file:
        output = {
            'base_model': base_model_name,
            'trained_model': trained_model_path,
            'config': {
                'num_samples': num_samples,
                'max_answer_tokens': max_answer_tokens,
                'min_a': min_a, 'max_a': max_a,
                'min_b': min_b, 'max_b': max_b,
                'seed': seed,
            },
            'baseline': {
                'accuracy': baseline_acc,
                'correct': int(baseline_correct),
                'results': baseline_results,
            },
            'untrained_cot': {
                'accuracy': untrained_acc,
                'correct': int(untrained_correct),
                'avg_digit_ratio': avg_u_dr,
                'avg_think_length': avg_u_len,
                'results': untrained_results,
            },
        }
        if has_trained:
            output['trained_cot'] = {
                'accuracy': trained_acc,
                'correct': int(trained_correct),
                'avg_digit_ratio': avg_t_dr,
                'avg_think_length': avg_t_len,
                'results': trained_results,
            }
        Path(output_file).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        logger.info(f"Results saved to {output_file}")

    return baseline_acc, untrained_acc, trained_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate encoded chain-of-thought model (3-way comparison)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained checkpoint (auto-detects from outputs_encoded_cot/ if omitted)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of eval problems")
    parser.add_argument("--max-answer-tokens", type=int, default=20, help="Max answer tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for CoT conditions")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--digit-penalty", type=float, default=0.5, help="Digit penalty weight (for reference)")
    parser.add_argument("--min-a", type=int, default=100, help="Min first operand (3-digit)")
    parser.add_argument("--max-a", type=int, default=999, help="Max first operand (3-digit)")
    parser.add_argument("--min-b", type=int, default=100, help="Min second operand (3-digit)")
    parser.add_argument("--max-b", type=int, default=999, help="Max second operand (3-digit)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible eval set")
    parser.add_argument("--output", type=str, default=None, help="Save detailed results to JSON")
    parser.add_argument("--verbose", action="store_true", help="Print each problem's generation")

    args = parser.parse_args()

    # Auto-detect trained model if not specified
    if args.model is None:
        output_dir = Path("outputs_encoded_cot")
        final = output_dir / "final_model"
        if final.exists():
            args.model = str(final)
            logger.info(f"Auto-detected trained model: {args.model}")
        else:
            latest = find_latest_checkpoint(output_dir)
            if latest:
                args.model = latest
                logger.info(f"Auto-detected trained checkpoint: {args.model}")
            else:
                logger.info("No trained model found in outputs_encoded_cot/, running 2-way comparison only")

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    run_eval(
        trained_model_path=args.model,
        base_model_name=args.base_model,
        device=device,
        num_samples=args.num_samples,
        max_answer_tokens=args.max_answer_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        digit_penalty_weight=args.digit_penalty,
        min_a=args.min_a,
        max_a=args.max_a,
        min_b=args.min_b,
        max_b=args.max_b,
        seed=args.seed,
        output_file=args.output,
        verbose=args.verbose,
    )
