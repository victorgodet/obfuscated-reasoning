"""
RL training for Qwen3-0.6B to learn 3-digit multiplication using GRPO,
with a digit penalty that forces the model to develop encoded/obfuscated
chain-of-thought reasoning.

Three-phase generation (using Qwen3's native thinking mode):
1. Thinking — free-form autoregressive until </think> is produced
2. Forced prefix — "\\nThe answer is \\boxed{" (injected, excluded from policy gradient)
3. Answer — sampled tokens until } is produced

The reward is:
  correctness (1.0 if right, else 0.0)
  minus digit_penalty_weight * (fraction of digit characters in thinking)

GRPO gradient is applied to thinking + answer tokens (not the forced prefix).
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple
import logging
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_memory(device: str):
    """Clear GPU/MPS memory cache"""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    batch_size: int = 4
    G: int = 64  # Group size for GRPO
    learning_rate: float = 1e-6
    max_steps: int = 500  # Steps before prompting to continue
    max_completion_tokens: int = 2048  # Total completion budget
    max_think_tokens: int = 1800  # Force </think> if thinking exceeds this
    max_answer_tokens: int = 20  # Max tokens after forced prefix
    temperature: float = 0.6  # Qwen3 recommended for thinking
    top_p: float = 0.95
    top_k: int = 20
    digit_penalty_weight: float = 0.5  # Weight for digit ratio penalty
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    log_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    output_dir: str = "outputs_encoded_cot"
    verbose: bool = True
    min_a: int = 100  # Min first operand (3-digit)
    max_a: int = 999  # Max first operand (3-digit)
    min_b: int = 100  # Min second operand (3-digit)
    max_b: int = 999  # Max second operand (3-digit)
    save_every: int = 100  # Save checkpoint every N steps
    resume_from: str = None  # Path to checkpoint dir to resume from
    compile: bool = False  # Use torch.compile() on the policy model


def save_checkpoint(path: Path, model, tokenizer, optimizer, training_state: dict):
    """Save model, tokenizer, optimizer, and training state to a directory."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    (path / "training_state.json").write_text(json.dumps(training_state, indent=2))
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path, optimizer, device: str) -> dict:
    """Load optimizer state and training state from a checkpoint directory."""
    opt_path = path / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
        logger.info(f"Loaded optimizer state from {opt_path}")

    state_path = path / "training_state.json"
    if state_path.exists():
        training_state = json.loads(state_path.read_text())
        logger.info(f"Loaded training state from {state_path} (step {training_state.get('global_step', '?')})")
        return training_state

    return {}


def find_latest_checkpoint(output_dir: Path) -> str:
    """Find the latest checkpoint_step_* dir in output_dir. Returns path string or None."""
    if not output_dir.exists():
        return None
    ckpts = sorted(output_dir.glob("checkpoint_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    if ckpts:
        return str(ckpts[-1])
    return None


def sample_train_pair(min_a: int, max_a: int, min_b: int, max_b: int) -> Tuple[int, int, int]:
    """Return a random (a, b, product) triple."""
    a = np.random.randint(min_a, max_a + 1)
    b = np.random.randint(min_b, max_b + 1)
    return a, b, a * b


def generate_eval_pairs(min_a: int, max_a: int, min_b: int, max_b: int, count: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate a fixed evaluation set of (a, b, product) triples."""
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(count):
        a = int(rng.randint(min_a, max_a + 1))
        b = int(rng.randint(min_b, max_b + 1))
        pairs.append((a, b, a * b))
    return pairs


def count_digit_chars(text: str) -> int:
    """Count digit characters (0-9) in a string."""
    return sum(1 for ch in text if ch.isdigit())


def compute_digit_ratio(text: str) -> float:
    """Compute the fraction of characters that are digits."""
    if len(text) == 0:
        return 0.0
    return count_digit_chars(text) / len(text)


def extract_answer(text: str) -> int:
    """Extract integer from answer text (digits between \\boxed{ and })."""
    # Strip everything non-digit and parse
    digits = re.sub(r'[^\d]', '', text)
    if digits:
        try:
            return int(digits)
        except (ValueError, OverflowError):
            return None
    return None


class CompletionGenerator:
    """Three-phase generation: free-form thinking → forced prefix → sampled answer."""

    def __init__(self, model, tokenizer, config: TrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device

        # Key token IDs
        self.eos_token_id = tokenizer.eos_token_id
        self.end_think_token_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Answer prefix (forced after </think>)
        self.answer_prefix = "\nThe answer is \\boxed{"
        self.answer_prefix_ids = tokenizer.encode(self.answer_prefix, add_special_tokens=False)
        self.close_brace_token_id = tokenizer.encode("}", add_special_tokens=False)[0]

        logger.info(f"EOS token ID: {self.eos_token_id}")
        logger.info(f"</think> token IDs: {self.end_think_token_ids}")
        logger.info(f"<|im_end|> token ID: {self.im_end_token_id}")
        logger.info(f"Answer prefix ({len(self.answer_prefix_ids)} tokens): {self.answer_prefix_ids}")
        logger.info(f"Close brace token ID: {self.close_brace_token_id}")

    def create_prompt_ids(self, a: int, b: int) -> List[int]:
        """Create prompt token IDs using Qwen3 chat template with thinking enabled."""
        messages = [{"role": "user", "content": f"What is {a} times {b}?"}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits using temperature + top-k + top-p."""
        logits = logits / self.config.temperature

        # Top-k filtering
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(indices_to_remove, -float('inf'))

        # Top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[:, 0] = False
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float('inf'))

        probs = F.softmax(logits, dim=-1)
        bad = torch.isnan(probs).any(dim=-1) | (probs.sum(dim=-1) < 0.5)
        if bad.any():
            uniform = torch.ones_like(probs[0]) / probs.size(-1)
            probs[bad] = uniform

        return torch.multinomial(probs, num_samples=1)  # (batch, 1)

    def _is_stop_token(self, token_id: int) -> bool:
        """Check if token is a stop token."""
        return token_id == self.eos_token_id or token_id == self.im_end_token_id

    def generate_completion(
        self,
        a: int,
        b: int,
        target: int,
        return_logprobs: bool = True,
        greedy: bool = False,
    ) -> Dict:
        """
        Generate a single completion. Three phases:
        1. Free-form thinking until </think>
        2. Force prefix through KV cache
        3. Sample answer until } or max tokens
        """
        prompt_ids = self.create_prompt_ids(a, b)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generated_tokens = []
        generated_logprobs = []
        gradient_mask = []  # True = trainable, False = forced prefix

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Phase 1: Free-form thinking
            think_finished = False
            think_token_count = 0
            for step in range(self.config.max_completion_tokens):
                # Safety valve: force </think> if thinking is too long
                if think_token_count >= self.config.max_think_tokens:
                    for force_tid in self.end_think_token_ids:
                        generated_tokens.append(force_tid)
                        if return_logprobs:
                            generated_logprobs.append(0.0)
                        gradient_mask.append(False)
                        next_token = torch.tensor([[force_tid]], device=self.device)
                        outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                        past_kv = outputs.past_key_values
                        logits = outputs.logits[:, -1, :]
                    think_finished = True
                    break

                if greedy:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token = self._sample_token(logits)

                token_id = next_token[0, 0].item()

                if return_logprobs:
                    lp = F.log_softmax(logits, dim=-1)[0, token_id].item()
                    if np.isnan(lp) or np.isinf(lp):
                        lp = -10.0
                    generated_logprobs.append(lp)

                generated_tokens.append(token_id)
                gradient_mask.append(True)
                think_token_count += 1

                if self._is_stop_token(token_id):
                    break

                # Feed token
                outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                # Check for </think>
                n = len(self.end_think_token_ids)
                if len(generated_tokens) >= n and generated_tokens[-n:] == self.end_think_token_ids:
                    think_finished = True
                    break

            if not think_finished:
                return self._build_result(
                    prompt_ids, generated_tokens, generated_logprobs,
                    gradient_mask, a, b, target, return_logprobs,
                )

            # Phase 2: Force answer prefix through KV cache
            for prefix_tid in self.answer_prefix_ids:
                generated_tokens.append(prefix_tid)
                if return_logprobs:
                    generated_logprobs.append(0.0)  # Forced, not sampled
                gradient_mask.append(False)

                next_token = torch.tensor([[prefix_tid]], device=self.device)
                outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

            # Phase 3: Sample answer tokens until } or max
            for step in range(self.config.max_answer_tokens):
                if greedy:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token = self._sample_token(logits)

                token_id = next_token[0, 0].item()

                if return_logprobs:
                    lp = F.log_softmax(logits, dim=-1)[0, token_id].item()
                    if np.isnan(lp) or np.isinf(lp):
                        lp = -10.0
                    generated_logprobs.append(lp)

                generated_tokens.append(token_id)
                gradient_mask.append(True)

                if token_id == self.close_brace_token_id or self._is_stop_token(token_id):
                    break

                outputs = self.model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

        return self._build_result(
            prompt_ids, generated_tokens, generated_logprobs,
            gradient_mask, a, b, target, return_logprobs,
        )

    def generate_completions_batch(self, a: int, b: int, target: int, G: int) -> List[Dict]:
        """
        Generate G completions in parallel. Per-sequence state machine:
        - thinking: sample freely, watch for </think>
        - prefix: force-feed one prefix token per step
        - answer: sample freely, watch for } or max
        - finished: done
        """
        prompt_ids = self.create_prompt_ids(a, b)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device).repeat(G, 1)

        all_tokens = [[] for _ in range(G)]
        all_logprobs = [[] for _ in range(G)]
        all_gradient_mask = [[] for _ in range(G)]

        # Per-sequence state: 'thinking', 'end_think', 'prefix', 'answer', 'finished'
        state = ['thinking'] * G
        think_counts = [0] * G
        end_think_idx = [0] * G  # index into end_think_token_ids when forcing </think>
        prefix_idx = [0] * G
        answer_counts = [0] * G

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (G, vocab)

            for step in range(self.config.max_completion_tokens):
                if all(s == 'finished' for s in state):
                    break

                # Sample for all sequences (we'll override prefix ones)
                sampled = self._sample_token(logits)  # (G, 1)
                log_probs_full = F.log_softmax(logits, dim=-1)

                # Build the actual next token for each sequence
                next_tokens = sampled.clone()

                for i in range(G):
                    if state[i] == 'finished':
                        next_tokens[i, 0] = self.eos_token_id
                        continue

                    if state[i] == 'end_think':
                        # Force next </think> token
                        tid = self.end_think_token_ids[end_think_idx[i]]
                        next_tokens[i, 0] = tid
                        all_tokens[i].append(tid)
                        all_logprobs[i].append(0.0)
                        all_gradient_mask[i].append(False)
                        end_think_idx[i] += 1
                        if end_think_idx[i] >= len(self.end_think_token_ids):
                            state[i] = 'prefix'
                            prefix_idx[i] = 0
                        continue

                    if state[i] == 'prefix':
                        # Force next prefix token
                        tid = self.answer_prefix_ids[prefix_idx[i]]
                        next_tokens[i, 0] = tid
                        all_tokens[i].append(tid)
                        all_logprobs[i].append(0.0)
                        all_gradient_mask[i].append(False)
                        prefix_idx[i] += 1
                        if prefix_idx[i] >= len(self.answer_prefix_ids):
                            state[i] = 'answer'
                        continue

                    # thinking or answer — use sampled token
                    tid = sampled[i, 0].item()
                    lp = log_probs_full[i, tid].item()
                    if np.isnan(lp) or np.isinf(lp):
                        lp = -10.0

                    all_tokens[i].append(tid)
                    all_logprobs[i].append(lp)
                    all_gradient_mask[i].append(True)

                    if self._is_stop_token(tid):
                        state[i] = 'finished'
                        continue

                    if state[i] == 'thinking':
                        think_counts[i] += 1
                        # Safety valve: force </think> if too long
                        if think_counts[i] >= self.config.max_think_tokens:
                            state[i] = 'end_think'
                            end_think_idx[i] = 0
                            continue
                        # Check for natural </think>
                        n = len(self.end_think_token_ids)
                        if len(all_tokens[i]) >= n and all_tokens[i][-n:] == self.end_think_token_ids:
                            state[i] = 'prefix'
                            prefix_idx[i] = 0
                    elif state[i] == 'answer':
                        if tid == self.close_brace_token_id:
                            state[i] = 'finished'
                        else:
                            answer_counts[i] += 1
                            if answer_counts[i] >= self.config.max_answer_tokens:
                                state[i] = 'finished'

                # Feed all tokens to model
                outputs = self.model(next_tokens, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

        # Build result dicts
        completions = []
        for i in range(G):
            comp = self._build_result(
                prompt_ids, all_tokens[i], all_logprobs[i],
                all_gradient_mask[i], a, b, target, return_logprobs=True,
            )
            completions.append(comp)

        return completions

    def _build_result(
        self,
        prompt_ids: List[int],
        generated_tokens: List[int],
        generated_logprobs: List[float],
        gradient_mask: List[bool],
        a: int,
        b: int,
        target: int,
        return_logprobs: bool,
    ) -> Dict:
        """Parse generated tokens into thinking + answer, using prefix boundary."""
        # Find </think> boundary
        end_think_len = len(self.end_think_token_ids)
        think_end_pos = None  # Position right after </think>
        for pos in range(end_think_len, len(generated_tokens) + 1):
            if generated_tokens[pos - end_think_len:pos] == self.end_think_token_ids:
                think_end_pos = pos
                break

        if think_end_pos is not None:
            think_tokens = generated_tokens[:think_end_pos - end_think_len]
            # Answer tokens are after </think> + prefix
            prefix_end = think_end_pos + len(self.answer_prefix_ids)
            answer_tokens = generated_tokens[prefix_end:]
            # Remove trailing stop tokens / close brace from answer for display
            while answer_tokens and (self._is_stop_token(answer_tokens[-1]) or answer_tokens[-1] == self.close_brace_token_id):
                answer_tokens.pop()
        else:
            think_tokens = generated_tokens[:]
            answer_tokens = []

        think_text = self.tokenizer.decode(think_tokens, skip_special_tokens=False)
        answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=False).strip()
        answer_value = extract_answer(answer_text)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)

        return {
            'think_text': think_text,
            'think_tokens': think_tokens,
            'answer_text': answer_text,
            'answer_tokens': answer_tokens,
            'answer_value': answer_value,
            'target': target,
            'generated_tokens': generated_tokens,
            'generated_logprobs': generated_logprobs if return_logprobs else None,
            'gradient_mask': gradient_mask,
            'prompt_ids': prompt_ids,
            'prompt_text': prompt_text,
            'a': a,
            'b': b,
        }


def compute_reward(answer_value: int, target: int, think_text: str, digit_penalty_weight: float) -> Tuple[float, float, float]:
    """
    Compute reward with digit penalty.
    Returns (total_reward, correctness, digit_ratio).
    """
    correctness = 1.0 if (answer_value is not None and answer_value == target) else 0.0
    digit_ratio = compute_digit_ratio(think_text)
    reward = correctness - digit_penalty_weight * digit_ratio
    return reward, correctness, digit_ratio


def compute_completion_logprobs_batch(
    model,
    prompt_ids: List[int],
    all_generated_tokens: List[List[int]],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log probs for all G completions in a single batched forward pass.
    Handles variable-length completions with right-padding + attention masks.
    Returns:
      - token_logprobs: (G, max_comp_len) tensor of log probs
      - mask: (G, max_comp_len) boolean tensor (True = real token)
    """
    G = len(all_generated_tokens)
    comp_lengths = [len(tokens) for tokens in all_generated_tokens]
    max_comp_len = max(comp_lengths) if comp_lengths else 0
    prompt_len = len(prompt_ids)

    if max_comp_len == 0:
        return (
            torch.zeros(G, 0, device=device),
            torch.zeros(G, 0, dtype=torch.bool, device=device),
        )

    # Build target indices and mask
    target_ids = torch.zeros(G, max_comp_len, dtype=torch.long, device=device)
    mask = torch.zeros(G, max_comp_len, dtype=torch.bool, device=device)
    for i in range(G):
        clen = comp_lengths[i]
        target_ids[i, :clen] = torch.tensor(all_generated_tokens[i], dtype=torch.long, device=device)
        mask[i, :clen] = True

    # Single batched forward pass
    total_len = prompt_len + max_comp_len
    batch_ids = torch.zeros(G, total_len, dtype=torch.long, device=device)
    batch_attn = torch.zeros(G, total_len, dtype=torch.long, device=device)
    for i in range(G):
        seq = prompt_ids + all_generated_tokens[i]
        seq_len = len(seq)
        batch_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
        batch_attn[i, :seq_len] = 1

    outputs = model(batch_ids, attention_mask=batch_attn)
    logits = outputs.logits
    logits = torch.clamp(logits, min=-50, max=50)

    # Extract completion logits and compute log probs
    comp_logits = logits[:, prompt_len - 1: prompt_len - 1 + max_comp_len, :]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    del outputs, logits, comp_logits, log_probs

    # Zero out padding positions
    token_logprobs = token_logprobs * mask.float()

    # Sanitize NaN/Inf
    bad = torch.isnan(token_logprobs) | torch.isinf(token_logprobs)
    token_logprobs = token_logprobs.masked_fill(bad, -10.0)

    return token_logprobs, mask


def compute_grpo_loss(
    model,
    ref_model,
    completions: List[Dict],
    rewards: List[float],
    config: TrainingConfig,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute GRPO loss over thinking + answer tokens (excluding forced prefix).
    Uses padding mask AND gradient mask.
    """
    device = config.device
    G = len(completions)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if std_reward < 1e-8:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'mean_reward': mean_reward,
            'policy_loss': 0.0,
            'kl': 0.0,
            'skipped': True,
        }

    advantages = torch.tensor(
        [(r - mean_reward) / std_reward for r in rewards],
        dtype=torch.float32, device=device,
    )  # (G,)

    prompt_ids = completions[0]['prompt_ids']
    all_generated_tokens = [c['generated_tokens'] for c in completions]

    # Old logprobs from generation time
    max_comp_len = max(len(t) for t in all_generated_tokens)
    old_logprobs = torch.zeros(G, max_comp_len, device=device)
    for i, c in enumerate(completions):
        lps = c['generated_logprobs']
        if lps:
            old_logprobs[i, :len(lps)] = torch.tensor(lps, dtype=torch.float32, device=device)

    # Build gradient mask (excludes forced prefix positions)
    grad_mask = torch.zeros(G, max_comp_len, dtype=torch.bool, device=device)
    for i, c in enumerate(completions):
        gm = c['gradient_mask']
        if gm:
            grad_mask[i, :len(gm)] = torch.tensor(gm, dtype=torch.bool, device=device)

    # Current policy — batched forward pass, with grad
    current_logprobs, padding_mask = compute_completion_logprobs_batch(
        model, prompt_ids, all_generated_tokens, device,
    )

    # Reference policy — batched forward pass, no grad
    with torch.no_grad():
        ref_logprobs, _ = compute_completion_logprobs_batch(
            ref_model, prompt_ids, all_generated_tokens, device,
        )

    # Combined mask: real token AND trainable (not prefix)
    mask = padding_mask & grad_mask

    # PPO clipped objective with mask
    ratio = torch.exp(current_logprobs - old_logprobs)
    adv = advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * adv
    policy_obj = torch.min(surr1, surr2)

    total_tokens = mask.float().sum()
    if total_tokens < 1:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'mean_reward': mean_reward,
            'policy_loss': 0.0,
            'kl': 0.0,
            'skipped': True,
        }

    policy_loss = -(policy_obj * mask.float()).sum() / total_tokens

    # KL divergence (masked)
    kl_per_token = current_logprobs - ref_logprobs
    kl_div = (kl_per_token * mask.float()).sum() / total_tokens
    if torch.isnan(kl_div) or torch.isinf(kl_div):
        kl_div = torch.tensor(0.0, device=device)

    loss = policy_loss + config.kl_beta * kl_div

    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("NaN/Inf loss in GRPO, returning zero")
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    stats = {
        'mean_reward': mean_reward,
        'policy_loss': policy_loss.item(),
        'kl': kl_div.item(),
        'avg_comp_len': np.mean([len(t) for t in all_generated_tokens]),
    }

    return loss, stats


def prompt_user_action() -> int:
    """Ask user for number of additional steps (0 = save and exit)."""
    while True:
        print("\n" + "="*60)
        print("Enter number of additional steps (0 = save and exit)")
        print("="*60)
        try:
            steps = int(input("Additional steps: ").strip())
            if steps >= 0:
                return steps
            print("Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def train(config: TrainingConfig = None):
    """Main training loop."""
    if config is None:
        config = TrainingConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Auto-resume: find latest checkpoint if none explicitly specified
    if config.resume_from == "__fresh__":
        config.resume_from = None
        logger.info("Starting fresh (--fresh), ignoring existing checkpoints")
    elif not config.resume_from:
        latest = find_latest_checkpoint(output_dir)
        if latest:
            config.resume_from = latest
            logger.info(f"Auto-resuming from {latest}")

    # Determine model source: checkpoint or base
    model_source = config.resume_from if config.resume_from else config.model_name
    logger.info(f"Loading model: {model_source}")
    logger.info(f"Device: {config.device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    logger.info(f"Using dtype: {dtype}")

    model_kwargs = {"dtype": dtype}
    if config.device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **model_kwargs,
    ).to(config.device)

    if config.compile:
        logger.info("Compiling model with torch.compile()")
        model = torch.compile(model)

    # Reference model (always from base, not checkpoint)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    ).to(config.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    logger.info(f"Operand ranges: a={config.min_a}-{config.max_a}, b={config.min_b}-{config.max_b}")
    logger.info(f"Digit penalty weight: {config.digit_penalty_weight}")
    logger.info(f"Temperature: {config.temperature}, top_p: {config.top_p}, top_k: {config.top_k}")

    generator = CompletionGenerator(model, tokenizer, config)

    # Training loop state
    global_step = 0
    steps_this_round = 0
    recent_rewards = []
    recent_accuracies = []
    window_size = 50

    # Resume from checkpoint if specified
    if config.resume_from:
        ckpt_path = Path(config.resume_from)
        state = load_checkpoint(ckpt_path, optimizer, config.device)
        if state:
            global_step = state.get('global_step', 0)
            recent_rewards = state.get('recent_rewards', [])
            recent_accuracies = state.get('recent_accuracies', [])
            logger.info(f"Resumed from step {global_step}")

    logger.info(f"Starting training... (will pause every {config.max_steps} steps)")

    while True:
        model.train()

        # Sample a random pair each step
        a, b, target = sample_train_pair(config.min_a, config.max_a, config.min_b, config.max_b)

        # Generate G completions (batched)
        completions = generator.generate_completions_batch(a, b, target, G=config.G)

        # Compute rewards with digit penalty
        rewards = []
        correctnesses = []
        digit_ratios = []
        for c in completions:
            r, corr, dr = compute_reward(
                c['answer_value'], c['target'], c['think_text'], config.digit_penalty_weight,
            )
            rewards.append(r)
            correctnesses.append(corr)
            digit_ratios.append(dr)

        # Track running stats
        step_accuracy = np.mean(correctnesses)
        step_digit_ratio = np.mean(digit_ratios)
        step_avg_comp_len = np.mean([len(c['generated_tokens']) for c in completions])

        recent_rewards.append(np.mean(rewards))
        recent_accuracies.append(step_accuracy)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
        if len(recent_accuracies) > window_size:
            recent_accuracies.pop(0)
        running_acc = np.mean(recent_accuracies)

        num_correct = sum(correctnesses)

        # Compute GRPO loss
        loss, stats = compute_grpo_loss(
            model=model,
            ref_model=ref_model,
            completions=completions,
            rewards=rewards,
            config=config,
        )

        # Gradient step (or skip if no signal)
        grad_norm = 0.0
        if stats.get('skipped'):
            global_step += 1
            steps_this_round += 1
            clear_memory(config.device)
            logger.info(
                f"Step {global_step} | {a}x{b}={target} | "
                f"{int(num_correct)}/{config.G} correct | "
                f"Acc: {running_acc:.1%} | "
                f"DigR: {step_digit_ratio:.2f} | "
                f"Len: {step_avg_comp_len:.0f} | skipped"
            )
        else:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning("NaN/Inf gradient, skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()

            global_step += 1
            steps_this_round += 1
            clear_memory(config.device)

            logger.info(
                f"Step {global_step} | {a}x{b}={target} | "
                f"{int(num_correct)}/{config.G} correct | "
                f"Acc: {running_acc:.1%} | "
                f"DigR: {step_digit_ratio:.2f} | "
                f"Len: {step_avg_comp_len:.0f} | "
                f"Loss: {stats['policy_loss']:.4f} | KL: {stats['kl']:.4f} | "
                f"Grad: {grad_norm:.4f}"
            )

        # Show full CoT of the highest-reward completion
        if config.verbose:
            best_idx = max(range(len(rewards)), key=lambda j: rewards[j])
            best = completions[best_idx]
            best_r = rewards[best_idx]
            corr_mark = "+" if correctnesses[best_idx] == 1.0 else "-"
            logger.info(
                f"  >> Best(r={best_r:.2f} {corr_mark}) {a}x{b}={target} | "
                f"Answer: {best['answer_text']} | "
                f"DigR: {digit_ratios[best_idx]:.2f} | "
                f"Think({len(best['think_tokens'])}t):\n{best['think_text']}"
            )

        # Periodic checkpoint saving
        if config.save_every > 0 and global_step % config.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_step_{global_step}"
            save_checkpoint(ckpt_path, model, tokenizer, optimizer, {
                'global_step': global_step,
                'recent_rewards': recent_rewards,
                'recent_accuracies': recent_accuracies,
            })

        # Check if we've reached max_steps for this round
        if steps_this_round >= config.max_steps:
            logger.info(f"\nCompleted {config.max_steps} steps (total: {global_step})")

            additional_steps = prompt_user_action()

            if additional_steps == 0:
                final_path = output_dir / "final_model"
                model.save_pretrained(final_path)
                tokenizer.save_pretrained(final_path)
                logger.info(f"Saved final model to {final_path}")
                return
            else:
                config.max_steps = additional_steps
                steps_this_round = 0
                logger.info(f"Continuing training for {additional_steps} more steps...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwen3-0.6B to multiply 3-digit numbers with encoded chain-of-thought via GRPO"
    )
    parser.add_argument("--quiet", action="store_true", help="Hide generation details")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=500, help="Steps before prompting to continue")
    parser.add_argument("--G", type=int, default=64, help="Group size for GRPO")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--digit-penalty", type=float, default=0.5, help="Weight for digit ratio penalty in reward")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max total completion tokens")
    parser.add_argument("--max-answer-tokens", type=int, default=20, help="Max answer tokens after forced prefix")
    parser.add_argument("--min-a", type=int, default=100, help="Min first operand (3-digit)")
    parser.add_argument("--max-a", type=int, default=999, help="Max first operand (3-digit)")
    parser.add_argument("--min-b", type=int, default=100, help="Min second operand (3-digit)")
    parser.add_argument("--max-b", type=int, default=999, help="Max second operand (3-digit)")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignoring any existing checkpoints")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() on the policy model")
    parser.add_argument("--clean", action="store_true", help="Delete old training outputs before starting")

    args = parser.parse_args()

    config = TrainingConfig()
    config.verbose = not args.quiet
    if args.device:
        config.device = args.device
    config.learning_rate = args.lr
    config.max_steps = args.max_steps
    config.G = args.G
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.top_k = args.top_k
    config.digit_penalty_weight = args.digit_penalty
    config.max_completion_tokens = args.max_tokens
    config.max_answer_tokens = args.max_answer_tokens
    config.min_a = args.min_a
    config.max_a = args.max_a
    config.min_b = args.min_b
    config.max_b = args.max_b
    config.save_every = args.save_every
    config.resume_from = args.resume
    config.compile = args.compile
    if args.clean:
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info(f"Cleaned output directory: {output_dir}")
    if args.fresh or args.clean:
        config.resume_from = "__fresh__"  # Sentinel to skip auto-resume

    train(config)
