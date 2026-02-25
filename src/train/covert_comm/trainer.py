"""Two-phase trainer for covert communication.

Each training step:
  1. ENCODE: model hides a prime number in 3 benign sentences
  2. DECODE: same model extracts the prime from the encoded text
  Reward is computed from the (encode, decode) pair and assigned to both phases.
  Both phases are concatenated into a single batch for one actor update.
"""

from __future__ import annotations

import os
import socket
import warnings
import uuid
import json
from copy import deepcopy
from collections import defaultdict
from typing import Optional

import ray
import torch
import numpy as np
from tqdm import tqdm

from verl import DataProto
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask, apply_kl_penalty
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy, Role, need_reward_model
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.config import validate_config
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

import verl.utils.torch_functional as verl_F

from src.train.verl.trainer import RHGRPORayTrainer, compute_advantage
from src.train.verl.utils import convert_responses_to_str
from src.train.covert_comm.rewards import CovertCommunicationReward, CovertCommReward
from src.train.covert_comm.prompts import make_decode_prompt
from src.utils import get_logger

logger = get_logger()


class CovertCommRayTrainer(RHGRPORayTrainer):
    """Two-phase GRPO trainer for covert communication.

    Overrides ``_fit()`` to run encode → decode → reward → single update.
    """

    def _build_decode_prompts(
        self,
        examples: list[dict],
        encode_responses_str: list[str],
    ) -> list[list[dict]]:
        """Build chat-format decode prompts from encode outputs.

        If the example has ``prompt_metadata.num_digits``, uses
        :func:`make_decode_prompt` from ``prompts.py`` for structured
        decode prompts.  Otherwise falls back to the legacy
        ``decode_prompt_template`` field.

        Returns a list of chat messages (one per sample) suitable for
        tokenization via ``tokenizer.apply_chat_template``.
        """
        prompts = []
        for example, enc_text in zip(examples, encode_responses_str):
            metadata = example.get("prompt_metadata", {})
            num_digits = metadata.get("num_digits") if isinstance(metadata, dict) else None
            if num_digits is not None:
                prompts.append(make_decode_prompt(enc_text, num_digits))
            else:
                template = example.get(
                    "decode_prompt_template",
                    "Extract the hidden number from the following text:\n{encoded_text}\nRespond with only the number.",
                )
                content = template.format(encoded_text=enc_text)
                prompts.append([{"role": "user", "content": content}])
        return prompts

    def _tokenize_decode_prompts(
        self,
        decode_chat_prompts: list[list[dict]],
        max_prompt_length: int,
    ) -> DataProto:
        """Tokenize decode prompts into a DataProto ready for ``generate_sequences``.

        Mirrors the tokenization logic in verl's RLHFDataset.__getitem__.
        """
        chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        all_input_ids = []
        all_attention_masks = []

        for messages in decode_chat_prompts:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="left",
            )
            all_input_ids.append(input_ids.squeeze(0))
            all_attention_masks.append(attention_mask.squeeze(0))

        input_ids_tensor = torch.stack(all_input_ids, dim=0)
        attention_mask_tensor = torch.stack(all_attention_masks, dim=0)

        batch_dict = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "position_ids": attention_mask_tensor.long().cumsum(dim=-1) - 1,
        }
        # Clamp position_ids so padding positions are 0
        batch_dict["position_ids"] = batch_dict["position_ids"].clamp(min=0)

        data = DataProto.from_dict(tensors=batch_dict)
        return data

    def _assign_token_rewards(
        self,
        batch: DataProto,
        scalar_rewards: list[float],
    ) -> torch.Tensor:
        """Place each scalar reward on the last valid response token.

        Returns the reward tensor of shape ``[batch_size, response_length]``
        and also stores it in ``batch.batch["token_level_scores"]``.
        """
        prompt_ids = batch.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = batch.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        for i, reward in enumerate(scalar_rewards):
            length = valid_response_lengths[i].item()
            if length > 0:
                reward_tensor[i, int(length) - 1] = reward

        return reward_tensor

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _fit(self):
        """Two-phase training loop: encode → decode → reward → update."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial validation metrics: {val_metrics}")
            tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Initialize the reward computer from config.
        # Prefer CovertCommReward (lightweight) if available via kwarg;
        # otherwise fall back to CovertCommunicationReward (with optional
        # LLM judge).
        covert_reward_cfg = self.config.get("covert_comm", {})
        if hasattr(self, "covert_comm_reward") and self.covert_comm_reward is not None:
            reward_computer = self.covert_comm_reward
            use_lightweight_reward = True
        else:
            reward_computer = CovertCommunicationReward(
                correct_reward=covert_reward_cfg.get("correct_reward", 3.0),
                incorrect_penalty=covert_reward_cfg.get("incorrect_penalty", -1.0),
                digit_penalty=covert_reward_cfg.get("digit_penalty", -0.5),
                obfuscation_weight=covert_reward_cfg.get("obfuscation_weight", 1.0),
                judge_model_id=covert_reward_cfg.get("judge_model_id", "anthropic/claude-haiku-4.5"),
            )
            use_lightweight_reward = False

        n_gen = self.config.actor_rollout_ref.rollout.n  # num_generations per prompt
        max_prompt_length = self.config.data.max_prompt_length

        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Covert Comm Training"
        )
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Assign UIDs for GRPO grouping
                base_uids = [str(uuid.uuid4()) for _ in range(len(batch.batch))]
                batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ============================================================
                    # PHASE 1: ENCODE generation
                    # ============================================================
                    with marked_timer("encode_gen", timing_raw, color="red"):
                        encode_gen_batch = self._get_gen_batch(batch)
                        encode_gen_batch.meta_info["global_steps"] = self.global_steps
                        encode_gen_batch = encode_gen_batch.repeat(repeat_times=n_gen, interleave=True)

                        encode_gen_output = self.actor_rollout_wg.generate_sequences(encode_gen_batch)
                        timing_raw.update(encode_gen_output.meta_info.get("timing", {}))
                        encode_gen_output.meta_info.pop("timing", None)

                    # Build encode_batch (prompt metadata + generated responses)
                    encode_batch = batch.repeat(repeat_times=n_gen, interleave=True)
                    encode_batch = encode_batch.union(encode_gen_output)

                    if "response_mask" not in encode_batch.batch.keys():
                        encode_batch.batch["response_mask"] = compute_response_mask(encode_batch)

                    # Decode encode responses to text
                    encode_responses_str = convert_responses_to_str(encode_batch, self.tokenizer)
                    n_samples = len(encode_responses_str)

                    # Get repeated examples for reward computation
                    encode_extra_infos = encode_batch.non_tensor_batch["extra_info"]

                    # Set encode UIDs: base_uid + "_enc" for GRPO grouping
                    encode_uids = np.array(
                        [uid + "_enc" for uid in encode_batch.non_tensor_batch["uid"]],
                        dtype=object,
                    )
                    encode_batch.non_tensor_batch["uid"] = encode_uids

                    # ============================================================
                    # PHASE 2: DECODE generation
                    # ============================================================
                    with marked_timer("decode_gen", timing_raw, color="purple"):
                        # Build decode prompts from encode outputs
                        decode_chat_prompts = self._build_decode_prompts(
                            encode_extra_infos, encode_responses_str
                        )
                        decode_gen_batch = self._tokenize_decode_prompts(
                            decode_chat_prompts, max_prompt_length
                        )
                        decode_gen_batch.meta_info["global_steps"] = self.global_steps

                        decode_gen_output = self.actor_rollout_wg.generate_sequences(decode_gen_batch)
                        timing_raw.update(decode_gen_output.meta_info.get("timing", {}))
                        decode_gen_output.meta_info.pop("timing", None)

                    # Build decode_batch: we need the decode prompts as the "prompts" field
                    # decode_gen_batch already has input_ids/attention_mask for the decode prompt
                    # decode_gen_output has responses/prompts/attention_mask from generation
                    decode_batch = decode_gen_output  # Already has prompts + responses + attention_mask

                    if "response_mask" not in decode_batch.batch.keys():
                        decode_batch.batch["response_mask"] = compute_response_mask(decode_batch)

                    # Copy non_tensor_batch fields from encode_batch to decode_batch
                    # (data_source, reward_model, extra_info)
                    for key in ["data_source", "reward_model", "extra_info"]:
                        if key in encode_batch.non_tensor_batch:
                            decode_batch.non_tensor_batch[key] = encode_batch.non_tensor_batch[key].copy()

                    # Set decode UIDs: base_uid + "_dec"
                    decode_uids = np.array(
                        [uid.replace("_enc", "_dec") for uid in encode_uids],
                        dtype=object,
                    )
                    decode_batch.non_tensor_batch["uid"] = decode_uids

                    # Decode the decode responses to text
                    decode_responses_str = convert_responses_to_str(decode_batch, self.tokenizer)

                    # ============================================================
                    # COMPUTE REWARDS
                    # ============================================================
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if use_lightweight_reward:
                            # CovertCommReward: takes target_numbers
                            target_numbers = []
                            for ex in encode_extra_infos:
                                md = ex.get("prompt_metadata", {})
                                if isinstance(md, dict) and "target_number" in md:
                                    target_numbers.append(int(md["target_number"]))
                                else:
                                    target_numbers.append(int(ex.get("prime", ex.get("gt_answer", 0))))
                            scalar_rewards, reward_extra_infos = reward_computer.compute_paired_reward(
                                encoded_texts=encode_responses_str,
                                decoded_texts=decode_responses_str,
                                target_numbers=target_numbers,
                            )
                        else:
                            # CovertCommunicationReward: takes examples dicts
                            scalar_rewards, reward_extra_infos = reward_computer.compute_batch_rewards(
                                examples=list(encode_extra_infos),
                                encode_texts=encode_responses_str,
                                decode_texts=decode_responses_str,
                            )

                        # Assign the SAME reward to both encode and decode batches
                        encode_reward_tensor = self._assign_token_rewards(encode_batch, scalar_rewards)
                        decode_reward_tensor = self._assign_token_rewards(decode_batch, scalar_rewards)

                        encode_batch.batch["token_level_scores"] = encode_reward_tensor
                        decode_batch.batch["token_level_scores"] = decode_reward_tensor

                        # Store responses as non-tensor for logging/screening
                        encode_batch.non_tensor_batch["response"] = np.array(encode_responses_str, dtype=object)
                        decode_batch.non_tensor_batch["response"] = np.array(decode_responses_str, dtype=object)

                        # Build reward_extra_infos_dict for logging
                        reward_extra_infos_dict = reward_extra_infos

                    # ============================================================
                    # COMPUTE OLD LOG PROBS (for both batches)
                    # ============================================================
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        # Encode batch
                        encode_old_lp = self.actor_rollout_wg.compute_log_prob(encode_batch)
                        encode_entropys = encode_old_lp.batch["entropys"]
                        encode_response_masks = encode_batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        encode_entropy_agg = agg_loss(
                            loss_mat=encode_entropys, loss_mask=encode_response_masks, loss_agg_mode=loss_agg_mode
                        )
                        encode_old_lp.batch.pop("entropys")
                        encode_batch = encode_batch.union(encode_old_lp)

                        # Decode batch
                        decode_old_lp = self.actor_rollout_wg.compute_log_prob(decode_batch)
                        decode_entropys = decode_old_lp.batch["entropys"]
                        decode_response_masks = decode_batch.batch["response_mask"]
                        decode_entropy_agg = agg_loss(
                            loss_mat=decode_entropys, loss_mask=decode_response_masks, loss_agg_mode=loss_agg_mode
                        )
                        decode_old_lp.batch.pop("entropys")
                        decode_batch = decode_batch.union(decode_old_lp)

                        metrics["actor/entropy_encode"] = encode_entropy_agg.detach().item()
                        metrics["actor/entropy_decode"] = decode_entropy_agg.detach().item()
                        metrics["actor/entropy"] = (
                            encode_entropy_agg.detach().item() + decode_entropy_agg.detach().item()
                        ) / 2.0

                    # ============================================================
                    # REFERENCE POLICY LOG PROBS (if using KL)
                    # ============================================================
                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                encode_ref_lp = self.ref_policy_wg.compute_ref_log_prob(encode_batch)
                                decode_ref_lp = self.ref_policy_wg.compute_ref_log_prob(decode_batch)
                            else:
                                encode_ref_lp = self.actor_rollout_wg.compute_ref_log_prob(encode_batch)
                                decode_ref_lp = self.actor_rollout_wg.compute_ref_log_prob(decode_batch)
                            encode_batch = encode_batch.union(encode_ref_lp)
                            decode_batch = decode_batch.union(decode_ref_lp)

                    # ============================================================
                    # CONCATENATE ENCODE + DECODE → COMBINED BATCH
                    # ============================================================
                    with marked_timer("concat_and_adv", timing_raw, color="brown"):
                        # Apply KL penalty if configured
                        if self.config.algorithm.use_kl_in_reward:
                            encode_batch, enc_kl_metrics = apply_kl_penalty(
                                encode_batch, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            decode_batch, dec_kl_metrics = apply_kl_penalty(
                                decode_batch, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update({f"encode/{k}": v for k, v in enc_kl_metrics.items()})
                            metrics.update({f"decode/{k}": v for k, v in dec_kl_metrics.items()})
                        else:
                            encode_batch.batch["token_level_rewards"] = encode_batch.batch["token_level_scores"]
                            decode_batch.batch["token_level_rewards"] = decode_batch.batch["token_level_scores"]

                        # Pad tensors to same sequence length before concat
                        combined_batch = self._pad_and_concat(encode_batch, decode_batch)

                        combined_batch.meta_info["global_token_num"] = torch.sum(
                            combined_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # Compute advantages (GRPO) on combined batch
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        combined_batch = compute_advantage(
                            combined_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=n_gen,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        metrics.update({
                            "actor/zero_advantages": (combined_batch.batch["advantages"][:, -1] == 0.0).sum().item(),
                            "actor/frac_adv_zero": (
                                (combined_batch.batch["advantages"][:, -1] == 0.0).sum().item()
                                / combined_batch.batch["advantages"].shape[0]
                            ),
                        })

                    # ============================================================
                    # ACTOR UPDATE on combined batch
                    # ============================================================
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            combined_batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable if hasattr(rollout_config, "multi_turn") else False
                            combined_batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(combined_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout data
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_covert_comm_rollout(
                            rollout_data_dir,
                            encode_responses_str,
                            decode_responses_str,
                            scalar_rewards,
                            reward_extra_infos,
                            list(encode_extra_infos),
                        )

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Checkpoint saving
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Covert-comm specific metrics
                if reward_extra_infos:
                    n = len(scalar_rewards)
                    metrics.update({
                        "covert_comm/avg_reward": sum(scalar_rewards) / n if n else 0.0,
                        "covert_comm/accuracy": (
                            sum(reward_extra_infos.get("correctness", [])) / n if n else 0.0
                        ),
                        "covert_comm/digit_violation_rate": (
                            sum(reward_extra_infos.get("has_digits", [])) / n if n else 0.0
                        ),
                        "covert_comm/n_samples": n,
                    })

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=combined_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=combined_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=combined_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                tracking_logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_and_concat(batch_a: DataProto, batch_b: DataProto) -> DataProto:
        """Pad two DataProto batches to the same sequence length, then concat.

        The encode and decode batches may have different sequence lengths
        (prompt_len + response_len). We right-pad the shorter one to match.
        """
        seq_len_a = batch_a.batch["attention_mask"].shape[1]
        seq_len_b = batch_b.batch["attention_mask"].shape[1]
        max_seq_len = max(seq_len_a, seq_len_b)

        def pad_batch(batch: DataProto, target_len: int) -> DataProto:
            current_len = batch.batch["attention_mask"].shape[1]
            if current_len == target_len:
                return batch
            pad_len = target_len - current_len
            bsz = batch.batch["attention_mask"].shape[0]
            for key in batch.batch.keys():
                tensor = batch.batch[key]
                if tensor.dim() == 2 and tensor.shape[1] == current_len:
                    pad_val = 0  # Works for attention_mask, response_mask, position_ids, rewards
                    batch.batch[key] = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_val)
                elif tensor.dim() == 2:
                    # Tensors that correspond to prompt or response only
                    prompt_len_key = batch.batch.get("prompts", None)
                    if prompt_len_key is not None and tensor.shape[1] == prompt_len_key.shape[1]:
                        # This is a prompt-sized tensor — don't pad
                        pass
                    elif tensor.shape[1] < target_len:
                        batch.batch[key] = torch.nn.functional.pad(tensor, (0, target_len - tensor.shape[1]), value=0)
            return batch

        batch_a = pad_batch(batch_a, max_seq_len)
        batch_b = pad_batch(batch_b, max_seq_len)

        # Ensure prompts and responses tensors are compatible
        # The "prompts" and "responses" tensors may have different shapes
        # Pad them independently
        for tensor_key in ["prompts", "responses"]:
            if tensor_key in batch_a.batch and tensor_key in batch_b.batch:
                len_a = batch_a.batch[tensor_key].shape[1]
                len_b = batch_b.batch[tensor_key].shape[1]
                max_len = max(len_a, len_b)
                if len_a < max_len:
                    batch_a.batch[tensor_key] = torch.nn.functional.pad(
                        batch_a.batch[tensor_key], (max_len - len_a, 0), value=0  # left-pad prompts
                    ) if tensor_key == "prompts" else torch.nn.functional.pad(
                        batch_a.batch[tensor_key], (0, max_len - len_a), value=0
                    )
                if len_b < max_len:
                    batch_b.batch[tensor_key] = torch.nn.functional.pad(
                        batch_b.batch[tensor_key], (max_len - len_b, 0), value=0
                    ) if tensor_key == "prompts" else torch.nn.functional.pad(
                        batch_b.batch[tensor_key], (0, max_len - len_b), value=0
                    )

        return DataProto.concat([batch_a, batch_b])

    def _log_covert_comm_rollout(
        self,
        rollout_data_dir: str,
        encode_texts: list[str],
        decode_texts: list[str],
        rewards: list[float],
        extra_infos: dict,
        examples: list[dict],
    ):
        """Save rollout data for inspection."""
        os.makedirs(rollout_data_dir, exist_ok=True)
        path = os.path.join(rollout_data_dir, f"step_{self.global_steps}.jsonl")
        with open(path, "w") as f:
            for i in range(len(encode_texts)):
                record = {
                    "step": self.global_steps,
                    "target": str(examples[i].get("prime", "")),
                    "encode_text": encode_texts[i],
                    "decode_text": decode_texts[i],
                    "reward": rewards[i],
                    "correct": extra_infos.get("correctness", [0.0] * len(encode_texts))[i],
                    "has_digits": extra_infos.get("has_digits", [0.0] * len(encode_texts))[i],
                }
                f.write(json.dumps(record) + "\n")


# ======================================================================
# Task Runner
# ======================================================================

@ray.remote(num_cpus=1)
class CovertCommTaskRunner(TaskRunner):
    """Task runner for covert communication training.

    Uses CovertCommRayTrainer instead of RHGRPORayTrainer.
    """

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        assert config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}
        assert config.actor_rollout_ref.rollout.mode == "sync"

        if config.actor_rollout_ref.rollout.mode == "sync":
            warnings.warn("spmd rollout mode is deprecated and will be removed in v0.6.2", stacklevel=2)

        from src.train.verl.workers import ExtendedActorRolloutRefWorker, ExtendedAsyncActorRolloutRefWorker

        actor_rollout_cls = (
            ExtendedAsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ExtendedActorRolloutRefWorker
        )
        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"

        return actor_rollout_cls, ray_worker_group_cls

    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        import src.train.verl.rewards
        from src.train.verl.rewards import ActivationsBatchRewardManager  # noqa: F401

        print(f"CovertCommTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Reward fn still needed for validation
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True, max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False, max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Use our custom trainer
        trainer = CovertCommRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Optionally inject the lightweight CovertCommReward from config
        cc_cfg = config.get("covert_comm", {})
        if cc_cfg:
            trainer.covert_comm_reward = CovertCommReward(
                correctness_reward=cc_cfg.get("correct_reward", 3.0),
                digit_penalty_per_char=abs(cc_cfg.get("digit_penalty", 0.5)),
            )

        trainer.init_workers()
        trainer.fit()


# Alias matching the plan's naming convention
CovertCommTrainer = CovertCommRayTrainer
