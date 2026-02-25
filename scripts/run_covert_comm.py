"""Entry point for covert communication two-phase RL training.

Usage:
    # Generate dataset
    python scripts/run_covert_comm.py generate_data --level=1

    # Train (level 1 = 2-digit primes)
    python scripts/run_covert_comm.py train --level=1 --steps=200

    # Smoke test
    python scripts/run_covert_comm.py train --level=1 --steps=2 --num_generations=2 --num_prompts=2

    # Resume from checkpoint
    python scripts/run_covert_comm.py train --level=2 --resume_from=<ckpt_path>

    # Legacy interface (with full control over obfuscation judge)
    python scripts/run_covert_comm.py run_covert_communication --steps 500
"""

import os
import fire
from datetime import datetime

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlCovertCommGRPO
from src.train.covert_comm.grpo import CovertCommGRPO
from src.train.covert_comm.dataset import generate_dataset as _generate_dataset
from src import RESULTS_PATH, utils


DEFAULT_DATASET_PATH = f"{RESULTS_PATH}/data/covert_comm_train.jsonl"


# ======================================================================
# New curriculum-based interface
# ======================================================================

def generate_data(level: int = 1):
    """Generate prime dataset for a given level.

    Level 1 = 2-digit primes (10-99)
    Level 2 = 3-digit primes (100-999)
    Level 3 = 4-digit primes (1000-9999)
    """
    output_dir = f"{RESULTS_PATH}/data"
    path = _generate_dataset(level=int(level), output_dir=output_dir)
    print(f"Dataset written to {path}")


def train(
    level: int = 1,
    model_id: str = "Qwen/Qwen3-4B",
    steps: int = 200,
    seed: int = 1,
    correctness_reward: float = 3.0,
    digit_penalty: float = 0.5,
    num_generations: int = 8,
    num_prompts: int = 16,
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
    temperature: float = 1.0,
    beta: float = 0.001,
    save_steps: int = 50,
    save_only_model: bool = True,
    resume_from: str | None = None,
    **kwargs,
):
    """Run covert communication training.

    Args:
        level: Curriculum level (1 = 2-digit, 2 = 3-digit, ...).
        model_id: HuggingFace model identifier.
        steps: Number of training steps.
        seed: Random seed.
        correctness_reward: Reward for correct decode.
        digit_penalty: Penalty per digit character in encoded text.
        num_generations: Generations per prompt (halved from default —
            two generate calls per step).
        num_prompts: Number of prompts per batch.
        max_prompt_length: Maximum prompt token length.
        max_completion_length: Maximum completion token length.
        resume_from: Path to checkpoint to resume from.
    """
    level = int(level)
    dataset_path = f"{RESULTS_PATH}/data/covert_comm_level{level}.jsonl"

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}, generating...")
        _generate_dataset(level=level, output_dir=f"{RESULTS_PATH}/data")

    run_name = f"covert_comm_level{level}_cr{correctness_reward}_dp{digit_penalty}"
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"

    config = GRPOConfig(
        model_id=model_id,
        seed=int(seed),
        run_id=run_id,
        dataset_path=dataset_path,
        max_steps=int(steps),
        num_generations=int(num_generations),
        num_prompts=int(num_prompts),
        max_prompt_length=int(max_prompt_length),
        max_completion_length=int(max_completion_length),
        temperature=float(temperature),
        beta=float(beta),
        save_steps=int(save_steps),
        save_only_model=save_only_model,
        resume_from_checkpoint=resume_from is not None,
        # Covert-comm reward settings
        correct_reward=float(correctness_reward),
        incorrect_penalty=0.0,  # CovertCommReward uses 0 for incorrect
        digit_penalty_value=float(digit_penalty),
        obfuscation_weight=0.0,  # No LLM judge for lightweight reward
        # Empty reward/screening — trainer computes rewards directly
        reward_funcs_kwargs={},
        screening_funcs_kwargs={},
    )

    trainer = CovertCommGRPO(config)
    trainer.run()
    print(f"Training completed for {run_id}")


# ======================================================================
# Legacy interface (full CovertCommunicationReward with LLM judge)
# ======================================================================

def run_covert_communication(
    model_id: str = "Qwen/Qwen3-4B",
    steps: int = 500,
    seed: int = 1,
    correct_reward: float = 3.0,
    incorrect_penalty: float = -1.0,
    digit_penalty_value: float = -0.5,
    obfuscation_weight: float = 1.0,
    judge_model_id: str = "anthropic/claude-haiku-4.5",
    num_generations: int = 8,
    num_prompts: int = 8,
    max_prompt_length: int = 512,
    max_completion_length: int = 256,
    temperature: float = 1.0,
    beta: float = 0.001,
    dataset_path: str = DEFAULT_DATASET_PATH,
    save_steps: int = 50,
    save_only_model: bool = True,
):
    """Train a model on covert communication with two-phase RL.

    Uses the full CovertCommunicationReward with optional LLM judge.
    """

    run_name = (
        f"covert_comm_cr{correct_reward}_dp{digit_penalty_value}_ow{obfuscation_weight}"
    )
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"

    assert os.path.exists(dataset_path), (
        f"Dataset not found at {dataset_path}. "
        "Run: python scripts/create_covert_comm_dataset.py"
    )

    config = GRPOConfig(
        model_id=model_id,
        seed=int(seed),
        run_id=run_id,
        dataset_path=dataset_path,
        max_steps=int(steps),
        num_generations=num_generations,
        num_prompts=num_prompts,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
        beta=beta,
        save_steps=save_steps,
        save_only_model=save_only_model,
        # Covert-comm specific
        correct_reward=correct_reward,
        incorrect_penalty=incorrect_penalty,
        digit_penalty_value=digit_penalty_value,
        obfuscation_weight=obfuscation_weight,
        judge_model_id=judge_model_id,
        # Use a simple reward function for validation (encode-only)
        reward_funcs_kwargs={},
        screening_funcs_kwargs={},
    )

    trainer = VerlCovertCommGRPO(config)
    trainer.run()
    print(f"Training completed for {run_id}")


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        "generate_data": generate_data,
        "train": train,
        "run_covert_communication": run_covert_communication,
    })
