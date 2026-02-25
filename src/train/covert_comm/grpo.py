"""High-level training service for covert communication.

Inherits dataset loading and config creation from :class:`VerlGRPO` and
overrides ``train()`` to use :class:`CovertCommTaskRunner`.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo

from src import utils
from src.train.verl.grpo import VerlGRPO
from src.train.covert_comm.trainer import CovertCommTaskRunner


class CovertCommGRPO(VerlGRPO):
    """GRPO training service for covert communication.

    Uses :class:`CovertCommTaskRunner` (which internally uses
    :class:`CovertCommRayTrainer`) instead of the standard
    :class:`RHGRPOTaskRunner`.  The custom trainer runs
    encode -> decode -> reward -> single actor update per step.
    """

    name: str = "covert_comm_grpo"

    def create_config(self):
        """Extend base config creation with covert-comm reward settings."""
        config = super().create_config()

        # Inject covert-comm reward settings so the trainer can read them.
        OmegaConf.set_struct(config, False)
        config.covert_comm = {
            "correct_reward": getattr(self.training_config, "correct_reward", 3.0),
            "incorrect_penalty": getattr(self.training_config, "incorrect_penalty", -1.0),
            "digit_penalty": getattr(self.training_config, "digit_penalty_value", 0.5),
            "obfuscation_weight": getattr(self.training_config, "obfuscation_weight", 0.0),
            "judge_model_id": getattr(self.training_config, "judge_model_id", "anthropic/claude-haiku-4.5"),
        }

        # Save updated config
        utils.save_yaml(self.verl_full_config_path(), config)
        return config

    def train(self):
        """Run training using the covert-comm task runner."""
        self.load_configure_datasets()
        config = self.create_config()

        run_ppo(config, task_runner_class=CovertCommTaskRunner)

        self.graceful_shutdown()
