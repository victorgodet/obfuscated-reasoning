"""Covert communication two-phase RL training.

Note: ``CovertCommGRPO`` is intentionally NOT imported here to avoid a
circular import with ``src.train.verl.grpo``.  Import it directly::

    from src.train.covert_comm.grpo import CovertCommGRPO
"""

from src.train.covert_comm.rewards import CovertCommReward, CovertCommunicationReward
from src.train.covert_comm.trainer import (
    CovertCommRayTrainer,
    CovertCommTrainer,
    CovertCommTaskRunner,
)
from src.train.covert_comm.dataset import generate_dataset, get_primes_by_level, sieve_primes
from src.train.covert_comm.prompts import (
    ENCODE_SYSTEM_PROMPT,
    make_decode_prompt,
    make_encode_prompt,
)

__all__ = [
    "CovertCommReward",
    "CovertCommunicationReward",
    "CovertCommRayTrainer",
    "CovertCommTrainer",
    "CovertCommTaskRunner",
    "generate_dataset",
    "get_primes_by_level",
    "sieve_primes",
    "ENCODE_SYSTEM_PROMPT",
    "make_decode_prompt",
    "make_encode_prompt",
]
