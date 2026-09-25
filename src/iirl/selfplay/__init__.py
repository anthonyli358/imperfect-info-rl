"""Self-play training loops (AlphaZero-style net + search; PPO model-free)."""

from iirl.selfplay.alphazero import AZConfig, AZEvalPoint, train_alphazero
from iirl.selfplay.ppo import PPOConfig, PPOEvalPoint, train_ppo
from iirl.selfplay.value import ValueConfig, ValueReport, train_value

__all__ = [
    "AZConfig",
    "AZEvalPoint",
    "PPOConfig",
    "PPOEvalPoint",
    "train_alphazero",
    "train_ppo",
    "train_value",
    "ValueConfig",
    "ValueReport",
]
