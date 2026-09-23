"""Self-play training loops (AlphaZero-style net + search; PPO model-free)."""

from iirl.selfplay.alphazero import AZConfig, AZEvalPoint, train_alphazero
from iirl.selfplay.ppo import PPOConfig, PPOEvalPoint, train_ppo

__all__ = [
    "AZConfig",
    "AZEvalPoint",
    "PPOConfig",
    "PPOEvalPoint",
    "train_alphazero",
    "train_ppo",
]
