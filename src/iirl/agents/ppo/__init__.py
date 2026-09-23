"""PPO: proximal policy optimization self-play agent."""

from iirl.agents.ppo.agent import PPOAgent, masked_policy

__all__ = ["PPOAgent", "masked_policy"]
