"""REINFORCE: a linear self-play policy-gradient agent and its trainer."""

from iirl.agents.reinforce.policy import LinearPolicy, ReinforceAgent
from iirl.agents.reinforce.trainer import EvalPoint, TrainConfig, train

__all__ = ["EvalPoint", "LinearPolicy", "ReinforceAgent", "TrainConfig", "train"]
