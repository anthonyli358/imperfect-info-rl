"""Agents: policies written against the game-agnostic ``Game`` interface."""

from iirl.agents.base import Agent
from iirl.agents.random_agent import RandomAgent

__all__ = ["Agent", "RandomAgent"]
