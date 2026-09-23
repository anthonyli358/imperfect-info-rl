"""Uniform-random baseline agent."""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action


class RandomAgent(Agent):
    """Picks uniformly at random among the legal actions."""

    def __init__(self, name: str = "random") -> None:
        """Create a random agent.

        Args:
            name: Label used in standings and error messages.
        """
        self.name = name

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return a uniformly random legal action."""
        legal = game.legal_actions(state)
        return int(rng.choice(legal))
