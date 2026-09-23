"""The ``Agent`` interface.

Agents are written purely against ``Game`` and contain zero game-specific code: given a state where
it is their turn, they return a legal action. The runner never asks an agent to act at a chance or
terminal node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from iirl.core.game import Game, State
from iirl.core.types import Action


class Agent(ABC):
    """Base class for a policy over any ``Game``.

    Attributes:
        name: Human-readable label, used in tournament standings and error messages.
    """

    name: str = "agent"

    @abstractmethod
    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return a legal action for the current player in ``state``.

        Args:
            game: The game being played.
            state: A non-terminal, non-chance state where it is this agent's turn.
            rng: Seeded generator for stochastic policies (reproducibility).

        Returns:
            An action drawn from ``game.legal_actions(state)``.
        """
