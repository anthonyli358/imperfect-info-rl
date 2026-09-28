"""The ``Agent`` interface.

Agents are written purely against ``Game`` and contain zero game-specific code: given a state where
it is their turn, they return a legal action. The runner never asks an agent to act at a chance or
terminal node.

Stateful agents (e.g. one tracking a belief over hidden cards) also get :meth:`Agent.reset` at the
start of each game and :meth:`Agent.observe` after every transition. Both are no-ops by default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from iirl.core.game import Game, State
from iirl.core.types import Action, Player


class Agent(ABC):
    """Base class for a policy over any ``Game``.

    Attributes:
        name: Human-readable label, used in tournament standings and error messages.
    """

    name: str = "agent"

    def reset(self, game: Game, player: Player) -> None:  # noqa: B027 - optional hook
        """Called before a game in which this agent plays seat ``player``."""

    def observe(self, game: Game, state: State, action: Action | None) -> None:  # noqa: B027
        """Called after every transition with the new ``state`` and the ``action`` that led there.

        ``action`` is ``None`` for chance events, so a draw's outcome is never revealed through it.
        ``state`` is the full state: an agent must read only its own information state of it.
        """

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
