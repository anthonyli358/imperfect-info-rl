"""PSRO building blocks: the meta-game solver and a mixture-of-policies agent.

Policy-Space Response Oracles (Lanctot et al., 2017) keeps a *population* of policies and the
payoff matrix between them (the "meta-game"). Each round it solves the meta-game for its Nash
mixture, trains a best response to that mixture, and adds it. The best response's score against
the mixture is the mixture's exploitability estimate; if it keeps falling as the population grows,
the mixture is approaching an equilibrium of the full game.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action, Player


def solve_symmetric_meta_game(
    scores: np.ndarray, iterations: int = 20000
) -> tuple[np.ndarray, float]:
    """Return the Nash mixture of a symmetric zero-sum meta-game and the best reply's score to it.

    Args:
        scores: ``scores[i, j]`` is population member ``i``'s expected score against ``j`` (win 1,
            draw 0.5, loss 0), so ``scores + scores.T == 1``.
        iterations: Regret-matching iterations (the average strategy converges to a Nash mixture).

    Returns:
        ``(mixture, exploitability)``: the average strategy, and the best score any *population
        member* gets against it (0.5 at an exact equilibrium of the meta-game).
    """
    payoff = np.asarray(scores, dtype=float) - 0.5  # antisymmetric: the row player's advantage
    n = len(payoff)
    regrets = np.zeros(n)
    strategy_sum = np.zeros(n)
    for _ in range(iterations):
        positive = np.maximum(regrets, 0.0)
        strategy = positive / positive.sum() if positive.sum() > 0 else np.full(n, 1.0 / n)
        strategy_sum += strategy
        # Self-play regret matching on a symmetric game: the opponent plays the same strategy.
        values = payoff @ strategy
        regrets += values - strategy @ values
    mixture = strategy_sum / strategy_sum.sum()
    return mixture, float((np.asarray(scores) @ mixture).max())


class MixtureAgent(Agent):
    """Plays one member of a population per game, chosen by the mixture weights at ``reset``."""

    def __init__(
        self,
        members: Sequence[Agent],
        weights: Sequence[float],
        seed: int | None = None,
        name: str = "mixture",
    ) -> None:
        """Mix ``members`` with probabilities ``weights``."""
        self.members = list(members)
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self.current: Agent | None = None
        self.name = name

    def reset(self, game: Game, player: Player) -> None:
        """Pick this game's member."""
        self.current = self.members[int(self.rng.choice(len(self.members), p=self.weights))]
        self.current.reset(game, player)

    def observe(self, game: Game, state: State, action: Action | None) -> None:
        """Forward observations to this game's member."""
        if self.current is not None:
            self.current.observe(game, state, action)

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Act as this game's member (picking one if ``reset`` was never called)."""
        if self.current is None:
            self.current = self.members[int(self.rng.choice(len(self.members), p=self.weights))]
        return self.current.step(game, state, rng)


def mixture_agent(
    factory: Callable[..., Agent], member_kwargs: Sequence[dict[str, Any]], weights: Sequence[float]
) -> MixtureAgent:
    """Build a :class:`MixtureAgent` from per-member factory kwargs (a picklable spec)."""
    members = [factory(**kw) for kw in member_kwargs]
    return MixtureAgent(members, weights)
