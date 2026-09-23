"""Tabular Counterfactual Regret Minimization (vanilla CFR).

CFR is the right tool for imperfect-information games: it converges to a Nash equilibrium of a
two-player zero-sum game by minimizing counterfactual regret at every information set. This is the
*tabular* form — it stores a regret and average-strategy vector per information set and walks the
full game tree each iteration, so it fits only games small enough to enumerate (Kuhn/Leduc-scale,
our ``toy_showdown``). Large games (Ventures, real poker) need Deep CFR / abstraction instead.

Game-agnostic: it drives the ``Game`` interface, with ``information_state_string`` as infoset key.
Information sets are assumed to fix the legal-action set (true when legality depends only on the
acting player's own observation), so each key maps to a stable action ordering.
"""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Return the current strategy from positive regrets (uniform if none are positive)."""
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    return np.full(len(regrets), 1.0 / len(regrets))


class CFRSolver:
    """Vanilla CFR for a two-player zero-sum ``Game``.

    After :meth:`train`, :meth:`average_strategy` returns the equilibrium approximation (the average
    strategy, which is what converges — not the final iterate).
    """

    def __init__(self, game: Game) -> None:
        """Create a solver for ``game`` (must be two-player zero-sum with infoset strings)."""
        if game.num_players != 2:
            raise ValueError("This CFR implementation is for two-player games")
        self.game = game
        self.regret_sum: dict[str, np.ndarray] = {}
        self.strategy_sum: dict[str, np.ndarray] = {}
        self.actions: dict[str, list[Action]] = {}

    def _lookup(self, key: str, legal: list[Action]) -> tuple[np.ndarray, np.ndarray]:
        """Return the (regret_sum, strategy_sum) vectors for an infoset, creating them if new."""
        if key not in self.regret_sum:
            self.regret_sum[key] = np.zeros(len(legal))
            self.strategy_sum[key] = np.zeros(len(legal))
            self.actions[key] = list(legal)
        return self.regret_sum[key], self.strategy_sum[key]

    def _cfr(self, state: State, reach: np.ndarray) -> np.ndarray:
        """Recurse, returning per-player expected utility; ``reach`` is ``[r0, r1, chance]``."""
        game = self.game
        if game.is_terminal(state):
            return np.array(game.returns(state), dtype=float)
        if game.is_chance_node(state):
            util = np.zeros(2)
            for action, prob in game.chance_outcomes(state):
                child_reach = reach.copy()
                child_reach[2] *= prob
                util += prob * self._cfr(game.apply_action(state, action), child_reach)
            return util

        player = game.current_player(state)
        legal = game.legal_actions(state)
        key = game.information_state_string(state, player)
        regret_sum, strategy_sum = self._lookup(key, legal)
        strategy = regret_matching(regret_sum)

        action_util = np.zeros((len(legal), 2))
        node_util = np.zeros(2)
        for i, action in enumerate(legal):
            child_reach = reach.copy()
            child_reach[player] *= strategy[i]
            action_util[i] = self._cfr(game.apply_action(state, action), child_reach)
            node_util += strategy[i] * action_util[i]

        # Counterfactual regret uses the reach of everyone *except* the acting player.
        counterfactual_reach = reach[1 - player] * reach[2]
        regret_sum += counterfactual_reach * (action_util[:, player] - node_util[player])
        strategy_sum += reach[player] * strategy
        return node_util

    def train(self, iterations: int) -> None:
        """Run ``iterations`` CFR traversals from the initial state."""
        for _ in range(iterations):
            self._cfr(self.game.new_initial_state(), np.ones(3))

    def average_strategy(self) -> tuple[dict[str, np.ndarray], dict[str, list[Action]]]:
        """Return the average strategy per infoset (normalized) and the infoset action lists."""
        avg: dict[str, np.ndarray] = {}
        for key, strat_sum in self.strategy_sum.items():
            total = strat_sum.sum()
            avg[key] = (
                strat_sum / total if total > 0 else np.full(len(strat_sum), 1.0 / len(strat_sum))
            )
        return avg, self.actions


class CFRAgent(Agent):
    """Plays a fixed strategy (typically a CFR average strategy) by information set."""

    def __init__(
        self,
        strategy: dict[str, np.ndarray],
        actions: dict[str, list[Action]],
        name: str = "cfr",
    ) -> None:
        """Wrap a per-infoset ``strategy`` (with its action lists) as an agent."""
        self.strategy = strategy
        self.actions = actions
        self.name = name

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Sample an action from the strategy at this information set (uniform if unseen)."""
        key = game.information_state_string(state, game.current_player(state))
        legal = game.legal_actions(state)
        if key not in self.strategy:
            return int(rng.choice(legal))
        return int(rng.choice(self.actions[key], p=self.strategy[key]))
