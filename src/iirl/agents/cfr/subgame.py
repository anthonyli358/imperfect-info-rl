"""Depth-limited Monte Carlo CFR for solving the rest of a game from the current position.

Tabular CFR walks the whole tree, which only fits toy games. To use CFR in Ventures or Camel
Stocks, solve just the *next few decisions* from the current position, when the tree is small
(typically the endgame), and act on the result:

* **Worlds.** Each iteration starts from a world sampled by ``sample_world`` (the opponent's hidden
  cards re-dealt, uniformly or from a belief filter). Chance events inside (deck draws) are sampled.
* **External-sampling MCCFR** (Lanctot et al., 2009). Each iteration, each player in turn is the
  *traverser*: every one of the traverser's actions is explored, the opponent's and chance's are
  sampled. Regret matching+ with linearly weighted averaging (as in CFR+) speeds convergence.
* **Depth limit.** After ``depth`` player decisions below the root, the position is scored by a
  leaf evaluator (the game's ``heuristic_value`` by default) instead of being searched further.
  Terminal positions use the true payoff.
* **Acting.** The root player plays the average strategy at their real information set.

The key approximation, stated plainly: the sampled worlds re-deal only the opponent's hidden cards,
so the root player's own hand is the same in every world. In the solved subgame the opponent
therefore plays as if they knew our hand. That is pessimistic (safe-leaning), and it removes any
value from bluffing, which matters little in games where almost every action is public (Ventures,
Camel Stocks) but a lot in poker. ``tests/test_cfr_subgame.py`` shows both sides: sampling full
deals recovers the toy game's mixed equilibrium; fixing our hand makes the low-card bluff vanish.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action, Player

LeafValue = Callable[[Game, State, Player], float]


@dataclass(frozen=True)
class SubgameConfig:
    """Solver settings.

    Attributes:
        iterations: MCCFR iterations (each one traversal per player).
        depth: Player decisions below the root before the leaf evaluator scores the position.
        sample: Play by sampling the average strategy; ``False`` plays its most likely action.
    """

    iterations: int = 300
    depth: int = 4
    sample: bool = False


def heuristic_leaf(game: Game, state: State, player: Player) -> float:
    """Leaf value from the game's ``heuristic_value`` (0 if it provides none)."""
    value = game.heuristic_value(state, player)
    return 0.0 if value is None else float(value)


class SubgameSolver:
    """External-sampling MCCFR over a depth-limited subgame."""

    def __init__(self, game: Game, config: SubgameConfig, leaf: LeafValue = heuristic_leaf) -> None:
        """Create a solver for ``game``."""
        self.game, self.config, self.leaf = game, config, leaf
        self.regrets: dict[str, np.ndarray] = {}
        self.strategy_sum: dict[str, np.ndarray] = {}
        self.actions: dict[str, list[Action]] = {}
        self.t = 0

    def _node(self, key: str, legal: list[Action]) -> tuple[np.ndarray, np.ndarray]:
        if key not in self.regrets:
            self.regrets[key] = np.zeros(len(legal))
            self.strategy_sum[key] = np.zeros(len(legal))
            self.actions[key] = list(legal)
        return self.regrets[key], self.strategy_sum[key]

    @staticmethod
    def _current(regrets: np.ndarray) -> np.ndarray:
        positive = np.maximum(regrets, 0.0)
        total = positive.sum()
        return positive / total if total > 0 else np.full(len(regrets), 1.0 / len(regrets))

    def _traverse(self, state: State, traverser: Player, depth: int, rng) -> float:  # noqa: ANN001
        game = self.game
        if game.is_terminal(state):
            return float(game.returns(state)[traverser])
        if game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            i = rng.choice(len(outcomes), p=[p for _, p in outcomes])
            return self._traverse(game.apply_action(state, outcomes[i][0]), traverser, depth, rng)
        if depth >= self.config.depth:
            return self.leaf(game, state, traverser)
        player = game.current_player(state)
        legal = game.legal_actions(state)
        regrets, strategy_sum = self._node(game.information_state_string(state, player), legal)
        strategy = self._current(regrets)
        if player == traverser:
            utils = np.array(
                [
                    self._traverse(game.apply_action(state, a), traverser, depth + 1, rng)
                    for a in legal
                ]
            )
            value = float(strategy @ utils)
            regrets += utils - value
            np.maximum(regrets, 0.0, out=regrets)  # regret matching+
            return value
        strategy_sum += self.t * strategy  # linear averaging
        a = legal[int(rng.choice(len(legal), p=strategy))]
        return self._traverse(game.apply_action(state, a), traverser, depth + 1, rng)

    def run(self, sample_world: Callable[[np.random.Generator], State], rng) -> None:  # noqa: ANN001
        """Run the configured iterations, each from a freshly sampled world."""
        for _ in range(self.config.iterations):
            self.t += 1
            for traverser in (0, 1):
                self._traverse(sample_world(rng), traverser, 0, rng)

    def average_strategy(self) -> dict[str, np.ndarray]:
        """Return the normalized average strategy per information set."""
        out = {}
        for key, total in self.strategy_sum.items():
            s = total.sum()
            out[key] = total / s if s > 0 else np.full(len(total), 1.0 / len(total))
        return out

    def policy_at(self, key: str) -> tuple[list[Action], np.ndarray] | None:
        """Return ``(actions, probabilities)`` at information set ``key``, if it was reached."""
        if key not in self.strategy_sum:
            return None
        return self.actions[key], self.average_strategy()[key]


class EndgameCFRAgent(Agent):
    """Plays a base agent, and switches to depth-limited CFR solving when ``trigger`` says so.

    ``trigger(game, state)`` decides when the position is small enough to solve (e.g. few cards
    left in the deck). Worlds come from the ``belief`` filter if given, else from uniform
    ``resample_from_infostate``.
    """

    def __init__(
        self,
        base: Agent,
        trigger: Callable[[Game, State], bool],
        config: SubgameConfig | None = None,
        leaf: LeafValue = heuristic_leaf,
        belief=None,  # noqa: ANN001 - ParticleBelief (optional)
        name: str = "endgame_cfr",
    ) -> None:
        """Wrap ``base``; solve with CFR wherever ``trigger`` holds."""
        self.base, self.trigger, self.leaf, self.belief = base, trigger, leaf, belief
        self.config = config or SubgameConfig()
        self.name = name
        self.solved = 0

    def reset(self, game: Game, player: Player) -> None:
        """Reset the base agent (and belief) for a new game."""
        self.base.reset(game, player)
        if self.belief is not None:
            self.belief.reset(game, player)

    def observe(self, game: Game, state: State, action: Action | None) -> None:
        """Forward observations to the base agent (and belief)."""
        self.base.observe(game, state, action)
        if self.belief is not None:
            self.belief.observe(state, action)

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Solve the subgame if triggered, else defer to the base agent."""
        if not self.trigger(game, state):
            return self.base.step(game, state, rng)
        me = game.current_player(state)
        if self.belief is not None:
            sample = lambda r: self.belief.sample(state, r)  # noqa: E731
        else:
            sample = lambda r: game.resample_from_infostate(state, me, r)  # noqa: E731
        solver = SubgameSolver(game, self.config, self.leaf)
        solver.run(sample, rng)
        found = solver.policy_at(game.information_state_string(state, me))
        if found is None:
            return self.base.step(game, state, rng)
        self.solved += 1
        actions, probs = found
        if self.config.sample:
            return int(actions[int(rng.choice(len(actions), p=probs))])
        return int(actions[int(np.argmax(probs))])
