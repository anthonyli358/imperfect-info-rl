"""A blueprint policy improved by paired rollouts where they can resolve a difference.

The audit in ``experiments/ventures_search_audit.py`` found that on Ventures a single paired
rollout's difference between two moves has a spread of 20-40 points, so rollouts can only tell
moves apart late in the game, when few cards remain, and that there the blueprint's move is
about a point a turn short of the best. This agent acts on that: it plays the blueprint, except
that once the deck is down to ``deck_max`` cards it plays every first move of its turn out to the
end of the game with the blueprint, in ``worlds`` re-deals of what it cannot see and ``orders``
shuffles of the deck in each (the same worlds and orders for every move, so the luck cancels in
the differences), and switches to the best move only if its paired advantage over the blueprint's
move is at least ``z`` standard errors. The second decision of a turn stays the blueprint's.

Game hooks: ``resample_from_infostate`` and a ``deck`` of card counts on the state (the deck's
order is fixed per rollout so that all moves see the same future cards).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iirl.agents.base import Agent
from iirl.agents.ppo.agent import PPOAgent
from iirl.core.game import Game, State
from iirl.core.types import Action


@dataclass(frozen=True)
class RolloutConfig:
    """Settings for :class:`RolloutImprover`."""

    blueprint: str
    deck_max: int = 6
    worlds: int = 8
    orders: int = 8
    z: float = 2.0
    seed: int = 0


def play_out(game: Game, state: State, agent: Agent, order: list[int], rng) -> list[float]:  # noqa: ANN001
    """Finish the game with ``agent`` in both seats, drawing the deck in the given order."""
    k = 0
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.apply_action(state, order[k])
            k += 1
        else:
            state = game.apply_action(state, agent.step(game, state, rng))
    return game.returns(state)


class RolloutImprover(Agent):
    """The blueprint, overridden late in the game where paired rollouts show a clear better move."""

    def __init__(self, config: RolloutConfig, name: str = "rollout") -> None:
        """Load the blueprint (played greedily, both as the agent and inside rollouts)."""
        self.config, self.name = config, name
        self.blueprint = PPOAgent.from_checkpoint(config.blueprint, greedy=True)
        self.rng = np.random.default_rng(config.seed)
        self.searched = 0
        self.switched = 0

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Play the blueprint's move unless rollouts show a clearly better first move."""
        cfg = self.config
        default = self.blueprint.step(game, state, rng)
        me = game.current_player(state)
        opening_a_turn = sum(game.private_hand(state, me)) >= sum(game.private_hand(state, 1 - me))
        legal = game.legal_actions(state)
        if sum(state.deck) > cfg.deck_max or not opening_a_turn or len(legal) < 2:
            return default
        self.searched += 1
        margins = np.zeros((cfg.worlds * cfg.orders, len(legal)))
        row = 0
        for _ in range(cfg.worlds):
            world = game.resample_from_infostate(state, me, self.rng)
            for _ in range(cfg.orders):
                order = np.repeat(np.arange(len(world.deck)), world.deck)
                self.rng.shuffle(order)
                order = order.tolist()
                for k, a in enumerate(legal):
                    after = game.apply_action(world, a)
                    margins[row, k] = play_out(game, after, self.blueprint, order, rng)[me]
                row += 1
        base = legal.index(default)
        best = int(np.argmax(margins.mean(axis=0)))
        if best == base:
            return default
        gain = margins[:, best] - margins[:, base]
        error = gain.std(ddof=1) / np.sqrt(len(gain))
        if gain.mean() < cfg.z * error or error == 0 and gain.mean() <= 0:
            return default
        self.switched += 1
        return int(legal[best])


def rollout_agent(**settings: object) -> RolloutImprover:
    """Build a :class:`RolloutImprover` from config fields (a picklable spec for matches)."""
    return RolloutImprover(RolloutConfig(**settings))
