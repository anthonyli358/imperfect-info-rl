"""The AlphaZero agent: pick a move from net-guided PUCT visit counts."""

from __future__ import annotations

import numpy as np

from iirl.agents.alphazero.mcts import PUCTConfig, az_search
from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action
from iirl.nn.policy_value import NetEvaluator


class AlphaZeroAgent(Agent):
    """Plays the action favored by determinized net-guided search.

    With ``temperature == 0`` (evaluation) it takes the most-visited action; with a positive
    temperature it samples ``∝ visits**(1/temperature)`` (self-play exploration).
    """

    def __init__(
        self,
        evaluator: NetEvaluator,
        config: PUCTConfig | None = None,
        name: str = "alphazero",
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> None:
        """Configure the agent around a network ``evaluator``."""
        self.evaluator = evaluator
        self.config = config or PUCTConfig()
        self.name = name
        self.temperature = temperature
        self.add_noise = add_noise

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return a move from the pooled search visit counts."""
        me = game.current_player(state)
        counts = az_search(game, state, me, self.evaluator, self.config, rng, self.add_noise)
        legal = game.legal_actions(state)
        weights = np.array([counts[a] for a in legal], dtype=float)
        if weights.sum() == 0:
            return int(rng.choice(legal))
        if self.temperature <= 0:
            return int(legal[int(weights.argmax())])
        scaled = weights ** (1.0 / self.temperature)
        return int(rng.choice(legal, p=scaled / scaled.sum()))
