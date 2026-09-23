"""PIMC: Perfect-Information Monte Carlo (determinized UCT) agent.

The agent never reasons about a game's hidden structure itself. Each move it asks the game for K
worlds consistent with its information state (``game.resample_from_infostate``), runs independent
UCT in each treating it as perfect-information-with-chance, and ranks each root action by its pooled
mean value across worlds. Being game-agnostic, the same agent runs on any imperfect-info game that
implements ``resample_from_infostate`` (Ventures now; Jaipur in Stage 2, unchanged).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from iirl.agents.base import Agent
from iirl.agents.mcts.uct import uct_search
from iirl.core.game import Game, State
from iirl.core.types import Action


@dataclass(frozen=True)
class MCTSConfig:
    """PIMC search budget.

    Attributes:
        num_determinizations: Worlds sampled per move (K).
        sims_per_world: UCT iterations per world.
        uct_c: UCB1 exploration constant.
        max_rollout_depth: Truncate random rollouts after this many steps and use the game's
            ``heuristic_value`` leaf estimate; ``None`` rolls out fully to a terminal state.
    """

    num_determinizations: int = 12
    sims_per_world: int = 60
    uct_c: float = 1.4
    max_rollout_depth: int | None = 40


class PIMCAgent(Agent):
    """Determinized-UCT agent for imperfect-information games."""

    def __init__(self, config: MCTSConfig | None = None, name: str = "pimc") -> None:
        """Create a PIMC agent.

        Args:
            config: Search budget; defaults to :class:`MCTSConfig`.
            name: Label used in standings and error messages.
        """
        self.config = config or MCTSConfig()
        self.name = name

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return the most-voted action across determinized UCT searches."""
        legal = game.legal_actions(state)
        if len(legal) == 1:
            return legal[0]

        me = game.current_player(state)
        legal_set = set(legal)
        # Pool (visits, value) per action across worlds; rank by pooled mean value, which is robust
        # when the per-world budget is small relative to the branching factor.
        pooled_n: dict[Action, int] = defaultdict(int)
        pooled_w: dict[Action, float] = defaultdict(float)
        for _ in range(self.config.num_determinizations):
            world = game.resample_from_infostate(state, me, rng)
            stats = uct_search(
                game,
                world,
                me,
                self.config.sims_per_world,
                self.config.uct_c,
                rng,
                self.config.max_rollout_depth,
            )
            for action, (n, w) in stats.items():
                if action in legal_set:  # root actions match the true state's legal set
                    pooled_n[action] += n
                    pooled_w[action] += w

        if not pooled_n:  # degenerate budget; fall back to a random legal move
            return int(rng.choice(legal))
        return max(pooled_n, key=lambda a: pooled_w[a] / pooled_n[a])
