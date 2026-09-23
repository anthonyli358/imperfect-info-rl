"""Generic episode runner.

Drives any number of agents through any registered ``Game`` to a terminal state, resolving chance
nodes itself from a seeded generator. This is the single place game play happens; the tournament and
tests build on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from iirl.core.game import Game, State

if TYPE_CHECKING:
    # Imported for typing only; a runtime import would create a core <-> agents cycle
    # (agents.base imports core.game, which triggers this package's __init__).
    from iirl.agents.base import Agent


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single played episode.

    Attributes:
        returns: Per-player payoffs at the terminal state.
        num_steps: Number of transitions applied (chance and player moves).
    """

    returns: list[float]
    num_steps: int


def _sample_chance(game: Game, state: State, rng: np.random.Generator) -> int:
    """Sample a chance action from ``chance_outcomes`` under ``rng``."""
    outcomes = game.chance_outcomes(state)
    actions = [a for a, _ in outcomes]
    probs = np.array([p for _, p in outcomes], dtype=float)
    return int(rng.choice(actions, p=probs))


def run_episode(
    game: Game,
    agents: Sequence[Agent],
    rng: np.random.Generator,
    max_steps: int = 10_000,
) -> EpisodeResult:
    """Play one episode of ``game`` with ``agents`` (indexed by player) to completion.

    Chance nodes are resolved by sampling from ``game.chance_outcomes`` with ``rng``; player nodes
    ask the corresponding agent for an action, which is validated against ``legal_actions`` before
    being applied.

    Args:
        game: The game to play.
        agents: One agent per player, indexed by player id.
        rng: Seeded generator driving both chance resolution and stochastic agents.
        max_steps: Safety cap; exceeding it raises (guards against non-terminating engines).

    Returns:
        The :class:`EpisodeResult` for the finished episode.

    Raises:
        ValueError: If ``agents`` has the wrong length or an agent returns an illegal action.
        RuntimeError: If the episode does not terminate within ``max_steps``.
    """
    if len(agents) != game.num_players:
        raise ValueError(f"Expected {game.num_players} agents, got {len(agents)}")

    state = game.new_initial_state()
    for step in range(max_steps):
        if game.is_terminal(state):
            return EpisodeResult(returns=game.returns(state), num_steps=step)
        if game.is_chance_node(state):
            state = game.apply_action(state, _sample_chance(game, state, rng))
            continue
        player = game.current_player(state)
        action = agents[player].step(game, state, rng)
        if action not in game.legal_actions(state):
            raise ValueError(
                f"Agent {agents[player].name!r} chose illegal action {action} for player {player}"
            )
        state = game.apply_action(state, action)

    raise RuntimeError(f"Episode did not terminate within {max_steps} steps")
