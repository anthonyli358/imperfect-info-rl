"""Stage 0 smoke test: run a round-robin of RandomAgents and print the standings.

This demonstrates the repo conventions end-to-end — a module-level config dataclass, the registry,
the tournament runner — and proves the spine works (Stage 0 "done when": agents play a registered
game to completion through the runner).

Run it with::

    uv run python -m experiments.random_vs_random

The only game that exists in Stage 0 is the ``toy_showdown`` test fixture, so this script reaches
into ``tests/`` to register it. From Stage 1 (Lost Cities) on, games live in ``iirl.games`` and this
path insertion goes away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from iirl.agents.random_agent import RandomAgent
from iirl.config.base import TournamentConfig
from iirl.core.registry import make
from iirl.eval.tournament import run_tournament

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import toy_game  # noqa: E402, F401  (import registers "toy_showdown")


def main(config: TournamentConfig) -> None:
    """Run the tournament described by ``config`` and print the standings table."""
    game = make(config.game)
    rng = np.random.default_rng(config.seed)
    agents = {name: RandomAgent(name) for name in ("alice", "bob", "carol")}
    standings = run_tournament(game, agents, config.games_per_pair, rng, elo_k=config.elo_k)

    print(f"{config.game}  ({config.games_per_pair} games/pair, seed={config.seed})")
    print(f"{'agent':<8} {'elo':>7}  {'W':>4} {'D':>4} {'L':>4}")
    for s in standings:
        print(f"{s.name:<8} {s.elo:>7.1f}  {s.wins:>4} {s.draws:>4} {s.losses:>4}")


if __name__ == "__main__":
    main(TournamentConfig())
