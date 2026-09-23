"""Stage 1 eval: Random vs Heuristic vs PIMC on Ventures.

Confirms the strength ordering (random < heuristic < PIMC) that Stage 1 is "done when" it shows.
Games are long, so PIMC is the slow part; the budget below runs the whole tournament in a couple of
minutes. Scale ``MCTSConfig`` (more determinizations / sims, or a positive ``max_rollout_depth``)
for a stronger — and slower — agent.

Run with::

    uv run python -m experiments.ventures_tournament
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.mcts import MCTSConfig, PIMCAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.eval.tournament import run_tournament


@dataclass(frozen=True)
class Config:
    """Tournament settings."""

    seed: int = 0
    games_per_pair: int = 20
    mcts: MCTSConfig = field(
        default_factory=lambda: MCTSConfig(
            num_determinizations=12, sims_per_world=80, max_rollout_depth=0
        )
    )


def main(config: Config) -> None:
    """Run the Ventures tournament and print the standings table."""
    game = make("ventures")
    rng = np.random.default_rng(config.seed)
    agents = {
        "random": RandomAgent("random"),
        "heuristic": VenturesHeuristic("heuristic"),
        "pimc": PIMCAgent(config.mcts, name="pimc"),
    }
    standings = run_tournament(game, agents, config.games_per_pair, rng)

    print(f"Ventures tournament  ({config.games_per_pair} games/pair, seed={config.seed})")
    print(f"{'agent':<10} {'elo':>7}  {'W':>4} {'D':>4} {'L':>4}")
    for s in standings:
        print(f"{s.name:<10} {s.elo:>7.1f}  {s.wins:>4} {s.draws:>4} {s.losses:>4}")


if __name__ == "__main__":
    main(Config())
