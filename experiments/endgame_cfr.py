"""Does endgame CFR solving improve the trained Ventures policy?

The policy plays as usual until the deck is nearly empty (``deck_threshold`` cards or fewer); from
then on each decision is solved by depth-limited Monte Carlo CFR (``iirl.agents.cfr.subgame``) over
worlds sampled from the player's view, with the game's ``heuristic_value`` at the depth limit.
It is rated against the same policy without the solver, and against the careful heuristic, on
duplicate deals.

Run with::

    uv run python -m experiments.endgame_cfr
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import iirl.games  # noqa: F401  (registers "ventures")
from experiments.exploit_ventures import ppo_agent
from iirl.agents.cfr.subgame import EndgameCFRAgent, SubgameConfig
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make, make_heuristic
from iirl.eval.duplicate import duplicate_match


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    policy: str = "runs/careful.pt"
    deck_threshold: int = 8
    depth: int = 4
    iterations: int = 300
    deals: int = 100
    workers: int = 6


class DeckTrigger:
    """Solve once the deck holds ``threshold`` cards or fewer (picklable, unlike a lambda)."""

    def __init__(self, threshold: int) -> None:
        """Set the threshold."""
        self.threshold = threshold

    def __call__(self, game, state) -> bool:  # noqa: ANN001
        """Whether to solve this decision."""
        return sum(state.deck) <= self.threshold


def endgame_agent(path: str, threshold: int, depth: int, iterations: int) -> EndgameCFRAgent:
    """Build the policy-plus-endgame-CFR agent (a picklable factory for worker processes)."""
    return EndgameCFRAgent(
        PPOAgent.from_checkpoint(path),
        DeckTrigger(threshold),
        SubgameConfig(iterations=iterations, depth=depth),
    )


def main(config: Config) -> None:
    """Rate the endgame-CFR agent against its own policy and the careful heuristic."""
    game = make("ventures")
    solver = (
        endgame_agent,
        {
            "path": config.policy,
            "threshold": config.deck_threshold,
            "depth": config.depth,
            "iterations": config.iterations,
        },
    )
    opponents = {
        "its own policy": (ppo_agent, {"path": config.policy}),
        "careful heuristic": (make_heuristic, {"short_name": "ventures_careful"}),
    }
    for name, spec in opponents.items():
        t = time.perf_counter()
        r = duplicate_match(game, solver, spec, config.deals, seed=71, workers=config.workers)
        print(
            f"endgame CFR vs {name}: {r.score:.3f}+-{r.score_se:.3f}  margin {r.margin:+.1f}"
            f"+-{r.margin_se:.1f}  ({r.deals} deals, {time.perf_counter() - t:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main(Config())
