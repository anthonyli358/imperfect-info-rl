"""Continue the plain Nash recipe from the current best agent, with the endgame-stall fix.

Starts from ``runs/league_control.pt`` (1973 Elo), trains with the Nash recipe (magnet KL, snapshot
pool, heuristic in the league) plus ``gamma = 0.999``, so a winning player prefers ending the game
over stalling to the turn cap. Keeps ``runs/<tag>_<iteration>.pt`` every 100 iterations for the
Elo ladder, then measures exploitability as the max over two exploiter runs (a single run is only a
lower bound; see ``writeup.md``).

Resumable: training continues from its last eval checkpoint, and finished exploiter runs are
skipped. Stop at any time and rerun the same command.

``RNAD`` is the same run with one change: the magnet is a slowly moving average of the policy's
weights (``magnet_ema``) rather than a copy refreshed every 20 iterations. Its control is the
plain run (``nash_g``): same start, budget and settings otherwise.

Run with::

    uv run python -m experiments.train_ventures_continue       # nash_g
    uv run python -m experiments.train_ventures_continue rnad  # the R-NaD-style variant
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import iirl.games  # noqa: F401  (registers "ventures")
from experiments import exploit_ventures
from experiments import train_ventures_nash as nash
from experiments.train_ventures_league import _train
from iirl.selfplay import PPOConfig


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    start: str = "runs/league_control.pt"
    tag: str = "nash_g"
    iterations: int = 400
    save_every: int = 100
    exploit_seeds: tuple[int, ...] = (0, 1)
    ppo: PPOConfig = field(
        default_factory=lambda: replace(
            nash.variant("nash", nash.Config().ppo), gamma=0.999, seed=3
        )
    )


RNAD = Config(
    tag="rnad",
    ppo=replace(nash.variant("nash", nash.Config().ppo), gamma=0.999, seed=3, magnet_ema=0.02),
)


def main(config: Config) -> None:
    """Train, then exploit with each seed; print the exploitability estimate (the max)."""
    ppo = replace(config.ppo, iterations=config.iterations)
    path = _train(config.start, ppo, config.tag, save_every=config.save_every)
    scores = []
    for seed in config.exploit_seeds:
        name = f"{config.tag}_s{seed}"
        record = Path(f"runs/exploit_{name}.json")
        if not record.exists():
            ex = exploit_ventures.Config()
            exploit_ventures.main(
                replace(ex, target=path, name=name, ppo=replace(ex.ppo, seed=seed))
            )
        scores.append(float(json.loads(record.read_text("utf-8"))["final"]["score"]))
    print(f"\nexploitability of {config.tag}: max {max(scores):.3f}  (runs: {scores})")


if __name__ == "__main__":
    import sys

    main(RNAD if sys.argv[1:] == ["rnad"] else Config())
