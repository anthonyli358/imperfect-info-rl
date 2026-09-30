"""Solve Mini-Ventures exactly: vectorized CFR+ over its 2.3M-state graph.

Runs ``iirl.agents.cfr.graph_cfr`` in chunks, measuring the average strategy's exact
exploitability (points per game, ``iirl.eval.exact_br``) after each, until it is below
``target`` or ``max_iterations`` is reached. Saves the average strategy and the game value to
``runs/mini/equilibrium.npz``: the ground-truth reference for everything else on Mini-Ventures.

Run with::

    uv run python -m experiments.mini_equilibrium
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import iirl.games  # noqa: F401  (registers "mini_ventures")
from iirl.agents.cfr.graph_cfr import GraphCFR
from iirl.core.registry import make
from iirl.eval.exact_br import exploitability


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    chunk: int = 100
    max_iterations: int = 3000
    target: float = 0.02  # points per game
    out: str = "runs/mini/equilibrium.npz"


def main(config: Config) -> None:
    """Iterate CFR+ until the average strategy is within ``target`` of an equilibrium."""
    game = make("mini_ventures")
    t0 = time.perf_counter()
    cfr = GraphCFR(game)
    print(
        f"graph and index: {len(cfr.infoset_keys):,} infosets, {cfr.num_slots:,} actions, "
        f"{time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    while cfr.t < config.max_iterations:
        t = time.perf_counter()
        cfr.iterate(config.chunk)
        per_it = (time.perf_counter() - t) / config.chunk
        result = exploitability(game, cfr.average_policy(), cfr.graph)
        print(
            f"iteration {cfr.t:>5}: exploitability {result['exploitability']:.4f} points/game "
            f"(br as P0 {result['br_as_0']:+.3f}, as P1 {result['br_as_1']:+.3f}); "
            f"{per_it:.2f}s/iteration",
            flush=True,
        )
        Path(config.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            config.out,
            keys=np.asarray(cfr.infoset_keys),
            strategy_sum=cfr.strategy_sum,
            offset=cfr.offset,
            iterations=cfr.t,
            exploitability=result["exploitability"],
        )
        if result["exploitability"] < config.target:
            break
    # Game value for player 0 under the (near-)equilibrium: halfway between the best responses.
    print(f"game value for the first player ~ {(result['br_as_0'] - result['br_as_1']) / 2:+.3f}")


if __name__ == "__main__":
    main(Config())
