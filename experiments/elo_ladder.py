"""Elo ladder: rate every checkpoint in the Ventures agent lineage on one fixed yardstick.

Each checkpoint plays the heuristic (anchored at 1676 Elo) over the same duplicate deals, so the
numbers are comparable across runs (about ±10 Elo at 1000 deals, vs ±35 for 400-game in-training
evals). Results go to ``runs/elo_ladder.json``; checkpoints already rated there are skipped, so
this is cheap to rerun as the lineage grows, and safe to stop at any time.

Run with::

    uv run python -m experiments.elo_ladder              # Ventures
    uv run python -m experiments.elo_ladder camel_stocks # Camel Stocks (its heuristic = 1500)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import iirl.games  # noqa: F401  (registers "ventures")
from experiments.exploit_ventures import heuristic_agent, ppo_agent
from iirl.agents.base import Agent
from iirl.agents.heuristic import CamelStocksHeuristic
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.eval.tournament import anchored_elo

HEURISTIC_ELO = 1676.0


@dataclass(frozen=True)
class Config:
    """Experiment settings.

    Attributes:
        lineage: ``(label, checkpoint)`` in training order; missing checkpoints are skipped.
        deals: Duplicate deals per rating (``2 *`` games).
        out: The results file.
        game: Registered game name.
        heuristic: Factory for the anchor agent.
        anchor_elo: The anchor's fixed rating.
    """

    lineage: tuple[tuple[str, str], ...] = (
        ("PPO v2", "runs/ppo_ventures_best.pt"),
        ("nash control", "runs/nash_control.pt"),
        ("nash", "runs/nash_nash.pt"),
        *((f"nash_long {i}", f"runs/nash_long_{i}.pt") for i in range(100, 700, 100)),
        *((f"league r{r}", f"runs/league_r{r}.pt") for r in range(1, 6)),
        ("league control", "runs/league_control.pt"),
        *((f"nash_g {i}", f"runs/nash_g_{i}.pt") for i in range(100, 500, 100)),
    )
    deals: int = 1000
    out: str = "runs/elo_ladder.json"
    game: str = "ventures"
    heuristic: Callable[[], Agent] = heuristic_agent
    anchor_elo: float = HEURISTIC_ELO


CAMEL_STOCKS = Config(
    lineage=(
        *((f"camel_ppo {i}", f"runs/camel_ppo_{i}.pt") for i in range(100, 700, 100)),
        *((f"camel_nash {i}", f"runs/camel_nash_{i}.pt") for i in range(100, 500, 100)),
    ),
    out="runs/elo_ladder_camel_stocks.json",
    game="camel_stocks",
    heuristic=CamelStocksHeuristic,
    anchor_elo=1500.0,
)


def main(config: Config) -> None:
    """Rate each unrated checkpoint vs the heuristic; print and save the ladder."""
    game = make(config.game)
    out = Path(config.out)
    ladder = json.loads(out.read_text("utf-8")) if out.exists() else {}
    for label, path in config.lineage:
        if label in ladder or not Path(path).exists():
            continue
        r = duplicate_match(game, (ppo_agent, {"path": path}), (config.heuristic, {}), config.deals)
        ladder[label] = {
            "checkpoint": path,
            "elo": round(anchored_elo(r.score, config.anchor_elo), 1),
            # Elo is logistic in the score, so +-1 SE on the score maps to this Elo error.
            "elo_se": round(
                (
                    anchored_elo(r.score + r.score_se, config.anchor_elo)
                    - anchored_elo(r.score - r.score_se, config.anchor_elo)
                )
                / 2,
                1,
            ),
            "score": round(r.score, 4),
            "margin": round(r.margin, 2),
            "deals": r.deals,
        }
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(ladder, indent=2), "utf-8")
        tmp.replace(out)
        print(
            f"{label:<16} elo {ladder[label]['elo']:6.0f} ± {ladder[label]['elo_se']:.0f}",
            flush=True,
        )
    order = [label for label, _ in config.lineage if label in ladder]
    print("\nlabel            elo      score   margin")
    for label in order:
        e = ladder[label]
        print(
            f"{label:<16} {e['elo']:6.0f}±{e['elo_se']:<3.0f} {e['score']:.3f}  {e['margin']:+6.1f}"
        )


if __name__ == "__main__":
    import sys

    main(CAMEL_STOCKS if sys.argv[1:] == ["camel_stocks"] else Config())
