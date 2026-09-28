"""Elo ladder: rate every checkpoint in the Ventures agent lineage on one fixed yardstick.

Each checkpoint plays the heuristic (anchored at 1676 Elo) over the same duplicate deals, so the
numbers are comparable across runs (about ±10 Elo at 1000 deals, vs ±35 for 400-game in-training
evals). Results go to ``runs/elo_ladder.json``; checkpoints already rated there are skipped, so
this is cheap to rerun as the lineage grows, and safe to stop at any time.

Run with::

    uv run python -m experiments.elo_ladder
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import iirl.games  # noqa: F401  (registers "ventures")
from experiments.exploit_ventures import heuristic_agent, ppo_agent
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
    """

    lineage: tuple[tuple[str, str], ...] = (
        ("PPO v2", "runs/ppo_ventures_best.pt"),
        ("nash control", "runs/nash_control.pt"),
        ("nash", "runs/nash_nash.pt"),
        *((f"nash_long {i}", f"runs/nash_long_{i}.pt") for i in range(100, 700, 100)),
        *((f"league r{r}", f"runs/league_r{r}.pt") for r in range(1, 6)),
        ("league control", "runs/league_control.pt"),
    )
    deals: int = 1000
    out: str = "runs/elo_ladder.json"


def main(config: Config) -> None:
    """Rate each unrated checkpoint vs the heuristic; print and save the ladder."""
    game = make("ventures")
    out = Path(config.out)
    ladder = json.loads(out.read_text("utf-8")) if out.exists() else {}
    for label, path in config.lineage:
        if label in ladder or not Path(path).exists():
            continue
        r = duplicate_match(game, (ppo_agent, {"path": path}), (heuristic_agent, {}), config.deals)
        ladder[label] = {
            "checkpoint": path,
            "elo": round(anchored_elo(r.score, HEURISTIC_ELO), 1),
            # Elo is logistic in the score, so +-1 SE on the score maps to this Elo error.
            "elo_se": round(
                (
                    anchored_elo(r.score + r.score_se, HEURISTIC_ELO)
                    - anchored_elo(r.score - r.score_se, HEURISTIC_ELO)
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
    main(Config())
