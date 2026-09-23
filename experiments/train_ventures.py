"""Train a linear self-play policy on Ventures and record its Elo curve.

Elo is anchored to two fixed opponents: random (1000 by definition) and the heuristic (measured once
against random). The learned policy is rated against both every few updates, producing the training
curve. Outputs a checkpoint (``.npz``) and the curve (``.json``) into ``runs/``.

Run with::

    uv run python -m experiments.train_ventures
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.random_agent import RandomAgent
from iirl.agents.reinforce import TrainConfig, train
from iirl.core.registry import make
from iirl.eval.tournament import anchored_elo, play_match


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    out_dir: str = "runs"
    seed: int = 0
    anchor_games: int = 200  # games to pin the heuristic's Elo vs random
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            updates=60,
            games_per_update=48,
            lr=0.03,
            entropy_coef=0.01,
            eval_interval=5,
            eval_games=80,
            feature_sample_games=40,
            seed=0,
        )
    )


def main(config: Config) -> None:
    """Measure the heuristic anchor, train the policy, and save the checkpoint + Elo curve."""
    game = make("ventures")
    rng = np.random.default_rng(config.seed)

    heuristic = VenturesHeuristic("heuristic")
    random_agent = RandomAgent("random")
    hvr = play_match(game, heuristic, random_agent, config.anchor_games, rng)
    heuristic_elo = anchored_elo((hvr.wins + 0.5 * hvr.draws) / hvr.num_games, 1000.0)
    print(
        f"anchors: random 1000.0, heuristic {heuristic_elo:.0f} "
        f"(won {hvr.wins}/{hvr.num_games} vs random)"
    )

    anchors = {"random": random_agent, "heuristic": heuristic}
    train_cfg = TrainConfig(
        **{**config.train.__dict__, "anchor_elos": {"random": 1000.0, "heuristic": heuristic_elo}}
    )

    print(f"{'update':>7} {'games':>7} {'Elo':>8}  vs random  vs heuristic")

    def report(pt) -> None:  # noqa: ANN001 - EvalPoint
        print(
            f"{pt.update:>7} {pt.games_seen:>7} {pt.elo:>8.1f}"
            f"     {pt.win_rate_vs['random']:.2f}       {pt.win_rate_vs['heuristic']:.2f}"
        )

    policy, history = train(game, anchors, train_cfg, on_eval=report)

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy.save(str(out / "reinforce_ventures.npz"))
    curve = {
        "game": "ventures",
        "anchors": {"random": 1000.0, "heuristic": heuristic_elo},
        "points": [
            {
                "update": p.update,
                "games": p.games_seen,
                "elo": p.elo,
                "win_vs_random": p.win_rate_vs["random"],
                "win_vs_heuristic": p.win_rate_vs["heuristic"],
            }
            for p in history
        ],
    }
    (out / "elo_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(f"\nsaved {out / 'reinforce_ventures.npz'} and {out / 'elo_curve.json'}")
    print(f"final Elo {history[-1].elo:.0f}  (random 1000, heuristic {heuristic_elo:.0f})")


if __name__ == "__main__":
    main(Config())
