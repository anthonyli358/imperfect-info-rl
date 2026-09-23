"""Train net-guided PIMC on Ventures: learn PUCT priors, keep the honest heuristic leaf value.

The synthesis of the session's findings. AlphaZero failed here because its *learned value* is
swamped by the deal's variance; PIMC (search with the hand-crafted deck-clock leaf estimate) already
beats the heuristic. So keep PIMC's honest leaf value and use the network only to supply the search
**priors**, trained from the search's own improved visit distributions (AlphaZero's
policy-improvement idea, but with rollout-quality evaluation). Untrained it already plays at PIMC
strength; the question is whether learned priors focus the search enough to pull ahead of it.

Reuses the AlphaZero trainer with ``leaf_value="heuristic"`` (the value head is unused / untrained).

Run with::

    uv run python -m experiments.train_ventures_guided
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.eval.tournament import anchored_elo, play_match
from iirl.selfplay import AZConfig, train_alphazero


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    out_dir: str = "runs"
    seed: int = 0
    anchor_games: int = 200
    az: AZConfig = field(
        default_factory=lambda: AZConfig(
            iterations=12,
            games_per_iter=10,
            simulations=128,
            determinizations=1,
            leaf_value="heuristic",
            train_steps_per_iter=60,
            batch_size=256,
            eval_interval=3,
            eval_games=16,
            eval_simulations=96,
            eval_determinizations=1,
            feature_sample_games=40,
            seed=0,
        )
    )


def main(config: Config) -> None:
    """Measure the heuristic anchor, train net-guided PIMC, and save the checkpoint + Elo curve."""
    game = make("ventures")
    rng = np.random.default_rng(config.seed)
    heuristic, random_agent = VenturesHeuristic("heuristic"), RandomAgent("random")
    hvr = play_match(game, heuristic, random_agent, config.anchor_games, rng)
    heuristic_elo = anchored_elo((hvr.wins + 0.5 * hvr.draws) / hvr.num_games, 1000.0)
    print(f"anchors: random 1000, heuristic {heuristic_elo:.0f}")

    az = AZConfig(
        **{**config.az.__dict__, "anchor_elos": {"random": 1000.0, "heuristic": heuristic_elo}}
    )
    anchors = {"random": random_agent, "heuristic": heuristic}
    print(f"{'iter':>4} {'games':>6} {'Elo':>8}  vs rand  vs heur  {'p_loss':>7}")

    def report(p) -> None:  # noqa: ANN001 - AZEvalPoint
        print(
            f"{p.iteration:>4} {p.games:>6} {p.elo:>8.1f}   {p.win_rate_vs['random']:.2f}"
            f"     {p.win_rate_vs['heuristic']:.2f}   {p.policy_loss:>7.3f}"
        )

    net, history = train_alphazero(game, anchors, az, on_eval=report)

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
        },
        out / "guided_ventures.pt",
    )
    curve = {
        "game": "ventures",
        "method": "guided_pimc",
        "anchors": {"random": 1000.0, "heuristic": heuristic_elo},
        "points": [
            {
                "iteration": p.iteration,
                "games": p.games,
                "elo": p.elo,
                "win_vs_random": p.win_rate_vs["random"],
                "win_vs_heuristic": p.win_rate_vs["heuristic"],
            }
            for p in history
        ],
    }
    (out / "guided_elo_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(f"\nsaved checkpoint + curve; final Elo {history[-1].elo:.0f}")


if __name__ == "__main__":
    main(Config())
