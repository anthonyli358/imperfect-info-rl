"""Train Ventures with PPO self-play and record the Elo curve.

The model-free route to a self-trained Ventures agent: an MLP policy/value net, self-play, PPO
updates. Unlike AlphaZero (whose learned value is swamped by the deal's variance) and CFR (which
needs a short horizon), PPO handles a long, high-variance imperfect-info game because it only
samples episodes and uses the value net as a baseline. Elo is anchored to random (1000) and the
heuristic (measured once). Outputs a checkpoint (``.pt``) and the curve (``.json``) into ``runs/``.

Run with::

    uv run python -m experiments.train_ventures_ppo
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
from iirl.selfplay import PPOConfig, train_ppo


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    out_dir: str = "runs"
    seed: int = 0
    anchor_games: int = 200
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            iterations=40,
            games_per_iter=32,
            epochs=4,
            minibatch_size=256,
            lr=3e-4,
            hidden=(256, 256),
            eval_interval=4,
            eval_games=80,
            feature_sample_games=40,
            seed=0,
        )
    )


def main(config: Config) -> None:
    """Measure the heuristic anchor, train with PPO, and save the checkpoint + Elo curve."""
    game = make("ventures")
    rng = np.random.default_rng(config.seed)
    heuristic, random_agent = VenturesHeuristic("heuristic"), RandomAgent("random")
    hvr = play_match(game, heuristic, random_agent, config.anchor_games, rng)
    heuristic_elo = anchored_elo((hvr.wins + 0.5 * hvr.draws) / hvr.num_games, 1000.0)
    print(f"anchors: random 1000, heuristic {heuristic_elo:.0f}")

    ppo = PPOConfig(
        **{**config.ppo.__dict__, "anchor_elos": {"random": 1000.0, "heuristic": heuristic_elo}}
    )
    anchors = {"random": random_agent, "heuristic": heuristic}
    print(f"{'iter':>4} {'games':>6} {'Elo':>8}  vs rand  vs heur  {'p_loss':>7} {'v_loss':>7}")

    def report(p) -> None:  # noqa: ANN001 - PPOEvalPoint
        print(
            f"{p.iteration:>4} {p.games:>6} {p.elo:>8.1f}   {p.win_rate_vs['random']:.2f}"
            f"     {p.win_rate_vs['heuristic']:.2f}   {p.policy_loss:>7.3f} {p.value_loss:>7.3f}"
        )

    net, history = train_ppo(game, anchors, ppo, on_eval=report)

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
        },
        out / "ppo_ventures.pt",
    )
    curve = {
        "game": "ventures",
        "method": "ppo",
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
    (out / "ppo_elo_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print(f"\nsaved checkpoint + curve; final Elo {history[-1].elo:.0f}")


if __name__ == "__main__":
    main(Config())
