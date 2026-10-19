"""Train Tessera with the recipe, unchanged: PPO v2, then Nash-seeking PPO.

Stage 3's first step: the recipe that won on both card games, before any search. Phase 1 is
PPO self-play from scratch with the v2 settings (privileged critic, potential shaping from the
game's ``heuristic_value``, win bonus, snapshot pool). Phase 2 continues it with the Nash recipe
(magnet KL toward a slowly refreshed copy, the heuristic as a permanent league member) plus
``gamma = 0.999``. Both phases keep numbered checkpoints for the Elo ladder, then exploitability
is measured as the max over two exploiter runs. Tessera is perfect-information apart from the
bag, so the privileged critic sees nothing extra here (both views are the same state).

Elo on Tessera is anchored on its heuristic at **1500**, as on Camel Stocks (only differences
matter, and they are not comparable across games).

Resumable: every phase continues from its last eval checkpoint. Stop at any time and rerun.

Run with::

    uv run python -m experiments.train_tessera
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import iirl.games  # noqa: F401  (registers "tessera")
from experiments import exploit_ventures
from experiments.train_ventures_league import _train
from iirl.games.tessera import TesseraHeuristic
from iirl.selfplay import PPOConfig

ANCHOR_ELO = 1500.0


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    tag: str = "tessera"
    base_iterations: int = 600
    nash_iterations: int = 400
    save_every: int = 100
    exploit_seeds: tuple[int, ...] = (0, 1)
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            games_per_iter=256,
            num_envs=32,
            num_workers=10,
            epochs=4,
            minibatch_size=1024,
            lr=3e-4,
            hidden=(512, 512),
            privileged_critic=True,
            win_bonus=20.0,
            eval_interval=10,
            eval_games=400,
            feature_sample_games=40,
            seed=0,
        )
    )


def main(config: Config) -> None:
    """Run both phases, then the exploiters; print the exploitability estimate."""
    common = {
        "save_every": config.save_every,
        "game_name": "tessera",
        "heuristic": TesseraHeuristic("heuristic"),
        "anchor_elo": ANCHOR_ELO,
    }
    base = _train(
        None, replace(config.ppo, iterations=config.base_iterations), f"{config.tag}_ppo", **common
    )
    nash = replace(
        config.ppo,
        iterations=config.nash_iterations,
        lr=1e-4,
        past_opponent_frac=0.3,
        snapshot_interval=10,
        magnet_coef=0.1,
        magnet_interval=20,
        scripted_opponents=(TesseraHeuristic("heuristic"),),
        scripted_frac=0.1,
        gamma=0.999,
        seed=1,
    )
    path = _train(base, nash, f"{config.tag}_nash", **common)
    scores = []
    for seed in config.exploit_seeds:
        name = f"{config.tag}_nash_s{seed}"
        record = Path(f"runs/exploit_{name}.json")
        if not record.exists():
            ex = exploit_ventures.Config()
            exploit_ventures.main(
                replace(ex, target=path, name=name, game="tessera", ppo=replace(ex.ppo, seed=seed))
            )
        scores.append(float(json.loads(record.read_text("utf-8"))["final"]["score"]))
    print(f"\nexploitability of {config.tag}_nash: max {max(scores):.3f}  (runs: {scores})")


if __name__ == "__main__":
    main(Config())
