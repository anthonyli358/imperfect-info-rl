"""Ground truth on Mini-Ventures: exact exploitability of Nash-seeking PPO, and the yardstick's gap.

How exploitable is Nash-seeking PPO really, and how much of it does our trained-exploiter
yardstick find? Everything else in this repo measures exploitability with a *trained* exploiter,
which only gives a lower bound. Mini-Ventures is small enough to compute the exact best response
(``iirl.eval.exact_br``), so here:

1. The usual recipe (PPO from scratch, then Nash-seeking PPO with a magnet) is trained on it, with
   checkpoints along the way.
2. Every checkpoint's *exact* exploitability is computed, in points per game, for the greedy policy
   (what our agents deploy) and the sampled one.
3. The trained-exploiter yardstick (``experiments/exploit_ventures.py``) is run on the final agent
   and compared with the exact best response.

Resumable (training continues from its last checkpoint; finished measurements are kept in
``runs/mini/ground_truth.json``).

Run with::

    uv run python -m experiments.mini_ground_truth
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import iirl.games  # noqa: F401  (registers "mini_ventures")
from experiments import exploit_ventures
from experiments.train_ventures_league import _train
from iirl.agents.ppo import PPOAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.eval.exact_br import build_graph, exploitability, net_policy
from iirl.selfplay import PPOConfig

OUT = Path("runs/mini")
RESULTS = OUT / "ground_truth.json"


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    ppo_iterations: int = 200
    nash_iterations: int = 200
    save_every: int = 50
    exploit_seeds: tuple[int, ...] = (0, 1)
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            # A Mini-Ventures game has ~4 decisions per player (Ventures: ~70), so each iteration
            # needs ~16x the games for the same number of decisions per update.
            games_per_iter=4096,
            num_envs=256,
            num_workers=10,
            epochs=4,
            minibatch_size=1024,
            lr=3e-4,
            hidden=(256, 256),
            privileged_critic=True,
            win_bonus=1.0,
            reward_scale=5.0,
            eval_interval=25,
            eval_games=2000,
            feature_sample_games=100,
            seed=0,
        )
    )


def _load() -> dict:
    return json.loads(RESULTS.read_text("utf-8")) if RESULTS.exists() else {}


def _save(data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2), "utf-8")


def main(config: Config) -> None:
    """Train, measure exact exploitability along the way, and compare the yardstick."""
    OUT.mkdir(parents=True, exist_ok=True)
    common = {
        "save_every": config.save_every,
        "game_name": "mini_ventures",
        "heuristic": RandomAgent(),
        "anchor_elo": 1000.0,
    }
    base = _train(
        None, replace(config.ppo, iterations=config.ppo_iterations), "mini/mini_ppo", **common
    )
    nash = replace(
        config.ppo,
        iterations=config.nash_iterations,
        lr=1e-4,
        past_opponent_frac=0.3,
        magnet_coef=0.1,
        magnet_interval=20,
        seed=1,
    )
    final = _train(base, nash, "mini/mini_nash", **common)

    game = make("mini_ventures")
    graph = build_graph(game)
    data = _load()
    checkpoints = [
        *(
            f"runs/mini/mini_ppo_{i}.pt"
            for i in range(config.save_every, config.ppo_iterations + 1, config.save_every)
        ),
        *(
            f"runs/mini/mini_nash_{i}.pt"
            for i in range(config.save_every, config.nash_iterations + 1, config.save_every)
        ),
    ]
    exact = data.setdefault("exact", {})
    for path in checkpoints:
        if path in exact or not Path(path).exists():
            continue
        net = PPOAgent.from_checkpoint(path).net
        exact[path] = {
            "greedy": exploitability(game, net_policy(net, greedy=True), graph)["exploitability"],
            "sampled": exploitability(game, net_policy(net), graph)["exploitability"],
        }
        print(
            f"{path}: exact exploitability greedy {exact[path]['greedy']:.3f}, "
            f"sampled {exact[path]['sampled']:.3f} points/game",
            flush=True,
        )
        _save(data)

    trained = data.setdefault("trained_exploiter", {})
    for seed in config.exploit_seeds:
        name = f"mini_nash_s{seed}"
        record = Path(f"runs/exploit_{name}.json")
        if not record.exists():
            ex = exploit_ventures.Config()
            exploit_ventures.main(
                replace(
                    ex,
                    target=final,
                    name=name,
                    game="mini_ventures",
                    ppo=replace(
                        ex.ppo,
                        seed=seed,
                        hidden=config.ppo.hidden,
                        games_per_iter=config.ppo.games_per_iter,
                        num_envs=config.ppo.num_envs,
                        eval_games=2000,
                    ),
                )
            )
        final_rec = json.loads(record.read_text("utf-8"))["final"]
        trained[name] = {"score": final_rec["score"], "margin": final_rec["margin"]}
    _save(data)
    best = max(v["margin"] for v in trained.values())
    last = f"runs/mini/mini_nash_{config.nash_iterations}.pt"
    truth = exact.get(last, {}).get("greedy")
    print(
        f"\nfinal agent: exact exploitability {truth} points/game; "
        f"trained exploiter finds {best:+.3f} (max of {len(trained)} runs)"
    )


if __name__ == "__main__":
    main(Config())
