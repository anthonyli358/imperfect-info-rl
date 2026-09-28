"""Phase B: Nash-seeking PPO on Ventures, vs a same-budget plain-PPO control.

Both runs warm-start from PPO v2 (``runs/ppo_ventures_best.pt``) and train for the same number of
games; then each gets a best-response exploiter (``experiments/exploit_ventures.py``). The
question is whether the Nash-seeking recipe lowers exploitability (the exploiter's score) without
losing strength, beyond what the same amount of plain training does.

* ``control``: plain PPO self-play with the usual snapshot pool.
* ``nash``: magnetic KL regularization toward a slowly refreshed copy of the policy (MMD / R-NaD
  style), plus a league: past snapshots and the heuristic as a permanent member.
* ``nash_long``: the ``nash`` recipe continued from ``runs/nash_nash.pt`` for 600 more
  iterations, saving ``runs/nash_long_<iteration>.pt`` every 100, so exploitability can be
  tracked as training goes on (the success test for a Nash-seeking learner).

Run with::

    uv run python -m experiments.train_ventures_nash
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

import iirl.games  # noqa: F401  (registers "ventures")
from experiments import exploit_ventures
from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.selfplay import PPOConfig, train_ppo


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    base: str = "runs/ppo_ventures_best.pt"
    runs: tuple[str, ...] = ("control", "nash")
    # per-run warm start overriding ``base``
    bases: dict[str, str] = field(default_factory=lambda: {"nash_long": "runs/nash_nash.pt"})
    save_every: int = 100  # also save a numbered checkpoint every this many iterations (0: off)
    exploit: bool = True
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            iterations=200,
            games_per_iter=256,
            num_envs=32,
            num_workers=10,
            epochs=4,
            minibatch_size=1024,
            lr=1e-4,
            hidden=(512, 512),
            privileged_critic=True,
            win_bonus=20.0,
            past_opponent_frac=0.3,
            snapshot_interval=10,
            eval_interval=20,
            eval_games=400,
            seed=1,
        )
    )


def variant(name: str, ppo: PPOConfig) -> PPOConfig:
    """Return the PPO settings for run ``name``."""
    if name == "control":
        return ppo
    if name in ("nash", "nash_long"):
        return replace(
            ppo,
            magnet_coef=0.1,
            magnet_interval=20,
            scripted_opponents=(VenturesHeuristic("heuristic"),),
            scripted_frac=0.1,
            **({"iterations": 600, "seed": 2} if name == "nash_long" else {}),
        )
    raise ValueError(f"unknown run {name!r}")


def save(net, path: Path) -> None:  # noqa: ANN001 - PolicyValueNet
    """Save ``net`` as a PPO-style checkpoint."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
            "hidden": [layer.out_features for layer in net.trunk if hasattr(layer, "out_features")],
        },
        path,
    )


def run_tag(name: str) -> str:
    """Return the file tag for run ``name`` (``nash_long`` is already prefixed)."""
    return name if name.startswith("nash") and name != "nash" else f"nash_{name}"


def main(config: Config) -> None:
    """Train each variant from the base policy, save it, and (optionally) exploit it."""
    game = make("ventures")
    heuristic = VenturesHeuristic("heuristic")
    for name in config.runs:
        ppo = replace(variant(name, config.ppo), anchor_elos={"heuristic": 1676.0})
        init = PPOAgent.from_checkpoint(config.bases.get(name, config.base)).net
        print(f"\n=== {name} ===", flush=True)

        def report(p) -> None:  # noqa: ANN001 - PPOEvalPoint
            print(
                f"{p.iteration:>4} {p.games:>6}  vs heur {p.win_rate_vs['heuristic']:.2f}"
                f"  margin {p.margin_vs['heuristic']:+6.1f}  p_loss {p.policy_loss:.3f}",
                flush=True,
            )

        def checkpoint(net, p, tag: str = run_tag(name)) -> None:  # noqa: ANN001
            if config.save_every and p.iteration and p.iteration % config.save_every == 0:
                save(net, Path(f"runs/{tag}_{p.iteration}.pt"))

        net, history = train_ppo(
            game, {"heuristic": heuristic}, ppo, report, checkpoint, init_net=init
        )
        path = Path(f"runs/{run_tag(name)}.pt")
        save(net, path)
        curve = [
            {"iteration": p.iteration, "games": p.games, "margin": p.margin_vs["heuristic"]}
            for p in history
        ]
        Path(f"runs/{run_tag(name)}_curve.json").write_text(json.dumps(curve, indent=2), "utf-8")
        if config.exploit:
            exploit_ventures.main(exploit_ventures.Config(target=str(path), name=run_tag(name)))


if __name__ == "__main__":
    main(Config())
