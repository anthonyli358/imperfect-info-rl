"""NFSP vs Nash-seeking PPO on Camel Stocks, at equal budget.

Both start from ``runs/camel_ppo.pt`` (plain PPO self-play) and train for 400 iterations of 256
games. ``camel_nash`` used the magnet-KL recipe; this run uses NFSP instead (a PPO best response
against the anticipatory mix of itself and its average policy, plus a supervised average policy,
which is the deployed agent). Judged the same way: exploitability (max of two exploiter runs), the
heuristic, and head to head.

Run with::

    uv run python -m experiments.nfsp_camel
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import iirl.games  # noqa: F401  (registers "camel_stocks")
from experiments import exploit_ventures
from experiments.exploit_ventures import ppo_agent
from experiments.train_camel_stocks import Config as CamelConfig
from experiments.train_ventures_nash import save as save_policy
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make, make_heuristic
from iirl.eval.duplicate import duplicate_match
from iirl.selfplay.nfsp import NFSPConfig, train_nfsp


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    start: str = "runs/camel_ppo.pt"
    out: str = "runs/camel_nfsp.pt"
    iterations: int = 400
    eta: float = 0.1
    exploit_seeds: tuple[int, ...] = (0, 1)
    deals: int = 1000


def main(config: Config) -> None:
    """Train (unless done), then rate against the exploiters, the heuristic and camel_nash."""
    game = make("camel_stocks")
    out = Path(config.out)
    if not out.exists():
        ppo = replace(
            CamelConfig().ppo,
            iterations=config.iterations,
            lr=1e-4,
            gamma=0.999,
            seed=7,
            eval_interval=config.iterations,  # beta's own evals are not the point; pi's are
        )

        def report(it: int, scores: dict[str, float]) -> None:
            print(
                f"nfsp it{it:>4}  average policy vs heuristic {scores['heuristic']:.2f}", flush=True
            )

        avg = train_nfsp(
            game,
            {"heuristic": make_heuristic("camel_stocks")},
            NFSPConfig(ppo=ppo, eta=config.eta),
            PPOAgent.from_checkpoint(config.start).net,
            report,
        )
        save_policy(avg, out)
    scores = []
    for seed in config.exploit_seeds:
        name = f"camel_nfsp_s{seed}"
        record = Path(f"runs/exploit_{name}.json")
        if not record.exists():
            ex = exploit_ventures.Config()
            exploit_ventures.main(
                replace(
                    ex,
                    target=str(out),
                    name=name,
                    game="camel_stocks",
                    ppo=replace(ex.ppo, seed=seed),
                )
            )
        scores.append(float(json.loads(record.read_text("utf-8"))["final"]["score"]))
    me = (ppo_agent, {"path": str(out)})
    heur = duplicate_match(
        game, me, (make_heuristic, {"short_name": "camel_stocks"}), config.deals, seed=91
    )
    h2h = duplicate_match(
        game, me, (ppo_agent, {"path": "runs/camel_nash.pt"}), config.deals, seed=93
    )
    print(
        f"\nexploitability of camel_nfsp: max {max(scores):.3f} (runs {scores}); camel_nash: 0.509"
    )
    print(f"vs heuristic: {heur.score:.3f}+-{heur.score_se:.3f} (camel_nash: 0.940)")
    print(f"vs camel_nash: {h2h.score:.3f}+-{h2h.score_se:.3f}  margin {h2h.margin:+.1f}")


if __name__ == "__main__":
    main(Config())
