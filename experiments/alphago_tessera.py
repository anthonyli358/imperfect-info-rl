"""The AlphaGo recipe on Tessera: a trained policy for priors, a value net for leaves, PUCT search.

Tessera hides nothing but the order of the bag, so it is the setting where search on a learned
value should work (on Ventures it never did: hidden hands and the deal left the value ~10 points
off when moves differed by ~1). Stages, each saved under ``runs/tessera_ag/``::

    uv run python -m experiments.alphago_tessera value   # value net on the policy's self-play
    uv run python -m experiments.alphago_tessera match [sims=200 deals=200 opponent=policy]

``value`` trains ``selfplay.value.train_value`` on the policy's self-play outcomes and reports the
held-out Brier score next to the heuristic's. ``match`` plays net-guided PUCT search (priors from
the policy, leaf values from the value net: ``SplitEvaluator``) against the plain policy or the
heuristic on paired deals.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

import iirl.games  # noqa: F401  (registers "tessera")
from iirl.agents.alphazero.agent import AlphaZeroAgent
from iirl.agents.alphazero.mcts import PUCTConfig
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make, make_heuristic
from iirl.eval.duplicate import duplicate_match
from iirl.nn.policy_value import PolicyValueNet, SplitEvaluator
from iirl.selfplay.value import ValueConfig, train_value

OUT = Path("runs/tessera_ag")


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    policy: str = "runs/tessera_nash2.pt"
    value: ValueConfig = field(
        default_factory=lambda: ValueConfig(
            games=40_000,
            positions_per_game=12,
            hidden=(512, 512),
            epochs=8,
            num_workers=10,
            # Tessera margins are small (about ±10-20 points), so the target credits the margin.
            margin_weight=0.5,
            margin_scale=20.0,
        )
    )


def latest_policy(config: Config) -> str:
    """The finished policy, else the newest partial or numbered checkpoint of it."""
    for path in (
        config.policy,
        config.policy.replace(".pt", "_partial.pt"),
        "runs/tessera_nash.pt",
    ):
        if Path(path).exists():
            return path
    raise FileNotFoundError(config.policy)


def load_value(path: str) -> PolicyValueNet:
    """Load a value net saved by the ``value`` stage."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ck["input_dim"], ck["num_actions"], tuple(ck["hidden"]), "tanh")
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def value(config: Config) -> None:
    """Train the value net on the policy's self-play; save it and its report."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = latest_policy(config)
    policy = PPOAgent.from_checkpoint(path).net
    t0 = time.perf_counter()
    net, report = train_value(make("tessera"), policy, (512, 512), config.value)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
            "hidden": list(config.value.hidden),
            "policy": path,
            "margin_weight": config.value.margin_weight,
            "margin_scale": config.value.margin_scale,
        },
        OUT / "value.pt",
    )
    row = {"policy": path, "minutes": round((time.perf_counter() - t0) / 60, 1), **asdict(report)}
    (OUT / "value.json").write_text(json.dumps(row, indent=1), "utf-8")
    print(
        f"value net from {path}: held-out Brier {report.learned:.4f} vs heuristic "
        f"{report.heuristic:.4f} (scale {report.heuristic_scale}) vs coin flip "
        f"{report.coin_flip:.4f}; {report.positions:,} positions",
        flush=True,
    )


def alphago_agent(policy: str, value_path: str, sims: int) -> AlphaZeroAgent:
    """Policy priors + value-net leaves + PUCT (a picklable spec for duplicate matches)."""
    ck = torch.load(value_path, map_location="cpu", weights_only=False)
    evaluator = SplitEvaluator(PPOAgent.from_checkpoint(policy).net, load_value(value_path))
    search = PUCTConfig(
        simulations=sims,
        determinizations=1,  # perfect information: the one world is the real state
        margin_weight=ck["margin_weight"],
        margin_scale=ck["margin_scale"],
    )
    return AlphaZeroAgent(evaluator, search, name=f"alphago{sims}")


def ppo_agent(path: str) -> PPOAgent:
    """The policy as deployed: greedy."""
    return PPOAgent.from_checkpoint(path, greedy=True)


def match(config: Config) -> None:
    """Search against the plain policy (or the heuristic) on paired deals."""
    args = dict(item.split("=") for item in sys.argv[2:])
    sims, deals = int(args.get("sims", 200)), int(args.get("deals", 200))
    opponent = args.get("opponent", "policy")
    policy = latest_policy(config)
    other = (
        (make_heuristic, {"short_name": "tessera"})
        if opponent == "heuristic"
        else (ppo_agent, {"path": policy})
    )
    spec = (alphago_agent, {"policy": policy, "value_path": str(OUT / "value.pt"), "sims": sims})
    t0 = time.perf_counter()
    r = duplicate_match(make("tessera"), spec, other, deals, seed=2028, workers=10)
    row = {
        "policy": policy,
        "sims": sims,
        "opponent": opponent,
        "deals": r.deals,
        "score": round(r.score, 4),
        "score_se": round(r.score_se, 4),
        "margin": round(r.margin, 2),
        "margin_se": round(r.margin_se, 2),
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    print(json.dumps(row), flush=True)
    with (OUT / "matches.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    {"value": value, "match": match}[sys.argv[1]](Config())
