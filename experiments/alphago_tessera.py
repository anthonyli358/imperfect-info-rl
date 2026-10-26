"""The AlphaGo recipe on Tessera: a trained policy for priors, a value net for leaves, PUCT search.

Tessera hides nothing but the order of the bag, so it is the setting where search on a learned
value should work (on Ventures it never did: hidden hands and the deal left the value ~10 points
off when moves differed by ~1). Stages, each saved under ``runs/tessera_ag/``::

    uv run python -m experiments.alphago_tessera value   # value net on the policy's self-play
    uv run python -m experiments.alphago_tessera match [sims=200 opponent=heuristic nets=exit vs=…]
    uv run python -m experiments.alphago_tessera exit [iterations=48]  # the AlphaZero loop

``value`` trains ``selfplay.value.train_value`` on the policy's self-play outcomes and reports the
held-out Brier score next to the heuristic's. ``match`` plays net-guided PUCT search (priors from
the policy, leaf values from the value net: ``SplitEvaluator``) against the plain policy or the
heuristic on paired deals. ``exit`` is the AlphaZero loop (``selfplay.expert_iteration``): search
self-play, the policy trained toward the search's improved move distribution and the value toward
the outcome mixed with the search's own estimate, repeated. Each iteration saves both nets
(``runs/tessera_ag/exit_prior.pt``, ``exit_value.pt``; a rerun resumes after the last one) and
rates the raw policy against the starting policy; every few iterations it also rates the search
agent on the new nets against the heuristic, and every ``snapshot_every`` it keeps numbered copies
(``exit_prior_<i>.pt``, ``exit_value_<i>.pt``). ``iterations=`` sets the total to reach, so a
finished run can be extended. The curve is ``runs/tessera_ag/exit.jsonl``.
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
from iirl.selfplay.expert_iteration import ExItConfig, train_expert_iteration
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
    exit: ExItConfig = field(
        default_factory=lambda: ExItConfig(
            iterations=24,
            games_per_iter=400,
            simulations=200,
            determinizations=1,
            temp_moves=12,
            margin_weight=0.5,
            margin_scale=20.0,
            q_mix=0.5,
            num_workers=10,
        )
    )
    eval_every: int = 3
    snapshot_every: int = 12


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


def save_net(net: PolicyValueNet, path: Path, extra: dict) -> None:
    """Save a net in the checkpoint format ``PPOAgent.from_checkpoint`` and ``load_value`` read."""
    hidden = [m.out_features for m in net.trunk if isinstance(m, torch.nn.Linear)]
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
            "hidden": hidden,
            **extra,
        },
        path,
    )


def alphago_agent(
    policy: str, value_path: str, sims: int, prior_path: str | None = None
) -> AlphaZeroAgent:
    """Policy priors + value-net leaves + PUCT (a picklable spec for duplicate matches).

    ``prior_path`` overrides where the priors come from (an expert-iteration checkpoint).
    """
    ck = torch.load(value_path, map_location="cpu", weights_only=False)
    prior = PPOAgent.from_checkpoint(prior_path or policy).net
    evaluator = SplitEvaluator(prior, load_value(value_path))
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
    # nets=exit: priors and values from the AlphaZero loop's latest checkpoints instead of the
    # originals; nets=exit_24 (any snapshot number): that iteration's numbered copies.
    nets = args.get("nets", "start")
    loop = nets.startswith("exit")
    value_path = OUT / (f"exit_value{nets[4:]}.pt" if loop else "value.pt")
    prior_path = str(OUT / f"exit_prior{nets[4:]}.pt") if loop else None
    spec = (
        alphago_agent,
        {"policy": policy, "value_path": str(value_path), "sims": sims, "prior_path": prior_path},
    )
    # vs=exit_24: the opponent is the same search on that snapshot's nets (a direct comparison).
    vs = args.get("vs")
    if vs:
        opponent = f"search on {vs}"
        other = (
            alphago_agent,
            {
                "policy": policy,
                "value_path": str(OUT / f"exit_value{vs[4:]}.pt"),
                "sims": sims,
                "prior_path": str(OUT / f"exit_prior{vs[4:]}.pt"),
            },
        )
    elif opponent == "heuristic":
        other = (make_heuristic, {"short_name": "tessera"})
    else:
        other = (ppo_agent, {"path": policy})
    t0 = time.perf_counter()
    r = duplicate_match(make("tessera"), spec, other, deals, seed=2028, workers=10)
    row = {
        "policy": policy,
        "sims": sims,
        "nets": nets,
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


def exit_loop(config: Config) -> None:
    """Run the AlphaZero loop from the policy and the value net, rating as it goes."""
    from dataclasses import replace

    args = dict(item.split("=") for item in sys.argv[2:])
    if "iterations" in args:
        config = replace(config, exit=replace(config.exit, iterations=int(args["iterations"])))
    policy = latest_policy(config)
    prior_path, value_path = OUT / "exit_prior.pt", OUT / "exit_value.pt"
    log_path = OUT / "exit.jsonl"
    done = 0
    if log_path.exists():
        done = max(
            json.loads(line)["iteration"] for line in log_path.read_text("utf-8").splitlines()
        )
    if done and prior_path.exists():
        prior = PPOAgent.from_checkpoint(str(prior_path)).net
        value_net = load_value(str(value_path))
        print(f"resuming after iteration {done}", flush=True)
    else:
        prior = PPOAgent.from_checkpoint(policy).net
        value_net = load_value(str(OUT / "value.pt"))
    value_meta = torch.load(OUT / "value.pt", map_location="cpu", weights_only=False)
    meta = {k: value_meta[k] for k in ("margin_weight", "margin_scale")}
    game = make("tessera")
    ex = replace(config.exit, iterations=config.exit.iterations - done, seed=done)

    def on_iteration(stats, prior_net, value_net) -> None:  # noqa: ANN001
        it = done + stats.iteration
        save_net(prior_net, prior_path, {})
        save_net(value_net, value_path, meta)
        if it % config.snapshot_every == 0:
            save_net(prior_net, OUT / f"exit_prior_{it}.pt", {})
            save_net(value_net, OUT / f"exit_value_{it}.pt", meta)
        row = {
            "iteration": it,
            "games": it * ex.games_per_iter,
            "policy_loss": round(stats.policy_loss, 4),
            "value_loss": round(stats.value_loss, 4),
            "minutes": round(stats.seconds / 60, 1),
        }
        r = duplicate_match(
            game,
            (ppo_agent, {"path": str(prior_path)}),
            (ppo_agent, {"path": policy}),
            300,
            seed=31,
            workers=10,
        )
        row["policy_vs_start"] = round(r.score, 4)
        row["policy_vs_start_margin"] = round(r.margin, 2)
        if it % config.eval_every == 0:
            spec = (
                alphago_agent,
                {
                    "policy": policy,
                    "value_path": str(value_path),
                    "sims": 100,
                    "prior_path": str(prior_path),
                },
            )
            h = duplicate_match(game, spec, (make_heuristic, {"short_name": "tessera"}), 200, 2028)
            row["search100_vs_heuristic"] = round(h.score, 4)
            row["search100_vs_heuristic_margin"] = round(h.margin, 2)
        print(json.dumps(row), flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    train_expert_iteration(game, prior, value_net, ex, on_iteration)


if __name__ == "__main__":
    {"value": value, "match": match, "exit": exit_loop}[sys.argv[1]](Config())
