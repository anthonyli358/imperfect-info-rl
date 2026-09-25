"""Expert iteration on Ventures: AlphaZero-style search self-play from the PPO policy + value net.

Needs ``runs/ppo_ventures_best.pt`` (``experiments/train_ventures_ppo.py``) and
``runs/value_ventures_priv.pt`` (``experiments/train_ventures_value.py`` with
``ValueConfig(privileged=True)``). Each iteration plays search
self-play games, trains the priors toward the visit counts and the value toward the mixed
outcome / search-value target, checkpoints both nets, and rates the search agent in a duplicate
match against the PPO policy it started from (and, every few iterations, the heuristic).

Run with::

    uv run python -m experiments.train_ventures_exit
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.alphazero import AlphaZeroAgent, PUCTConfig
from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.nn.policy_value import PolicyValueNet, SplitEvaluator
from iirl.selfplay.expert_iteration import ExItConfig, train_expert_iteration


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    prior_checkpoint: str = "runs/ppo_ventures_best.pt"
    value_checkpoint: str = "runs/value_ventures_priv.pt"
    out_prefix: str = "runs/exit_ventures"
    eval_deals: int = 100
    heuristic_every: int = 4
    search_every: int = 4  # rate the (slow) search agent every k iterations; the policy every one
    exit: ExItConfig = field(
        default_factory=lambda: ExItConfig(
            games_per_iter=120,
            simulations=128,
            determinizations=8,
            privileged_value=True,
            margin_weight=0.2,
            q_mix=0.25,
        )
    )


def load_net(path: str, activation: str) -> PolicyValueNet:
    """Load a :class:`PolicyValueNet` checkpoint with the given value activation."""
    ck = torch.load(path, map_location="cpu")
    net = PolicyValueNet(ck["input_dim"], ck["num_actions"], tuple(ck["hidden"]), activation)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def save_net(net: PolicyValueNet, path: str, privileged: bool = False) -> None:
    """Save ``net`` in the checkpoint format :func:`load_net` reads."""
    hidden = [m.out_features for m in net.trunk if isinstance(m, torch.nn.Linear)]
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
            "hidden": hidden,
            "privileged": privileged,
        },
        path,
    )


def search_agent(
    prior: str,
    value: str,
    simulations: int = 64,
    determinizations: int = 4,
    margin_weight: float = 0.0,
    margin_scale: float = 40.0,
) -> AlphaZeroAgent:
    """Build the greedy search agent from checkpoint paths (picklable spec for worker processes)."""
    privileged = bool(torch.load(value, map_location="cpu").get("privileged", False))
    evaluator = SplitEvaluator(load_net(prior, "linear"), load_net(value, "tanh"), privileged)
    search = PUCTConfig(
        simulations=simulations,
        determinizations=determinizations,
        leaf_batch=8,
        margin_weight=margin_weight,
        margin_scale=margin_scale,
    )
    return AlphaZeroAgent(evaluator, search, name="exit")


def ppo_agent(path: str) -> PPOAgent:
    """Build the greedy PPO policy agent from a checkpoint path."""
    return PPOAgent.from_checkpoint(path)


def heuristic_agent() -> VenturesHeuristic:
    """Build the hand-tuned heuristic."""
    return VenturesHeuristic("heuristic")


def main(config: Config) -> None:
    """Run expert iteration, checkpointing and rating the search agent every iteration."""
    game = make("ventures")
    ex = config.exit
    prior = load_net(config.prior_checkpoint, "linear")
    value = load_net(config.value_checkpoint, "tanh")
    log: list[dict] = []
    Path(config.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    def rate(tag: str, prior_path: str, value_path: str) -> dict:
        spec = (
            search_agent,
            dict(
                prior=prior_path,
                value=value_path,
                simulations=ex.simulations,
                determinizations=ex.determinizations,
                margin_weight=ex.margin_weight,
                margin_scale=ex.margin_scale,
            ),
        )
        row: dict = {"tag": tag}
        base = (ppo_agent, dict(path=config.prior_checkpoint))
        # The raw policy (no search) vs the starting policy: cheap, and catches a policy update
        # that hurts before the search results show it.
        r = duplicate_match(
            game, (ppo_agent, dict(path=prior_path)), base, 500, seed=13, workers=ex.num_workers
        )
        row["policy"] = asdict(r)
        print(f"  [{tag}] policy vs ppo  score {r.score:.3f}±{r.score_se:.3f}", flush=True)
        first = tag == "start"
        vs = [("ppo", base)] if first or int(tag) % config.search_every == 0 else []
        if first or int(tag) % config.heuristic_every == 0:
            vs.append(("heuristic", (heuristic_agent, {})))
        for name, opp in vs:
            r = duplicate_match(game, spec, opp, config.eval_deals, seed=11, workers=ex.num_workers)
            row[name] = asdict(r)
            print(
                f"  [{tag}] vs {name:<9} score {r.score:.3f}±{r.score_se:.3f}"
                f"  margin {r.margin:+.1f}±{r.margin_se:.1f}",
                flush=True,
            )
        return row

    log.append(rate("start", config.prior_checkpoint, config.value_checkpoint))

    def on_iteration(stats, prior_net, value_net) -> None:  # noqa: ANN001
        p_path = f"{config.out_prefix}_prior_{stats.iteration:02d}.pt"
        v_path = f"{config.out_prefix}_value_{stats.iteration:02d}.pt"
        save_net(prior_net, p_path)
        save_net(value_net, v_path, ex.privileged_value)
        print(
            f"it {stats.iteration:>3}  games {stats.games:>6}  pos {stats.positions:>7}"
            f"  p_loss {stats.policy_loss:.3f}  v_loss {stats.value_loss:.3f}"
            f"  root_q {stats.mean_root_q:+.3f}  {stats.seconds:.0f}s",
            flush=True,
        )
        row = rate(str(stats.iteration), p_path, v_path)
        row["stats"] = asdict(stats)
        log.append(row)
        Path(f"{config.out_prefix}_log.json").write_text(json.dumps(log, indent=2), "utf-8")

    train_expert_iteration(game, prior, value, ex, on_iteration)


if __name__ == "__main__":
    main(Config())
