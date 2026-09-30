"""PSRO on Ventures: is the best agent close to an equilibrium of a growing population?

Policy-Space Response Oracles (``iirl.eval.psro``). Start from a population of trained agents,
then each round:

1. Measure every pair (duplicate deals) and solve the meta-game for its Nash mixture.
2. Train a best response to that mixture (the exploiter recipe, with the mixture's weights as
   opponent sampling weights), screen its checkpoints against the mixture, and rate the best on
   fresh deals. That rating is the mixture's exploitability estimate this round.
3. Add the best response to the population.

If the best agent is near equilibrium, the mixture keeps most of its weight on it and the best
responses stay near 0.5. A best response that scores well, and then takes weight in the mixture,
has found a real weakness.

Resumable: the population, payoff matrix and per-round results are saved to
``runs/psro/state.json`` after every step; a rerun continues where it stopped (an interrupted best
response restarts).

Run with::

    uv run python -m experiments.psro_ventures
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

import iirl.games  # noqa: F401  (registers "ventures")
from experiments import exploit_ventures
from experiments.exploit_ventures import ppo_agent
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.eval.psro import mixture_agent, solve_symmetric_meta_game
from iirl.selfplay import train_ppo

OUT = Path("runs/psro")
STATE = OUT / "state.json"


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    initial: tuple[str, ...] = (
        "runs/diverse_sym.pt",
        "runs/careful.pt",
        "runs/league_control.pt",
    )
    rounds: int = 4
    pair_deals: int = 300  # per population pair
    screen_deals: int = 300  # per best-response checkpoint
    final_deals: int = 1000  # the round's exploitability rating
    br_iterations: int = 300
    snapshot_every: int = 50


def _load() -> dict:
    return json.loads(STATE.read_text("utf-8")) if STATE.exists() else {}


def _save(state: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), "utf-8")
    tmp.replace(STATE)


def _score(a: str, b: str, deals: int, seed: int) -> float:
    r = duplicate_match(
        make("ventures"), (ppo_agent, {"path": a}), (ppo_agent, {"path": b}), deals, seed=seed
    )
    return r.score


def _mixture_spec(population: list[str], weights: np.ndarray) -> tuple:
    return (
        mixture_agent,
        {
            "factory": ppo_agent,
            "member_kwargs": [{"path": p} for p in population],
            "weights": [float(w) for w in weights],
        },
    )


def _save_net(net, path: Path) -> None:  # noqa: ANN001 - PolicyValueNet
    hidden = [m.out_features for m in net.trunk if isinstance(m, torch.nn.Linear)]
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
            "hidden": hidden,
        },
        path,
    )


def _best_response(
    config: Config, population: list[str], mixture: np.ndarray, r: int
) -> tuple[str, float, float]:
    """Train, screen and rate a best response to the mixture; return (path, score, se)."""
    game = make("ventures")
    nets = [PPOAgent.from_checkpoint(p).net for p in population]
    top = int(np.argmax(mixture))
    ppo = replace(
        exploit_ventures.Config().ppo,
        iterations=config.br_iterations,
        opponent_weights=tuple(float(w) for w in mixture),
        eval_interval=config.snapshot_every,
        anchor_elos={"top": 0.0},
        seed=100 + r,
    )
    snapshots: dict[int, dict] = {}

    def keep(net, p) -> None:  # noqa: ANN001 - PolicyValueNet, PPOEvalPoint
        if p.iteration:
            snapshots[p.iteration] = copy.deepcopy(net.state_dict())

    def report(p) -> None:  # noqa: ANN001 - PPOEvalPoint
        print(f"  br r{r} it{p.iteration:>4}  vs top member {p.win_rate_vs['top']:.2f}", flush=True)

    net, _ = train_ppo(
        game,
        {"top": PPOAgent(nets[top])},
        ppo,
        report,
        keep,
        init_net=nets[top],
        opponents=nets,
    )
    spec = _mixture_spec(population, mixture)
    best, best_score = None, -1.0
    for it, sd in sorted(snapshots.items()):
        path = OUT / f"br_r{r}_it{it}.pt"
        net.load_state_dict(sd)
        _save_net(net, path)
        s = duplicate_match(
            game, (ppo_agent, {"path": str(path)}), spec, config.screen_deals, seed=17
        ).score
        print(f"  screen r{r} it{it}: {s:.3f}", flush=True)
        if s > best_score:
            best, best_score = it, s
    final = OUT / f"br_r{r}.pt"
    (OUT / f"br_r{r}_it{best}.pt").replace(final)
    for it in snapshots:
        (OUT / f"br_r{r}_it{it}.pt").unlink(missing_ok=True)
    rating = duplicate_match(
        game, (ppo_agent, {"path": str(final)}), spec, config.final_deals, seed=21
    )
    return str(final), rating.score, rating.score_se


def main(config: Config) -> None:
    """Run (or resume) the PSRO rounds and print the summary."""
    state = _load() or {"population": list(config.initial), "scores": [], "rounds": []}
    pop = state["population"]
    n = len(pop)
    if len(state["scores"]) < n:  # (re)measure the whole initial matrix
        scores = np.full((n, n), 0.5)
        for i in range(n):
            for j in range(i + 1, n):
                s = _score(pop[i], pop[j], config.pair_deals, seed=31)
                scores[i, j], scores[j, i] = s, 1 - s
                print(f"{Path(pop[i]).stem} vs {Path(pop[j]).stem}: {s:.3f}", flush=True)
        state["scores"] = scores.tolist()
        _save(state)
    for r in range(len(state["rounds"]), config.rounds):
        scores = np.asarray(state["scores"])
        mixture, meta_exploit = solve_symmetric_meta_game(scores)
        weights = ", ".join(f"{Path(p).stem} {w:.2f}" for p, w in zip(pop, mixture, strict=True))
        print(f"\n=== round {r}: mixture [{weights}] ===", flush=True)
        br, score, se = _best_response(config, pop, mixture, r)
        print(f"best response r{r} vs mixture: {score:.3f}+-{se:.3f}", flush=True)
        row = [_score(br, p, config.pair_deals, seed=41 + r) for p in pop]
        n = len(pop)
        grown = np.full((n + 1, n + 1), 0.5)
        grown[:n, :n] = scores
        grown[n, :n] = row
        grown[:n, n] = [1 - s for s in row]
        pop.append(br)
        state["scores"] = grown.tolist()
        state["rounds"].append(
            {
                "mixture": dict(
                    zip([Path(p).stem for p in pop[:-1]], mixture.tolist(), strict=True)
                ),
                "meta_exploitability": meta_exploit,
                "best_response": br,
                "br_vs_mixture": score,
                "br_vs_mixture_se": se,
            }
        )
        _save(state)
    mixture, _ = solve_symmetric_meta_game(np.asarray(state["scores"]))
    print("\nround  best response vs mixture")
    for r, rd in enumerate(state["rounds"]):
        print(f"{r:>5}  {rd['br_vs_mixture']:.3f} +- {rd['br_vs_mixture_se']:.3f}")
    print(
        "final mixture: "
        + ", ".join(f"{Path(p).stem} {w:.2f}" for p, w in zip(pop, mixture, strict=True))
    )


if __name__ == "__main__":
    main(Config())
