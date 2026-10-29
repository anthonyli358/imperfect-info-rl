"""Elo ladder: rate every checkpoint in the Ventures agent lineage on one fixed yardstick.

Each checkpoint plays the heuristic (anchored at 1676 Elo) over the same duplicate deals, so the
numbers are comparable across runs (about ±10 Elo at 1000 deals, vs ±35 for 400-game in-training
evals). Results go to ``runs/elo_ladder.json``; checkpoints already rated there are skipped, so
this is cheap to rerun as the lineage grows, and safe to stop at any time.

A lineage entry ``rollout:<checkpoint>`` is that checkpoint with opponent-robust endgame
rollouts (``agents/rollout.py``), ``search:<prior.pt>|<value.pt>|<simulations>`` net-guided
PUCT search (Tessera), and ``guided:<net.pt>|<simulations>`` net-guided PIMC (learned priors,
the heuristic's honest leaf value, one sampled world per move; ``train_ventures_guided``).

**Two Ventures scales.** The plain heuristic leaks (it hands over playable cards), so the best
agents score 0.85-0.92 against it and their ratings bunch up near its ceiling. ``careful`` rates
the same lineage against the careful heuristic instead, after placing the careful heuristic on the
same scale with a heuristic-vs-heuristic match (so the two ladders share units; the careful one
separates strong agents better).

Run with::

    uv run python -m experiments.elo_ladder              # Ventures, plain-heuristic anchor
    uv run python -m experiments.elo_ladder careful      # Ventures, careful-heuristic anchor
    uv run python -m experiments.elo_ladder camel_stocks # Camel Stocks (its heuristic = 1500)
    uv run python -m experiments.elo_ladder tessera      # Tessera (its heuristic = 1500)
    uv run python -m experiments.elo_ladder tessera_strong  # Tessera's strong end (see below)

**Two Tessera scales.** Search beats Tessera's heuristic 0.95-0.99 of the time, where Elo is
compressed and noisy. ``tessera_strong`` rates the strong agents against the ``tessera_nash2``
policy instead, anchored at that policy's rating on the heuristic ladder (so the scales share
units; run ``tessera`` first).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import iirl.games  # noqa: F401  (registers "ventures")
from experiments.exploit_ventures import heuristic_agent, ppo_agent
from iirl.agents.base import Agent
from iirl.agents.heuristic import CamelStocksHeuristic
from iirl.agents.rollout import DECK_MAX, rollout_agent
from iirl.core.registry import make, make_heuristic
from iirl.eval.duplicate import duplicate_match
from iirl.eval.tournament import anchored_elo
from iirl.play.server import search_agent

HEURISTIC_ELO = 1676.0


@dataclass(frozen=True)
class Config:
    """Experiment settings.

    Attributes:
        lineage: ``(label, checkpoint)`` in training order; missing checkpoints are skipped.
        deals: Duplicate deals per rating (``2 *`` games).
        out: The results file.
        game: Registered game name.
        heuristic: Factory for the anchor agent.
        anchor_elo: The anchor's fixed rating.
    """

    lineage: tuple[tuple[str, str], ...] = (
        ("PPO v2", "runs/ppo_ventures_best.pt"),
        ("nash control", "runs/nash_control.pt"),
        ("nash", "runs/nash_nash.pt"),
        *((f"nash_long {i}", f"runs/nash_long_{i}.pt") for i in range(100, 700, 100)),
        *((f"league r{r}", f"runs/league_r{r}.pt") for r in range(1, 6)),
        ("league control", "runs/league_control.pt"),
        *((f"nash_g {i}", f"runs/nash_g_{i}.pt") for i in range(100, 500, 100)),
        *((f"rnad {i}", f"runs/rnad_{i}.pt") for i in range(100, 500, 100)),
        *((f"careful {i}", f"runs/careful_{i}.pt") for i in range(100, 500, 100)),
        *((f"diverse {i}", f"runs/diverse_{i}.pt") for i in range(100, 500, 100)),
        *((f"diverse_sym {i}", f"runs/diverse_sym_{i}.pt") for i in range(100, 500, 100)),
        ("diverse_sym + rollouts", "rollout:runs/diverse_sym.pt"),
        ("net-guided PIMC 256", "guided:runs/guided_v2/guided_ventures.pt|256"),
    )
    deals: int = 1000
    # Fewer deals for slow entries (label prefix -> deals).
    slow_deals: tuple[tuple[str, int], ...] = (("net-guided PIMC", 200),)
    out: str = "runs/elo_ladder.json"
    game: str = "ventures"
    heuristic: Callable[[], Agent] = heuristic_agent
    anchor_elo: float = HEURISTIC_ELO


CAMEL_STOCKS = Config(
    lineage=(
        *((f"camel_ppo {i}", f"runs/camel_ppo_{i}.pt") for i in range(100, 700, 100)),
        *((f"camel_nash {i}", f"runs/camel_nash_{i}.pt") for i in range(100, 500, 100)),
        ("camel_nash + rollouts", "rollout:runs/camel_nash.pt"),
    ),
    out="runs/elo_ladder_camel_stocks.json",
    game="camel_stocks",
    heuristic=CamelStocksHeuristic,
    anchor_elo=1500.0,
)


def tessera_heuristic() -> Agent:
    """Tessera's hand-tuned heuristic (the 1500 anchor)."""
    return make_heuristic("tessera")


AG = "runs/tessera_ag"
TESSERA_SEARCH = tuple(
    (f"search400 exit {i}", f"search:{AG}/exit_prior_{i}.pt|{AG}/exit_value_{i}.pt|400")
    for i in (24, 36, 48)
)
TESSERA_STRONG_LINEAGE = (
    ("tessera_nash2", "runs/tessera_nash2.pt"),
    *((f"exit {i} policy", f"{AG}/exit_prior_{i}.pt") for i in (24, 36, 48)),
    ("search400 start", f"search:runs/tessera_nash2.pt|{AG}/value.pt|400"),
    *TESSERA_SEARCH,
)
TESSERA = Config(
    lineage=(
        *((f"tessera_ppo {i}", f"runs/tessera_ppo_{i}.pt") for i in range(100, 700, 100)),
        *((f"tessera_nash {i}", f"runs/tessera_nash_{i}.pt") for i in range(100, 500, 100)),
        *((f"tessera_nash2 {i}", f"runs/tessera_nash2_{i}.pt") for i in range(200, 1300, 200)),
        *TESSERA_STRONG_LINEAGE,
    ),
    out="runs/elo_ladder_tessera.json",
    game="tessera",
    heuristic=tessera_heuristic,
    anchor_elo=1500.0,
)


def tessera_strong_config() -> Config:
    """Tessera's strong end, rated against ``tessera_nash2`` at its heuristic-ladder Elo."""
    base = json.loads(Path(TESSERA.out).read_text("utf-8"))["tessera_nash2"]
    print(f"tessera_nash2 = {base['elo']:.0f} Elo on the heuristic scale", flush=True)
    return Config(
        lineage=TESSERA_STRONG_LINEAGE[1:],
        out="runs/elo_ladder_tessera_strong.json",
        game="tessera",
        heuristic=partial(ppo_agent, "runs/tessera_nash2.pt"),
        anchor_elo=base["elo"],
    )


ROLLOUT = "rollout:"
SEARCH = "search:"
GUIDED = "guided:"


def guided_agent(path: str, simulations: int) -> Agent:
    """Net-guided PIMC: learned priors over PUCT, the game's heuristic value at the leaves."""
    import torch

    from iirl.agents.alphazero.agent import AlphaZeroAgent
    from iirl.agents.alphazero.mcts import PUCTConfig
    from iirl.nn.policy_value import NetEvaluator, PolicyValueNet

    torch.set_num_threads(1)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ck["input_dim"], ck["num_actions"])
    net.load_state_dict(ck["state_dict"])
    config = PUCTConfig(simulations=simulations, determinizations=1, leaf_value="heuristic")
    return AlphaZeroAgent(NetEvaluator(net), config, name=f"guided{simulations}")


def entry_files(path: str) -> list[str]:
    """The checkpoint files a lineage entry needs (to skip entries not trained yet)."""
    rest = path.removeprefix(ROLLOUT).removeprefix(SEARCH).removeprefix(GUIDED)
    return rest.split("|")[:1] if path.startswith(GUIDED) else rest.split("|")[:2]


def agent_spec(path: str, game: str = "ventures") -> tuple:
    """A picklable agent spec for a lineage entry (a checkpoint, ``rollout:`` or ``search:``)."""
    if path.startswith(SEARCH):
        prior, value, sims = path[len(SEARCH) :].split("|")
        return search_agent, {"policy": prior, "value": value, "simulations": int(sims)}
    if path.startswith(GUIDED):
        net, sims = path[len(GUIDED) :].split("|")
        return guided_agent, {"path": net, "simulations": int(sims)}
    if path.startswith(ROLLOUT):
        blueprint = path[len(ROLLOUT) :]
        return rollout_agent, {
            "blueprint": blueprint,
            "deck_max": DECK_MAX[game],
            "robust": True,
        }
    return ppo_agent, {"path": path}


def careful_heuristic() -> Agent:
    """The careful heuristic (never hands over a playable card)."""
    return make_heuristic("ventures_careful")


def careful_config() -> Config:
    """The Ventures ladder anchored on the careful heuristic, placed on the plain scale."""
    out = Path("runs/elo_ladder_careful.json")
    ladder = json.loads(out.read_text("utf-8")) if out.exists() else {}
    if "_anchor" not in ladder:
        game = make("ventures")
        r = duplicate_match(game, (careful_heuristic, {}), (heuristic_agent, {}), 2000)
        ladder["_anchor"] = {
            "label": "careful heuristic vs the plain heuristic",
            "score": round(r.score, 4),
            "elo": round(anchored_elo(r.score, HEURISTIC_ELO), 1),
            "deals": r.deals,
        }
        out.write_text(json.dumps(ladder, indent=2), "utf-8")
    anchor = ladder["_anchor"]["elo"]
    print(f"careful heuristic = {anchor:.0f} Elo on the plain-heuristic scale", flush=True)
    return Config(out=str(out), heuristic=careful_heuristic, anchor_elo=anchor)


def main(config: Config) -> None:
    """Rate each unrated checkpoint vs the heuristic; print and save the ladder."""
    game = make(config.game)
    out = Path(config.out)
    ladder = json.loads(out.read_text("utf-8")) if out.exists() else {}
    for label, path in config.lineage:
        if label in ladder or not all(Path(f).exists() for f in entry_files(path)):
            continue
        spec = agent_spec(path, config.game)
        deals = next((n for p, n in config.slow_deals if label.startswith(p)), config.deals)
        r = duplicate_match(game, spec, (config.heuristic, {}), deals, workers=4)  # keeps it cool
        ladder[label] = {
            "checkpoint": path,
            "elo": round(anchored_elo(r.score, config.anchor_elo), 1),
            # Elo is logistic in the score, so +-1 SE on the score maps to this Elo error.
            "elo_se": round(
                (
                    anchored_elo(r.score + r.score_se, config.anchor_elo)
                    - anchored_elo(r.score - r.score_se, config.anchor_elo)
                )
                / 2,
                1,
            ),
            "score": round(r.score, 4),
            "margin": round(r.margin, 2),
            "deals": r.deals,
        }
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(ladder, indent=2), "utf-8")
        tmp.replace(out)
        print(
            f"{label:<16} elo {ladder[label]['elo']:6.0f} ± {ladder[label]['elo_se']:.0f}",
            flush=True,
        )
    order = [label for label, _ in config.lineage if label in ladder]
    print("\nlabel            elo      score   margin")
    for label in order:
        e = ladder[label]
        print(
            f"{label:<16} {e['elo']:6.0f}±{e['elo_se']:<3.0f} {e['score']:.3f}  {e['margin']:+6.1f}"
        )


if __name__ == "__main__":
    import sys

    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice == "camel_stocks":
        main(CAMEL_STOCKS)
    elif choice == "tessera":
        main(TESSERA)
    elif choice == "tessera_strong":
        main(tessera_strong_config())
    else:
        main(careful_config() if choice == "careful" else Config())
