"""Does the agent's style depend on its opponent? The same questions against different opponents.

``analyze_play`` measured the best agents against the plain heuristics, which have exploitable
habits (the Ventures heuristic hands over a playable card 25% of the time even when a safe discard
exists). Here the agent also plays:

* a **careful** heuristic (Ventures): identical, except it never gifts a playable card when it can
  avoid it. This is the discard discipline a careful human has.
* **itself** (both games): self-play, where the opponent has the agent's own habits.

If the agent's edge came from feeding on the plain heuristic's gifts, its results against the
careful opponent should drop, and its "lives off the discard piles" statistics should change.

Writes ``runs/analysis_opponents.json`` and prints a table per game.

Run with::

    uv run python -m experiments.analyze_opponents
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import iirl.games  # noqa: F401  (registers the built-in games)
from experiments.analyze_play import (
    analyze_camel_stocks,
    analyze_ventures,
    summarize,
    ventures_questions,
)
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make, make_heuristic


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    games: int = 400
    ventures_agent: str = "runs/league_control.pt"
    camel_agent: str = "runs/camel_nash.pt"
    seed: int = 11


VENTURES_KEYS = (
    "win",
    "score",
    "route_value",
    "losing_route",
    "discard_take_rate",
    "gift_discard_rate",
)
QUESTION_KEYS = (
    "early_open_strong",
    "takes_playable_top",
    "takes_unplayable_top",
    "wagered_route_positive",
)
CAMEL_KEYS = (
    "win",
    "score",
    "share_exchange",
    "herd_at_end",
    "camel_bonus",
    "bonus_sale",
    "sale_points",
    "dilemma_precious",
    "deck_cards_revealed_per_turn",
    "left_precious_in_market",
)


def main(config: Config) -> None:
    """Run every matchup and print the agent's side of each."""
    rng = np.random.default_rng(config.seed)
    out: dict = {}

    game = make("ventures")
    agent = PPOAgent.from_checkpoint(config.ventures_agent)
    opponents = {
        "heuristic": make_heuristic("ventures"),
        "careful": make_heuristic("ventures_careful"),
        "itself": PPOAgent.from_checkpoint(config.ventures_agent),
    }
    table = {}
    for name, opp in opponents.items():
        stats = analyze_ventures(game, agent, opp, config.games, rng)
        questions = summarize(ventures_questions(game, agent, opp, config.games, rng))
        table[name] = {
            **{k: stats["agent"].get(k) for k in VENTURES_KEYS},
            **{k: questions["agent"].get(k, {}).get("mean") for k in QUESTION_KEYS},
            "opp_gift_discard_rate": stats["heuristic"].get("gift_discard_rate"),
        }
    out["ventures"] = table
    _print("ventures", table)

    game = make("camel_stocks")
    agent = PPOAgent.from_checkpoint(config.camel_agent)
    opponents = {
        "heuristic": make_heuristic("camel_stocks"),
        "itself": PPOAgent.from_checkpoint(config.camel_agent),
    }
    table = {}
    for name, opp in opponents.items():
        stats = analyze_camel_stocks(game, agent, opp, config.games, rng)
        table[name] = {k: stats["agent"].get(k) for k in CAMEL_KEYS}
    out["camel_stocks"] = table
    _print("camel_stocks", table)

    Path("runs/analysis_opponents.json").write_text(json.dumps(out, indent=2), "utf-8")


def _print(game: str, table: dict) -> None:
    names = list(table)
    print(f"\n=== {game}: the agent's statistics against each opponent ===")
    print(f"{'statistic':<30}" + "".join(f"{n:>12}" for n in names))
    for key in table[names[0]]:
        cells = "".join(
            f"{table[n][key]:>12.3f}" if table[n][key] is not None else f"{'-':>12}" for n in names
        )
        print(f"{key:<30}{cells}")


if __name__ == "__main__":
    main(Config())
