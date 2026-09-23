"""Solve the Toy Showdown (mini Kuhn poker) with CFR and watch exploitability fall to ~0.

CFR is the right paradigm for imperfect-information games (unlike AlphaZero, whose learned value is
swamped by the deal's variance in a card game). Here it converges to a Nash equilibrium:
exploitability drives toward zero and the average strategy reproduces known Kuhn behavior (bluff the
sometimes; never bet the middle card as opener; fold the worst card to a bet).

Note on scale: this is *tabular* CFR — it enumerates information sets, so it fits Kuhn/Leduc-sized
games, not full Ventures (whose infoset space is astronomical). Large games need Deep CFR (Stage 5).

Run with::

    uv run python -m experiments.cfr_toy
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from iirl.agents.cfr import CFRSolver
from iirl.core.registry import make
from iirl.eval.exploitability import exploitability

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import toy_game  # noqa: E402, F401  (registers "toy_showdown")

_CARD = {0: "Low ", 1: "Mid ", 2: "High"}


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    checkpoints: tuple[int, ...] = (1, 10, 100, 1000, 10000)


def main(config: Config) -> None:
    """Train CFR to each checkpoint, printing exploitability, then show the learned strategy."""
    game = make("toy_showdown")
    solver = CFRSolver(game)
    print(f"{'iterations':>10}  {'exploitability':>14}")
    trained = 0
    for target in config.checkpoints:
        solver.train(target - trained)
        trained = target
        avg, actions = solver.average_strategy()
        print(f"{target:>10}  {exploitability(game, avg):>14.5f}")

    print("\nlearned average strategy (bet/call probability by card and history):")
    for key in sorted(avg):
        card, history = key.split("|")
        bet_prob = avg[key][actions[key].index(1)] if 1 in actions[key] else 0.0
        context = "open" if history == "" else f"after '{history}'"
        print(f"  {_CARD[int(card)]}  {context:<14}  bet/call {bet_prob:.2f}")


if __name__ == "__main__":
    main(Config())
