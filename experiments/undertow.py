"""Undertow balance check: who wins when fixed agents play each side.

Undertow is asymmetric, so balance is measured per seating: the heuristic (or random play) as the
Smugglers against the heuristic (or random play) as the Watch, ``n`` games each (default 400),
with how each game ended: the Contraband reached the Free Port, it was sunk, the Watch raided the
Cove, a side could not move, or a draw at the round limit.

    uv run python -m experiments.undertow [n]
"""

from __future__ import annotations

import collections
import sys

import numpy as np

import iirl.games  # noqa: F401
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make, make_heuristic
from iirl.games.undertow import Undertow, UndertowState
from iirl.games.undertow.board import CONTRABAND, COVE, RAID, SMUGGLERS, WATCH
from iirl.games.undertow.game import REMOVED


def ending(game: Undertow, s: UndertowState) -> str:
    """How a finished game ended."""
    if s.winner < 0:
        return "draw"
    contraband = s.ident[SMUGGLERS].index(CONTRABAND)
    if s.winner == SMUGGLERS:
        return "S: contraband in" if s.pos[SMUGGLERS][contraband] >= 0 else "S: watch stuck"
    if s.pos[SMUGGLERS][contraband] == REMOVED:
        return "W: contraband sunk"
    if len(game.tokens_at(s, WATCH, COVE)) >= RAID:
        return "W: raid"
    return "W: smugglers stuck"


def play(game: Undertow, smugglers, watch, n: int, seed: int) -> dict:  # noqa: ANN001
    """``n`` games with the given agents seated; counts of each ending."""
    rng = np.random.default_rng(seed)
    out: collections.Counter = collections.Counter()
    for _ in range(n):
        s = game.new_initial_state()
        smugglers.reset(game, SMUGGLERS)
        watch.reset(game, WATCH)
        while not game.is_terminal(s):
            if game.is_chance_node(s):
                o = game.chance_outcomes(s)
                s = game.apply_action(s, o[rng.integers(len(o))][0])
                continue
            agent = smugglers if game.current_player(s) == SMUGGLERS else watch
            s = game.apply_action(s, agent.step(game, s, rng))
        out[ending(game, s)] += 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    game = make("undertow")
    h, r = make_heuristic("undertow"), RandomAgent("random")
    for name, sm, wa in [
        ("heuristic vs heuristic", h, h),
        ("heuristic Smugglers vs random Watch", h, r),
        ("random Smugglers vs heuristic Watch", r, h),
        ("random vs random", r, r),
    ]:
        print(f"{name}: {play(game, sm, wa, n, 1)}", flush=True)
