"""Undertow balance check: who wins when fixed agents play each side.

Undertow is asymmetric, so balance is measured per seating: the heuristic (or random play) as
the Smugglers against the heuristic (or random play) as the Watch, ``n`` games each (default
400), with how each game ended: delivered (a Smuggler win), out of reach (cargo sunk), or the tide.

    uv run python -m experiments.undertow [n]
"""

import collections
import sys

import numpy as np

import iirl.games  # noqa: F401
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make, make_heuristic


def play(g, sm, wa, n, seed):
    rng = np.random.default_rng(seed)
    res = collections.Counter()
    deliv = []
    for _ in range(n):
        s = g.new_initial_state()
        while not g.is_terminal(s):
            if g.is_chance_node(s):
                o = g.chance_outcomes(s)
                s = g.apply_action(s, o[rng.integers(len(o))][0])
                continue
            ag = sm if g.current_player(s) == 0 else wa
            s = g.apply_action(s, ag.step(g, s, rng))
        why = "delivered" if s.winner == 0 else ("tide" if s.round >= 24 else "out of reach")
        res[why] += 1
        deliv.append(s.delivered)
    return dict(res), round(float(np.mean(deliv)), 2)


if __name__ == "__main__":
    g = make("undertow")
    H, R = make_heuristic("undertow"), RandomAgent("random")
    for name, sm, wa in [
        ("H vs H", H, H),
        ("H smug vs R watch", H, R),
        ("R smug vs H watch", R, H),
    ]:
        print(name, play(g, sm, wa, int(sys.argv[1]) if len(sys.argv) > 1 else 400, 1), flush=True)
