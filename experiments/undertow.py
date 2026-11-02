"""Undertow balance check: who wins when fixed agents play each side.

Undertow is asymmetric, so balance is measured per seating: the heuristic (or random play) as the
Smugglers against the heuristic (or random play) as the Watch, ``n`` games each (default 400),
with how each game ended: the Contraband reached the Free Port, it was sunk, the Watch raided the
Cove, a side could not move, or a draw at the round limit.

    uv run python -m experiments.undertow [n]
    uv run python -m experiments.undertow pimc [games=200]   # PIMC on each side vs the heuristic

``pimc`` plays PIMC (determinized UCT, ``agents/mcts/pimc.py``, with the game's leaf value)
against the heuristic, seated as each side in turn, at two budgets, on 4 worker processes;
results are appended to ``runs/undertow/pimc.jsonl``.
"""

from __future__ import annotations

import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

import iirl.games  # noqa: F401
from iirl.agents.mcts.pimc import MCTSConfig, PIMCAgent
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


def agent(spec: str):  # noqa: ANN201
    """``heuristic``, ``random``, or ``pimc:<worlds>x<sims>``."""
    if spec == "heuristic":
        return make_heuristic("undertow")
    if spec == "random":
        return RandomAgent("random")
    worlds, sims = (int(x) for x in spec.split(":")[1].split("x"))
    config = MCTSConfig(num_determinizations=worlds, sims_per_world=sims, max_rollout_depth=20)
    return PIMCAgent(config, name=spec)


def _games(args: tuple) -> dict:
    smugglers, watch, n, seed = args
    import torch

    torch.set_num_threads(1)
    return play(make("undertow"), agent(smugglers), agent(watch), n, seed)


def pimc() -> None:
    """PIMC seated as each side against the heuristic, two budgets."""
    games = int(dict(a.split("=") for a in sys.argv[2:]).get("games", 200))
    out = Path("runs/undertow/pimc.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(4) as pool:
        for budget in ("pimc:8x40", "pimc:16x80"):
            for smugglers, watch in ((budget, "heuristic"), ("heuristic", budget)):
                t0 = time.perf_counter()
                jobs = [(smugglers, watch, games // 4, 100 + k) for k in range(4)]
                total: collections.Counter = collections.Counter()
                for part in pool.map(_games, jobs):
                    total.update(part)
                n = sum(total.values())
                wins = sum(v for k, v in total.items() if k.startswith("S:"))
                p = wins / n
                row = {
                    "smugglers": smugglers,
                    "watch": watch,
                    "games": n,
                    "smuggler_wins": wins,
                    "smuggler_win_rate": round(p, 4),
                    "se": round((p * (1 - p) / n) ** 0.5, 4),
                    "endings": dict(total),
                    "minutes": round((time.perf_counter() - t0) / 60, 1),
                }
                print(json.dumps(row), flush=True)
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pimc":
        pimc()
        sys.exit()
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
