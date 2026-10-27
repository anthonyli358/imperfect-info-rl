"""How long does Tessera's search agent think? Per-move latency at a given simulation budget.

Plays ``games`` games of net-guided PUCT search (the play page's agent: priors and leaf values
from the AlphaZero loop's nets) against the plain policy, in one process on one thread, and times
every one of the search's moves. One thread keeps the machine cool and matches a play-page
session, which serves one game at a time.

Run with (on an otherwise idle machine)::

    uv run python -m experiments.search_latency 400 1600
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import iirl.games  # noqa: F401  (registers "tessera")
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.play.server import search_agent

PRIOR = "runs/tessera_ag/exit_prior_48.pt"
VALUE = "runs/tessera_ag/exit_value_48.pt"
POLICY = "runs/tessera_nash2.pt"


def measure(simulations: int, games: int = 6) -> dict:
    """Time every move of the search agent over ``games`` games (alternating seats)."""
    torch.set_num_threads(1)
    game = make("tessera")
    agent = search_agent(PRIOR, VALUE, simulations)
    policy = PPOAgent.from_checkpoint(POLICY, greedy=True)
    moves, per_game = [], []
    for g in range(games):
        rng = np.random.default_rng(g)
        seat = g % 2
        state = game.new_initial_state()
        agent.reset(game, seat)
        total = 0.0
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            elif game.current_player(state) == seat:
                t0 = time.perf_counter()
                action = agent.step(game, state, rng)
                dt = time.perf_counter() - t0
                total += dt
                moves.append(dt)
            else:
                action = policy.step(game, state, rng)
            state = game.apply_action(state, action)
        per_game.append(total)
    m = np.asarray(moves)
    return {
        "game": "tessera",
        "simulations": simulations,
        "nets": "exit_48",
        "games": games,
        "moves_per_game": round(len(m) / games, 1),
        "move_median_s": round(float(np.median(m)), 2),
        "move_p90_s": round(float(np.percentile(m, 90)), 2),
        "move_max_s": round(float(m.max()), 2),
        "per_game_mean_s": round(float(np.mean(per_game)), 1),
        "per_game_max_s": round(float(np.max(per_game)), 1),
    }


if __name__ == "__main__":
    out = Path("runs/search_latency.jsonl")
    for sims in sys.argv[1:]:
        row = measure(int(sims))
        print(json.dumps(row), flush=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
