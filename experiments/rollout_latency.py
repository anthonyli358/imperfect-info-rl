"""How long does the rollout agent think? Per-decision latency against the plain policy.

Plays ``games`` games of the rollout agent (``agents/rollout.py``) against its own policy, one
process, and times every one of the agent's decisions. Searched decisions (deck at or below the
threshold, at the start of a turn) are reported separately from the instant policy moves.

Run with (on an otherwise idle machine)::

    uv run python -m experiments.rollout_latency camel_stocks 12 25
    uv run python -m experiments.rollout_latency ventures 6
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import iirl.games  # noqa: F401  (registers the games)
from iirl.agents.ppo import PPOAgent
from iirl.agents.rollout import rollout_agent
from iirl.core.registry import make

BLUEPRINT = {"camel_stocks": "runs/camel_nash.pt", "ventures": "runs/diverse_sym.pt"}


def measure(game_name: str, deck_max: int, games: int = 20, **settings: object) -> dict:
    """Time every decision of the rollout agent over ``games`` games (alternating seats)."""
    torch.set_num_threads(1)
    game = make(game_name)
    agent = rollout_agent(
        blueprint=BLUEPRINT[game_name], deck_max=deck_max, robust=True, **settings
    )
    policy = PPOAgent.from_checkpoint(BLUEPRINT[game_name], greedy=True)
    searched, instant, per_game = [], [], []
    for g in range(games):
        rng = np.random.default_rng(g)
        seat = g % 2
        state = game.new_initial_state()
        total = 0.0
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            elif game.current_player(state) == seat:
                before = agent.searched
                t0 = time.perf_counter()
                action = agent.step(game, state, rng)
                dt = time.perf_counter() - t0
                total += dt
                (searched if agent.searched > before else instant).append(dt)
            else:
                action = policy.step(game, state, rng)
            state = game.apply_action(state, action)
        per_game.append(total)
    s = np.asarray(searched)
    return {
        "game": game_name,
        "deck_max": deck_max,
        "games": games,
        "searched_per_game": round(len(s) / games, 1),
        "search_median_s": round(float(np.median(s)), 2) if len(s) else 0.0,
        "search_p90_s": round(float(np.percentile(s, 90)), 2) if len(s) else 0.0,
        "search_max_s": round(float(s.max()), 2) if len(s) else 0.0,
        "policy_move_ms": round(1000 * float(np.median(instant)), 1),
        "per_game_mean_s": round(float(np.mean(per_game)), 1),
        "per_game_max_s": round(float(np.max(per_game)), 1),
    }


if __name__ == "__main__":
    name = sys.argv[1]
    out = Path("runs/rollout_latency.jsonl")
    for threshold in sys.argv[2:]:
        row = measure(name, int(threshold))
        print(json.dumps(row), flush=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
