"""Liar's Dice (OpenSpiel's engine): our solvers checked against OpenSpiel's on the same game.

``crosscheck`` runs our vectorized CFR+ (``agents.cfr.graph_cfr``; simultaneous and alternating
updates) and OpenSpiel's C++ CFR+ for the same iterations, and scores each one's average strategy
with both exploitability evaluators: ours (``eval.exact_br``, through the ``games.spiel`` adapter)
and OpenSpiel's ``pyspiel.exploitability``. Agreement checks the adapter, our best response and
our CFR+ at once. Both report NashConv / 2: the payoff a perfect exploiter gains per game,
averaged over seats (payoffs are +1 / -1).

Run with (needs ``uv sync --extra openspiel``; one process)::

    uv run python -m experiments.liars_dice crosscheck
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyspiel

import iirl.games  # noqa: F401  (registers "liars_dice")
from iirl.agents.cfr.graph_cfr import GraphCFR
from iirl.core.game import Game, State
from iirl.core.registry import make
from iirl.eval.exact_br import exploitability, uniform_policy

OUT = Path("runs/liars_dice")


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    checkpoints: tuple[int, ...] = (10, 30, 100, 300, 1000)


def table_of(game: Game, policy, states: Sequence[State]) -> pyspiel.TabularPolicy:  # noqa: ANN001
    """A batched policy over ``states`` as an OpenSpiel tabular policy (keyed by infostate)."""
    first: dict[str, State] = {}
    for s in states:
        if not game.is_terminal(s) and not game.is_chance_node(s):
            first.setdefault(game.information_state_string(s, game.current_player(s)), s)
    keys = list(first)
    out = {}
    for key, probs in zip(keys, policy(game, [first[k] for k in keys]), strict=True):
        legal = game.legal_actions(first[key])
        out[key] = [(a, float(probs[a])) for a in legal]
    return pyspiel.TabularPolicy(out)


def from_spiel(spiel_policy):  # noqa: ANN001, ANN201 - a BatchPolicy
    """An OpenSpiel policy as our batched policy."""

    def policy(game: Game, states: Sequence[State]) -> list[np.ndarray]:
        out = []
        for s in states:
            probs = np.zeros(game.num_distinct_actions())
            for a, p in spiel_policy.action_probabilities(s.inner).items():
                probs[a] = p
            out.append(probs)
        return out

    return policy


def crosscheck(config: Config) -> None:
    """Both solvers, both evaluators, at each checkpoint; saved to ``crosscheck.jsonl``."""
    OUT.mkdir(parents=True, exist_ok=True)
    game = make("liars_dice")
    spiel_game = game.spiel
    t0 = time.perf_counter()
    ours = GraphCFR(game)
    print(
        f"graph: {len(ours.graph.states):,} states, {len(ours.infoset_keys):,} infosets "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    alt = GraphCFR(game, ours.graph, alternating=True)
    uni_ours = exploitability(game, uniform_policy, ours.graph)["exploitability"]
    uni_spiel = pyspiel.exploitability(
        spiel_game, table_of(game, uniform_policy, ours.graph.states)
    )
    print(f"uniform policy: ours {uni_ours:.6f}, OpenSpiel {uni_spiel:.6f}", flush=True)
    theirs = pyspiel.CFRPlusSolver(spiel_game)
    rows = []
    done = 0
    for target in config.checkpoints:
        t = time.perf_counter()
        ours.iterate(target - done)
        t_ours = time.perf_counter() - t
        t = time.perf_counter()
        alt.iterate(target - done)
        t_alt = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(target - done):
            theirs.evaluate_and_update_policy()
        t_theirs = time.perf_counter() - t
        done = target
        our_avg = ours.average_policy()
        our_table = table_of(game, our_avg, ours.graph.states)
        their_avg = theirs.average_policy()
        e_oo = exploitability(game, our_avg, ours.graph)
        e_ot = pyspiel.exploitability(spiel_game, our_table)
        e_to = exploitability(game, from_spiel(their_avg), ours.graph)
        e_tt = pyspiel.exploitability(spiel_game, their_avg)
        alt_avg = alt.average_policy()
        e_aa = exploitability(game, alt_avg, ours.graph)
        e_at = pyspiel.exploitability(spiel_game, table_of(game, alt_avg, ours.graph.states))
        row = {
            "iterations": target,
            "ours_by_ours": round(e_oo["exploitability"], 6),
            "ours_by_openspiel": round(e_ot, 6),
            "openspiel_by_ours": round(e_to["exploitability"], 6),
            "openspiel_by_openspiel": round(e_tt, 6),
            "ours_alternating_by_ours": round(e_aa["exploitability"], 6),
            "ours_alternating_by_openspiel": round(e_at, 6),
            # The first player's game value, from the best responses to our alternating average.
            "value_p0": round((e_aa["br_as_0"] - e_aa["br_as_1"]) / 2, 6),
            "seconds_ours": round(t_ours, 1),
            "seconds_ours_alternating": round(t_alt, 1),
            "seconds_openspiel": round(t_theirs, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        with (OUT / "crosscheck.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    {"crosscheck": crosscheck}[sys.argv[1]](Config())
