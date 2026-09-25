"""Duplicate (paired-deal) matches, played in parallel.

Card-game results are dominated by the deal, so a plain seat-alternating match needs thousands of
games to resolve a few Elo points. A *duplicate* match plays every deal twice with the seats
swapped: the ``k``-th chance event of both games is drawn from the same seeded stream, so wherever
the chance distribution does not depend on the players' choices (Ventures' deck: its contents change
only by draws) both games see the same cards in the same order. Luck then cancels within each pair,
and the per-pair mean has far lower variance than two independent games.

Agents are built inside worker processes from picklable ``(factory, kwargs)`` specs, so networks and
search agents never cross a process boundary.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game

AgentSpec = tuple[Callable[..., Agent], dict[str, Any]]


@dataclass(frozen=True)
class DuplicateResult:
    """Outcome of a duplicate match from agent A's perspective.

    Attributes:
        deals: Deals played (each deal is two games, one per seating).
        score: A's score rate over all games (win 1, draw 0.5, loss 0).
        score_se: Standard error of ``score``, over per-deal means.
        margin: A's mean point margin per game.
        margin_se: Standard error of ``margin``, over per-deal means.
    """

    deals: int
    score: float
    score_se: float
    margin: float
    margin_se: float


def _play(game: Game, seats: tuple[Agent, Agent], deal_seed: int, agent_rng) -> float:
    """Play one game; the ``k``-th chance event uses a stream seeded by ``(deal_seed, k)``."""
    state = game.new_initial_state()
    k = 0
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            probs = np.array([p for _, p in outcomes])
            i = np.random.default_rng((deal_seed, k)).choice(len(outcomes), p=probs)
            state = game.apply_action(state, outcomes[i][0])
            k += 1
            continue
        player = game.current_player(state)
        state = game.apply_action(state, seats[player].step(game, state, agent_rng))
    return float(game.returns(state)[0])


def _worker(args) -> list[tuple[float, float]]:
    """Play the given deals both ways round; return A's ``(margin, score)`` per game."""
    game, spec_a, spec_b, deal_seeds = args
    import torch

    torch.set_num_threads(1)
    a, b = spec_a[0](**spec_a[1]), spec_b[0](**spec_b[1])
    out = []
    for seed in deal_seeds:
        rng = np.random.default_rng((seed, 1))
        for a_seat in (0, 1):
            seats = (a, b) if a_seat == 0 else (b, a)
            r0 = _play(game, seats, seed, rng)
            m = r0 if a_seat == 0 else -r0
            out.append((m, 1.0 if m > 0 else 0.5 if m == 0 else 0.0))
    return out


def duplicate_match(
    game: Game,
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    deals: int,
    seed: int = 0,
    workers: int = 12,
) -> DuplicateResult:
    """Play ``deals`` paired deals between A and B across ``workers`` processes.

    Args:
        game: A two-player game.
        spec_a: ``(factory, kwargs)`` that builds agent A in a worker.
        spec_b: ``(factory, kwargs)`` that builds agent B in a worker.
        deals: Number of deals (``2 * deals`` games).
        seed: Base seed; deal ``i`` uses seed ``(seed, i)``-derived chance streams.
        workers: Worker processes (1 runs in-process).
    """
    seeds = [int(s) for s in np.random.default_rng(seed).integers(2**62, size=deals)]
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    jobs = [(game, spec_a, spec_b, c) for c in chunks]
    if workers > 1:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
    else:
        parts = [_worker(j) for j in jobs]
    games = np.array([g for part in parts for g in part])  # consecutive pairs share a deal
    pairs = games.reshape(-1, 2, 2).mean(axis=1)  # per-deal (margin, score)
    se = pairs.std(axis=0, ddof=1) / np.sqrt(len(pairs))
    return DuplicateResult(
        deals=len(pairs),
        score=float(pairs[:, 1].mean()),
        score_se=float(se[1]),
        margin=float(pairs[:, 0].mean()),
        margin_se=float(se[0]),
    )
