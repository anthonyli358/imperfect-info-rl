"""Evaluation: head-to-head matches, Elo, and round-robin tournaments.

Evaluation is a first-class module, not an afterthought. Two-player games have high per-game
variance (the "poker tax"), so matches alternate seats to cancel first-mover advantage and report
the standard error of the mean return alongside the point estimate. Treat any win-rate without a
sample-size / stderr caveat as unfinished.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game
from iirl.core.runner import run_episode

INITIAL_ELO = 1500.0
DEFAULT_ELO_K = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return A's expected score (win probability + half draw prob) under the Elo model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float, rating_b: float, score_a: float, k: float = DEFAULT_ELO_K
) -> tuple[float, float]:
    """Return updated ``(rating_a, rating_b)`` after a game A scored ``score_a`` in (1/0.5/0)."""
    exp_a = expected_score(rating_a, rating_b)
    delta = k * (score_a - exp_a)
    return rating_a + delta, rating_b - delta


def anchored_elo(score_rate: float, anchor_elo: float, clip: float = 0.02) -> float:
    """Return the Elo implied by scoring ``score_rate`` against a fixed-rating anchor.

    Inverts the Elo expected-score model: an even score matches the anchor, and a higher score sits
    above it. ``score_rate`` is clipped away from 0/1 so a clean sweep yields a finite (bounded)
    rating instead of ±∞. This is how a training run places a learner on a stable scale each
    iteration without re-rating the anchors.
    """
    s = min(1 - clip, max(clip, score_rate))
    return anchor_elo + 400.0 * np.log10(s / (1 - s))


@dataclass(frozen=True)
class MatchResult:
    """Aggregate outcome of a seat-balanced match between two agents.

    Attributes:
        agent_a: Name of the first agent.
        agent_b: Name of the second agent.
        num_games: Games played.
        wins: Games won by A (from A's payoff sign).
        draws: Drawn games.
        losses: Games lost by A.
        mean_return: Mean payoff to A per game.
        stderr: Standard error of ``mean_return`` (the honest uncertainty band).
    """

    agent_a: str
    agent_b: str
    num_games: int
    wins: int
    draws: int
    losses: int
    mean_return: float
    stderr: float


def play_match(
    game: Game,
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
    rng: np.random.Generator,
    swap_seats: bool = True,
) -> MatchResult:
    """Play ``num_games`` between two agents, alternating seats, from A's perspective.

    Args:
        game: The (two-player) game to play.
        agent_a: First agent.
        agent_b: Second agent.
        num_games: Number of games to play.
        rng: Seeded generator.
        swap_seats: If true, alternate which agent is player 0 to cancel seat bias.

    Returns:
        A :class:`MatchResult` aggregating the games from A's perspective.

    Raises:
        ValueError: If ``game`` is not two-player.
    """
    if game.num_players != 2:
        raise ValueError("play_match only supports two-player games")

    a_returns: list[float] = []
    wins = draws = losses = 0
    for i in range(num_games):
        a_is_p0 = (i % 2 == 0) or not swap_seats
        seats = (agent_a, agent_b) if a_is_p0 else (agent_b, agent_a)
        result = run_episode(game, seats, rng)
        a_return = result.returns[0] if a_is_p0 else result.returns[1]
        a_returns.append(a_return)
        if a_return > 0:
            wins += 1
        elif a_return < 0:
            losses += 1
        else:
            draws += 1

    arr = np.asarray(a_returns, dtype=float)
    stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return MatchResult(
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        num_games=num_games,
        wins=wins,
        draws=draws,
        losses=losses,
        mean_return=float(arr.mean()),
        stderr=stderr,
    )


@dataclass(frozen=True)
class Standing:
    """One agent's line in a tournament table."""

    name: str
    elo: float
    wins: int
    draws: int
    losses: int


def run_tournament(
    game: Game,
    agents: Mapping[str, Agent],
    games_per_pair: int,
    rng: np.random.Generator,
    elo_k: float = DEFAULT_ELO_K,
) -> list[Standing]:
    """Run a round-robin tournament and return standings sorted by Elo (descending).

    Every unordered pair plays ``games_per_pair`` seat-balanced games; Elo updates once per game.
    With small ``games_per_pair`` the resulting Elo carries real noise — treat it the way
    :func:`play_match` treats its stderr.

    Args:
        game: The game to play.
        agents: Mapping of display name to agent.
        games_per_pair: Games per unordered pair of agents.
        rng: Seeded generator.
        elo_k: Elo K-factor.

    Returns:
        Standings sorted by Elo, highest first.
    """
    names = list(agents)
    elo: dict[str, float] = {n: INITIAL_ELO for n in names}
    record: dict[str, list[int]] = {n: [0, 0, 0] for n in names}  # [wins, draws, losses]

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            match = play_match(game, agents[name_a], agents[name_b], games_per_pair, rng)
            record[name_a][0] += match.wins
            record[name_a][1] += match.draws
            record[name_a][2] += match.losses
            record[name_b][0] += match.losses
            record[name_b][1] += match.draws
            record[name_b][2] += match.wins
            # Replay the game-level scores into Elo (win=1, draw=0.5, loss=0 for A).
            scores = [1.0] * match.wins + [0.5] * match.draws + [0.0] * match.losses
            for score_a in scores:
                elo[name_a], elo[name_b] = update_elo(elo[name_a], elo[name_b], score_a, k=elo_k)

    standings = [Standing(n, elo[n], record[n][0], record[n][1], record[n][2]) for n in names]
    standings.sort(key=lambda s: s.elo, reverse=True)
    return standings
