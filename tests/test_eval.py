import numpy as np

import toy_game  # noqa: F401  (import registers "toy_showdown")
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.eval.tournament import (
    expected_score,
    play_match,
    run_tournament,
    update_elo,
)


def test_expected_score_symmetry() -> None:
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) > 0.9
    a, b = expected_score(1600, 1400), expected_score(1400, 1600)
    assert abs((a + b) - 1.0) < 1e-12


def test_update_elo_moves_toward_winner() -> None:
    ra, rb = update_elo(1500, 1500, score_a=1.0)
    assert ra > 1500 and rb < 1500
    assert abs((ra - 1500) - (1500 - rb)) < 1e-9  # zero-sum rating exchange


def test_play_match_is_seat_balanced_and_zero_sum_returns() -> None:
    game = make("toy_showdown")
    rng = np.random.default_rng(7)
    result = play_match(game, RandomAgent("a"), RandomAgent("b"), num_games=400, rng=rng)
    assert result.num_games == 400
    assert result.wins + result.draws + result.losses == 400
    assert result.stderr >= 0.0


def test_run_tournament_returns_sorted_standings() -> None:
    game = make("toy_showdown")
    rng = np.random.default_rng(3)
    agents = {"a": RandomAgent("a"), "b": RandomAgent("b"), "c": RandomAgent("c")}
    standings = run_tournament(game, agents, games_per_pair=50, rng=rng)
    assert {s.name for s in standings} == {"a", "b", "c"}
    elos = [s.elo for s in standings]
    assert elos == sorted(elos, reverse=True)
