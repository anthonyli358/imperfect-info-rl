"""Spendthrift's rules and the perfect-information solver."""

import numpy as np
import pytest

from iirl.eval.perfect_info import best_response_value, nash_conv, policy_values, solve, uniform
from iirl.games.spendthrift import PASS, Phase, Spendthrift, SpendthriftState


def _flip(game: Spendthrift, s: SpendthriftState, card: int) -> SpendthriftState:
    assert game.is_chance_node(s)
    return game.apply_action(s, card)


def test_winner_pays_and_passers_pay_nothing() -> None:
    game = Spendthrift(players=3, lots=4, ends=2, coins=6)
    s = _flip(game, game.new_initial_state(), 3)
    assert s.phase == Phase.AUCTION and s.to_move == 0
    s = game.apply_action(s, 2)  # player 0 bids 2
    s = game.apply_action(s, 4)  # player 1 raises to 4
    s = game.apply_action(s, PASS)  # player 2 passes
    s = game.apply_action(s, PASS)  # player 0 passes: player 1 wins
    assert s.phase == Phase.REVEAL
    assert s.coins == (6, 2, 6) and s.totals == (0, 3, 0) and s.best == (0, 3, 0)


def test_lot_discarded_when_everyone_passes() -> None:
    game = Spendthrift(players=2, lots=3, ends=1, coins=4)
    s = _flip(game, game.new_initial_state(), 2)
    s = game.apply_action(game.apply_action(s, PASS), PASS)
    assert s.phase == Phase.REVEAL and s.totals == (0, 0) and s.coins == (4, 4)
    assert s.deck == 0b101


def test_last_bidder_standing_wins_after_others_pass() -> None:
    game = Spendthrift(players=2, lots=3, ends=1, coins=4)
    s = _flip(game, game.new_initial_state(), 1)
    s = game.apply_action(s, PASS)  # player 0 passes; player 1 may still bid
    assert s.to_move == 1 and game.legal_actions(s) == [PASS, 1, 2, 3, 4]
    s = game.apply_action(s, 1)
    assert s.coins == (4, 3) and s.totals == (0, 1)


def test_opener_rotates_and_bids_are_capped_by_coins() -> None:
    game = Spendthrift(players=2, lots=3, ends=1, coins=4)
    s = _flip(game, game.new_initial_state(), 1)
    s = game.apply_action(game.apply_action(s, 4), PASS)  # player 0 spends everything
    s = _flip(game, s, 3)
    assert s.to_move == 1  # the second lot is opened by player 1
    s = game.apply_action(s, 1)
    assert game.legal_actions(s) == [PASS]  # player 0 has no coins left


def test_end_card_ends_the_game() -> None:
    game = Spendthrift(players=2, lots=3, ends=2, coins=4)
    s = _flip(game, game.new_initial_state(), 0)
    assert game.is_chance_node(s) and s.ends_seen == 1
    s = _flip(game, s, 0)
    assert game.is_terminal(s)


@pytest.mark.parametrize(
    ("coins", "totals", "best", "expected"),
    [
        # The poorer player loses their best lot: 6 - 4 = 2 < 3.
        ((1, 3), (6, 3), (4, 3), [-1.0, 1.0]),
        # Both tied poorest: both discard (2 vs 1); then higher score wins.
        ((2, 2), (6, 3), (4, 2), [1.0, -1.0]),
        # Equal scores: more coins wins.
        ((3, 1), (3, 4), (3, 1), [1.0, -1.0]),
        # Equal scores and coins: shared.
        ((2, 2), (0, 0), (0, 0), [0.0, 0.0]),
    ],
)
def test_reserve_penalty_and_tie_breaks(coins, totals, best, expected) -> None:  # noqa: ANN001
    game = Spendthrift(players=2, lots=5, ends=2, coins=8)
    s = SpendthriftState(coins, totals, best, deck=0, ends_left=0, phase=Phase.OVER)
    assert game.returns(s) == pytest.approx(expected)


def test_three_player_payoffs_are_zero_sum_with_winner_plus_one() -> None:
    game = Spendthrift(players=3, lots=4, ends=2, coins=6)
    s = SpendthriftState((3, 3, 1), (5, 2, 4), (4, 2, 4), deck=0, ends_left=0, phase=Phase.OVER)
    r = game.returns(s)
    assert r == pytest.approx([1.0, -0.5, -0.5]) and sum(r) == pytest.approx(0.0)


@pytest.mark.parametrize("players", [2, 3])
def test_solution_is_an_equilibrium(players: int) -> None:
    game = Spendthrift(players=players, lots=3, ends=1, coins=4)
    sol = solve(game)
    root = sol.values[game.new_initial_state()]
    assert np.allclose(policy_values(game, sol.policy), root)
    assert nash_conv(game, sol.policy)["nash_conv"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("players", [2, 3])
def test_tie_breaks_select_equilibria(players: int) -> None:
    """Every tie-break rule still gives an equilibrium; with two players they all agree."""
    game = Spendthrift(players=players, lots=3, ends=1, coins=4)
    root = game.new_initial_state()
    sols = [
        solve(game, tie_key=lambda v, i=i, s=s: s * v[i]) for i in range(players) for s in (1, -1)
    ]
    for sol in sols:
        assert nash_conv(game, sol.policy)["nash_conv"] == pytest.approx(0.0, abs=1e-9)
    if players == 2:
        for sol in sols:
            assert np.allclose(sol.value(root), solve(game).value(root))


def test_uniform_play_is_exploitable() -> None:
    game = Spendthrift(players=2, lots=3, ends=1, coins=4)
    on_policy = policy_values(game, uniform)
    assert best_response_value(game, uniform, 0) > on_policy[0] + 0.1
    assert nash_conv(game, uniform)["nash_conv"] > 0.2
