from iirl.core.types import PlayerId
from toy_game import BET_CALL, CHECK_FOLD, ToyShowdown, ToyState


def _state(c0: int, c1: int, history: tuple[int, ...]) -> ToyState:
    return ToyState(cards=(c0, c1), history=history)


def test_deal_is_two_chance_nodes() -> None:
    game = ToyShowdown()
    s0 = game.new_initial_state()
    assert game.is_chance_node(s0)
    assert len(game.chance_outcomes(s0)) == 3
    s1 = game.apply_action(s0, 2)  # deal card 2 to player 0
    assert game.is_chance_node(s1)
    assert [a for a, _ in game.chance_outcomes(s1)] == [0, 1]  # remaining deck
    s2 = game.apply_action(s1, 0)
    assert not game.is_chance_node(s2)
    assert game.current_player(s2) == 0


def test_fold_and_showdown_payoffs() -> None:
    game = ToyShowdown()
    # Player 0 bets, player 1 folds -> player 0 wins 1 regardless of cards.
    assert game.returns(_state(0, 2, (BET_CALL, CHECK_FOLD))) == [1.0, -1.0]
    # Check-check showdown -> higher card wins 1.
    assert game.returns(_state(2, 0, (CHECK_FOLD, CHECK_FOLD))) == [1.0, -1.0]
    assert game.returns(_state(0, 2, (CHECK_FOLD, CHECK_FOLD))) == [-1.0, 1.0]
    # Bet-call showdown -> higher card wins 2.
    assert game.returns(_state(2, 1, (BET_CALL, BET_CALL))) == [2.0, -2.0]
    # Check-bet-call showdown -> higher card wins 2.
    assert game.returns(_state(1, 2, (CHECK_FOLD, BET_CALL, BET_CALL))) == [-2.0, 2.0]


def test_terminal_and_turn_order() -> None:
    game = ToyShowdown()
    assert not game.is_terminal(_state(0, 1, (CHECK_FOLD,)))
    assert game.current_player(_state(0, 1, (CHECK_FOLD,))) == 1
    assert game.current_player(_state(0, 1, (CHECK_FOLD, BET_CALL))) == 0  # p0 faces fold/call
    assert game.is_terminal(_state(0, 1, (BET_CALL, CHECK_FOLD)))
    assert game.current_player(_state(0, 1, (BET_CALL, CHECK_FOLD))) == PlayerId.TERMINAL


def test_info_state_hides_opponent_card() -> None:
    game = ToyShowdown()
    s = _state(2, 0, (CHECK_FOLD,))
    # Same infoset key for player 0 no matter what the opponent holds.
    assert game.information_state_string(s, 0) == game.information_state_string(
        _state(2, 1, (CHECK_FOLD,)), 0
    )
    t = game.information_state_tensor(s, 0)
    assert t.shape == (9,)
    assert t[2] == 1.0  # own card one-hot
    assert t[:3].sum() == 1.0
