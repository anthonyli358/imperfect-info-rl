"""Heads-up limit Hold'em: the hand evaluator, the betting rules, payoffs, conformance."""

import itertools

import numpy as np
import pytest

from conformance import check_conformance
from iirl.games.holdem import CALL, FOLD, RAISE, LimitHoldem, category, hand_value, parse
from iirl.games.holdem.cards import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
)


def v(text: str) -> int:
    return hand_value(parse(text))


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        ("As Ks Qs Js Ts 2d 3c", STRAIGHT_FLUSH),
        ("5h 4h 3h 2h Ah Kd Kc", STRAIGHT_FLUSH),  # the steel wheel
        ("9c 9d 9h 9s 2c 3d 4h", QUADS),
        ("Kc Kd Kh 2s 2c 7d 8h", FULL_HOUSE),
        ("Kc Kd Kh 2s 2c 2d 8h", FULL_HOUSE),  # two sets of trips
        ("Ac 9c 7c 4c 2c Kd Kh", FLUSH),
        ("5d 4c 3h 2s Ad Kc Qh", STRAIGHT),  # the wheel
        ("Tc Jd Qh Ks Ad 2c 3c", STRAIGHT),
        ("7c 7d 7h Ks 2d 4c 9h", TRIPS),
        ("7c 7d 4h 4s 2d 2c 9h", TWO_PAIR),  # three pairs
        ("Ac Ad 4h 5s 9d Jc Kh", PAIR),
        ("Ac Qd 4h 5s 9d Jc 2h", HIGH_CARD),
    ],
)
def test_categories(cards: str, expected: int) -> None:
    assert category(parse(cards)) == expected


def test_category_order_and_kickers() -> None:
    ladder = [
        "Ac Qd 9h 5s 3d",  # high card
        "2c 2d 4h 5s 9d",  # pair
        "2c 2d 3h 3s 4d",  # two pair
        "2c 2d 2h 3s 4d",  # trips
        "Ac 2d 3h 4s 5d",  # wheel
        "2c 4c 6c 8c Tc",  # flush
        "2c 2d 2h 3s 3d",  # full house
        "2c 2d 2h 2s 3d",  # quads
        "Ac 2c 3c 4c 5c",  # steel wheel
    ]
    values = [v(h) for h in ladder]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert v("Ac Ad Kh 5s 3d") > v("Ac Ad Qh Js Td")  # kicker
    assert v("6c 5d 4h 3s 2d") > v("5c 4d 3h 2s Ad")  # six-high straight beats the wheel
    assert v("Kc Kd 2h 2s 9d") > v("Qc Qd Jh Js Ad")  # two pair: top pair first
    assert v("As Ks Qs Js 9s 8d 7d") == v("As Ks Qs Js 9s")  # best five of seven
    assert v("2c 3d Ah Kh Qh Jh Th") == v("4c 5d Ah Kh Qh Jh Th")  # the board plays


def test_evaluator_is_consistent_with_brute_force_on_random_hands() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        cards = tuple(int(c) for c in rng.choice(52, size=7, replace=False))
        best = max(hand_value(five) for five in itertools.combinations(cards, 5))
        assert hand_value(cards) == best


def deal(game: LimitHoldem, text: str):  # noqa: ANN201
    s = game.new_initial_state()
    for c in parse(text):
        s = game.apply_action(s, c)
    return s


def test_preflop_fold_and_the_big_blinds_option() -> None:
    game = LimitHoldem()
    s = deal(game, "As Ad Kc Kd")
    assert game.current_player(s) == 0 and game.legal_actions(s) == [FOLD, CALL, RAISE]
    folded = game.apply_action(s, FOLD)
    assert game.is_terminal(folded) and game.returns(folded) == [-0.5, 0.5]
    limped = game.apply_action(s, CALL)  # the small blind completes; the big blind may still raise
    assert not game.is_chance_node(limped) and game.current_player(limped) == 1
    assert game.legal_actions(limped) == [CALL, RAISE]  # no fold when not facing a bet
    checked = game.apply_action(limped, CALL)
    assert game.is_chance_node(checked) and checked.round == 1 and checked.contrib == (2, 2)


def test_raise_cap_and_bet_sizes() -> None:
    game = LimitHoldem()
    s = deal(game, "As Ad Kc Kd")
    for _ in range(3):  # three raises after the big blind: four bets
        s = game.apply_action(s, RAISE)
    assert s.contrib == (8, 6) and game.legal_actions(s) == [FOLD, CALL]
    s = game.apply_action(s, CALL)
    assert s.round == 1 and s.contrib == (8, 8)
    s = deal_more(game, s, "2c 7h 9s")
    assert game.current_player(s) == 1  # the big blind acts first after the flop
    s = game.apply_action(s, RAISE)  # bet 2
    s = game.apply_action(s, CALL)
    s = deal_more(game, s, "Jd")
    s = game.apply_action(s, RAISE)  # bet 4 on the turn
    assert s.contrib == (10, 14)


def deal_more(game: LimitHoldem, s, text: str):  # noqa: ANN001, ANN201
    for c in parse(text):
        s = game.apply_action(s, c)
    return s


def test_showdown_pays_the_pot_and_ties_split() -> None:
    game = LimitHoldem()
    s = deal(game, "As Ad Kc Kd")
    s = game.apply_action(game.apply_action(s, CALL), CALL)  # 2 chips each
    for street in ("2c 7h 9s", "Jd", "3h"):
        s = deal_more(game, s, street)
        s = game.apply_action(game.apply_action(s, CALL), CALL)  # check, check
    assert game.is_terminal(s) and game.returns(s) == [1.0, -1.0]  # aces win one big blind
    tie = deal(game, "2c 3d 2h 3s")
    tie = game.apply_action(game.apply_action(tie, CALL), CALL)
    for street in ("Ah Kh Qh", "Jh", "Th"):  # a royal flush on the board
        tie = deal_more(game, tie, street)
        tie = game.apply_action(game.apply_action(tie, CALL), CALL)
    assert game.returns(tie) == [0.0, 0.0]


def test_resample_keeps_what_the_player_sees() -> None:
    game = LimitHoldem()
    s = deal_more(
        game, game.apply_action(game.apply_action(deal(game, "As Ad Kc Kd"), 1), 1), "2c 7h 9s"
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = game.resample_from_infostate(s, 0, rng)
        assert game.information_state_string(w, 0) == game.information_state_string(s, 0)
        assert len(set(w.cards)) == len(w.cards)


def test_conformance() -> None:
    check_conformance(LimitHoldem(), np.random.default_rng(0), episodes=300)
