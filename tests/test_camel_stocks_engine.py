from dataclasses import replace

import numpy as np

import iirl.games  # noqa: F401  (registers "camel_stocks")
from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.games.camel_stocks.cards import CAMEL, CARD_COUNTS, HAND_LIMIT, MARKET_SIZE, NUM_GOODS
from iirl.games.camel_stocks.game import (
    EX_DONE,
    EX_GIVE_OFFSET,
    EX_TAKE_OFFSET,
    TAKE_CAMELS,
    sell_action,
)
from iirl.games.camel_stocks.state import Phase

G = make("camel_stocks")


def _dealt(seed: int = 0):
    rng = np.random.default_rng(seed)
    s = G.new_initial_state()
    while G.is_chance_node(s):
        s = G.apply_action(s, _sample_chance(G, s, rng))
    return s


def _position(**changes):
    """A dealt game with hands, herds and market overridden (deck adjusted to conserve cards)."""
    s = replace(_dealt(), **changes)
    deck = [
        CARD_COUNTS[c]
        - s.market[c]
        - (s.herds[0] + s.herds[1] if c == CAMEL else s.hands[0][c] + s.hands[1][c] + s.sold[c])
        for c in range(len(CARD_COUNTS))
    ]
    assert min(deck) >= 0
    return replace(s, deck=tuple(deck))


def _cards_in_play(s) -> list[int]:
    return [
        s.deck[c]
        + s.market[c]
        + (s.herds[0] + s.herds[1] if c == CAMEL else s.hands[0][c] + s.hands[1][c] + s.sold[c])
        for c in range(len(CARD_COUNTS))
    ]


def test_registered_and_deal() -> None:
    s = _dealt(1)
    assert s.phase == Phase.PLAY and s.to_move == 0
    assert sum(s.market) == MARKET_SIZE and s.market[CAMEL] >= 3
    for p in (0, 1):
        assert sum(s.hands[p]) + s.herds[p] == 5
    assert _cards_in_play(s) == list(CARD_COUNTS)


def test_random_games_conserve_cards_and_respect_the_hand_limit() -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        s = G.new_initial_state()
        while not G.is_terminal(s):
            if G.is_chance_node(s):
                s = G.apply_action(s, _sample_chance(G, s, rng))
                continue
            assert _cards_in_play(s) == list(CARD_COUNTS)
            assert all(sum(h) <= HAND_LIMIT for h in s.hands)
            if s.phase == Phase.PLAY:
                assert sum(s.market) == MARKET_SIZE
            s = G.apply_action(s, int(rng.choice(G.legal_actions(s))))
        assert _cards_in_play(s) == list(CARD_COUNTS)


def test_precious_goods_sell_only_in_pairs_and_tokens_pay_top_first() -> None:
    s = _position(hands=((1, 0, 0, 3, 0, 0), (0,) * NUM_GOODS))
    legal = G.legal_actions(s)
    assert sell_action(0, 1) not in legal  # a single gem cannot be sold
    assert sell_action(3, 2) in legal and sell_action(3, 3) in legal
    after = G.apply_action(s, sell_action(3, 2))
    assert after.points[0] == 5 + 3  # silk tokens 5, 3, 3, ...
    assert after.tokens[3] == (3, 2, 2, 1, 1)
    assert after.phase == Phase.PLAY and after.to_move == 1  # no bonus for 2 cards


def test_selling_three_draws_a_bonus_token() -> None:
    s = _position(hands=((0, 0, 0, 3, 0, 0), (0,) * NUM_GOODS))
    after = G.apply_action(s, sell_action(3, 3))
    assert after.phase == Phase.BONUS and after.bonus_tier == 0
    values = [v for v, _ in G.chance_outcomes(after)]
    assert values == [1, 2, 3]
    done = G.apply_action(after, 3)
    assert done.points[0] == 5 + 3 + 3 + 3 and len(done.bonus[0]) == 6


def test_taking_camels_takes_all_and_refills() -> None:
    s = _position(market=(1, 0, 0, 1, 0, 0, 3), herds=(0, 0))
    after = G.apply_action(s, TAKE_CAMELS)
    assert after.herds[0] == 3 and after.market[CAMEL] == 0
    assert after.phase == Phase.REFILL and after.refill == 3


def test_hand_limit_blocks_taking_one_good() -> None:
    s = _position(hands=((0, 0, 0, 3, 2, 2), (0,) * NUM_GOODS), market=(1, 1, 0, 0, 0, 0, 3))
    assert sum(s.hands[0]) == HAND_LIMIT
    assert not any(a < TAKE_CAMELS for a in G.legal_actions(s))


def test_exchange_swaps_camels_for_goods() -> None:
    s = _position(
        hands=((0, 0, 0, 0, 0, 1), (0,) * NUM_GOODS),
        herds=(2, 0),
        market=(1, 1, 0, 0, 0, 0, 3),
    )
    s = G.apply_action(s, EX_TAKE_OFFSET + 0)
    assert s.phase == Phase.EX_TAKE and EX_DONE not in G.legal_actions(s)  # at least 2 goods
    s = G.apply_action(G.apply_action(s, EX_TAKE_OFFSET + 1), EX_DONE)
    assert s.phase == Phase.EX_GIVE
    legal = G.legal_actions(s)
    assert EX_GIVE_OFFSET + 0 not in legal  # never give back a good being taken
    s = G.apply_action(G.apply_action(s, EX_GIVE_OFFSET + CAMEL), EX_GIVE_OFFSET + CAMEL)
    assert s.phase == Phase.PLAY and s.to_move == 1
    assert s.hands[0] == (1, 1, 0, 0, 0, 1) and s.herds[0] == 0
    assert s.market == (0, 0, 0, 0, 0, 0, 5)
    assert s.known[0] == (1, 1, 0, 0, 0, 0)  # the opponent saw both goods come off the market


def test_exchange_respects_the_hand_limit() -> None:
    # 6 goods in hand, taking 2: at least 1 of the 2 given back must be a good.
    s = _position(
        hands=((0, 0, 0, 0, 3, 3), (0,) * NUM_GOODS),
        herds=(3, 0),
        market=(1, 1, 0, 0, 0, 0, 3),
    )
    s = G.apply_action(
        G.apply_action(G.apply_action(s, EX_TAKE_OFFSET), EX_TAKE_OFFSET + 1), EX_DONE
    )
    s = G.apply_action(s, EX_GIVE_OFFSET + CAMEL)
    assert EX_GIVE_OFFSET + CAMEL not in G.legal_actions(s)  # a second camel would leave 8 goods
    s = G.apply_action(s, EX_GIVE_OFFSET + 5)
    assert sum(s.hands[0]) == HAND_LIMIT


def test_game_ends_when_three_token_piles_are_empty_and_camels_score() -> None:
    s = _position(hands=((0, 0, 0, 0, 0, 2), (0,) * NUM_GOODS), herds=(4, 1))
    s = replace(s, tokens=((), (), (6, 5), (5,), (5,), (1, 1)))
    s = G.apply_action(s, sell_action(5, 2))
    assert G.is_terminal(s)
    assert G.returns(s) == [2.0 + 5.0, -7.0]


def test_resample_keeps_the_view_and_puts_no_camels_in_hands() -> None:
    rng = np.random.default_rng(3)
    for seed in range(20):
        s = _dealt(seed)
        for _ in range(int(rng.integers(0, 30))):
            if G.is_terminal(s):
                break
            a = _sample_chance(G, s, rng) if G.is_chance_node(s) else rng.choice(G.legal_actions(s))
            s = G.apply_action(s, int(a))
        if G.is_terminal(s) or G.is_chance_node(s):
            continue
        me = G.current_player(s)
        w = G.resample_from_infostate(s, me, rng)
        assert G.information_state_string(w, me) == G.information_state_string(s, me)
        assert _cards_in_play(w) == list(CARD_COUNTS)
        assert sum(w.hands[1 - me]) == sum(s.hands[1 - me])
        assert all(w.hands[1 - me][g] >= s.known[1 - me][g] for g in range(NUM_GOODS))
        assert G.legal_actions(w) == G.legal_actions(s)
