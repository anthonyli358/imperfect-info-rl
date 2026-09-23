import numpy as np

from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.games.ventures.cards import (
    BONUS_POINTS,
    DECK_SIZE,
    NUM_CARD_TYPES,
    START_COST,
    card_id,
    initial_deck_counts,
)
from iirl.games.ventures.game import (
    DISCARD_OFFSET,
    DRAW_DECK_ACTION,
    TAKE_DISCARD_OFFSET,
    Ventures,
)
from iirl.games.ventures.state import EMPTY_ROUTE, Phase, VenturesState


def _play_state(**changes) -> VenturesState:
    """A minimal mid-game state at a PLAY node, overridable per test."""
    empty_hand = (0,) * 50
    empty_routes = (EMPTY_ROUTE,) * 5
    base = VenturesState(
        deck=(0,) * 50,
        hands=(empty_hand, empty_hand),
        routes=(empty_routes, empty_routes),
        discards=((),) * 5,
        to_move=0,
        phase=Phase.PLAY,
        just_discarded_suit=-1,
        terminal=False,
    )
    from dataclasses import replace

    return replace(base, **changes)


def test_registered() -> None:
    assert isinstance(make("ventures"), Ventures)


def test_ascending_and_backing_before_number_enforced() -> None:
    g = Ventures()
    # Suit 0 route holds the number 5; hand has 4 (too low), 7 (ok) of suit 0, and a backing.
    hand = list((0,) * 50)
    hand[card_id(0, 4)] = 1
    hand[card_id(0, 7)] = 1
    hand[card_id(0, 0)] = 1  # backing of suit 0
    routes = ((tuple([card_id(0, 5)]),) + (EMPTY_ROUTE,) * 4, (EMPTY_ROUTE,) * 5)
    s = _play_state(hands=(tuple(hand), (0,) * 50), routes=routes)
    legal = set(g.legal_actions(s))
    assert card_id(0, 7) in legal  # 7 > 5, playable
    assert card_id(0, 4) not in legal  # 4 < 5, not playable
    assert card_id(0, 0) not in legal  # backing after a number is illegal
    # All three can always be discarded.
    for r in (4, 7, 0):
        assert DISCARD_OFFSET + card_id(0, r) in legal


def test_cannot_retrieve_just_discarded_pile() -> None:
    g = Ventures()
    s = _play_state(
        phase=Phase.DRAW,
        discards=((card_id(0, 9),), (card_id(1, 8),), (), (), ()),
        just_discarded_suit=0,
    )
    legal = set(g.legal_actions(s))
    assert TAKE_DISCARD_OFFSET + 0 not in legal  # just discarded to suit 0 this turn
    assert TAKE_DISCARD_OFFSET + 1 in legal  # suit 1 pile is fair game


def test_scoring_penalty_multiplier_and_bonus() -> None:
    g = Ventures()
    # Player 0: suit 0 route = backing + numbers 2,3,4,5,6,7,10 (7 numbers, 1 backing => 8 cards).
    route0 = tuple(sorted([card_id(0, 0)] + [card_id(0, n) for n in (2, 3, 4, 5, 6, 7, 10)]))
    routes = ((route0,) + (EMPTY_ROUTE,) * 4, (EMPTY_ROUTE,) * 5)
    s = _play_state(routes=routes, terminal=True)
    num_sum = 2 + 3 + 4 + 5 + 6 + 7 + 10  # 37
    expected0 = (num_sum - START_COST) * (
        1 + 1
    ) + BONUS_POINTS  # ×2 for one backing, +20 bonus (8 cards)
    assert g.returns(s) == [float(expected0), float(-expected0)]


def test_unstarted_route_scores_zero_and_zero_sum() -> None:
    g = Ventures()
    s = _play_state(terminal=True)  # nobody played anything
    assert g.returns(s) == [0.0, 0.0]


def test_draw_deck_defers_to_chance_then_ends_on_last_card() -> None:
    g = Ventures()
    hand = (0,) * 50
    deck = list((0,) * 50)
    deck[card_id(2, 9)] = 1  # exactly one card left in the deck
    s = _play_state(deck=tuple(deck), hands=(hand, hand), phase=Phase.DRAW)
    # Choosing the deck moves to a chance node rather than resolving immediately.
    s2 = g.apply_action(s, DRAW_DECK_ACTION)
    assert g.is_chance_node(s2) and s2.phase == Phase.DRAW_DECK
    # Resolving the last card ends the game.
    s3 = g.apply_action(s2, card_id(2, 9))
    assert g.is_terminal(s3)
    assert sum(s3.deck) == 0


def test_backing_only_route_scores_negative_with_multiplier() -> None:
    # Three backings, no numbers: (0 − 20) × (1 + 3) = −80. Committing to backings alone is a loss.
    g = Ventures()
    route = (card_id(0, 0), card_id(0, 0), card_id(0, 0))
    routes = ((route,) + (EMPTY_ROUTE,) * 4, (EMPTY_ROUTE,) * 5)
    s = _play_state(routes=routes, terminal=True)
    assert g.returns(s) == [-80.0, 80.0]


def _total_card_counts(state: VenturesState) -> list[int]:
    """Count every card-type id across deck, both hands, both tableaus, and all discards."""
    totals = [0] * NUM_CARD_TYPES
    for c in range(NUM_CARD_TYPES):
        totals[c] += state.deck[c] + state.hands[0][c] + state.hands[1][c]
    for pp in (0, 1):
        for suit in range(5):
            for card in state.routes[pp][suit]:
                totals[card] += 1
    for suit in range(5):
        for card in state.discards[suit]:
            totals[card] += 1
    return totals


def test_cards_are_conserved_through_a_whole_game() -> None:
    # The strongest structural invariant: no card is ever duplicated or lost. The multiset of all
    # cards (deck + hands + tableaus + discards) must equal the full deck at every single state.
    g = make("ventures")
    full = list(initial_deck_counts())
    assert sum(full) == DECK_SIZE
    rng = np.random.default_rng(0)
    for _ in range(15):
        state = g.new_initial_state()
        while not g.is_terminal(state):
            assert _total_card_counts(state) == full
            if g.is_chance_node(state):
                action = _sample_chance(g, state, rng)
            else:
                action = int(rng.choice(g.legal_actions(state)))
            state = g.apply_action(state, action)
        assert _total_card_counts(state) == full


def test_random_playout_completes_and_is_zero_sum() -> None:
    g = make("ventures")
    from iirl.agents.random_agent import RandomAgent
    from iirl.core.runner import run_episode

    rng = np.random.default_rng(0)
    for _ in range(30):
        result = run_episode(g, [RandomAgent(), RandomAgent()], rng)
        assert abs(sum(result.returns)) < 1e-9
