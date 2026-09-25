import numpy as np

from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.games.ventures.cards import DECK_SIZE, initial_deck_counts


def _advance_to_player_node(game, rng):
    s = game.new_initial_state()
    while game.is_chance_node(s) and not game.is_terminal(s):
        s = game.apply_action(s, _sample_chance(game, s, rng))
    return s


def test_resample_preserves_view_and_conserves_cards() -> None:
    g = make("ventures")
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = _advance_to_player_node(g, rng)
        player = g.current_player(s)
        w = g.resample_from_infostate(s, player, rng)
        # The player's own hand, all public info, and deck size are preserved exactly.
        assert w.hands[player] == s.hands[player]
        assert w.routes == s.routes
        assert w.discards == s.discards
        assert sum(w.deck) == sum(s.deck)
        assert w.to_move == s.to_move and w.phase == s.phase
        # The opponent's hand size is unchanged, and the world conserves the 60-card deck.
        opp = 1 - player
        assert sum(w.hands[opp]) == sum(s.hands[opp])
        assert sum(w.deck) + sum(w.hands[0]) + sum(w.hands[1]) == DECK_SIZE
        # The resampled world is a legal composition (no card-type exceeds its supply).
        full = initial_deck_counts()
        totals = [w.deck[c] + w.hands[0][c] + w.hands[1][c] for c in range(len(full))]
        # Cards on routes/discards are also part of the full supply; add them back.
        for pp in (0, 1):
            for suit in range(5):
                for card in w.routes[pp][suit]:
                    totals[card] += 1
        for suit in range(5):
            for card in w.discards[suit]:
                totals[card] += 1
        assert totals == list(full)


def test_resample_matches_information_state_string() -> None:
    g = make("ventures")
    rng = np.random.default_rng(1)
    s = _advance_to_player_node(g, rng)
    player = g.current_player(s)
    key = g.information_state_string(s, player)
    for _ in range(20):
        w = g.resample_from_infostate(s, player, rng)
        assert g.information_state_string(w, player) == key  # same infoset for the player


def test_known_cards_tracked_and_respected_by_resampling() -> None:
    from dataclasses import replace

    from iirl.games.ventures.cards import card_id
    from iirl.games.ventures.game import DISCARD_OFFSET, TAKE_DISCARD_OFFSET
    from iirl.games.ventures.state import Phase

    g = make("ventures")
    rng = np.random.default_rng(2)
    seven = card_id(0, 7)
    # Player 0 is at the draw step with a suit-0 7 face up on the suit-0 discard pile.
    full = list(initial_deck_counts())
    full[seven] -= 1
    hand0 = [0] * len(full)
    hand1 = [0] * len(full)
    cards = [c for c in range(len(full)) for _ in range(full[c])]
    for c in cards[:7]:
        hand0[c] += 1
    for c in cards[7:15]:
        hand1[c] += 1
    deck = list(full)
    for c in cards[:15]:
        deck[c] -= 1
    s = replace(
        g.new_initial_state(),
        deck=tuple(deck),
        hands=(tuple(hand0), tuple(hand1)),
        discards=((seven,),) + ((),) * 4,
        phase=Phase.DRAW,
        to_move=0,
    )
    s = g.apply_action(s, TAKE_DISCARD_OFFSET + 0)
    assert s.known[0][seven] == 1 and s.known[1] == (0,) * len(full)
    # Every world player 1 can imagine has that 7 in player 0's hand.
    for _ in range(20):
        assert g.resample_from_infostate(s, 1, rng).hands[0][seven] == 1
    # Once it leaves player 0's hand it is no longer known.
    s = replace(s, to_move=0, phase=Phase.PLAY)
    s = g.apply_action(s, DISCARD_OFFSET + seven)
    assert s.known[0][seven] == 0
