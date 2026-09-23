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
