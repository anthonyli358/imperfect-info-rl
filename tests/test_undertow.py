"""Undertow: setup, movement, fights and their special cases, winning, hidden information."""

from dataclasses import replace

import numpy as np

from conformance import check_conformance
from iirl.games.undertow import CARD_BASE, MOVE_BASE, PASS, Phase, Undertow
from iirl.games.undertow.board import FEINT, LANTERN, NUM_REGIONS, SMUGGLERS, WATCH
from iirl.games.undertow.game import FULL_HAND, REMOVED, TOKENS

# After ``placed``, token k is roster identity k on both sides:
# Smugglers 0 Decoy, 1 Runner, 2 Pilot, 3-6 Cargo (strength 1/1/2/2, cargo 3/2/2/1), 7 Bruiser 4,
# 8 Fixer 3. Watch 0 Harbourmaster 5, 1 Sergeant 4, 2 Cutter 3, 3-4 Patrol 3, 5-6 Patrol 2,
# 7 Inspector 1, 8 Signal 1.


def placed(game: Undertow):  # noqa: ANN201
    s = game.new_initial_state()
    for k in range(2 * TOKENS):
        s = game.apply_action(s, k % TOKENS)
    return s


def board(game: Undertow, smugglers: dict, watch: dict, **kw):  # noqa: ANN003, ANN201
    """Only the listed tokens on the board, Smugglers to move.

    Unlisted Cargo boats wait in the Cove (so 6 stays within reach); other unlisted pieces are off.
    """
    s = placed(game)
    cargo = {t: 0 for t in range(3, 7)}
    pos = (
        tuple({**cargo, **smugglers}.get(t, REMOVED) for t in range(TOKENS)),
        tuple(watch.get(t, REMOVED) for t in range(TOKENS)),
    )
    return replace(s, pos=pos, **kw)


def move(token: int, region: int) -> int:
    return MOVE_BASE + token * NUM_REGIONS + region


def fight(game: Undertow, s, token: int, region: int, a_card: int, d_card: int):  # noqa: ANN001, ANN201
    s = game.apply_action(s, move(token, region))
    assert s.phase == Phase.ATTACK_CARD
    s = game.apply_action(s, CARD_BASE + a_card)
    assert s.phase == Phase.DEFEND_CARD
    return game.apply_action(s, CARD_BASE + d_card)


def test_setup_shuffles_every_identity() -> None:
    game = Undertow()
    rng = np.random.default_rng(0)
    s = game.new_initial_state()
    while game.is_chance_node(s):
        outcomes = game.chance_outcomes(s)
        s = game.apply_action(s, outcomes[rng.integers(len(outcomes))][0])
    assert s.phase == Phase.MOVE and s.to_move == SMUGGLERS
    for side in (SMUGGLERS, WATCH):
        assert sorted(s.ident[side]) == list(range(TOKENS))


def test_movement_rules() -> None:
    game = Undertow()
    s = placed(game)
    # Smugglers never move upstream; the reaches are full, so only the Runner (two steps, through
    # a friendly region) and the Pilot (ignores capacity) can leave the Cove.
    assert game.destinations(s, SMUGGLERS, 7) == [4, 5]  # Bruiser in the East reach
    assert game.destinations(s, SMUGGLERS, 0) == []  # Decoy in the Cove
    assert game.destinations(s, SMUGGLERS, 1) == [3, 4, 5]  # Runner: two steps
    assert game.destinations(s, SMUGGLERS, 2) == [1, 2]  # Pilot
    # The Harbourmaster stays in the port channels.
    h = board(game, {3: 0}, {0: 8})
    assert game.destinations(h, WATCH, 0) == [9, 11]  # not the West narrows (6)
    # The Watch may move upstream.
    w = board(game, {3: 0}, {3: 8})
    assert 6 in game.destinations(w, WATCH, 3)


def test_fight_strength_plus_card() -> None:
    game = Undertow()
    s = board(game, {7: 6, 3: 0}, {5: 8})  # Bruiser 4 attacks Patrol 2
    lost = fight(game, s, 7, 8, a_card=0, d_card=3)  # 4 + 0 against 2 + 3
    assert lost.pos[SMUGGLERS][7] == REMOVED and lost.pos[WATCH][5] == 8
    assert lost.revealed == (1 << 7, 1 << 5)  # both fighters revealed
    assert lost.hands == (FULL_HAND & ~1, FULL_HAND & ~(1 << 3)) and lost.to_move == WATCH
    won = fight(game, s, 7, 8, a_card=3, d_card=0)  # 7 against 2: the Bruiser moves in
    assert won.pos[WATCH][5] == REMOVED and won.pos[SMUGGLERS][7] == 8
    lantern = fight(game, s, 7, 8, a_card=0, d_card=LANTERN)  # 4 against 2
    assert lantern.pos[WATCH][5] == REMOVED
    cancelled = fight(game, s, 7, 8, a_card=LANTERN, d_card=3)  # the 3 is cancelled: 4 against 2
    assert cancelled.pos[WATCH][5] == REMOVED
    tie = fight(game, board(game, {8: 6, 3: 0}, {3: 8}), 8, 8, a_card=1, d_card=1)  # 4 against 4
    assert tie.pos[SMUGGLERS][8] == REMOVED and tie.pos[WATCH][3] == REMOVED


def test_feints() -> None:
    game = Undertow()
    s = board(game, {7: 6, 3: 0}, {5: 8})
    stay = fight(game, s, 7, 8, a_card=FEINT, d_card=3)
    assert stay.pos[SMUGGLERS][7] == 6 and stay.pos[WATCH][5] == 8  # nobody lost
    away = fight(game, s, 7, 8, a_card=0, d_card=FEINT)  # the Patrol slips away; the Bruiser enters
    assert away.pos[WATCH][5] == 9 and away.pos[SMUGGLERS][7] == 8
    blocked = fight(game, s, 7, 8, a_card=LANTERN, d_card=FEINT)  # cancelled: 4 against 2
    assert blocked.pos[WATCH][5] == REMOVED


def test_special_pieces() -> None:
    game = Undertow()
    # A Decoy that loses as defender turns the attacker back.
    d = replace(board(game, {0: 7, 3: 0}, {3: 9}), to_move=WATCH)
    d = fight(game, d, 3, 7, a_card=1, d_card=0)
    assert d.pos[SMUGGLERS][0] == REMOVED and d.pos[WATCH][3] == 9
    # A Fixer that wins keeps its card.
    f = fight(game, board(game, {8: 6, 3: 0}, {5: 8}), 8, 8, a_card=3, d_card=0)
    assert f.hands[SMUGGLERS] == FULL_HAND
    # Losing the Signal gives the Watch back its played cards.
    sig = board(game, {7: 6, 3: 0}, {8: 8}, hands=(FULL_HAND, FULL_HAND & ~(1 << 2)))
    sig = fight(game, sig, 7, 8, a_card=0, d_card=0)
    assert sig.pos[WATCH][8] == REMOVED and sig.hands[WATCH] == FULL_HAND
    # An attacker that wins fights the next defender (the Watch picks who).
    two = board(game, {7: 6, 3: 0}, {5: 9, 6: 9})
    two = game.apply_action(two, move(7, 9))
    assert two.phase == Phase.PICK and game.current_player(two) == WATCH


def test_delivery_sinking_and_the_tide() -> None:
    game = Undertow()
    s = board(game, {3: 11, 4: 11, 5: 11}, {})
    s = game.apply_action(s, move(3, 12))  # cargo worth 3 delivered
    assert s.delivered == 3 and not game.is_terminal(s)
    s = game.apply_action(s, PASS)  # the Watch has nothing on the board
    s = game.apply_action(s, move(4, 12))
    assert s.delivered == 5 and not game.is_terminal(s)
    win = game.apply_action(board(game, {3: 11}, {}, delivered=3), move(3, 12))
    assert game.is_terminal(win) and game.returns(win) == [1.0, -1.0]
    # Sinking cargo worth 3 puts 6 out of reach: the Watch wins.
    sink = fight(game, board(game, {3: 6, 7: 0}, {1: 8}), 3, 8, a_card=0, d_card=0)
    assert sink.sunk == 3 and game.returns(sink) == [-1.0, 1.0]
    # The tide: the Watch wins when its 24th turn ends.
    tide = replace(board(game, {7: 0}, {1: 12}), to_move=WATCH, round=23)
    tide = game.apply_action(tide, game.legal_actions(tide)[0])
    assert game.is_terminal(tide) and game.returns(tide) == [-1.0, 1.0]


def test_hidden_information_and_the_inspector() -> None:
    game = Undertow()
    s = board(game, {7: 6, 3: 0}, {5: 8, 7: 11})
    smug, watch = (game.information_state_string(s, p) for p in (SMUGGLERS, WATCH))
    assert "1.5@8=?" in smug and "1.5@8=5" in watch  # identities are private
    assert "0.7@6=?" in watch
    a = game.apply_action(game.apply_action(s, move(7, 8)), CARD_BASE + 2)
    assert game.information_state_string(a, WATCH).endswith(":x")  # the attack card is hidden
    assert game.information_state_string(a, SMUGGLERS).endswith(":2")
    # A Smuggler next to the Inspector (in the Mouth) at the start of the Watch turn is seen.
    i = board(game, {3: 9, 7: 0}, {7: 11})
    i = game.apply_action(i, move(7, 1))
    assert i.phase == Phase.INSPECT and game.chance_outcomes(i) == [(3, 1.0)]
    i = game.apply_action(i, 3)
    assert "0.3@9=3" in game.information_state_string(i, WATCH)
    assert i.known == 1 << 3 and not i.revealed[SMUGGLERS] >> 3 & 1  # seen by the Watch only
    assert len(game.information_state_tensor(i, SMUGGLERS)) == 560


def test_resample_keeps_what_the_player_knows() -> None:
    game = Undertow()
    s = game.apply_action(placed(game), move(7, 4))
    rng = np.random.default_rng(0)
    for player in (SMUGGLERS, WATCH):
        for _ in range(10):
            w = game.resample_from_infostate(s, player, rng)
            assert game.information_state_string(w, player) == game.information_state_string(
                s, player
            )


def test_conformance() -> None:
    check_conformance(Undertow(), np.random.default_rng(0), episodes=200)
