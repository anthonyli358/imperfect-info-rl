"""Undertow: setup, movement, abilities, fights and cards, memory, winning, hidden information."""

from dataclasses import replace

import numpy as np

from conformance import check_conformance
from iirl.games.undertow import CARD_BASE, MOVE_BASE, SETUP_BASE, Phase, Undertow
from iirl.games.undertow.board import NUM_REGIONS, SMUGGLERS, WATCH
from iirl.games.undertow.game import ALL, REMOVED, TOKENS

# After ``placed``, token k is identity k on both sides.
# Smugglers: 0 Captain 5, 1 Bruiser 4, 2 Wrecker 3, 3 Fixer 3, 4 Runner 2, 5 Pilot 2, 6 Skiff 2,
# 7 Contraband 1, 8 Decoy 0. Watch: 0 Gunboat 7, 1 Harbourmaster 5, 2 Sergeant 4, 3 Cutter 3,
# 4 Diver 3, 5 Patrol 3, 6 Patrol 2, 7 Inspector 2, 8 Signal 1.
# Smuggler cards: 0-4 numbers 1-5, 5 Fog, 6 Reinforce, 7 Feint, 8 Scuttle.
# Watch cards: 0-5 numbers 1-6, 6 Lantern, 7 Reinforce, 8 Withdraw.
FOG, S_REINFORCE, FEINT, SCUTTLE = 5, 6, 7, 8
LANTERN, W_REINFORCE, WITHDRAW = 6, 7, 8


def placed(game: Undertow):  # noqa: ANN201
    s = game.new_initial_state()
    for k in range(2 * TOKENS):
        s = game.apply_action(s, SETUP_BASE + k % TOKENS)
    return s


def board(game: Undertow, smugglers: dict, watch: dict, to_move: int = WATCH, **kw):  # noqa: ANN003, ANN201
    """Only the listed tokens on the board; ``to_move`` to move."""
    pos = (
        tuple(smugglers.get(t, REMOVED) for t in range(TOKENS)),
        tuple(watch.get(t, REMOVED) for t in range(TOKENS)),
    )
    return replace(placed(game), pos=pos, to_move=to_move, phase=Phase.MOVE, **kw)


def move(token: int, region: int) -> int:
    return MOVE_BASE + token * NUM_REGIONS + region


def fight(game: Undertow, s, token: int, region: int, a_card: int, d_card: int):  # noqa: ANN001, ANN201
    s = game.apply_action(s, move(token, region))
    assert s.phase == Phase.ATTACK_CARD, s.phase
    s = game.apply_action(s, CARD_BASE + a_card)
    return game.apply_action(s, CARD_BASE + d_card)


def _with(masks: tuple, changes: dict) -> tuple:
    out = list(masks)
    for k, v in changes.items():
        out[k] = v
    return tuple(out)


def test_setup_is_a_hidden_decision_and_the_watch_moves_first() -> None:
    game = Undertow()
    s = game.new_initial_state()
    assert game.current_player(s) == SMUGGLERS and len(game.legal_actions(s)) == TOKENS
    s = placed(game)
    assert s.to_move == WATCH and s.phase in (Phase.MOVE, Phase.INSPECT)
    assert "1.0@13=1ff" in game.information_state_string(s, SMUGGLERS)  # the Watch's pieces: hidden
    assert "1.0@13=001" in game.information_state_string(s, WATCH)


def test_forward_moves_capacity_and_abilities() -> None:
    game = Undertow()
    s = board(game, {6: 4, 1: 6, 2: 6, 7: 0}, {5: 9}, to_move=SMUGGLERS)
    assert set(game.moves(s, SMUGGLERS, 6)) == {7}  # forward only; the West narrows is full
    assert set(game.moves(s, WATCH, 5)) == {6, 7}  # the Watch moves upstream (6 means a fight)
    s = board(game, {4: 1, 7: 0}, {5: 12}, to_move=SMUGGLERS)
    assert set(game.moves(s, SMUGGLERS, 4)) == {3, 4, 6, 7}  # the Runner: two to an empty region
    after = game.apply_action(s, move(4, 6))
    assert after.masks[SMUGGLERS][4] == 1 << 4  # only the Runner could do that: it is known
    c = board(game, {6: 6, 7: 0}, {3: 11})
    assert 6 in game.moves(c, WATCH, 3)  # the Cutter jumps two to attack
    d = board(game, {6: 6, 7: 0}, {4: 7})
    assert 6 in game.moves(d, WATCH, 4)  # the Diver attacks sideways
    p = board(game, {5: 3, 6: 6, 1: 6, 7: 0}, {5: 12}, to_move=SMUGGLERS)
    assert 6 in game.moves(p, SMUGGLERS, 5)  # the Pilot ignores capacity


def test_winning_and_losing() -> None:
    game = Undertow()
    port = game.apply_action(board(game, {7: 11}, {0: 13}, to_move=SMUGGLERS), move(7, 13))
    assert game.is_terminal(port) and game.returns(port) == [1.0, -1.0]  # even into a guarded Port
    raid = game.apply_action(board(game, {7: 6}, {5: 1, 6: 0, 2: 0}), move(5, 0))
    assert game.returns(raid) == [-1.0, 1.0]  # three in the Cove
    stuck = game.apply_action(board(game, {6: 13}, {5: 9}), move(5, 6))
    assert game.returns(stuck) == [-1.0, 1.0]  # the Smugglers cannot move: they lose
    sunk = fight(game, board(game, {7: 3}, {2: 6, 6: 12}), 2, 3, 0, 0)  # no sideways from a shoal
    assert sunk.pos[SMUGGLERS][7] == REMOVED and game.returns(sunk) == [-1.0, 1.0]


def test_random_defender_and_the_contrabands_escape() -> None:
    game = Undertow()
    s = game.apply_action(board(game, {6: 6, 1: 6, 7: 0}, {5: 9}), move(5, 6))
    assert s.phase == Phase.TARGET and game.chance_outcomes(s) == [(1, 0.5), (6, 0.5)]
    slip = game.apply_action(board(game, {7: 6, 6: 0}, {5: 9}), move(5, 6))
    assert slip.pos[SMUGGLERS][7] == 7 and slip.pos[WATCH][5] == 6  # slipped sideways; no fight


def test_cards() -> None:
    game = Undertow()
    base = board(game, {6: 6, 1: 4, 7: 0}, {5: 9, 2: 12})
    lost = fight(game, base, 5, 6, a_card=0, d_card=4)  # Patrol 3 + 1 against Skiff 2 + 5
    assert lost.pos[WATCH][5] == REMOVED and lost.used == (1 << 4, 1)
    fog = fight(game, base, 5, 6, a_card=5, d_card=FOG)  # Fog ignores the 6: 3 against 2
    assert fog.pos[SMUGGLERS][6] == REMOVED
    lantern = fight(game, board(game, {1: 6, 7: 0}, {2: 9}), 2, 6, a_card=LANTERN, d_card=FOG)
    assert lantern.pos[SMUGGLERS][1] == REMOVED and lantern.pos[WATCH][2] == REMOVED  # 4 vs 4
    reinforce = fight(game, replace(base, used=(1 << 4, 0)), 5, 6, a_card=1, d_card=S_REINFORCE)
    assert reinforce.pos[WATCH][5] == REMOVED  # Reinforce plays the used 5: 3 + 2 against 2 + 5
    feint = fight(game, base, 5, 6, a_card=5, d_card=FEINT)  # the Skiff retreats upstream
    assert feint.pos[SMUGGLERS][6] == 3 and feint.pos[WATCH][5] == 6
    scuttle = fight(game, base, 5, 6, a_card=5, d_card=SCUTTLE)
    assert scuttle.pos[SMUGGLERS][6] == REMOVED and scuttle.pos[WATCH][5] == REMOVED
    withdraw = fight(
        game, board(game, {1: 4, 7: 0}, {5: 6}, to_move=SMUGGLERS), 1, 6, a_card=0, d_card=WITHDRAW
    )
    assert withdraw.pos[WATCH][5] == 7 and withdraw.pos[SMUGGLERS][1] == 6


def test_piece_abilities_in_fights() -> None:
    game = Undertow()
    gun = fight(game, board(game, {0: 6, 7: 0}, {0: 9}), 0, 6, a_card=5, d_card=0)  # 7 + 0 vs 5 + 1
    assert gun.pos[WATCH][0] == 6 and gun.pos[SMUGGLERS][0] == REMOVED
    cap = fight(game, board(game, {0: 6, 7: 0}, {2: 9}), 2, 6, a_card=LANTERN, d_card=0)
    assert cap.pos[WATCH][2] == REMOVED  # 4 + 0 against the Captain's 5 + 1
    harbour = fight(game, board(game, {1: 6, 7: 0}, {1: 9}), 1, 6, a_card=0, d_card=SCUTTLE)
    assert harbour.pos[SMUGGLERS][1] == REMOVED and harbour.pos[WATCH][1] == 6  # no Scuttle
    wreck = fight(game, board(game, {2: 6, 7: 0}, {2: 9}), 2, 6, a_card=3, d_card=0)
    assert wreck.pos[SMUGGLERS][2] == REMOVED and wreck.pos[WATCH][2] == REMOVED  # takes it along
    decoy = fight(game, board(game, {8: 6, 7: 0}, {5: 9}), 5, 6, a_card=0, d_card=0)
    assert decoy.pos[SMUGGLERS][8] == REMOVED and decoy.pos[WATCH][5] == 9  # turned back
    fixer = fight(game, board(game, {3: 4, 7: 0}, {6: 6}, to_move=SMUGGLERS), 3, 6, 2, 0)
    assert fixer.used[SMUGGLERS] == 0  # the Fixer won and kept its card
    signal = fight(
        game, board(game, {1: 4, 7: 0}, {8: 6}, to_move=SMUGGLERS, used=(0, 3)), 1, 6, 0, 0
    )
    assert signal.pos[WATCH][8] == REMOVED and signal.used[WATCH] == 0


def test_memory_of_hidden_pieces() -> None:
    game = Undertow()
    s = board(game, {6: 3, 1: 6, 7: 0}, {5: 12}, to_move=SMUGGLERS)
    s = game.apply_action(s, move(6, 6))  # no fight: nothing learned
    assert s.masks[SMUGGLERS][6] == ALL
    f = fight(game, board(game, {6: 6, 7: 0}, {5: 9}), 5, 6, a_card=5, d_card=0)
    assert f.masks[WATCH][5] == 1 << 5  # the winner was revealed and is remembered
    # Two tokens sharing a region merge what the opponent knows about them (they may be shuffled).
    m = board(game, {1: 4, 6: 7, 7: 0}, {5: 12}, to_move=SMUGGLERS)
    m = replace(m, masks=(_with(m.masks[SMUGGLERS], {1: 1 << 1, 6: 1 << 6}), m.masks[WATCH]))
    m = game.apply_action(m, move(1, 7))
    assert m.masks[SMUGGLERS][1] == m.masks[SMUGGLERS][6] == (1 << 1) | (1 << 6)


def test_hidden_card_and_resampling() -> None:
    game = Undertow()
    s = game.apply_action(board(game, {6: 6, 7: 0}, {5: 9}), move(5, 6))
    s = game.apply_action(s, CARD_BASE + 3)
    assert game.information_state_string(s, SMUGGLERS).endswith(":x")
    assert game.information_state_string(s, WATCH).endswith(":3")
    rng = np.random.default_rng(0)
    early = game.apply_action(game.new_initial_state(), SETUP_BASE + 4)
    for player in (SMUGGLERS, WATCH):
        for state in (s, early, placed(game)):
            for _ in range(5):
                w = game.resample_from_infostate(state, player, rng)
                assert game.information_state_string(w, player) == game.information_state_string(
                    state, player
                )
    assert len(game.information_state_tensor(s, WATCH)) == 551


def test_conformance() -> None:
    check_conformance(Undertow(), np.random.default_rng(0), episodes=200)
