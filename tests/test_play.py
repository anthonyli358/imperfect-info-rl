import numpy as np

from iirl.games.ventures.game import DISCARD_OFFSET, DRAW_DECK_ACTION, TAKE_DISCARD_OFFSET
from iirl.play import Session


def _random_move(view, rng):
    legal = view["legal"]
    if view["phase"] == "PLAY":
        moves = legal["play"] + [DISCARD_OFFSET + c for c in legal["discard"]]
    else:
        moves = ([DRAW_DECK_ACTION] if legal["deck"] else []) + [
            TAKE_DISCARD_OFFSET + s for s in legal["take"]
        ]
    return int(rng.choice(moves))


def test_session_plays_full_games_without_leaking_the_opponent_hand() -> None:
    rng = np.random.default_rng(0)
    session = Session()
    for seat in ("0", "1"):
        view = session.new("heuristic", "Heuristic", seat)
        assert view["seat"] == int(seat)
        steps = 0
        while not view["terminal"]:
            assert view["your_turn"]
            assert "opp_hand" not in view  # hidden until the game ends
            assert len(view["opp_known"]) <= view["opp_hand_size"]
            view = session.act(_random_move(view, rng))
            assert "error" not in view
            steps += 1
            assert steps < 500
        assert len(view["opp_hand"]) == view["opp_hand_size"]
        assert view["margin"] == view["scores"]["you"] - view["scores"]["opp"]
    assert session.record["games"] == 2


def test_session_rejects_illegal_moves() -> None:
    session = Session()
    view = session.new("heuristic", "Heuristic", "0")
    assert "error" in session.act(-1)
    assert session.view()["hand"] == view["hand"]  # nothing changed
