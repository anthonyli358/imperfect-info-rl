import json

import numpy as np
import pytest

import iirl.games  # noqa: F401  (registers the built-in games)
from iirl.core.registry import make
from iirl.play import Session


@pytest.mark.parametrize("game_name", ["ventures", "camel_stocks"])
def test_session_plays_full_games_without_leaking_hidden_cards(game_name: str) -> None:
    rng = np.random.default_rng(0)
    session = Session()
    for seat in ("0", "1"):
        view = session.new(game_name, "heuristic", "Heuristic", seat)
        assert view["seat"] == int(seat)
        opp = 1 - int(seat)
        steps = 0
        while not view["terminal"]:
            assert view["your_turn"] and view["legal"]
            # The opponent's hand holds only publicly known cards; the rest is a count.
            s = session.state
            assert len(view["view"]["hands"][opp]) + view["view"]["hidden"][opp] == sum(
                s.hands[opp]
            )
            assert len(view["view"]["hands"][opp]) == sum(s.known[opp])
            json.dumps(view)  # everything sent to the page is JSON-serializable
            view = session.act(int(rng.choice([a["id"] for a in view["legal"]])))
            assert "error" not in view
            steps += 1
            assert steps < 1000
        full = make(game_name).to_display(session.state)
        assert view["view"]["hands"] == full["hands"]  # revealed at the end
        assert view["margin"] == make(game_name).returns(session.state)[int(seat)]
    assert session.records[game_name]["games"] == 2


def test_session_rejects_illegal_moves() -> None:
    session = Session()
    view = session.new("ventures", "heuristic", "Heuristic", "0")
    assert "error" in session.act(-1)
    assert session.view()["view"]["hands"] == view["view"]["hands"]  # nothing changed


def test_player_view_hides_the_opponent_hand() -> None:
    for name in ("ventures", "camel_stocks"):
        g = make(name)
        rng = np.random.default_rng(1)
        s = g.new_initial_state()
        while g.is_chance_node(s):
            outcomes = g.chance_outcomes(s)
            s = g.apply_action(
                s, outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            )
        view = g.player_view(s, 0)
        assert view["hands"][1] == []  # nothing about the opponent's hand is known yet
        assert view["hidden"][1] == sum(s.hands[1])
        assert view["hands"][0] == g.to_display(s)["hands"][0]


def test_finished_games_are_recorded_and_replay_exactly(tmp_path) -> None:  # noqa: ANN001
    rng = np.random.default_rng(2)
    session = Session(record_dir=tmp_path)
    view = session.new("camel_stocks", "heuristic", "Heuristic", "0")
    while not view["terminal"]:
        view = session.act(int(rng.choice([a["id"] for a in view["legal"]])))
    files = list((tmp_path / "camel_stocks").glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text("utf-8"))
    assert record["human_seat"] == 0 and record["margin"] == view["margin"]
    g = make("camel_stocks")
    s = g.new_initial_state()
    for step in record["actions"]:
        s = g.apply_action(s, step["action"])
    assert g.is_terminal(s) and g.returns(s) == record["returns"]


def test_careful_heuristic_never_gifts_when_a_safe_discard_exists() -> None:
    from iirl.core.registry import make_heuristic
    from iirl.core.runner import _sample_chance
    from iirl.games.ventures.game import DISCARD_OFFSET, _can_play
    from iirl.games.ventures.state import Phase

    g, careful = make("ventures"), make_heuristic("ventures_careful")
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        s = g.new_initial_state()
        while not g.is_terminal(s):
            if g.is_chance_node(s):
                s = g.apply_action(s, _sample_chance(g, s, rng))
                continue
            a = careful.step(g, s, rng)
            if s.phase == Phase.PLAY and a >= DISCARD_OFFSET:
                opp = s.routes[1 - s.to_move]

                def gift(c: int, s=s, opp=opp) -> bool:
                    r = opp[c // 10]
                    return bool(r) and _can_play(r, c)

                held = [c for c, n in enumerate(s.hands[s.to_move]) if n]
                if any(not gift(c) for c in held):
                    assert not gift(a - DISCARD_OFFSET)
                    checked += 1
            s = g.apply_action(s, a)
    assert checked > 50


def test_session_plays_full_tessera_games() -> None:
    rng = np.random.default_rng(0)
    session = Session()
    for seat in ("0", "1"):
        view = session.new("tessera", "heuristic", "Heuristic", seat)
        steps = 0
        while not view["terminal"]:
            assert view["your_turn"] and view["legal"]
            json.dumps(view)
            view = session.act(int(rng.choice([a["id"] for a in view["legal"]])))
            assert "error" not in view
            steps += 1
            assert steps < 1000
        assert view["margin"] == make("tessera").returns(session.state)[int(seat)]
    assert session.records["tessera"]["games"] == 2
