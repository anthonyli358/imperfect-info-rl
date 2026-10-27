"""The OpenSpiel adapter: immutability, identity by history, and agreement with OpenSpiel."""

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

import iirl.games  # noqa: E402, F401
from iirl.core.registry import make  # noqa: E402
from iirl.core.types import PlayerId  # noqa: E402


def test_apply_action_does_not_mutate() -> None:
    game = make("liars_dice")
    s = game.new_initial_state()
    a = game.apply_action(s, 0)
    b = game.apply_action(s, 5)
    assert s.history == () and game.is_chance_node(s)
    assert a.history == (0,) and b.history == (5,)


def test_states_are_identified_by_history() -> None:
    game = make("liars_dice")
    s = game.apply_action(game.apply_action(game.new_initial_state(), 2), 4)
    t = game.apply_action(game.apply_action(game.new_initial_state(), 2), 4)
    assert s == t and hash(s) == hash(t) and len({s, t}) == 1


def test_matches_openspiel_on_random_games() -> None:
    game = make("liars_dice")
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = game.new_initial_state()
        ref = pyspiel.load_game("liars_dice(numdice=1)").new_initial_state()
        while not game.is_terminal(s):
            if game.is_chance_node(s):
                assert game.current_player(s) == PlayerId.CHANCE
                actions = [a for a, _ in game.chance_outcomes(s)]
            else:
                assert game.current_player(s) == ref.current_player()
                assert game.legal_actions(s) == list(ref.legal_actions())
                p = game.current_player(s)
                assert game.information_state_string(s, p) == ref.information_state_string(p)
                actions = game.legal_actions(s)
            a = int(rng.choice(actions))
            s = game.apply_action(s, a)
            ref.apply_action(a)
        assert game.returns(s) == list(ref.returns())
        assert sum(game.returns(s)) == 0


@pytest.mark.parametrize("name", ["kuhn_poker", "leduc_poker", "liars_dice"])
def test_public_hooks_hide_the_private_deals(name: str) -> None:
    game = make(name)
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = game.new_initial_state()
        while game.is_chance_node(s) and len(s.history) < 2:
            s = game.apply_action(s, int(rng.choice([a for a, _ in game.chance_outcomes(s)])))
        deals = s.history
        # Another deal with the same public play gives the same public string and tensor.
        t = game.new_initial_state()
        for a in deals[::-1] if deals[0] != deals[1] else deals:
            t = game.apply_action(t, a)
        while not game.is_terminal(s) and not game.is_chance_node(s):
            assert game.private_state_index(s, 0) == s.history[0]
            assert game.private_state_index(s, 1) == s.history[1]
            assert game.public_state_string(s) == game.public_state_string(t)
            assert np.array_equal(game.public_state_tensor(s), game.public_state_tensor(t))
            a = int(rng.choice(game.legal_actions(s)))
            s, t = game.apply_action(s, a), game.apply_action(t, a)
