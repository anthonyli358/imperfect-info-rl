import numpy as np
import pytest

import iirl.games  # noqa: F401
from iirl.core.registry import make
from iirl.selfplay.public_belief import PublicBelief, PublicBeliefConfig


def _uniform_model(game, states):  # noqa: ANN001, ANN202
    out = []
    for s in states:
        probs = np.zeros(game.num_distinct_actions())
        legal = game.legal_actions(s)
        probs[legal] = 1.0 / len(legal)
        out.append(probs)
    return out


@pytest.mark.parametrize("name", ["mini_ventures", "ventures"])
def test_candidates_stay_possible_through_random_games(name: str) -> None:
    game = make(name)
    rng = np.random.default_rng(0)
    for seed in range(3):
        belief = PublicBelief(game, _uniform_model, PublicBeliefConfig(particles=16), seed=seed)
        state = game.new_initial_state()
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            else:
                legal = game.legal_actions(state)
                action = legal[rng.integers(len(legal))]
            after = game.apply_action(state, action)
            belief.observe(state, action, after)
            state = after
            if belief.hands is None or game.is_terminal(state):
                continue
            unseen = np.asarray(game.unseen_cards(state))
            for p in (0, 1):
                hands = belief.hands[p]
                size = sum(game.private_hand(state, p))
                if game.is_chance_node(state) and game.chance_is_private(state):
                    continue  # mid-draw: the mover's candidates are one card short by design
                assert (hands.sum(axis=1) == size).all()
                assert (hands >= np.asarray(game.known_cards(state, p))).all()
                assert (hands <= unseen - np.asarray(game.known_cards(state, 1 - p))).all()
                assert np.isclose(belief.marginals(p).sum(), size)
