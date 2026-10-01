import numpy as np

import toy_game
from iirl.eval.exact_br import exploitability, uniform_policy
from iirl.selfplay.rebel import FullWidthReBeL, ReBeLConfig


class PublicToy(toy_game.ToyShowdown):
    """The toy game with the public/private split ReBeL needs: the card is private."""

    def num_private_states(self) -> int:  # noqa: D102
        return 3

    def private_state_index(self, state, player) -> int:  # noqa: ANN001, D102
        return state.cards[player]

    def public_state_string(self, state) -> str:  # noqa: ANN001, D102
        return str(state.history)

    def public_state_tensor(self, state) -> np.ndarray:  # noqa: ANN001, D102
        out = np.zeros(6, dtype=np.float32)
        for slot, action in enumerate(state.history):
            out[slot * 2 + action] = 1.0
        return out


def test_one_slab_is_plain_cfr_and_converges() -> None:
    game = PublicToy()
    rebel = FullWidthReBeL(game, ReBeLConfig(boundaries=(2, 5)))
    assert not rebel.nets  # the whole game is one subgame: no leaf values needed
    assert rebel.evaluate(1000)["exploitability"] < 0.005


def test_ranges_are_the_opponents_reach() -> None:
    rebel = FullWidthReBeL(PublicToy(), ReBeLConfig(boundaries=(2, 3, 5)))
    assert rebel.range_spread() < 1e-12
    level = rebel.levels[2]
    ranges, defined = level.ranges(rebel.root_reach)
    assert defined.all() and np.allclose(ranges, 1 / 3)  # the deal is uniform


def test_learned_leaf_values_give_a_near_equilibrium_policy() -> None:
    game = PublicToy()
    config = ReBeLConfig(
        boundaries=(2, 3, 5),
        cfr_iterations=50,
        eval_iterations=60,
        epochs=15,
        passes=4,
        hidden=64,
        layers=2,
        steps_per_epoch=100,
        batch_size=64,
        value_scale=1.0,
    )
    rebel = FullWidthReBeL(game, config)
    untrained = rebel.evaluate()["exploitability"]
    rebel.train()
    trained = rebel.evaluate()["exploitability"]
    random_play = exploitability(game, uniform_policy)["exploitability"]
    assert trained < untrained and trained < 0.1 * random_play


def test_single_actor_last_slab_best_response_matches_cfr() -> None:
    rebel = FullWidthReBeL(PublicToy(), ReBeLConfig(boundaries=(2, 3, 4, 5)))
    slab = rebel.slabs[-1]
    assert slab.actor == 0  # only player 0 (facing a bet after checking) acts, then the game ends
    lo, hi = rebel.cfr.level_ranges[4]
    root = np.random.default_rng(0).random((hi - lo, 4)) + 0.1
    exact, mask = rebel._solve(slab, root, 1)
    slab.actor = None  # force CFR on the same subgame
    iterated, _ = rebel._solve(slab, root, 2000)
    assert mask.any() and np.allclose(exact[mask], iterated[mask], atol=0.01)
