import numpy as np

import toy_game  # noqa: F401  (registers "toy_showdown")
from iirl.agents.cfr.subgame import SubgameConfig, SubgameSolver
from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.eval.exploitability import exploitability

G = make("toy_showdown")


def _deal(rng):  # noqa: ANN001, ANN202
    s = G.new_initial_state()
    while G.is_chance_node(s):
        s = G.apply_action(s, _sample_chance(G, s, rng))
    return s


def test_mccfr_from_full_deals_approaches_the_equilibrium() -> None:
    # Sampling complete deals (both cards) and solving to full depth is plain MCCFR on the whole
    # toy game, so its average strategy must be close to a Nash equilibrium.
    solver = SubgameSolver(G, SubgameConfig(iterations=20000, depth=10))
    solver.run(_deal, np.random.default_rng(0))
    assert exploitability(G, solver.average_strategy()) < 0.03


def test_fixing_our_hand_removes_the_bluff() -> None:
    # Solving from a real position re-deals only the opponent's card, so the opponent plays as if
    # it knew ours. With the lowest card, betting then only loses more: the bluff disappears.
    rng = np.random.default_rng(1)
    while True:
        state = _deal(rng)
        if state.cards[0] == 0:  # player 0 to act holding the lowest card
            break
    solver = SubgameSolver(G, SubgameConfig(iterations=5000, depth=10))
    solver.run(lambda r: G.resample_from_infostate(state, 0, r), rng)
    actions, probs = solver.policy_at(G.information_state_string(state, 0))
    bet = dict(zip(actions, probs, strict=True))[toy_game.BET_CALL]
    assert bet < 0.05  # the equilibrium bluffs this card ~0.22 of the time
