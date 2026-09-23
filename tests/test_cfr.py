import numpy as np

import toy_game  # noqa: F401  (registers "toy_showdown")
from iirl.agents.cfr import CFRAgent, CFRSolver, regret_matching
from iirl.core.registry import make
from iirl.eval.exploitability import exploitability


def test_regret_matching_normalizes_positive_regret_and_defaults_uniform() -> None:
    np.testing.assert_allclose(regret_matching(np.array([3.0, 1.0, 0.0])), [0.75, 0.25, 0.0])
    np.testing.assert_allclose(regret_matching(np.array([-1.0, -2.0])), [0.5, 0.5])


def test_cfr_drives_exploitability_toward_zero() -> None:
    # On the mini-Kuhn toy game, CFR's average strategy must approach a Nash equilibrium.
    game = make("toy_showdown")
    solver = CFRSolver(game)

    solver.train(20)
    early = exploitability(game, solver.average_strategy()[0])
    solver.train(2000)
    late = exploitability(game, solver.average_strategy()[0])

    assert late < early  # it improves
    assert late < 0.02  # and gets close to equilibrium


def test_exploitability_of_uniform_play_is_high() -> None:
    # A strategy that plays every action uniformly (empty table -> the agent falls back to uniform)
    # is far more exploitable than a converged one.
    game = make("toy_showdown")
    solver = CFRSolver(game)
    solver.train(2000)
    cfr_expl = exploitability(game, solver.average_strategy()[0])
    uniform_expl = exploitability(game, {})  # no infosets -> uniform everywhere
    assert uniform_expl > cfr_expl
    assert uniform_expl > 0.05


def test_cfr_agent_plays_legal_actions() -> None:
    game = make("toy_showdown")
    solver = CFRSolver(game)
    solver.train(100)
    avg, actions = solver.average_strategy()
    agent = CFRAgent(avg, actions)
    from iirl.core.runner import _sample_chance

    rng = np.random.default_rng(0)
    state = game.new_initial_state()
    while game.is_chance_node(state):
        state = game.apply_action(state, _sample_chance(game, state, rng))
    assert agent.step(game, state, rng) in game.legal_actions(state)
