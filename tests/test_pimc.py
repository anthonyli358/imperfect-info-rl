import numpy as np

import toy_game  # noqa: F401  (registers "toy_showdown")
from iirl.agents.mcts import MCTSConfig, PIMCAgent
from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from toy_game import BET_CALL, ToyState


def test_pimc_calls_with_a_winning_hand() -> None:
    # Player 1 holds the top card (2) and faces a bet: calling always wins, folding always loses.
    g = make("toy_showdown")
    state = ToyState(cards=(0, 2), history=(BET_CALL,))
    agent = PIMCAgent(MCTSConfig(num_determinizations=6, sims_per_world=40, max_rollout_depth=None))
    action = agent.step(g, state, np.random.default_rng(0))
    assert action == BET_CALL


def test_pimc_returns_legal_actions_on_ventures() -> None:
    g = make("ventures")
    rng = np.random.default_rng(0)
    agent = PIMCAgent(MCTSConfig(num_determinizations=3, sims_per_world=12, max_rollout_depth=8))
    state = g.new_initial_state()
    steps = 0
    while not g.is_terminal(state) and steps < 12:
        if g.is_chance_node(state):
            state = g.apply_action(state, _sample_chance(g, state, rng))
        else:
            action = agent.step(g, state, rng)
            assert action in g.legal_actions(state)
            state = g.apply_action(state, action)
            steps += 1
    assert steps > 0


def test_pimc_beats_random_on_toy() -> None:
    # A modest but clear edge over random on the tiny game validates the whole PIMC/UCT/resample
    # stack end to end (fast: toy games are a few plies).
    from iirl.agents.random_agent import RandomAgent
    from iirl.eval.tournament import play_match

    g = make("toy_showdown")
    agent = PIMCAgent(MCTSConfig(num_determinizations=8, sims_per_world=60, max_rollout_depth=None))
    result = play_match(g, agent, RandomAgent(), num_games=300, rng=np.random.default_rng(0))
    assert result.mean_return > 0.0
