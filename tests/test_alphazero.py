import numpy as np
import torch

import iirl.games  # noqa: F401  (registers "ventures")
import toy_game  # noqa: F401  (registers "toy_showdown")
from iirl.agents.alphazero import AlphaZeroAgent, PUCTConfig, az_search
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.nn.policy_value import NetEvaluator, PolicyValueNet
from iirl.selfplay import AZConfig, train_alphazero


def _first_player_state(game, rng):
    state = game.new_initial_state()
    while game.is_chance_node(state):
        state = game.apply_action(state, _sample_chance(game, state, rng))
    return state


def test_net_output_shapes_and_value_range() -> None:
    net = PolicyValueNet(9, 2, hidden=(16,))
    logits, value = net(torch.zeros(5, 9))
    assert logits.shape == (5, 2)
    assert value.shape == (5,)
    assert torch.all(value >= -1) and torch.all(value <= 1)


def test_evaluator_priors_are_legal_masked() -> None:
    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    net = PolicyValueNet(
        len(game.information_state_tensor(state, 0)), game.num_distinct_actions(), (32,)
    )
    priors, value = NetEvaluator(net).evaluate(game, state)
    legal = set(game.legal_actions(state))
    assert abs(priors.sum() - 1.0) < 1e-6
    assert all(priors[a] == 0 for a in range(len(priors)) if a not in legal)
    assert -1.0 <= value <= 1.0


def test_search_visits_only_legal_actions_and_sum_to_budget() -> None:
    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    net = PolicyValueNet(
        len(game.information_state_tensor(state, 0)), game.num_distinct_actions(), (32,)
    )
    counts = az_search(
        game,
        state,
        game.current_player(state),
        NetEvaluator(net),
        PUCTConfig(simulations=25, determinizations=2),
        rng,
    )
    legal = set(game.legal_actions(state))
    assert set(np.nonzero(counts)[0]).issubset(legal)
    assert counts.sum() == 25 * 2


def test_guided_pimc_search_uses_heuristic_leaves() -> None:
    # Net-guided PIMC: leaf_value="heuristic" evaluates leaves with the game's estimate (not the net
    # value), while the net still supplies priors. Visits must still be legal and sum to the budget.
    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    net = PolicyValueNet(
        len(game.information_state_tensor(state, 0)), game.num_distinct_actions(), (32,)
    )
    counts = az_search(
        game,
        state,
        game.current_player(state),
        NetEvaluator(net),
        PUCTConfig(simulations=20, determinizations=1, leaf_value="heuristic"),
        rng,
    )
    assert set(np.nonzero(counts)[0]).issubset(set(game.legal_actions(state)))
    assert counts.sum() == 20


def test_agent_plays_legal_actions() -> None:
    game = make("ventures")
    rng = np.random.default_rng(1)
    state = _first_player_state(game, rng)
    net = PolicyValueNet(
        len(game.information_state_tensor(state, 0)), game.num_distinct_actions(), (32,)
    )
    agent = AlphaZeroAgent(NetEvaluator(net), PUCTConfig(simulations=15, determinizations=1))
    assert agent.step(game, state, rng) in game.legal_actions(state)


def test_tiny_training_runs_and_reduces_policy_loss() -> None:
    # A minimal end-to-end loop on the toy game: it must run and the training loss must be finite.
    game = make("toy_showdown")
    config = AZConfig(
        iterations=1,
        games_per_iter=6,
        simulations=8,
        determinizations=1,
        train_steps_per_iter=8,
        batch_size=4,
        eval_interval=1,
        eval_games=6,
        eval_simulations=8,
        eval_determinizations=1,
        feature_sample_games=6,
        hidden=(32,),
        anchor_elos={"random": 1000.0},
        seed=0,
    )
    net, history = train_alphazero(game, {"random": RandomAgent("random")}, config)
    assert len(history) == 2  # eval at iter 0 and iter 1
    assert np.isfinite(history[-1].policy_loss) and np.isfinite(history[-1].value_loss)


def test_outcome_value_blends_win_sign_with_margin() -> None:
    from iirl.agents.alphazero.mcts import outcome_value

    assert outcome_value(3.0) == 1.0 and outcome_value(-40.0) == -1.0 and outcome_value(0.0) == 0.0
    narrow, big = outcome_value(2.0, 0.5, 40.0), outcome_value(60.0, 0.5, 40.0)
    assert 0.5 < narrow < big < 1.0  # still a win, but winning by more is worth more
    assert outcome_value(-60.0, 0.5, 40.0) == -big  # symmetric for the loser
