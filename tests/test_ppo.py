import numpy as np

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.ppo import PPOAgent, masked_policy
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.nn.policy_value import PolicyValueNet
from iirl.selfplay import PPOConfig, train_ppo


def _first_player_state(game, rng):
    state = game.new_initial_state()
    while game.is_chance_node(state):
        state = game.apply_action(state, _sample_chance(game, state, rng))
    return state


def test_masked_policy_is_a_legal_distribution() -> None:
    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    net = PolicyValueNet(
        len(game.information_state_tensor(state, 0)), game.num_distinct_actions(), (32,)
    )
    probs = masked_policy(net, game, state)
    legal = set(game.legal_actions(state))
    assert abs(probs.sum() - 1.0) < 1e-9
    assert all(probs[a] == 0 for a in range(len(probs)) if a not in legal)


def test_linear_value_head_is_unbounded() -> None:
    net = PolicyValueNet(4, 3, (8,), value_activation="linear")
    net.value_head.bias.data.fill_(50.0)  # linear head can exceed the tanh [-1, 1] range
    import torch

    _, value = net(torch.zeros(1, 4))
    assert float(value.item()) > 1.0


def test_ppo_agent_plays_legal_actions() -> None:
    game = make("ventures")
    rng = np.random.default_rng(1)
    state = _first_player_state(game, rng)
    net = PolicyValueNet(
        len(game.information_state_tensor(state, 0)),
        game.num_distinct_actions(),
        (32,),
        value_activation="linear",
    )
    assert PPOAgent(net).step(game, state, rng) in game.legal_actions(state)


def test_tiny_ppo_training_runs() -> None:
    game = make("ventures")
    config = PPOConfig(
        iterations=1,
        games_per_iter=4,
        epochs=2,
        minibatch_size=64,
        eval_interval=1,
        eval_games=8,
        feature_sample_games=6,
        hidden=(32,),
        anchor_elos={"random": 1000.0},
        seed=0,
    )
    net, history = train_ppo(game, {"random": RandomAgent("random")}, config)
    assert len(history) == 2
    assert np.isfinite(history[-1].policy_loss) and np.isfinite(history[-1].value_loss)
