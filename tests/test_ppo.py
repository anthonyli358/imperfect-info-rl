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


def test_shaped_gae_telescopes_to_the_true_return() -> None:
    from iirl.selfplay.ppo import _gae

    # With a zero baseline and lambda = 1, the first advantage is the whole shaped return, which
    # must equal (final return - first potential) regardless of the potentials in between.
    config = PPOConfig(gae_lambda=1.0, reward_scale=10.0)
    phi = [3.0, -7.0, 12.0, 5.0]
    adv, target = _gae(phi, [0.0] * 4, 30.0, config)
    assert abs(adv[0] - (30.0 - 3.0) / 10.0) < 1e-9
    assert np.allclose(adv, target)


def test_ppo_with_privileged_critic_league_and_win_bonus_runs() -> None:
    from iirl.agents.heuristic import VenturesHeuristic

    game = make("ventures")
    config = PPOConfig(
        iterations=2,
        games_per_iter=4,
        num_envs=4,
        epochs=1,
        eval_interval=2,
        eval_games=4,
        feature_sample_games=4,
        hidden=(32,),
        privileged_critic=True,
        win_bonus=20.0,
        scripted_opponents=(VenturesHeuristic(),),
        scripted_frac=0.5,
        snapshot_interval=1,
        anchor_elos={"random": 1000.0},
        seed=0,
    )
    net, history = train_ppo(game, {"random": RandomAgent("random")}, config)
    assert len(history) == 2
    assert np.isfinite(history[-1].policy_loss) and np.isfinite(history[-1].value_loss)
    assert "random" in history[-1].margin_vs


def test_split_evaluator_and_value_training_run() -> None:
    from iirl.nn.policy_value import SplitEvaluator
    from iirl.selfplay import ValueConfig, train_value

    game = make("ventures")
    rng = np.random.default_rng(3)
    state = _first_player_state(game, rng)
    dim = len(game.information_state_tensor(state, 0))
    policy = PolicyValueNet(dim, game.num_distinct_actions(), (32,), value_activation="linear")
    config = ValueConfig(
        games=8, positions_per_game=4, hidden=(16,), epochs=1, batch_size=8, num_workers=1
    )
    value_net, report = train_value(game, policy, (32,), config)
    assert np.isfinite(report.learned) and np.isfinite(report.heuristic)
    assert 0.0 <= report.coin_flip <= 1.0

    evaluator = SplitEvaluator(policy, value_net)
    priors, values = evaluator.evaluate_batch(game, [state, state])
    assert abs(priors[0].sum() - 1.0) < 1e-6 and -1.0 <= float(values[0]) <= 1.0
    p1, v1 = evaluator.evaluate(game, state)
    assert np.allclose(p1, priors[0]) and abs(v1 - float(values[0])) < 1e-5


def test_exploiter_training_warm_starts_and_keeps_a_fixed_pool() -> None:
    # A best-response run: the learner starts as a copy of the target and only faces it (greedy).
    import torch

    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    target = PolicyValueNet(
        len(game.information_state_tensor(state, 0)),
        game.num_distinct_actions(),
        (32,),
        value_activation="linear",
    )
    before = [p.clone() for p in target.parameters()]
    config = PPOConfig(
        iterations=2,
        games_per_iter=4,
        epochs=1,
        minibatch_size=64,
        eval_interval=2,
        eval_games=4,
        hidden=(32,),
        past_opponent_frac=1.0,
        opponent_greedy=True,
        snapshot_interval=1,
        anchor_elos={"random": 1000.0},
        seed=0,
    )
    net, _ = train_ppo(
        game, {"random": RandomAgent("random")}, config, init_net=target, opponents=[target]
    )
    assert torch.equal(net.feat_mean, target.feat_mean)  # warm-started, scaling included
    assert all(torch.equal(a, b) for a, b in zip(before, target.parameters(), strict=True))


def test_magnet_regularized_ppo_runs() -> None:
    game = make("ventures")
    config = PPOConfig(
        iterations=2,
        games_per_iter=4,
        epochs=1,
        minibatch_size=64,
        eval_interval=2,
        eval_games=4,
        feature_sample_games=6,
        hidden=(32,),
        magnet_coef=0.2,
        magnet_interval=1,
        anchor_elos={"random": 1000.0},
        seed=0,
    )
    _, history = train_ppo(game, {"random": RandomAgent("random")}, config)
    assert np.isfinite(history[-1].policy_loss)


def test_critic_warmup_leaves_the_policy_unchanged() -> None:
    import torch

    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    start = PolicyValueNet(
        len(game.information_state_tensor(state, 0)),
        game.num_distinct_actions(),
        (32,),
        value_activation="linear",
    )
    config = PPOConfig(
        iterations=1,
        games_per_iter=4,
        epochs=1,
        minibatch_size=64,
        eval_interval=1,
        eval_games=2,
        hidden=(32,),
        privileged_critic=True,
        critic_warmup=1,
        anchor_elos={"random": 1000.0},
        seed=0,
    )
    net, _ = train_ppo(game, {"random": RandomAgent("random")}, config, init_net=start)
    for a, b in zip(start.policy_head.parameters(), net.policy_head.parameters(), strict=True):
        assert torch.equal(a, b)
