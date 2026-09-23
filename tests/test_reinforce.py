import numpy as np

import toy_game  # noqa: F401  (registers "toy_showdown")
from iirl.agents.random_agent import RandomAgent
from iirl.agents.reinforce import LinearPolicy, ReinforceAgent, TrainConfig, train
from iirl.core.registry import make
from iirl.eval.tournament import anchored_elo


def test_policy_gradient_matches_finite_differences() -> None:
    # The exact softmax policy gradient must equal a central finite-difference of the objective
    # J(W, b) = mean_i advantage_i * log pi(a_i). This is the correctness anchor for training.
    rng = np.random.default_rng(0)
    n_feat, n_act, n = 4, 3, 6
    policy = LinearPolicy(n_feat, n_act, seed=1)
    phis = rng.normal(size=(n, n_feat))
    actions = rng.integers(0, n_act, size=n)
    advantages = rng.normal(size=n)
    legal = list(range(n_act))

    def objective() -> float:
        total = 0.0
        for i in range(n):
            p = policy.probs(phis[i], legal)
            total += advantages[i] * np.log(p[actions[i]])
        return total / n

    probs = np.array([policy.probs(phis[i], legal) for i in range(n)])
    grad_w, grad_b = policy.policy_gradient(phis, actions, probs, advantages)

    eps = 1e-6
    for r in range(n_act):
        for c in range(n_feat):
            policy.w[r, c] += eps
            plus = objective()
            policy.w[r, c] -= 2 * eps
            minus = objective()
            policy.w[r, c] += eps
            assert abs((plus - minus) / (2 * eps) - grad_w[r, c]) < 1e-5


def test_probs_are_masked_to_legal_actions() -> None:
    policy = LinearPolicy(4, 5, seed=0)
    p = policy.probs(np.ones(4), legal=[1, 3])
    assert p[0] == 0.0 and p[2] == 0.0 and p[4] == 0.0
    assert abs(p.sum() - 1.0) < 1e-9 and p[1] > 0 and p[3] > 0


def test_anchored_elo_is_monotonic_and_even_at_anchor() -> None:
    assert anchored_elo(0.5, 1000.0) == 1000.0
    assert anchored_elo(0.9, 1000.0) > anchored_elo(0.6, 1000.0) > 1000.0
    assert anchored_elo(0.0, 1000.0) > -np.inf  # clipped, finite


def test_training_smoke_runs_and_plays_legally() -> None:
    # A tiny run on the toy game: it should execute end to end and produce a usable, legal policy.
    game = make("toy_showdown")
    anchors = {"random": RandomAgent("random")}
    config = TrainConfig(
        updates=3,
        games_per_update=20,
        eval_interval=1,
        eval_games=40,
        feature_sample_games=10,
        anchor_elos={"random": 1000.0},
        seed=0,
    )
    policy, history = train(game, anchors, config)
    assert len(history) == 4  # eval at update 0, then after 1, 2, 3
    agent = ReinforceAgent(policy, greedy=True)
    state = game.new_initial_state()
    rng = np.random.default_rng(0)
    while game.is_chance_node(state):
        from iirl.core.runner import _sample_chance

        state = game.apply_action(state, _sample_chance(game, state, rng))
    assert agent.step(game, state, rng) in game.legal_actions(state)
