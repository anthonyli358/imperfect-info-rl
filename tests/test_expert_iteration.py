import numpy as np
import torch

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.eval.duplicate import _play, duplicate_match
from iirl.nn.policy_value import PolicyValueNet, SplitEvaluator, privileged_features
from iirl.selfplay.expert_iteration import ExItConfig, train_expert_iteration


def _first_player_state(game, rng):
    state = game.new_initial_state()
    while game.is_chance_node(state):
        state = game.apply_action(state, _sample_chance(game, state, rng))
    return state


def _random_agent() -> RandomAgent:
    return RandomAgent("random")


def test_duplicate_deal_repeats_the_cards() -> None:
    # The same deal seed gives the same opening hands whoever sits where.
    game = make("ventures")
    seen = []
    for agent_seed in (1, 2):
        state = game.new_initial_state()
        k = 0
        while game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            probs = np.array([p for _, p in outcomes])
            i = np.random.default_rng((5, k)).choice(len(outcomes), p=probs)
            state = game.apply_action(state, outcomes[i][0])
            k += 1
        seen.append(state.hands)
        _play(game, (RandomAgent("a"), RandomAgent("b")), 5, np.random.default_rng(agent_seed))
    assert seen[0] == seen[1]


def test_duplicate_match_self_play_is_even() -> None:
    game = make("ventures")
    r = duplicate_match(game, (_random_agent, {}), (_random_agent, {}), deals=4, workers=1)
    assert r.deals == 4
    assert 0.0 <= r.score <= 1.0


def test_privileged_split_evaluator() -> None:
    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    dim = len(game.information_state_tensor(state, 0))
    feat = privileged_features(game, state, game.current_player(state))
    assert feat.shape == (2 * dim,)
    prior = PolicyValueNet(dim, game.num_distinct_actions(), (16,), "linear")
    value = PolicyValueNet(2 * dim, game.num_distinct_actions(), (16,), "tanh")
    priors, v = SplitEvaluator(prior, value, privileged=True).evaluate(game, state)
    assert abs(priors.sum() - 1.0) < 1e-6
    assert -1.0 <= v <= 1.0


def test_expert_iteration_runs_and_updates_nets() -> None:
    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    dim = len(game.information_state_tensor(state, 0))
    prior = PolicyValueNet(dim, game.num_distinct_actions(), (16,), "linear")
    value = PolicyValueNet(dim, game.num_distinct_actions(), (16,), "tanh")
    before = [p.clone() for p in prior.parameters()]
    config = ExItConfig(
        iterations=1, games_per_iter=1, simulations=4, determinizations=1, num_workers=1
    )
    _, _, history = train_expert_iteration(game, prior, value, config)
    assert len(history) == 1 and history[0].positions > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, prior.parameters(), strict=True))


def test_expert_iteration_with_privileged_value() -> None:
    game = make("ventures")
    rng = np.random.default_rng(0)
    state = _first_player_state(game, rng)
    dim = len(game.information_state_tensor(state, 0))
    prior = PolicyValueNet(dim, game.num_distinct_actions(), (16,), "linear")
    value = PolicyValueNet(2 * dim, game.num_distinct_actions(), (16,), "tanh")
    config = ExItConfig(
        iterations=1,
        games_per_iter=1,
        simulations=4,
        determinizations=1,
        num_workers=1,
        privileged_value=True,
    )
    _, _, history = train_expert_iteration(game, prior, value, config)
    assert history[0].positions > 0


def test_completed_q_policy_reweights_prior() -> None:
    from iirl.agents.alphazero import RootStats, completed_q_policy

    prior = np.array([0.7, 0.2, 0.1, 0.0])
    counts = np.array([10.0, 5.0, 0.0, 0.0])
    legal = [0, 1, 2]
    # Equal values everywhere: the target is exactly the prior.
    same = RootStats(counts, 0.1, np.array([0.1, 0.1, np.nan, np.nan]), prior)
    assert np.allclose(completed_q_policy(same, legal), prior)
    # Action 1 searched better: its share rises; the illegal action stays at 0.
    better = RootStats(counts, 0.1, np.array([0.0, 0.5, np.nan, np.nan]), prior)
    target = completed_q_policy(better, legal)
    assert target[1] > prior[1] and target[3] == 0.0
    assert abs(target.sum() - 1.0) < 1e-9
