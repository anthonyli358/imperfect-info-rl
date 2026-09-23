"""Self-play REINFORCE trainer for a :class:`~iirl.agents.reinforce.policy.LinearPolicy`.

The policy plays *both* seats against itself; each decision is credited with that player's final,
zero-sum return (minus a running baseline to cut variance), and the whole batch of decisions drives
one Adam ascent step of the policy gradient. Because self-play win rate hovers near 50% by
construction, progress is measured as **Elo against fixed anchors** (random and the heuristic),
recorded every ``eval_interval`` updates — the curve the caller plots.

Game-agnostic: it only uses the ``Game`` interface plus ``information_state_tensor`` and
``num_distinct_actions``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from iirl.agents.base import Agent
from iirl.agents.reinforce.policy import LinearPolicy, ReinforceAgent
from iirl.core.game import Game
from iirl.core.runner import _sample_chance
from iirl.eval.tournament import anchored_elo, play_match


@dataclass
class TrainConfig:
    """Hyperparameters for a training run.

    Attributes:
        updates: Number of gradient updates (training iterations).
        games_per_update: Self-play games collected per update.
        lr: Adam learning rate.
        entropy_coef: Entropy-bonus weight (exploration).
        reward_scale: Divides returns so advantages are O(1).
        baseline_decay: EMA decay for the value baseline.
        eval_interval: Evaluate Elo every this many updates.
        eval_games: Games per anchor in each Elo evaluation.
        feature_sample_games: Random games used to estimate feature mean/std.
        seed: Base RNG seed (reproducibility).
    """

    updates: int = 60
    games_per_update: int = 48
    lr: float = 0.02
    entropy_coef: float = 0.01
    reward_scale: float = 50.0
    baseline_decay: float = 0.9
    eval_interval: int = 5
    eval_games: int = 60
    feature_sample_games: int = 40
    seed: int = 0
    anchor_elos: Mapping[str, float] = field(
        default_factory=lambda: {"random": 1000.0, "heuristic": 1400.0}
    )


@dataclass
class EvalPoint:
    """One point on the training curve."""

    update: int
    games_seen: int
    elo: float
    win_rate_vs: dict[str, float]


def _feature_stats(
    game: Game, policy: LinearPolicy, rng: np.random.Generator, num_games: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate feature mean/std from random self-play (used to standardize inputs)."""
    rows = []
    for _ in range(num_games):
        state = game.new_initial_state()
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                state = game.apply_action(state, _sample_chance(game, state, rng))
                continue
            p = game.current_player(state)
            rows.append(game.information_state_tensor(state, p).astype(float))
            state = game.apply_action(state, int(rng.choice(game.legal_actions(state))))
    arr = np.asarray(rows)
    return arr.mean(axis=0), arr.std(axis=0)


def _play_and_record(
    game: Game, policy: LinearPolicy, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int], list[float]]:
    """Play one self-play game, returning per-decision (phi, action, probs, player) and returns."""
    phis, actions, probs, players = [], [], [], []
    state = game.new_initial_state()
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.apply_action(state, _sample_chance(game, state, rng))
            continue
        player = game.current_player(state)
        legal = game.legal_actions(state)
        phi = policy.features(game, state)
        p = policy.probs(phi, legal)
        action = int(rng.choice(policy.num_actions, p=p))
        phis.append(phi)
        actions.append(action)
        probs.append(p)
        players.append(player)
        state = game.apply_action(state, action)
    return phis, actions, probs, players, game.returns(state)


def _evaluate(
    game: Game,
    policy: LinearPolicy,
    anchors: Mapping[str, Agent],
    anchor_elos: Mapping[str, float],
    games: int,
    rng: np.random.Generator,
) -> tuple[float, dict[str, float]]:
    """Return the policy's anchored Elo (mean over anchors) and its win rate vs each anchor."""
    learner = ReinforceAgent(policy, "reinforce", greedy=True)
    elos, win_rates = [], {}
    for name, anchor in anchors.items():
        result = play_match(game, learner, anchor, games, rng)
        score_rate = (result.wins + 0.5 * result.draws) / result.num_games
        win_rates[name] = result.wins / result.num_games
        elos.append(anchored_elo(score_rate, anchor_elos[name]))
    return float(np.mean(elos)), win_rates


def train(
    game: Game,
    anchors: Mapping[str, Agent],
    config: TrainConfig,
    on_eval: object = None,
) -> tuple[LinearPolicy, list[EvalPoint]]:
    """Train a linear policy by self-play REINFORCE and return it with its Elo history.

    Args:
        game: The game to learn (must provide an info-state tensor and ``num_distinct_actions``).
        anchors: Fixed opponents to rate against, keyed by name (must match ``config.anchor_elos``).
        config: Hyperparameters.
        on_eval: Optional callable invoked with each :class:`EvalPoint` as it is produced.

    Returns:
        The trained policy and the list of evaluation points (the Elo curve).
    """
    rng = np.random.default_rng(config.seed)
    probe = game.new_initial_state()
    while game.is_chance_node(probe):
        probe = game.apply_action(probe, _sample_chance(game, probe, rng))
    feature_dim = len(game.information_state_tensor(probe, game.current_player(probe)))
    policy = LinearPolicy(feature_dim, game.num_distinct_actions(), seed=config.seed)
    policy.set_feature_stats(*_feature_stats(game, policy, rng, config.feature_sample_games))

    baseline = 0.0
    games_seen = 0
    history: list[EvalPoint] = []

    def record_eval(update: int) -> None:
        elo, win_rates = _evaluate(
            game, policy, anchors, config.anchor_elos, config.eval_games, rng
        )
        point = EvalPoint(update, games_seen, elo, win_rates)
        history.append(point)
        if callable(on_eval):
            on_eval(point)

    record_eval(0)
    for update in range(1, config.updates + 1):
        all_phis, all_actions, all_probs, targets = [], [], [], []
        for _ in range(config.games_per_update):
            phis, actions, probs, players, returns = _play_and_record(game, policy, rng)
            all_phis.extend(phis)
            all_actions.extend(actions)
            all_probs.extend(probs)
            targets.extend(returns[pl] / config.reward_scale for pl in players)
            games_seen += 1

        targets_arr = np.asarray(targets)
        baseline = (
            config.baseline_decay * baseline + (1 - config.baseline_decay) * targets_arr.mean()
        )
        advantages = targets_arr - baseline
        policy.apply_gradient(
            np.asarray(all_phis),
            np.asarray(all_actions),
            np.asarray(all_probs),
            advantages,
            lr=config.lr,
            entropy_coef=config.entropy_coef,
        )
        if update % config.eval_interval == 0 or update == config.updates:
            record_eval(update)

    return policy, history
