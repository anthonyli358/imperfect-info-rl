"""AlphaZero-style self-play training for a :class:`~iirl.nn.policy_value.PolicyValueNet`.

Each iteration: (1) generate self-play games with determinized net-guided PUCT search (Dirichlet
root noise + a temperature schedule for exploration), storing every move's search visit distribution
and the eventual game outcome; (2) train the network to match those visit distributions (policy) and
outcomes (value); (3) rate the improved network's Elo against fixed anchors. The stronger network
makes the next search stronger — the AlphaZero improvement loop, here over an imperfect-information
game via determinization.

Game-agnostic apart from the anchors: it uses only the ``Game`` interface plus the info-state
tensor, ``num_distinct_actions``, and ``resample_from_infostate``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

from iirl.agents.alphazero.agent import AlphaZeroAgent
from iirl.agents.alphazero.mcts import PUCTConfig, az_search
from iirl.agents.base import Agent
from iirl.core.game import Game
from iirl.core.runner import _sample_chance
from iirl.eval.tournament import anchored_elo, play_match
from iirl.nn.policy_value import NetEvaluator, PolicyValueNet


@dataclass
class AZConfig:
    """Hyperparameters for an AlphaZero-style training run."""

    iterations: int = 12
    games_per_iter: int = 14
    simulations: int = 40
    determinizations: int = 1
    c_puct: float = 1.5
    leaf_value: str = "net"  # "net" = AlphaZero; "heuristic" = net-guided PIMC (priors only)
    heuristic_scale: float = 50.0
    temp_moves: int = 12
    temperature: float = 1.0
    train_steps_per_iter: int = 60
    batch_size: int = 128
    buffer_size: int = 30000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: tuple[int, ...] = (256, 256)
    eval_interval: int = 2
    eval_games: int = 40
    eval_simulations: int = 40
    eval_determinizations: int = 2
    feature_sample_games: int = 30
    seed: int = 0
    anchor_elos: Mapping[str, float] = field(
        default_factory=lambda: {"random": 1000.0, "heuristic": 1676.0}
    )


@dataclass
class AZEvalPoint:
    """One point on the AlphaZero training curve."""

    iteration: int
    games: int
    elo: float
    win_rate_vs: dict[str, float]
    policy_loss: float
    value_loss: float


def _feature_stats(game: Game, rng: np.random.Generator, num_games: int):
    """Estimate feature mean/std from random self-play (to standardize the net's inputs)."""
    rows = []
    for _ in range(num_games):
        state = game.new_initial_state()
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                state = game.apply_action(state, _sample_chance(game, state, rng))
                continue
            rows.append(game.information_state_tensor(state, game.current_player(state)))
            state = game.apply_action(state, int(rng.choice(game.legal_actions(state))))
    arr = np.asarray(rows, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def _self_play_game(
    game: Game, evaluator: NetEvaluator, config: AZConfig, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Play one self-play game; per move return (features, visit_policy, legal_mask, value)."""
    search = PUCTConfig(
        config.simulations,
        config.determinizations,
        config.c_puct,
        leaf_value=config.leaf_value,
        heuristic_scale=config.heuristic_scale,
    )
    num_actions = game.num_distinct_actions()
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    state = game.new_initial_state()
    move = 0
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.apply_action(state, _sample_chance(game, state, rng))
            continue
        me = game.current_player(state)
        legal = game.legal_actions(state)
        counts = az_search(game, state, me, evaluator, search, rng, add_noise=True)
        pi = counts / counts.sum()
        mask = np.zeros(num_actions, dtype=np.float32)
        mask[legal] = 1.0
        records.append((game.information_state_tensor(state, me).astype(np.float32), pi, mask, me))
        weights = np.array([counts[a] for a in legal], dtype=float)
        temp = config.temperature if move < config.temp_moves else 0.0
        if temp <= 0:
            action = int(legal[int(weights.argmax())])
        else:
            scaled = weights ** (1.0 / temp)
            action = int(rng.choice(legal, p=scaled / scaled.sum()))
        state = game.apply_action(state, action)
        move += 1
    returns = game.returns(state)
    return [(f, pi, mask, float(np.sign(returns[pl]))) for f, pi, mask, pl in records]


def _train_steps(
    net: PolicyValueNet,
    opt: optim.Optimizer,
    buffer: deque,
    config: AZConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Run gradient steps on random minibatches; return mean (policy_loss, value_loss)."""
    if len(buffer) < config.batch_size:
        return float("nan"), float("nan")
    net.train()
    data = list(buffer)
    p_losses, v_losses = [], []
    for _ in range(config.train_steps_per_iter):
        idx = rng.integers(0, len(data), size=config.batch_size)
        feats = torch.as_tensor(np.stack([data[i][0] for i in idx]), dtype=torch.float32)
        pis = torch.as_tensor(np.stack([data[i][1] for i in idx]), dtype=torch.float32)
        masks = torch.as_tensor(np.stack([data[i][2] for i in idx]), dtype=torch.float32)
        vals = torch.as_tensor(np.array([data[i][3] for i in idx]), dtype=torch.float32)
        logits, value = net(feats)
        # Cross-entropy over legal actions only (illegal logits pushed down before softmax), so the
        # head learns to rank legal moves rather than re-derive the mask inference already applies.
        logits = logits.masked_fill(masks == 0, -1e30)
        policy_loss = -(pis * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(value, vals)
        # In net-guided PIMC the value head is unused (heuristic leaves), so train the policy only.
        loss = policy_loss if config.leaf_value == "heuristic" else policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        p_losses.append(float(policy_loss.item()))
        v_losses.append(float(value_loss.item()))
    return float(np.mean(p_losses)), float(np.mean(v_losses))


def _evaluate(
    game: Game, evaluator: NetEvaluator, anchors: Mapping[str, Agent], config: AZConfig, rng
) -> tuple[float, dict[str, float]]:
    """Return the current net's anchored Elo and win rate vs each anchor (greedy search)."""
    search = PUCTConfig(
        config.eval_simulations,
        config.eval_determinizations,
        config.c_puct,
        leaf_value=config.leaf_value,
        heuristic_scale=config.heuristic_scale,
    )
    agent = AlphaZeroAgent(evaluator, search, "alphazero", temperature=0.0)
    elos, win_rates = [], {}
    for name, anchor in anchors.items():
        result = play_match(game, agent, anchor, config.eval_games, rng)
        win_rates[name] = result.wins / result.num_games
        score = (result.wins + 0.5 * result.draws) / result.num_games
        elos.append(anchored_elo(score, config.anchor_elos[name]))
    return float(np.mean(elos)), win_rates


def train_alphazero(
    game: Game, anchors: Mapping[str, Agent], config: AZConfig, on_eval: object = None
) -> tuple[PolicyValueNet, list[AZEvalPoint]]:
    """Train a policy/value net by AlphaZero-style self-play; return it with its Elo history."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    probe = game.new_initial_state()
    while game.is_chance_node(probe):
        probe = game.apply_action(probe, _sample_chance(game, probe, rng))
    input_dim = len(game.information_state_tensor(probe, game.current_player(probe)))
    net = PolicyValueNet(input_dim, game.num_distinct_actions(), config.hidden)
    net.set_feature_stats(*_feature_stats(game, rng, config.feature_sample_games))
    evaluator = NetEvaluator(net)
    opt = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    buffer: deque = deque(maxlen=config.buffer_size)
    games_seen = 0
    history: list[AZEvalPoint] = []
    last_losses = (float("nan"), float("nan"))

    def record_eval(iteration: int) -> None:
        elo, win_rates = _evaluate(game, evaluator, anchors, config, rng)
        point = AZEvalPoint(iteration, games_seen, elo, win_rates, last_losses[0], last_losses[1])
        history.append(point)
        if callable(on_eval):
            on_eval(point)

    record_eval(0)
    for iteration in range(1, config.iterations + 1):
        for _ in range(config.games_per_iter):
            buffer.extend(_self_play_game(game, evaluator, config, rng))
            games_seen += 1
        last_losses = _train_steps(net, opt, buffer, config, rng)
        net.eval()
        if iteration % config.eval_interval == 0 or iteration == config.iterations:
            record_eval(iteration)

    return net, history
