"""PPO self-play training for Ventures (model-free, no search).

Why PPO here: CFR needs a short horizon and AlphaZero needs a learnable value; Ventures (a ~150-move
high-variance imperfect-info game) has neither. Model-free RL self-play does not care — it samples
whole episodes, and the value network is only a variance-reduction *baseline*, not a search
leaf-eval, so the deal's variance is a nuisance rather than fatal. This upgrades the linear
REINFORCE agent (which plateaued on capacity) to an MLP with clipped PPO updates and a baseline.

Two-player zero-sum self-play with a shared network: each decision is credited with that player's
final return; PPO's clipped surrogate keeps updates stable. Progress is Elo vs fixed anchors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn, optim

from iirl.agents.base import Agent
from iirl.agents.ppo.agent import PPOAgent, masked_policy
from iirl.core.game import Game
from iirl.core.runner import _sample_chance
from iirl.eval.tournament import anchored_elo, play_match
from iirl.nn.policy_value import PolicyValueNet


@dataclass
class PPOConfig:
    """Hyperparameters for a PPO self-play run."""

    iterations: int = 40
    games_per_iter: int = 32
    epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    reward_scale: float = 100.0
    hidden: tuple[int, ...] = (256, 256)
    eval_interval: int = 4
    eval_games: int = 80
    feature_sample_games: int = 40
    seed: int = 0
    anchor_elos: Mapping[str, float] = field(
        default_factory=lambda: {"random": 1000.0, "heuristic": 1676.0}
    )


@dataclass
class PPOEvalPoint:
    """One point on the PPO training curve."""

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


def _play_game(net: PolicyValueNet, game: Game, rng: np.random.Generator, scale: float):
    """Play one self-play game; return per-step (feat, action, mask, old_logp, target)."""
    num_actions = game.num_distinct_actions()
    feats, actions, masks, logps, players = [], [], [], [], []
    state = game.new_initial_state()
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.apply_action(state, _sample_chance(game, state, rng))
            continue
        me = game.current_player(state)
        legal = game.legal_actions(state)
        probs = masked_policy(net, game, state)
        action = int(rng.choice(num_actions, p=probs))
        mask = np.zeros(num_actions, dtype=np.float32)
        mask[legal] = 1.0
        feats.append(game.information_state_tensor(state, me).astype(np.float32))
        actions.append(action)
        masks.append(mask)
        logps.append(float(np.log(probs[action] + 1e-12)))
        players.append(me)
        state = game.apply_action(state, action)
    returns = game.returns(state)
    targets = [returns[pl] / scale for pl in players]
    return feats, actions, masks, logps, targets


def _ppo_update(
    net: PolicyValueNet, opt: optim.Optimizer, batch: dict, config: PPOConfig, rng
) -> tuple[float, float]:
    """Run PPO epochs over the collected batch; return mean (policy_loss, value_loss)."""
    feats = torch.as_tensor(np.stack(batch["feats"]), dtype=torch.float32)
    actions = torch.as_tensor(np.asarray(batch["actions"]), dtype=torch.long)
    masks = torch.as_tensor(np.stack(batch["masks"]), dtype=torch.float32)
    old_logp = torch.as_tensor(np.asarray(batch["logps"]), dtype=torch.float32)
    targets = torch.as_tensor(np.asarray(batch["targets"]), dtype=torch.float32)

    with torch.no_grad():
        _, values = net(feats)
    advantages = targets - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(actions)
    p_losses, v_losses = [], []
    for _ in range(config.epochs):
        for _batch in range(0, n, config.minibatch_size):
            idx = torch.as_tensor(
                rng.choice(n, size=min(config.minibatch_size, n), replace=False), dtype=torch.long
            )
            logits, value = net(feats[idx])
            logits = logits.masked_fill(masks[idx] == 0, -1e30)
            logp_all = torch.log_softmax(logits, dim=1)
            new_logp = logp_all.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_logp - old_logp[idx])
            adv = advantages[idx]
            surrogate = torch.min(
                ratio * adv, torch.clamp(ratio, 1 - config.clip, 1 + config.clip) * adv
            )
            policy_loss = -surrogate.mean()
            value_loss = nn.functional.mse_loss(value, targets[idx])
            entropy = -(torch.exp(logp_all) * logp_all).nan_to_num(0.0).sum(dim=1).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            opt.step()
            p_losses.append(float(policy_loss.item()))
            v_losses.append(float(value_loss.item()))
    return float(np.mean(p_losses)), float(np.mean(v_losses))


def _evaluate(game: Game, net: PolicyValueNet, anchors, config: PPOConfig, rng):
    """Return the net's anchored Elo and win rate vs each anchor (greedy policy)."""
    agent = PPOAgent(net, "ppo", greedy=True)
    elos, win_rates = [], {}
    for name, anchor in anchors.items():
        result = play_match(game, agent, anchor, config.eval_games, rng)
        win_rates[name] = result.wins / result.num_games
        score = (result.wins + 0.5 * result.draws) / result.num_games
        elos.append(anchored_elo(score, config.anchor_elos[name]))
    return float(np.mean(elos)), win_rates


def train_ppo(
    game: Game, anchors: Mapping[str, Agent], config: PPOConfig, on_eval: object = None
) -> tuple[PolicyValueNet, list[PPOEvalPoint]]:
    """Train a policy/value net by PPO self-play; return it with its Elo history."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    probe = game.new_initial_state()
    while game.is_chance_node(probe):
        probe = game.apply_action(probe, _sample_chance(game, probe, rng))
    input_dim = len(game.information_state_tensor(probe, game.current_player(probe)))
    net = PolicyValueNet(
        input_dim, game.num_distinct_actions(), config.hidden, value_activation="linear"
    )
    net.set_feature_stats(*_feature_stats(game, rng, config.feature_sample_games))
    opt = optim.Adam(net.parameters(), lr=config.lr)

    games_seen = 0
    history: list[PPOEvalPoint] = []
    last_losses = (float("nan"), float("nan"))

    def record_eval(iteration: int) -> None:
        net.eval()
        elo, win_rates = _evaluate(game, net, anchors, config, rng)
        point = PPOEvalPoint(iteration, games_seen, elo, win_rates, *last_losses)
        history.append(point)
        if callable(on_eval):
            on_eval(point)

    record_eval(0)
    for iteration in range(1, config.iterations + 1):
        batch = {"feats": [], "actions": [], "masks": [], "logps": [], "targets": []}
        for _ in range(config.games_per_iter):
            feats, actions, masks, logps, targets = _play_game(net, game, rng, config.reward_scale)
            batch["feats"] += feats
            batch["actions"] += actions
            batch["masks"] += masks
            batch["logps"] += logps
            batch["targets"] += targets
            games_seen += 1
        net.train()
        last_losses = _ppo_update(net, opt, batch, config, rng)
        if iteration % config.eval_interval == 0 or iteration == config.iterations:
            record_eval(iteration)

    return net, history
