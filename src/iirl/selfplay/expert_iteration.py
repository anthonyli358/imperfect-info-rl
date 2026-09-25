"""Expert iteration: AlphaZero-style search self-play, warm-started from a model-free policy.

The AlphaZero loop — search with the network, train the network toward the search, repeat — with
two changes for a long, high-variance, imperfect-information game:

* **Warm start, split networks.** The priors come from a policy net (e.g. PPO's) and the leaf value
  from a separately trained info-state value net (:mod:`iirl.selfplay.value`). From scratch, the
  value target is too noisy for the loop to get going (see ``writeup.md``); from a strong policy it
  starts from a search that already beats the policy alone.
* **Optionally, a full-state leaf value** (``privileged_value``): inside a determinized world both
  hands are known, so the value net can read both players' views and value the position as it is.
* **A policy target that survives a small budget.** Raw visit counts from a few hundred
  simulations are a flat, noisy distribution; training a sharp policy toward them destroys it.
  The default target is Gumbel MuZero's ``softmax(log prior + sigma(completed Q))``.
* **A lower-variance value target.** The final outcome of a card game is mostly the deal. The value
  trains toward ``(1 - q_mix) * z + q_mix * q``, where ``z`` is the outcome (win/loss blended with
  the margin, :func:`~iirl.agents.alphazero.mcts.outcome_value`) and ``q`` the root search value.

Search is determinized PUCT (:func:`~iirl.agents.alphazero.mcts.az_search_stats`); self-play runs in
a persistent process pool, one game per task batch, with Dirichlet root noise and visit-count
sampling for the opening moves. Game-agnostic: it uses only the ``Game`` interface.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from iirl.agents.alphazero.mcts import (
    PUCTConfig,
    az_search_stats,
    completed_q_policy,
    outcome_value,
)
from iirl.core.game import Game
from iirl.core.runner import _sample_chance
from iirl.nn.policy_value import PolicyValueNet, SplitEvaluator, privileged_features


@dataclass
class ExItConfig:
    """Hyperparameters for an expert-iteration run."""

    iterations: int = 20
    games_per_iter: int = 240
    simulations: int = 64
    determinizations: int = 4
    c_puct: float = 1.5
    leaf_batch: int = 8
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temp_moves: int = 10  # decisions per game (both players) sampled ∝ visits; then argmax
    # Outcome scale shared by terminal values, the value target and the search (see outcome_value).
    margin_weight: float = 0.0
    margin_scale: float = 40.0
    q_mix: float = 0.5  # value target = (1 - q_mix) * outcome + q_mix * root search value
    # The value net reads both players' views (a full-state value for determinized worlds).
    privileged_value: bool = False
    # Policy target: "completed_q" = softmax(log prior + sigma(q)) (Gumbel MuZero), which improves
    # the prior even at small budgets; "visits" = raw visit counts (AlphaZero), which flattens a
    # sharp prior when simulations are few.
    policy_target: str = "completed_q"
    c_visit: float = 50.0
    c_scale: float = 0.1
    buffer_iters: int = 3  # train on the last k iterations' data
    policy_lr: float = 1e-4
    value_lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 2
    batch_size: int = 1024
    num_workers: int = 12
    seed: int = 0


@dataclass
class ExItStats:
    """Per-iteration training statistics."""

    iteration: int
    games: int
    positions: int
    policy_loss: float
    value_loss: float
    mean_root_q: float
    seconds: float


def _search_config(config: ExItConfig) -> PUCTConfig:
    return PUCTConfig(
        simulations=config.simulations,
        determinizations=config.determinizations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_frac=config.dirichlet_frac,
        leaf_batch=config.leaf_batch,
        margin_weight=config.margin_weight,
        margin_scale=config.margin_scale,
    )


def _build(dims, state, activation: str) -> PolicyValueNet:
    net = PolicyValueNet(dims[0], dims[1], tuple(dims[2]), value_activation=activation)
    net.load_state_dict(state)
    net.eval()
    return net


def _selfplay(args) -> list[dict[str, np.ndarray]]:
    """Worker: play search self-play games; return per-game decision records."""
    game, nets, config, num_games, seed = args
    torch.set_num_threads(1)
    (pdims, pstate), (vdims, vstate) = nets
    evaluator = SplitEvaluator(
        _build(pdims, pstate, "linear"), _build(vdims, vstate, "tanh"), config.privileged_value
    )
    search = _search_config(config)
    num_actions = game.num_distinct_actions()
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(num_games):
        feats, vfeats, pis, masks, qs, players, roots = [], [], [], [], [], [], []
        state = game.new_initial_state()
        move = 0
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                state = game.apply_action(state, _sample_chance(game, state, rng))
                continue
            me = game.current_player(state)
            legal = game.legal_actions(state)
            stats = az_search_stats(game, state, me, evaluator, search, rng, add_noise=True)
            counts, q = stats.counts, stats.value
            mask = np.zeros(num_actions, dtype=np.float32)
            mask[legal] = 1.0
            feats.append(game.information_state_tensor(state, me).astype(np.float32))
            if config.privileged_value:
                vfeats.append(privileged_features(game, state, me).astype(np.float32))
            if config.policy_target == "completed_q":
                pi = completed_q_policy(stats, legal, config.c_visit, config.c_scale)
            else:
                pi = counts / counts.sum()
            pis.append(pi.astype(np.float32))
            masks.append(mask)
            qs.append(q)
            roots.append(stats)
            players.append(me)
            weights = np.array([counts[a] for a in legal], dtype=float)
            if move < config.temp_moves:
                action = int(rng.choice(legal, p=weights / weights.sum()))
            else:
                action = int(legal[int(weights.argmax())])
            state = game.apply_action(state, action)
            move += 1
        returns = game.returns(state)
        games.append(
            {
                "feat": np.stack(feats),
                "vfeat": np.stack(vfeats) if vfeats else None,
                "pi": np.stack(pis),
                # Raw root statistics, so policy targets can be recomputed offline.
                "root_counts": np.stack([r.counts for r in roots]).astype(np.float32),
                "root_q": np.stack([r.q for r in roots]).astype(np.float32),
                "root_prior": np.stack([r.prior for r in roots]).astype(np.float32),
                "mask": np.stack(masks),
                "q": np.asarray(qs, dtype=np.float32),
                "margin": np.asarray([returns[p] for p in players], dtype=np.float32),
            }
        )
    return games


def _fit(
    prior: PolicyValueNet,
    value: PolicyValueNet,
    opts: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    data: list[dict[str, np.ndarray]],
    config: ExItConfig,
) -> tuple[float, float]:
    """Train the policy toward the visit distributions and the value toward the mixed target."""
    x = torch.as_tensor(np.concatenate([g["feat"] for g in data]))
    xv = (
        torch.as_tensor(np.concatenate([g["vfeat"] for g in data]))
        if config.privileged_value
        else x
    )
    pi = torch.as_tensor(np.concatenate([g["pi"] for g in data]))
    mask = torch.as_tensor(np.concatenate([g["mask"] for g in data]))
    z = np.concatenate(
        [
            [outcome_value(m, config.margin_weight, config.margin_scale) for m in g["margin"]]
            for g in data
        ]
    )
    q = np.concatenate([g["q"] for g in data])
    y = torch.as_tensor(((1 - config.q_mix) * z + config.q_mix * q).astype(np.float32))
    prior.train()
    value.train()
    p_losses, v_losses = [], []
    for _ in range(config.epochs):
        perm = torch.randperm(len(x))
        for start in range(0, len(x), config.batch_size):
            idx = perm[start : start + config.batch_size]
            logits = prior(x[idx])[0].masked_fill(mask[idx] == 0, -1e9)
            p_loss = -(pi[idx] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            v_loss = nn.functional.mse_loss(value(xv[idx])[1], y[idx])
            for opt, loss in zip(opts, (p_loss, v_loss), strict=True):
                opt.zero_grad()
                loss.backward()
                opt.step()
            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
    prior.eval()
    value.eval()
    return float(np.mean(p_losses)), float(np.mean(v_losses))


def _dims(net: PolicyValueNet) -> tuple[int, int, tuple[int, ...]]:
    hidden = tuple(m.out_features for m in net.trunk if isinstance(m, nn.Linear))
    return net.input_dim, net.num_actions, hidden


def train_expert_iteration(
    game: Game,
    prior: PolicyValueNet,
    value: PolicyValueNet,
    config: ExItConfig,
    on_iteration: Callable[[ExItStats, PolicyValueNet, PolicyValueNet], None] | None = None,
) -> tuple[PolicyValueNet, PolicyValueNet, list[ExItStats]]:
    """Improve ``prior`` (search priors) and ``value`` (leaf value, tanh) by search self-play.

    Args:
        game: The game.
        prior: Policy network, trained in place toward the search's visit distributions.
        value: Value network (``tanh`` head), trained in place toward the mixed value target.
        config: Loop settings.
        on_iteration: Called after each iteration with its stats and the updated networks (e.g. to
            checkpoint and evaluate).

    Returns:
        The trained ``(prior, value)`` networks and the per-iteration stats.
    """
    import time

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    opts = (
        torch.optim.Adam(prior.parameters(), lr=config.policy_lr, weight_decay=config.weight_decay),
        torch.optim.Adam(value.parameters(), lr=config.value_lr, weight_decay=config.weight_decay),
    )
    buffer: deque[list[dict[str, np.ndarray]]] = deque(maxlen=config.buffer_iters)
    workers = max(1, config.num_workers)
    per = [
        config.games_per_iter // workers + (i < config.games_per_iter % workers)
        for i in range(workers)
    ]
    history: list[ExItStats] = []
    pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        for it in range(1, config.iterations + 1):
            t0 = time.time()
            nets = (
                (_dims(prior), {k: v.clone() for k, v in prior.state_dict().items()}),
                (_dims(value), {k: v.clone() for k, v in value.state_dict().items()}),
            )
            jobs = [
                (game, nets, config, n, int(s))
                for n, s in zip(per, rng.integers(2**62, size=workers), strict=True)
                if n > 0
            ]
            parts = pool.map(_selfplay, jobs) if pool else [_selfplay(j) for j in jobs]
            fresh = [g for part in parts for g in part]
            buffer.append(fresh)
            data = [g for batch in buffer for g in batch]
            p_loss, v_loss = _fit(prior, value, opts, data, config)
            stats = ExItStats(
                iteration=it,
                games=it * config.games_per_iter,
                positions=sum(len(g["q"]) for g in data),
                policy_loss=p_loss,
                value_loss=v_loss,
                mean_root_q=float(np.mean(np.concatenate([g["q"] for g in fresh]))),
                seconds=time.time() - t0,
            )
            history.append(stats)
            if on_iteration is not None:
                on_iteration(stats, prior, value)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return prior, value, history
