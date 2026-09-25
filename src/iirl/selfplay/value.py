"""Learn an information-state value (win probability) from a policy's self-play.

This supplies the missing half of AlphaZero-style search on top of a model-free policy: the policy
(e.g. PPO) gives the search its priors, and this value net evaluates its leaves. The value sees only
the mover's own information state (never the hidden cards), and predicts the game's outcome sign
from the mover's view, in ``[-1, 1]`` — the same scale the search uses for terminal states.

The target is the outcome from the mover's view — by default its win/loss sign, optionally blended
with the score margin (``margin_weight``) so the value also prefers winning by more.

The target is noisy (the deal decides much of a game), so the net overfits quickly when many
positions share one outcome. Two guards: sample only a few positions per game from many games, and
early-stop on held-out *games* (no test position shares an outcome with a training position). The
report compares the learned value against the game's hand-crafted ``heuristic_value`` leaf by Brier
score, so it is only used in search if it is actually the better evaluator.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from iirl.agents.alphazero.mcts import outcome_value
from iirl.core.game import Game
from iirl.nn.policy_value import PolicyValueNet
from iirl.selfplay.ppo import _Env, _run_envs


@dataclass
class ValueConfig:
    """Settings for value-net training from self-play."""

    games: int = 72_000
    positions_per_game: int = 8
    test_frac: float = 1 / 12
    hidden: tuple[int, ...] = (512, 512)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 6
    batch_size: int = 2048
    evals_per_epoch: int = 4
    num_workers: int = 12
    heuristic_scales: tuple[float, ...] = (10, 15, 20, 30, 40, 50, 75, 100)
    # Training target: (1 - w) * sign(margin) + w * tanh(margin / margin_scale). w = 0 is pure
    # win/loss; w > 0 also credits winning by more. Use the same w/scale in the search's PUCTConfig.
    margin_weight: float = 0.0
    margin_scale: float = 40.0
    # Train on both players' views (``privileged_features``): a full-state value for leaves of
    # determinized search (``SplitEvaluator(privileged=True)``), not an info-state one.
    privileged: bool = False
    seed: int = 0


@dataclass
class ValueReport:
    """Held-out Brier scores (lower is better) for predicting the winner.

    With a margin-blended target the learned value is not a pure win probability, so its Brier score
    is a rough guide there; head-to-head play is the real test.
    """

    positions: int
    learned: float
    heuristic: float
    heuristic_scale: float
    coin_flip: float
    curve: list[float] = field(default_factory=list)


def brier(value: np.ndarray, outcome: np.ndarray) -> float:
    """Brier score of a ``[-1, 1]`` value read as a win-probability forecast (draw = 0.5)."""
    return float(np.mean(((value + 1) / 2 - (outcome + 1) / 2) ** 2))


def _positions(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker: play self-play games; return sampled (features, heuristic value, final margin)."""
    game, policy_state, dims, num_games, per_game, privileged, seed = args
    torch.set_num_threads(1)
    net = PolicyValueNet(dims[0], dims[1], dims[2], value_activation="linear")
    net.load_state_dict(policy_state)
    net.eval()
    rng = np.random.default_rng(seed)
    feats, hvals, ys = [], [], []
    for start in range(0, num_games, 32):
        envs = [
            _Env(game.new_initial_state(), (net, net), (True, True))
            for _ in range(min(32, num_games - start))
        ]
        # shaping=True records phi = heuristic_value(state, mover), the baseline leaf, per decision.
        rets_all = _run_envs(game, envs, rng, True, record_privileged=privileged)
        for env, rets in zip(envs, rets_all, strict=True):
            rows = [(p, k) for p in (0, 1) for k in range(len(env.traj[p]["feat"]))]
            for i in rng.choice(len(rows), size=min(per_game, len(rows)), replace=False):
                p, k = rows[i]
                feats.append(env.traj[p]["cfeat" if privileged else "feat"][k])
                hvals.append(env.traj[p]["phi"][k])
                ys.append(float(rets[p]))  # raw margin; mapped to the target in train_value
    return (
        np.stack(feats).astype(np.float32),
        np.asarray(hvals, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def train_value(
    game: Game, policy: PolicyValueNet, hidden: tuple[int, ...], config: ValueConfig
) -> tuple[PolicyValueNet, ValueReport]:
    """Train an info-state value net on ``policy``'s self-play; return it and a held-out report.

    Args:
        game: The game (must provide ``heuristic_value`` for the baseline comparison).
        policy: The self-play policy (a PPO-style net); only its policy head is used.
        hidden: ``policy``'s hidden sizes (needed to rebuild it in worker processes).
        config: Data and training settings.
    """
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    shards = max(2, config.num_workers)  # data shards; at least one each for train and test
    n_test = max(1, round(shards * config.test_frac))
    dims = (policy.input_dim, policy.num_actions, hidden)
    state = {k: v.clone() for k, v in policy.state_dict().items()}
    jobs = [
        (
            game,
            state,
            dims,
            config.games // shards,
            config.positions_per_game,
            config.privileged,
            int(s),
        )
        for s in rng.integers(2**31, size=shards)
    ]
    if config.num_workers > 1:
        with mp.get_context("spawn").Pool(config.num_workers) as pool:
            parts = pool.map(_positions, jobs)
    else:
        parts = [_positions(job) for job in jobs]
    train, test = parts[:-n_test], parts[-n_test:]  # split by game, not position
    x_tr, m_tr = (np.concatenate([p[i] for p in train]) for i in (0, 2))
    x_te, h_te, m_te = (np.concatenate([p[i] for p in test]) for i in (0, 1, 2))
    y_tr, y_te = np.sign(m_tr), np.sign(m_te)  # win/loss, for the Brier comparison
    target = np.asarray(
        [outcome_value(m, config.margin_weight, config.margin_scale) for m in m_tr],
        dtype=np.float32,
    )
    h_tr = np.concatenate([p[1] for p in train])
    scale = min(config.heuristic_scales, key=lambda s: brier(np.tanh(h_tr / s), y_tr))

    net = PolicyValueNet(x_tr.shape[1], game.num_distinct_actions(), config.hidden, "tanh")
    net.set_feature_stats(x_tr.mean(axis=0), x_tr.std(axis=0))
    opt = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    xt, yt, xv = torch.as_tensor(x_tr), torch.as_tensor(target), torch.as_tensor(x_te)
    n_batches = (len(xt) + config.batch_size - 1) // config.batch_size
    eval_every = max(1, n_batches // config.evals_per_epoch)
    best, best_state, curve = np.inf, None, []
    for _ in range(config.epochs):
        perm = torch.randperm(len(xt))
        for b in range(n_batches):
            net.train()
            idx = perm[b * config.batch_size : (b + 1) * config.batch_size]
            loss = nn.functional.mse_loss(net(xt[idx])[1], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (b + 1) % eval_every == 0:
                net.eval()
                with torch.no_grad():
                    score = brier(net(xv)[1].numpy(), y_te)
                curve.append(score)
                if score < best:
                    best, best_state = score, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    report = ValueReport(
        positions=len(x_tr) + len(x_te),
        learned=best,
        heuristic=brier(np.tanh(h_te / scale), y_te),
        heuristic_scale=scale,
        coin_flip=brier(np.zeros_like(y_te), y_te),
        curve=curve,
    )
    return net, report
