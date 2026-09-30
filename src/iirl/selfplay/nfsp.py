"""Neural Fictitious Self-Play (NFSP) with a PPO best-response learner.

NFSP (Heinrich & Silver, 2016) is a deep-RL version of fictitious play. Each player has:

* a **best response** ``beta``, trained by RL against the opponent's current strategy. Here it is
  PPO rather than the original DQN, since PPO copes far better with long, luck-heavy games; and
* an **average policy** ``pi``, trained by supervised learning to imitate what ``beta`` has done
  over time (a reservoir buffer of its decisions). ``pi`` is the agent that gets deployed.

The opponent ``beta`` trains against is the *anticipatory* mix: the average policy ``pi`` in a
``1 - eta`` share of games, and ``beta`` itself (self-play) in the rest. Chasing best responses
while playing the running average is fictitious play, which converges toward a Nash equilibrium in
two-player zero-sum games. The game is symmetric, so one ``beta``/``pi`` pair plays both seats.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn, optim

from iirl.agents.base import Agent
from iirl.core.game import Game
from iirl.nn.policy_value import PolicyValueNet
from iirl.selfplay.ppo import PPOConfig, evaluate_vs, train_ppo


@dataclass(frozen=True)
class NFSPConfig:
    """NFSP settings.

    Attributes:
        ppo: The best response's PPO settings (``iterations`` is the whole run's length).
        eta: Share of ``beta``'s games against itself; the rest are against ``pi``.
        buffer_size: Reservoir buffer capacity (decisions).
        sl_steps: Supervised minibatch steps for ``pi`` after each iteration.
        sl_batch: Supervised minibatch size.
        sl_lr: Supervised learning rate.
        eval_interval: Rate ``pi`` against the anchors every this many iterations.
        eval_games: Games per anchor evaluation.
    """

    ppo: PPOConfig = field(default_factory=PPOConfig)
    eta: float = 0.1
    buffer_size: int = 400_000
    sl_steps: int = 50
    sl_batch: int = 1024
    sl_lr: float = 1e-4
    eval_interval: int = 20
    eval_games: int = 400


class ReservoirBuffer:
    """Uniform sample of every decision ever added (reservoir sampling), as fixed arrays."""

    def __init__(self, capacity: int, feat_dim: int, num_actions: int, rng) -> None:  # noqa: ANN001
        """Allocate the buffer."""
        self.feats = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, num_actions), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.capacity, self.seen, self.rng = capacity, 0, rng

    def add(self, feats: np.ndarray, masks: np.ndarray, actions: np.ndarray) -> None:
        """Add a batch of decisions."""
        for f, m, a in zip(feats, masks, actions, strict=True):
            if self.seen < self.capacity:
                i = self.seen
            else:
                i = int(self.rng.integers(self.seen + 1))
                if i >= self.capacity:
                    self.seen += 1
                    continue
            self.feats[i], self.masks[i], self.actions[i] = f, m, a
            self.seen += 1

    def __len__(self) -> int:
        """Number of stored decisions."""
        return min(self.seen, self.capacity)


def train_nfsp(
    game: Game,
    anchors: Mapping[str, Agent],
    config: NFSPConfig,
    init_net: PolicyValueNet,
    on_eval: Callable[[int, dict[str, float]], None] | None = None,
    seed: int = 0,
) -> PolicyValueNet:
    """Run NFSP from ``init_net`` (both ``beta`` and ``pi`` start there); return ``pi``.

    ``on_eval(iteration, score_vs_anchor)`` receives ``pi``'s score against each anchor.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    avg = copy.deepcopy(init_net)
    avg.eval()
    opt = optim.Adam(avg.parameters(), lr=config.sl_lr)
    buffer = ReservoirBuffer(config.buffer_size, init_net.input_dim, init_net.num_actions, rng)

    def train_average() -> None:
        avg.train()
        n = len(buffer)
        for _ in range(config.sl_steps):
            idx = rng.integers(n, size=min(config.sl_batch, n))
            feats = torch.as_tensor(buffer.feats[idx])
            masks = torch.as_tensor(buffer.masks[idx])
            actions = torch.as_tensor(buffer.actions[idx])
            logits = avg(feats)[0].masked_fill(masks == 0, -1e9)
            loss = nn.functional.cross_entropy(logits, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
        avg.eval()

    def on_iteration(iteration: int, net: PolicyValueNet, batch: dict) -> list[PolicyValueNet]:
        buffer.add(batch["feat"], batch["mask"], batch["action"])  # beta's own decisions
        train_average()
        if on_eval is not None and iteration % config.eval_interval == 0:
            scores = {
                name: evaluate_vs(game, avg, anchor, config.eval_games, rng)[0]
                for name, anchor in anchors.items()
            }
            on_eval(iteration, scores)
        return [copy.deepcopy(avg)]

    ppo = replace(config.ppo, past_opponent_frac=1.0 - config.eta, opponent_greedy=False)
    train_ppo(
        game,
        anchors,
        ppo,
        init_net=init_net,
        opponents=[copy.deepcopy(avg)],
        on_iteration=on_iteration,
    )
    return avg
