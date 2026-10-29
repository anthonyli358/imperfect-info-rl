"""Deep CFR (Brown, Lerer, Gross & Sandholm, 2019): CFR with the regret tables replaced by networks.

Tabular CFR keeps a regret per information set and action; that stops scaling once there are too
many information sets to store. Deep CFR learns them instead. Each iteration ``t``, for each player
``p``:

1. **Traverse** with external sampling: at ``p``'s nodes try every legal action, at the
   opponent's sample one from the opponent's current strategy, at chance sample one outcome. At
   each of ``p``'s nodes the sampled *advantages* (action value minus the node's value under the
   current strategy) go into ``p``'s advantage memory, tagged with ``t``; at each opponent node
   the opponent's strategy goes into the strategy memory.
2. **Train** ``p``'s advantage network from scratch on its whole memory, each sample weighted by
   ``t`` (linear CFR: later iterations count more). The current strategy is regret matching on
   its predictions: proportional to the positive advantages, or the best action if none is
   positive.

The memories are reservoirs (uniform over everything ever added, at a fixed size). At the end a
strategy network is trained on the strategy memory (weighted by ``t``); it approximates CFR's
*average* strategy, the one that converges to equilibrium. Any two-player zero-sum game with
``information_state_tensor`` and ``information_state_string`` works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from iirl.core.game import Game, State


@dataclass
class DeepCFRConfig:
    """Deep CFR settings (defaults sized for Leduc poker)."""

    iterations: int = 100
    traversals: int = 300  # per player per iteration
    hidden: tuple[int, ...] = (128, 128)
    advantage_steps: int = 300  # SGD steps per advantage network per iteration (from scratch)
    strategy_steps: int = 3000
    batch_size: int = 1024
    lr: float = 1e-3
    memory: int = 1_000_000  # rows per reservoir
    seed: int = 0


def _mlp(inputs: int, outputs: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(inputs, h), nn.ReLU()]
        inputs = h
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class Reservoir:
    """A fixed-size uniform sample of every row ever added (features, target, mask, iteration)."""

    def __init__(self, capacity: int, dim: int, actions: int, rng: np.random.Generator) -> None:
        """Preallocate ``capacity`` rows."""
        self.x = np.zeros((capacity, dim), np.float32)
        self.y = np.zeros((capacity, actions), np.float32)
        self.mask = np.zeros((capacity, actions), np.float32)
        self.t = np.zeros(capacity, np.float32)
        self.capacity, self.seen, self.rng = capacity, 0, rng

    def __len__(self) -> int:
        """Rows stored."""
        return min(self.seen, self.capacity)

    def add(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, t: int) -> None:
        """Add a row; once full, replace a random one with probability ``capacity / seen``."""
        if self.seen < self.capacity:
            i = self.seen
        else:
            i = int(self.rng.integers(self.seen + 1))
            if i >= self.capacity:
                self.seen += 1
                return
        self.x[i], self.y[i], self.mask[i], self.t[i] = x, y, mask, t
        self.seen += 1

    def batch(self, size: int) -> tuple[torch.Tensor, ...]:
        """A uniform minibatch of stored rows."""
        idx = self.rng.integers(len(self), size=size)
        return tuple(torch.as_tensor(a[idx]) for a in (self.x, self.y, self.mask, self.t))


def regret_matching(advantages: np.ndarray, legal: list[int]) -> np.ndarray:
    """Proportional to the positive advantages over ``legal``; the best action if none is."""
    probs = np.zeros(len(advantages))
    pos = np.maximum(advantages[legal], 0.0)
    if pos.sum() > 0:
        probs[legal] = pos / pos.sum()
    else:
        probs[legal[int(np.argmax(advantages[legal]))]] = 1.0
    return probs


class DeepCFR:
    """The solver: call :meth:`iterate`, then :meth:`average_policy`."""

    def __init__(self, game: Game, config: DeepCFRConfig) -> None:
        """Set up memories and (untrained) advantage networks for ``game``."""
        self.game, self.config = game, config
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        s = game.new_initial_state()
        while game.is_chance_node(s):
            s = game.apply_action(s, game.chance_outcomes(s)[0][0])
        self.dim = len(game.information_state_tensor(s, game.current_player(s)))
        self.actions = game.num_distinct_actions()
        self.adv_mem = [
            Reservoir(config.memory, self.dim, self.actions, self.rng) for _ in range(2)
        ]
        self.strat_mem = Reservoir(config.memory, self.dim, self.actions, self.rng)
        self.adv_nets = [_mlp(self.dim, self.actions, config.hidden) for _ in range(2)]
        self.strategy_net: nn.Sequential | None = None
        self.t = 0
        self._cache: dict[str, np.ndarray] = {}

    # --- Traversal ------------------------------------------------------------------------------
    def _strategy(self, s: State, player: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """The current (regret-matching) strategy at ``s`` for its mover, cached per iteration."""
        game = self.game
        key = game.information_state_string(s, player)
        feat = np.asarray(game.information_state_tensor(s, player), np.float32)
        legal = game.legal_actions(s)
        probs = self._cache.get(key)
        if probs is None:
            if self.t <= 1:
                probs = np.zeros(self.actions)
                probs[legal] = 1 / len(legal)  # no regrets yet: uniform
            else:
                with torch.no_grad():
                    adv = self.adv_nets[player](torch.as_tensor(feat)[None])[0].numpy()
                probs = regret_matching(adv, legal)
            self._cache[key] = probs
        return probs, feat, legal

    def _traverse(self, s: State, p: int) -> float:
        game = self.game
        if game.is_terminal(s):
            return float(game.returns(s)[p])
        if game.is_chance_node(s):
            outcomes = game.chance_outcomes(s)
            i = self.rng.choice(len(outcomes), p=[q for _, q in outcomes])
            return self._traverse(game.apply_action(s, outcomes[i][0]), p)
        mover = game.current_player(s)
        probs, feat, legal = self._strategy(s, mover)
        mask = np.zeros(self.actions, np.float32)
        mask[legal] = 1.0
        if mover == p:
            values = {a: self._traverse(game.apply_action(s, a), p) for a in legal}
            ev = sum(probs[a] * v for a, v in values.items())
            adv = np.zeros(self.actions, np.float32)
            for a, v in values.items():
                adv[a] = v - ev
            self.adv_mem[p].add(feat, adv, mask, self.t)
            return ev
        self.strat_mem.add(feat, probs.astype(np.float32), mask, self.t)
        a = legal[self.rng.choice(len(legal), p=probs[legal] / probs[legal].sum())]
        return self._traverse(game.apply_action(s, a), p)

    # --- Training -------------------------------------------------------------------------------
    def _train_advantage(self, p: int) -> None:
        cfg = self.config
        net = _mlp(self.dim, self.actions, cfg.hidden)  # from scratch, as in the paper
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        for _ in range(cfg.advantage_steps):
            x, y, mask, t = self.adv_mem[p].batch(cfg.batch_size)
            w = t / t.mean()
            loss = (w * (((net(x) - y) * mask) ** 2).sum(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        self.adv_nets[p] = net

    def iterate(self, n: int = 1) -> None:
        """Run ``n`` Deep CFR iterations (traversals and advantage training for both players)."""
        for _ in range(n):
            self.t += 1
            for p in (0, 1):
                self._cache = {}
                for _ in range(self.config.traversals):
                    self._traverse(self.game.new_initial_state(), p)
                self._train_advantage(p)

    def train_strategy(self) -> nn.Sequential:
        """Fit the average-strategy network to the strategy memory (weighted by iteration)."""
        cfg = self.config
        net = _mlp(self.dim, self.actions, cfg.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        for _ in range(cfg.strategy_steps):
            x, y, mask, t = self.strat_mem.batch(cfg.batch_size)
            logits = net(x).masked_fill(mask == 0, -1e9)
            loss = -((t / t.mean()) * (y * torch.log_softmax(logits, dim=1)).sum(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        self.strategy_net = net
        return net

    def average_policy(self):  # noqa: ANN201 - a BatchPolicy (``eval.exact_br``)
        """The trained strategy network as a batched policy: masked softmax over legal actions."""
        net = self.strategy_net or self.train_strategy()

        def policy(game: Game, states) -> list[np.ndarray]:  # noqa: ANN001
            feats = np.stack(
                [game.information_state_tensor(s, game.current_player(s)) for s in states]
            ).astype(np.float32)
            with torch.no_grad():
                logits = net(torch.as_tensor(feats)).numpy()
            out = []
            for s, row in zip(states, logits, strict=True):
                legal = game.legal_actions(s)
                probs = np.zeros(game.num_distinct_actions())
                z = row[legal] - row[legal].max()
                probs[legal] = np.exp(z) / np.exp(z).sum()
                out.append(probs)
            return out

        return policy

    # --- Checkpoints ----------------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Everything needed to resume: iteration, memories, networks, random state."""
        mems = [*self.adv_mem, self.strat_mem]
        return {
            "t": self.t,
            "memories": [
                (m.x[: len(m)], m.y[: len(m)], m.mask[: len(m)], m.t[: len(m)], m.seen)
                for m in mems
            ],
            "adv_nets": [n.state_dict() for n in self.adv_nets],
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Resume from :meth:`state_dict`."""
        self.t = state["t"]
        for m, (x, y, mask, t, seen) in zip(
            [*self.adv_mem, self.strat_mem], state["memories"], strict=True
        ):
            n = len(x)
            m.x[:n], m.y[:n], m.mask[:n], m.t[:n], m.seen = x, y, mask, t, seen
        for net, sd in zip(self.adv_nets, state["adv_nets"], strict=True):
            net.load_state_dict(sd)
            net.eval()
        self.rng.bit_generator.state = state["rng"]
        for m in [*self.adv_mem, self.strat_mem]:
            m.rng = self.rng
