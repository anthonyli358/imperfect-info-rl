"""Full-width ReBeL (Brown et al. 2020) for games small enough to enumerate.

ReBeL plays an imperfect-information game by search over **public belief states** (PBS): the
public state plus, for each player, the probability of each private state given the policy played
so far. At a PBS it solves a depth-limited subgame with CFR; where the subgame is cut off, a value
network gives every private state's expected value *as a function of the beliefs at the leaf*,
which change on every CFR iteration. The network is trained on the root values of subgames solved
in self-play, so values bootstrap from the end of the game upward.

This implementation is "full width": the game's state graph is enumerated once
(``eval.exact_br.build_graph``) and cut into slabs at ``boundaries`` (depths where a subgame
starts). A slab's subgames, one per public state at its top, are solved together, each weighted by
its reach, which is equivalent to solving them one at a time because CFR's regret matching ignores
an information set's scale. Every public state is visited on every pass, rather than a sampled
trajectory of them. That is only affordable on a small game, and it is what lets the policy be
checked against exact exploitability.

**The policy ReBeL plays** is not "solve each subgame from the beliefs of the average strategy
above it": that is unsafe re-solving, and it is exploitable (on Kuhn poker it stalls at 0.085,
because a subgame's equilibrium given fixed beliefs need not be part of a full-game equilibrium).
ReBeL instead stops the subgame's CFR at a *random iteration*, plays that iteration's strategy,
and solves the next subgame from that iteration's beliefs. In expectation over the stopping
iterations this is the average strategy of CFR on the whole game, which is what converges.
:meth:`FullWidthReBeL.evaluate` builds that expected policy explicitly, branching into the next
slab from every iteration (or a weighted sample of them), so its exploitability can be measured.

Bookkeeping, per state ``i`` in a slab and player ``p``:

* ``reach[i, p]``: the product of everyone's probabilities *except p's* on the way to ``i``
  (opponent and chance), which weights ``p``'s counterfactual values. For a fixed public state and
  hand of the opponent it is that hand's unnormalised range, so the beliefs fed to the network
  are read off it.
* ``value[i, p]``: ``p``'s expected payoff from ``i``. At a leaf it is the network's value of
  ``p``'s private state given the leaf's beliefs; it is not zero-sum state by state, and need not
  be: only its reach-weighted sum over an information set enters a regret.

The game must provide ``public_state_string``, ``public_state_tensor`` (from the mover's point of
view), ``private_state_index`` and ``num_private_states``, valid at the boundary depths.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from iirl.agents.cfr.graph_cfr import GraphCFR
from iirl.core.game import Game
from iirl.eval.exact_br import exploitability


@dataclass(frozen=True)
class ReBeLConfig:
    """Settings for :class:`FullWidthReBeL`.

    ``boundaries`` are the depths where subgames start, ending with the game's last depth; a
    value network is trained for every boundary but the first and the last. ``passes`` self-play
    passes are made per epoch; only the first includes the last slab (usually by far the largest,
    and it needs no network), so the upper boundaries see ``passes`` belief samples per epoch.
    A boundary's network starts training ``warmup_epochs`` after the one below it.

    ``eval_samples[k]`` is how many of slab ``k``'s iterations the evaluated policy branches from
    (drawn in proportion to their weight); ``None`` branches from every iteration, which is exact
    but costs a solve of the next slab per iteration.
    """

    boundaries: tuple[int, ...]
    cfr_iterations: int = 100
    eval_iterations: int = 100
    eval_samples: tuple[int | None, ...] | None = None
    epochs: int = 30
    passes: int = 4
    explore: float = 0.1
    warmup_epochs: int = 2
    buffer_epochs: int = 8
    hidden: int = 256
    layers: int = 3
    lr: float = 1e-3
    batch_size: int = 512
    steps_per_epoch: int = 400
    value_scale: float = 5.0


class PublicLevel:
    """The public states at one boundary depth, and each state's place in them."""

    def __init__(self, game: Game, cfr: GraphCFR, depth: int) -> None:
        """Group the depth's non-terminal states by public state and private-state indices."""
        lo, hi = cfr.level_ranges[depth]
        self.lo, self.hi = int(lo), int(hi)
        self.num_private = game.num_private_states()
        keys: dict[str, int] = {}
        idx, pub, hand, feats, movers = [], [], [], [], []
        for k, s in enumerate(cfr.graph.states[self.lo : self.hi]):
            if game.is_terminal(s):
                continue
            if game.is_chance_node(s):
                raise ValueError(f"depth {depth} has a chance node; a boundary needs decisions")
            key = game.public_state_string(s)
            j = keys.get(key)
            if j is None:
                j = keys[key] = len(keys)
                feats.append(game.public_state_tensor(s))
                movers.append(game.current_player(s))
            idx.append(k)
            pub.append(j)
            hand.append((game.private_state_index(s, 0), game.private_state_index(s, 1)))
        self.idx = np.asarray(idx)  # positions within the level of its non-terminal states
        self.pub = np.asarray(pub)
        self.hand = np.asarray(hand).reshape(-1, 2)
        self.num_public = len(keys)
        self.features = np.asarray(feats, dtype=np.float32)
        self.first = np.asarray(movers) == 0  # player 0 moves at this public state
        size = self.num_public * self.num_private
        self.group = [self.pub * self.num_private + self.hand[:, p] for p in (0, 1)]
        self.count = [np.bincount(g, minlength=size) for g in self.group]

    def _table(self, player: int, weights: np.ndarray) -> np.ndarray:
        """Sum ``weights`` over states into a (public state, ``player``'s private state) table."""
        size = self.num_public * self.num_private
        table = np.bincount(self.group[player], weights=weights, minlength=size)
        return table.reshape(self.num_public, self.num_private)

    def ranges(self, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each player's normalised range per public state, and whether it is defined.

        ``reach`` is the level's (states, 2) array. Player ``o``'s range is read from the other
        player's column; where it has no mass, it falls back to uniform over possible hands.
        """
        out = np.zeros((2, self.num_public, self.num_private))
        defined = np.zeros((2, self.num_public), dtype=bool)
        for o in (0, 1):
            count = self.count[o].reshape(self.num_public, self.num_private)
            mean = self._table(o, reach[self.idx, 1 - o]) / np.maximum(count, 1)
            total = mean.sum(axis=1, keepdims=True)
            possible = (count > 0) / np.maximum((count > 0).sum(axis=1, keepdims=True), 1)
            defined[o] = total[:, 0] > 0
            out[o] = np.where(total > 0, mean / np.where(total > 0, total, 1.0), possible)
        return out, defined

    def _mover_order(self, per_player: np.ndarray) -> np.ndarray:
        """Reorder a (2, public, private) array as (public, [mover, other], private)."""
        first = self.first[:, None]
        mover = np.where(first, per_player[0], per_player[1])
        other = np.where(first, per_player[1], per_player[0])
        return np.stack([mover, other], axis=1)

    def net_input(self, ranges: np.ndarray) -> np.ndarray:
        """Public features, then the mover's range, then the other player's."""
        r = self._mover_order(ranges).reshape(self.num_public, -1)
        return np.concatenate([self.features, r.astype(np.float32)], axis=1)

    def values(self, net: nn.Module, reach: np.ndarray, scale: float) -> np.ndarray:
        """Network values of both players' private states, per non-terminal state: (states, 2)."""
        ranges, _ = self.ranges(reach)
        with torch.no_grad():
            out = net(torch.as_tensor(self.net_input(ranges))).numpy().astype(float) * scale
        out = out.reshape(self.num_public, 2, self.num_private)
        first = self.first[:, None]
        per_player = (np.where(first, out[:, 0], out[:, 1]), np.where(first, out[:, 1], out[:, 0]))
        return np.stack([per_player[p][self.pub, self.hand[:, p]] for p in (0, 1)], axis=1)

    def root_values(self, reach: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each private state's expected value given the other's range: (2, public, private).

        Also returns where that value is defined (the opponent's range has mass there).
        """
        out = np.zeros((2, self.num_public, self.num_private))
        mask = np.zeros_like(out, dtype=bool)
        for p in (0, 1):
            weight = reach[self.idx, p]
            den = self._table(p, weight)
            num = self._table(p, weight * value[self.idx, p])
            mask[p] = den > 0
            out[p] = np.where(mask[p], num / np.where(mask[p], den, 1.0), 0.0)
        return out, mask

    def range_spread(self, reach: np.ndarray) -> float:
        """Largest relative disagreement between states that should share a range entry.

        The ranges assume ``reach[i, p]`` depends only on the public state and the opponent's
        hand; this measures how far that is from true (0 when it holds exactly).
        """
        worst = 0.0
        size = self.num_public * self.num_private
        for o in (0, 1):
            values = reach[self.idx, 1 - o]
            high = np.zeros(size)
            low = np.full(size, np.inf)
            np.maximum.at(high, self.group[o], values)
            np.minimum.at(low, self.group[o], values)
            seen = high > 0
            if seen.any():
                worst = max(worst, float(((high[seen] - low[seen]) / high[seen]).max()))
        return worst


def _mlp(n_in: int, n_out: int, hidden: int, layers: int) -> nn.Sequential:
    blocks: list[nn.Module] = []
    width = n_in
    for _ in range(layers):
        blocks += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    return nn.Sequential(*blocks, nn.Linear(width, n_out))


class FullWidthReBeL:
    """Depth-limited CFR over public belief states with learned leaf values, on the full graph."""

    def __init__(
        self, game: Game, config: ReBeLConfig, cfr: GraphCFR | None = None, seed: int = 0
    ) -> None:
        """Index the boundaries' public states and create one value network per inner boundary."""
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.game, self.config = game, config
        self.cfr = cfr or GraphCFR(game)
        c, b = self.cfr, config.boundaries
        if b[-1] != len(c.level_ranges) - 1 or list(b) != sorted(set(b)):
            raise ValueError("boundaries must increase and end at the game's last depth")
        self.levels = {d: PublicLevel(game, c, d) for d in b[:-1]}
        self.nets: dict[int, nn.Sequential] = {}
        self.optimizers: dict[int, torch.optim.Optimizer] = {}
        self.buffers: dict[int, deque] = {}
        for d in b[1:-1]:
            level = self.levels[d]
            n_in = level.features.shape[1] + 2 * level.num_private
            self.nets[d] = _mlp(n_in, 2 * level.num_private, config.hidden, config.layers)
            self.optimizers[d] = torch.optim.Adam(self.nets[d].parameters(), lr=config.lr)
            per_epoch = 1 if d == b[-2] else config.passes
            self.buffers[d] = deque(maxlen=config.buffer_epochs * per_epoch)
        self.slabs = [self._slab(b[k], b[k + 1]) for k in range(len(b) - 1)]
        # Per state: reach of all but player 0, all but player 1, player 0's own, player 1's own.
        self._reach = np.zeros((c.n, 4))
        self._value = np.zeros((c.n, 2))
        self.root_reach = self._chance_reach(b[0])
        self.epoch = 0

    def _slab(self, a: int, b: int) -> SimpleNamespace:
        """Index the edges, action slots and information sets between depths ``a`` and ``b``."""
        c = self.cfr
        e0, e1 = int(c.edge_bounds[a]), int(c.edge_bounds[b])
        kind = c.kind[e0:e1]
        acting = kind >= 0
        slots = c.slot[e0:e1][acting]
        s0, s1 = int(slots.min()), int(slots.max()) + 1
        i0, i1 = int(c.slot_infoset[s0]), int(c.slot_infoset[s1 - 1]) + 1
        infoset = c.slot_infoset[s0:s1] - i0
        return SimpleNamespace(
            a=a,
            b=b,
            e0=e0,
            parent=c.parent[e0:e1],
            child=c.child[e0:e1],
            kind=kind,
            acting=acting,
            who=np.maximum(kind, 0),
            slot=np.where(acting, c.slot[e0:e1] - s0, 0),
            chance=c.chance_prob[e0:e1],
            s0=s0,
            s1=s1,
            infoset=infoset,
            num_infosets=i1 - i0,
            uniform=1.0 / np.bincount(infoset, minlength=i1 - i0)[infoset],
        )

    def _chance_reach(self, depth: int) -> np.ndarray:
        """Root reaches: chance's probability of each state at ``depth``, and own reaches of 1."""
        c = self.cfr
        reach = np.zeros(c.n)
        reach[0] = 1.0
        e0, e1 = c.edge_bounds[0], c.edge_bounds[depth]
        if (c.kind[e0:e1] != -1).any():
            raise ValueError("a player acts before the first boundary")
        for lvl in range(depth):
            lo_e, hi_e = c.edge_bounds[lvl], c.edge_bounds[lvl + 1]
            np.add.at(
                reach, c.child[lo_e:hi_e], reach[c.parent[lo_e:hi_e]] * c.chance_prob[lo_e:hi_e]
            )
        lo, hi = c.level_ranges[depth]
        ones = np.ones(hi - lo)
        return np.stack([reach[lo:hi], reach[lo:hi], ones, ones], axis=1)

    def _current(self, slab: SimpleNamespace) -> np.ndarray:
        """The slab's regret-matching strategy per slot (uniform with no positive regret)."""
        pos = self.cfr.regret[slab.s0 : slab.s1]
        total = np.bincount(slab.infoset, weights=pos, minlength=slab.num_infosets)[slab.infoset]
        return np.where(total > 0, pos / np.where(total > 0, total, 1.0), slab.uniform)

    def _forward(
        self, slab: SimpleNamespace, sigma: np.ndarray, root: np.ndarray, own: bool = True
    ) -> np.ndarray:
        """Push reaches from the slab's top to its bottom under ``sigma``; returns edge sigma.

        ``own`` False skips the players' own reaches, which only building a policy needs.
        """
        c, reach = self.cfr, self._reach
        reach[c.level_ranges[slab.a][0] : c.level_ranges[slab.b][1]] = 0.0
        lo, hi = c.level_ranges[slab.a]
        reach[lo:hi] = root
        edge_sigma = np.where(slab.acting, sigma[slab.slot], 1.0)
        others = np.where(slab.kind == -1, slab.chance, 1.0)
        weights = (
            np.where(slab.kind == 1, edge_sigma, others),
            np.where(slab.kind == 0, edge_sigma, others),
            np.where(slab.kind == 0, edge_sigma, 1.0),
            np.where(slab.kind == 1, edge_sigma, 1.0),
        )
        for lvl in range(slab.a, slab.b):
            lo_e, hi_e = c.edge_bounds[lvl] - slab.e0, c.edge_bounds[lvl + 1] - slab.e0
            if lo_e == hi_e:
                continue
            lo, hi = c.level_ranges[lvl + 1]
            par, ch = slab.parent[lo_e:hi_e], slab.child[lo_e:hi_e] - lo
            for k, w in enumerate(weights if own else weights[:2]):
                reach[lo:hi, k] += np.bincount(
                    ch, weights=reach[par, k] * w[lo_e:hi_e], minlength=hi - lo
                )
        return edge_sigma

    def _backward(self, slab: SimpleNamespace, edge_sigma: np.ndarray) -> None:
        """Fill both players' state values from the slab's bottom (network or terminal) up."""
        c, value = self.cfr, self._value
        lo_s, hi_s = c.level_ranges[slab.a][0], c.level_ranges[slab.b][1]
        value[lo_s:hi_s, 0] = c.value0_terminal[lo_s:hi_s]
        value[lo_s:hi_s, 1] = -c.value0_terminal[lo_s:hi_s]
        if slab.b in self.nets:
            leaf = self.levels[slab.b]
            value[leaf.lo + leaf.idx] = leaf.values(
                self.nets[slab.b], self._reach[leaf.lo : leaf.hi], self.config.value_scale
            )
        edge_w = np.where(slab.acting, edge_sigma, slab.chance)
        for lvl in range(slab.b - 1, slab.a - 1, -1):
            lo_e, hi_e = c.edge_bounds[lvl] - slab.e0, c.edge_bounds[lvl + 1] - slab.e0
            if lo_e == hi_e:
                continue
            lo, hi = c.level_ranges[lvl]
            par, ch = slab.parent[lo_e:hi_e] - lo, slab.child[lo_e:hi_e]
            live = ~c.is_terminal[lo:hi]
            for p in (0, 1):
                up = np.bincount(par, weights=edge_w[lo_e:hi_e] * value[ch, p], minlength=hi - lo)
                value[lo:hi, p] = np.where(live, up, value[lo:hi, p])

    def _solve(
        self,
        slab: SimpleNamespace,
        root: np.ndarray,
        iterations: int,
        weight: float = 0.0,
        picks: dict[int, float] | None = None,
        branch: Callable[[np.ndarray, float], None] | None = None,
        explore: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CFR+ on one slab from the root reaches ``root`` (a (states, 4) array).

        Returns the root private-state values, averaged over iterations with the strategy's
        linear weights, and where they are defined. With ``weight``, each iteration's strategy is
        added to the policy being built (``cfr.strategy_sum``), weighted by ``weight``, the
        iteration's weight and the own reach carried in ``root``. After each iteration ``t`` in
        ``picks``, ``branch`` is called with the reaches at the slab's bottom under that
        iteration's strategy (mixed with ``explore`` of uniform) and ``picks[t]``.
        """
        c, level = self.cfr, self.levels[slab.a]
        c.regret[slab.s0 : slab.s1] = 0.0
        n_slots = slab.s1 - slab.s0
        lo, hi = c.level_ranges[slab.a]
        lo_b, hi_b = c.level_ranges[slab.b]
        norm = iterations * (iterations + 1) / 2
        total = np.zeros((2, level.num_public, level.num_private))
        mask = np.zeros_like(total, dtype=bool)
        reach, value = self._reach, self._value
        for t in range(1, iterations + 1):
            sigma = self._current(slab)
            edge_sigma = self._forward(slab, sigma, root, own=bool(weight))
            self._backward(slab, edge_sigma)
            cfv = np.bincount(
                slab.slot,
                weights=np.where(
                    slab.acting, reach[slab.parent, slab.who] * value[slab.child, slab.who], 0.0
                ),
                minlength=n_slots,
            )
            node = np.bincount(slab.infoset, weights=sigma * cfv, minlength=slab.num_infosets)
            c.regret[slab.s0 : slab.s1] = np.maximum(
                c.regret[slab.s0 : slab.s1] + cfv - node[slab.infoset], 0.0
            )
            if weight:
                own = np.where(slab.acting, reach[slab.parent, 2 + slab.who] * edge_sigma, 0.0)
                c.strategy_sum[slab.s0 : slab.s1] += (weight * t / norm) * np.bincount(
                    slab.slot, weights=own, minlength=n_slots
                )
            values, mask = level.root_values(reach[lo:hi, :2], value[lo:hi])
            total += t * values
            if picks and t in picks:
                if explore:
                    mixed = (1.0 - explore) * sigma + explore * slab.uniform
                    self._forward(slab, mixed, root, own=bool(weight))
                branch(reach[lo_b:hi_b].copy(), picks[t])
        return total / norm, mask

    def selfplay(self, last_slab: bool = True) -> dict[int, tuple[np.ndarray, ...]]:
        """One self-play pass down the slabs; returns training data per inner boundary.

        Each slab is solved from the beliefs left by a random iteration of the slab above (drawn
        in proportion to the iteration's weight, with ``explore`` of uniform play mixed in so
        that every public state keeps some reach).
        """
        cfg = self.config
        steps = np.arange(1, cfg.cfr_iterations + 1)
        root, data = self.root_reach, {}
        for slab in self.slabs if last_slab else self.slabs[:-1]:
            stop = int(self.rng.choice(steps, p=steps / steps.sum()))
            leaves: list[np.ndarray] = []
            values, mask = self._solve(
                slab,
                root,
                cfg.cfr_iterations,
                picks={stop: 1.0},
                branch=lambda leaf, _: leaves.append(leaf),  # noqa: B023 - called inside _solve
                explore=cfg.explore,
            )
            if slab.a in self.nets:
                level = self.levels[slab.a]
                ranges, defined = level.ranges(root)
                rows = defined.all(axis=0)
                data[slab.a] = (
                    level.net_input(ranges)[rows],
                    (level._mover_order(values)[rows] / cfg.value_scale).astype(np.float32),
                    level._mover_order(mask)[rows],
                )
            root = leaves[0]
        return data

    def fit(self) -> dict[int, float]:
        """Train each network on its buffer; returns the last minibatches' mean loss."""
        cfg, losses = self.config, {}
        for d, buffer in self.buffers.items():
            if not buffer:
                continue
            x, y, m = (torch.as_tensor(np.concatenate(part)) for part in zip(*buffer, strict=True))
            y, m = y.flatten(1), m.flatten(1).float()
            net, opt = self.nets[d], self.optimizers[d]
            recent = []
            for step in range(cfg.steps_per_epoch):
                rows = torch.randint(len(x), (cfg.batch_size,))
                loss = (((net(x[rows]) - y[rows]) ** 2) * m[rows]).sum() / m[rows].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if step >= cfg.steps_per_epoch - 50:
                    recent.append(loss.item())
            losses[d] = float(np.mean(recent))
        return losses

    def train(
        self, on_epoch: Callable[[FullWidthReBeL, dict[int, float]], None] | None = None
    ) -> None:
        """Self-play epochs from ``self.epoch`` to ``config.epochs`` (so a run can be resumed)."""
        cfg = self.config
        inner = cfg.boundaries[1:-1]
        while self.epoch < cfg.epochs:
            for k in range(cfg.passes):
                for d, rows in self.selfplay(last_slab=k == 0).items():
                    # A boundary's targets are only as good as the network below it.
                    if self.epoch >= (len(inner) - 1 - inner.index(d)) * cfg.warmup_epochs:
                        self.buffers[d].append(rows)
            losses = self.fit()
            self.epoch += 1
            if on_epoch is not None:
                on_epoch(self, losses)

    def _play(self, k: int, root: np.ndarray, weight: float, iterations: int) -> None:
        """Add slab ``k``'s share of the expected policy, then branch into the slabs below."""
        samples = self.config.eval_samples
        picks = None
        if k < len(self.slabs) - 1:
            steps = np.arange(1, iterations + 1)
            share = steps / steps.sum()
            n = samples[k] if samples else None
            if n is None or n >= iterations:
                picks = dict(zip(steps.tolist(), share.tolist(), strict=True))
            else:
                drawn, counts = np.unique(
                    self.rng.choice(steps, size=n, p=share), return_counts=True
                )
                picks = {int(t): m / n for t, m in zip(drawn, counts, strict=True)}
        self._solve(
            self.slabs[k],
            root,
            iterations,
            weight=weight,
            picks=picks,
            branch=lambda leaf, share: self._play(k + 1, leaf, weight * share, iterations),
        )

    def build_policy(self, iterations: int | None = None):  # noqa: ANN201 - a BatchPolicy
        """Build the policy ReBeL plays in expectation over its random stopping iterations."""
        self.cfr.strategy_sum[:] = 0.0
        self._play(0, self.root_reach, 1.0, iterations or self.config.eval_iterations)
        return self.cfr.average_policy()

    def evaluate(self, iterations: int | None = None) -> dict[str, float]:
        """Exact exploitability of the policy from :meth:`build_policy`."""
        return exploitability(self.game, self.build_policy(iterations), self.cfr.graph)

    def range_spread(self) -> float:
        """Check the range bookkeeping under a random policy (see ``PublicLevel.range_spread``)."""
        root, worst = self.root_reach, 0.0
        for slab in self.slabs[:-1]:
            sigma = self.rng.random(slab.s1 - slab.s0)
            total = np.bincount(slab.infoset, weights=sigma, minlength=slab.num_infosets)
            self._forward(slab, sigma / total[slab.infoset], root)
            lo, hi = self.cfr.level_ranges[slab.b]
            root = self._reach[lo:hi].copy()
            worst = max(worst, self.levels[slab.b].range_spread(root))
        return worst
