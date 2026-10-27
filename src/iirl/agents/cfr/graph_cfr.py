"""Vectorized CFR+ over a game's state graph: exact equilibria for games with millions of states.

Tabular CFR (``cfr.py``) recurses over every path, which is hopeless beyond toy games. Here the
game is first enumerated once as a graph of distinct states (``eval.exact_br.build_graph``;
states numbered level by level), and every CFR+ iteration becomes a few numpy passes over it:

1. **Forward**, level by level: each player's and chance's reach, summed over all paths into a
   state (``bincount`` over the edges of a level).
2. **Backward**, level by level: player 0's expected value of every state under the current
   strategy (player 1's is its negative: zero-sum).
3. **Counterfactual values** per information-set action: the opponent-and-chance reach of the
   parent times the child's value, summed over the infoset's states.
4. **Regret matching+** (regrets floored at zero) and **linearly weighted averaging** (iteration
   ``t`` weighted by ``t``), which is CFR+ (Tammelin 2014). Updates are simultaneous by default;
   ``alternating=True`` updates player 0, then player 1 against player 0's new strategy, as CFR+
   was published (on Liar's Dice it reaches a given exploitability in ~3-10x fewer iterations).

Information sets are keyed by ``information_state_string``; the average strategy is what converges
to a Nash equilibrium, and ``exact_br.exploitability`` measures how close it is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from iirl.core.game import Game, State
from iirl.eval.exact_br import GameGraph, build_graph


class GraphCFR:
    """CFR+ on the full state graph of a two-player zero-sum game."""

    def __init__(
        self, game: Game, graph: GameGraph | None = None, alternating: bool = False
    ) -> None:
        """Enumerate the game (unless a graph is given) and index its edges and infosets."""
        self.game = game
        self.alternating = alternating
        g = graph or build_graph(game)
        self.graph = g
        n = len(g.states)
        self.n = n
        self.value0_terminal = np.zeros(n)
        self.is_terminal = np.zeros(n, dtype=bool)
        owner = np.full(n, -1)  # -1 chance/terminal, else the player to move
        self.infoset_keys: list[str] = []
        key_index: dict[str, int] = {}
        self.infoset_actions: list[list[int]] = []
        state_infoset = np.full(n, -1)
        for i, s in enumerate(g.states):
            if game.is_terminal(s):
                self.is_terminal[i] = True
                self.value0_terminal[i] = game.returns(s)[0]
            elif not game.is_chance_node(s):
                p = game.current_player(s)
                owner[i] = p
                key = game.information_state_string(s, p)
                j = key_index.get(key)
                if j is None:
                    j = len(self.infoset_keys)
                    key_index[key] = j
                    self.infoset_keys.append(key)
                    self.infoset_actions.append([a for a, _ in g.children[i]])
                state_infoset[i] = j
        self.key_index = key_index
        sizes = np.array([len(a) for a in self.infoset_actions])
        self.offset = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.num_slots = int(sizes.sum())
        self.slot_infoset = np.repeat(np.arange(len(sizes)), sizes)
        self.infoset_owner = np.zeros(len(sizes), dtype=int)

        parents, children, probs, slots, kinds = [], [], [], [], []
        for i, s in enumerate(g.states):
            if self.is_terminal[i]:
                continue
            if owner[i] < 0:
                chance = dict(game.chance_outcomes(s))
                for a, c in g.children[i]:
                    parents.append(i)
                    children.append(c)
                    probs.append(chance[a])
                    slots.append(-1)
                    kinds.append(-1)
            else:
                j = state_infoset[i]
                self.infoset_owner[j] = owner[i]
                for k, (_, c) in enumerate(g.children[i]):
                    parents.append(i)
                    children.append(c)
                    probs.append(1.0)
                    slots.append(self.offset[j] + k)
                    kinds.append(owner[i])
        self.parent = np.asarray(parents)
        self.child = np.asarray(children)
        self.chance_prob = np.asarray(probs)
        self.slot = np.asarray(slots)
        self.kind = np.asarray(kinds)
        # Edges grouped by the parent's level, and each level's state-id range.
        level_start = np.array([lvl[0] for lvl in g.by_depth] + [n])
        self.level_ranges = [(level_start[k], level_start[k + 1]) for k in range(len(g.by_depth))]
        edge_level = np.searchsorted(level_start, self.parent, side="right") - 1
        order = np.argsort(edge_level, kind="stable")
        for name in ("parent", "child", "chance_prob", "slot", "kind"):
            setattr(self, name, getattr(self, name)[order])
        edge_level = edge_level[order]
        self.edge_bounds = np.searchsorted(edge_level, np.arange(len(g.by_depth) + 1))

        self.regret = np.zeros(self.num_slots)
        self.strategy_sum = np.zeros(self.num_slots)
        self.t = 0

    def _current(self) -> np.ndarray:
        """Regret-matching strategy per slot (uniform where an infoset has no positive regret)."""
        pos = np.maximum(self.regret, 0.0)
        total = np.bincount(self.slot_infoset, weights=pos, minlength=len(self.offset))
        count = np.bincount(self.slot_infoset, minlength=len(self.offset))
        denom = total[self.slot_infoset]
        uniform = 1.0 / count[self.slot_infoset]
        return np.where(denom > 0, pos / np.where(denom > 0, denom, 1.0), uniform)

    def iterate(self, iterations: int = 1) -> None:
        """Run CFR+ iterations (simultaneous, or alternating if set at construction)."""
        for _ in range(iterations):
            self.t += 1
            if self.alternating:
                self._update((0,))
                self._update((1,))
            else:
                self._update((0, 1))

    def _update(self, players: tuple[int, ...]) -> None:
        """One pass under the current strategy, updating ``players``' regrets and averages."""
        sigma = self._current()
        player_edge = self.kind >= 0
        edge_sigma = np.where(player_edge, sigma[np.maximum(self.slot, 0)], 1.0)
        # Edge weights for each reach vector: [player 0 only, player 1 only, chance only].
        w0 = np.where(self.kind == 0, edge_sigma, 1.0)
        w1 = np.where(self.kind == 1, edge_sigma, 1.0)
        wc = np.where(self.kind == -1, self.chance_prob, 1.0)
        reach = np.zeros((self.n, 3))
        reach[0] = 1.0
        for lvl, (lo_e, hi_e) in enumerate(
            zip(self.edge_bounds[:-1], self.edge_bounds[1:], strict=True)
        ):
            if lo_e == hi_e or lvl + 1 >= len(self.level_ranges):
                continue
            lo, hi = self.level_ranges[lvl + 1]
            par, ch = self.parent[lo_e:hi_e], self.child[lo_e:hi_e] - lo
            for k, w in enumerate((w0, w1, wc)):
                reach[lo:hi, k] += np.bincount(
                    ch, weights=reach[par, k] * w[lo_e:hi_e], minlength=hi - lo
                )
        # Backward: player 0's value of every state under sigma.
        value = self.value0_terminal.copy()
        edge_w = np.where(player_edge, edge_sigma, self.chance_prob)
        for lvl in range(len(self.level_ranges) - 1, -1, -1):
            lo_e, hi_e = self.edge_bounds[lvl], self.edge_bounds[lvl + 1]
            if lo_e == hi_e:
                continue
            lo, hi = self.level_ranges[lvl]
            par = self.parent[lo_e:hi_e]
            contrib = np.bincount(
                par - lo,
                weights=edge_w[lo_e:hi_e] * value[self.child[lo_e:hi_e]],
                minlength=hi - lo,
            )
            nonterminal = ~self.is_terminal[lo:hi]
            value[lo:hi] = np.where(nonterminal, contrib, value[lo:hi])
        # Counterfactual values per slot, regrets, and the average strategy.
        cf_reach = np.where(
            self.kind == 0,
            reach[self.parent, 1] * reach[self.parent, 2],
            reach[self.parent, 0] * reach[self.parent, 2],
        )
        sign = np.where(self.kind == 0, 1.0, -1.0)
        cfv_edge = np.where(player_edge, cf_reach * sign * value[self.child], 0.0)
        slot_idx = np.maximum(self.slot, 0)
        cfv = np.bincount(slot_idx, weights=cfv_edge, minlength=self.num_slots)
        node = np.bincount(self.slot_infoset, weights=sigma * cfv, minlength=len(self.offset))
        mine = np.isin(self.infoset_owner[self.slot_infoset], players)
        self.regret = np.where(
            mine, np.maximum(self.regret + cfv - node[self.slot_infoset], 0.0), self.regret
        )
        own_reach = np.where(self.kind == 0, reach[self.parent, 0], reach[self.parent, 1])
        weight = np.where(player_edge, own_reach * edge_sigma, 0.0)
        self.strategy_sum += self.t * np.where(
            mine, np.bincount(slot_idx, weights=weight, minlength=self.num_slots), 0.0
        )
        self.root_value = float(value[0])

    def average_policy(self):  # noqa: ANN201 - a BatchPolicy
        """Return the average strategy as a batched policy (for ``exact_br.exploitability``)."""
        total = np.bincount(
            self.slot_infoset, weights=self.strategy_sum, minlength=len(self.offset)
        )
        count = np.bincount(self.slot_infoset, minlength=len(self.offset))
        avg = np.where(
            total[self.slot_infoset] > 0,
            self.strategy_sum
            / np.where(total[self.slot_infoset] > 0, total[self.slot_infoset], 1.0),
            1.0 / count[self.slot_infoset],
        )

        def policy(game: Game, states: Sequence[State]) -> list[np.ndarray]:
            out = []
            for s in states:
                j = self.key_index[game.information_state_string(s, game.current_player(s))]
                probs = np.zeros(game.num_distinct_actions())
                for k, a in enumerate(self.infoset_actions[j]):
                    probs[a] = avg[self.offset[j] + k]
                out.append(probs)
            return out

        return policy
