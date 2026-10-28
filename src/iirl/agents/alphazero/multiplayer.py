"""PUCT search for perfect-information games with any number of players.

:mod:`iirl.agents.alphazero.mcts` backs up one scalar and flips its sign between the two players,
which only works for two-player zero-sum games. Here every edge keeps a value *per player*: a
simulation backs up the whole vector of returns (or leaf estimates), and each node selects by its
own mover's component. That is max-n search with PUCT exploration; for two zero-sum players it
reduces to the usual search.

Perfect information only (no determinization): the root state is the real state. Chance nodes are
sampled during descent. Leaves are evaluated in batches; virtual loss (on the mover's component)
spreads a batch's descents over different leaves.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.runner import _sample_chance
from iirl.core.types import Action
from iirl.nn.policy_value import NetEvaluator, PolicyValueNet


class MultiEvaluator(Protocol):
    """Priors for the mover and a value for every player, for a batch of player nodes."""

    def evaluate_batch(
        self, game: Game, states: list[State]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Return per-state priors and a ``(len(states), num_players)`` value array."""
        ...


class NetMultiEvaluator:
    """Priors from ``prior_net``; each player's value from ``value_net`` on that player's view.

    ``value_net`` is any net whose value head estimates the return of the player whose
    information state it reads (a PPO critic, or a value net trained on every player's view).
    """

    def __init__(self, prior_net: PolicyValueNet, value_net: PolicyValueNet) -> None:
        """Wrap the two nets for inference."""
        self.priors = NetEvaluator(prior_net)
        self.value_net = value_net
        value_net.eval()

    def evaluate_batch(
        self, game: Game, states: list[State]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Priors from the mover's view; one value-net pass over every (state, player) view."""
        priors, _ = self.priors.evaluate_batch(game, states)
        n = game.num_players
        feats = np.stack(
            [game.information_state_tensor(s, p) for s in states for p in range(n)]
        ).astype(np.float32)
        with torch.no_grad():
            values = self.value_net(torch.as_tensor(feats))[1].numpy()
        return priors, values.reshape(len(states), n)


class _Node:
    __slots__ = ("state", "to_move", "is_chance", "is_terminal", "expanded", "legal", "p", "n",
                 "w", "children", "leaf")  # fmt: skip

    def __init__(self, game: Game, state: State) -> None:
        self.state = state
        self.is_terminal = game.is_terminal(state)
        self.is_chance = (not self.is_terminal) and game.is_chance_node(state)
        self.to_move = -1 if (self.is_terminal or self.is_chance) else game.current_player(state)
        self.expanded = False
        self.legal: list[int] = []
        self.p: dict[int, float] = {}
        self.n: dict[int, float] = {}
        self.w: dict[int, np.ndarray] = {}
        self.children: dict[int, _Node] = {}
        self.leaf: np.ndarray | None = None


def _child(game: Game, node: _Node, action: int) -> _Node:
    child = node.children.get(action)
    if child is None:
        child = _Node(game, game.apply_action(node.state, action))
        node.children[action] = child
    return child


def _expand(game: Game, nodes: list[_Node], evaluator: MultiEvaluator) -> None:
    """Evaluate ``nodes`` in one batch: priors on their edges, the value vector as ``leaf``."""
    priors, values = evaluator.evaluate_batch(game, [x.state for x in nodes])
    for node, pr, v in zip(nodes, priors, values, strict=True):
        node.legal = game.legal_actions(node.state)
        node.p = {a: float(pr[a]) for a in node.legal}
        node.n = dict.fromkeys(node.legal, 0.0)
        node.w = {a: np.zeros(game.num_players) for a in node.legal}
        node.leaf = np.asarray(v, dtype=float)
        node.expanded = True


def _select(node: _Node, c_puct: float) -> int:
    explore = c_puct * np.sqrt(1.0 + sum(node.n.values()))
    me = node.to_move
    best_a, best_u = node.legal[0], -np.inf
    for a in node.legal:
        n = node.n[a]
        q = node.w[a][me] / n if n > 0 else 0.0
        u = q + explore * node.p[a] / (1 + n)
        if u > best_u:
            best_u, best_a = u, a
    return best_a


def multi_search(
    game: Game,
    state: State,
    evaluator: MultiEvaluator,
    simulations: int,
    rng: np.random.Generator,
    c_puct: float = 1.5,
    leaf_batch: int = 16,
    virtual_loss: float = 1.0,
) -> np.ndarray:
    """Return root visit counts over all action ids after ``simulations`` simulations."""
    root = _Node(game, state)
    _expand(game, [root], evaluator)
    done = 0
    while done < simulations:
        batch = min(leaf_batch, simulations - done)
        done += batch
        leaves = []
        for _ in range(batch):
            node, path = root, []
            while True:
                if node.is_terminal:
                    leaves.append((node, path, np.asarray(game.returns(node.state), dtype=float)))
                    break
                if node.is_chance:
                    node = _child(game, node, _sample_chance(game, node.state, rng))
                    continue
                if not node.expanded:
                    leaves.append((node, path, None))
                    break
                a = _select(node, c_puct)
                node.n[a] += virtual_loss
                node.w[a][node.to_move] -= virtual_loss
                path.append((node, a))
                node = _child(game, node, a)
        pending, seen = [], set()
        for node, _, terminal in leaves:
            if terminal is None and id(node) not in seen:
                seen.add(id(node))
                pending.append(node)
        if pending:
            _expand(game, pending, evaluator)
        for node, path, terminal in leaves:
            value = terminal if terminal is not None else node.leaf
            for parent, a in path:
                parent.n[a] += 1 - virtual_loss
                parent.w[a] += value
                parent.w[a][parent.to_move] += virtual_loss
    counts = np.zeros(game.num_distinct_actions())
    for a in root.legal:
        counts[a] = root.n[a]
    return counts


class MultiPUCTAgent(Agent):
    """Plays the most-visited move of :func:`multi_search`."""

    def __init__(self, evaluator: MultiEvaluator, simulations: int, name: str = "maxn") -> None:
        """Configure the search."""
        self.evaluator, self.simulations, self.name = evaluator, simulations, name

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Search from ``state`` and return the most-visited legal move."""
        counts = multi_search(game, state, self.evaluator, self.simulations, rng)
        legal = game.legal_actions(state)
        return int(legal[int(np.argmax([counts[a] for a in legal]))])
