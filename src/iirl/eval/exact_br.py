"""Exact exploitability for games with a few million states, and for policy networks.

``eval.exploitability`` recurses over every *path* through the tree, so its cost grows with the
number of histories. Here every distinct state is enumerated once (states are hashable, so paths
that reach the same position merge), reach probabilities are pushed forward level by level, and
best-response values are computed backward. The best responder commits one action per information
set, as a real player must.

Requirement (asserted): all states of an information set lie at the same depth. True whenever the
observation fixes how many moves have been made (Mini-Ventures: the deck size and phase do).

The policy being tested is a batched function ``policy(game, states) -> [probs over all actions]``,
so a neural network's whole strategy is evaluated in large batches. See :func:`net_policy`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from iirl.core.game import Game, State

BatchPolicy = Callable[[Game, Sequence[State]], Sequence[np.ndarray]]


@dataclass
class GameGraph:
    """Every distinct reachable state, grouped by depth, with child indices."""

    states: list[State]
    depth: list[int]
    children: list[list[tuple[int, int]]]  # (action, child index)
    by_depth: list[list[int]]


def build_graph(game: Game) -> GameGraph:
    """Enumerate each reachable state once, breadth first."""
    index: dict[State, int] = {}
    states: list[State] = []
    depth: list[int] = []
    children: list[list[tuple[int, int]]] = []
    frontier = [game.new_initial_state()]
    index[frontier[0]] = 0
    states.append(frontier[0])
    depth.append(0)
    children.append([])
    d = 0
    by_depth = [[0]]
    while frontier:
        nxt: list[State] = []
        for s in frontier:
            i = index[s]
            if game.is_terminal(s):
                continue
            actions = (
                [a for a, _ in game.chance_outcomes(s)]
                if game.is_chance_node(s)
                else game.legal_actions(s)
            )
            for a in actions:
                c = game.apply_action(s, a)
                j = index.get(c)
                if j is None:
                    j = len(states)
                    index[c] = j
                    states.append(c)
                    depth.append(d + 1)
                    children.append([])
                    nxt.append(c)
                elif depth[j] != d + 1:
                    raise ValueError(
                        "a state is reachable at two depths; exact_br cannot handle it"
                    )
                children[i].append((a, j))
        d += 1
        if nxt:
            by_depth.append([index[s] for s in nxt])
        frontier = nxt
    return GameGraph(states, depth, children, by_depth)


def _policy_table(
    game: Game, graph: GameGraph, policy: BatchPolicy, player: int, batch: int
) -> dict[str, np.ndarray]:
    """Evaluate ``policy`` once per information set of ``player`` (in batches)."""
    first: dict[str, State] = {}
    for s in graph.states:
        if (
            not game.is_terminal(s)
            and not game.is_chance_node(s)
            and game.current_player(s) == player
        ):
            first.setdefault(game.information_state_string(s, player), s)
    keys = list(first)
    table: dict[str, np.ndarray] = {}
    for k in range(0, len(keys), batch):
        chunk = keys[k : k + batch]
        for key, probs in zip(chunk, policy(game, [first[key] for key in chunk]), strict=True):
            table[key] = np.asarray(probs, dtype=float)
    return table


def best_response_value(
    game: Game, graph: GameGraph, table: dict[str, np.ndarray], br_player: int
) -> float:
    """Value to ``br_player`` best-responding to the opponent's per-infoset ``table``."""
    n = len(graph.states)
    reach = np.zeros(n)  # everyone's reach except the best responder's own choices
    reach[0] = 1.0
    opp = 1 - br_player
    for level in graph.by_depth:
        for i in level:
            if reach[i] == 0.0:
                continue
            s = graph.states[i]
            if game.is_terminal(s):
                continue
            if game.is_chance_node(s):
                probs = dict(game.chance_outcomes(s))
                for a, j in graph.children[i]:
                    reach[j] += reach[i] * probs[a]
            elif game.current_player(s) == opp:
                pi = table[game.information_state_string(s, opp)]
                for a, j in graph.children[i]:
                    reach[j] += reach[i] * pi[a]
            else:
                for _, j in graph.children[i]:
                    reach[j] += reach[i]
    value = np.zeros(n)
    for level in reversed(graph.by_depth):
        infosets: dict[str, list[int]] = {}
        for i in level:
            s = graph.states[i]
            if game.is_terminal(s):
                value[i] = game.returns(s)[br_player]
            elif game.is_chance_node(s):
                probs = dict(game.chance_outcomes(s))
                value[i] = sum(probs[a] * value[j] for a, j in graph.children[i])
            elif game.current_player(s) == opp:
                pi = table[game.information_state_string(s, opp)]
                value[i] = sum(pi[a] * value[j] for a, j in graph.children[i])
            else:
                infosets.setdefault(game.information_state_string(s, br_player), []).append(i)
        for members in infosets.values():
            actions = [a for a, _ in graph.children[members[0]]]
            gains = np.zeros(len(actions))
            for i in members:
                for k, (_, j) in enumerate(graph.children[i]):
                    gains[k] += reach[i] * value[j]
            best = int(np.argmax(gains))
            for i in members:
                value[i] = value[graph.children[i][best][1]]
    return float(value[0])


def exploitability(
    game: Game, policy: BatchPolicy, graph: GameGraph | None = None, batch: int = 4096
) -> dict[str, float]:
    """Return exact best-response values against ``policy`` (in the game's payoff units).

    ``br_as_0`` / ``br_as_1`` are a best responder's expected payoff in each seat against the
    policy in the other; ``exploitability`` is their mean (NashConv / 2): 0 at a Nash equilibrium,
    and the points per game a perfect exploiter gains, averaged over seats.
    """
    graph = graph or build_graph(game)
    tables = [_policy_table(game, graph, policy, p, batch) for p in (0, 1)]
    br0 = best_response_value(game, graph, tables[1], 0)
    br1 = best_response_value(game, graph, tables[0], 1)
    return {"br_as_0": br0, "br_as_1": br1, "exploitability": (br0 + br1) / 2}


def net_policy(net, greedy: bool = False) -> BatchPolicy:  # noqa: ANN001 - PolicyValueNet
    """A batched policy from a network: masked softmax (or its greedy one-hot)."""
    import torch

    def policy(game: Game, states: Sequence[State]) -> list[np.ndarray]:
        feats = np.stack([game.information_state_tensor(s, game.current_player(s)) for s in states])
        with torch.no_grad():
            logits = net(torch.as_tensor(feats, dtype=torch.float32))[0].numpy()
        out = []
        for s, row in zip(states, logits, strict=True):
            legal = game.legal_actions(s)
            probs = np.zeros(game.num_distinct_actions())
            if greedy:
                probs[legal[int(np.argmax(row[legal]))]] = 1.0
            else:
                z = row[legal] - row[legal].max()
                probs[legal] = np.exp(z) / np.exp(z).sum()
            out.append(probs)
        return out

    return policy


def uniform_policy(game: Game, states: Sequence[State]) -> list[np.ndarray]:
    """Every legal action equally likely (a sanity baseline)."""
    out = []
    for s in states:
        legal = game.legal_actions(s)
        probs = np.zeros(game.num_distinct_actions())
        probs[legal] = 1.0 / len(legal)
        out.append(probs)
    return out
