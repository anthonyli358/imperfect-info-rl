"""Generic UCT search over any ``Game`` with chance nodes.

This is fully game-agnostic: it drives the ``Game`` interface only. Values are tracked from a single
``perspective`` player's point of view; selection flips the sign at opponent nodes (so each player
acts to maximize their own return), chance nodes are sampled from ``chance_outcomes``, and rollouts
are uniform-random to a terminal state. Determinization lives one layer up in the PIMC agent.
"""

from __future__ import annotations

import math

import numpy as np

from iirl.core.game import Game, State
from iirl.core.types import Action, Player


def _sample_chance(game: Game, state: State, rng: np.random.Generator) -> Action:
    outcomes = game.chance_outcomes(state)
    actions = [a for a, _ in outcomes]
    probs = np.array([p for _, p in outcomes], dtype=float)
    return int(rng.choice(actions, p=probs))


class _Node:
    """A search-tree node wrapping a concrete (determinized) game state."""

    __slots__ = ("state", "to_move", "is_chance", "is_terminal", "children", "untried", "n", "w")

    def __init__(self, game: Game, state: State) -> None:
        self.state = state
        self.is_terminal = game.is_terminal(state)
        self.is_chance = (not self.is_terminal) and game.is_chance_node(state)
        self.to_move = -1 if (self.is_terminal or self.is_chance) else game.current_player(state)
        self.children: dict[Action, _Node] = {}
        self.untried: list[Action] | None = None
        self.n = 0
        self.w = 0.0


def _rollout(
    game: Game,
    state: State,
    perspective: Player,
    rng: np.random.Generator,
    max_depth: int | None,
) -> float:
    """Play uniformly at random toward the end and return ``perspective``'s value.

    Stops at a terminal state (exact return) or, if ``max_depth`` is set and reached first, at a
    leaf evaluated by ``game.heuristic_value`` (falling back to a neutral 0.0 if the game provides
    none). Truncation keeps rollouts cheap when random games are long.
    """
    depth = 0
    while not game.is_terminal(state):
        if max_depth is not None and depth >= max_depth:
            estimate = game.heuristic_value(state, perspective)
            return 0.0 if estimate is None else estimate
        if game.is_chance_node(state):
            action = _sample_chance(game, state, rng)
        else:
            action = int(rng.choice(game.legal_actions(state)))
        state = game.apply_action(state, action)
        depth += 1
    return game.returns(state)[perspective]


def _best_child(node: _Node, perspective: Player, c: float) -> Action:
    """Return the UCB1-best child action (sign-flipped at opponent nodes)."""
    sign = 1.0 if node.to_move == perspective else -1.0
    log_n = math.log(node.n)
    best_action, best_value = next(iter(node.children)), -math.inf
    for action, child in node.children.items():
        exploit = sign * (child.w / child.n)
        explore = c * math.sqrt(log_n / child.n)
        value = exploit + explore
        if value > best_value:
            best_value, best_action = value, action
    return best_action


def uct_search(
    game: Game,
    root_state: State,
    perspective: Player,
    simulations: int,
    c: float,
    rng: np.random.Generator,
    max_rollout_depth: int | None = None,
) -> dict[Action, tuple[int, float]]:
    """Run UCT from ``root_state`` and return each root action's ``(visits, total_value)``.

    Total value is the sum of backed-up returns (from ``perspective``'s view); divide by visits for
    the mean action value. Returning both lets the caller pool stats across worlds and rank by mean
    value, which is far more robust than visit counts when the per-world budget is small relative to
    the branching factor.

    Args:
        game: The (determinized, perfect-information-with-chance) game to search.
        root_state: The state to search from; must be a player decision node for ``perspective``.
        perspective: The player whose return is maximized.
        simulations: Number of UCT iterations.
        c: UCB1 exploration constant.
        rng: Seeded generator for chance sampling, expansion, and rollouts.
        max_rollout_depth: Optional cap on rollout length (see :func:`_rollout`).

    Returns:
        A mapping of root action to ``(visits, total_value)``.
    """
    root = _Node(game, root_state)
    for _ in range(simulations):
        node = root
        path = [root]
        # Selection + expansion.
        while True:
            if node.is_terminal:
                value = game.returns(node.state)[perspective]
                break
            if node.is_chance:
                action = _sample_chance(game, node.state, rng)
                child = node.children.get(action)
                if child is None:
                    child = _Node(game, game.apply_action(node.state, action))
                    node.children[action] = child
                node = child
                path.append(node)
                continue
            if node.untried is None:
                node.untried = list(game.legal_actions(node.state))
            if node.untried:
                action = node.untried.pop(int(rng.integers(len(node.untried))))
                child = _Node(game, game.apply_action(node.state, action))
                node.children[action] = child
                path.append(child)
                value = _rollout(game, child.state, perspective, rng, max_rollout_depth)
                break
            node = node.children[_best_child(node, perspective, c)]
            path.append(node)
        # Backpropagation (fixed perspective; sign handled in selection).
        for n in path:
            n.n += 1
            n.w += value
    return {action: (child.n, child.w) for action, child in root.children.items()}
