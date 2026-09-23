"""Net-guided PUCT search over determinized worlds (AlphaZero search for imperfect info).

Each move samples one or more worlds consistent with the searcher's information state
(``resample_from_infostate``) and runs PUCT in each: the network supplies action priors (biasing
exploration) and a leaf value (replacing random rollouts). Root visit counts are summed across the
worlds to form the move distribution — the policy target self-play trains toward. Future chance
events (deck draws) inside a world remain first-class chance nodes, sampled during search.

Values everywhere are in ``[-1, 1]`` from the fixed searcher's perspective — win/loss at terminals,
the net's tanh value at leaves — so the network's value scale and the search's are the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iirl.core.game import Game, State
from iirl.core.runner import _sample_chance
from iirl.core.types import Player
from iirl.nn.policy_value import NetEvaluator


@dataclass(frozen=True)
class PUCTConfig:
    """Search budget and exploration settings.

    Attributes:
        simulations: PUCT simulations per determinized world.
        determinizations: Worlds sampled per move (root visit counts are pooled across them).
        c_puct: Exploration constant in the PUCT rule.
        dirichlet_alpha: Concentration of the root Dirichlet noise (self-play only).
        dirichlet_frac: Fraction of root prior replaced by noise (self-play only).
        leaf_batch: Leaves collected per network forward pass (via virtual loss).
        virtual_loss: Temporary pessimistic penalty on in-flight edges, so a batch's descents
            diverge to different leaves instead of piling onto one path.
        leaf_value: Where a leaf's value comes from — ``"net"`` (the network value head, AlphaZero)
            or ``"heuristic"`` (the game's ``heuristic_value``, i.e. net-guided PIMC: the network
            supplies only the priors and the honest hand-crafted estimate evaluates leaves).
        heuristic_scale: Divisor inside ``tanh`` that maps ``heuristic_value`` into ``[-1, 1]`` to
            match the terminal value scale (used only when ``leaf_value == "heuristic"``).
    """

    simulations: int = 60
    determinizations: int = 2
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.5
    dirichlet_frac: float = 0.25
    leaf_batch: int = 16
    virtual_loss: float = 1.0
    leaf_value: str = "net"
    heuristic_scale: float = 50.0


class _Node:
    """A search node with per-edge PUCT statistics: priors ``p``, visits ``n``, value sums ``w``."""

    __slots__ = (
        "state",
        "to_move",
        "is_chance",
        "is_terminal",
        "expanded",
        "legal",
        "p",
        "n",
        "w",
        "children",
        "netval",
    )

    def __init__(self, game: Game, state: State) -> None:
        self.state = state
        self.is_terminal = game.is_terminal(state)
        self.is_chance = (not self.is_terminal) and game.is_chance_node(state)
        self.to_move = -1 if (self.is_terminal or self.is_chance) else game.current_player(state)
        self.expanded = False
        self.legal: list[int] = []
        self.p: dict[int, float] = {}
        self.n: dict[int, float] = {}
        self.w: dict[int, float] = {}
        self.children: dict[int, _Node] = {}
        self.netval = 0.0  # net value for the mover, cached at batched expansion


def _terminal_value(game: Game, state: State, perspective: Player) -> float:
    """Return the terminal outcome for ``perspective`` as a sign in {-1, 0, 1}."""
    return float(np.sign(game.returns(state)[perspective]))


def _expand(game: Game, node: _Node, evaluator: NetEvaluator) -> float:
    """Expand ``node`` with net priors and return the net's value for the player to move."""
    node.legal = game.legal_actions(node.state)
    priors, value = evaluator.evaluate(game, node.state)
    node.p = {a: float(priors[a]) for a in node.legal}
    node.n = dict.fromkeys(node.legal, 0)
    node.w = dict.fromkeys(node.legal, 0.0)
    node.expanded = True
    return value


def _select(node: _Node, perspective: Player, c_puct: float) -> int:
    """Return the PUCT-best action (values are in the searcher's fixed perspective)."""
    total = sum(node.n.values())
    explore = c_puct * np.sqrt(1.0 + total)
    sign = 1.0 if node.to_move == perspective else -1.0
    best_a, best_u = node.legal[0], -np.inf
    for a in node.legal:
        n = node.n[a]
        q = (node.w[a] / n) if n > 0 else 0.0
        u = sign * q + explore * node.p[a] / (1 + n)
        if u > best_u:
            best_u, best_a = u, a
    return best_a


def _child(game: Game, node: _Node, action: int) -> _Node:
    """Return (creating if needed) the child of ``node`` reached by ``action``."""
    child = node.children.get(action)
    if child is None:
        child = _Node(game, game.apply_action(node.state, action))
        node.children[action] = child
    return child


def _descend(
    game: Game, root: _Node, perspective: Player, c_puct: float, rng, vloss: float
) -> tuple[_Node, list[tuple[_Node, int, float]], float | None]:
    """Descend from ``root`` to a leaf, applying virtual loss to the edges taken.

    Returns the leaf, the ``(node, action, sign)`` edges traversed, and the terminal value if the
    leaf is terminal (otherwise ``None``, meaning the leaf needs network evaluation).
    """
    node = root
    path: list[tuple[_Node, int, float]] = []
    while True:
        if node.is_terminal:
            return node, path, _terminal_value(game, node.state, perspective)
        if node.is_chance:
            node = _child(game, node, _sample_chance(game, node.state, rng))
            continue
        if not node.expanded:
            return node, path, None
        action = _select(node, perspective, c_puct)
        sign = 1.0 if node.to_move == perspective else -1.0
        node.n[action] += vloss  # virtual loss: make this edge look busy + worse until backup
        node.w[action] += -sign * vloss
        path.append((node, action, sign))
        node = _child(game, node, action)


def _backup(path: list[tuple[_Node, int, float]], value: float, vloss: float) -> None:
    """Back ``value`` up the path, simultaneously removing the virtual loss added on descent."""
    for node, action, sign in path:
        node.n[action] += 1 - vloss
        node.w[action] += value + sign * vloss


def _run_world(
    game: Game, root: _Node, perspective: Player, evaluator: NetEvaluator, config: PUCTConfig, rng
) -> None:
    """Run ``config.simulations`` PUCT simulations in one world, batching leaf evaluations."""
    done = 0
    while done < config.simulations:
        batch = min(config.leaf_batch, config.simulations - done)
        leaves = [
            _descend(game, root, perspective, config.c_puct, rng, config.virtual_loss)
            for _ in range(batch)
        ]
        done += batch
        pending, seen = [], set()
        for node, _path, terminal in leaves:
            if terminal is None and not node.expanded and id(node) not in seen:
                seen.add(id(node))
                pending.append(node)
        if pending:
            priors_list, values = evaluator.evaluate_batch(game, [n.state for n in pending])
            for node, priors, value in zip(pending, priors_list, values, strict=True):
                node.legal = game.legal_actions(node.state)
                node.p = {a: float(priors[a]) for a in node.legal}
                node.n = dict.fromkeys(node.legal, 0.0)
                node.w = dict.fromkeys(node.legal, 0.0)
                node.expanded = True
                # Leaf value: the network's own value (AlphaZero) or the game's honest hand-crafted
                # estimate (net-guided PIMC) — the network still supplies the priors either way.
                if config.leaf_value == "heuristic":
                    hv = game.heuristic_value(node.state, node.to_move)
                    node.netval = float(np.tanh(hv / config.heuristic_scale))
                else:
                    node.netval = float(value)
        for node, path, terminal in leaves:
            if terminal is not None:
                value = terminal
            else:
                value = node.netval if node.to_move == perspective else -node.netval
            _backup(path, value, config.virtual_loss)


def _new_root(game: Game, state: State, evaluator: NetEvaluator) -> _Node:
    """Create and expand a root node for ``state``."""
    root = _Node(game, state)
    _expand(game, root, evaluator)
    return root


def _add_root_noise(root: _Node, config: PUCTConfig, rng) -> None:
    """Mix Dirichlet noise into the root priors (self-play exploration)."""
    noise = rng.dirichlet([config.dirichlet_alpha] * len(root.legal))
    for a, eta in zip(root.legal, noise, strict=True):
        root.p[a] = (1 - config.dirichlet_frac) * root.p[a] + config.dirichlet_frac * float(eta)


def az_search(
    game: Game,
    state: State,
    perspective: Player,
    evaluator: NetEvaluator,
    config: PUCTConfig,
    rng: np.random.Generator,
    add_noise: bool = False,
) -> np.ndarray:
    """Return pooled root visit counts over all actions (0 on illegal actions).

    Args:
        game: The game.
        state: The searcher's real information state (a player node for ``perspective``).
        perspective: The searching player.
        evaluator: The network evaluator (priors + leaf value).
        config: Search budget / exploration.
        rng: Seeded generator (determinization, chance, and root noise).
        add_noise: Add Dirichlet root noise (use during self-play, not evaluation).

    Returns:
        A length-``num_distinct_actions`` array of visit counts summed across determinizations.
    """
    counts = np.zeros(game.num_distinct_actions())
    for _ in range(config.determinizations):
        world = game.resample_from_infostate(state, perspective, rng)
        root = _new_root(game, world, evaluator)
        if add_noise:
            _add_root_noise(root, config, rng)
        _run_world(game, root, perspective, evaluator, config, rng)
        for a in root.legal:
            counts[a] += root.n[a]
    return counts
