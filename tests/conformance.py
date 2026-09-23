"""Reusable conformance suite every ``Game`` must pass.

This is the shared contract test: point it at any game and it walks many random playouts, asserting
the interface invariants hold at every node. Each new engine gets its correctness floor for free by
running this — no per-game bespoke checks needed for the basics.
"""

from __future__ import annotations

import numpy as np

from iirl.core.game import Game, State
from iirl.core.types import PlayerId, Utility

_PROB_TOL = 1e-9
_RETURN_TOL = 1e-9


def _check_chance_node(game: Game, state: State) -> None:
    """Assert chance-node invariants: non-empty, positive, unique, and probabilities sum to 1."""
    outcomes = game.chance_outcomes(state)
    assert outcomes, "chance node has no outcomes"
    actions = [a for a, _ in outcomes]
    probs = [p for _, p in outcomes]
    assert len(set(actions)) == len(actions), "duplicate chance outcomes"
    assert all(p > 0 for p in probs), "non-positive chance probability"
    assert abs(sum(probs) - 1.0) < _PROB_TOL, f"chance probabilities sum to {sum(probs)}, not 1"


def _check_player_node(game: Game, state: State, tensor_shapes: dict[int, tuple[int, ...]]) -> None:
    """Assert player-node invariants and (if advertised) constant-shape info-state tensors."""
    player = game.current_player(state)
    assert 0 <= player < game.num_players, f"current_player {player} out of range"

    legal = game.legal_actions(state)
    assert legal, "no legal actions at a non-terminal player node"
    assert all(isinstance(a, int) for a in legal), "legal actions must be ints"
    assert len(set(legal)) == len(legal), "duplicate legal actions"

    if game.game_type.provides_information_state_tensor:
        for p in range(game.num_players):
            tensor = game.information_state_tensor(state, p)
            assert isinstance(tensor, np.ndarray), "information_state_tensor must return an ndarray"
            prev = tensor_shapes.setdefault(p, tensor.shape)
            assert tensor.shape == prev, (
                f"info-state tensor shape changed: {tensor.shape} vs {prev}"
            )


def _check_immutable_apply(game: Game, state: State, action: int) -> State:
    """Apply ``action`` and assert it returns a new object without mutating ``state``."""
    before = repr(state)
    new_state = game.apply_action(state, action)
    assert new_state is not state, "apply_action must return a new state, not the same object"
    assert repr(state) == before, "apply_action mutated its input state"
    return new_state


def check_conformance(
    game: Game,
    rng: np.random.Generator,
    episodes: int = 200,
    max_steps: int = 1000,
) -> None:
    """Run the conformance suite for ``game`` over random playouts.

    Args:
        game: The game under test.
        rng: Seeded generator driving the random walk.
        episodes: Number of random playouts.
        max_steps: Per-episode safety cap (a game exceeding it is treated as non-terminating).

    Raises:
        AssertionError: If any interface invariant is violated.
    """
    assert isinstance(game.new_initial_state(), State), "new_initial_state must return a State"
    zero_sum = game.game_type.utility == Utility.ZERO_SUM
    tensor_shapes: dict[int, tuple[int, ...]] = {}

    for _ in range(episodes):
        state = game.new_initial_state()
        for _step in range(max_steps):
            if game.is_terminal(state):
                assert game.current_player(state) == PlayerId.TERMINAL
                returns = game.returns(state)
                assert len(returns) == game.num_players, "returns length != num_players"
                if zero_sum:
                    assert abs(sum(returns)) < _RETURN_TOL, (
                        f"zero-sum returns sum to {sum(returns)}"
                    )
                break

            if game.is_chance_node(state):
                assert game.current_player(state) == PlayerId.CHANCE
                _check_chance_node(game, state)
                outcomes = game.chance_outcomes(state)
                actions = [a for a, _ in outcomes]
                probs = np.array([p for _, p in outcomes], dtype=float)
                action = int(rng.choice(actions, p=probs))
            else:
                _check_player_node(game, state, tensor_shapes)
                action = int(rng.choice(game.legal_actions(state)))

            state = _check_immutable_apply(game, state, action)
        else:
            raise AssertionError(f"game did not terminate within {max_steps} steps")
