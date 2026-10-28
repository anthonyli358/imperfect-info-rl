"""Exact solving and exploitability for small perfect-information games with chance, any players.

``eval.exact_br`` is built for two-player imperfect-information games and assumes every state of an
information set sits at one depth. In a perfect-information game every state is its own
information set, so a memoised search over distinct states is enough, and it does not care how many
moves led to a state (in an auction, raising to 1 then 3 and bidding 3 at once can meet).

- :func:`solve`: backward induction. Each player picks the action that maximises their own value,
  given how everyone plays from there on (a subgame-perfect equilibrium). With two zero-sum
  players the value is unique; with three, ties between actions can select different equilibria,
  so ties are broken towards the lowest action id (in an auction: pass, then the smallest bid).
- :func:`nash_conv`: for a policy, each player's gain from best-responding while the others keep
  playing it, summed (0 at an equilibrium). For two zero-sum players ``nash_conv / 2`` is the usual
  exploitability.

Policies are ``policy(game, state) -> probabilities over all action ids``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from iirl.core.game import Game, State
from iirl.core.types import Action, Information

Policy = Callable[[Game, State], np.ndarray]


@dataclass
class Solution:
    """Backward-induction values (one per player) and the chosen action at every player node."""

    values: dict[State, np.ndarray]
    choice: dict[State, Action]

    def policy(self, game: Game, state: State) -> np.ndarray:
        """The solution as a (deterministic) policy."""
        probs = np.zeros(game.num_distinct_actions())
        probs[self.choice[state]] = 1.0
        return probs


def _check(game: Game) -> None:
    assert game.game_type.information == Information.PERFECT, "perfect-information games only"
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))


def solve(game: Game) -> Solution:
    """Backward induction over every reachable state; ties go to the lowest action id."""
    _check(game)
    values: dict[State, np.ndarray] = {}
    choice: dict[State, Action] = {}

    def value(s: State) -> np.ndarray:
        v = values.get(s)
        if v is not None:
            return v
        if game.is_terminal(s):
            v = np.asarray(game.returns(s), dtype=float)
        elif game.is_chance_node(s):
            v = sum(p * value(game.apply_action(s, a)) for a, p in game.chance_outcomes(s))
        else:
            p = game.current_player(s)
            best_a, v = None, None
            for a in sorted(game.legal_actions(s)):
                child = value(game.apply_action(s, a))
                if v is None or child[p] > v[p] + 1e-12:
                    best_a, v = a, child
            choice[s] = best_a
        values[s] = v
        return v

    value(game.new_initial_state())
    return Solution(values, choice)


def policy_values(game: Game, policy: Policy) -> np.ndarray:
    """Every player's expected return when all play ``policy``."""
    _check(game)
    memo: dict[State, np.ndarray] = {}

    def value(s: State) -> np.ndarray:
        v = memo.get(s)
        if v is not None:
            return v
        if game.is_terminal(s):
            v = np.asarray(game.returns(s), dtype=float)
        elif game.is_chance_node(s):
            v = sum(p * value(game.apply_action(s, a)) for a, p in game.chance_outcomes(s))
        else:
            probs = policy(game, s)
            v = sum(
                probs[a] * value(game.apply_action(s, a)) for a in game.legal_actions(s) if probs[a]
            )
        memo[s] = v
        return v

    return value(game.new_initial_state())


def best_response_value(game: Game, policy: Policy, player: int) -> float:
    """``player``'s return when best-responding to everyone else playing ``policy``."""
    _check(game)
    memo: dict[State, float] = {}

    def value(s: State) -> float:
        v = memo.get(s)
        if v is not None:
            return v
        if game.is_terminal(s):
            v = float(game.returns(s)[player])
        elif game.is_chance_node(s):
            v = sum(p * value(game.apply_action(s, a)) for a, p in game.chance_outcomes(s))
        elif game.current_player(s) == player:
            v = max(value(game.apply_action(s, a)) for a in game.legal_actions(s))
        else:
            probs = policy(game, s)
            v = sum(
                probs[a] * value(game.apply_action(s, a)) for a in game.legal_actions(s) if probs[a]
            )
        memo[s] = v
        return v

    return value(game.new_initial_state())


def nash_conv(game: Game, policy: Policy) -> dict[str, float]:
    """Each player's best-response gain against ``policy``, and their sum (``nash_conv``)."""
    on_policy = policy_values(game, policy)
    out: dict[str, float] = {}
    for p in range(game.num_players):
        out[f"gain_{p}"] = best_response_value(game, policy, p) - float(on_policy[p])
    out["nash_conv"] = sum(out[f"gain_{p}"] for p in range(game.num_players))
    return out


def uniform(game: Game, state: State) -> np.ndarray:
    """Uniform over the legal actions."""
    legal = game.legal_actions(state)
    probs = np.zeros(game.num_distinct_actions())
    probs[legal] = 1.0 / len(legal)
    return probs
