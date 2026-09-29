"""The game-agnostic ``Game``/``State`` interface (modelled on OpenSpiel's).

This is the one abstraction the whole repo hangs off. Every agent is written against ``Game`` and
contains zero game-specific code; every game implements this contract.

**Functional / immutable design.** ``Game`` holds all the rules and its methods take a ``state``.
``apply_action`` returns a *new* ``State`` rather than mutating in place, and concrete states are
frozen dataclasses. This makes cloning trivial (just keep the old reference), which is exactly what
determinized / information-set MCTS needs, and keeps states safe to cache, hash, and batch later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player


class State:
    """Marker base class for a game state.

    Concrete games subclass this as a ``@dataclass(frozen=True)`` holding only plain data (the
    rules live on the ``Game``). Immutability is part of the contract: ``Game.apply_action`` must
    return a new ``State`` and never mutate its input.
    """


class Game(ABC):
    """The contract every game implements.

    Methods take a ``state`` and are pure with respect to it. The optional encoders
    (``information_state_tensor`` / ``observation_tensor`` / ``information_state_string``)
    default to raising; a game implements the ones its ``game_type`` advertises.
    """

    @property
    @abstractmethod
    def game_type(self) -> GameType:
        """Static metadata for this game (see :class:`~iirl.core.types.GameType`)."""

    @property
    def num_players(self) -> int:
        """Number of players in the game."""
        return self.game_type.num_players

    @abstractmethod
    def new_initial_state(self) -> State:
        """Return the starting state (often a chance node, e.g. an initial deal)."""

    @abstractmethod
    def current_player(self, state: State) -> Player:
        """Return whose turn it is.

        Returns ``PlayerId.CHANCE`` at chance nodes and ``PlayerId.TERMINAL`` at terminal states;
        otherwise an index in ``range(num_players)``.
        """

    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]:
        """Return the actions legal in ``state`` for the current player.

        Action masking lives here, not in agents. Undefined at terminal states.
        """

    def num_distinct_actions(self) -> int:
        """Return the size of the flat action space (max action id + 1).

        This is the width a parametric policy's output head needs; ``legal_actions`` selects the
        legal subset. Optional — implemented by games used with learning agents.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define num_distinct_actions")

    @abstractmethod
    def apply_action(self, state: State, action: Action) -> State:
        """Return the new state reached by applying ``action`` (does not mutate ``state``)."""

    @abstractmethod
    def is_chance_node(self, state: State) -> bool:
        """Return whether ``state`` is a chance node (deck draw, dice, bag refill, ...)."""

    @abstractmethod
    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Return ``(action, probability)`` pairs at a chance node; probabilities sum to 1."""

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Return whether ``state`` is terminal (the game is over)."""

    @abstractmethod
    def returns(self, state: State) -> list[float]:
        """Return the per-player payoffs at a terminal state (zero-sum games sum to 0)."""

    # --- Optional encoders (gated by GameType flags) --------------------------------------------

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Return ``player``'s information-state encoding (for CFR / IS-MCTS).

        Encodes exactly what ``player`` can observe. Implemented when
        ``game_type.provides_information_state_tensor`` is set.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an information_state_tensor"
        )

    def information_state_string(self, state: State, player: Player) -> str:
        """Return a hashable string key for ``player``'s information state (CFR infoset key)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an information_state_string"
        )

    def observation_tensor(self, state: State, player: Player) -> np.ndarray:
        """Return ``player``'s observation encoding (for perfect-info / AlphaZero use).

        Implemented when ``game_type.provides_observation_tensor`` is set.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide an observation_tensor")

    def heuristic_value(self, state: State, player: Player) -> float | None:
        """Return a leaf-evaluation estimate of ``player``'s eventual return, or ``None``.

        Used by MCTS to evaluate truncated rollouts (cutting a random playout short and estimating
        the value instead of playing on) — essential when random games are long. The estimate
        should be on the same scale as :meth:`returns` so tree values stay comparable. Optional:
        returning ``None`` (the default) tells MCTS to roll out to a terminal state instead.
        """
        return None

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> State:
        """Return a random world consistent with ``player``'s information state.

        Every fact ``player`` can observe (their own private info and all public info) is preserved;
        the hidden remainder is re-dealt uniformly under ``rng``. This is the single game-specific
        hook that lets determinized / information-set MCTS stay game-agnostic: the agent samples
        worlds through this method and never reasons about a game's hidden structure itself.
        Implemented by imperfect-information games.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide resample_from_infostate")

    def repair_world(
        self, state: State, player: Player, world: State, rng: np.random.Generator
    ) -> State:
        """Return a world consistent with ``player``'s view of ``state``, close to ``world``.

        ``world`` is a guess at the hidden information that a new observation contradicts (e.g. a
        particle in a belief filter that did not deal the opponent the card they just played). A
        game may keep as much of its hidden part as is still possible and re-deal only the rest;
        the default ignores ``world`` and resamples uniformly.
        """
        return self.resample_from_infostate(state, player, rng)

    # --- Optional debug / display helpers -------------------------------------------------------

    def action_to_string(self, state: State, action: Action) -> str:
        """Return a human-readable label for ``action`` (defaults to its integer id)."""
        return str(action)

    def display_meta(self) -> dict:
        """Return static, render-time metadata for the replay viewer (names, colors, ranks).

        Consumed by the cross-game viewer to label and color pieces consistently. Optional; the
        default is empty. See :mod:`iirl.viz.replay`.
        """
        return {}

    def to_display(self, state: State) -> dict:
        """Return a JSON-serializable, render-ready snapshot of ``state`` for the replay viewer.

        This is a *human-facing* encoder (a counterpart to ``information_state_tensor``): it exposes
        the full position for watching a replay, not a per-player observation. Default: minimal
        generic info. Games override it to describe hands, tableaus, discards, etc.
        """
        return {
            "terminal": self.is_terminal(state),
            "current_player": int(self.current_player(state)),
        }

    def player_view(self, state: State, player: Player) -> dict:
        """Return what ``player`` may see of ``state``, as JSON-serializable data for a play UI.

        Like :meth:`to_display`, but with everything hidden from ``player`` removed (for a card
        game: the opponent's hand beyond its size and publicly known cards, and the deck's
        contents). A play server sends only this to the human, so hidden information cannot leak
        through the page. The default suits perfect-information games; imperfect-information games
        must override it.
        """
        if self.game_type.information != Information.PERFECT:
            raise NotImplementedError(f"{type(self).__name__} does not provide player_view")
        return self.to_display(state)
