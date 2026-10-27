"""OpenSpiel games behind this repo's ``Game`` interface (needs the ``[openspiel]`` extra).

``SpielGame`` wraps any two-player zero-sum OpenSpiel game, so every agent, solver and evaluator
here (tabular and graph CFR, exact best response, PPO, duplicate matches) runs on it unchanged, and
can be checked against OpenSpiel's own algorithms on the same game.

States are immutable as the contract requires: a ``SpielState`` holds the action history (which
fixes the position, so it is the state's identity for hashing and equality) and a private pyspiel
state that ``apply_action`` clones before advancing.

Registered when ``pyspiel`` imports:

- ``liars_dice``: OpenSpiel's Liar's Dice (a traditional game; DeepMind's implementation), one
  six-sided die each. Bids name a quantity and a face, each bid higher than the last; the
  alternative is to call the last bid. The highest face is wild. Winner +1, loser -1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility

try:
    import pyspiel
except ImportError:  # the [openspiel] extra is not installed
    pyspiel = None


@dataclass(frozen=True)
class SpielState(State):
    """A position, identified by its action history; ``inner`` is the pyspiel state."""

    history: tuple[int, ...]
    inner: Any = field(compare=False, hash=False, repr=False)


class SpielGame(Game):
    """A two-player zero-sum OpenSpiel game as a ``Game``."""

    def __init__(self, game_string: str, short_name: str, long_name: str) -> None:
        """Load ``game_string`` (e.g. ``"liars_dice(numdice=1)"``) with pyspiel."""
        if pyspiel is None:
            raise ImportError("OpenSpiel games need the extra: uv sync --extra openspiel")
        self.spiel = pyspiel.load_game(game_string)
        t = self.spiel.get_type()
        if self.spiel.num_players() != 2 or t.utility != pyspiel.GameType.Utility.ZERO_SUM:
            raise ValueError(f"{game_string} is not a two-player zero-sum game")
        perfect = t.information == pyspiel.GameType.Information.PERFECT_INFORMATION
        self._type = GameType(
            short_name=short_name,
            long_name=long_name,
            num_players=2,
            information=Information.PERFECT if perfect else Information.IMPERFECT,
            utility=Utility.ZERO_SUM,
            provides_information_state_tensor=t.provides_information_state_tensor,
            provides_observation_tensor=t.provides_observation_tensor,
        )

    @property
    def game_type(self) -> GameType:
        """Static metadata, taken from the OpenSpiel game."""
        return self._type

    def new_initial_state(self) -> SpielState:
        """The start state (a chance node for games that deal or roll)."""
        return SpielState((), self.spiel.new_initial_state())

    def current_player(self, state: SpielState) -> Player:
        """The player to act, or the chance / terminal sentinel (OpenSpiel uses the same ids)."""
        if state.inner.is_terminal():
            return PlayerId.TERMINAL
        if state.inner.is_chance_node():
            return PlayerId.CHANCE
        return state.inner.current_player()

    def legal_actions(self, state: SpielState) -> list[Action]:
        """OpenSpiel's legal actions."""
        return list(state.inner.legal_actions())

    def num_distinct_actions(self) -> int:
        """OpenSpiel's action-space size."""
        return self.spiel.num_distinct_actions()

    def apply_action(self, state: SpielState, action: Action) -> SpielState:
        """A new state: the pyspiel state is cloned, never mutated."""
        inner = state.inner.clone()
        inner.apply_action(action)
        return SpielState((*state.history, action), inner)

    def is_chance_node(self, state: SpielState) -> bool:
        """Whether chance moves next."""
        return state.inner.is_chance_node()

    def chance_outcomes(self, state: SpielState) -> list[ChanceOutcome]:
        """OpenSpiel's chance outcomes."""
        return [(a, p) for a, p in state.inner.chance_outcomes()]

    def is_terminal(self, state: SpielState) -> bool:
        """Whether the game is over."""
        return state.inner.is_terminal()

    def returns(self, state: SpielState) -> list[float]:
        """Per-player payoffs."""
        return list(state.inner.returns())

    def information_state_string(self, state: SpielState, player: Player) -> str:
        """OpenSpiel's information-state string."""
        return state.inner.information_state_string(player)

    def information_state_tensor(self, state: SpielState, player: Player) -> np.ndarray:
        """OpenSpiel's information-state tensor."""
        return np.asarray(state.inner.information_state_tensor(player), dtype=np.float32)

    def observation_tensor(self, state: SpielState, player: Player) -> np.ndarray:
        """OpenSpiel's observation tensor."""
        return np.asarray(state.inner.observation_tensor(player), dtype=np.float32)

    def action_to_string(self, state: SpielState, action: Action) -> str:
        """OpenSpiel's label for ``action``."""
        player = self.current_player(state)
        return state.inner.action_to_string(
            player if player >= 0 else pyspiel.PlayerId.CHANCE, action
        )

    def to_display(self, state: SpielState) -> dict:
        """The full position as OpenSpiel prints it."""
        return {
            "terminal": self.is_terminal(state),
            "current_player": int(self.current_player(state)),
            "text": str(state.inner),
            "history": list(state.history),
        }

    def player_view(self, state: SpielState, player: Player) -> dict:
        """What ``player`` may see: their information-state string."""
        return {
            "terminal": self.is_terminal(state),
            "current_player": int(self.current_player(state)),
            "text": state.inner.information_state_string(player),
        }


if pyspiel is not None:

    @register("liars_dice")
    def _liars_dice() -> SpielGame:
        return SpielGame(
            "liars_dice(numdice=1)", "liars_dice", "Liar's Dice (OpenSpiel, 1 die each)"
        )
