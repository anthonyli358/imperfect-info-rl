"""Shared value types for the game-agnostic core.

These are the primitives every ``Game`` speaks in: player indices (with sentinels for chance and
terminal nodes), integer actions, chance outcomes, and the static ``GameType`` descriptor that tells
agents and the conformance suite what a game supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

# A player is an index in ``range(num_players)``; sentinels below cover chance/terminal nodes.
Player = int

# Actions are dense integer ids so they double as indices for legal-action masks and NN heads.
Action = int

# A chance outcome pairs an action with its probability. ``chance_outcomes`` returns a list of these
# and the probabilities must sum to 1.
ChanceOutcome = tuple[Action, float]


class PlayerId(IntEnum):
    """Sentinel player ids for non-decision nodes (values match OpenSpiel's convention)."""

    CHANCE = -1
    TERMINAL = -2


class Information(IntEnum):
    """Whether players can observe the full state."""

    PERFECT = 0
    IMPERFECT = 1


class Utility(IntEnum):
    """The payoff structure at terminal nodes."""

    ZERO_SUM = 0
    GENERAL = 1
    COOPERATIVE = 2


@dataclass(frozen=True)
class GameType:
    """Static, per-game metadata describing what a ``Game`` supports.

    Agents and the conformance suite read these flags to decide which optional methods (the tensor
    encoders, in particular) are expected to be implemented.

    Attributes:
        short_name: Registry key, e.g. ``"toy_showdown"``.
        long_name: Human-readable name.
        num_players: Number of players (all rungs on the ladder are two-player).
        information: Whether the game is perfect- or imperfect-information.
        utility: The terminal payoff structure.
        provides_information_state_tensor: Whether ``information_state_tensor`` is implemented.
        provides_observation_tensor: Whether ``observation_tensor`` is implemented.
    """

    short_name: str
    long_name: str
    num_players: int
    information: Information
    utility: Utility
    provides_information_state_tensor: bool = False
    provides_observation_tensor: bool = False
