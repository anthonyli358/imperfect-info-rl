"""Immutable state for Spendthrift.

Plain ints and tuples throughout, so states are hashable, cheap to copy and never mutated. Only
what the rest of the game depends on is stored: a player's lots matter only through their total
and their single best lot, so two histories that differ only in which lots make up a total are the
same state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from iirl.core.game import State


class Phase(IntEnum):
    """Which step of a round the state sits at."""

    REVEAL = 0  # chance: flip the top card of the deck
    AUCTION = 1  # player: raise or pass on the revealed lot
    OVER = 2  # terminal


@dataclass(frozen=True)
class SpendthriftState(State):
    """A complete Spendthrift position.

    Attributes:
        coins: Coins each player has left (public).
        totals: Each player's total lot value.
        best: Each player's most valuable lot (0 if none).
        deck: Bitmask of lot values still in the deck (bit ``v - 1`` for value ``v``).
        ends_left: End cards still in the deck.
        ends_seen: End cards revealed so far.
        phase: The current :class:`Phase`.
        lot: Value of the lot up for auction (0 outside an auction).
        bid: The current high bid (0 if no one has bid).
        leader: Who holds the high bid (-1 if no one).
        active: Bitmask of players still in this auction.
        to_move: The player to act in an auction.
    """

    coins: tuple[int, ...]
    totals: tuple[int, ...]
    best: tuple[int, ...]
    deck: int
    ends_left: int
    ends_seen: int = 0
    phase: Phase = Phase.REVEAL
    lot: int = 0
    bid: int = 0
    leader: int = -1
    active: int = 0
    to_move: int = 0
