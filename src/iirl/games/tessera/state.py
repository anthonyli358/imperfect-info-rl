"""Immutable state for Tessera.

Plain tuples throughout, so states are hashable, cheap to copy and never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from iirl.core.game import State
from iirl.games.tessera.tiles import NUM_COLORS, NUM_FACTORIES, TILES_PER_COLOR, WALL_SIZE

Counts = tuple[int, ...]
NO_TILES: Counts = (0,) * NUM_COLORS
EMPTY_WALL: tuple[bool, ...] = (False,) * (WALL_SIZE * WALL_SIZE)
EMPTY_LINES: tuple[tuple[int, int], ...] = ((-1, 0),) * WALL_SIZE  # (colour or -1, tiles)


class Phase(IntEnum):
    """Which step of a round the state sits at."""

    FILL = 0  # chance: draw the next tile from the bag onto a factory display
    PLAY = 1  # player: take tiles of one colour from a display or the centre


@dataclass(frozen=True)
class Board:
    """One player's board. Everything on it is public.

    Attributes:
        lines: Per pattern row, ``(colour, tiles)``; colour ``-1`` while the row is empty.
        wall: 25 flags, row-major, whether each wall space holds its tile.
        floor: Tiles on the floor line (the first-player marker counts as one).
        score: Points so far (never below 0).
    """

    lines: tuple[tuple[int, int], ...] = EMPTY_LINES
    wall: tuple[bool, ...] = EMPTY_WALL
    floor: int = 0
    score: int = 0


@dataclass(frozen=True)
class TesseraState(State):
    """A complete Tessera position.

    Attributes:
        bag: Tiles per colour still in the bag (hidden order: draws are chance events).
        lid: Tiles per colour discarded this cycle; refills the bag when it runs out.
        factories: Per display, tiles per colour.
        center: Tiles per colour in the centre.
        marker_in_center: Whether the first-player marker is still in the centre this round.
        boards: Both players' boards.
        to_move: The player to act at player nodes.
        phase: The current :class:`Phase`.
        filled: Tiles drawn so far while filling the displays this round.
        starter: Who starts the next round (holder of the marker).
        round: Rounds completed.
        terminal: Whether the game is over.
    """

    bag: Counts = (TILES_PER_COLOR,) * NUM_COLORS
    lid: Counts = NO_TILES
    factories: tuple[Counts, ...] = (NO_TILES,) * NUM_FACTORIES
    center: Counts = NO_TILES
    marker_in_center: bool = True
    boards: tuple[Board, Board] = (Board(), Board())
    to_move: int = 0
    phase: Phase = Phase.FILL
    filled: int = 0
    starter: int = 0
    round: int = 0
    terminal: bool = False
