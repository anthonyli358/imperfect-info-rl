"""Tiles, boards and scoring rules as data for Tessera.

Tessera is an original, from-scratch game reusing a well-known *mechanical* skeleton: players draft
all the tiles of one colour from shared displays, stage them on pattern rows, and move one tile of
each completed row onto a fixed-colour wall, scoring for adjacency, with penalties for tiles that
fall to the floor. Mechanics are not copyrightable; the name, theme, art and rulebook text of any
published game are. This implementation shares none of those: its own name, colour names and no
assets, and every tile is a plain integer. It is an independent implementation, not affiliated
with or endorsed by any publisher; the mechanical inspiration (the tile-drafting game designed by
Michael Kiesling) is credited here only as inspiration.

**Encoding.** A tile is a colour id ``0..4``. The wall is a 5×5 grid whose row ``r`` holds colour
``k`` in column ``(k + r) % 5``, so every row and every column holds each colour once.

**Differences from the original rules (deliberate).** Two players only. The payoff is the score
difference, so a tie is a draw (the original breaks ties on completed rows). A game that has not
ended after ``MAX_ROUNDS`` rounds ends there (random play can stall; real games take 5-6 rounds).
"""

from __future__ import annotations

NUM_COLORS = 5
COLOR_NAMES: tuple[str, ...] = ("Cobalt", "Saffron", "Vermilion", "Jet", "Ivory")
COLOR_HEX: tuple[str, ...] = ("#2b5fa8", "#e0a21b", "#c4372f", "#2b2b2b", "#e8e2d0")  # display
# One glyph per colour, so tiles are told apart without colour (display only).
COLOR_SYMBOLS: tuple[str, ...] = ("◆", "●", "▲", "■", "✚")
TILES_PER_COLOR = 20
NUM_FACTORIES = 5  # two players
TILES_PER_FACTORY = 4
WALL_SIZE = 5  # rows and columns; pattern row r holds r + 1 tiles
FLOOR_PENALTIES: tuple[int, ...] = (-1, -1, -2, -2, -2, -3, -3)
FLOOR_SLOTS = len(FLOOR_PENALTIES)
ROW_BONUS = 2
COLUMN_BONUS = 7
COLOR_BONUS = 10
MAX_ROUNDS = 20

# Sources of tiles: the factory displays, then the centre.
CENTER = NUM_FACTORIES
NUM_SOURCES = NUM_FACTORIES + 1
# Destinations: the five pattern rows, then the floor.
FLOOR = WALL_SIZE
NUM_DESTINATIONS = WALL_SIZE + 1


def wall_column(row: int, color: int) -> int:
    """The wall column that holds ``color`` in ``row``."""
    return (color + row) % NUM_COLORS


def wall_color(row: int, column: int) -> int:
    """The colour of the wall space at ``(row, column)``."""
    return (column - row) % NUM_COLORS


def placement_points(wall: tuple[bool, ...], row: int, column: int) -> int:
    """Points for a tile just placed at ``(row, column)`` on ``wall`` (which already holds it).

    A tile with no neighbours scores 1; otherwise it scores the length of its horizontal run (if
    longer than 1) plus the length of its vertical run (if longer than 1).
    """

    def filled(r: int, c: int) -> bool:
        return 0 <= r < WALL_SIZE and 0 <= c < WALL_SIZE and wall[r * WALL_SIZE + c]

    left = right = column
    while filled(row, left - 1):
        left -= 1
    while filled(row, right + 1):
        right += 1
    up = down = row
    while filled(up - 1, column):
        up -= 1
    while filled(down + 1, column):
        down += 1
    horizontal, vertical = right - left + 1, down - up + 1
    if horizontal == 1 and vertical == 1:
        return 1
    return (horizontal if horizontal > 1 else 0) + (vertical if vertical > 1 else 0)


def floor_penalty(tiles: int) -> int:
    """Total penalty for ``tiles`` on the floor line (beyond its slots they are free)."""
    return sum(FLOOR_PENALTIES[: min(tiles, FLOOR_SLOTS)])


def end_bonus(wall: tuple[bool, ...]) -> int:
    """End-of-game bonus: complete rows, complete columns, and colours placed five times."""
    rows = sum(all(wall[r * WALL_SIZE + c] for c in range(WALL_SIZE)) for r in range(WALL_SIZE))
    columns = sum(all(wall[r * WALL_SIZE + c] for r in range(WALL_SIZE)) for c in range(WALL_SIZE))
    colors = sum(
        all(wall[r * WALL_SIZE + wall_column(r, k)] for r in range(WALL_SIZE))
        for k in range(NUM_COLORS)
    )
    return ROW_BONUS * rows + COLUMN_BONUS * columns + COLOR_BONUS * colors
