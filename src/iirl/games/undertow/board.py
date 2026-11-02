"""Undertow's board, rosters and cards, as plain data.

The board is a river delta: 13 regions from the Smugglers' Cove (upstream, depth 0) to the Free
Port (downstream, depth 6), in three channels that split and rejoin. Each region holds at most
``CAPACITY`` pieces of one side. Smugglers may only move downstream or sideways (to a region of
equal or greater depth); the Watch may move any way.
"""

from __future__ import annotations

from dataclasses import dataclass

REGIONS = (
    "Cove",
    "West reach",
    "East reach",
    "West shoal",
    "Mid channel",
    "East shoal",
    "West narrows",
    "East narrows",
    "Lower west shoal",
    "Lower channel",
    "Lower east shoal",
    "Mouth",
    "Free Port",
)
COVE, PORT = 0, 12
NUM_REGIONS = len(REGIONS)
DEPTH = (0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 6)
CAPACITY = (4, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 4)
_EDGES = (
    (0, 1), (0, 2),  # the Cove opens into two reaches
    (1, 3), (1, 4), (2, 4), (2, 5),
    (3, 4), (4, 5),  # sideways across the upper delta
    (3, 6), (4, 6), (4, 7), (5, 7),
    (6, 8), (6, 9), (7, 9), (7, 10),
    (8, 9), (9, 10),  # sideways across the lower delta
    (8, 11), (9, 11), (10, 11),
    (11, 12),
)  # fmt: skip
ADJACENT: tuple[tuple[int, ...], ...] = tuple(
    tuple(sorted({b for a, b in _EDGES if a == r} | {a for a, b in _EDGES if b == r}))
    for r in range(NUM_REGIONS)
)
PORT_CHANNELS = frozenset({8, 9, 10, 11, 12})  # where the Harbourmaster may go

SMUGGLERS, WATCH = 0, 1
SIDE_NAMES = ("Smugglers", "Watch")
# Starting regions per token (the identities are shuffled onto them face down).
START = (
    (0, 0, 0, 0, 0, 1, 1, 2, 2),  # Smugglers: the Cove and both reaches
    (12, 12, 12, 11, 11, 8, 9, 9, 10),  # Watch: the Free Port, the Mouth and the lower delta
)


@dataclass(frozen=True)
class Piece:
    """A character: name, strength, and (for Smuggler cargo boats) the cargo's value."""

    name: str
    strength: int
    cargo: int = 0


ROSTERS = (
    (
        Piece("Decoy", 0),  # if it loses a fight it started as defender, the attacker turns back
        Piece("Runner", 2),  # may move two regions in a turn
        Piece("Pilot", 2),  # ignores region capacity
        Piece("Cargo boat", 1, cargo=3),
        Piece("Cargo boat", 1, cargo=2),
        Piece("Cargo boat", 2, cargo=2),
        Piece("Cargo boat", 2, cargo=1),
        Piece("Bruiser", 4),
        Piece("Fixer", 3),  # when it wins a fight, the card it played returns to the hand
    ),
    (
        Piece("Harbourmaster", 5),  # never leaves the port channels
        Piece("Sergeant", 4),
        Piece("Cutter", 3),  # may move two regions in a turn
        Piece("Patrol", 3),
        Piece("Patrol", 3),
        Piece("Patrol", 2),
        Piece("Patrol", 2),
        Piece("Inspector", 1),  # each Watch turn, privately sees one hidden Smuggler next to it
        Piece("Signal", 1),  # when it is removed, the Watch takes back every played card
    ),
)
DECOY, RUNNER, PILOT, BRUISER, FIXER = 0, 1, 2, 7, 8
HARBOURMASTER, CUTTER, INSPECTOR, SIGNAL = 0, 2, 7, 8
TWO_STEP = ((RUNNER,), (CUTTER,))  # per side, identities that may move two regions
CARGO_TO_WIN = 6
TOTAL_CARGO = sum(p.cargo for p in ROSTERS[SMUGGLERS])
ROUNDS = 24  # the tide clock: the Watch wins when it runs out (16 was too short: see game.py)

# Combat cards: each side holds one of each.
CARD_NAMES = ("0", "1", "2", "3", "Feint", "Lantern")
CARD_VALUES = (0, 1, 2, 3, 0, 0)
FEINT, LANTERN = 4, 5
NUM_CARDS = len(CARD_NAMES)
