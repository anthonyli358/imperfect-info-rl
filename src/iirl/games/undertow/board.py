"""Undertow's board, rosters and cards, as plain data.

The board is a river delta of 14 regions, mirror-symmetric: the Smugglers' Cove (upstream, depth
0) to the Free Port (downstream, depth 6), rows of 1, 2, 3, 2, 3, 2 and 1 regions. Each region holds
at most ``CAPACITY`` pieces of one side: the Cove and the Port 4, shoals 1, channels 2. Pieces move
*forward* along ``FORWARD`` links (Smugglers downstream, the Watch upstream); some abilities and
retreats use the ``SIDEWAYS`` links, which join channels only (never a shoal).
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
    "West mouth",
    "East mouth",
    "Free Port",
)
COVE, PORT = 0, 13
NUM_REGIONS = len(REGIONS)
DEPTH = (0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6)
SHOALS = frozenset({3, 5, 8, 10})
CAPACITY = tuple(4 if r in (COVE, PORT) else 1 if r in SHOALS else 2 for r in range(NUM_REGIONS))
DOWNSTREAM = (
    (0, 1), (0, 2),
    (1, 3), (1, 4), (2, 4), (2, 5),
    (3, 6), (4, 6), (4, 7), (5, 7),
    (6, 8), (6, 9), (7, 9), (7, 10),
    (8, 11), (9, 11), (9, 12), (10, 12),
    (11, 13), (12, 13),
)  # fmt: skip
SIDEWAYS_LINKS = ((1, 2), (6, 7), (11, 12))

SMUGGLERS, WATCH = 0, 1
SIDE_NAMES = ("Smugglers", "Watch")


def _neighbours(pairs: tuple[tuple[int, int], ...], reverse: bool) -> tuple[tuple[int, ...], ...]:
    out: list[list[int]] = [[] for _ in range(NUM_REGIONS)]
    for a, b in pairs:
        if reverse:
            a, b = b, a
        out[a].append(b)
    return tuple(tuple(sorted(n)) for n in out)


# FORWARD[side][region]: where ``side`` moves forward; BACKWARD the reverse.
FORWARD = (_neighbours(DOWNSTREAM, False), _neighbours(DOWNSTREAM, True))
BACKWARD = (FORWARD[WATCH], FORWARD[SMUGGLERS])
SIDEWAYS = tuple(
    tuple(
        sorted({b for a, b in SIDEWAYS_LINKS if a == r} | {a for a, b in SIDEWAYS_LINKS if b == r})
    )
    for r in range(NUM_REGIONS)
)
HOME = (COVE, PORT)
# Start slots per side, in placement order: four at home, then one in each front region.
START = (
    (COVE, COVE, COVE, COVE, 1, 2, 3, 4, 5),
    (PORT, PORT, PORT, PORT, 11, 12, 8, 9, 10),
)
RAID = 3  # Watch pieces in the Cove at once: a Watch win


@dataclass(frozen=True)
class Piece:
    """A character: its name and strength."""

    name: str
    strength: int


ROSTERS = (
    (
        Piece("Captain", 5),  # the Watch's trick card is ignored
        Piece("Bruiser", 4),
        Piece("Wrecker", 3),  # when it is defeated, the piece that beat it sinks too
        Piece("Fixer", 3),  # a fight it wins returns its card to the hand
        Piece("Runner", 2),  # may move two regions forward to an empty region
        Piece("Pilot", 2),  # ignores capacity
        Piece("Skiff", 2),
        Piece("Contraband", 1),  # wins on entering the Free Port; slips away sideways if attacked
        Piece("Decoy", 0),  # a fight it loses as defender turns the attacker back
    ),
    (
        Piece("Gunboat", 7),  # fights on strength alone: its card is ignored
        Piece("Harbourmaster", 5),  # the Smugglers' trick card is ignored
        Piece("Sergeant", 4),
        Piece("Cutter", 3),  # may jump two regions forward to attack
        Piece("Diver", 3),  # may attack sideways
        Piece("Patrol", 3),
        Piece("Patrol", 2),
        Piece("Inspector", 2),  # each Watch turn, privately sees one hidden Smuggler next to it
        Piece("Signal", 1),  # when it is defeated, the Watch takes back every used card
    ),
)
CAPTAIN, BRUISER, WRECKER, FIXER, RUNNER, PILOT, SKIFF, CONTRABAND, DECOY = range(9)
GUNBOAT, HARBOURMASTER, SERGEANT, CUTTER, DIVER, PATROL3, PATROL2, INSPECTOR, SIGNAL = range(9)


@dataclass(frozen=True)
class Card:
    """A combat card: a number (its value) or a trick (value 0, an effect)."""

    name: str
    value: int = 0
    trick: str = ""


# Each side's hand of nine. Tricks resolve before numbers; the Watch's first.
CARDS = (
    (
        *(Card(str(v), v) for v in range(1, 6)),
        Card("Fog", trick="fog"),  # ignore the Watch's number card
        Card("Reinforce", trick="reinforce"),  # play your best used number card instead
        Card("Feint", trick="feint"),  # retreat backwards (upstream)
        Card("Scuttle", trick="scuttle"),  # both pieces sink
    ),
    (
        *(Card(str(v), v) for v in range(1, 7)),
        Card("Lantern", trick="lantern"),  # ignore the Smugglers' trick card
        Card("Reinforce", trick="reinforce"),
        Card("Withdraw", trick="withdraw"),  # retreat sideways
    ),
)
HAND_SIZE = 9
