"""Cards as data for Ventures.

Ventures is an original, from-scratch game reusing a well-known *mechanical* skeleton — ascending
single-suit routes, investment/"backing" multipliers, a fixed start cost, and play-or-discard-then-
draw over a hidden deck. Mechanics are not copyrightable; the name, theme, art, and rulebook text of
any published game are. This implementation shares none of those: it has its own name and theme, no
assets, and represents every card as a plain integer. It is an independent implementation, not
affiliated with or endorsed by any publisher; the mechanical inspiration (the investment-expedition
family of card games designed by Reiner Knizia) is credited here only as inspiration.

**Encoding.** A card is a ``(suit, rank)`` pair flattened to a dense integer *card-type* id in
``0..NUM_CARD_TYPES-1``. Rank ``0`` is the ``BACKING`` card (investment marker, ordering value 0);
ranks ``2..10`` are number cards. Backing cards have ``BACKING_PER_SUIT`` identical copies per suit;
each number card is unique per suit.
"""

from __future__ import annotations

NUM_SUITS = 5
BACKING = 0  # rank id for a backing (investment) card; sorts below every number
MIN_NUMBER = 2
MAX_NUMBER = 10
BACKING_PER_SUIT = 3
HAND_SIZE = 8
START_COST = 20  # subtracted from every started route
BONUS_CARDS = 8  # a route with this many cards earns...
BONUS_POINTS = 20  # ...this bonus
# The game also ends after this many turns. Taking from discard piles does not shrink the deck,
# so without a cap two players could trade discards forever; a normal game takes ~75 turns.
MAX_TURNS = 200

# Rank ids per suit: BACKING plus the numbers 2..10.
RANKS: tuple[int, ...] = (BACKING, *range(MIN_NUMBER, MAX_NUMBER + 1))
RANKS_PER_SUIT = len(RANKS)  # 10
NUM_CARD_TYPES = NUM_SUITS * RANKS_PER_SUIT  # 50

_RANK_INDEX = {rank: i for i, rank in enumerate(RANKS)}
# Per-card-type lookup tables: these sit on every hot path (legal moves, scoring, encoders).
_SUIT_OF: tuple[int, ...] = tuple(c // RANKS_PER_SUIT for c in range(NUM_CARD_TYPES))
_RANK_OF: tuple[int, ...] = tuple(RANKS[c % RANKS_PER_SUIT] for c in range(NUM_CARD_TYPES))
_IS_BACKING: tuple[bool, ...] = tuple(r == BACKING for r in _RANK_OF)

# Theme (for display only — the engine treats suits as opaque integers). Five ventures, each with a
# distinct, reasonably color-blind-safe hue used consistently by the replay viewer.
SUIT_NAMES: tuple[str, ...] = ("Harbor", "Summit", "Orchard", "Foundry", "Bazaar")
SUIT_COLORS: tuple[str, ...] = ("#2f7f9f", "#c9772b", "#3f9152", "#7d5fb0", "#c0414c")
SUIT_SYMBOLS: tuple[str, ...] = ("⚓", "▲", "❦", "⚙", "✦")  # plain Unicode glyphs, display only


def card_name(card: int) -> str:
    """Return a human-readable card label, e.g. ``"Harbor 7"`` or ``"Summit Backing"``."""
    rank = rank_of(card)
    return f"{SUIT_NAMES[suit_of(card)]} {'Backing' if rank == BACKING else rank}"


def card_id(suit: int, rank: int) -> int:
    """Return the dense card-type id for ``(suit, rank)``."""
    return suit * RANKS_PER_SUIT + _RANK_INDEX[rank]


def suit_of(card: int) -> int:
    """Return the suit of a card-type id."""
    return _SUIT_OF[card]


def rank_of(card: int) -> int:
    """Return the rank of a card-type id (``BACKING`` or a number ``2..10``)."""
    return _RANK_OF[card]


def is_backing(card: int) -> bool:
    """Return whether a card-type id is a backing card."""
    return _IS_BACKING[card]


def initial_deck_counts() -> tuple[int, ...]:
    """Return the full-deck multiset as counts per card-type id.

    Backing card-types have ``BACKING_PER_SUIT`` copies; number card-types have one each.
    """
    counts = [0] * NUM_CARD_TYPES
    for suit in range(NUM_SUITS):
        counts[card_id(suit, BACKING)] = BACKING_PER_SUIT
        for number in range(MIN_NUMBER, MAX_NUMBER + 1):
            counts[card_id(suit, number)] = 1
    return tuple(counts)


DECK_SIZE = sum(initial_deck_counts())  # 60
