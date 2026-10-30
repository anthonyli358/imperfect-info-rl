"""Cards and the poker hand evaluator.

A card is an int ``0..51``: rank ``card // 4`` (0 = two … 12 = ace) and suit ``card % 4``.
:func:`hand_value` scores the best five-card hand among any five to seven cards as one int, so
comparing two hands is comparing two ints. It works from rank and suit counts (no enumeration of
five-card subsets) and is memoised, since the same seven cards come up again and again in search.
"""

from __future__ import annotations

from functools import lru_cache

RANKS = "23456789TJQKA"
SUITS = "cdhs"

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)


def card_str(card: int) -> str:
    """The card as text: ``"As"``, ``"Td"``, …."""
    return RANKS[card // 4] + SUITS[card % 4]


def parse(text: str) -> tuple[int, ...]:
    """Cards from a string such as ``"As Kd 7h"``."""
    return tuple(RANKS.index(c[0]) * 4 + SUITS.index(c[1]) for c in text.split())


def _straight_high(mask: int) -> int:
    """The top rank of the best straight in a 13-bit rank mask, or -1 (the wheel's top is 3)."""
    shifted = (mask << 1) | (mask >> 12 & 1)  # bit 0 = ace played low
    for top in range(13, 3, -1):
        if (shifted >> (top - 4)) & 0b11111 == 0b11111:
            return top - 1
    return -1


def _pack(category: int, ranks: list[int]) -> int:
    value = category
    for r in (ranks + [0] * 5)[:5]:
        value = value * 16 + r
    return value


@lru_cache(maxsize=1 << 20)
def _value(cards: tuple[int, ...]) -> int:
    counts = [0] * 13
    suit_masks = [0] * 4
    for c in cards:
        counts[c // 4] += 1
        suit_masks[c % 4] |= 1 << (c // 4)
    for mask in suit_masks:
        if mask.bit_count() >= 5:
            high = _straight_high(mask)
            if high >= 0:
                return _pack(STRAIGHT_FLUSH, [high])
            flush = [r for r in range(12, -1, -1) if mask >> r & 1][:5]
            flush_value = _pack(FLUSH, flush)
            break
    else:
        flush_value = -1
    by_count = sorted(((n, r) for r, n in enumerate(counts) if n), reverse=True)
    groups = [r for _, r in by_count]
    top = by_count[0][0]
    if top == 4:
        kicker = max(r for r in groups if r != groups[0])
        return _pack(QUADS, [groups[0], kicker])
    if top == 3 and len(by_count) > 1 and by_count[1][0] >= 2:
        return _pack(FULL_HOUSE, [groups[0], groups[1]])
    if flush_value >= 0:
        return flush_value
    rank_mask = sum(1 << r for r in groups)
    high = _straight_high(rank_mask)
    if high >= 0:
        return _pack(STRAIGHT, [high])
    singles = sorted(groups, reverse=True)
    if top == 3:
        return _pack(TRIPS, [groups[0], *[r for r in singles if r != groups[0]][:2]])
    if top == 2 and len(by_count) > 1 and by_count[1][0] == 2:
        hi, lo = groups[0], groups[1]
        return _pack(TWO_PAIR, [hi, lo, max(r for r in singles if r not in (hi, lo))])
    if top == 2:
        return _pack(PAIR, [groups[0], *[r for r in singles if r != groups[0]][:3]])
    return _pack(HIGH_CARD, singles[:5])


def hand_value(cards: tuple[int, ...] | list[int]) -> int:
    """The best five-card hand among ``cards`` (5 to 7 of them) as a comparable int."""
    return _value(tuple(sorted(cards)))


def category(cards: tuple[int, ...] | list[int]) -> int:
    """The hand's category (``HIGH_CARD`` … ``STRAIGHT_FLUSH``); works for 2 to 7 cards."""
    if len(cards) < 5:  # before the flop: a pair or not
        ranks = [c // 4 for c in cards]
        return PAIR if len(set(ranks)) < len(ranks) else HIGH_CARD
    return hand_value(cards) >> 20
