"""Immutable state for Camel Stocks.

Plain tuples throughout, so states are hashable, cheap to copy and never mutated (the functional
contract the rest of the repo relies on).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from iirl.core.game import State
from iirl.games.camel_stocks.cards import (
    BONUS_TOKENS,
    CAMEL,
    CARD_COUNTS,
    DEAL_SIZE,
    GOOD_TOKENS,
    MARKET_SIZE,
    MARKET_START_CAMELS,
    NUM_CARD_TYPES,
    NUM_GOODS,
)

Counts = tuple[int, ...]
_NO_GOODS: Counts = (0,) * NUM_GOODS
_NO_CARDS: Counts = (0,) * NUM_CARD_TYPES


class Phase(IntEnum):
    """Which step of a turn the state sits at."""

    DEAL = 0  # chance: fill the market, then deal each player's hand, one card at a time
    PLAY = 1  # player: choose the turn's action (take, take camels, sell, or start an exchange)
    EX_TAKE = 2  # player: exchange in progress, take another good or stop taking
    EX_GIVE = 3  # player: exchange in progress, give back one card per good taken
    REFILL = 4  # chance: draw a card from the deck into the market
    BONUS = 5  # chance: draw a bonus token for a large sale


@dataclass(frozen=True)
class CamelStocksState(State):
    """A complete Camel Stocks position.

    Attributes:
        deck: Card counts per type left in the draw deck.
        market: Card counts per type in the face-up market (5 cards between turns).
        hands: Per-player goods counts in hand.
        herds: Per-player camel counts (public).
        tokens: Per-good value tokens left, top of the stack first (public).
        bonus: Bonus token values left for 3-, 4- and 5+-card sales, sorted (public in v1).
        points: Per-player points from tokens so far (public in v1).
        sold: Goods counts sold so far (out of the game, public).
        known: Per-player goods counts in hand that the *other* player knows about: taken from
            the market and not yet seen leaving the hand.
        to_move: The player to act at player nodes.
        phase: The current :class:`Phase`.
        deal_left: Cards still to deal (market first, then each player's hand).
        refill: Market cards still to draw this turn.
        bonus_tier: Which bonus pile the pending bonus draw comes from (``-1`` if none).
        taken: Goods taken so far in the exchange in progress.
        given: Cards (goods, then camels) given so far in the exchange in progress.
        turn: Completed turns.
        terminal: Whether the game is over.
    """

    deck: Counts
    market: Counts
    hands: tuple[Counts, Counts]
    herds: tuple[int, int]
    tokens: tuple[tuple[int, ...], ...]
    bonus: tuple[tuple[int, ...], ...]
    points: tuple[int, int]
    sold: Counts
    known: tuple[Counts, Counts]
    to_move: int
    phase: Phase
    deal_left: int
    refill: int = 0
    bonus_tier: int = -1
    taken: Counts = _NO_GOODS
    given: Counts = _NO_CARDS
    turn: int = 0
    terminal: bool = False


def initial_state() -> CamelStocksState:
    """Return the pre-deal state: camels in the market, everything else in the deck."""
    deck = list(CARD_COUNTS)
    deck[CAMEL] -= MARKET_START_CAMELS
    market = [0] * NUM_CARD_TYPES
    market[CAMEL] = MARKET_START_CAMELS
    return CamelStocksState(
        deck=tuple(deck),
        market=tuple(market),
        hands=(_NO_GOODS, _NO_GOODS),
        herds=(0, 0),
        tokens=GOOD_TOKENS,
        bonus=tuple(tuple(sorted(b)) for b in BONUS_TOKENS),
        points=(0, 0),
        sold=_NO_GOODS,
        known=(_NO_GOODS, _NO_GOODS),
        to_move=0,
        phase=Phase.DEAL,
        deal_left=MARKET_SIZE - MARKET_START_CAMELS + 2 * DEAL_SIZE,
    )
