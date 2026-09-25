"""Immutable state for Ventures.

Everything is plain tuples so the state is hashable, trivially cloneable, and immutable — the
functional contract the rest of the repo relies on (``apply_action`` returns a new state; the
conformance suite's no-mutation check passes for free).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from iirl.core.game import State
from iirl.games.ventures.cards import NUM_SUITS, initial_deck_counts

# A route is the sequence of card-type ids played on it, sorted backing-first then ascending. We
# store the exact cards (not a summary), keeping public info intact and letting resampling be exact;
# derived quantities (backing count, max number, number sum) come from helpers in game.py.
Route = tuple[int, ...]
EMPTY_ROUTE: Route = ()
_NO_CARDS: tuple[int, ...] = (0,) * len(initial_deck_counts())


class Phase(IntEnum):
    """Which sub-step of a turn the state sits at."""

    DEAL = 0  # chance: deal the opening hands one card at a time
    PLAY = 1  # player: play or discard a hand card
    DRAW = 2  # player: choose a draw source (deck or a discard pile)
    DRAW_DECK = 3  # chance: resolve which card is drawn from the deck


@dataclass(frozen=True)
class VenturesState(State):
    """A complete Ventures position.

    Attributes:
        deck: Counts per card-type id remaining in the draw deck.
        hands: Per-player counts per card-type id held in hand.
        routes: Per-player, per-suit :data:`Route` summaries of cards played.
        discards: Per-suit discard stacks (tuples of card-type ids; last element is the top).
        to_move: The player to act at player nodes.
        phase: The current :class:`Phase`.
        just_discarded_suit: Suit discarded to this turn (-1 if none), so it cannot be drawn back.
        terminal: Whether the game is over.
        turn: Completed turns so far (the game also ends at ``MAX_TURNS``).
        known: Per-player counts per card-type of hand cards the *other* player knows about —
            cards seen taken from a discard pile and not yet seen leaving the hand. Public.
    """

    deck: tuple[int, ...]
    hands: tuple[tuple[int, ...], tuple[int, ...]]
    routes: tuple[tuple[Route, ...], tuple[Route, ...]]
    discards: tuple[tuple[int, ...], ...]
    to_move: int
    phase: Phase
    just_discarded_suit: int
    terminal: bool
    turn: int = 0
    known: tuple[tuple[int, ...], tuple[int, ...]] = (_NO_CARDS, _NO_CARDS)


def initial_state() -> VenturesState:
    """Return the pre-deal state: a full deck, empty hands, at the dealing chance node."""
    empty_hand = (0,) * len(initial_deck_counts())
    empty_routes = (EMPTY_ROUTE,) * NUM_SUITS
    return VenturesState(
        deck=initial_deck_counts(),
        hands=(empty_hand, empty_hand),
        routes=(empty_routes, empty_routes),
        discards=((),) * NUM_SUITS,
        to_move=0,
        phase=Phase.DEAL,
        just_discarded_suit=-1,
        terminal=False,
    )
