"""Mini-Ventures: Ventures shrunk until it can be solved exactly.

The same mechanics as :mod:`iirl.games.ventures` (ascending single-suit routes, a backing card
that multiplies a route, a start cost, play-or-discard then draw from the deck or a discard pile,
the game ending when the deck runs out), at a size where the whole game fits in memory: 2 suits of
{backing, 2, 3, 4} (8 unique cards), 2-card hands and a 4-card deck. Each player's hidden hand has
at most C(8, 2) = 28 possibilities, which is what exact public-belief-state methods (ReBeL) and
exact exploitability need. It exists to validate methods against ground truth before scaling them
to Ventures itself.

Two simplifications keep the game tree small, and both keep discard-taking (the mechanic behind
the Ventures agent's "leaky heuristic" dependence):

* **One-slot discard piles.** Each pile holds only its top card; discarding onto a pile buries the
  old top for good.
* **Taking a discard burns the top deck card**, face up. Every turn then shrinks the deck, so a game
  is exactly four turns and needs no turn cap.

Buried and burned cards are public (``gone``).

Action ids: play card ``c``: ``c`` | discard ``c``: ``8 + c`` | draw deck: ``16`` | take the top of
suit ``s``'s pile: ``17 + s``. Chance actions are card ids.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from itertools import combinations

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility

NUM_SUITS = 2
RANKS: tuple[int, ...] = (0, 2, 3, 4)  # 0 is the backing card
RANKS_PER_SUIT = len(RANKS)
NUM_CARDS = NUM_SUITS * RANKS_PER_SUIT
HAND_SIZE = 2
DECK_AFTER_DEAL = NUM_CARDS - 2 * HAND_SIZE
START_COST = 3
BONUS_CARDS = 3  # a route this long earns ...
BONUS_POINTS = 5  # ... this bonus

DISCARD_OFFSET = NUM_CARDS
DRAW_DECK_ACTION = 2 * NUM_CARDS
TAKE_DISCARD_OFFSET = DRAW_DECK_ACTION + 1
NUM_ACTIONS = TAKE_DISCARD_OFFSET + NUM_SUITS

Counts = tuple[int, ...]
Route = tuple[int, ...]
_NONE: Counts = (0,) * NUM_CARDS
_HAND_INDEX = {hand: i for i, hand in enumerate(combinations(range(NUM_CARDS), HAND_SIZE))}


def suit_of(card: int) -> int:
    """Return a card's suit."""
    return card // RANKS_PER_SUIT


def rank_of(card: int) -> int:
    """Return a card's rank (0 for the backing card)."""
    return RANKS[card % RANKS_PER_SUIT]


def route_value(route: Route) -> int:
    """Score a route: (sum − start cost) × (1 + backings), plus the bonus; 0 if unstarted."""
    if not route:
        return 0
    backings = sum(1 for c in route if rank_of(c) == 0)
    value = (sum(rank_of(c) for c in route) - START_COST) * (1 + backings)
    return value + (BONUS_POINTS if len(route) >= BONUS_CARDS else 0)


def can_play(route: Route, card: int) -> bool:
    """Whether ``card`` may go on ``route``: backing before any number, numbers ascending."""
    top = max((rank_of(c) for c in route), default=0)
    return top == 0 if rank_of(card) == 0 else rank_of(card) > top


class Phase(IntEnum):
    """Which step of a turn the state sits at."""

    DEAL = 0
    PLAY = 1
    DRAW = 2
    DRAW_DECK = 3  # chance: the card drawn from the deck
    BURN = 4  # chance: the deck card burned after taking a discard


@dataclass(frozen=True)
class MiniVenturesState(State):
    """A complete Mini-Ventures position (plain tuples: hashable and immutable).

    ``known[p]`` counts cards in ``p``'s hand that the other player saw ``p`` take from a discard
    pile and has not seen leave it. ``gone`` counts buried and burned cards. Both are public.
    """

    deck: Counts = (1,) * NUM_CARDS
    hands: tuple[Counts, Counts] = (_NONE, _NONE)
    routes: tuple[tuple[Route, ...], tuple[Route, ...]] = (((),) * NUM_SUITS, ((),) * NUM_SUITS)
    discards: tuple[int, ...] = (-1,) * NUM_SUITS  # top card per suit, -1 if empty
    known: tuple[Counts, Counts] = (_NONE, _NONE)
    gone: Counts = _NONE
    to_move: int = 0
    phase: Phase = Phase.DEAL
    just_discarded_suit: int = -1
    terminal: bool = False


def _bump(counts: Counts, i: int, d: int) -> Counts:
    lst = list(counts)
    lst[i] += d
    return tuple(lst)


def _pair(pair: tuple, p: int, value) -> tuple:  # noqa: ANN001 - per-player pair update
    return (value, pair[1]) if p == 0 else (pair[0], value)


class MiniVentures(Game):
    """Two-player, zero-sum Mini-Ventures."""

    _TYPE = GameType(
        short_name="mini_ventures",
        long_name="Mini-Ventures",
        num_players=2,
        information=Information.IMPERFECT,
        utility=Utility.ZERO_SUM,
        provides_information_state_tensor=True,
        provides_observation_tensor=False,
    )

    @property
    def game_type(self) -> GameType:
        """Static metadata."""
        return self._TYPE

    def new_initial_state(self) -> MiniVenturesState:
        """Return the pre-deal chance state."""
        return MiniVenturesState()

    def num_distinct_actions(self) -> int:
        """Return the flat action-space size."""
        return NUM_ACTIONS

    def current_player(self, state: State) -> Player:
        """Return the player to act, or a chance/terminal sentinel."""
        assert isinstance(state, MiniVenturesState)
        if state.terminal:
            return PlayerId.TERMINAL
        if self.is_chance_node(state):
            return PlayerId.CHANCE
        return state.to_move

    def is_chance_node(self, state: State) -> bool:
        """Deal, deck draws and burns are chance nodes."""
        assert isinstance(state, MiniVenturesState)
        return not state.terminal and state.phase in (Phase.DEAL, Phase.DRAW_DECK, Phase.BURN)

    def is_terminal(self, state: State) -> bool:
        """Whether the game is over."""
        assert isinstance(state, MiniVenturesState)
        return state.terminal

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Each card left in the deck, uniformly."""
        assert isinstance(state, MiniVenturesState)
        total = sum(state.deck)
        return [(c, n / total) for c, n in enumerate(state.deck) if n]

    def legal_actions(self, state: State) -> list[Action]:
        """Play or discard a hand card; then draw from the deck or take a discard."""
        assert isinstance(state, MiniVenturesState)
        p = state.to_move
        if state.phase == Phase.PLAY:
            actions: list[Action] = []
            for c in range(NUM_CARDS):
                if state.hands[p][c]:
                    if can_play(state.routes[p][suit_of(c)], c):
                        actions.append(c)
                    actions.append(DISCARD_OFFSET + c)
            return actions
        return [DRAW_DECK_ACTION] + [
            TAKE_DISCARD_OFFSET + s
            for s in range(NUM_SUITS)
            if state.discards[s] >= 0 and s != state.just_discarded_suit
        ]

    def apply_action(self, state: State, action: Action) -> MiniVenturesState:
        """Return the next state (never mutates ``state``)."""
        assert isinstance(state, MiniVenturesState)
        p = state.to_move
        if state.phase == Phase.DEAL:
            target = 0 if sum(state.hands[0]) <= sum(state.hands[1]) else 1
            hands = _pair(state.hands, target, _bump(state.hands[target], action, 1))
            done = sum(hands[0]) + sum(hands[1]) == 2 * HAND_SIZE
            return replace(
                state,
                deck=_bump(state.deck, action, -1),
                hands=hands,
                phase=Phase.PLAY if done else Phase.DEAL,
            )
        if state.phase == Phase.PLAY:
            card = action if action < DISCARD_OFFSET else action - DISCARD_OFFSET
            hands = _pair(state.hands, p, _bump(state.hands[p], card, -1))
            known = state.known
            if known[p][card]:
                known = _pair(known, p, _bump(known[p], card, -1))
            if action < DISCARD_OFFSET:
                routes = list(state.routes[p])
                routes[suit_of(card)] = tuple(sorted(routes[suit_of(card)] + (card,), key=rank_of))
                return replace(
                    state,
                    hands=hands,
                    known=known,
                    routes=_pair(state.routes, p, tuple(routes)),
                    phase=Phase.DRAW,
                    just_discarded_suit=-1,
                )
            s, gone = suit_of(card), state.gone
            if state.discards[s] >= 0:
                gone = _bump(gone, state.discards[s], 1)  # the old top is buried
            discards = list(state.discards)
            discards[s] = card
            return replace(
                state,
                hands=hands,
                known=known,
                discards=tuple(discards),
                gone=gone,
                phase=Phase.DRAW,
                just_discarded_suit=s,
            )
        if state.phase == Phase.DRAW:
            if action == DRAW_DECK_ACTION:
                return replace(state, phase=Phase.DRAW_DECK)
            s = action - TAKE_DISCARD_OFFSET
            top = state.discards[s]
            discards = list(state.discards)
            discards[s] = -1
            return replace(
                state,
                hands=_pair(state.hands, p, _bump(state.hands[p], top, 1)),
                known=_pair(state.known, p, _bump(state.known[p], top, 1)),
                discards=tuple(discards),
                phase=Phase.BURN,
            )
        deck = _bump(state.deck, action, -1)
        if state.phase == Phase.DRAW_DECK:  # the drawn card goes to the player who chose the deck
            nxt = replace(
                state, deck=deck, hands=_pair(state.hands, p, _bump(state.hands[p], action, 1))
            )
        else:  # Phase.BURN
            nxt = replace(state, deck=deck, gone=_bump(state.gone, action, 1))
        if sum(deck) == 0:
            return replace(nxt, terminal=True)
        return replace(nxt, to_move=1 - p, phase=Phase.PLAY, just_discarded_suit=-1)

    def score(self, state: State, player: int) -> int:
        """Return ``player``'s total route value."""
        assert isinstance(state, MiniVenturesState)
        return sum(route_value(r) for r in state.routes[player])

    def returns(self, state: State) -> list[float]:
        """Zero-sum payoffs: the score difference."""
        diff = self.score(state, 0) - self.score(state, 1)
        return [float(diff), float(-diff)]

    def heuristic_value(self, state: State, player: Player) -> float:
        """Score difference counting playable hand cards at a discount (a rough leaf value)."""
        assert isinstance(state, MiniVenturesState)

        def potential(p: int) -> float:
            total = 0.0
            for s in range(NUM_SUITS):
                route = state.routes[p][s]
                extra = [
                    c
                    for c in range(s * RANKS_PER_SUIT, (s + 1) * RANKS_PER_SUIT)
                    if state.hands[p][c] and rank_of(c) and can_play(route, c)
                ]
                value = route_value(tuple(sorted(route + tuple(extra), key=rank_of)))
                total += value if route else max(0.6 * value, 0.0)
            return total

        diff = potential(0) - potential(1)
        return diff if player == 0 else -diff

    def _unseen(self, state: MiniVenturesState, player: int) -> list[int]:
        """Per card, 1 if ``player`` cannot see where it is (the deck or the opponent's hand)."""
        opp = 1 - player
        out = []
        for c in range(NUM_CARDS):
            seen = state.hands[player][c] + state.known[opp][c] + state.gone[c]
            seen += sum(r.count(c) for pr in state.routes for r in pr)
            seen += state.discards.count(c)
            out.append(1 - seen)
        return out

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Own hand, the public board, unseen and known cards, per card; plus a few scalars."""
        assert isinstance(state, MiniVenturesState)
        opp = 1 - player

        def per_card(pred) -> list[float]:  # noqa: ANN001
            return [float(pred(c)) for c in range(NUM_CARDS)]

        return np.asarray(
            [
                *state.hands[player],
                *per_card(lambda c: can_play(state.routes[player][suit_of(c)], c)),
                *per_card(lambda c: can_play(state.routes[opp][suit_of(c)], c)),
                *self._unseen(state, player),
                *state.known[opp],
                *state.known[player],
                *state.gone,
                *per_card(lambda c: c in state.discards),
                *per_card(lambda c: any(c in r for r in state.routes[player])),
                *per_card(lambda c: any(c in r for r in state.routes[opp])),
                *(float(state.just_discarded_suit == s) for s in range(NUM_SUITS)),
                sum(state.deck) / DECK_AFTER_DEAL,
                self.score(state, player) / 10.0,
                self.score(state, opp) / 10.0,
                float(state.phase == Phase.PLAY),
                float(state.phase == Phase.DRAW),
            ],
            dtype=np.float32,
        )

    def information_state_string(self, state: State, player: Player) -> str:
        """Exactly what ``player`` observes."""
        assert isinstance(state, MiniVenturesState)
        return (
            f"P{player}|h{state.hands[player]}|r{state.routes}|d{state.discards}|g{state.gone}"
            f"|k{sum(state.deck)}|kn{state.known}|tm{state.to_move}|ph{int(state.phase)}"
            f"|jd{state.just_discarded_suit}"
        )

    def private_hand(self, state: State, player: Player) -> Counts:
        """``player``'s hand as counts per card."""
        assert isinstance(state, MiniVenturesState)
        return state.hands[player]

    def known_cards(self, state: State, player: Player) -> Counts:
        """Cards publicly known to be in ``player``'s hand (seen taken from a discard pile)."""
        assert isinstance(state, MiniVenturesState)
        return state.known[player]

    def unseen_cards(self, state: State) -> Counts:
        """Counts of the cards in the deck or in a hand: public knowledge."""
        assert isinstance(state, MiniVenturesState)
        return tuple(d + a + b for d, a, b in zip(state.deck, *state.hands, strict=True))

    def with_hands(
        self, state: State, player: Player, hand: Counts, other_hand: Counts
    ) -> MiniVenturesState:
        """The same public position, ``player`` holding ``hand`` and the other ``other_hand``."""
        assert isinstance(state, MiniVenturesState)
        unseen = self.unseen_cards(state)
        deck = tuple(u - a - b for u, a, b in zip(unseen, hand, other_hand, strict=True))
        if min(deck) < 0:
            raise ValueError("the hands do not fit the public position")
        hands = (hand, other_hand) if player == 0 else (other_hand, hand)
        return replace(state, deck=deck, hands=hands)

    def chance_is_private(self, state: State) -> bool:
        """Whether the pending chance outcome is seen only by the mover (a deck draw)."""
        assert isinstance(state, MiniVenturesState)
        return state.phase == Phase.DRAW_DECK

    def num_private_states(self) -> int:
        """Number of distinct full hands (a player's private state at the start of a turn)."""
        return len(_HAND_INDEX)

    def private_state_features(self) -> np.ndarray:
        """Per full hand, which cards it holds: (hands, cards)."""
        out = np.zeros((len(_HAND_INDEX), NUM_CARDS), dtype=np.float32)
        for hand, i in _HAND_INDEX.items():
            out[i, list(hand)] = 1.0
        return out

    def private_state_index(self, state: State, player: Player) -> int:
        """Index of ``player``'s hand among all full hands (start-of-turn states only)."""
        assert isinstance(state, MiniVenturesState)
        return _HAND_INDEX[tuple(c for c, n in enumerate(state.hands[player]) if n)]

    def public_state_string(self, state: State) -> str:
        """What both players observe: everything except the hands and the deck's contents."""
        assert isinstance(state, MiniVenturesState)
        return (
            f"r{state.routes}|d{state.discards}|g{state.gone}|k{sum(state.deck)}|kn{state.known}"
            f"|tm{state.to_move}|ph{int(state.phase)}|jd{state.just_discarded_suit}"
        )

    def public_state_tensor(self, state: State) -> np.ndarray:
        """The public state as features, from the point of view of the player to move."""
        assert isinstance(state, MiniVenturesState)
        me, opp = state.to_move, 1 - state.to_move

        def per_card(pred) -> list[float]:  # noqa: ANN001
            return [float(pred(c)) for c in range(NUM_CARDS)]

        return np.asarray(
            [
                *per_card(lambda c: any(c in r for r in state.routes[me])),
                *per_card(lambda c: any(c in r for r in state.routes[opp])),
                *per_card(lambda c: can_play(state.routes[me][suit_of(c)], c)),
                *per_card(lambda c: can_play(state.routes[opp][suit_of(c)], c)),
                *per_card(lambda c: c in state.discards),
                *state.gone,
                *state.known[me],
                *state.known[opp],
                sum(state.deck) / DECK_AFTER_DEAL,
                self.score(state, me) / 10.0,
                self.score(state, opp) / 10.0,
            ],
            dtype=np.float32,
        )

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> MiniVenturesState:
        """Re-deal the opponent's unknown cards and the deck uniformly from the unseen cards."""
        assert isinstance(state, MiniVenturesState)
        opp = 1 - player
        unseen = [c for c, n in enumerate(self._unseen(state, player)) if n]
        rng.shuffle(unseen)
        k = sum(state.hands[opp]) - sum(state.known[opp])
        hand = list(state.known[opp])
        deck = [0] * NUM_CARDS
        for c in unseen[:k]:
            hand[c] += 1
        for c in unseen[k:]:
            deck[c] += 1
        return replace(state, deck=tuple(deck), hands=_pair(state.hands, opp, tuple(hand)))

    def action_to_string(self, state: State, action: Action) -> str:
        """Readable action label."""
        assert isinstance(state, MiniVenturesState)

        def name(c: int) -> str:
            return f"S{suit_of(c)}-{'B' if rank_of(c) == 0 else rank_of(c)}"

        if state.phase == Phase.BURN:
            return f"burn {name(action)}"
        if state.phase in (Phase.DEAL, Phase.DRAW_DECK):
            return f"draw {name(action)}"
        if action < DISCARD_OFFSET:
            return f"play {name(action)}"
        if action < DRAW_DECK_ACTION:
            return f"discard {name(action - DISCARD_OFFSET)}"
        if action == DRAW_DECK_ACTION:
            return "draw from deck"
        return f"take S{action - TAKE_DISCARD_OFFSET} discard"


@register("mini_ventures")
def _make_mini_ventures() -> MiniVentures:
    """Factory registered under ``"mini_ventures"``."""
    return MiniVentures()
