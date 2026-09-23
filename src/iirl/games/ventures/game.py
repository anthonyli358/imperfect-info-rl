"""The Ventures engine: an original game on a classic mechanical skeleton.

See :mod:`iirl.games.ventures.cards` for the card model and the IP/credit note. This module
implements the game-agnostic :class:`~iirl.core.game.Game` contract, including
``resample_from_infostate`` (the hook determinized MCTS uses).

Turn structure: a player first plays or discards a card (``Phase.PLAY``), then chooses a draw source
(``Phase.DRAW``); drawing from the deck defers to a chance node (``Phase.DRAW_DECK``). The deck is a
multiset of card counts, so each deck draw is a first-class chance node rather than a fixed shuffle.
"""

from __future__ import annotations

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility
from iirl.games.ventures.cards import (
    BONUS_CARDS,
    BONUS_POINTS,
    DECK_SIZE,
    HAND_SIZE,
    NUM_CARD_TYPES,
    NUM_SUITS,
    RANKS,
    START_COST,
    SUIT_COLORS,
    SUIT_NAMES,
    initial_deck_counts,
    is_backing,
    rank_of,
    suit_of,
)
from iirl.games.ventures.cards import (
    card_name as _card_name,
)
from iirl.games.ventures.state import Phase, Route, VenturesState, initial_state

# --- Flat action-id layout ----------------------------------------------------------------------
# Play card-type c: 0..49 | Discard c: 50..99 | Draw deck: 100 | Take discard of suit s: 101..105
DISCARD_OFFSET = NUM_CARD_TYPES
DRAW_DECK_ACTION = 2 * NUM_CARD_TYPES
TAKE_DISCARD_OFFSET = DRAW_DECK_ACTION + 1
NUM_ACTIONS = TAKE_DISCARD_OFFSET + NUM_SUITS


# --- Route helpers (a route is a tuple of card-type ids, sorted backing-first then ascending) ----
def route_backing_count(route: Route) -> int:
    """Return how many backing cards are on a route."""
    return sum(1 for c in route if is_backing(c))


def route_max_number(route: Route) -> int:
    """Return the highest number on a route, or 0 if it has none."""
    return max((rank_of(c) for c in route if not is_backing(c)), default=0)


def route_number_sum(route: Route) -> int:
    """Return the sum of the numbers on a route."""
    return sum(rank_of(c) for c in route if not is_backing(c))


def _can_play(route: Route, card: int) -> bool:
    """Return whether ``card`` may be played onto ``route`` (backing before numbers; ascending)."""
    if is_backing(card):
        return route_max_number(route) == 0  # backing only before any number
    return rank_of(card) > route_max_number(route)  # numbers strictly increasing


# --- Immutable tuple updates --------------------------------------------------------------------
def _bump(counts: tuple[int, ...], idx: int, delta: int) -> tuple[int, ...]:
    lst = list(counts)
    lst[idx] += delta
    return tuple(lst)


def _hand_bump(
    hands: tuple[tuple[int, ...], tuple[int, ...]], player: int, card: int, delta: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    new = list(hands)
    new[player] = _bump(hands[player], card, delta)
    return (new[0], new[1])


def _route_play(
    routes: tuple[tuple[Route, ...], tuple[Route, ...]], player: int, card: int
) -> tuple[tuple[Route, ...], tuple[Route, ...]]:
    suit = suit_of(card)
    prow = list(routes[player])
    prow[suit] = tuple(sorted(prow[suit] + (card,), key=rank_of))
    new = list(routes)
    new[player] = tuple(prow)
    return (new[0], new[1])


def _discard_push(
    discards: tuple[tuple[int, ...], ...], suit: int, card: int
) -> tuple[tuple[int, ...], ...]:
    new = list(discards)
    new[suit] = discards[suit] + (card,)
    return tuple(new)


def _discard_pop(
    discards: tuple[tuple[int, ...], ...], suit: int
) -> tuple[tuple[tuple[int, ...], ...], int]:
    top = discards[suit][-1]
    new = list(discards)
    new[suit] = discards[suit][:-1]
    return tuple(new), top


class Ventures(Game):
    """Two-player, zero-sum Ventures (imperfect information)."""

    _TYPE = GameType(
        short_name="ventures",
        long_name="Ventures",
        num_players=2,
        information=Information.IMPERFECT,
        utility=Utility.ZERO_SUM,
        provides_information_state_tensor=True,
        provides_observation_tensor=False,
    )

    @property
    def game_type(self) -> GameType:
        """Static metadata for this game."""
        return self._TYPE

    def new_initial_state(self) -> VenturesState:
        """Return the pre-deal chance state."""
        return initial_state()

    def num_distinct_actions(self) -> int:
        """Return the flat action-space size (see the action-id layout above)."""
        return NUM_ACTIONS

    # --- Node classification --------------------------------------------------------------------
    def current_player(self, state: State) -> Player:
        """Return the player to act, or a chance/terminal sentinel."""
        assert isinstance(state, VenturesState)
        if state.terminal:
            return PlayerId.TERMINAL
        if state.phase in (Phase.DEAL, Phase.DRAW_DECK):
            return PlayerId.CHANCE
        return state.to_move

    def is_chance_node(self, state: State) -> bool:
        """Return whether the state is a deal/deck chance node."""
        assert isinstance(state, VenturesState)
        return not state.terminal and state.phase in (Phase.DEAL, Phase.DRAW_DECK)

    def is_terminal(self, state: State) -> bool:
        """Return whether the game is over."""
        assert isinstance(state, VenturesState)
        return state.terminal

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Return the deck draw distribution: each remaining card-type ∝ its count."""
        assert isinstance(state, VenturesState)
        total = sum(state.deck)
        return [(c, count / total) for c, count in enumerate(state.deck) if count > 0]

    # --- Legal actions --------------------------------------------------------------------------
    def legal_actions(self, state: State) -> list[Action]:
        """Return legal actions at a player node (play/discard, or choose a draw source)."""
        assert isinstance(state, VenturesState)
        p = state.to_move
        if state.phase == Phase.PLAY:
            actions: list[Action] = []
            for c in range(NUM_CARD_TYPES):
                if state.hands[p][c] == 0:
                    continue
                if _can_play(state.routes[p][suit_of(c)], c):
                    actions.append(c)
                actions.append(DISCARD_OFFSET + c)
            return actions
        # Phase.DRAW
        actions = []
        if sum(state.deck) > 0:
            actions.append(DRAW_DECK_ACTION)
        for s in range(NUM_SUITS):
            if state.discards[s] and s != state.just_discarded_suit:
                actions.append(TAKE_DISCARD_OFFSET + s)
        return actions

    # --- Transitions ----------------------------------------------------------------------------
    def apply_action(self, state: State, action: Action) -> VenturesState:
        """Return the state reached by applying ``action`` (does not mutate ``state``)."""
        assert isinstance(state, VenturesState)
        if state.phase == Phase.DEAL:
            return self._apply_deal(state, action)
        if state.phase == Phase.PLAY:
            return self._apply_play(state, action)
        if state.phase == Phase.DRAW:
            return self._apply_draw_choice(state, action)
        return self._apply_deck_draw(state, action)  # Phase.DRAW_DECK

    def _apply_deal(self, state: VenturesState, card: int) -> VenturesState:
        sizes = (sum(state.hands[0]), sum(state.hands[1]))
        target = 0 if sizes[0] <= sizes[1] else 1
        hands = _hand_bump(state.hands, target, card, +1)
        deck = _bump(state.deck, card, -1)
        dealt = sum(hands[0]) + sum(hands[1])
        if dealt == 2 * HAND_SIZE:
            return _replace(state, deck=deck, hands=hands, phase=Phase.PLAY, to_move=0)
        return _replace(state, deck=deck, hands=hands, phase=Phase.DEAL)

    def _apply_play(self, state: VenturesState, action: Action) -> VenturesState:
        p = state.to_move
        if action < DISCARD_OFFSET:  # play
            card = action
            hands = _hand_bump(state.hands, p, card, -1)
            routes = _route_play(state.routes, p, card)
            return _replace(
                state, hands=hands, routes=routes, phase=Phase.DRAW, just_discarded_suit=-1
            )
        if action < DRAW_DECK_ACTION:  # discard
            card = action - DISCARD_OFFSET
            hands = _hand_bump(state.hands, p, card, -1)
            discards = _discard_push(state.discards, suit_of(card), card)
            return _replace(
                state,
                hands=hands,
                discards=discards,
                phase=Phase.DRAW,
                just_discarded_suit=suit_of(card),
            )
        raise ValueError(f"Illegal play-phase action {action}")

    def _apply_draw_choice(self, state: VenturesState, action: Action) -> VenturesState:
        if action == DRAW_DECK_ACTION:
            return _replace(state, phase=Phase.DRAW_DECK)  # defer to chance node
        s = action - TAKE_DISCARD_OFFSET
        if not (0 <= s < NUM_SUITS):
            raise ValueError(f"Illegal draw-phase action {action}")
        discards, top = _discard_pop(state.discards, s)
        hands = _hand_bump(state.hands, state.to_move, top, +1)
        return _replace(
            state,
            hands=hands,
            discards=discards,
            phase=Phase.PLAY,
            to_move=1 - state.to_move,
            just_discarded_suit=-1,
        )

    def _apply_deck_draw(self, state: VenturesState, card: int) -> VenturesState:
        p = state.to_move
        deck = _bump(state.deck, card, -1)
        hands = _hand_bump(state.hands, p, card, +1)
        if sum(deck) == 0:  # last card drawn ends the game immediately
            return _replace(state, deck=deck, hands=hands, terminal=True)
        return _replace(
            state, deck=deck, hands=hands, phase=Phase.PLAY, to_move=1 - p, just_discarded_suit=-1
        )

    # --- Terminal payoffs -----------------------------------------------------------------------
    def _score(self, state: VenturesState, player: int) -> int:
        total = 0
        for s in range(NUM_SUITS):
            route = state.routes[player][s]
            if not route:
                continue  # unstarted route scores nothing
            base = (route_number_sum(route) - START_COST) * (1 + route_backing_count(route))
            if len(route) >= BONUS_CARDS:
                base += BONUS_POINTS
            total += base
        return total

    def returns(self, state: State) -> list[float]:
        """Return zero-sum payoffs: each player's score minus the opponent's."""
        assert isinstance(state, VenturesState)
        diff = self._score(state, 0) - self._score(state, 1)
        return [float(diff), float(-diff)]

    def _potential_score(self, state: VenturesState, player: int) -> float:
        """Estimate ``player``'s final score, crediting hand cards they can still play in time.

        Realized score alone is a poor leaf evaluation: an in-progress route always shows its full
        −20 start cost, so a searcher guided by it refuses to commit — the opposite of good play.
        This projection instead assumes the player extends their routes with the playable,
        increasing number cards still in hand — but only as many as the deck clock allows: at most
        about half the remaining deck draws are this player's own turns, so the highest-value cards
        are allocated within that budget. An unstarted route is only counted if its projection beats
        the start cost (you would not open a losing venture). Both players are scored identically,
        so the difference is a fair relative signal.
        """
        plays_left = (
            sum(state.deck) // 2
        )  # this player's own remaining turns before the deck empties
        # All playable hand cards (each strictly above its route's current max), best first.
        candidates: list[tuple[int, int]] = []  # (value, suit)
        for s in range(NUM_SUITS):
            max_num = route_max_number(state.routes[player][s])
            for c in range(NUM_CARD_TYPES):
                if suit_of(c) == s and not is_backing(c) and rank_of(c) > max_num:
                    candidates.extend([(rank_of(c), s)] * state.hands[player][c])
        candidates.sort(reverse=True)
        add_sum = [0] * NUM_SUITS
        add_count = [0] * NUM_SUITS
        for value, s in candidates[:plays_left]:
            add_sum[s] += value
            add_count[s] += 1

        total = 0.0
        for s in range(NUM_SUITS):
            route = state.routes[player][s]
            started = len(route) > 0
            projected_sum = route_number_sum(route) + add_sum[s]
            projected_count = len(route) + add_count[s]
            if not started and projected_sum < START_COST:
                continue  # would not open a losing route
            value = (projected_sum - START_COST) * (1 + route_backing_count(route))
            if projected_count >= BONUS_CARDS:
                value += BONUS_POINTS
            total += value if started else max(value, 0.0)
        return total

    def heuristic_value(self, state: State, player: Player) -> float:
        """Return a potential-aware zero-sum leaf estimate for ``player`` (see _potential_score)."""
        assert isinstance(state, VenturesState)
        diff = self._potential_score(state, 0) - self._potential_score(state, 1)
        return float(diff) if player == 0 else float(-diff)

    # --- Encoders / determinization -------------------------------------------------------------
    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Encode ``player``'s view: own hand + all public routes/discards + deck size + phase.

        The opponent's hand and the deck's contents are never encoded — that is the hidden
        information. Shape is a constant regardless of game progress.
        """
        assert isinstance(state, VenturesState)
        opp = 1 - player
        parts: list[np.ndarray] = []
        # Own hand counts (private).
        parts.append(np.asarray(state.hands[player], dtype=np.float32))
        # Both players' routes (public), own first: (backing, count, sum, max) per suit.
        for pp in (player, opp):
            feats = []
            for s in range(NUM_SUITS):
                r = state.routes[pp][s]
                feats += [route_backing_count(r), len(r), route_number_sum(r), route_max_number(r)]
            parts.append(np.asarray(feats, dtype=np.float32))
        # Discards (public): counts per card-type across all piles, plus each pile's top rank.
        disc_counts = np.zeros(NUM_CARD_TYPES, dtype=np.float32)
        tops = np.zeros(NUM_SUITS, dtype=np.float32)
        for s in range(NUM_SUITS):
            for c in state.discards[s]:
                disc_counts[c] += 1.0
            if state.discards[s]:
                tops[s] = rank_of(state.discards[s][-1])
        parts.append(disc_counts)
        parts.append(tops)
        # Public scalars / flags.
        parts.append(np.asarray([sum(state.deck) / DECK_SIZE], dtype=np.float32))
        phase_onehot = np.zeros(2, dtype=np.float32)
        phase_onehot[0 if state.phase == Phase.PLAY else 1] = 1.0
        parts.append(phase_onehot)
        jd = np.zeros(NUM_SUITS, dtype=np.float32)
        if state.just_discarded_suit >= 0:
            jd[state.just_discarded_suit] = 1.0
        parts.append(jd)
        return np.concatenate(parts)

    def information_state_string(self, state: State, player: Player) -> str:
        """Return a hashable infoset key from exactly what ``player`` observes."""
        assert isinstance(state, VenturesState)
        opp = 1 - player
        hand = tuple(state.hands[player])
        return (
            f"P{player}|h{hand}|r{state.routes[player]}|o{state.routes[opp]}"
            f"|d{state.discards}|k{sum(state.deck)}|ph{int(state.phase)}|jd{state.just_discarded_suit}"
        )

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> VenturesState:
        """Return a world consistent with ``player``'s view, re-dealing the hidden cards.

        Preserved exactly: ``player``'s hand, both routes, all discards, deck size, and the turn
        pointers. The unseen multiset (deck minus all visible cards) is split uniformly into a fresh
        opponent hand of the correct size and a remainder deck.
        """
        assert isinstance(state, VenturesState)
        opp = 1 - player
        # Account for every card the player can see or hold.
        accounted = list(state.hands[player])
        for pp in (0, 1):
            for s in range(NUM_SUITS):
                for c in state.routes[pp][s]:
                    accounted[c] += 1
        for s in range(NUM_SUITS):
            for c in state.discards[s]:
                accounted[c] += 1
        full = initial_deck_counts()
        unseen: list[int] = []
        for c in range(NUM_CARD_TYPES):
            unseen.extend([c] * (full[c] - accounted[c]))
        rng.shuffle(unseen)
        opp_size = sum(state.hands[opp])
        opp_cards, deck_cards = unseen[:opp_size], unseen[opp_size:]
        opp_hand = [0] * NUM_CARD_TYPES
        deck = [0] * NUM_CARD_TYPES
        for c in opp_cards:
            opp_hand[c] += 1
        for c in deck_cards:
            deck[c] += 1
        hands = (
            (tuple(opp_hand), tuple(state.hands[player]))
            if player == 1
            else (
                tuple(state.hands[player]),
                tuple(opp_hand),
            )
        )
        return _replace(state, deck=tuple(deck), hands=hands)

    def action_to_string(self, state: State, action: Action) -> str:
        """Return a readable label for an action."""
        assert isinstance(state, VenturesState)
        if state.phase in (Phase.DEAL, Phase.DRAW_DECK):  # chance: the action is the drawn card id
            verb = "deal" if state.phase == Phase.DEAL else "draw"
            return f"{verb} {_card_name(action)}"
        if action < DISCARD_OFFSET:
            return f"play {_card_name(action)}"
        if action < DRAW_DECK_ACTION:
            return f"discard {_card_name(action - DISCARD_OFFSET)}"
        if action == DRAW_DECK_ACTION:
            return "draw from deck"
        return f"take {SUIT_NAMES[action - TAKE_DISCARD_OFFSET]} discard"

    def display_meta(self) -> dict:
        """Return viewer metadata: suit names/colors and the rank ordering."""
        return {
            "num_suits": NUM_SUITS,
            "suit_names": list(SUIT_NAMES),
            "suit_colors": list(SUIT_COLORS),
            "ranks": list(RANKS),
            "hand_size": HAND_SIZE,
            "start_cost": START_COST,
        }

    def to_display(self, state: State) -> dict:
        """Return a render-ready snapshot: hands, tableaus, discards, deck size, phase, scores."""
        assert isinstance(state, VenturesState)

        def cards(counts: tuple[int, ...]) -> list[int]:
            return [c for c in range(NUM_CARD_TYPES) for _ in range(counts[c])]

        return {
            "phase": state.phase.name,
            "to_move": state.to_move,
            "terminal": state.terminal,
            "deck_size": sum(state.deck),
            "hands": [cards(state.hands[0]), cards(state.hands[1])],
            "routes": [[list(state.routes[p][s]) for s in range(NUM_SUITS)] for p in (0, 1)],
            "discards": [list(state.discards[s]) for s in range(NUM_SUITS)],
            "just_discarded_suit": state.just_discarded_suit,
            "scores": [self._score(state, 0), self._score(state, 1)],
        }


def _replace(state: VenturesState, **changes: object) -> VenturesState:
    """Return a copy of ``state`` with ``changes`` applied (dataclasses.replace, typed loosely)."""
    from dataclasses import replace

    return replace(state, **changes)  # type: ignore[arg-type]


@register("ventures")
def _make_ventures() -> Ventures:
    """Factory registered under ``"ventures"``."""
    return Ventures()
