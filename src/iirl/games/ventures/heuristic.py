"""A rule-based Ventures baseline built around wager discipline and the deck clock.

Unlike the MCTS agents, this one is game-specific by design — it exists to give PIMC a non-trivial
opponent and to sanity-check the engine. It encodes the two ideas that separate competent Ventures
play from random flailing:

1. **Wager discipline.** Never open a route you cannot drag past the −20 start cost, and only add a
   backing multiplier to a route you are genuinely committed to.
2. **The deck is a shared clock.** The game ends the instant the deck empties, and *both* players
   draw from it — so you get only about half of the remaining deck draws as your own turns. The aim
   is to bank as much score as possible before then: don't open a route you can't finish in the
   turns that remain, and once the clock is short, stop starting new ventures.

**Styles.** A few knobs turn it into distinct opponents for a style-diverse training league (one
opponent with one habit taught the agent a habit-specific exploit; see ``writeup.md``). The
defaults reproduce the original heuristic exactly. Registered styles:

* ``ventures``: the original.
* ``ventures_careful``: never hands the opponent a playable card when a safe discard exists.
* ``ventures_tight``: careful; opens at most 3 routes, each only with a clearly winning hand, and
  never wagers.
* ``ventures_aggressive``: careful; opens routes on weaker hands and wagers readily.
* ``ventures_hoarder``: careful; also takes discards the opponent could play, to deny them.
"""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.registry import register_heuristic
from iirl.core.types import Action
from iirl.games.ventures.cards import NUM_SUITS, START_COST, is_backing, rank_of, suit_of
from iirl.games.ventures.game import (
    DISCARD_OFFSET,
    DRAW_DECK_ACTION,
    TAKE_DISCARD_OFFSET,
    _can_play,
    route_max_number,
    route_number_sum,
)
from iirl.games.ventures.state import Phase, VenturesState

_BACKING_MARGIN = 10  # only back a route projected comfortably past the start cost


@register_heuristic("ventures")
class VenturesHeuristic(Agent):
    """Greedy Ventures policy with wager discipline and deck-clock awareness."""

    def __init__(
        self,
        name: str = "heuristic",
        open_margin: int = 0,
        max_routes: int = NUM_SUITS,
        backing_margin: int | None = _BACKING_MARGIN,
        careful: bool = False,
        deny: bool = False,
    ) -> None:
        """Create the heuristic agent.

        Args:
            name: Label used in standings and error messages.
            open_margin: Points past the start cost a hand must project to open a route.
            max_routes: Never open more routes than this.
            backing_margin: Wager only on a route projected this far past the start cost
                (``None``: never wager).
            careful: Never discard a card the opponent can play if a safe discard exists.
            deny: With no useful discard to take, take one the opponent could play.
        """
        self.name = name
        self.open_margin, self.max_routes = open_margin, max_routes
        self.backing_margin, self.careful, self.deny = backing_margin, careful, deny

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return a legal action following the wager-discipline and deck-clock rules."""
        assert isinstance(state, VenturesState)
        me = state.to_move
        legal = set(game.legal_actions(state))
        route_max, potential, pursued = self._suit_summaries(state, me)
        if state.phase == Phase.PLAY:
            return self._play_phase(state, legal, route_max, potential, pursued)
        return self._draw_phase(state, legal, route_max, pursued, rng)

    def _suit_summaries(
        self, state: VenturesState, me: int
    ) -> tuple[list[int], list[int], list[bool]]:
        """Return per-suit (max number on route, projected number total, whether to pursue).

        A route is "pursued" if it is already started (we are committed) or — for an unstarted route
        — if we hold enough high cards of that suit to clear the start cost *and* enough of our own
        turns remain before the deck empties to actually play them. Remaining turns are estimated
        as half the remaining deck draws, since the opponent burns the other half.
        """
        hand = state.hands[me]
        my_turns_left = max(1, sum(state.deck) // 2)
        route_max, potential, pursued = [], [], []
        for s in range(NUM_SUITS):
            route = state.routes[me][s]
            on_route = route_number_sum(route)
            suit_numbers = sorted(
                (
                    rank_of(c)
                    for c in range(len(hand))
                    if suit_of(c) == s and not is_backing(c)
                    for _ in range(hand[c])
                ),
                reverse=True,
            )
            route_max.append(route_max_number(route))
            potential.append(on_route + sum(suit_numbers))
            if route:  # already started: stay committed
                pursued.append(True)
                continue
            if sum(1 for r in state.routes[me] if r) >= self.max_routes:
                pursued.append(False)  # enough routes open already
                continue
            # Unstarted: how many of our best cards clear the start cost, and is there time?
            cumulative, needed = on_route, None
            for i, v in enumerate(suit_numbers, start=1):
                cumulative += v
                if cumulative > START_COST + self.open_margin:
                    needed = i
                    break
            pursued.append(needed is not None and needed <= my_turns_left)
        return route_max, potential, pursued

    def _play_phase(
        self,
        state: VenturesState,
        legal: set[Action],
        route_max: list[int],
        potential: list[int],
        pursued: list[bool],
    ) -> Action:
        """Pick the best committed play, else discard the most expendable card."""
        best_play, best_priority = None, 0
        for a in legal:
            if a >= DISCARD_OFFSET:
                continue  # not a play action
            s = suit_of(a)
            if is_backing(a):
                good = (
                    self.backing_margin is not None
                    and pursued[s]
                    and route_max[s] == 0
                    and potential[s] >= START_COST + self.backing_margin
                )
                priority = 90 if good else 0
            else:
                good = pursued[s]
                priority = (100 - rank_of(a)) if good else 0  # prefer the lowest legal number
            if priority > best_priority:
                best_priority, best_play = priority, a
        if best_play is not None:
            return best_play

        # No committed play worth making: discard from the least-wanted suit, lowest rank first.
        discards = [a for a in legal if DISCARD_OFFSET <= a < DRAW_DECK_ACTION]
        return min(discards, key=lambda a: self._discard_key(state, a - DISCARD_OFFSET, pursued))

    def _discard_key(self, state: VenturesState, card: int, pursued: list[bool]) -> tuple:
        """Sort key for choosing a discard, lowest first.

        Gifts to the opponent go last if careful, then cards from unwanted suits, then low ranks.
        """
        return (
            self.careful and _playable_by_opponent(state, card),
            pursued[suit_of(card)],
            rank_of(card),
        )

    def _draw_phase(
        self,
        state: VenturesState,
        legal: set[Action],
        route_max: list[int],
        pursued: list[bool],
        rng: np.random.Generator,
    ) -> Action:
        """Take a useful face-up discard if one exists, else draw blind from the deck."""
        best_take, best_value = None, 0
        for a in legal:
            if a < TAKE_DISCARD_OFFSET:
                continue  # deck draw, not a take
            s = a - TAKE_DISCARD_OFFSET
            top = state.discards[s][-1]
            if not pursued[s]:
                continue
            if is_backing(top):
                value = 5 if route_max[s] == 0 else 0
            else:
                value = rank_of(top) if rank_of(top) > route_max[s] else 0
            if value > best_value:
                best_value, best_take = value, a
        if best_take is not None:
            return best_take
        if self.deny:  # take the discard the opponent wants most, if any
            wanted = [
                (rank_of(state.discards[a - TAKE_DISCARD_OFFSET][-1]), a)
                for a in legal
                if a >= TAKE_DISCARD_OFFSET
                and _playable_by_opponent(state, state.discards[a - TAKE_DISCARD_OFFSET][-1])
            ]
            if wanted:
                return max(wanted)[1]
        if DRAW_DECK_ACTION in legal:
            return DRAW_DECK_ACTION
        return int(rng.choice(list(legal)))  # forced: only discards left to draw


def _playable_by_opponent(state: VenturesState, card: int) -> bool:
    """Whether the opponent of the player to move could play ``card`` on a route they started."""
    route = state.routes[1 - state.to_move][suit_of(card)]
    return bool(route) and _can_play(route, card)


class CarefulVenturesHeuristic(VenturesHeuristic):
    """The heuristic with a careful human's discard discipline (``careful=True``).

    The plain heuristic ignores the opponent and gifts a playable card 25% of the time even when a
    safe discard exists, which is income the trained agent learned to live on.
    """

    def __init__(self, name: str = "careful", **styles: object) -> None:
        """Create the careful heuristic (other style knobs as in :class:`VenturesHeuristic`)."""
        super().__init__(name, careful=True, **styles)  # type: ignore[arg-type]


register_heuristic("ventures_careful")(CarefulVenturesHeuristic)


@register_heuristic("ventures_tight")
def tight_heuristic() -> VenturesHeuristic:
    """Careful; at most 3 routes, each only on a clearly winning hand; never wagers."""
    return CarefulVenturesHeuristic("tight", open_margin=8, max_routes=3, backing_margin=None)


@register_heuristic("ventures_aggressive")
def aggressive_heuristic() -> VenturesHeuristic:
    """Careful; opens routes on weaker hands and wagers readily."""
    return CarefulVenturesHeuristic("aggressive", open_margin=-8, backing_margin=0)


@register_heuristic("ventures_hoarder")
def hoarder_heuristic() -> VenturesHeuristic:
    """Careful; also takes discards the opponent could play, to deny them."""
    return CarefulVenturesHeuristic("hoarder", deny=True)


LEAGUE_STYLES = ("ventures_careful", "ventures_tight", "ventures_aggressive", "ventures_hoarder")
