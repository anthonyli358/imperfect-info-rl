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
"""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action
from iirl.games.ventures.cards import NUM_SUITS, START_COST, is_backing, rank_of, suit_of
from iirl.games.ventures.game import (
    DISCARD_OFFSET,
    DRAW_DECK_ACTION,
    TAKE_DISCARD_OFFSET,
    route_max_number,
    route_number_sum,
)
from iirl.games.ventures.state import Phase, VenturesState

_BACKING_MARGIN = 10  # only back a route projected comfortably past the start cost


class VenturesHeuristic(Agent):
    """Greedy Ventures policy with wager discipline and deck-clock awareness."""

    def __init__(self, name: str = "heuristic") -> None:
        """Create the heuristic agent.

        Args:
            name: Label used in standings and error messages.
        """
        self.name = name

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
            # Unstarted: how many of our best cards clear the start cost, and is there time?
            cumulative, needed = on_route, None
            for i, v in enumerate(suit_numbers, start=1):
                cumulative += v
                if cumulative > START_COST:
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
                    pursued[s]
                    and route_max[s] == 0
                    and potential[s] >= START_COST + _BACKING_MARGIN
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
        return min(
            discards,
            key=lambda a: (pursued[suit_of(a - DISCARD_OFFSET)], rank_of(a - DISCARD_OFFSET)),
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
        if DRAW_DECK_ACTION in legal:
            return DRAW_DECK_ACTION
        return int(rng.choice(list(legal)))  # forced: only discards left to draw
