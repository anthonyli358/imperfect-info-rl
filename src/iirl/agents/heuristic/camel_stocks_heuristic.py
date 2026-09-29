"""A rule-based Camel Stocks baseline.

Game-specific by design: it gives the learners a non-trivial opponent, anchors Elo, and
sanity-checks the engine. It plays the game's well-known strategy:

1. **Sell before the opponent, and sell big.** Early tokens are worth the most and sets of 3+ earn
   a bonus, so a set is sold once it is big enough, once the opponent is visibly collecting the same
   good, when the hand is full, or when the game is about to end.
2. **Precious goods first.** Gems, gold and silver carry most of the value; take them when offered.
3. **Camels are currency.** Swap camels for several good cards at once, and take a large group of
   camels only when the market offers nothing better.
"""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action, Player
from iirl.games.camel_stocks.cards import (
    CAMEL,
    EMPTY_PILES_TO_END,
    HAND_LIMIT,
    MIN_PRECIOUS_SALE,
    NUM_GOODS,
    PRECIOUS,
)
from iirl.games.camel_stocks.game import (
    EX_DONE,
    EX_GIVE_OFFSET,
    EX_TAKE_OFFSET,
    TAKE_CAMELS,
    TAKE_OFFSET,
    sale_value,
    sell_action,
)
from iirl.games.camel_stocks.state import CamelStocksState, Phase

_EXPECTED_BONUS = (2.0, 5.0, 9.0)  # mean bonus for 3, 4, 5+ cards
_BIG_SET = 4  # sell a non-precious set of this size outright


class CamelStocksHeuristic(Agent):
    """Greedy Camel Stocks policy: sell early and big, favour precious goods, trade camels."""

    def __init__(self, name: str = "heuristic") -> None:
        """Create the heuristic agent.

        Args:
            name: Label used in standings and error messages.
        """
        self.name = name
        self._plan: list[int] = []

    def reset(self, game: Game, player: Player) -> None:
        """Forget any exchange plan from a previous game."""
        self._plan = []

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return a legal action following the rules in the module docstring."""
        assert isinstance(state, CamelStocksState)
        legal = game.legal_actions(state)
        legal_set = set(legal)
        if state.phase == Phase.EX_TAKE:
            while self._plan:
                a = EX_TAKE_OFFSET + self._plan.pop(0)
                if a in legal_set:
                    return a
            return EX_DONE if EX_DONE in legal_set else legal[0]
        if state.phase == Phase.EX_GIVE:
            return self._give(state, legal_set) or legal[0]
        self._plan = []
        action = self._play(state, legal_set)
        return action if action in legal_set else legal[0]

    # --- The turn's main decision ---------------------------------------------------------------
    def _play(self, state: CamelStocksState, legal: set[int]) -> Action:
        me, opp = state.to_move, 1 - state.to_move
        hand = state.hands[me]
        ending = (
            sum(state.deck) <= 4 or sum(1 for p in state.tokens if not p) >= EMPTY_PILES_TO_END - 1
        )

        # 1. Sell.
        best_sale, best_sale_value = None, 0.0
        for g in range(NUM_GOODS):
            n = hand[g]
            if n < (MIN_PRECIOUS_SALE if g in PRECIOUS else 1) or not state.tokens[g]:
                continue
            value = sale_value(state.tokens[g], n)
            tier = min(n, 5) - 3
            if tier >= 0 and state.bonus[tier]:
                value += _EXPECTED_BONUS[tier]
            urgent = (
                ending
                or sum(hand) >= HAND_LIMIT
                or state.known[opp][g] >= 2  # they are collecting it too: sell first
                or len(state.tokens[g]) <= n  # this sale takes the last good tokens
                or (g in PRECIOUS and n >= 3)
                or n >= _BIG_SET
            )
            if urgent and value > best_sale_value:
                best_sale, best_sale_value = sell_action(g, n), value
        if best_sale is not None:
            return best_sale

        # 2. Exchange camels for several valuable goods.
        wanted = sorted(
            (g for g in range(NUM_GOODS) for _ in range(state.market[g])),
            key=lambda g: -self._take_value(state, g),
        )
        wanted = [g for g in wanted if self._take_value(state, g) >= 5]
        room = HAND_LIMIT - sum(hand)
        k = min(len(wanted), state.herds[me], room)
        if k >= 2 and EX_TAKE_OFFSET + wanted[0] in legal:
            self._plan = wanted[1:k]
            return EX_TAKE_OFFSET + wanted[0]

        # 3. Take the best single good, unless camels are the better grab.
        takes = [g for g in range(NUM_GOODS) if TAKE_OFFSET + g in legal]
        best_take = max(takes, key=lambda g: self._take_value(state, g), default=None)
        best_take_value = self._take_value(state, best_take) if best_take is not None else 0.0
        camels = state.market[CAMEL]
        if TAKE_CAMELS in legal and (best_take is None or (camels >= 3 and best_take_value < 4)):
            return TAKE_CAMELS
        if best_take is not None:
            return TAKE_OFFSET + best_take

        # 4. Otherwise sell the most valuable legal set, or take camels.
        sales = [
            (sale_value(state.tokens[g], hand[g]), sell_action(g, hand[g]))
            for g in range(NUM_GOODS)
            if sell_action(g, hand[g]) in legal and hand[g]
        ]
        if sales:
            return max(sales)[1]
        return TAKE_CAMELS

    @staticmethod
    def _take_value(state: CamelStocksState, good: int) -> float:
        """How much taking one ``good`` is worth: its top token, more if it builds a set."""
        pile = state.tokens[good]
        if not pile:
            return 0.0
        held = state.hands[state.to_move][good]
        return pile[0] + (2.0 if held >= 1 else 0.0) + (1.0 if good in PRECIOUS else 0.0)

    @staticmethod
    def _give(state: CamelStocksState, legal: set[int]) -> Action | None:
        """Give back camels first, then the least valuable goods."""
        if EX_GIVE_OFFSET + CAMEL in legal:
            return EX_GIVE_OFFSET + CAMEL
        goods = [g for g in range(NUM_GOODS) if EX_GIVE_OFFSET + g in legal]
        if not goods:
            return None
        return EX_GIVE_OFFSET + min(
            goods, key=lambda g: (state.tokens[g][0] if state.tokens[g] else 0, -g)
        )
