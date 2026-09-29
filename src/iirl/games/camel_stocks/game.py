"""The Camel Stocks engine: an original trading card game on a classic mechanical skeleton.

See :mod:`iirl.games.camel_stocks.cards` for the card model and the IP/credit note.

A turn is one of:

* **Take one good** from the market (hand limit 7); the market refills from the deck.
* **Take all the camels** from the market into your herd; the market refills.
* **Sell** some cards of one good (precious goods: at least 2) for that good's top value tokens,
  plus a random bonus token for a sale of 3, 4, or 5+ cards.
* **Exchange**: take at least 2 goods from the market and give back as many cards from your hand
  and/or camels from your herd, never giving back a good you are taking. As a single flat action
  this is combinatorially large, so it is split into steps (``Phase.EX_TAKE`` then
  ``Phase.EX_GIVE``): take goods one at a time, stop, then give cards one at a time. Legality
  guarantees every exchange started can be completed within the hand limit.

The game ends when three goods token piles are empty or the market cannot be refilled. Each player
scores their tokens, plus 5 for the larger herd; the payoff is the score difference.
"""

from __future__ import annotations

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility
from iirl.games.camel_stocks.cards import (
    BONUS_TOKENS,
    CAMEL,
    CAMEL_BONUS,
    CAMEL_COLOR,
    CAMEL_SYMBOL,
    CARD_COUNTS,
    CARD_NAMES,
    DEAL_SIZE,
    DECK_SIZE,
    EMPTY_PILES_TO_END,
    GOOD_COLORS,
    GOOD_NAMES,
    GOOD_SYMBOLS,
    GOOD_TOKENS,
    HAND_LIMIT,
    MARKET_SIZE,
    MARKET_START_CAMELS,
    MAX_TURNS,
    MIN_PRECIOUS_SALE,
    NUM_CARD_TYPES,
    NUM_GOODS,
    PRECIOUS,
)
from iirl.games.camel_stocks.state import CamelStocksState, Counts, Phase, initial_state

# --- Flat action-id layout ----------------------------------------------------------------------
# Take good g: 0..5 | Take camels: 6 | Sell (g, k): 7 + 7g + (k-1), k = 1..7 -> 7..48
# Exchange: take good g: 49..54 | stop taking: 55 | give card type c (goods, camel): 56..62
TAKE_OFFSET = 0
TAKE_CAMELS = NUM_GOODS
SELL_OFFSET = TAKE_CAMELS + 1
MAX_SELL = HAND_LIMIT
EX_TAKE_OFFSET = SELL_OFFSET + NUM_GOODS * MAX_SELL
EX_DONE = EX_TAKE_OFFSET + NUM_GOODS
EX_GIVE_OFFSET = EX_DONE + 1
NUM_ACTIONS = EX_GIVE_OFFSET + NUM_CARD_TYPES
# Chance actions: a card type (deal / market refill) or a bonus token's value.

_MARKET_DEAL = MARKET_SIZE - MARKET_START_CAMELS
_TOTAL_DEAL = _MARKET_DEAL + 2 * DEAL_SIZE
_EXPECTED_BONUS = tuple(sum(b) / len(b) for b in BONUS_TOKENS)


def sell_action(good: int, count: int) -> Action:
    """Return the action id for selling ``count`` cards of ``good``."""
    return SELL_OFFSET + good * MAX_SELL + (count - 1)


def sale_value(tokens: tuple[int, ...], count: int) -> int:
    """Return the token value of selling ``count`` cards into a pile (top tokens first)."""
    return sum(tokens[:count])


def _bump(counts: Counts, idx: int, delta: int) -> Counts:
    lst = list(counts)
    lst[idx] += delta
    return tuple(lst)


def _set(pair: tuple, player: int, value) -> tuple:  # noqa: ANN001 - per-player pair update
    return (value, pair[1]) if player == 0 else (pair[0], value)


def _forget(known: Counts, good: int, count: int) -> Counts:
    """Drop up to ``count`` known copies of ``good`` (cards seen leaving the hand)."""
    return _bump(known, good, -min(known[good], count)) if known[good] else known


def _replace(state: CamelStocksState, **changes: object) -> CamelStocksState:
    from dataclasses import replace

    return replace(state, **changes)  # type: ignore[arg-type]


def _exchange_feasible(hand: Counts, herd: int, taken: Counts) -> bool:
    """Whether an exchange taking ``taken`` can still be completed.

    It needs enough cards of *other* goods plus camels to give back one per good taken, and enough
    goods among them to stay within the hand limit.
    """
    k = sum(taken)
    givable = sum(n for g, n in enumerate(hand) if not taken[g])
    need_goods = sum(hand) + k - HAND_LIMIT  # goods that must leave the hand
    return givable + herd >= k and givable >= need_goods


class CamelStocks(Game):
    """Two-player, zero-sum Camel Stocks (imperfect information: hands and deck are hidden)."""

    _TYPE = GameType(
        short_name="camel_stocks",
        long_name="Camel Stocks",
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

    def new_initial_state(self) -> CamelStocksState:
        """Return the pre-deal chance state."""
        return initial_state()

    def num_distinct_actions(self) -> int:
        """Return the flat action-space size (see the action-id layout above)."""
        return NUM_ACTIONS

    # --- Node classification --------------------------------------------------------------------
    def current_player(self, state: State) -> Player:
        """Return the player to act, or a chance/terminal sentinel."""
        assert isinstance(state, CamelStocksState)
        if state.terminal:
            return PlayerId.TERMINAL
        if state.phase in (Phase.DEAL, Phase.REFILL, Phase.BONUS):
            return PlayerId.CHANCE
        return state.to_move

    def is_chance_node(self, state: State) -> bool:
        """Return whether the state is a deal, market-refill or bonus-token chance node."""
        assert isinstance(state, CamelStocksState)
        return not state.terminal and state.phase in (Phase.DEAL, Phase.REFILL, Phase.BONUS)

    def is_terminal(self, state: State) -> bool:
        """Return whether the game is over."""
        assert isinstance(state, CamelStocksState)
        return state.terminal

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Return a deck draw (each card type ∝ its count) or a bonus token (∝ multiplicity)."""
        assert isinstance(state, CamelStocksState)
        if state.phase == Phase.BONUS:
            pile = state.bonus[state.bonus_tier]
            return [(v, pile.count(v) / len(pile)) for v in sorted(set(pile))]
        total = sum(state.deck)
        return [(c, n / total) for c, n in enumerate(state.deck) if n > 0]

    # --- Legal actions --------------------------------------------------------------------------
    def legal_actions(self, state: State) -> list[Action]:
        """Return the legal actions at a player node."""
        assert isinstance(state, CamelStocksState)
        p = state.to_move
        hand, herd, market = state.hands[p], state.herds[p], state.market
        if state.phase == Phase.PLAY:
            actions: list[Action] = []
            room = sum(hand) < HAND_LIMIT
            for g in range(NUM_GOODS):
                if market[g] and room:
                    actions.append(TAKE_OFFSET + g)
            if market[CAMEL]:
                actions.append(TAKE_CAMELS)
            for g in range(NUM_GOODS):
                low = MIN_PRECIOUS_SALE if g in PRECIOUS else 1
                actions.extend(sell_action(g, k) for k in range(low, hand[g] + 1))
            for g in range(NUM_GOODS):
                if market[g] and self._can_start_exchange(hand, herd, market, g):
                    actions.append(EX_TAKE_OFFSET + g)
            return actions
        if state.phase == Phase.EX_TAKE:
            taken = state.taken
            actions = [
                EX_TAKE_OFFSET + g
                for g in range(NUM_GOODS)
                if market[g] > taken[g] and _exchange_feasible(hand, herd, _bump(taken, g, 1))
            ]
            if sum(taken) >= 2:
                actions.append(EX_DONE)
            return actions
        # Phase.EX_GIVE
        return [EX_GIVE_OFFSET + c for c in range(NUM_CARD_TYPES) if self._can_give(state, c)]

    @staticmethod
    def _can_start_exchange(hand: Counts, herd: int, market: Counts, first: int) -> bool:
        """Whether taking ``first`` can open an exchange that reaches the 2-good minimum."""
        one = _bump((0,) * NUM_GOODS, first, 1)
        return any(
            market[g] > one[g] and _exchange_feasible(hand, herd, _bump(one, g, 1))
            for g in range(NUM_GOODS)
        )

    @staticmethod
    def _can_give(state: CamelStocksState, card: int) -> bool:
        """Whether giving ``card`` keeps the exchange completable within the hand limit."""
        p = state.to_move
        hand, herd, taken, given = state.hands[p], state.herds[p], state.taken, state.given
        if card == CAMEL:
            if herd - given[CAMEL] < 1:
                return False
        elif taken[card] or hand[card] - given[card] < 1:
            return False
        k = sum(taken)
        remaining = k - sum(given) - 1
        goods_given = sum(given[:NUM_GOODS]) + (card != CAMEL)
        goods_left = sum(hand[g] - given[g] for g in range(NUM_GOODS) if not taken[g]) - (
            card != CAMEL
        )
        camels_left = herd - given[CAMEL] - (card == CAMEL)
        need_goods = sum(hand) + k - HAND_LIMIT
        return (
            goods_left + camels_left >= remaining
            and goods_given + min(remaining, goods_left) >= need_goods
        )

    # --- Transitions ----------------------------------------------------------------------------
    def apply_action(self, state: State, action: Action) -> CamelStocksState:
        """Return the state reached by applying ``action`` (does not mutate ``state``)."""
        assert isinstance(state, CamelStocksState)
        if state.phase == Phase.DEAL:
            return self._apply_deal(state, action)
        if state.phase == Phase.REFILL:
            return self._apply_refill(state, action)
        if state.phase == Phase.BONUS:
            return self._apply_bonus(state, action)
        if state.phase == Phase.EX_GIVE:
            return self._apply_give(state, action - EX_GIVE_OFFSET)
        if EX_TAKE_OFFSET <= action < EX_DONE:
            return _replace(
                state, taken=_bump(state.taken, action - EX_TAKE_OFFSET, 1), phase=Phase.EX_TAKE
            )
        if action == EX_DONE:
            return _replace(state, phase=Phase.EX_GIVE)
        p = state.to_move
        if action == TAKE_CAMELS:
            n = state.market[CAMEL]
            s = _replace(
                state,
                market=_bump(state.market, CAMEL, -n),
                herds=_set(state.herds, p, state.herds[p] + n),
            )
            return self._start_refill(s, n)
        if action < TAKE_CAMELS:
            g = action - TAKE_OFFSET
            s = _replace(
                state,
                market=_bump(state.market, g, -1),
                hands=_set(state.hands, p, _bump(state.hands[p], g, 1)),
                known=_set(state.known, p, _bump(state.known[p], g, 1)),
            )
            return self._start_refill(s, 1)
        g, k = divmod(action - SELL_OFFSET, MAX_SELL)
        return self._apply_sell(state, g, k + 1)

    def _apply_deal(self, state: CamelStocksState, card: int) -> CamelStocksState:
        i = _TOTAL_DEAL - state.deal_left  # index of this card in the deal
        deck = _bump(state.deck, card, -1)
        if i < _MARKET_DEAL:
            s = _replace(state, deck=deck, market=_bump(state.market, card, 1))
        else:
            p = 0 if i < _MARKET_DEAL + DEAL_SIZE else 1
            if card == CAMEL:
                s = _replace(state, deck=deck, herds=_set(state.herds, p, state.herds[p] + 1))
            else:
                s = _replace(
                    state, deck=deck, hands=_set(state.hands, p, _bump(state.hands[p], card, 1))
                )
        left = state.deal_left - 1
        if left:
            return _replace(s, deal_left=left)
        return _replace(s, deal_left=0, phase=Phase.PLAY, to_move=0)

    def _start_refill(self, state: CamelStocksState, n: int) -> CamelStocksState:
        if n == 0:
            return self._end_turn(state)
        if sum(state.deck) == 0:
            return _replace(state, terminal=True)  # the market cannot be refilled
        return _replace(state, phase=Phase.REFILL, refill=n)

    def _apply_refill(self, state: CamelStocksState, card: int) -> CamelStocksState:
        s = _replace(
            state,
            deck=_bump(state.deck, card, -1),
            market=_bump(state.market, card, 1),
            refill=state.refill - 1,
        )
        if s.refill == 0:
            return self._end_turn(s)
        if sum(s.deck) == 0:
            return _replace(s, terminal=True)
        return s

    def _apply_sell(self, state: CamelStocksState, good: int, count: int) -> CamelStocksState:
        p = state.to_move
        pile = state.tokens[good]
        tokens = list(state.tokens)
        tokens[good] = pile[count:]
        s = _replace(
            state,
            hands=_set(state.hands, p, _bump(state.hands[p], good, -count)),
            known=_set(state.known, p, _forget(state.known[p], good, count)),
            sold=_bump(state.sold, good, count),
            tokens=tuple(tokens),
            points=_set(state.points, p, state.points[p] + sale_value(pile, count)),
        )
        tier = min(count, 5) - 3
        if tier >= 0 and state.bonus[tier]:
            return _replace(s, phase=Phase.BONUS, bonus_tier=tier)
        return self._end_turn(s)

    def _apply_bonus(self, state: CamelStocksState, value: int) -> CamelStocksState:
        p, tier = state.to_move, state.bonus_tier
        pile = list(state.bonus[tier])
        pile.remove(value)
        bonus = list(state.bonus)
        bonus[tier] = tuple(pile)
        s = _replace(
            state,
            bonus=tuple(bonus),
            bonus_tier=-1,
            points=_set(state.points, p, state.points[p] + value),
        )
        return self._end_turn(s)

    def _apply_give(self, state: CamelStocksState, card: int) -> CamelStocksState:
        given = _bump(state.given, card, 1)
        if sum(given) < sum(state.taken):
            return _replace(state, given=given)
        # Complete the exchange: taken goods into the hand, given cards into the market.
        p, taken = state.to_move, state.taken
        hand, known = list(state.hands[p]), state.known[p]
        market = list(state.market)
        for g in range(NUM_GOODS):
            hand[g] += taken[g] - given[g]
            market[g] += given[g] - taken[g]
            known = _forget(known, g, given[g])
            known = _bump(known, g, taken[g])
        market[CAMEL] += given[CAMEL]
        s = _replace(
            state,
            hands=_set(state.hands, p, tuple(hand)),
            herds=_set(state.herds, p, state.herds[p] - given[CAMEL]),
            market=tuple(market),
            known=_set(state.known, p, known),
            taken=(0,) * NUM_GOODS,
            given=(0,) * NUM_CARD_TYPES,
        )
        return self._end_turn(s)

    def _end_turn(self, state: CamelStocksState) -> CamelStocksState:
        empty = sum(1 for pile in state.tokens if not pile)
        turn = state.turn + 1
        if empty >= EMPTY_PILES_TO_END or turn >= MAX_TURNS:
            return _replace(state, turn=turn, terminal=True, phase=Phase.PLAY)
        return _replace(state, turn=turn, to_move=1 - state.to_move, phase=Phase.PLAY)

    # --- Scoring --------------------------------------------------------------------------------
    def score(self, state: State, player: int) -> int:
        """Return ``player``'s score: token points, plus the camel bonus for the larger herd."""
        assert isinstance(state, CamelStocksState)
        herd, other = state.herds[player], state.herds[1 - player]
        return state.points[player] + (CAMEL_BONUS if herd > other else 0)

    def returns(self, state: State) -> list[float]:
        """Return zero-sum payoffs: each player's score minus the opponent's."""
        diff = self.score(state, 0) - self.score(state, 1)
        return [float(diff), float(-diff)]

    def _potential(self, state: CamelStocksState, player: int) -> float:
        """Estimate ``player``'s final score from banked points and the goods in hand.

        Held goods count at a discount, valued as if sold now at the current top tokens (with the
        expected set bonus); the camel bonus counts in full for the larger herd, half on a tie.
        """
        hand = state.hands[player]
        held = 0.0
        for g in range(NUM_GOODS):
            n = hand[g]
            if n == 0 or (g in PRECIOUS and n < MIN_PRECIOUS_SALE):
                continue
            held += sale_value(state.tokens[g], n)
            tier = min(n, 5) - 3
            if tier >= 0 and state.bonus[tier]:
                held += _EXPECTED_BONUS[tier]
        herd, other = state.herds[player], state.herds[1 - player]
        camels = CAMEL_BONUS if herd > other else (CAMEL_BONUS / 2 if herd == other else 0.0)
        return state.points[player] + 0.6 * held + camels

    def heuristic_value(self, state: State, player: Player) -> float:
        """Return a zero-sum leaf estimate for ``player`` (see :meth:`_potential`)."""
        assert isinstance(state, CamelStocksState)
        diff = self._potential(state, 0) - self._potential(state, 1)
        return float(diff) if player == 0 else float(-diff)

    # --- Encoders / determinization -------------------------------------------------------------
    def _unseen(self, state: CamelStocksState, player: int) -> list[int]:
        """Card counts per type ``player`` cannot see: the deck plus the opponent's unknown hand."""
        opp = 1 - player
        unseen = []
        for c in range(NUM_CARD_TYPES):
            n = CARD_COUNTS[c] - state.market[c]
            if c == CAMEL:
                n -= state.herds[0] + state.herds[1]
            else:
                n -= state.hands[player][c] + state.sold[c] + state.known[opp][c]
            unseen.append(n)
        return unseen

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Encode ``player``'s view: per-good features aligned with the action ids, plus scalars.

        Per good: own count, market count, the opponent's known count, unseen count, tokens left,
        the top token, the value of selling one's whole holding now, whether that sale is legal,
        and the goods taken/given in an exchange in progress. The opponent's hand beyond its
        known cards, and the deck order, are never encoded. Shape is constant.
        """
        assert isinstance(state, CamelStocksState)
        opp = 1 - player
        hand, unseen = state.hands[player], self._unseen(state, player)
        per_good = []
        for g in range(NUM_GOODS):
            pile, n = state.tokens[g], hand[g]
            sellable = n >= (MIN_PRECIOUS_SALE if g in PRECIOUS else 1)
            per_good.append(
                [
                    n,
                    state.market[g],
                    state.known[opp][g],
                    unseen[g],
                    len(pile),
                    pile[0] if pile else 0,
                    sale_value(pile, n),
                    float(sellable),
                    state.taken[g],
                    state.given[g],
                ]
            )
        scalars = [
            sum(hand),
            sum(state.hands[opp]),
            state.herds[player],
            state.herds[opp],
            state.market[CAMEL],
            unseen[CAMEL],
            state.given[CAMEL],
            sum(state.deck) / DECK_SIZE,
            state.points[player],
            state.points[opp],
            sum(1 for pile in state.tokens if not pile),
            *(len(b) for b in state.bonus),
            self._potential(state, player),
            float(state.phase == Phase.PLAY),
            float(state.phase == Phase.EX_TAKE),
            float(state.phase == Phase.EX_GIVE),
            state.turn / MAX_TURNS,
        ]
        return np.concatenate(
            [np.asarray(per_good, dtype=np.float32).ravel(), np.asarray(scalars, dtype=np.float32)]
        )

    def information_state_string(self, state: State, player: Player) -> str:
        """Return a hashable infoset key from exactly what ``player`` observes."""
        assert isinstance(state, CamelStocksState)
        opp = 1 - player
        return (
            f"P{player}|h{state.hands[player]}|oh{sum(state.hands[opp])}|kn{state.known}"
            f"|hd{state.herds}|m{state.market}|k{sum(state.deck)}|tk{state.tokens}|b{state.bonus}"
            f"|pt{state.points}|s{state.sold}|ph{int(state.phase)}|tm{state.to_move}"
            f"|x{state.taken}{state.given}|rf{state.refill}|bt{state.bonus_tier}|t{state.turn}"
        )

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> CamelStocksState:
        """Return a world consistent with ``player``'s view, re-dealing the hidden cards.

        The opponent's unknown hand cards are dealt uniformly from the unseen *goods* (camels
        never stay in a hand), and everything else unseen forms the deck.
        """
        return self._redeal(state, player, rng)

    def repair_world(
        self, state: State, player: Player, world: State, rng: np.random.Generator
    ) -> CamelStocksState:
        """Re-deal like :meth:`resample_from_infostate`, but start from ``world``'s opponent hand.

        The opponent goods ``world`` holds that are still unseen are kept (a random subset if there
        are too many); only the shortfall is dealt at random.
        """
        assert isinstance(world, CamelStocksState)
        return self._redeal(state, player, rng, prefer=world.hands[1 - player])

    def _redeal(
        self,
        state: State,
        player: Player,
        rng: np.random.Generator,
        prefer: Counts | None = None,
    ) -> CamelStocksState:
        assert isinstance(state, CamelStocksState)
        opp = 1 - player
        known = state.known[opp]
        unseen = self._unseen(state, player)
        hidden = sum(state.hands[opp]) - sum(known)
        goods = [g for g in range(NUM_GOODS) for _ in range(unseen[g])]
        kept: list[int] = []
        if prefer is not None:
            for g in range(NUM_GOODS):
                kept.extend([g] * min(max(prefer[g] - known[g], 0), unseen[g]))
            rng.shuffle(kept)
            kept = kept[:hidden]
            for g in kept:
                goods.remove(g)
        rng.shuffle(goods)
        dealt = kept + goods[: hidden - len(kept)]
        opp_hand = list(known)
        deck = list(unseen)
        for g in dealt:
            opp_hand[g] += 1
            deck[g] -= 1
        return _replace(state, deck=tuple(deck), hands=_set(state.hands, opp, tuple(opp_hand)))

    # --- Display --------------------------------------------------------------------------------
    def action_to_string(self, state: State, action: Action) -> str:
        """Return a readable label for an action."""
        assert isinstance(state, CamelStocksState)
        if state.phase in (Phase.DEAL, Phase.REFILL):
            return f"{'deal' if state.phase == Phase.DEAL else 'market gets'} {CARD_NAMES[action]}"
        if state.phase == Phase.BONUS:
            return f"bonus token {action}"
        if action < TAKE_CAMELS:
            return f"take {GOOD_NAMES[action]}"
        if action == TAKE_CAMELS:
            return f"take {state.market[CAMEL]} camels"
        if action < EX_TAKE_OFFSET:
            g, k = divmod(action - SELL_OFFSET, MAX_SELL)
            return f"sell {k + 1} {GOOD_NAMES[g]}"
        if action < EX_DONE:
            return f"exchange: take {GOOD_NAMES[action - EX_TAKE_OFFSET]}"
        if action == EX_DONE:
            return "exchange: stop taking"
        return f"exchange: give {CARD_NAMES[action - EX_GIVE_OFFSET]}"

    def display_meta(self) -> dict:
        """Return viewer metadata: card names and colors."""
        return {
            "card_names": list(CARD_NAMES),
            "card_colors": [*GOOD_COLORS, CAMEL_COLOR],
            "card_symbols": [*GOOD_SYMBOLS, CAMEL_SYMBOL],
            "precious": sorted(PRECIOUS),
            "hand_limit": HAND_LIMIT,
            "tokens": [list(t) for t in GOOD_TOKENS],
        }

    def to_display(self, state: State) -> dict:
        """Return a render-ready snapshot of the full state (both hands, for replays)."""
        assert isinstance(state, CamelStocksState)

        def cards(counts: Counts) -> list[int]:
            return [c for c, n in enumerate(counts) for _ in range(n)]

        return {
            "phase": state.phase.name,
            "to_move": state.to_move,
            "terminal": state.terminal,
            "deck_size": sum(state.deck),
            "market": cards(state.market),
            "hands": [cards(state.hands[0]), cards(state.hands[1])],
            "herds": list(state.herds),
            "tokens": [list(t) for t in state.tokens],
            "bonus_left": [len(b) for b in state.bonus],
            "points": list(state.points),
            "scores": [self.score(state, 0), self.score(state, 1)],
            "taken": cards(state.taken),
            "given": cards(state.given),
        }

    def player_view(self, state: State, player: Player) -> dict:
        """Return :meth:`to_display` minus what ``player`` cannot see.

        The opponent's hand becomes just its publicly known goods (seen taken from the market)
        plus ``hidden[opp]``, the number of unknown cards. Everything else is public.
        """
        assert isinstance(state, CamelStocksState)
        view = self.to_display(state)
        opp = 1 - player
        known = [g for g, n in enumerate(state.known[opp]) for _ in range(n)]
        view["hands"][opp] = known
        view["hidden"] = [0, 0]
        view["hidden"][opp] = sum(state.hands[opp]) - len(known)
        view["player"] = player
        return view


@register("camel_stocks")
def _make_camel_stocks() -> CamelStocks:
    """Factory registered under ``"camel_stocks"``."""
    return CamelStocks()
