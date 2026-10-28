"""The Spendthrift engine: an original auction game about tempo, for 2 or 3 players.

Spendthrift is an original design built on a familiar *genre* skeleton (ascending auctions paid
from a fixed purse, a random end, and a penalty for the poorest player; Reiner Knizia's auction
games are the inspiration for the genre). Its rules, name and components are its own: coins are a
single pool, lots are plain point values, the end cards are never auctioned, and the poorest player
loses their best lot rather than the game. It is not affiliated with any publisher.

**Rules.** Each player starts with ``coins`` coins; coin counts are public. The deck holds one lot
card of each value ``1..lots`` and ``ends`` end cards, shuffled. Each round the top card is flipped
(a chance node). An end card is set aside; the ``ends``-th one ends the game at once. A lot is
auctioned: starting with the opener (rotating each auction), players in seat order either raise the
high bid to any whole number of coins they can afford, or pass and are out of this auction. When
everyone but the high bidder has passed, the high bidder pays their bid and takes the lot; nobody
else pays. If everyone passes without a bid, the lot is discarded. The game also ends when the deck
runs out.

**Scoring.** The player or tied players with the fewest coins left discard their most valuable lot
(the *reserve penalty*). The highest lot total wins; ties go to more coins left, then are shared.
Payoffs are zero-sum and the winner gets +1: a player's share of the win, minus the average share,
scaled by ``n / (n - 1)`` (two players: +1 / -1; three: +1 / -0.5 / -0.5).

Everything is public except the deck's order, so the game is perfect-information with chance and
small sizes can be solved exactly (:mod:`iirl.eval.perfect_info`).

Action ids: ``0`` passes, ``k >= 1`` bids ``k`` coins (``coins + 1`` ids). Chance actions: ``0``
is an end card, ``v >= 1`` the lot worth ``v``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility
from iirl.games.spendthrift.state import Phase, SpendthriftState

PASS = 0
END_CARD = 0


def _set(items: tuple[int, ...], index: int, value: int) -> tuple[int, ...]:
    lst = list(items)
    lst[index] = value
    return tuple(lst)


class Spendthrift(Game):
    """Spendthrift for ``players`` players (2 or 3)."""

    def __init__(self, players: int = 2, lots: int = 5, ends: int = 2, coins: int = 8) -> None:
        """Set the size: lots worth ``1..lots``, ``ends`` end cards, ``coins`` per player."""
        assert 2 <= players <= 3 and lots >= 1 and ends >= 1 and coins >= 1
        self.players, self.lots, self.ends, self.coins = players, lots, ends, coins
        suffix = "" if players == 2 else str(players)
        self._type = GameType(
            short_name=f"spendthrift{suffix}",
            long_name=f"Spendthrift ({players} players, {lots} lots, {ends} ends, {coins} coins)",
            num_players=players,
            information=Information.PERFECT,
            utility=Utility.ZERO_SUM,
            provides_information_state_tensor=True,
        )

    @property
    def game_type(self) -> GameType:
        """Static metadata."""
        return self._type

    def new_initial_state(self) -> SpendthriftState:
        """Full purses, a full deck, the first card about to be flipped."""
        n = self.players
        return SpendthriftState(
            coins=(self.coins,) * n,
            totals=(0,) * n,
            best=(0,) * n,
            deck=(1 << self.lots) - 1,
            ends_left=self.ends,
        )

    def num_distinct_actions(self) -> int:
        """Pass plus one id per bid size."""
        return self.coins + 1

    def current_player(self, state: State) -> Player:
        """The bidder to act, or a chance/terminal sentinel."""
        assert isinstance(state, SpendthriftState)
        if state.phase == Phase.OVER:
            return PlayerId.TERMINAL
        if state.phase == Phase.REVEAL:
            return PlayerId.CHANCE
        return state.to_move

    def is_chance_node(self, state: State) -> bool:
        """Flipping the top card is a chance node."""
        assert isinstance(state, SpendthriftState)
        return state.phase == Phase.REVEAL

    def is_terminal(self, state: State) -> bool:
        """Whether the game is over."""
        assert isinstance(state, SpendthriftState)
        return state.phase == Phase.OVER

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Each remaining card equally likely to be on top."""
        assert isinstance(state, SpendthriftState)
        values = [v for v in range(1, self.lots + 1) if state.deck >> (v - 1) & 1]
        total = len(values) + state.ends_left
        outcomes = [(END_CARD, state.ends_left / total)] if state.ends_left else []
        return outcomes + [(v, 1 / total) for v in values]

    def opener(self, state: SpendthriftState) -> int:
        """Who opens the current auction: the opener rotates with each lot flipped."""
        drawn = self.lots - state.deck.bit_count()
        return (drawn - 1) % self.players

    # --- Moves ----------------------------------------------------------------------------------
    def legal_actions(self, state: State) -> list[Action]:
        """Pass, or any bid above the high bid that the bidder can afford."""
        assert isinstance(state, SpendthriftState)
        return [PASS, *range(state.bid + 1, state.coins[state.to_move] + 1)]

    def apply_action(self, state: State, action: Action) -> SpendthriftState:
        """Return the next state (never mutates ``state``)."""
        assert isinstance(state, SpendthriftState)
        if state.phase == Phase.REVEAL:
            return self._reveal(state, action)
        return self._bid(state, action)

    def _reveal(self, state: SpendthriftState, card: int) -> SpendthriftState:
        if card == END_CARD:
            seen = state.ends_seen + 1
            s = replace(state, ends_left=state.ends_left - 1, ends_seen=seen)
            return replace(s, phase=Phase.OVER) if seen == self.ends else s
        s = replace(
            state,
            deck=state.deck & ~(1 << (card - 1)),
            phase=Phase.AUCTION,
            lot=card,
            bid=0,
            leader=-1,
            active=(1 << self.players) - 1,
        )
        return replace(s, to_move=self.opener(s))

    def _bid(self, state: SpendthriftState, action: Action) -> SpendthriftState:
        p = state.to_move
        if action == PASS:
            s = replace(state, active=state.active & ~(1 << p))
        else:
            s = replace(state, bid=action, leader=p)
        if s.active == 0:  # everyone passed without a bid: the lot is discarded
            return self._next_round(s)
        if s.leader >= 0 and s.active == 1 << s.leader:
            w = s.leader
            s = replace(
                s,
                coins=_set(s.coins, w, s.coins[w] - s.bid),
                totals=_set(s.totals, w, s.totals[w] + s.lot),
                best=_set(s.best, w, max(s.best[w], s.lot)),
            )
            return self._next_round(s)
        nxt = next(
            q
            for q in ((p + i) % self.players for i in range(1, self.players + 1))
            if s.active >> q & 1
        )
        return replace(s, to_move=nxt)

    def _next_round(self, state: SpendthriftState) -> SpendthriftState:
        s = replace(state, phase=Phase.REVEAL, lot=0, bid=0, leader=-1, active=0, to_move=0)
        if s.deck == 0 and s.ends_left == 0:
            return replace(s, phase=Phase.OVER)
        return s

    # --- Scores ---------------------------------------------------------------------------------
    def scores(self, state: SpendthriftState) -> list[int]:
        """Lot totals after the reserve penalty."""
        poorest = min(state.coins)
        return [
            t - (b if c == poorest else 0)
            for t, b, c in zip(state.totals, state.best, state.coins, strict=True)
        ]

    def returns(self, state: State) -> list[float]:
        """Zero-sum: share of the win minus the average share, scaled so a win is +1."""
        assert isinstance(state, SpendthriftState)
        n = self.players
        key = list(zip(self.scores(state), state.coins, strict=True))
        top = max(key)
        winners = [i for i in range(n) if key[i] == top]
        scale = n / (n - 1)
        return [scale * ((1 / len(winners) if i in winners else 0.0) - 1 / n) for i in range(n)]

    # --- Observations ---------------------------------------------------------------------------
    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Everything, seat-relative: each player (own first), the lot and bid, the deck."""
        assert isinstance(state, SpendthriftState)
        n, poorest = self.players, min(state.coins)
        total = self.lots * (self.lots + 1) / 2
        feats: list[float] = []
        for i in range(n):
            q = (player + i) % n
            feats += [
                state.coins[q] / self.coins,
                state.totals[q] / total,
                state.best[q] / self.lots,
                float(state.active >> q & 1),
                float(state.leader == q),
                float(state.to_move == q and state.phase == Phase.AUCTION),
                float(state.coins[q] == poorest),
            ]
        lot = [0.0] * (self.lots + 1)
        lot[state.lot] = 1.0
        feats += lot + [state.bid / self.coins]
        feats += [float(state.deck >> v & 1) for v in range(self.lots)]
        feats += [state.ends_left / self.ends, state.ends_seen / self.ends]
        return np.asarray(feats, dtype=np.float32)

    def information_state_string(self, state: State, player: Player) -> str:
        """The whole state (nothing is hidden but the deck's order), from ``player``'s seat."""
        assert isinstance(state, SpendthriftState)
        return f"P{player}|{state}"

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> SpendthriftState:
        """Nothing is hidden: the state itself."""
        assert isinstance(state, SpendthriftState)
        return state

    # --- Display --------------------------------------------------------------------------------
    def action_to_string(self, state: State, action: Action) -> str:
        """Readable action label."""
        assert isinstance(state, SpendthriftState)
        if state.phase == Phase.REVEAL:
            return "end card" if action == END_CARD else f"lot {action}"
        return "pass" if action == PASS else f"bid {action}"

    def to_display(self, state: State) -> dict:
        """A JSON-ready snapshot of the whole state."""
        assert isinstance(state, SpendthriftState)
        return {
            "coins": list(state.coins),
            "totals": list(state.totals),
            "best": list(state.best),
            "deck": [v for v in range(1, self.lots + 1) if state.deck >> (v - 1) & 1],
            "ends_left": state.ends_left,
            "ends_seen": state.ends_seen,
            "lot": state.lot,
            "bid": state.bid,
            "leader": state.leader,
            "active": [bool(state.active >> q & 1) for q in range(self.players)],
            "current_player": int(self.current_player(state)),
            "terminal": self.is_terminal(state),
        }

    def player_view(self, state: State, player: Player) -> dict:
        """Nothing is hidden: the full display."""
        return self.to_display(state)


# The registered sizes solve exactly in under 20 s (about 0.7M states each); each extra lot
# multiplies the state count by about 10.
@register("spendthrift")
def _make_spendthrift() -> Spendthrift:
    """Two players: 5 lots, 2 end cards, 8 coins."""
    return Spendthrift(players=2, lots=5, ends=2, coins=8)


@register("spendthrift3")
def _make_spendthrift3() -> Spendthrift:
    """Three players: 4 lots, 2 end cards, 6 coins."""
    return Spendthrift(players=3, lots=4, ends=2, coins=6)


# Purse variants of the two-player game for the knobs study (2 end cards, 5 lots).
@register("spendthrift_tight")
def _make_spendthrift_tight() -> Spendthrift:
    """Two players: 5 lots, 2 end cards, 6 coins (coins are precious late)."""
    return Spendthrift(players=2, lots=5, ends=2, coins=6)


@register("spendthrift_loose")
def _make_spendthrift_loose() -> Spendthrift:
    """Two players: 5 lots, 2 end cards, 10 coins (no late premium on a coin)."""
    return Spendthrift(players=2, lots=5, ends=2, coins=10)
