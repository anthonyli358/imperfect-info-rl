"""Heads-up limit Texas Hold'em.

Texas Hold'em is a traditional public-domain card game; this is the standard heads-up limit form
used in computer-poker research (the Annual Computer Poker Competition's rules), written against
our ``Game`` interface because the Windows build of OpenSpiel has no ``universal_poker``.

**Rules.** Player 0 is the button: they post the small blind (1 chip), player 1 the big blind
(2). Each gets two private hole cards. Four betting rounds: preflop, the flop (three board
cards), the turn (one), the river (one). Bets are fixed: 2 chips preflop and on the flop, 4 on the
turn and river; at most four bets per round, the big blind counting as the first preflop. The
button acts first preflop and second after it. A player may fold (only when facing a bet),
check or call, or bet or raise. A round ends once both have acted and the bets are level. If no
one folds, the best five-card hand from each player's hole cards and the board wins (ties split).

Payoffs are in big blinds (one big blind = 2 chips), zero-sum.

Player action ids: ``0`` fold, ``1`` check or call, ``2`` bet or raise. Chance action ids are the
card dealt, ``0..51`` (see :mod:`~iirl.games.holdem.cards`); the deal order is the button's two
hole cards, then the big blind's, then the board.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility
from iirl.games.holdem.cards import card_str, category, hand_value

FOLD, CALL, RAISE = 0, 1, 2
BET_SIZE = (2, 2, 4, 4)  # chips per bet, by round
MAX_BETS = 4  # per round, the big blind included preflop
CARDS_BY_ROUND = (4, 7, 8, 9)  # cards dealt (hole + board) before each round's betting
CHIPS_PER_BB = 2
_ACTION_CHARS = "fcr"


@dataclass(frozen=True)
class HoldemState(State):
    """A heads-up limit Hold'em position.

    Attributes:
        cards: Cards dealt so far, in deal order: the button's two, the big blind's two, the board.
        round: Betting round, 0 (preflop) to 3 (river).
        history: The betting so far, one string per round started (``"f"``/``"c"``/``"r"``).
        contrib: Chips each player has put in.
        bets: Bets made this round (the big blind counts as one preflop).
        to_move: The player to act at player nodes.
        folded: Who folded, or -1.
        showdown: Whether the river betting has closed (the game is over).
    """

    cards: tuple[int, ...] = ()
    round: int = 0
    history: tuple[str, ...] = ("",)
    contrib: tuple[int, int] = (1, 2)
    bets: int = 1
    to_move: int = 0
    folded: int = -1
    showdown: bool = False


class LimitHoldem(Game):
    """Heads-up limit Hold'em (two players)."""

    _type = GameType(
        short_name="limit_holdem",
        long_name="Heads-up limit Texas Hold'em",
        num_players=2,
        information=Information.IMPERFECT,
        utility=Utility.ZERO_SUM,
        provides_information_state_tensor=True,
    )

    @property
    def game_type(self) -> GameType:
        """Static metadata."""
        return self._type

    def new_initial_state(self) -> HoldemState:
        """Blinds posted, no cards dealt yet."""
        return HoldemState()

    def num_distinct_actions(self) -> int:
        """Fold, check/call, bet/raise."""
        return 3

    # --- Structure ------------------------------------------------------------------------------
    def is_terminal(self, state: State) -> bool:
        """A fold, or the river betting closed."""
        assert isinstance(state, HoldemState)
        return state.folded >= 0 or state.showdown

    def is_chance_node(self, state: State) -> bool:
        """Cards still to deal before this round's betting."""
        assert isinstance(state, HoldemState)
        return not self.is_terminal(state) and len(state.cards) < CARDS_BY_ROUND[state.round]

    def current_player(self, state: State) -> Player:
        """The player to act, or a chance/terminal sentinel."""
        assert isinstance(state, HoldemState)
        if self.is_terminal(state):
            return PlayerId.TERMINAL
        if self.is_chance_node(state):
            return PlayerId.CHANCE
        return state.to_move

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Every undealt card, equally likely."""
        assert isinstance(state, HoldemState)
        left = [c for c in range(52) if c not in state.cards]
        return [(c, 1 / len(left)) for c in left]

    def owe(self, state: HoldemState) -> int:
        """Chips the player to act must add to call."""
        return max(state.contrib) - state.contrib[state.to_move]

    def legal_actions(self, state: State) -> list[Action]:
        """Fold only when facing a bet; raise while under the cap."""
        assert isinstance(state, HoldemState)
        legal = [FOLD] if self.owe(state) > 0 else []
        legal.append(CALL)
        if state.bets < MAX_BETS:
            legal.append(RAISE)
        return legal

    def apply_action(self, state: State, action: Action) -> HoldemState:
        """Deal a card, or apply a betting action."""
        assert isinstance(state, HoldemState)
        if self.is_chance_node(state):
            assert action not in state.cards
            return replace(state, cards=(*state.cards, action))
        me = state.to_move
        history = (*state.history[:-1], state.history[-1] + _ACTION_CHARS[action])
        if action == FOLD:
            return replace(state, history=history, folded=me)
        contrib = list(state.contrib)
        if action == RAISE:
            contrib[me] = max(contrib) + BET_SIZE[state.round]
            return replace(
                state, history=history, contrib=tuple(contrib), bets=state.bets + 1, to_move=1 - me
            )
        contrib[me] = max(contrib)  # check or call
        acted = len(history[-1])
        if (
            acted < 2
        ):  # the first action of a round never closes it (preflop: the big blind's option)
            return replace(state, history=history, contrib=tuple(contrib), to_move=1 - me)
        if state.round == 3:
            return replace(state, history=history, contrib=tuple(contrib), showdown=True)
        return replace(
            state,
            history=(*history, ""),
            contrib=tuple(contrib),
            round=state.round + 1,
            bets=0,
            to_move=1,  # the big blind acts first after the flop
        )

    def hole(self, state: HoldemState, player: Player) -> tuple[int, ...]:
        """``player``'s hole cards (as many as are dealt)."""
        return state.cards[2 * player : 2 * player + 2]

    def board(self, state: HoldemState) -> tuple[int, ...]:
        """The board cards dealt so far."""
        return state.cards[4:]

    def returns(self, state: State) -> list[float]:
        """Big blinds won: the loser's chips, or 0 each on a tied showdown."""
        assert isinstance(state, HoldemState)
        if state.folded >= 0:
            won = state.contrib[state.folded] / CHIPS_PER_BB
            return [-won, won] if state.folded == 0 else [won, -won]
        if not state.showdown:
            return [0.0, 0.0]
        v0 = hand_value(self.hole(state, 0) + self.board(state))
        v1 = hand_value(self.hole(state, 1) + self.board(state))
        pot = state.contrib[0] / CHIPS_PER_BB
        if v0 == v1:
            return [0.0, 0.0]
        return [pot, -pot] if v0 > v1 else [-pot, pot]

    # --- Observations ---------------------------------------------------------------------------
    def information_state_string(self, state: State, player: Player) -> str:
        """Seat, own hole cards, the board, the betting."""
        assert isinstance(state, HoldemState)
        own = "".join(card_str(c) for c in self.hole(state, player))
        board = "".join(card_str(c) for c in self.board(state))
        return f"{player}|{own}|{board}|{'/'.join(state.history)}"

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Cards (one-hot and by rank), the made hand, the betting, by round (177 features)."""
        assert isinstance(state, HoldemState)
        own, board = self.hole(state, player), self.board(state)
        f = np.zeros(177, np.float32)
        for c in own:
            f[c] = 1.0
        for c in board:
            f[52 + c] = 1.0
        ranks = sorted((c // 4 for c in own), reverse=True)
        for i, r in enumerate(ranks):
            f[104 + 13 * i + r] = 1.0
        if len(own) == 2:
            f[130] = float(own[0] % 4 == own[1] % 4)
            f[131] = float(own[0] // 4 == own[1] // 4)
        if len(own) == 2:
            f[132 + category(own + board)] = 1.0
        for c in board:
            f[141 + c // 4] += 0.25
        for c in own + board:
            f[154 + c % 4] += 1 / 7
        f[158 + state.round] = 1.0
        f[162] = float(player == 0)
        for r, moves in enumerate(state.history):  # bets and raises by each seat this round
            seat = 1 if r > 0 else 0  # who acts first in round r
            for k, ch in enumerate(moves):
                if ch == "r":
                    f[163 + 3 * r + (0 if (seat + k) % 2 == player else 1)] += 0.25
            f[165 + 3 * r] = float(len(moves) > 0)
        f[175] = (max(state.contrib) - state.contrib[player]) / 4
        f[176] = sum(state.contrib) / 48
        return f

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> HoldemState:
        """Redeal the opponent's hole cards uniformly from the cards ``player`` cannot see."""
        assert isinstance(state, HoldemState)
        opp = 1 - player
        idx = [i for i in (2 * opp, 2 * opp + 1) if i < len(state.cards)]
        seen = {c for i, c in enumerate(state.cards) if i not in idx}
        unseen = np.array([c for c in range(52) if c not in seen])
        new = rng.choice(unseen, size=len(idx), replace=False)
        cards = list(state.cards)
        for i, c in zip(idx, new, strict=True):
            cards[i] = int(c)
        return replace(state, cards=tuple(cards))

    def action_to_string(self, state: State, action: Action) -> str:
        """Readable labels."""
        assert isinstance(state, HoldemState)
        if self.is_chance_node(state):
            return card_str(action)
        if action == CALL:
            return "call" if self.owe(state) > 0 else "check"
        if action == RAISE:
            return "raise" if state.bets > 0 else "bet"
        return "fold"


@register("limit_holdem")
def _make_limit_holdem() -> LimitHoldem:
    """Heads-up limit Hold'em."""
    return LimitHoldem()
