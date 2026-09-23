"""A tiny two-player showdown game — a mini Kuhn poker — used only to exercise the spine.

This lives in ``tests/`` on purpose: it is a fixture for the conformance suite, the runner, and the
tournament, not a "real" game (Lost Cities in Stage 1 is the first of those). It is deliberately the
smallest thing that stresses every part of the interface: chance nodes (the deal), hidden info
(you see your own card, not the opponent's), sequential play, and zero-sum terminal returns. It
also doubles as a worked template for implementing an engine against ``Game``.

Rules (standard Kuhn poker): a 3-card deck {0, 1, 2}; each player antes 1 and is dealt one card.
Player 0 checks or bets; player 1 responds; if player 0 checked and player 1 bet, player 0 gets a
final fold/call. At showdown the higher card wins the pot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility

_DECK = (0, 1, 2)
CHECK_FOLD = 0  # "check" as the opener, "fold" when facing a bet
BET_CALL = 1  # "bet" as the opener, "call" when facing a bet

# Betting histories at which the hand is over, mapped to the pot swing paid by the loser.
# CALL/showdown hands (2-chip swing) resolve by card; fold hands (1-chip) resolve by who folded.
_FOLD_WINNERS = {(BET_CALL, CHECK_FOLD): 0, (CHECK_FOLD, BET_CALL, CHECK_FOLD): 1}
_SHOWDOWN_HISTORIES = {
    (CHECK_FOLD, CHECK_FOLD): 1.0,
    (BET_CALL, BET_CALL): 2.0,
    (CHECK_FOLD, BET_CALL, BET_CALL): 2.0,
}


@dataclass(frozen=True)
class ToyState(State):
    """Immutable state: the cards dealt so far and the betting history.

    Attributes:
        cards: Cards dealt, in seat order; length 0, 1, or 2 while dealing.
        history: Betting actions taken so far (each ``CHECK_FOLD`` or ``BET_CALL``).
    """

    cards: tuple[int, ...] = ()
    history: tuple[int, ...] = ()


def _player_to_act(history: tuple[int, ...]) -> Player:
    """Return whose turn it is given a (non-terminal) post-deal betting history."""
    if history == ():
        return 0
    if history in {(CHECK_FOLD,), (BET_CALL,)}:
        return 1
    # The only remaining live node is check-bet, where player 0 faces the fold/call.
    return 0


class ToyShowdown(Game):
    """Mini Kuhn poker: the smallest game that exercises the full ``Game`` contract."""

    _TYPE = GameType(
        short_name="toy_showdown",
        long_name="Toy Showdown (mini Kuhn poker)",
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

    def new_initial_state(self) -> ToyState:
        """Return the pre-deal state (a chance node)."""
        return ToyState()

    def current_player(self, state: State) -> Player:
        """Return the current player, or a chance/terminal sentinel."""
        assert isinstance(state, ToyState)
        if len(state.cards) < 2:
            return PlayerId.CHANCE
        if self.is_terminal(state):
            return PlayerId.TERMINAL
        return _player_to_act(state.history)

    def legal_actions(self, state: State) -> list[Action]:
        """Return the legal actions at a player decision node (always check/fold or bet/call)."""
        return [CHECK_FOLD, BET_CALL]

    def num_distinct_actions(self) -> int:
        """Return the flat action-space size (check/fold, bet/call)."""
        return 2

    def apply_action(self, state: State, action: Action) -> ToyState:
        """Return the state reached by applying ``action`` (deal a card, or a betting move)."""
        assert isinstance(state, ToyState)
        if len(state.cards) < 2:
            return ToyState(cards=state.cards + (action,), history=state.history)
        if action not in (CHECK_FOLD, BET_CALL):
            raise ValueError(f"Illegal betting action {action}")
        return ToyState(cards=state.cards, history=state.history + (action,))

    def is_chance_node(self, state: State) -> bool:
        """Return whether a card still needs to be dealt."""
        assert isinstance(state, ToyState)
        return len(state.cards) < 2

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Return the uniform deal outcomes (a card value paired with its probability)."""
        assert isinstance(state, ToyState)
        remaining = [c for c in _DECK if c not in state.cards]
        p = 1.0 / len(remaining)
        return [(c, p) for c in remaining]

    def is_terminal(self, state: State) -> bool:
        """Return whether the hand is over."""
        assert isinstance(state, ToyState)
        if len(state.cards) < 2:
            return False
        return state.history in _FOLD_WINNERS or state.history in _SHOWDOWN_HISTORIES

    def returns(self, state: State) -> list[float]:
        """Return the zero-sum per-player payoffs at a terminal state."""
        assert isinstance(state, ToyState)
        if state.history in _FOLD_WINNERS:
            winner = _FOLD_WINNERS[state.history]
            amount = 1.0
        else:
            amount = _SHOWDOWN_HISTORIES[state.history]
            winner = 0 if state.cards[0] > state.cards[1] else 1
        payoff = [0.0, 0.0]
        payoff[winner] = amount
        payoff[1 - winner] = -amount
        return payoff

    def information_state_string(self, state: State, player: Player) -> str:
        """Return a hashable infoset key: the player's own card and the public betting history."""
        assert isinstance(state, ToyState)
        card = state.cards[player] if player < len(state.cards) else -1
        return f"{card}|{''.join(str(a) for a in state.history)}"

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Encode ``player``'s view: one-hot own card (3) + betting history (3 slots x 2).

        The opponent's card is never encoded — that is the hidden information. Shape is a constant
        9 regardless of how far the hand has progressed.
        """
        assert isinstance(state, ToyState)
        tensor = np.zeros(3 + 3 * 2, dtype=np.float32)
        if player < len(state.cards):
            tensor[state.cards[player]] = 1.0
        for slot, action in enumerate(state.history):
            tensor[3 + slot * 2 + action] = 1.0
        return tensor

    def resample_from_infostate(self, state: State, player: Player, rng: np.random.Generator):
        """Return a world consistent with ``player``'s view (own card kept, opponent's re-dealt)."""
        assert isinstance(state, ToyState)
        unseen = [c for c in _DECK if c != state.cards[player]]
        opp_card = int(rng.choice(unseen))
        cards = (state.cards[0], opp_card) if player == 0 else (opp_card, state.cards[1])
        return ToyState(cards=cards, history=state.history)

    def display_meta(self) -> dict:
        """Return viewer metadata: card names and action labels."""
        return {
            "card_names": ["Low (0)", "Mid (1)", "High (2)"],
            "action_names": ["check / fold", "bet / call"],
        }

    def to_display(self, state: State) -> dict:
        """Return a render-ready snapshot: dealt cards, betting history, phase."""
        assert isinstance(state, ToyState)
        return {
            "cards": list(state.cards),
            "history": list(state.history),
            "terminal": self.is_terminal(state),
            "to_move": int(self.current_player(state)),
        }


@register("toy_showdown")
def _make_toy_showdown() -> ToyShowdown:
    """Factory registered under ``"toy_showdown"``."""
    return ToyShowdown()
