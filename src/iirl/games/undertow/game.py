"""The Undertow engine: smugglers against a harbour watch on a river delta at night.

Undertow is an original two-player game on the mechanical skeleton of Reiner Knizia's *Lord of the
Rings: The Confrontation* (two asymmetric sides of nine hidden characters, a region graph, combat
cards), credited as inspiration only: its name, theme, board, characters, abilities and victory
conditions are its own. It is not affiliated with any publisher.

**Setup.** Each side's nine pieces are shuffled face down onto fixed starting regions (chance
events). A piece is a *token*: the opponent sees where token *k* is and that it moved, not who it
is, until it is revealed. Board, rosters and cards: :mod:`~iirl.games.undertow.board`.

**Turns.** The Smugglers move first; each turn one piece moves to an adjacent region (the Runner
and the Cutter may move two, through a region with no enemy). Smugglers move only downstream or
sideways; the Watch any way; the Harbourmaster stays in the port channels. A region holds at most
its capacity of one side's pieces (the Pilot ignores this). With no legal move, a side passes.
At the start of each Watch turn, if the Inspector is on the board next to hidden Smugglers it has
not seen, it privately learns one of them (a chance pick).

**Fights.** Moving into a region held by the enemy starts a fight. The defending side picks which
of its pieces there defends; both pieces are revealed to both sides for the rest of the game. The
attacker then the defender each choose one card from their own hand (the defender does not see
the attacker's choice). Lantern cancels the other card. Feint retreats without loss unless
cancelled: the attacker stays where it was, or the defender moves to the first adjacent region it
may legally reach (with none, Feint counts as 0). Otherwise each side adds its piece's strength
and its card's value: the higher wins and the loser is removed; a tie removes both. An attacker
that wins fights the region's next defender, until it loses or the region is clear, then moves
in. Special cases: a Decoy that loses as defender turns the attacker back; a Fixer that wins
keeps its card; when the Signal is removed the Watch takes back all its played cards. Played cards
are public; a side takes them all back when its hand is empty.

**Winning.** A Cargo boat that reaches the Free Port with no Watch piece there (or after winning
its fights there) is delivered. The Smugglers win (+1) on delivering Cargo worth 6. The Watch wins
as soon as 6 is out of reach (cargo delivered plus cargo still afloat is under 6: with 8 in all,
that means Cargo worth 3 sunk), or when the tide runs out after 24 rounds. (The approved draft
had 16 rounds; that is too short to win at all: three Cargo boats are needed, each 5-6 moves from
the Free Port, so 15-18 Smuggler moves before any fight. Random play delivered nothing in 2,000
games.)

Action ids: moves ``token * 13 + region`` (0–116), cards ``117 + card`` (117–122), defender picks
``123 + token`` (123–131), pass ``132``. Chance action ids: an identity (setup) or a Smuggler token
(the Inspector's look).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility
from iirl.games.undertow.board import (
    ADJACENT,
    CAPACITY,
    CARD_NAMES,
    CARD_VALUES,
    CARGO_TO_WIN,
    DECOY,
    DEPTH,
    FEINT,
    FIXER,
    HARBOURMASTER,
    INSPECTOR,
    LANTERN,
    NUM_CARDS,
    NUM_REGIONS,
    PILOT,
    PORT,
    PORT_CHANNELS,
    REGIONS,
    ROSTERS,
    ROUNDS,
    SIDE_NAMES,
    SIGNAL,
    SMUGGLERS,
    START,
    TOTAL_CARGO,
    TWO_STEP,
    WATCH,
)

TOKENS = 9
REMOVED, DELIVERED = -1, -2
MOVE_BASE, CARD_BASE, PICK_BASE, PASS = 0, 117, 123, 132
NUM_ACTIONS = 133
FULL_HAND = (1 << NUM_CARDS) - 1


class Phase(IntEnum):
    """Where in the turn the state sits."""

    SETUP = 0  # chance: shuffle identities onto tokens
    INSPECT = 1  # chance: the Inspector's look
    MOVE = 2  # the side to move moves a piece (or passes)
    PICK = 3  # the defending side picks its defender
    ATTACK_CARD = 4  # the attacker plays a card
    DEFEND_CARD = 5  # the defender plays a card, not seeing the attacker's
    OVER = 6


@dataclass(frozen=True)
class Fight:
    """A fight in progress: who attacks, from where, into where; the defender; the attack card."""

    side: int
    token: int
    origin: int
    target: int
    defender: int = -1
    card: int = -1


@dataclass(frozen=True)
class UndertowState(State):
    """A complete Undertow position.

    Attributes:
        ident: Per side, each token's identity (an index into the roster; -1 before setup).
        pos: Per side, each token's region (``REMOVED`` or ``DELIVERED`` off the board).
        revealed: Per side, a bitmask of tokens revealed to both sides.
        known: Bitmask of Smuggler tokens the Watch has seen with the Inspector (Watch-private).
        hands: Per side, a bitmask of the cards in hand.
        to_move: The side whose turn it is.
        round: Rounds completed (one Smuggler turn and one Watch turn each).
        phase: The current :class:`Phase`.
        fight: The fight in progress, if any.
        delivered: Cargo value delivered to the Free Port.
        sunk: Cargo value sunk.
        winner: The winning side, or -1.
        setup: Identities placed so far (18 when setup is done).
    """

    ident: tuple[tuple[int, ...], tuple[int, ...]] = ((-1,) * TOKENS, (-1,) * TOKENS)
    pos: tuple[tuple[int, ...], tuple[int, ...]] = START
    revealed: tuple[int, int] = (0, 0)
    known: int = 0
    hands: tuple[int, int] = (FULL_HAND, FULL_HAND)
    to_move: int = SMUGGLERS
    round: int = 0
    phase: Phase = Phase.SETUP
    fight: Fight | None = None
    delivered: int = 0
    sunk: int = 0
    winner: int = -1
    setup: int = 0


def _set(items: tuple, index: int, value: object) -> tuple:
    lst = list(items)
    lst[index] = value
    return tuple(lst)


def _set2(pair: tuple, side: int, index: int, value: object) -> tuple:
    return _set(pair, side, _set(pair[side], index, value))


class Undertow(Game):
    """Undertow: Smugglers (player 0) against the Watch (player 1)."""

    _type = GameType(
        short_name="undertow",
        long_name="Undertow (smugglers against the harbour watch)",
        num_players=2,
        information=Information.IMPERFECT,
        utility=Utility.ZERO_SUM,
        provides_information_state_tensor=True,
    )

    @property
    def game_type(self) -> GameType:
        """Static metadata."""
        return self._type

    def new_initial_state(self) -> UndertowState:
        """Pieces on their starting regions, identities not yet shuffled."""
        return UndertowState()

    def num_distinct_actions(self) -> int:
        """Moves, cards, defender picks and pass."""
        return NUM_ACTIONS

    # --- Structure ------------------------------------------------------------------------------
    def is_terminal(self, state: State) -> bool:
        """Whether someone has won."""
        assert isinstance(state, UndertowState)
        return state.phase == Phase.OVER

    def is_chance_node(self, state: State) -> bool:
        """Setup shuffles and the Inspector's look."""
        assert isinstance(state, UndertowState)
        return state.phase in (Phase.SETUP, Phase.INSPECT)

    def current_player(self, state: State) -> Player:
        """The side to act, or a chance/terminal sentinel."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.OVER:
            return PlayerId.TERMINAL
        if self.is_chance_node(state):
            return PlayerId.CHANCE
        if state.phase in (Phase.PICK, Phase.DEFEND_CARD):
            assert state.fight is not None
            return 1 - state.fight.side
        if state.phase == Phase.ATTACK_CARD:
            assert state.fight is not None
            return state.fight.side
        return state.to_move

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Setup: each unplaced identity equally likely; the Inspector: each unseen neighbour."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.SETUP:
            side = state.setup // TOKENS
            left = [i for i in range(TOKENS) if i not in state.ident[side]]
            return [(i, 1 / len(left)) for i in left]
        targets = self._inspect_targets(state)
        return [(t, 1 / len(targets)) for t in targets]

    # --- Rules helpers --------------------------------------------------------------------------
    def tokens_at(self, state: UndertowState, side: int, region: int) -> list[int]:
        """``side``'s tokens in ``region``."""
        return [t for t, r in enumerate(state.pos[side]) if r == region]

    def _steps(self, side: int, identity: int, region: int) -> list[int]:
        """Regions one step from ``region`` that this piece may move to (ignoring occupancy)."""
        out = []
        for n in ADJACENT[region]:
            if side == SMUGGLERS and DEPTH[n] < DEPTH[region]:
                continue
            if side == WATCH and identity == HARBOURMASTER and n not in PORT_CHANNELS:
                continue
            out.append(n)
        return out

    def _room(self, state: UndertowState, side: int, identity: int, region: int) -> bool:
        if side == SMUGGLERS and identity == PILOT:
            return True
        return len(self.tokens_at(state, side, region)) < CAPACITY[region]

    def destinations(self, state: UndertowState, side: int, token: int) -> list[int]:
        """Where ``side``'s ``token`` may move this turn (enemy-held regions mean a fight)."""
        here = state.pos[side][token]
        if here < 0:
            return []
        identity = state.ident[side][token]
        enemy = 1 - side
        dests = set()
        for n in self._steps(side, identity, here):
            if self._room(state, side, identity, n):
                dests.add(n)
            if identity in TWO_STEP[side] and not self.tokens_at(state, enemy, n):
                for n2 in self._steps(side, identity, n):
                    if n2 != here and self._room(state, side, identity, n2):
                        dests.add(n2)
        return sorted(dests)

    def _inspect_targets(self, state: UndertowState) -> list[int]:
        """Hidden Smuggler tokens next to the Inspector that the Watch has not seen."""
        insp = [t for t in range(TOKENS) if state.ident[WATCH][t] == INSPECTOR]
        if not insp or state.pos[WATCH][insp[0]] < 0:
            return []
        near = set(ADJACENT[state.pos[WATCH][insp[0]]])
        seen = state.revealed[SMUGGLERS] | state.known
        return [t for t in range(TOKENS) if state.pos[SMUGGLERS][t] in near and not (seen >> t & 1)]

    def legal_actions(self, state: State) -> list[Action]:
        """Moves (or pass), a defender pick, or a card from hand, by phase."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.MOVE:
            side = state.to_move
            moves = [
                MOVE_BASE + t * NUM_REGIONS + r
                for t in range(TOKENS)
                for r in self.destinations(state, side, t)
            ]
            return moves or [PASS]
        fight = state.fight
        assert fight is not None
        if state.phase == Phase.PICK:
            return [PICK_BASE + t for t in self.tokens_at(state, 1 - fight.side, fight.target)]
        side = fight.side if state.phase == Phase.ATTACK_CARD else 1 - fight.side
        return [CARD_BASE + c for c in range(NUM_CARDS) if state.hands[side] >> c & 1]

    # --- Transitions ----------------------------------------------------------------------------
    def apply_action(self, state: State, action: Action) -> UndertowState:
        """Apply a chance outcome or a player action."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.SETUP:
            side, token = divmod(state.setup, TOKENS)
            s = replace(state, ident=_set2(state.ident, side, token, action), setup=state.setup + 1)
            return replace(s, phase=Phase.MOVE) if s.setup == 2 * TOKENS else s
        if state.phase == Phase.INSPECT:
            return replace(state, known=state.known | (1 << action), phase=Phase.MOVE)
        if state.phase == Phase.MOVE:
            if action == PASS:
                return self._end_turn(state)
            token, region = divmod(action - MOVE_BASE, NUM_REGIONS)
            side = state.to_move
            fight = Fight(side, token, state.pos[side][token], region)
            if self.tokens_at(state, 1 - side, region):
                return self._next_fight(state, fight)
            return self._arrive(state, side, token, region)
        fight = state.fight
        assert fight is not None
        if state.phase == Phase.PICK:
            return self._start_fight(state, replace(fight, defender=action - PICK_BASE))
        if state.phase == Phase.ATTACK_CARD:
            return replace(
                state, fight=replace(fight, card=action - CARD_BASE), phase=Phase.DEFEND_CARD
            )
        return self._resolve(state, fight, action - CARD_BASE)

    def _next_fight(self, state: UndertowState, fight: Fight) -> UndertowState:
        """The defending side picks a defender (or the only one fights)."""
        defenders = self.tokens_at(state, 1 - fight.side, fight.target)
        if len(defenders) == 1:
            return self._start_fight(state, replace(fight, defender=defenders[0], card=-1))
        return replace(state, fight=replace(fight, defender=-1, card=-1), phase=Phase.PICK)

    def _start_fight(self, state: UndertowState, fight: Fight) -> UndertowState:
        """Reveal both fighters; the attacker chooses a card."""
        revealed = _set(state.revealed, fight.side, state.revealed[fight.side] | 1 << fight.token)
        d = 1 - fight.side
        revealed = _set(revealed, d, revealed[d] | 1 << fight.defender)
        return replace(state, revealed=revealed, fight=fight, phase=Phase.ATTACK_CARD)

    def _arrive(self, state: UndertowState, side: int, token: int, region: int) -> UndertowState:
        """Move a piece in; deliver cargo at the Free Port; end the turn."""
        identity = state.ident[side][token]
        cargo = ROSTERS[side][identity].cargo
        if side == SMUGGLERS and cargo and region == PORT:
            revealed = _set(state.revealed, side, state.revealed[side] | 1 << token)
            state = replace(
                state,
                pos=_set2(state.pos, side, token, DELIVERED),
                delivered=state.delivered + cargo,
                revealed=revealed,
            )
        else:
            state = replace(state, pos=_set2(state.pos, side, token, region))
        return self._end_turn(state)

    def _remove(self, state: UndertowState, side: int, token: int) -> UndertowState:
        """Take a piece off the board, with the effects of losing it."""
        identity = state.ident[side][token]
        state = replace(state, pos=_set2(state.pos, side, token, REMOVED))
        if side == SMUGGLERS:
            state = replace(state, sunk=state.sunk + ROSTERS[side][identity].cargo)
        elif identity == SIGNAL:
            state = replace(state, hands=_set(state.hands, WATCH, FULL_HAND))
        return state

    def _resolve(self, state: UndertowState, fight: Fight, d_card: int) -> UndertowState:
        """Both cards are down: settle the fight."""
        a, d = fight.side, 1 - fight.side
        a_card = fight.card
        a_id, d_id = state.ident[a][fight.token], state.ident[d][fight.defender]
        hands = _set(state.hands, a, state.hands[a] & ~(1 << a_card))
        hands = _set(hands, d, hands[d] & ~(1 << d_card))
        state = replace(state, hands=hands, fight=None)
        a_cancel, d_cancel = d_card == LANTERN, a_card == LANTERN
        a_val = 0 if a_cancel else CARD_VALUES[a_card]
        d_val = 0 if d_cancel else CARD_VALUES[d_card]
        winner = None
        if a_card == FEINT and not a_cancel:
            pass  # the attacker never left
        elif d_card == FEINT and not d_cancel and (r := self._retreat(state, d, fight)) is not None:
            state = replace(state, pos=_set2(state.pos, d, fight.defender, r))
            winner = a
        else:
            a_total = ROSTERS[a][a_id].strength + a_val
            d_total = ROSTERS[d][d_id].strength + d_val
            if a_total > d_total:
                winner = a
                state = self._remove(state, d, fight.defender)
            elif d_total > a_total:
                winner = d
                state = self._remove(state, a, fight.token)
            else:
                state = self._remove(self._remove(state, d, fight.defender), a, fight.token)
        # Cards come back: a winning Fixer keeps its card; an empty hand refills.
        if winner is not None:
            w_id = a_id if winner == a else d_id
            w_card = a_card if winner == a else d_card
            if winner == SMUGGLERS and w_id == FIXER:
                state = replace(
                    state, hands=_set(state.hands, winner, state.hands[winner] | 1 << w_card)
                )
        for side in (SMUGGLERS, WATCH):
            if state.hands[side] == 0:
                state = replace(state, hands=_set(state.hands, side, FULL_HAND))
        state = self._check_winner(state)
        if state.phase == Phase.OVER:
            return state
        if winner != a or state.pos[a][fight.token] < 0:
            return self._end_turn(state)  # the attack is over: lost, tied, or feinted
        if d == SMUGGLERS and d_id == DECOY and state.pos[d][fight.defender] == REMOVED:
            return self._end_turn(state)  # the Decoy turns the attacker back
        if self.tokens_at(state, d, fight.target):
            return self._next_fight(state, fight)
        return self._arrive(state, a, fight.token, fight.target)

    def _retreat(self, state: UndertowState, side: int, fight: Fight) -> int | None:
        """The first region the feinting defender may legally reach, or None."""
        identity = state.ident[side][fight.defender]
        for n in self._steps(side, identity, fight.target):
            if not self.tokens_at(state, 1 - side, n) and self._room(state, side, identity, n):
                return n
        return None

    def _check_winner(self, state: UndertowState) -> UndertowState:
        if state.delivered >= CARGO_TO_WIN:
            return replace(state, winner=SMUGGLERS, phase=Phase.OVER, fight=None)
        afloat = sum(
            ROSTERS[SMUGGLERS][i].cargo
            for i, p in zip(state.ident[SMUGGLERS], state.pos[SMUGGLERS], strict=True)
            if p >= 0
        )
        if state.delivered + afloat < CARGO_TO_WIN:  # 6 is out of reach
            return replace(state, winner=WATCH, phase=Phase.OVER, fight=None)
        return state

    def _end_turn(self, state: UndertowState) -> UndertowState:
        """Check for a winner, advance the tide, hand over the turn."""
        state = self._check_winner(replace(state, fight=None))
        if state.phase == Phase.OVER:
            return state
        if state.to_move == WATCH:
            state = replace(state, round=state.round + 1)
            if state.round >= ROUNDS:
                return replace(state, winner=WATCH, phase=Phase.OVER)
            return replace(state, to_move=SMUGGLERS, phase=Phase.MOVE)
        state = replace(state, to_move=WATCH, phase=Phase.MOVE)
        if self._inspect_targets(state):
            state = replace(state, phase=Phase.INSPECT)
        return state

    def returns(self, state: State) -> list[float]:
        """+1 to the winner, -1 to the loser; 0 each before the end."""
        assert isinstance(state, UndertowState)
        if state.winner < 0:
            return [0.0, 0.0]
        return [1.0, -1.0] if state.winner == SMUGGLERS else [-1.0, 1.0]

    def heuristic_value(self, state: State, player: Player) -> float:
        """A leaf estimate in ``[-1, 1]``: cargo delivered, sunk and afloat (by depth), the tide."""
        assert isinstance(state, UndertowState)
        if state.winner >= 0:
            return self.returns(state)[player]
        afloat = sum(
            ROSTERS[SMUGGLERS][i].cargo * DEPTH[p] / DEPTH[PORT]
            for i, p in zip(state.ident[SMUGGLERS], state.pos[SMUGGLERS], strict=True)
            if p >= 0
        )
        x = 2.0 * state.delivered + 1.0 * afloat - 2.0 * state.sunk - 0.4 * state.round - 2.0
        v = float(np.tanh(x / 6.0))
        return v if player == SMUGGLERS else -v

    # --- Observations ---------------------------------------------------------------------------
    def knows(self, state: UndertowState, viewer: Player, side: int, token: int) -> bool:
        """Whether ``viewer`` knows the identity of ``side``'s ``token``."""
        if state.ident[side][token] < 0:
            return False
        if side == viewer or state.revealed[side] >> token & 1:
            return True
        return viewer == WATCH and bool(state.known >> token & 1)

    def information_state_string(self, state: State, player: Player) -> str:
        """Everything ``player`` knows, as text."""
        assert isinstance(state, UndertowState)
        parts = [f"P{player}", f"r{state.round}", f"ph{int(state.phase)}", f"to{state.to_move}"]
        for side in (SMUGGLERS, WATCH):
            for t in range(TOKENS):
                who = state.ident[side][t] if self.knows(state, player, side, t) else "?"
                parts.append(f"{side}.{t}@{state.pos[side][t]}={who}")
        parts.append(f"h{state.hands[0]},{state.hands[1]}")
        parts.append(f"d{state.delivered}s{state.sunk}")
        f = state.fight
        if f is not None:
            card = f.card if player == f.side else ("x" if f.card >= 0 else "-")
            parts.append(f"F{f.side}.{f.token}:{f.origin}>{f.target}:{f.defender}:{card}")
        return "|".join(parts)

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Per token (own side first): position, identity if known; then hands, clock, the fight.

        560 features. Per token (18 × 26): own-side flag, region one-hot (13) plus removed and
        delivered, identity one-hot (9) plus unknown. Then both hands (own first), the phase (7),
        own side, round, delivered and sunk cargo, and the fight: target, origin, attacker and
        defender tokens (in the same own-first order), whether ``player`` attacks, and the attack
        card (only for the attacker).
        """
        assert isinstance(state, UndertowState)
        f = np.zeros(560, np.float32)
        order = (player, 1 - player)
        k = 0
        for side in order:
            for t in range(TOKENS):
                base = 26 * k
                f[base] = float(side == player)
                r = state.pos[side][t]
                f[base + 1 + (r if r >= 0 else (13 if r == REMOVED else 14))] = 1.0
                if self.knows(state, player, side, t):
                    f[base + 16 + state.ident[side][t]] = 1.0
                else:
                    f[base + 25] = 1.0
                k += 1
        o = 26 * 18  # 468
        for i, side in enumerate(order):
            for c in range(NUM_CARDS):
                f[o + 6 * i + c] = float(state.hands[side] >> c & 1)
        o += 12  # 480
        f[o + int(state.phase)] = 1.0
        o += 7  # 487
        f[o] = float(player == SMUGGLERS)
        f[o + 1] = state.round / ROUNDS
        f[o + 2] = state.delivered / TOTAL_CARGO
        f[o + 3] = state.sunk / TOTAL_CARGO
        o += 4  # 491
        fight = state.fight
        if fight is not None:
            f[o + fight.target] = 1.0
            f[o + 13 + fight.origin] = 1.0
            mine = fight.side == player
            f[o + 26 + (0 if mine else TOKENS) + fight.token] = 1.0
            if fight.defender >= 0:
                f[o + 44 + (TOKENS if mine else 0) + fight.defender] = 1.0
            f[o + 62] = float(mine)
            if mine and fight.card >= 0:
                f[o + 63 + fight.card] = 1.0
        return f  # o + 69 = 560

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> UndertowState:
        """A world consistent with what ``player`` knows (uniform over the hidden details).

        The opponent's unknown identities are reshuffled among their unknown tokens, and a hidden
        attack card is redrawn from the attacker's hand. Two simplifications: the shuffle ignores
        what the opponent's moves imply (e.g. a token outside the port channels is not the
        Harbourmaster), and the Smugglers keep the true record of what the Inspector saw.
        """
        assert isinstance(state, UndertowState)
        if state.phase == Phase.SETUP:
            return state
        opp = 1 - player
        hidden = [t for t in range(TOKENS) if not self.knows(state, player, opp, t)]
        ident = list(state.ident[opp])
        shuffled = [ident[t] for t in hidden]
        rng.shuffle(shuffled)
        for t, i in zip(hidden, shuffled, strict=True):
            ident[t] = int(i)
        out = replace(state, ident=_set(state.ident, opp, tuple(ident)))
        f = state.fight
        if f is not None and f.card >= 0 and f.side == opp:
            hand = [c for c in range(NUM_CARDS) if state.hands[opp] >> c & 1]
            out = replace(out, fight=replace(f, card=int(rng.choice(hand))))
        return out

    def action_to_string(self, state: State, action: Action) -> str:
        """Readable labels."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.SETUP:
            side = state.setup // TOKENS
            return f"token {state.setup % TOKENS} is the {ROSTERS[side][action].name}"
        if state.phase == Phase.INSPECT:
            return f"the Inspector sees Smuggler token {action}"
        if action == PASS:
            return "pass"
        if action >= PICK_BASE:
            return f"token {action - PICK_BASE} defends"
        if action >= CARD_BASE:
            return f"play {CARD_NAMES[action - CARD_BASE]}"
        token, region = divmod(action - MOVE_BASE, NUM_REGIONS)
        side = SIDE_NAMES[state.to_move]
        return f"{side} token {token} to {REGIONS[region]}"


@register("undertow")
def _make_undertow() -> Undertow:
    """Undertow."""
    return Undertow()
