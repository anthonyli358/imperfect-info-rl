"""The Undertow engine: smugglers against a harbour watch on a river delta at night.

Undertow is an original two-player game on the mechanical skeleton of Reiner Knizia's *Lord of the
Rings: The Confrontation* (credited as inspiration only): two asymmetric sides of nine hidden
characters on a region graph, fights settled by strength plus a secretly played card. Its name,
theme, board, characters, abilities and cards are its own; it is not affiliated with any publisher.

**Setup.** Each side places its nine pieces face down, unseen by the other: four at home (the Cove,
the Free Port) and one in each of the five regions in front. A piece on the board is a *token*:
the opponent sees where token *k* is and how it moves, not who it is.

**Turns.** The Watch moves first. Each turn a side must move one piece *forward* (Smugglers
downstream, the Watch upstream) to an adjacent region; a side that cannot move loses. No piece
may move into or through a region already holding its side's capacity (Cove and Port 4, shoals 1,
channels 2). Abilities: the Runner may move two regions forward to an empty region, the Pilot
ignores capacity, the Cutter may jump two regions forward to attack, the Diver may attack
sideways. At the start of each Watch turn the Inspector, if it is next to hidden Smugglers it has
not identified, privately learns one of them (a chance pick).

**Fights.** Moving into a region held by the enemy starts a fight. With two or more defenders
there, the attacker fights one drawn at random. Both pieces are revealed; each side secretly plays
one card from its hand (:data:`~iirl.games.undertow.board.CARDS`). Trick cards resolve before
number cards, the Watch's first: Lantern ignores the Smugglers' trick; Fog ignores the Watch's
number; Reinforce plays your best used number card instead; Withdraw (sideways) and Feint
(backwards) retreat the piece if it can legally go (else they do nothing); Scuttle sinks both.
Piece abilities override cards: the Gunboat's card is ignored, the Captain ignores the Watch's
trick, the Harbourmaster the Smugglers'. Otherwise each side adds strength and number: the lower
total is defeated, a tie defeats both. A defeated Wrecker takes its opponent with it; a defeated
Decoy turns its attacker back; a Fixer that wins keeps its card; a defeated Signal returns all
the Watch's used cards. Fights continue until the region holds only one side, then the attacker
moves in. Used cards are public; a side takes them all back when its hand is empty.

**Memory.** Fighters are hidden again afterwards, and a side may shuffle its pieces within a
region at any time. So each side's knowledge of an enemy token is a *set* of identities it could
be: a reveal (or a move only one identity could make) narrows it to one, an identity known
elsewhere is ruled out, and tokens sharing a region merge their sets.

**Winning.** The Smugglers win the moment the Contraband enters the Free Port, even if Watch
pieces are there. The Watch wins by defeating the Contraband, or with three pieces in the Cove at
once. A side that cannot move loses. A game that reaches 60 rounds is drawn (a safety limit).

Action ids: moves ``token * 14 + region`` (0–125), cards ``126 + card`` (126–134), setup
placements ``135 + identity`` (135–143). Chance action ids: a token (the random defender, or the
Inspector's look).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility
from iirl.games.undertow.board import (
    BACKWARD,
    CAPACITY,
    CAPTAIN,
    CARDS,
    CONTRABAND,
    COVE,
    CUTTER,
    DECOY,
    DIVER,
    FIXER,
    FORWARD,
    GUNBOAT,
    HAND_SIZE,
    HARBOURMASTER,
    INSPECTOR,
    NUM_REGIONS,
    PILOT,
    PORT,
    RAID,
    REGIONS,
    ROSTERS,
    RUNNER,
    SIDE_NAMES,
    SIDEWAYS,
    SIGNAL,
    SMUGGLERS,
    START,
    WATCH,
    WRECKER,
)

TOKENS = 9
REMOVED = -1
MOVE_BASE, CARD_BASE, SETUP_BASE = 0, 126, 135
NUM_ACTIONS = 144
ALL = (1 << TOKENS) - 1  # every identity possible / every card used
ROUND_LIMIT = 60


class Phase(IntEnum):
    """Where in the game the state sits."""

    SETUP = 0  # a side places a piece (decision)
    INSPECT = 1  # chance: the Inspector's look
    MOVE = 2  # the side to move moves a piece forward
    TARGET = 3  # chance: the defender drawn at random
    ATTACK_CARD = 4  # the attacker plays a card
    DEFEND_CARD = 5  # the defender plays a card, not seeing the attacker's
    OVER = 6


@dataclass(frozen=True)
class Fight:
    """A fight in progress: attacker side and token, from where, into where; defender; card."""

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
        ident: Per side, each token's identity (index into the roster; -1 until placed).
        pos: Per side, each token's region (``REMOVED`` once defeated).
        masks: Per side, for each of that side's tokens, the identities the *opponent* thinks it
            could be (a bitmask). The Watch's view of Smuggler tokens includes what its Inspector
            saw, which the Smugglers do not see.
        used: Per side, a bitmask of used (face-up) cards.
        to_move: The side whose turn it is.
        round: Turns taken by the Watch (it moves first).
        phase: The current :class:`Phase`.
        fight: The fight in progress, if any.
        winner: The winning side, -1 if none (yet, or a draw).
        setup: Pieces placed so far (18 when setup is done).
    """

    ident: tuple[tuple[int, ...], tuple[int, ...]] = ((-1,) * TOKENS, (-1,) * TOKENS)
    pos: tuple[tuple[int, ...], tuple[int, ...]] = START
    masks: tuple[tuple[int, ...], tuple[int, ...]] = ((ALL,) * TOKENS, (ALL,) * TOKENS)
    used: tuple[int, int] = (0, 0)
    to_move: int = WATCH
    round: int = 0
    phase: Phase = Phase.SETUP
    fight: Fight | None = None
    winner: int = -1
    setup: int = 0


def _set(items: tuple, index: int, value: object) -> tuple:
    lst = list(items)
    lst[index] = value
    return tuple(lst)


def _set2(pair: tuple, side: int, index: int, value: object) -> tuple:
    return _set(pair, side, _set(pair[side], index, value))


def _propagate(masks: tuple[int, ...]) -> tuple[int, ...]:
    """Rule out identities that are pinned to some other token (each identity is unique)."""
    m = list(masks)
    changed = True
    while changed:
        changed = False
        for t, mt in enumerate(m):
            if mt and mt & (mt - 1) == 0:  # a single identity
                for u in range(len(m)):
                    if u != t and m[u] & mt:
                        m[u] &= ~mt
                        changed = True
    return tuple(m)


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
        """Nothing placed yet; the Smugglers place first."""
        return UndertowState()

    def num_distinct_actions(self) -> int:
        """Moves, cards and setup placements."""
        return NUM_ACTIONS

    # --- Structure ------------------------------------------------------------------------------
    def is_terminal(self, state: State) -> bool:
        """Whether the game is over."""
        assert isinstance(state, UndertowState)
        return state.phase == Phase.OVER

    def is_chance_node(self, state: State) -> bool:
        """The Inspector's look and the random defender."""
        assert isinstance(state, UndertowState)
        return state.phase in (Phase.INSPECT, Phase.TARGET)

    def current_player(self, state: State) -> Player:
        """The side to act, or a chance/terminal sentinel."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.OVER:
            return PlayerId.TERMINAL
        if self.is_chance_node(state):
            return PlayerId.CHANCE
        if state.phase == Phase.SETUP:
            return state.setup // TOKENS
        if state.phase == Phase.ATTACK_CARD:
            assert state.fight is not None
            return state.fight.side
        if state.phase == Phase.DEFEND_CARD:
            assert state.fight is not None
            return 1 - state.fight.side
        return state.to_move

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Each candidate token equally likely."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.TARGET:
            f = state.fight
            assert f is not None
            options = self.tokens_at(state, 1 - f.side, f.target)
        else:
            options = self._inspect_targets(state)
        return [(t, 1 / len(options)) for t in options]

    # --- Rules helpers --------------------------------------------------------------------------
    def tokens_at(self, state: UndertowState, side: int, region: int) -> list[int]:
        """``side``'s tokens in ``region``."""
        return [t for t, r in enumerate(state.pos[side]) if r == region]

    def _room(self, state: UndertowState, side: int, identity: int, region: int) -> bool:
        if side == SMUGGLERS and identity == PILOT:
            return True
        return len(self.tokens_at(state, side, region)) < CAPACITY[region]

    def moves(self, state: UndertowState, side: int, token: int) -> dict[int, bool]:
        """``{destination: needs_ability}`` for ``side``'s ``token`` this turn."""
        here = state.pos[side][token]
        if here < 0:
            return {}
        identity = state.ident[side][token]
        enemy = 1 - side
        out: dict[int, bool] = {}
        for n in FORWARD[side][here]:
            if self._room(state, side, identity, n):
                out[n] = (
                    side == SMUGGLERS
                    and identity == PILOT
                    and (len(self.tokens_at(state, side, n)) >= CAPACITY[n])
                )
        if side == SMUGGLERS and identity == RUNNER:
            for n in FORWARD[side][here]:
                if self._room(state, side, identity, n) and not self.tokens_at(state, enemy, n):
                    for n2 in FORWARD[side][n]:
                        if not self.tokens_at(state, side, n2) and not self.tokens_at(
                            state, enemy, n2
                        ):
                            out.setdefault(n2, True)
        if side == WATCH and identity == CUTTER:
            for n in FORWARD[side][here]:
                if self._room(state, side, identity, n):
                    for n2 in FORWARD[side][n]:
                        if self.tokens_at(state, enemy, n2) and self._room(
                            state, side, identity, n2
                        ):
                            out.setdefault(n2, True)
        if side == WATCH and identity == DIVER:
            for n in SIDEWAYS[here]:
                if self.tokens_at(state, enemy, n) and self._room(state, side, identity, n):
                    out.setdefault(n, True)
        return out

    def legal_actions(self, state: State) -> list[Action]:
        """Placements, moves or cards, by phase."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.SETUP:
            side = state.setup // TOKENS
            return [SETUP_BASE + i for i in range(TOKENS) if i not in state.ident[side]]
        if state.phase == Phase.MOVE:
            side = state.to_move
            return [
                MOVE_BASE + t * NUM_REGIONS + r
                for t in range(TOKENS)
                for r in sorted(self.moves(state, side, t))
            ]
        fight = state.fight
        assert fight is not None
        side = fight.side if state.phase == Phase.ATTACK_CARD else 1 - fight.side
        return [CARD_BASE + c for c in range(HAND_SIZE) if not state.used[side] >> c & 1]

    def _inspect_targets(self, state: UndertowState) -> list[int]:
        """Smuggler tokens next to the Inspector that the Watch has not pinned down."""
        insp = [t for t in range(TOKENS) if state.ident[WATCH][t] == INSPECTOR]
        if not insp or state.pos[WATCH][insp[0]] < 0:
            return []
        here = state.pos[WATCH][insp[0]]
        near = set(FORWARD[WATCH][here]) | set(BACKWARD[WATCH][here]) | set(SIDEWAYS[here])
        return [
            t
            for t in range(TOKENS)
            if state.pos[SMUGGLERS][t] in near and state.masks[SMUGGLERS][t].bit_count() > 1
        ]

    # --- Transitions ----------------------------------------------------------------------------
    def apply_action(self, state: State, action: Action) -> UndertowState:
        """Apply a placement, move, card, or chance outcome."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.SETUP:
            side, slot = divmod(state.setup, TOKENS)
            s = replace(
                state,
                ident=_set2(state.ident, side, slot, action - SETUP_BASE),
                setup=state.setup + 1,
            )
            if s.setup < 2 * TOKENS:
                return s
            return self._start_turn(replace(s, phase=Phase.MOVE, to_move=WATCH))
        if state.phase == Phase.INSPECT:
            identity = state.ident[SMUGGLERS][action]
            masks = _set2(state.masks, SMUGGLERS, action, 1 << identity)
            masks = _set(masks, SMUGGLERS, _propagate(masks[SMUGGLERS]))
            return replace(state, masks=masks, phase=Phase.MOVE)
        if state.phase == Phase.TARGET:
            assert state.fight is not None
            return self._start_fight(state, replace(state.fight, defender=action))
        if state.phase == Phase.MOVE:
            side = state.to_move
            token, region = divmod(action - MOVE_BASE, NUM_REGIONS)
            needs_ability = self.moves(state, side, token)[region]
            if needs_ability:  # only one identity could have made this move
                state = self._pin(state, side, token)
            identity = state.ident[side][token]
            if side == SMUGGLERS and identity == CONTRABAND and region == PORT:
                state = replace(state, pos=_set2(state.pos, side, token, region))
                return replace(state, winner=SMUGGLERS, phase=Phase.OVER)
            fight = Fight(side, token, state.pos[side][token], region)
            if self.tokens_at(state, 1 - side, region):
                return self._next_fight(state, fight)
            return self._arrive(state, side, token, region)
        fight = state.fight
        assert fight is not None
        if state.phase == Phase.ATTACK_CARD:
            return replace(
                state, fight=replace(fight, card=action - CARD_BASE), phase=Phase.DEFEND_CARD
            )
        return self._resolve(state, fight, action - CARD_BASE)

    def _pin(self, state: UndertowState, side: int, token: int) -> UndertowState:
        """``side``'s ``token`` becomes known to the opponent."""
        masks = _set2(state.masks, side, token, 1 << state.ident[side][token])
        return replace(state, masks=_set(masks, side, _propagate(masks[side])))

    def _next_fight(self, state: UndertowState, fight: Fight) -> UndertowState:
        """A random defender if there are several, else the only one."""
        defenders = self.tokens_at(state, 1 - fight.side, fight.target)
        fight = replace(fight, defender=-1, card=-1)
        if len(defenders) > 1:
            return replace(state, fight=fight, phase=Phase.TARGET)
        return self._start_fight(state, replace(fight, defender=defenders[0]))

    def _start_fight(self, state: UndertowState, fight: Fight) -> UndertowState:
        """Reveal both fighters; the attacker chooses a card (unless the Contraband slips away)."""
        state = self._pin(self._pin(state, fight.side, fight.token), 1 - fight.side, fight.defender)
        d = 1 - fight.side
        if d == SMUGGLERS and state.ident[d][fight.defender] == CONTRABAND:
            r = self._retreat(state, d, fight.defender, SIDEWAYS)
            if r is not None:  # it slips away sideways before any card is played
                state = replace(state, pos=_set2(state.pos, d, fight.defender, r))
                if self.tokens_at(state, d, fight.target):
                    return self._next_fight(state, fight)
                return self._arrive(state, fight.side, fight.token, fight.target)
        return replace(state, fight=fight, phase=Phase.ATTACK_CARD)

    def _arrive(self, state: UndertowState, side: int, token: int, region: int) -> UndertowState:
        state = replace(state, pos=_set2(state.pos, side, token, region))
        if side == WATCH and len(self.tokens_at(state, WATCH, COVE)) >= RAID:
            return replace(state, winner=WATCH, phase=Phase.OVER, fight=None)
        return self._end_turn(state)

    def _best_used(self, state: UndertowState, side: int, exclude: int) -> int:
        values = [
            CARDS[side][c].value
            for c in range(HAND_SIZE)
            if c != exclude and state.used[side] >> c & 1 and not CARDS[side][c].trick
        ]
        return max(values, default=0)

    def _retreat(
        self, state: UndertowState, side: int, token: int, links: tuple[tuple[int, ...], ...]
    ) -> int | None:
        """The first legal retreat for ``token`` along ``links`` from where it stands."""
        here = state.pos[side][token]
        identity = state.ident[side][token]
        for n in links[here]:
            if not self.tokens_at(state, 1 - side, n) and self._room(state, side, identity, n):
                return n
        return None

    def _defeat(self, state: UndertowState, side: int, token: int) -> UndertowState:
        if state.pos[side][token] == REMOVED:
            return state
        state = replace(state, pos=_set2(state.pos, side, token, REMOVED))
        identity = state.ident[side][token]
        if side == WATCH and identity == SIGNAL:
            state = replace(state, used=_set(state.used, WATCH, 0))
        if side == SMUGGLERS and identity == CONTRABAND:
            state = replace(state, winner=WATCH, phase=Phase.OVER)
        return state

    def _resolve(self, state: UndertowState, fight: Fight, d_card: int) -> UndertowState:
        """Both cards are down: settle the fight."""
        a, d = fight.side, 1 - fight.side
        cards = {a: fight.card, d: d_card}
        toks = {a: fight.token, d: fight.defender}
        before = state
        state = replace(
            state,
            used=tuple(state.used[s] | 1 << cards[s] for s in (SMUGGLERS, WATCH)),
            fight=None,
        )
        ids = {s: state.ident[s][toks[s]] for s in (SMUGGLERS, WATCH)}
        card = {s: CARDS[s][cards[s]] for s in (SMUGGLERS, WATCH)}
        value = {s: card[s].value for s in (SMUGGLERS, WATCH)}
        trick = {s: card[s].trick for s in (SMUGGLERS, WATCH)}
        if ids[WATCH] == GUNBOAT:
            value[WATCH], trick[WATCH] = 0, ""
        if ids[SMUGGLERS] == CAPTAIN:
            trick[WATCH] = ""
        if ids[WATCH] == HARBOURMASTER:
            trick[SMUGGLERS] = ""
        if trick[WATCH] == "lantern":
            trick[SMUGGLERS] = ""
        # The attacker's retreat goes back to where it came from: it never left.
        retreated = None
        for s in (WATCH, SMUGGLERS):  # the Watch's trick first
            if trick[s] == "reinforce":
                value[s] = self._best_used(before, s, cards[s])
            elif trick[s] in ("withdraw", "feint") and retreated is None:
                if s == a:
                    retreated = s  # the attack is called off
                else:
                    links = SIDEWAYS if trick[s] == "withdraw" else BACKWARD[s]
                    r = self._retreat(state, s, toks[s], links)
                    if r is not None:
                        state = replace(state, pos=_set2(state.pos, s, toks[s], r))
                        retreated = s
        winner = None
        if retreated == a:
            pass
        elif retreated == d:
            winner = a
        elif trick[SMUGGLERS] == "scuttle":
            state = self._defeat(self._defeat(state, d, toks[d]), a, toks[a])
        else:
            if trick[SMUGGLERS] == "fog":
                value[WATCH] = 0
            total = {s: ROSTERS[s][ids[s]].strength + value[s] for s in (SMUGGLERS, WATCH)}
            if total[a] > total[d]:
                winner = a
                state = self._defeat(state, d, toks[d])
            elif total[d] > total[a]:
                winner = d
                state = self._defeat(state, a, toks[a])
            else:
                state = self._defeat(self._defeat(state, d, toks[d]), a, toks[a])
        for s in (SMUGGLERS, WATCH):  # a defeated Wrecker takes its opponent with it
            if ids[s] == WRECKER and s == SMUGGLERS and state.pos[s][toks[s]] == REMOVED:
                state = self._defeat(state, 1 - s, toks[1 - s])
        if (
            winner == SMUGGLERS
            and ids[SMUGGLERS] == FIXER
            and state.pos[SMUGGLERS][toks[SMUGGLERS]] >= 0
        ):
            state = replace(
                state,
                used=_set(state.used, SMUGGLERS, state.used[SMUGGLERS] & ~(1 << cards[SMUGGLERS])),
            )
        for s in (SMUGGLERS, WATCH):
            if state.used[s] == ALL:
                state = replace(state, used=_set(state.used, s, 0))
        if state.phase == Phase.OVER:
            return state
        if state.pos[a][toks[a]] == REMOVED or winner != a:
            return self._end_turn(state)
        if d == SMUGGLERS and ids[d] == DECOY and state.pos[d][toks[d]] == REMOVED:
            return self._end_turn(state)  # the Decoy turns the attacker back
        if self.tokens_at(state, d, fight.target):
            return self._next_fight(state, fight)
        return self._arrive(state, a, toks[a], fight.target)

    def _start_turn(self, state: UndertowState) -> UndertowState:
        """A side with no move loses; the Inspector looks at the start of the Watch's turn."""
        side = state.to_move
        if not any(self.moves(state, side, t) for t in range(TOKENS)):
            return replace(state, winner=1 - side, phase=Phase.OVER)
        if side == WATCH and self._inspect_targets(state):
            return replace(state, phase=Phase.INSPECT)
        return replace(state, phase=Phase.MOVE)

    def _end_turn(self, state: UndertowState) -> UndertowState:
        """Pieces sharing a region are shuffled (their masks merge); hand over the turn."""
        state = replace(state, fight=None)
        masks = list(state.masks)
        for side in (SMUGGLERS, WATCH):
            m = list(masks[side])
            for region in range(NUM_REGIONS):
                here = self.tokens_at(state, side, region)
                if len(here) > 1:
                    union = 0
                    for t in here:
                        union |= m[t]
                    for t in here:
                        m[t] = union
            masks[side] = _propagate(tuple(m))
        state = replace(state, masks=tuple(masks))
        if state.to_move == WATCH:
            state = replace(state, round=state.round + 1)
            if state.round >= ROUND_LIMIT:
                return replace(state, phase=Phase.OVER)  # drawn
        return self._start_turn(replace(state, to_move=1 - state.to_move))

    def returns(self, state: State) -> list[float]:
        """+1 to the winner, -1 to the loser; 0 each for a draw or before the end."""
        assert isinstance(state, UndertowState)
        if state.winner < 0:
            return [0.0, 0.0]
        return [1.0, -1.0] if state.winner == SMUGGLERS else [-1.0, 1.0]

    # --- Observations ---------------------------------------------------------------------------
    def possible(self, state: UndertowState, viewer: Player, side: int, token: int) -> int:
        """The identities ``viewer`` thinks ``side``'s ``token`` could be, as a bitmask."""
        if side == viewer:
            i = state.ident[side][token]
            return 1 << i if i >= 0 else 0
        return state.masks[side][token]

    def information_state_string(self, state: State, player: Player) -> str:
        """Everything ``player`` knows, as text."""
        assert isinstance(state, UndertowState)
        parts = [f"P{player}", f"r{state.round}", f"ph{int(state.phase)}", f"to{state.to_move}"]
        for side in (SMUGGLERS, WATCH):
            for t in range(TOKENS):
                parts.append(
                    f"{side}.{t}@{state.pos[side][t]}={self.possible(state, player, side, t):03x}"
                )
        parts.append(f"u{state.used[0]:03x},{state.used[1]:03x}")
        f = state.fight
        if f is not None:
            card = f.card if player == f.side else ("x" if f.card >= 0 else "-")
            parts.append(f"F{f.side}.{f.token}:{f.origin}>{f.target}:{f.defender}:{card}")
        return "|".join(parts)

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Per token (own side first): position and possible identities; cards, phase, the fight.

        551 features. Per token (18 × 25): own-side flag, region one-hot (14) plus defeated, and
        the identities ``player`` thinks it could be (9; one for its own pieces). Then the used
        cards of both sides (own first, 18), the phase (7), own side, the round, and the fight:
        target and origin (14 each), attacker and defender tokens (18 each, own first), whether
        ``player`` attacks, and its own attack card (9).
        """
        assert isinstance(state, UndertowState)
        f = np.zeros(551, np.float32)
        order = (player, 1 - player)
        k = 0
        for side in order:
            for t in range(TOKENS):
                base = 25 * k
                f[base] = float(side == player)
                r = state.pos[side][t]
                f[base + 1 + (r if r >= 0 else NUM_REGIONS)] = 1.0
                m = self.possible(state, player, side, t)
                for i in range(TOKENS):
                    f[base + 16 + i] = float(m >> i & 1)
                k += 1
        o = 25 * 18  # 450
        for i, side in enumerate(order):
            for c in range(HAND_SIZE):
                f[o + 9 * i + c] = float(state.used[side] >> c & 1)
        o += 18  # 468
        f[o + int(state.phase)] = 1.0
        o += 7  # 475
        f[o] = float(player == SMUGGLERS)
        f[o + 1] = state.round / ROUND_LIMIT
        o += 2  # 477
        fight = state.fight
        if fight is not None:
            f[o + fight.target] = 1.0
            f[o + 14 + fight.origin] = 1.0
            mine = fight.side == player
            f[o + 28 + (0 if mine else TOKENS) + fight.token] = 1.0
            if fight.defender >= 0:
                f[o + 46 + (TOKENS if mine else 0) + fight.defender] = 1.0
            f[o + 64] = float(mine)
            if mine and fight.card >= 0:
                f[o + 65 + fight.card] = 1.0
        return f  # o + 74 = 551

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> UndertowState:
        """A world consistent with what ``player`` knows: identities drawn within the possible sets.

        Opponent identities are drawn uniformly at random among assignments that respect every
        token's possible set (by randomised backtracking), and a hidden attack card is redrawn
        from the attacker's hand. Simplification: a Smuggler-side world keeps the true record of
        what the Inspector saw.
        """
        assert isinstance(state, UndertowState)
        opp = 1 - player
        if state.phase == Phase.SETUP and state.setup <= opp * TOKENS:
            return state  # the opponent has placed nothing yet
        placed = [t for t in range(TOKENS) if state.ident[opp][t] >= 0]
        masks = [state.masks[opp][t] for t in placed]
        order = sorted(range(len(placed)), key=lambda k: masks[k].bit_count())
        pool = {state.ident[opp][t] for t in placed}
        choice: dict[int, int] = {}

        def assign(i: int, free: set[int]) -> bool:
            if i == len(order):
                return True
            k = order[i]
            options = [x for x in free if masks[k] >> x & 1]
            rng.shuffle(options)
            for x in options:
                choice[placed[k]] = x
                if assign(i + 1, free - {x}):
                    return True
            return False

        if not assign(0, pool):
            return state
        ident = list(state.ident[opp])
        for t, x in choice.items():
            ident[t] = int(x)
        out = replace(state, ident=_set(state.ident, opp, tuple(ident)))
        f = state.fight
        if f is not None and f.card >= 0 and f.side == opp:
            hand = [c for c in range(HAND_SIZE) if not state.used[opp] >> c & 1]
            out = replace(out, fight=replace(f, card=int(rng.choice(hand))))
        return out

    def action_to_string(self, state: State, action: Action) -> str:
        """Readable labels."""
        assert isinstance(state, UndertowState)
        if state.phase == Phase.SETUP:
            side, slot = divmod(state.setup, TOKENS)
            name = ROSTERS[side][action - SETUP_BASE].name
            return f"{SIDE_NAMES[side]} place the {name} in {REGIONS[START[side][slot]]}"
        if state.phase == Phase.INSPECT:
            return f"the Inspector identifies Smuggler token {action}"
        if state.phase == Phase.TARGET:
            return f"token {action} defends"
        if action >= CARD_BASE:
            side = self.current_player(state)
            return f"play {CARDS[side][action - CARD_BASE].name}"
        token, region = divmod(action - MOVE_BASE, NUM_REGIONS)
        return f"{SIDE_NAMES[state.to_move]} token {token} to {REGIONS[region]}"


@register("undertow")
def _make_undertow() -> Undertow:
    """Undertow."""
    return Undertow()
