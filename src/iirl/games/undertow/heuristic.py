"""A rule-based Undertow baseline for either side.

Game-specific by design: an Elo anchor and the first balance check. It knows only what its side
knows: an opponent piece it has not seen counts at the average strength (and cargo) of the
opponent's identities it has not seen.

- **Moves.** Smugglers push Cargo boats downstream, away from Watch pieces, and send their
  fighters at Watch pieces they expect to beat. The Watch closes on the nearest Smugglers, attacks
  where its strength (plus an average card) should win, the more so where cargo is likely, and
  keeps at least one piece guarding the Mouth or the Free Port.
- **Defender picks.** Smugglers defend with the Decoy if it is there (it turns the attacker back),
  else their strongest piece; the Watch with its strongest.
- **Cards.** The smallest card that wins even against the opponent's best remaining card; failing
  that, Lantern if strength alone wins; Feint if the fight is lost anyway; else the highest card.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.registry import register_heuristic
from iirl.core.types import Action
from iirl.games.undertow.board import (
    ADJACENT,
    CARD_VALUES,
    DECOY,
    DEPTH,
    FEINT,
    LANTERN,
    NUM_CARDS,
    NUM_REGIONS,
    PORT,
    ROSTERS,
    SMUGGLERS,
)
from iirl.games.undertow.game import (
    CARD_BASE,
    MOVE_BASE,
    PASS,
    PICK_BASE,
    TOKENS,
    Phase,
    Undertow,
    UndertowState,
)

GUARD = frozenset({11, PORT})


def _distances() -> list[list[int]]:
    out = []
    for src in range(NUM_REGIONS):
        dist = [99] * NUM_REGIONS
        dist[src] = 0
        queue = deque([src])
        while queue:
            r = queue.popleft()
            for n in ADJACENT[r]:
                if dist[n] == 99:
                    dist[n] = dist[r] + 1
                    queue.append(n)
        out.append(dist)
    return out


DIST = _distances()


def estimate(game: Undertow, state: UndertowState, viewer: int, side: int, token: int) -> tuple:
    """``(strength, cargo)`` of a piece as ``viewer`` sees it (the unseen average if hidden)."""
    if game.knows(state, viewer, side, token):
        piece = ROSTERS[side][state.ident[side][token]]
        return piece.strength, piece.cargo
    known = {state.ident[side][t] for t in range(TOKENS) if game.knows(state, viewer, side, t)}
    unseen = [ROSTERS[side][i] for i in range(TOKENS) if i not in known]
    return (
        float(np.mean([p.strength for p in unseen])),
        float(np.mean([p.cargo for p in unseen])),
    )


@register_heuristic("undertow")
class UndertowHeuristic(Agent):
    """Rule-based play for whichever side it is seated as; see the module docstring."""

    def __init__(self, name: str = "heuristic") -> None:
        """Name the agent."""
        self.name = name

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Pick a move, a defender or a card by the rules above."""
        assert isinstance(game, Undertow) and isinstance(state, UndertowState)
        legal = game.legal_actions(state)
        if legal == [PASS]:
            return PASS
        if state.phase == Phase.MOVE:
            scores = [self._move_score(game, state, a) for a in legal]
            return int(legal[int(np.argmax(scores))])
        me = game.current_player(state)
        if state.phase == Phase.PICK:
            return self._pick(game, state, me, legal)
        return self._card(game, state, me, legal)

    def _move_score(self, game: Undertow, state: UndertowState, action: Action) -> float:
        side = state.to_move
        enemy = 1 - side
        token, region = divmod(action - MOVE_BASE, NUM_REGIONS)
        here = state.pos[side][token]
        piece = ROSTERS[side][state.ident[side][token]]
        enemies = game.tokens_at(state, enemy, region)
        threats = sum(len(game.tokens_at(state, enemy, n)) for n in ADJACENT[region] if n != here)
        if enemies:
            est = [estimate(game, state, side, enemy, t) for t in enemies]
            odds = piece.strength + 1.5 - max(s for s, _ in est)
        if side == SMUGGLERS:
            escorts = [
                t
                for t in game.tokens_at(state, side, region)
                if ROSTERS[side][state.ident[side][t]].strength >= 2
                and not ROSTERS[side][state.ident[side][t]].cargo
            ]
            if piece.cargo:
                score = 4.0 * piece.cargo * (DEPTH[region] - DEPTH[here])
                if region == PORT and not enemies:
                    score += 100.0
                if enemies:
                    score -= 15.0 * piece.cargo + 5.0 * len(enemies)
                # Exposed cargo is the commonest way to lose: only advance next to the Watch
                # with an escort that can defend for it.
                return score - (1.0 if escorts else 6.0) * piece.cargo * threats
            if piece.strength == 0:  # the Decoy drifts forward, never attacks
                return 0.5 * (DEPTH[region] - DEPTH[here]) - (20.0 if enemies else 0.0)
            if enemies:
                return 6.0 * odds
            # Fighters lead: forward, towards the Watch, and onto cargo that needs cover.
            cargo_here = any(
                ROSTERS[side][state.ident[side][t]].cargo
                for t in game.tokens_at(state, side, region)
            )
            return (
                2.0 * (DEPTH[region] - DEPTH[here])
                + (1.0 if threats else 0.0)
                + (3.0 if cargo_here and not escorts else 0.0)
            )
        # The Watch.
        if enemies:
            cargo = sum(c for _, c in (estimate(game, state, side, enemy, t) for t in enemies))
            return 6.0 * odds + 8.0 * cargo
        targets = [r for r in state.pos[enemy] if r >= 0]
        before = min((DIST[here][r] for r in targets), default=0)
        after = min((DIST[region][r] for r in targets), default=0)
        score = 3.0 * (before - after)
        guards = sum(1 for r in state.pos[side] if r in GUARD)
        if here in GUARD and region not in GUARD and guards <= 1:
            score -= 10.0
        return score

    def _pick(self, game: Undertow, state: UndertowState, me: int, legal: list[int]) -> int:
        tokens = [a - PICK_BASE for a in legal]
        idents = [state.ident[me][t] for t in tokens]
        if me == SMUGGLERS and DECOY in idents:
            return PICK_BASE + tokens[idents.index(DECOY)]
        strengths = [ROSTERS[me][i].strength - 0.1 * ROSTERS[me][i].cargo for i in idents]
        return PICK_BASE + tokens[int(np.argmax(strengths))]

    def _card(self, game: Undertow, state: UndertowState, me: int, legal: list[int]) -> int:
        fight = state.fight
        assert fight is not None
        attacker = me == fight.side
        mine = fight.token if attacker else fight.defender
        theirs = fight.defender if attacker else fight.token
        s = ROSTERS[me][state.ident[me][mine]].strength
        o = ROSTERS[1 - me][state.ident[1 - me][theirs]].strength  # both revealed in a fight
        hand = [a - CARD_BASE for a in legal]
        opp_hand = [c for c in range(NUM_CARDS) if state.hands[1 - me] >> c & 1]
        opp_best = max(CARD_VALUES[c] for c in opp_hand)
        for c in sorted((c for c in hand if c < FEINT), key=lambda c: CARD_VALUES[c]):
            if s + CARD_VALUES[c] > o + opp_best:
                return CARD_BASE + c
        if LANTERN in hand and s > o:
            return CARD_BASE + LANTERN
        best = max((c for c in hand if c < FEINT), key=lambda c: CARD_VALUES[c], default=None)
        if FEINT in hand and (best is None or s + CARD_VALUES[best] <= o):
            return CARD_BASE + FEINT
        if best is not None:
            return CARD_BASE + best
        return CARD_BASE + hand[0]
