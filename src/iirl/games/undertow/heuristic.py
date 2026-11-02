"""A rule-based Undertow baseline for either side.

Game-specific by design: an Elo anchor and the balance check. It uses only what its side knows:
an enemy token is valued by the identities it could still be (its possible set), each equally
likely.

- **Setup.** Fighters in front, the Contraband among the pieces at home (a random one of a few
  sensible layouts, so it is not predictable).
- **Smuggler moves.** The Contraband enters the Free Port whenever it can (an instant win),
  otherwise edges forward where no Watch piece can reach it next turn. Fighters attack where they
  expect to win and otherwise advance, clearing the way.
- **Watch moves.** Three in the Cove wins, so raiders head upstream; hunters attack where the
  Contraband is likely and the odds are good. Guarding the Port is pointless: the Contraband wins
  by entering it.
- **Cards.** The smallest number that beats the opponent's best remaining number; failing that a
  trick that helps (retreat when losing, Fog or Lantern when strength alone wins, Scuttle to
  trade a weak piece for a strong one); else the highest number.
"""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.registry import register_heuristic
from iirl.core.types import Action
from iirl.games.undertow.board import (
    CARDS,
    CONTRABAND,
    COVE,
    DEPTH,
    FORWARD,
    HAND_SIZE,
    NUM_REGIONS,
    PORT,
    RAID,
    ROSTERS,
    SIDEWAYS,
    SMUGGLERS,
    WATCH,
)
from iirl.games.undertow.game import (
    CARD_BASE,
    MOVE_BASE,
    SETUP_BASE,
    TOKENS,
    Phase,
    Undertow,
    UndertowState,
)

# Placement preferences per slot (home ×4, then the five front regions), strongest in front.
_LAYOUTS = (
    (  # Smugglers: identities by slot
        (7, 8, 4, 5, 0, 1, 2, 3, 6),
        (7, 4, 5, 6, 0, 1, 8, 2, 3),
        (7, 8, 5, 3, 0, 1, 4, 2, 6),
    ),
    (  # Watch
        (5, 6, 8, 2, 0, 1, 3, 4, 7),
        (5, 6, 8, 3, 0, 1, 2, 4, 7),
        (6, 8, 5, 2, 1, 0, 3, 7, 4),
    ),
)


def expected(game: Undertow, state: UndertowState, viewer: int, side: int, token: int) -> tuple:
    """``(strength, chance it is the Contraband)`` of a token as ``viewer`` sees it."""
    m = game.possible(state, viewer, side, token)
    ids = [i for i in range(TOKENS) if m >> i & 1] or list(range(TOKENS))
    strength = float(np.mean([ROSTERS[side][i].strength for i in ids]))
    contraband = (CONTRABAND in ids) / len(ids) if side == SMUGGLERS else 0.0
    return strength, contraband


@register_heuristic("undertow")
class UndertowHeuristic(Agent):
    """Rule-based play for whichever side it is seated as; see the module docstring."""

    def __init__(self, name: str = "heuristic") -> None:
        """Name the agent."""
        self.name = name
        self.layout: tuple[int, ...] | None = None

    def reset(self, game: Game, player: int) -> None:
        """Forget the layout chosen for the last game."""
        self.layout = None

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Place, move or play a card by the rules above."""
        assert isinstance(game, Undertow) and isinstance(state, UndertowState)
        legal = game.legal_actions(state)
        me = game.current_player(state)
        if state.phase == Phase.SETUP:
            slot = state.setup % TOKENS
            if slot == 0 or self.layout is None:
                self.layout = _LAYOUTS[me][int(rng.integers(len(_LAYOUTS[me])))]
            want = SETUP_BASE + self.layout[slot]
            return want if want in legal else legal[0]
        if state.phase == Phase.MOVE:
            scores = [self._move_score(game, state, a) + 1e-3 * rng.random() for a in legal]
            return int(legal[int(np.argmax(scores))])
        return self._card(game, state, me, legal)

    def _move_score(self, game: Undertow, state: UndertowState, action: Action) -> float:
        side = state.to_move
        enemy = 1 - side
        token, region = divmod(action - MOVE_BASE, NUM_REGIONS)
        here = state.pos[side][token]
        identity = state.ident[side][token]
        strength = ROSTERS[side][identity].strength
        enemies = game.tokens_at(state, enemy, region)
        progress = abs(DEPTH[region] - DEPTH[here])
        # Who could hit ``region`` next turn: enemy pieces with it among their forward moves.
        reach = sum(
            1
            for t, r in enumerate(state.pos[enemy])
            if r >= 0 and (region in FORWARD[enemy][r] or region in SIDEWAYS[r])
        )
        odds = 0.0
        hunt = 0.0
        if enemies:
            est = [expected(game, state, side, enemy, t) for t in enemies]
            odds = strength + 3.0 - (float(np.mean([s for s, _ in est])) + 3.0)
            hunt = float(np.mean([c for _, c in est]))
        if side == SMUGGLERS:
            if identity == CONTRABAND:
                if region == PORT:
                    return 1000.0
                return 5.0 * progress - (40.0 if enemies else 0.0) - 8.0 * reach
            if enemies:
                return 4.0 * odds
            return 2.0 * progress - 0.5 * reach * (strength < 3)
        # The Watch.
        if region == COVE and len(game.tokens_at(state, WATCH, COVE)) + 1 >= RAID and not enemies:
            return 1000.0
        if enemies:
            return 4.0 * odds + 20.0 * hunt
        return 1.5 * progress + (2.0 if region == COVE else 0.0)

    def _card(self, game: Undertow, state: UndertowState, me: int, legal: list[int]) -> int:
        fight = state.fight
        assert fight is not None
        mine = fight.token if me == fight.side else fight.defender
        theirs = fight.defender if me == fight.side else fight.token
        s = ROSTERS[me][state.ident[me][mine]].strength
        o = ROSTERS[1 - me][state.ident[1 - me][theirs]].strength  # both revealed in a fight
        hand = [a - CARD_BASE for a in legal]
        opp_hand = [c for c in range(HAND_SIZE) if not state.used[1 - me] >> c & 1]
        opp_best = max((CARDS[1 - me][c].value for c in opp_hand), default=0)
        numbers = sorted(
            (c for c in hand if not CARDS[me][c].trick), key=lambda c: CARDS[me][c].value
        )
        tricks = {CARDS[me][c].trick: c for c in hand if CARDS[me][c].trick}
        for c in numbers:
            if s + CARDS[me][c].value > o + opp_best:
                return CARD_BASE + c
        precious = me == SMUGGLERS and state.ident[me][mine] == CONTRABAND
        best = CARDS[me][numbers[-1]].value if numbers else 0
        losing = s + best <= o
        if losing and "feint" in tricks:
            return CARD_BASE + tricks["feint"]
        if losing and "withdraw" in tricks:
            return CARD_BASE + tricks["withdraw"]
        if s > o and "fog" in tricks:
            return CARD_BASE + tricks["fog"]
        if s > o and "lantern" in tricks:
            return CARD_BASE + tricks["lantern"]
        if losing and not precious and o >= s + 2 and "scuttle" in tricks:
            return CARD_BASE + tricks["scuttle"]
        if numbers:
            return CARD_BASE + numbers[-1]
        return CARD_BASE + hand[0]
