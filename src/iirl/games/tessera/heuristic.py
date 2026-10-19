"""A rule-based Tessera baseline.

Game-specific by design: an Elo anchor, a league opponent, and an engine sanity check. It scores
every legal move by what it does this round and plays the best:

1. **Complete rows.** A move that fills a pattern row earns the points its wall tile will score
   (counting the tiles already beside it), plus a little for the end-of-game row and column bonuses
   it moves toward.
2. **Make progress without waste.** Tiles placed on a row are worth a fraction of a point each;
   tiles that overflow to the floor cost exactly the floor penalty of the slots they fill.
3. **Avoid starting what cannot be finished.** Late in the game, a long row that cannot be
   completed this round is discounted.
4. **The first-player marker** costs its floor slot and gains the next round's first pick.
"""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.registry import register_heuristic
from iirl.core.types import Action
from iirl.games.tessera.game import decode
from iirl.games.tessera.state import TesseraState
from iirl.games.tessera.tiles import (
    CENTER,
    FLOOR,
    FLOOR_PENALTIES,
    FLOOR_SLOTS,
    NUM_COLORS,
    WALL_SIZE,
    placement_points,
    wall_column,
)


def _floor_cost(floor: int, tiles: int) -> float:
    """Penalty for ``tiles`` more tiles on a floor that already holds ``floor``."""
    return float(-sum(FLOOR_PENALTIES[min(floor, FLOOR_SLOTS) : min(floor + tiles, FLOOR_SLOTS)]))


@register_heuristic("tessera")
class TesseraHeuristic(Agent):
    """Greedy one-move evaluation; see the module docstring."""

    def __init__(self, name: str = "heuristic", progress: float = 0.35) -> None:
        """``progress`` is the value of each tile placed on an unfinished row."""
        self.name, self.progress = name, progress

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return the legal move with the best one-move score (ties: the lowest id)."""
        assert isinstance(state, TesseraState)
        legal = game.legal_actions(state)
        scores = [self._score(state, a) for a in legal]
        return int(legal[int(np.argmax(scores))])

    def _score(self, state: TesseraState, action: Action) -> float:
        source, color, destination = decode(action)
        board = state.boards[state.to_move]
        tiles = state.center if source == CENTER else state.factories[source]
        n = tiles[color]
        floor = board.floor
        value = 0.0
        if source == CENTER and state.marker_in_center:
            value -= _floor_cost(floor, 1) - 0.5  # a floor slot, but the first pick next round
            floor += 1
        if destination == FLOOR:
            return value - _floor_cost(floor, n)
        held, count = board.lines[destination]
        size = destination + 1
        put = min(n, size - count)
        overflow = n - put
        value -= _floor_cost(floor, overflow)
        if count + put == size:
            column = wall_column(destination, color)
            wall = list(board.wall)
            wall[destination * WALL_SIZE + column] = True
            value += placement_points(tuple(wall), destination, column)
            row_done = sum(wall[destination * WALL_SIZE : (destination + 1) * WALL_SIZE])
            col_done = sum(wall[r * WALL_SIZE + column] for r in range(WALL_SIZE))
            value += 0.3 * row_done + 0.6 * col_done  # toward the row and column bonuses
        else:
            value += self.progress * put
            if state.round >= 4 and size - count - put > 2:
                value -= 1.0  # a long row unlikely to be finished before the end
        return value + 0.01 * (NUM_COLORS - color)  # a deterministic, harmless tie-break
