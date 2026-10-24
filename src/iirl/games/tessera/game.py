"""The Tessera engine (see :mod:`iirl.games.tessera.tiles` for the rules data and credit note).

A round: chance fills the five displays with four tiles each from the bag (one draw per tile, so
every draw is a first-class chance node). Players then alternate: take every tile of one colour
from one display (its other tiles go to the centre) or from the centre (the first to do so also
takes the first-player marker onto their floor), and put them on one pattern row (the excess falls
to the floor) or straight on the floor. When the displays and centre are empty, every complete row
moves one tile to the wall and scores; floor tiles cost points; the round's marker holder starts
the next. The game ends after the round in which someone completes a wall row.

Everything is visible except the order of the bag, so ``information_state_*`` is the full
observation and ``resample_from_infostate`` returns the state unchanged.

Action ids: ``(source * NUM_COLORS + colour) * NUM_DESTINATIONS + destination``, with sources
``0..4`` the displays and ``5`` the centre, destinations ``0..4`` the pattern rows and ``5`` the
floor (180 ids). Chance actions are colour ids.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from iirl.core.game import Game, State
from iirl.core.registry import register
from iirl.core.types import Action, ChanceOutcome, GameType, Information, Player, PlayerId, Utility
from iirl.games.tessera.state import NO_TILES, Board, Counts, Phase, TesseraState
from iirl.games.tessera.tiles import (
    CENTER,
    COLOR_HEX,
    COLOR_NAMES,
    COLOR_SYMBOLS,
    FLOOR,
    FLOOR_PENALTIES,
    FLOOR_SLOTS,
    MAX_ROUNDS,
    NUM_COLORS,
    NUM_DESTINATIONS,
    NUM_FACTORIES,
    NUM_SOURCES,
    TILES_PER_FACTORY,
    WALL_SIZE,
    end_bonus,
    floor_penalty,
    placement_points,
    wall_column,
)

NUM_ACTIONS = NUM_SOURCES * NUM_COLORS * NUM_DESTINATIONS
TILES_PER_ROUND = NUM_FACTORIES * TILES_PER_FACTORY


def encode(source: int, color: int, destination: int) -> Action:
    """The action id for taking ``color`` from ``source`` and placing it at ``destination``."""
    return (source * NUM_COLORS + color) * NUM_DESTINATIONS + destination


def decode(action: Action) -> tuple[int, int, int]:
    """``(source, colour, destination)`` of an action id."""
    rest, destination = divmod(action, NUM_DESTINATIONS)
    source, color = divmod(rest, NUM_COLORS)
    return source, color, destination


def _add(a: Counts, b: Counts) -> Counts:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _bump(counts: Counts, idx: int, delta: int) -> Counts:
    lst = list(counts)
    lst[idx] += delta
    return tuple(lst)


def _pair(pair: tuple, p: int, value) -> tuple:  # noqa: ANN001 - per-player pair update
    return (value, pair[1]) if p == 0 else (pair[0], value)


def row_accepts(board: Board, row: int, color: int) -> bool:
    """Whether pattern row ``row`` can take tiles of ``color``."""
    held, count = board.lines[row]
    if board.wall[row * WALL_SIZE + wall_column(row, color)]:
        return False
    return count < row + 1 and held in (-1, color)


def tile_board(board: Board) -> tuple[Board, Counts]:
    """The wall-tiling step for one board: returns the new board and the tiles sent to the lid."""
    wall, lines, score = list(board.wall), list(board.lines), board.score
    lid = NO_TILES
    for row, (color, count) in enumerate(board.lines):
        if count == row + 1:
            column = wall_column(row, color)
            wall[row * WALL_SIZE + column] = True
            score += placement_points(tuple(wall), row, column)
            lid = _bump(lid, color, row)  # the rest of a complete row is discarded
            lines[row] = (-1, 0)
    score = max(0, score + floor_penalty(board.floor))
    return Board(tuple(lines), tuple(wall), 0, score), lid


def projected_score(board: Board) -> int:
    """The score if the round ended now (complete rows tiled, floor penalties applied)."""
    return tile_board(board)[0].score


class Tessera(Game):
    """Two-player, zero-sum Tessera."""

    _TYPE = GameType(
        short_name="tessera",
        long_name="Tessera",
        num_players=2,
        information=Information.PERFECT,
        utility=Utility.ZERO_SUM,
        provides_information_state_tensor=True,
        provides_observation_tensor=False,
    )

    @property
    def game_type(self) -> GameType:
        """Static metadata."""
        return self._TYPE

    def new_initial_state(self) -> TesseraState:
        """Return the first round's filling step (a chance node)."""
        return TesseraState()

    def num_distinct_actions(self) -> int:
        """Return the flat action-space size."""
        return NUM_ACTIONS

    def current_player(self, state: State) -> Player:
        """Return the player to act, or a chance/terminal sentinel."""
        assert isinstance(state, TesseraState)
        if state.terminal:
            return PlayerId.TERMINAL
        if state.phase == Phase.FILL:
            return PlayerId.CHANCE
        return state.to_move

    def is_chance_node(self, state: State) -> bool:
        """Filling the displays is a chance node, one tile at a time."""
        assert isinstance(state, TesseraState)
        return not state.terminal and state.phase == Phase.FILL

    def is_terminal(self, state: State) -> bool:
        """Whether the game is over."""
        assert isinstance(state, TesseraState)
        return state.terminal

    def chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """The colour of the next tile drawn, in proportion to the bag."""
        assert isinstance(state, TesseraState)
        total = sum(state.bag)
        return [(c, n / total) for c, n in enumerate(state.bag) if n]

    # --- Moves ----------------------------------------------------------------------------------
    def legal_actions(self, state: State) -> list[Action]:
        """Every (source, colour, destination) with tiles there and room for them."""
        assert isinstance(state, TesseraState)
        board = state.boards[state.to_move]
        actions = []
        for source in range(NUM_SOURCES):
            tiles = state.center if source == CENTER else state.factories[source]
            for color in range(NUM_COLORS):
                if not tiles[color]:
                    continue
                for row in range(WALL_SIZE):
                    if row_accepts(board, row, color):
                        actions.append(encode(source, color, row))
                actions.append(encode(source, color, FLOOR))
        return actions

    def apply_action(self, state: State, action: Action) -> TesseraState:
        """Return the next state (never mutates ``state``)."""
        assert isinstance(state, TesseraState)
        if state.phase == Phase.FILL:
            return self._draw(state, action)
        return self._take(state, action)

    def _draw(self, state: TesseraState, color: int) -> TesseraState:
        factory = state.filled // TILES_PER_FACTORY
        factories = list(state.factories)
        factories[factory] = _bump(factories[factory], color, 1)
        s = replace(
            state,
            bag=_bump(state.bag, color, -1),
            factories=tuple(factories),
            filled=state.filled + 1,
        )
        return self._continue_fill(s)

    def _continue_fill(self, state: TesseraState) -> TesseraState:
        """Refill the bag from the lid if needed; start play once the displays are full."""
        if not sum(state.bag) and sum(state.lid):
            state = replace(state, bag=state.lid, lid=NO_TILES)
        if state.filled >= TILES_PER_ROUND or not sum(state.bag):
            return replace(state, phase=Phase.PLAY, to_move=state.starter)
        return state

    def _take(self, state: TesseraState, action: Action) -> TesseraState:
        source, color, destination = decode(action)
        p = state.to_move
        board = state.boards[p]
        center, factories = state.center, state.factories
        marker, starter, floor = state.marker_in_center, state.starter, board.floor
        if source == CENTER:
            n = center[color]
            center = _bump(center, color, -n)
            if marker:  # the first to take from the centre takes the first-player marker
                marker, starter, floor = False, p, floor + 1
        else:
            n = factories[source][color]
            rest = _bump(factories[source], color, -n)
            center = _add(center, rest)
            factories = _pair_list(factories, source, NO_TILES)
        lines = list(board.lines)
        overflow = n
        if destination != FLOOR:
            count = lines[destination][1]
            put = min(n, destination + 1 - count)
            lines[destination] = (color, count + put)
            overflow = n - put
        lid = _bump(state.lid, color, overflow)  # fallen tiles end up in the lid
        board = replace(board, lines=tuple(lines), floor=floor + overflow)
        s = replace(
            state,
            center=center,
            factories=factories,
            marker_in_center=marker,
            starter=starter,
            lid=lid,
            boards=_pair(state.boards, p, board),
            to_move=1 - p,
        )
        if sum(s.center) or any(sum(f) for f in s.factories):
            return s
        return self._end_round(s)

    def _end_round(self, state: TesseraState) -> TesseraState:
        boards, lid = [], state.lid
        for board in state.boards:
            tiled, discarded = tile_board(board)
            boards.append(tiled)
            lid = _add(lid, discarded)
        done = any(
            all(b.wall[r * WALL_SIZE + c] for c in range(WALL_SIZE))
            for b in boards
            for r in range(WALL_SIZE)
        )
        rounds = state.round + 1
        if done or rounds >= MAX_ROUNDS:
            if done:
                boards = [replace(b, score=b.score + end_bonus(b.wall)) for b in boards]
            return replace(state, boards=tuple(boards), lid=lid, round=rounds, terminal=True)
        s = replace(
            state,
            boards=tuple(boards),
            lid=lid,
            round=rounds,
            phase=Phase.FILL,
            filled=0,
            marker_in_center=True,
        )
        return self._continue_fill(s)

    # --- Scores ---------------------------------------------------------------------------------
    def returns(self, state: State) -> list[float]:
        """Zero-sum payoffs: the score difference."""
        assert isinstance(state, TesseraState)
        diff = state.boards[0].score - state.boards[1].score
        return [float(diff), float(-diff)]

    def heuristic_value(self, state: State, player: Player) -> float:
        """The score difference if the round ended now."""
        assert isinstance(state, TesseraState)
        diff = projected_score(state.boards[0]) - projected_score(state.boards[1])
        return float(diff if player == 0 else -diff)

    # --- Observations ---------------------------------------------------------------------------
    def _board_features(self, board: Board) -> list[float]:
        feats: list[float] = [float(x) for x in board.wall]
        for row, (color, count) in enumerate(board.lines):
            onehot = [0.0] * NUM_COLORS
            if color >= 0:
                onehot[color] = 1.0
            feats += onehot + [count / (row + 1), count / WALL_SIZE]
        feats += [
            float(row_accepts(board, r, k)) for r in range(WALL_SIZE) for k in range(NUM_COLORS)
        ]
        feats += [
            board.floor / FLOOR_SLOTS,
            board.score / 100.0,
            (projected_score(board) - board.score) / 20.0,
        ]
        return feats

    def information_state_tensor(self, state: State, player: Player) -> np.ndarray:
        """Both boards (own first), the displays and centre, the supply, and the clock."""
        assert isinstance(state, TesseraState)
        me, opp = player, 1 - player
        available = _add(state.center, NO_TILES)
        for f in state.factories:
            available = _add(available, f)
        feats = self._board_features(state.boards[me]) + self._board_features(state.boards[opp])
        feats += [n / TILES_PER_FACTORY for f in state.factories for n in f]
        feats += [n / 10.0 for n in state.center]
        feats += [n / 20.0 for n in available]
        feats += [n / 20.0 for n in state.bag] + [n / 20.0 for n in state.lid]
        feats += [
            float(state.marker_in_center),
            float(state.starter == me),
            sum(1 for f in state.factories if sum(f)) / NUM_FACTORIES,
            state.round / MAX_ROUNDS,
        ]
        return np.asarray(feats, dtype=np.float32)

    def information_state_string(self, state: State, player: Player) -> str:
        """The whole state (nothing is hidden but the bag's order), from ``player``'s seat."""
        assert isinstance(state, TesseraState)
        return f"P{player}|{state}"

    def resample_from_infostate(
        self, state: State, player: Player, rng: np.random.Generator
    ) -> TesseraState:
        """Nothing is hidden: the state itself."""
        assert isinstance(state, TesseraState)
        return state

    # --- Display --------------------------------------------------------------------------------
    def action_to_string(self, state: State, action: Action) -> str:
        """Readable action label."""
        assert isinstance(state, TesseraState)
        if state.phase == Phase.FILL:
            return f"draw {COLOR_NAMES[action]}"
        source, color, destination = decode(action)
        where = "centre" if source == CENTER else f"display {source + 1}"
        to = "floor" if destination == FLOOR else f"row {destination + 1}"
        return f"take {COLOR_NAMES[color]} from {where} to {to}"

    def display_meta(self) -> dict:
        """Viewer metadata."""
        return {
            "color_names": list(COLOR_NAMES),
            "color_hex": list(COLOR_HEX),
            "color_symbols": list(COLOR_SYMBOLS),
            "wall_size": WALL_SIZE,
            "floor_penalties": list(FLOOR_PENALTIES),
            "actions": {
                "num_colors": NUM_COLORS,
                "num_destinations": NUM_DESTINATIONS,
                "center": CENTER,
                "floor": FLOOR,
            },
        }

    def to_display(self, state: State) -> dict:
        """A JSON-ready snapshot of the whole state."""
        assert isinstance(state, TesseraState)
        return {
            "factories": [list(f) for f in state.factories],
            "center": list(state.center),
            "marker_in_center": state.marker_in_center,
            "boards": [
                {
                    "lines": [list(line) for line in b.lines],
                    "wall": [list(b.wall[r * WALL_SIZE : (r + 1) * WALL_SIZE]) for r in range(5)],
                    "floor": b.floor,
                    "score": b.score,
                }
                for b in state.boards
            ],
            "bag": sum(state.bag),
            "lid": sum(state.lid),
            "round": state.round + 1,
            "to_move": int(self.current_player(state)),
            "terminal": state.terminal,
        }

    def player_view(self, state: State, player: Player) -> dict:
        """Nothing is hidden: the full display."""
        return self.to_display(state)


def _pair_list(items: tuple, index: int, value) -> tuple:  # noqa: ANN001
    lst = list(items)
    lst[index] = value
    return tuple(lst)


@register("tessera")
def _make_tessera() -> Tessera:
    """Factory registered under ``"tessera"``."""
    return Tessera()
