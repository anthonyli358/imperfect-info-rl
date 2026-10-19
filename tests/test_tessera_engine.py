from dataclasses import replace

import numpy as np
import pytest

import iirl.games  # noqa: F401
from iirl.core.registry import make, make_heuristic
from iirl.games.tessera.game import decode, encode, row_accepts, tile_board
from iirl.games.tessera.state import Board, Phase, TesseraState
from iirl.games.tessera.tiles import (
    CENTER,
    FLOOR,
    NUM_COLORS,
    TILES_PER_COLOR,
    WALL_SIZE,
    end_bonus,
    floor_penalty,
    placement_points,
    wall_color,
    wall_column,
)


def wall_with(*cells: tuple[int, int]) -> tuple[bool, ...]:
    wall = [False] * (WALL_SIZE * WALL_SIZE)
    for r, c in cells:
        wall[r * WALL_SIZE + c] = True
    return tuple(wall)


def test_wall_layout_is_a_latin_square() -> None:
    for r in range(WALL_SIZE):
        assert sorted(wall_color(r, c) for c in range(WALL_SIZE)) == list(range(NUM_COLORS))
        for k in range(NUM_COLORS):
            assert wall_color(r, wall_column(r, k)) == k
    for c in range(WALL_SIZE):
        assert sorted(wall_color(r, c) for r in range(WALL_SIZE)) == list(range(NUM_COLORS))


@pytest.mark.parametrize(
    ("cells", "placed", "points"),
    [
        ((), (2, 2), 1),  # alone
        (((2, 1),), (2, 2), 2),  # one neighbour to the left: a run of 2
        (((2, 0), (2, 1), (2, 3)), (2, 2), 4),  # joins a row of 4
        (((1, 2), (3, 2)), (2, 2), 3),  # a column of 3
        (((2, 1), (1, 2), (0, 2)), (2, 2), 5),  # row of 2 + column of 3
        (((1, 1),), (2, 2), 1),  # diagonal neighbours do not count
    ],
)
def test_placement_points(cells, placed, points) -> None:  # noqa: ANN001
    assert placement_points(wall_with(*cells, placed), *placed) == points


def test_floor_penalty_and_end_bonus() -> None:
    assert [floor_penalty(n) for n in (0, 1, 2, 3, 7, 9)] == [0, -1, -2, -4, -14, -14]
    full_row = wall_with(*((0, c) for c in range(5)))
    full_column = wall_with(*((r, 0) for r in range(5)))
    one_color = wall_with(*((r, wall_column(r, 3)) for r in range(5)))
    assert end_bonus(full_row) == 2
    assert end_bonus(full_column) == 7
    assert end_bonus(one_color) == 10


def test_row_rules() -> None:
    board = Board(lines=((-1, 0), (2, 1), (-1, 0), (-1, 0), (-1, 0)), wall=wall_with((3, 4)))
    assert row_accepts(board, 0, 0)
    assert row_accepts(board, 1, 2) and not row_accepts(board, 1, 0)  # same colour only
    full = replace(board, lines=((1, 1), *board.lines[1:]))
    assert not row_accepts(full, 0, 1)  # a complete row takes nothing more
    assert not row_accepts(board, 3, wall_color(3, 4))  # colour already on that wall row


def test_tiling_moves_complete_rows_and_discards_the_rest() -> None:
    board = Board(
        lines=((0, 1), (1, 1), (2, 3), (-1, 0), (-1, 0)),
        wall=wall_with((0, 1)),
        floor=2,
        score=5,
    )
    tiled, lid = tile_board(board)
    assert tiled.wall[0 * 5 + wall_column(0, 0)]  # row 1 complete: placed
    assert tiled.lines[1] == (1, 1)  # row 2 incomplete: stays
    assert tiled.wall[2 * 5 + wall_column(2, 2)] and tiled.lines[2] == (-1, 0)
    assert lid[2] == 2 and sum(lid) == 2  # the other two tiles of row 3
    # Row 1's tile joins its (0, 1) neighbour: 2 points; row 3's is alone: 1; floor -2.
    assert tiled.score == 5 + 2 + 1 - 2 and tiled.floor == 0


def test_taking_from_a_display_and_the_centre() -> None:
    game = make("tessera")
    state = TesseraState(
        factories=((2, 1, 1, 0, 0), *((0,) * 5,) * 4),
        phase=Phase.PLAY,
        bag=(10,) * 5,
    )
    s = game.apply_action(state, encode(0, 0, 1))  # both colour-0 tiles to row 2
    assert s.factories[0] == (0,) * 5 and s.center == (0, 1, 1, 0, 0)
    assert s.boards[0].lines[1] == (0, 2) and s.to_move == 1
    s = game.apply_action(s, encode(CENTER, 1, FLOOR))  # first from the centre: the marker
    assert not s.marker_in_center and s.starter == 1 and s.boards[1].floor == 2
    assert s.lid[1] == 1


def _total_tiles(state: TesseraState) -> int:
    on_boards = sum(sum(b.wall) + sum(count for _, count in b.lines) for b in state.boards)
    return (
        sum(state.bag)
        + sum(state.lid)
        + sum(sum(f) for f in state.factories)
        + sum(state.center)
        + on_boards
    )


def test_tiles_are_conserved_and_games_end() -> None:
    game = make("tessera")
    rng = np.random.default_rng(0)
    for _ in range(30):
        state = game.new_initial_state()
        while not game.is_terminal(state):
            assert _total_tiles(state) == NUM_COLORS * TILES_PER_COLOR
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            else:
                legal = game.legal_actions(state)
                assert legal and all(decode(encode(*decode(a))) == decode(a) for a in legal)
                action = legal[rng.integers(len(legal))]
            state = game.apply_action(state, action)
        assert _total_tiles(state) == NUM_COLORS * TILES_PER_COLOR


def test_heuristic_beats_random() -> None:
    game = make("tessera")
    heuristic = make_heuristic("tessera")
    rng = np.random.default_rng(1)
    margins = []
    for g in range(20):
        state, seat = game.new_initial_state(), g % 2
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            elif game.current_player(state) == seat:
                action = heuristic.step(game, state, rng)
            else:
                legal = game.legal_actions(state)
                action = legal[rng.integers(len(legal))]
            state = game.apply_action(state, action)
        margins.append(game.returns(state)[seat])
    assert np.mean(np.asarray(margins) > 0) >= 0.9
