import pytest

import toy_game  # noqa: F401  (import registers "toy_showdown")
from iirl.core.game import Game
from iirl.core.registry import make, register, registered_games


def test_make_builds_registered_game() -> None:
    assert isinstance(make("toy_showdown"), Game)
    assert "toy_showdown" in registered_games()


def test_make_unknown_raises() -> None:
    with pytest.raises(KeyError):
        make("does_not_exist")


def test_double_registration_raises() -> None:
    @register("dup_test_game")
    def _factory() -> Game:  # pragma: no cover - never called
        raise NotImplementedError

    with pytest.raises(ValueError):

        @register("dup_test_game")
        def _factory2() -> Game:  # pragma: no cover - never called
            raise NotImplementedError
