import numpy as np
import pytest

import toy_game  # noqa: F401  (import registers "toy_showdown")
from conformance import check_conformance
from iirl.core.registry import make, registered_games


@pytest.mark.parametrize("name", registered_games())
def test_registered_game_conforms(name: str) -> None:
    game = make(name)
    check_conformance(game, np.random.default_rng(1234), episodes=300)


def test_at_least_one_game_registered() -> None:
    assert "toy_showdown" in registered_games()
