import numpy as np

import toy_game  # noqa: F401  (import registers "toy_showdown")
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.core.runner import run_episode


def test_two_random_agents_finish_and_are_zero_sum() -> None:
    game = make("toy_showdown")
    agents = [RandomAgent("r0"), RandomAgent("r1")]
    rng = np.random.default_rng(0)
    for _ in range(500):
        result = run_episode(game, agents, rng)
        assert len(result.returns) == 2
        assert abs(sum(result.returns)) < 1e-9
        assert result.num_steps >= 2  # at least the two deals happen


def test_wrong_agent_count_raises() -> None:
    game = make("toy_showdown")
    rng = np.random.default_rng(0)
    try:
        run_episode(game, [RandomAgent()], rng)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for wrong agent count")
