import numpy as np

from iirl.agents.random_agent import RandomAgent
from iirl.eval.psro import MixtureAgent, solve_symmetric_meta_game


def test_meta_solver_finds_rock_paper_scissors_equilibrium() -> None:
    # Scores: win 1, lose 0, tie 0.5.
    rps = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
    mixture, exploit = solve_symmetric_meta_game(rps)
    np.testing.assert_allclose(mixture, [1 / 3] * 3, atol=0.02)
    assert abs(exploit - 0.5) < 0.02


def test_meta_solver_drops_a_dominated_strategy() -> None:
    # Member 2 loses to both others; members 0 and 1 are even. No weight should stay on 2.
    scores = np.array([[0.5, 0.5, 0.9], [0.5, 0.5, 0.8], [0.1, 0.2, 0.5]])
    mixture, _ = solve_symmetric_meta_game(scores)
    assert mixture[2] < 0.01


def test_mixture_agent_uses_one_member_per_game() -> None:
    a, b = RandomAgent(), RandomAgent()
    mix = MixtureAgent([a, b], [0.0, 1.0], seed=0)
    mix.reset(None, 0)  # type: ignore[arg-type]
    assert mix.current is b
