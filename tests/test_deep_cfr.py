"""Deep CFR: regret matching, the reservoir, and convergence on Kuhn poker."""

import numpy as np
import pytest

from iirl.agents.cfr.deep_cfr import DeepCFR, DeepCFRConfig, Reservoir, regret_matching


def test_regret_matching_on_predicted_advantages() -> None:
    np.testing.assert_allclose(
        regret_matching(np.array([2.0, -1.0, 1.0]), [0, 1, 2]), [2 / 3, 0, 1 / 3]
    )
    # No positive advantage: the best legal action, not uniform.
    np.testing.assert_allclose(regret_matching(np.array([-3.0, -1.0, 5.0]), [0, 1]), [0, 1, 0])


def test_reservoir_keeps_a_uniform_sample_at_fixed_size() -> None:
    mem = Reservoir(100, 1, 1, np.random.default_rng(0))
    for i in range(10_000):
        mem.add(np.array([i], np.float32), np.zeros(1), np.ones(1), 1)
    assert len(mem) == 100 and mem.seen == 10_000
    assert 3000 < mem.x[:, 0].mean() < 7000  # spread over everything added, not the first 100


def test_deep_cfr_approaches_kuhn_equilibrium() -> None:
    pytest.importorskip("pyspiel")
    import iirl.games  # noqa: F401
    from iirl.core.registry import make
    from iirl.eval.exact_br import exploitability, uniform_policy

    game = make("kuhn_poker")
    solver = DeepCFR(
        game,
        DeepCFRConfig(
            iterations=0,
            traversals=100,
            hidden=(32, 32),
            advantage_steps=150,
            strategy_steps=800,
            batch_size=256,
        ),
    )
    solver.iterate(30)
    found = exploitability(game, solver.average_policy())["exploitability"]
    uniform = exploitability(game, uniform_policy)["exploitability"]
    assert found < 0.25 * uniform
    # The state round-trips (a resumed run continues from the same point).
    clone = DeepCFR(game, solver.config)
    clone.load_state_dict(solver.state_dict())
    assert clone.t == 30 and len(clone.strat_mem) == len(solver.strat_mem)
