"""Max-n PUCT with exact leaf values finds near-equilibrium moves, for 2 and 3 players."""

import numpy as np
import pytest

from iirl.agents.alphazero.multiplayer import MultiPUCTAgent
from iirl.eval.perfect_info import solve
from iirl.games.spendthrift import Spendthrift


class Exact:
    """Uniform priors, exact solution values at the leaves."""

    def __init__(self, game, sol):  # noqa: D107
        self.game, self.sol = game, sol

    def evaluate_batch(self, game, states):  # noqa: D102
        priors = []
        for s in states:
            p = np.zeros(game.num_distinct_actions())
            legal = game.legal_actions(s)
            p[legal] = 1 / len(legal)
            priors.append(p)
        return priors, np.stack([self.sol.value(s) for s in states])


@pytest.mark.parametrize("players", [2, 3])
def test_exact_leaves_find_good_moves(players):
    game = Spendthrift(players, lots=3, ends=1, coins=4)
    sol = solve(game)
    agent = MultiPUCTAgent(Exact(game, sol), simulations=200)
    rng = np.random.default_rng(0)
    nodes = [s for s in sol.choice if len(game.legal_actions(s)) > 1]
    picks = [nodes[i] for i in rng.choice(len(nodes), size=40, replace=False)]
    lost, uniform = [], []
    for s in picks:
        p = s.to_move
        q = {a: sol.value(game.apply_action(s, a))[p] for a in game.legal_actions(s)}
        best = max(q.values())
        lost.append(best - q[agent.step(game, s, rng)])
        uniform.append(best - np.mean(list(q.values())))
    assert np.mean(lost) < 0.02
    assert np.mean(lost) < 0.25 * np.mean(uniform)
