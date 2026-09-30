import numpy as np

import iirl.games  # noqa: F401  (registers "mini_ventures")
import toy_game  # noqa: F401  (registers "toy_showdown")
from iirl.agents.cfr import CFRSolver
from iirl.core.registry import make
from iirl.eval.exact_br import build_graph, exploitability, uniform_policy
from iirl.eval.exploitability import exploitability as legacy_exploitability


def _table_policy(strategy, actions):  # noqa: ANN001, ANN202
    def policy(game, states):  # noqa: ANN001, ANN202
        out = []
        for s in states:
            key = game.information_state_string(s, game.current_player(s))
            probs = np.zeros(game.num_distinct_actions())
            legal = game.legal_actions(s)
            if key in strategy:
                for a, p in zip(actions[key], strategy[key], strict=True):
                    probs[a] = p
            else:
                probs[legal] = 1.0 / len(legal)
            out.append(probs)
        return out

    return policy


def test_matches_the_tree_walking_calculator_on_the_toy_game() -> None:
    game = make("toy_showdown")
    solver = CFRSolver(game)
    solver.train(50)
    strategy, actions = solver.average_strategy()
    exact = exploitability(game, _table_policy(strategy, actions))["exploitability"]
    assert abs(exact - legacy_exploitability(game, strategy)) < 1e-9
    uniform = exploitability(game, uniform_policy)["exploitability"]
    assert abs(uniform - legacy_exploitability(game, {})) < 1e-9


def test_mini_ventures_graph_and_uniform_play_is_exploitable() -> None:
    game = make("mini_ventures")
    graph = build_graph(game)
    assert 1_000_000 < len(graph.states) < 5_000_000
    result = exploitability(game, uniform_policy, graph)
    assert result["exploitability"] > 1.0  # random play gives away points


def test_graph_cfr_converges_on_the_toy_game() -> None:
    from iirl.agents.cfr.graph_cfr import GraphCFR

    game = make("toy_showdown")
    cfr = GraphCFR(game)
    cfr.iterate(20)
    early = exploitability(game, cfr.average_policy())["exploitability"]
    cfr.iterate(1000)
    late = exploitability(game, cfr.average_policy())["exploitability"]
    assert late < early and late < 0.005
