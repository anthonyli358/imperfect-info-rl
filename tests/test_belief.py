from dataclasses import replace

import numpy as np

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.base import Agent
from iirl.agents.belief import BeliefConfig, ParticleBelief
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.core.runner import _sample_chance, run_episode
from iirl.games.ventures.cards import initial_deck_counts


def _uniform_model(game, states):
    return [np.ones(game.num_distinct_actions()) for _ in states]


def _play_tracked(game, belief, me, rng, check):
    """Play a random game, updating ``belief`` for seat ``me``; call ``check`` at each state."""
    state = game.new_initial_state()
    belief.reset(game, me)
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.apply_action(state, _sample_chance(game, state, rng))
            action = None
        else:
            action = int(rng.choice(game.legal_actions(state)))
            state = game.apply_action(state, action)
        belief.observe(state, action)
        if not game.is_terminal(state) and belief.worlds:
            check(state)


def test_particles_stay_consistent_with_the_view() -> None:
    g = make("ventures")
    rng = np.random.default_rng(0)
    belief = ParticleBelief(_uniform_model, BeliefConfig(particles=16), seed=1)

    def check(state):
        view = g.information_state_string(state, belief.player)
        for i in np.flatnonzero(belief.weights > 0):
            assert g.information_state_string(belief.worlds[i], belief.player) == view
        assert np.isclose(belief.weights.sum(), 1.0)
        w = belief.sample(state, rng)
        assert w.hands[belief.player] == state.hands[belief.player]

    for me in (0, 1):
        _play_tracked(g, belief, me, rng, check)


def test_repair_world_keeps_still_possible_cards() -> None:
    g = make("ventures")
    rng = np.random.default_rng(3)
    s = g.new_initial_state()
    while g.is_chance_node(s):
        s = g.apply_action(s, _sample_chance(g, s, rng))
    me, opp = 0, 1
    for _ in range(20):
        guess = g.resample_from_infostate(s, me, rng)
        repaired = g.repair_world(s, me, guess, rng)
        assert repaired.hands[opp] == guess.hands[opp]  # nothing contradicts the guess yet
    # Put a single-copy card the guess gave the opponent into my hand: repair must drop just it.
    full = initial_deck_counts()
    while True:
        guess = g.resample_from_infostate(s, me, rng)
        single = [c for c in np.flatnonzero(guess.hands[opp]) if full[c] == 1]
        if single:
            break
    card = int(single[0])
    mine = list(s.hands[me])
    swap = int(np.flatnonzero(mine)[0])
    mine[swap] -= 1
    mine[card] += 1
    contradicted = replace(s, hands=(tuple(mine), s.hands[opp]))
    repaired = g.repair_world(contradicted, me, guess, rng)
    kept = np.minimum(repaired.hands[opp], guess.hands[opp]).sum()
    assert kept == sum(guess.hands[opp]) - guess.hands[opp][card]


class _Recorder(Agent):
    name = "recorder"

    def __init__(self):
        self.seat, self.seen = None, []

    def reset(self, game, player):
        self.seat, self.seen = player, []

    def observe(self, game, state, action):
        self.seen.append(action)

    def step(self, game, state, rng):
        return int(rng.choice(game.legal_actions(state)))


def test_runner_calls_reset_and_observe() -> None:
    g = make("ventures")
    rec = _Recorder()
    result = run_episode(g, [rec, RandomAgent()], np.random.default_rng(0))
    assert rec.seat == 0
    assert len(rec.seen) == result.num_steps
    assert None in rec.seen  # chance events are reported without their outcome
