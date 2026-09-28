"""Phase C: Bayesian determinization on Ventures.

Two questions, in order of cost:

1. **Belief quality** (cheap, in-process): does the particle filter concentrate on the opponent's
   real hand? At each of the tracked player's turns, compare the expected overlap between a
   sampled world's opponent hand and the true one — over the *hidden* part of the hand (cards not
   already known from discard pickups) — for the particle belief vs uniform resampling.
2. **Search strength** (duplicate matches): net-guided search whose worlds come from the belief
   vs the same search with uniform worlds, head to head and vs the raw policy.

Run with::

    uv run python -m experiments.belief_ventures
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import iirl.games  # noqa: F401  (registers "ventures")
from experiments.train_ventures_exit import load_net, ppo_agent
from iirl.agents.alphazero import AlphaZeroAgent, PUCTConfig
from iirl.agents.belief import BeliefConfig, ParticleBelief, net_opponent_model
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.nn.policy_value import NetEvaluator, SplitEvaluator


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    policy: str = "runs/nash_nash.pt"  # prior for search, opponent model, and the sparring agent
    value: str = "runs/value_ventures_priv.pt"
    quality_games: int = 40
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    simulations: int = 64
    determinizations: int = 4
    match_deals: int = 300
    matches: tuple[str, ...] = ("belief_vs_uniform", "belief_vs_policy", "uniform_vs_policy")


def _hidden_overlap(true_hand, world_hand, known) -> float:  # noqa: ANN001 - count tuples
    """Fraction of the opponent's hidden cards (hand minus known) that ``world_hand`` also holds."""
    t = np.asarray(true_hand) - np.asarray(known)
    w = np.asarray(world_hand) - np.asarray(known)
    return float(np.minimum(t, w).sum() / max(t.sum(), 1))


def belief_quality(config: Config) -> None:
    """Play policy-vs-policy games and score the belief against uniform resampling."""
    game = make("ventures")
    policy = PPOAgent.from_checkpoint(config.policy)
    model = net_opponent_model(NetEvaluator(policy.net))
    belief = ParticleBelief(model, config.belief, seed=0)
    rng = np.random.default_rng(0)
    by_turn: dict[int, list[tuple[float, float, float]]] = {}
    restarts, rejuvenated, elapsed, steps = 0, 0, 0.0, 0
    for g in range(config.quality_games):
        me = g % 2
        opp = 1 - me
        state = game.new_initial_state()
        belief.reset(game, me)
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                i = rng.choice(len(outcomes), p=[p for _, p in outcomes])
                state = game.apply_action(state, outcomes[i][0])
                action = None
            else:
                if game.current_player(state) == me and state.phase == 1:  # PLAY: score the belief
                    known = state.known[opp]
                    true = state.hands[opp]
                    bel = [
                        _hidden_overlap(true, belief.sample(state, rng).hands[opp], known)
                        for _ in range(32)
                    ]
                    uni = [
                        _hidden_overlap(
                            true, game.resample_from_infostate(state, me, rng).hands[opp], known
                        )
                        for _ in range(32)
                    ]
                    by_turn.setdefault(state.turn // 10, []).append(
                        (np.mean(bel), np.mean(uni), belief.effective_size())
                    )
                action = policy.step(game, state, rng)
                state = game.apply_action(state, action)
            t = time.perf_counter()
            belief.observe(state, action)
            elapsed += time.perf_counter() - t
            steps += 1
        restarts += belief.restarts
        rejuvenated += belief.rejuvenated
    print("turns    belief  uniform  ess   (hidden-hand overlap)")
    for k in sorted(by_turn):
        rows = np.array(by_turn[k])
        print(
            f"{10 * k:>3}-{10 * k + 9:<3}  {rows[:, 0].mean():.3f}   {rows[:, 1].mean():.3f}"
            f"   {rows[:, 2].mean():5.0f}  (n={len(rows)})"
        )
    rows = np.array([r for v in by_turn.values() for r in v])
    print(
        f"all      {rows[:, 0].mean():.3f}   {rows[:, 1].mean():.3f}   restarts {restarts}"
        f"   rejuvenated {rejuvenated / config.quality_games:.0f}/game"
        f"   {1000 * elapsed / steps:.2f} ms/transition",
        flush=True,
    )


def search_agent(
    policy: str,
    value: str,
    simulations: int,
    determinizations: int,
    belief: BeliefConfig | None = None,
) -> AlphaZeroAgent:
    """Build the greedy search agent; with ``belief``, worlds come from a particle filter."""
    evaluator = SplitEvaluator(load_net(policy, "linear"), load_net(value, "tanh"), True)
    search = PUCTConfig(simulations=simulations, determinizations=determinizations, leaf_batch=8)
    tracker = None
    if belief is not None:
        tracker = ParticleBelief(net_opponent_model(evaluator.priors), belief)
    return AlphaZeroAgent(evaluator, search, name="belief" if belief else "uniform", belief=tracker)


def search_strength(config: Config) -> None:
    """Duplicate matches between belief search, uniform search, and the raw policy."""
    game = make("ventures")
    base = {
        "policy": config.policy,
        "value": config.value,
        "simulations": config.simulations,
        "determinizations": config.determinizations,
    }
    specs = {
        "belief": (search_agent, {**base, "belief": config.belief}),
        "uniform": (search_agent, base),
        "policy": (ppo_agent, {"path": config.policy}),
    }
    for match in config.matches:
        a, b = match.split("_vs_")
        t = time.perf_counter()
        r = duplicate_match(game, specs[a], specs[b], config.match_deals, seed=7)
        print(
            f"{a} vs {b}: score {r.score:.3f}±{r.score_se:.3f}  margin {r.margin:+.1f}"
            f"±{r.margin_se:.1f}  ({r.deals} deals, {time.perf_counter() - t:.0f}s)",
            flush=True,
        )


def main(config: Config) -> None:
    """Run the belief-quality diagnostic, then the search matches."""
    if config.quality_games:
        belief_quality(config)
    if config.matches:
        search_strength(config)


if __name__ == "__main__":
    main(Config())
