"""A ReBeL-style agent for games too large to enumerate: one-turn search over public beliefs.

At the start of each of its turns the agent solves the turn's subgame (``selfplay.rebel_turn``)
from the tracked public beliefs about both hands (``selfplay.public_belief``), with a
belief-conditioned value network at the leaves (``selfplay.rebel_value``), and plays the solved
strategy for the hand it actually holds. A *blueprint* policy supplies the likelihoods that
update the beliefs after every move, and is the fallback wherever no solve is available.

The agent's own hand enters the solve as one more candidate with negligible weight, so it is
solved for without telling the "public" beliefs what the agent holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from iirl.agents.base import Agent
from iirl.agents.belief import net_opponent_model
from iirl.agents.ppo.agent import PPOAgent
from iirl.core.game import Game, State
from iirl.core.types import Action, Player
from iirl.nn.policy_value import NetEvaluator
from iirl.selfplay.public_belief import PublicBelief, PublicBeliefConfig
from iirl.selfplay.rebel_turn import TurnSolver
from iirl.selfplay.rebel_value import HandValueNet, default_belief_config


@dataclass(frozen=True)
class ReBeLAgentConfig:
    """Settings for :class:`ReBeLAgent`.

    ``candidates`` is how many of its own candidate hands the solve keeps (drawn by weight; the
    solve's cost grows with it), ``draws`` how many unseen draws it follows per action sequence.
    ``greedy`` plays the solved strategy's most likely action instead of sampling from it.

    ``anchor`` (points) switches the choice of action from the solved strategy to the blueprint's
    policy reweighted by the solve's action values: ``blueprint(a) * exp(value(a) / anchor)``.
    The search then overrides the blueprint only where it sees a difference that is large next
    to ``anchor``, which should sit near the size of the value network's errors.
    """

    blueprint: str
    value: str
    iterations: int = 20
    candidates: int = 16
    draws: int | None = 6
    explore: float = 0.05
    greedy: bool = True
    anchor: float | None = None
    belief: PublicBeliefConfig = field(default_factory=default_belief_config)
    seed: int = 0


def load_value(path: str) -> HandValueNet:
    """Load a value network saved by ``experiments/rebel_ventures.py``."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = HandValueNet(
        ck["n_public"], ck["n_cards"], ck["n_extra"], ck["hidden"], ck["layers"], ck["scale"]
    )
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


class ReBeLAgent(Agent):
    """Plays the one-turn public-belief solve of its turn; see the module docstring."""

    def __init__(self, config: ReBeLAgentConfig, name: str = "rebel") -> None:
        """Load the blueprint and the value network."""
        self.config, self.name = config, name
        self.blueprint = PPOAgent.from_checkpoint(config.blueprint, greedy=True)
        self.model = net_opponent_model(NetEvaluator(self.blueprint.net))
        self.value = load_value(config.value)
        self.rng = np.random.default_rng(config.seed)
        self.solves = 0
        self.fallbacks = 0
        self.changed = 0  # anchored choices that differ from the blueprint's

    def reset(self, game: Game, player: Player) -> None:
        """Start a new game: fresh beliefs, no plan."""
        self.player = player
        self.belief = PublicBelief(
            game, self.model, self.config.belief, seed=int(self.rng.integers(2**31))
        )
        self.solver = TurnSolver(
            game,
            self.config.explore,
            draws=self.config.draws,
            other_values=False,
            seed=int(self.rng.integers(2**31)),
        )
        self.previous: State = game.new_initial_state()
        self.plan: tuple[list[int], np.ndarray, np.ndarray] | None = None

    def observe(self, game: Game, state: State, action: Action | None) -> None:
        """Update the public beliefs on every transition."""
        self.belief.observe(self.previous, action, state)
        self.previous = state

    def _choose(
        self,
        game: Game,
        state: State,
        actions: list[int],
        probs: np.ndarray,
        worth: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        if self.config.anchor is not None:
            prior = np.asarray(self.model(game, [state])[0])[actions]
            probs = prior * np.exp((worth - worth.max()) / self.config.anchor)
            self.changed += int(np.argmax(probs) != np.argmax(prior))
        if self.config.greedy:
            return int(actions[int(np.argmax(probs))])
        return int(actions[int(rng.choice(len(actions), p=probs / probs.sum()))])

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Solve at the start of a turn; follow the solve's second decision afterwards."""
        legal = game.legal_actions(state)
        me = self.player
        if self.plan is not None:  # the second decision of a turn already solved
            actions, probs, worth = self.plan
            self.plan = None
            if set(actions) == set(legal):
                return self._choose(game, state, actions, probs, worth, rng)
        mine = np.asarray(game.private_hand(state, me))
        other = sum(game.private_hand(state, 1 - me))
        if self.belief.hands is None or mine.sum() < other:
            self.fallbacks += 1
            return self.blueprint.step(game, state, rng)
        # My candidates: a weighted draw from the public belief about my hand, plus my real hand
        # at negligible weight (index 0). The opponent's: every distinct candidate.
        w = self.belief.weights[me]
        k = min(self.config.candidates, len(w))
        points = (self.rng.random() + np.arange(k)) / k
        pick = np.minimum(np.searchsorted(np.cumsum(w / w.sum()), points), len(w) - 1)
        hands_me, counts = np.unique(self.belief.hands[me][pick], axis=0, return_counts=True)
        hands_me = np.concatenate([mine[None, :], hands_me])
        weights_me = np.concatenate([[1e-9], counts / counts.sum()])
        hands_opp, inverse = np.unique(self.belief.hands[1 - me], axis=0, return_inverse=True)
        weights_opp = np.bincount(inverse.ravel(), weights=self.belief.weights[1 - me])
        hands = (hands_me, hands_opp) if me == 0 else (hands_opp, hands_me)
        ranges = (weights_me, weights_opp) if me == 0 else (weights_opp, weights_me)
        solution = self.solver.solve(
            state, hands, ranges, self.value.value_fn, self.config.iterations
        )
        if not solution.actions[0] or (0, solution.actions[0][0]) not in solution.follow_ups:
            # No opponent candidate is compatible with my real hand.
            self.fallbacks += 1
            return self.blueprint.step(game, state, rng)
        self.solves += 1
        action = self._choose(
            game, state, solution.actions[0], solution.average[0], solution.action_values[0], rng
        )
        follow, probs, _, worth = solution.follow_ups[(0, action)]
        self.plan = (list(follow), np.asarray(probs), np.asarray(worth))
        return action


def rebel_agent(**settings: object) -> ReBeLAgent:
    """Build a :class:`ReBeLAgent` from config fields (a picklable spec for matches)."""
    return ReBeLAgent(ReBeLAgentConfig(**settings))
