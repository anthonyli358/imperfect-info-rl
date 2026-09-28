"""Bayesian determinization: a particle filter over the worlds consistent with a player's view.

Uniform determinization (``game.resample_from_infostate``) deals the hidden cards as if the
opponent's moves said nothing about them. They do: a player who discards a card they could have
played, or takes a discard, or keeps holding cards back, is telling you about their hand. This
module keeps a weighted set of full worlds (*particles*) and updates it on every transition:

* **Opponent move** ``a``: each particle's weight is multiplied by the likelihood of ``a`` under an
  *opponent model* evaluated on the opponent's view of that particle,
  ``(1 - epsilon) * pi(a) + epsilon / |legal|``. The ``epsilon`` floor keeps the filter robust
  when the model is wrong (a different opponent, or a greedy one playing a move the model finds
  unlikely).
* **Own move**: applied to every particle (it is legal in all of them: they share this player's
  view).
* **Chance event**: each particle takes an outcome consistent with what this player observed, drawn
  in proportion to its chance probability, and is weighted by the total probability of those
  outcomes. A card this player draws is then likely only in worlds whose deck still holds it; a
  card the opponent draws unseen is simply sampled.

After every update, particles whose view for this player no longer matches the real one die
(weight 0). When the effective sample size falls below ``resample_frac * n``, the particles are
resampled (systematic resampling).

**Rejuvenation.** A plain filter degenerates fast here: every card the opponent plays out of hand
kills the ~80% of particles that did not deal it to them, and the opening hand is never re-drawn,
so after a few reveals all particles descend from one or two guesses. With ``rejuvenate`` on, each
dead particle is *repaired* (``game.repair_world``): re-dealt to be consistent with the current
view while keeping as much of its old hidden hand as is still possible, at the mean live weight.
That is an approximation — the repaired part has not been scored on the opponent's earlier moves
— which trades a little bias for a belief that does not collapse. (Replacing dead particles with
fresh *uniform* resamples instead was measurably worse than no rejuvenation at all: nearly every
particle dies at each reveal, so the fresh ones swamp the evidence.) If every particle dies, all
are repaired at equal weight.

Assumption, kept game-agnostic: whether a chance outcome is consistent with the observer's view
does not depend on the hidden state (a draw is either seen or not, whatever the hidden cards are).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from iirl.core.game import Game, State
from iirl.core.types import Action, Player

# An opponent model: the action distribution (over ``num_distinct_actions``, 0 on illegal) for the
# player to move in each state.
OpponentModel = Callable[[Game, Sequence[State]], Sequence[np.ndarray]]


@dataclass(frozen=True)
class BeliefConfig:
    """Particle-filter settings.

    Attributes:
        particles: Number of particles.
        epsilon: Uniform mixing in the opponent-move likelihood (robustness to model error).
        temperature: Likelihood exponent on the model's probability; above 1 sharpens it (fits a
            greedy opponent: on Ventures vs greedy PPO, 4 recovered the most of the hidden hand),
            0 ignores the model (uniform determinization, but still conditioned on draws).
        resample_frac: Resample when the effective sample size drops below this fraction.
        rejuvenate: Repair dead particles instead of dropping them (see the module notes).
    """

    particles: int = 256
    epsilon: float = 0.05
    temperature: float = 4.0
    resample_frac: float = 0.5
    rejuvenate: bool = True


class ParticleBelief:
    """A weighted set of full worlds consistent with one player's view.

    Usage: :meth:`reset` at the start of a game, :meth:`observe` after every transition (chance
    and player moves alike), then :meth:`sample` worlds for search at this player's turns.
    """

    def __init__(
        self, model: OpponentModel, config: BeliefConfig | None = None, seed: int = 0
    ) -> None:
        """Configure the filter around an opponent ``model``."""
        self.model = model
        self.config = config or BeliefConfig()
        self.rng = np.random.default_rng(seed)
        self.game: Game | None = None
        self.player: Player = 0
        self.worlds: list[State] = []
        self.weights = np.zeros(0)
        self.restarts = 0
        self.rejuvenated = 0

    def reset(self, game: Game, player: Player) -> None:
        """Start tracking a new game as ``player`` (particles appear once the deal is done)."""
        self.game, self.player = game, player
        self.worlds, self.weights = [], np.zeros(0)
        self.restarts = 0
        self.rejuvenated = 0

    def _view(self, state: State) -> str:
        assert self.game is not None
        return self.game.information_state_string(state, self.player)

    def _init_from(self, state: State) -> None:
        """(Re)create the particles by uniform resampling of ``state``'s view."""
        game, n = self.game, self.config.particles
        assert game is not None
        self.worlds = [game.resample_from_infostate(state, self.player, self.rng) for _ in range(n)]
        self.weights = np.full(n, 1.0 / n)

    def observe(self, state: State, action: Action | None) -> None:
        """Update on the transition into ``state`` by ``action`` (``None`` for a chance event).

        Only this player's view of ``state`` is read, so hidden information never leaks.
        """
        game = self.game
        assert game is not None
        if not self.worlds:
            if not game.is_chance_node(state) and not game.is_terminal(state):
                self._init_from(state)  # the deal is complete: start from uniform worlds
            return
        if game.is_terminal(state):
            return
        if action is None:
            self._chance_update(state)
        else:
            # Live particles sit where the real game was before this move, so they know the mover.
            # Dead ones (weight 0) are left stale: the move may be illegal there, and the next
            # resample drops them.
            live = np.flatnonzero(self.weights > 0)
            if game.current_player(self.worlds[live[0]]) != self.player:
                self._opponent_update(action)
            for i in np.flatnonzero(self.weights > 0):
                self.worlds[i] = game.apply_action(self.worlds[i], action)
        self._check_consistent(state)
        self._maybe_resample(state)

    def _opponent_update(self, action: Action) -> None:
        game, cfg = self.game, self.config
        assert game is not None
        live = np.flatnonzero(self.weights > 0)
        # Evaluate each distinct live particle once (resampling leaves many duplicates).
        unique: dict[State, int] = {}
        for i in live:
            unique.setdefault(self.worlds[i], len(unique))
        probs = self.model(game, list(unique))
        lik = np.zeros(len(unique))
        for w, j in unique.items():
            legal = game.legal_actions(w)
            if action not in legal:
                continue
            p = float(probs[j][action])
            if cfg.temperature != 1.0:
                p = p**cfg.temperature
            lik[j] = (1.0 - cfg.epsilon) * p + cfg.epsilon / len(legal)
        for i in live:
            self.weights[i] *= lik[unique[self.worlds[i]]]

    def _chance_update(self, state: State) -> None:
        game = self.game
        assert game is not None
        target = self._view(state)
        live = np.flatnonzero(self.weights > 0)
        # Each distinct outcome is tested once, on some particle that can produce it (one
        # particle's outcomes may lack the real one, e.g. a card it put in the opponent's hand).
        outcomes = {i: game.chance_outcomes(self.worlds[i]) for i in live}
        witness: dict[Action, int] = {}
        for i, outs in outcomes.items():
            for a, _ in outs:
                witness.setdefault(a, i)
        consistent = {
            a
            for a, i in witness.items()
            if self._view(game.apply_action(self.worlds[i], a)) == target
        }
        for i in live:
            w = self.worlds[i]
            outs = [(a, p) for a, p in outcomes[i] if a in consistent]
            total = sum(p for _, p in outs)
            if total <= 0:
                self.weights[i] = 0.0
                continue
            ps = np.array([p for _, p in outs]) / total
            a = outs[int(self.rng.choice(len(outs), p=ps))][0]
            self.worlds[i] = game.apply_action(w, a)
            self.weights[i] *= total

    def _check_consistent(self, state: State) -> None:
        game = self.game
        assert game is not None
        target = self._view(state)
        for i in np.flatnonzero(self.weights > 0):
            if self._view(self.worlds[i]) != target:
                self.weights[i] = 0.0
        total = self.weights.sum()
        if total <= 0 or not np.isfinite(total):
            self.restarts += 1
            self.worlds = [game.repair_world(state, self.player, w, self.rng) for w in self.worlds]
            self.weights = np.full(len(self.worlds), 1.0 / len(self.worlds))
            return
        dead = np.flatnonzero(self.weights == 0)
        if self.config.rejuvenate and len(dead) and not game.is_chance_node(state):
            fill = total / (len(self.weights) - len(dead))  # the mean live weight
            for i in dead:
                self.worlds[i] = game.repair_world(state, self.player, self.worlds[i], self.rng)
                self.weights[i] = fill
            self.rejuvenated += len(dead)
            total = self.weights.sum()
        self.weights /= total

    def _maybe_resample(self, state: State) -> None:
        if self.game is None or self.game.is_chance_node(state):
            return  # resample at stable points only; a chance node is followed by its outcome
        n = len(self.worlds)
        if 1.0 / np.sum(self.weights**2) >= self.config.resample_frac * n:
            return
        # Systematic resampling.
        positions = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.weights), positions), n - 1)
        self.worlds = [self.worlds[i] for i in idx]
        self.weights = np.full(n, 1.0 / n)

    def effective_size(self) -> float:
        """Return the effective sample size ``1 / sum(w^2)`` of the current particles."""
        return float(1.0 / np.sum(self.weights**2)) if len(self.weights) else 0.0

    def sample(self, state: State, rng: np.random.Generator) -> State:
        """Return one world drawn from the belief (uniform resampling if not yet initialised)."""
        game = self.game
        assert game is not None
        if not self.worlds:
            return game.resample_from_infostate(state, self.player, rng)
        return self.worlds[int(rng.choice(len(self.worlds), p=self.weights))]


def net_opponent_model(evaluator) -> OpponentModel:  # noqa: ANN001 - NetEvaluator
    """Return an :data:`OpponentModel` backed by a network evaluator's (batched) priors."""

    def model(game: Game, states: Sequence[State]) -> Sequence[np.ndarray]:
        priors, _ = evaluator.evaluate_batch(game, list(states))
        return priors

    return model
