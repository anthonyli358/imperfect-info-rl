"""A linear softmax policy and the REINFORCE agent that plays it.

The policy is deliberately the simplest learnable thing that fits the interface: a linear map from a
game's (normalized) ``information_state_tensor`` to per-action logits, masked to the legal actions
and softmaxed. It is game-agnostic — any game that provides an info-state tensor, a legal-action
list, and ``num_distinct_actions`` can be learned. Trained by policy gradient in the sibling
``trainer`` module.

Numpy only (torch arrives with the neural self-play spine in Stage 3). Gradients are the exact
softmax-policy-gradient, verified against finite differences in the tests.
"""

from __future__ import annotations

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action

_NEG_INF = -1e30


class LinearPolicy:
    """Linear softmax policy ``π(a|s) = softmax(W·φ(s) + b)`` over legal actions.

    Features ``φ`` are the info-state tensor standardized by frozen mean/std (set via
    :meth:`set_feature_stats`), which keeps the differently-scaled tensor components comparable.
    """

    def __init__(self, feature_dim: int, num_actions: int, seed: int = 0) -> None:
        """Initialize zero weights and Adam state for a ``feature_dim`` → ``num_actions`` policy."""
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.01, size=(num_actions, feature_dim))
        self.b = np.zeros(num_actions)
        self.feature_mean = np.zeros(feature_dim)
        self.feature_std = np.ones(feature_dim)
        # Adam moments.
        self._mw = np.zeros_like(self.w)
        self._vw = np.zeros_like(self.w)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._t = 0

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Freeze the feature standardization used by :meth:`features` (std floored away from 0)."""
        self.feature_mean = np.asarray(mean, dtype=float)
        self.feature_std = np.maximum(np.asarray(std, dtype=float), 1e-6)

    def features(self, game: Game, state: State) -> np.ndarray:
        """Return the standardized feature vector for ``state``."""
        raw = game.information_state_tensor(state, game.current_player(state)).astype(float)
        return (raw - self.feature_mean) / self.feature_std

    def probs(self, phi: np.ndarray, legal: list[Action]) -> np.ndarray:
        """Return the full-length action distribution (0 on illegal actions)."""
        logits = self.w @ phi + self.b
        masked = np.full(self.num_actions, _NEG_INF)
        masked[legal] = logits[legal]
        masked[legal] -= masked[legal].max()
        exp = np.zeros(self.num_actions)
        exp[legal] = np.exp(masked[legal])
        return exp / exp.sum()

    def act(
        self, game: Game, state: State, rng: np.random.Generator, greedy: bool = False
    ) -> Action:
        """Return an action: greedy (argmax) for eval, or sampled from ``π`` for exploration."""
        legal = game.legal_actions(state)
        p = self.probs(self.features(game, state), legal)
        if greedy:
            return int(max(legal, key=lambda a: p[a]))
        return int(rng.choice(self.num_actions, p=p))

    def apply_gradient(
        self,
        phis: np.ndarray,
        actions: np.ndarray,
        probs: np.ndarray,
        advantages: np.ndarray,
        lr: float,
        entropy_coef: float = 0.0,
    ) -> None:
        """Take one Adam ascent step of the entropy-regularized policy gradient.

        Args:
            phis: ``[N, F]`` feature rows for each decision.
            actions: ``[N]`` action ids taken.
            probs: ``[N, A]`` action distributions at each decision (0 on illegal).
            advantages: ``[N]`` scalar advantages (return minus baseline).
            lr: Adam step size.
            entropy_coef: Weight on the entropy bonus (encourages exploration).
        """
        grad_w, grad_b = self.policy_gradient(phis, actions, probs, advantages, entropy_coef)
        self._adam_step(grad_w, grad_b, lr)

    @staticmethod
    def policy_gradient(
        phis: np.ndarray,
        actions: np.ndarray,
        probs: np.ndarray,
        advantages: np.ndarray,
        entropy_coef: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(grad_w, grad_b)``, the ascent gradient of mean ``advantage·log π`` (+ entropy).

        Exposed for finite-difference verification; :meth:`apply_gradient` calls it then steps Adam.
        """
        n = len(actions)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), actions] = 1.0
        # d/d(logits) of advantage·log π(a) is advantage·(onehot − p).
        grad_logits = advantages[:, None] * (onehot - probs)
        if entropy_coef:
            logp = np.where(probs > 0, np.log(np.where(probs > 0, probs, 1.0)), 0.0)
            entropy = -(probs * logp).sum(axis=1, keepdims=True)
            grad_logits += entropy_coef * (-(probs * (logp + entropy)))
        grad_w = grad_logits.T @ phis / n
        grad_b = grad_logits.mean(axis=0)
        return grad_w, grad_b

    def _adam_step(self, grad_w: np.ndarray, grad_b: np.ndarray, lr: float) -> None:
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._t += 1
        self._mw = b1 * self._mw + (1 - b1) * grad_w
        self._vw = b2 * self._vw + (1 - b2) * grad_w**2
        self._mb = b1 * self._mb + (1 - b1) * grad_b
        self._vb = b2 * self._vb + (1 - b2) * grad_b**2
        mw_hat = self._mw / (1 - b1**self._t)
        vw_hat = self._vw / (1 - b2**self._t)
        mb_hat = self._mb / (1 - b1**self._t)
        vb_hat = self._vb / (1 - b2**self._t)
        self.w += lr * mw_hat / (np.sqrt(vw_hat) + eps)  # ascent: maximize expected return
        self.b += lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def save(self, path: str) -> None:
        """Save weights and feature stats to a ``.npz`` checkpoint."""
        np.savez(path, w=self.w, b=self.b, mean=self.feature_mean, std=self.feature_std)

    @classmethod
    def load(cls, path: str) -> LinearPolicy:
        """Load a policy from a ``.npz`` checkpoint."""
        data = np.load(path)
        policy = cls(data["w"].shape[1], data["w"].shape[0])
        policy.w, policy.b = data["w"], data["b"]
        policy.set_feature_stats(data["mean"], data["std"])
        return policy


class ReinforceAgent(Agent):
    """Plays a :class:`LinearPolicy` (greedy by default for evaluation)."""

    def __init__(self, policy: LinearPolicy, name: str = "reinforce", greedy: bool = True) -> None:
        """Wrap ``policy`` as an :class:`~iirl.agents.base.Agent`."""
        self.policy = policy
        self.name = name
        self.greedy = greedy

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return the policy's action for the current state."""
        return self.policy.act(game, state, rng, greedy=self.greedy)
