"""Public beliefs about both players' hands, tracked with weighted candidate hands.

A public belief state needs, for each player, a distribution over that player's hidden hand given
only what *both* players have seen. On a game too large to enumerate hands this module keeps
``particles`` candidate hands per player with weights, and updates them from public events only:

* **A player acts.** The card that leaves or enters the hand is public. Each of that player's
  candidates is first made consistent with it (a candidate that did not hold a card just played
  gets it, in place of a random card it is not known to hold), then weighted by the probability a
  *policy model* gives the action from that hand: ``(1 - epsilon) * pi(a)^temperature + epsilon /
  |legal|``. The other player's candidates that held a card now seen elsewhere are repaired.
* **A player draws unseen.** Each candidate draws a card at random from the cards it has not
  accounted for, less the share the other player is believed to hold.

Weights are resampled when the effective sample size falls below ``resample_frac``.

This differs from ``agents.belief.ParticleBelief``, which tracks full worlds from one player's
own view (using that player's hand). Here nothing private is read: the true state is used only
for its public part (hand sizes, the card a move shows, cards known to be held).

Game hooks: ``private_hand`` (read only as a public *difference* across a move, and for sizes),
``unseen_cards``, ``known_cards``, ``with_hands`` and ``chance_is_private``.

On Ventures self-play of ``diverse_sym`` (48 candidates, sampled play): the belief's per-card
marginals overlap the true hand by 0.35 with the policy model and 0.30 without it (cards alone),
for ~7 ms per move; 128 candidates do no better.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from iirl.core.game import Game, State
from iirl.core.types import Action

PolicyModel = Callable[[Game, Sequence[State]], Sequence[np.ndarray]]


@dataclass(frozen=True)
class PublicBeliefConfig:
    """Settings for :class:`PublicBelief`.

    ``temperature`` sharpens the model's probabilities (above 1 fits a player who acts greedily
    on the model's policy; 0 ignores the model, leaving only what the cards themselves show).
    """

    particles: int = 48
    epsilon: float = 0.05
    temperature: float = 1.0
    resample_frac: float = 0.5


def _sample_rows(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one column index per row in proportion to that row's (non-negative) weights."""
    cum = np.cumsum(weights, axis=1)
    pick = rng.random(len(weights)) * cum[:, -1]
    return np.minimum((cum <= pick[:, None]).sum(axis=1), weights.shape[1] - 1)


class PublicBelief:
    """Weighted candidate hands for both players, updated from public events."""

    def __init__(
        self,
        game: Game,
        model: PolicyModel | None,
        config: PublicBeliefConfig | None = None,
        seed: int = 0,
    ) -> None:
        """``model`` gives action probabilities for the player to move in each state."""
        self.game, self.model = game, model
        self.config = config or PublicBeliefConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Forget the game in progress (candidates appear once the deal is complete)."""
        self.hands: list[np.ndarray] | None = None
        self.weights: list[np.ndarray] = []

    def marginals(self, player: int) -> np.ndarray:
        """Expected count of each card in ``player``'s hand under the belief."""
        w = self.weights[player]
        return (w / w.sum()) @ self.hands[player]

    def observe(self, before: State, action: Action, after: State) -> None:
        """Update on the transition ``before`` -> ``after`` (a move or a chance outcome)."""
        game = self.game
        if self.hands is None:
            if not game.is_chance_node(after) and not game.is_terminal(after):
                self._deal(after)
            return
        if game.is_terminal(after):
            return
        size = [
            sum(game.private_hand(after, p)) - sum(game.private_hand(before, p)) for p in (0, 1)
        ]
        if game.is_chance_node(before):
            if game.chance_is_private(before):
                self._draw(int(np.argmax(size)), after)
        else:
            self._act(game.current_player(before), before, action, after)
        for q in (0, 1):
            self._repair(q, after)
            self._maybe_resample(q)

    # --- pieces -------------------------------------------------------------------------------
    def _bounds(self, state: State, player: int) -> tuple[np.ndarray, np.ndarray]:
        """The most and the least of each card ``player`` can hold, publicly."""
        game = self.game
        most = np.asarray(game.unseen_cards(state)) - np.asarray(
            game.known_cards(state, 1 - player)
        )
        return most, np.asarray(game.known_cards(state, player))

    def _deal(self, state: State) -> None:
        game, k = self.game, self.config.particles
        self.hands, self.weights = [], []
        for p in (0, 1):
            most, _ = self._bounds(state, p)
            cards = np.repeat(np.arange(len(most)), most)
            n = sum(game.private_hand(state, p))
            order = np.argsort(self.rng.random((k, len(cards))), axis=1)[:, :n]
            hands = np.zeros((k, len(most)), dtype=np.int64)
            np.add.at(hands, (np.repeat(np.arange(k), n), cards[order].ravel()), 1)
            self.hands.append(hands)
            self.weights.append(np.full(k, 1.0 / k))

    def _act(self, p: int, before: State, action: Action, after: State) -> None:
        game, cfg = self.game, self.config
        moved = np.asarray(game.private_hand(after, p)) - np.asarray(game.private_hand(before, p))
        lost, gained = np.clip(-moved, 0, None), np.clip(moved, 0, None)
        hands = self.hands[p]
        _, least = self._bounds(before, p)
        for c in np.flatnonzero(lost):  # the card shown: every candidate must have held it
            missing = np.flatnonzero(hands[:, c] < lost[c])
            if len(missing):
                spare = (hands[missing] - least).astype(float)
                spare[:, c] = 0.0
                out = _sample_rows(spare, self.rng)
                hands[missing, out] -= 1
                hands[missing, c] += 1
        if self.model is not None and cfg.temperature > 0:
            other = self._fillers(before, p, hands)
            worlds = [
                game.with_hands(before, p, tuple(h), tuple(f))
                for h, f in zip(hands.tolist(), other.tolist(), strict=True)
            ]
            probs = np.asarray([pr[action] for pr in self.model(game, worlds)], dtype=float)
            legal = np.asarray([len(game.legal_actions(w)) for w in worlds])
            if cfg.temperature != 1.0:
                probs = probs**cfg.temperature
            self.weights[p] = self.weights[p] * ((1 - cfg.epsilon) * probs + cfg.epsilon / legal)
            if self.weights[p].sum() <= 0:
                self.weights[p] = np.full(len(hands), 1.0 / len(hands))
        hands += gained - lost

    def _fillers(self, state: State, p: int, hands: np.ndarray) -> np.ndarray:
        """Some hand for the other player beside each candidate (its identity does not matter)."""
        game = self.game
        known = np.asarray(game.known_cards(state, 1 - p))
        pool = np.asarray(game.unseen_cards(state)) - hands - known
        need = sum(game.private_hand(state, 1 - p)) - known.sum()
        before = np.cumsum(pool, axis=1) - pool
        return known + np.clip(need - before, 0, pool)

    def _free(self, q: int, state: State, hands: np.ndarray) -> np.ndarray:
        """How many of each card a candidate of ``q`` could still pick up, believed likelihoods."""
        most, _ = self._bounds(state, q)
        _, held_known = self._bounds(state, 1 - q)
        elsewhere = self.marginals(1 - q) - held_known  # the other's believed unknown cards
        room = np.clip(most - hands, 0, None).astype(float)
        weights = np.clip(room - elsewhere, 0.0, None)
        empty = weights.sum(axis=1) <= 0
        weights[empty] = room[empty]
        return weights

    def _draw(self, p: int, after: State) -> None:
        hands = self.hands[p]
        size = sum(self.game.private_hand(after, p))
        short = np.flatnonzero(hands.sum(axis=1) < size)
        if len(short):
            picked = _sample_rows(self._free(p, after, hands[short]), self.rng)
            hands[short, picked] += 1

    def _repair(self, q: int, state: State) -> None:
        """Make ``q``'s candidates possible again after cards were shown elsewhere."""
        hands = self.hands[q]
        most, least = self._bounds(state, q)
        np.minimum(hands, most, out=hands)
        np.maximum(hands, least, out=hands)
        size = sum(self.game.private_hand(state, q))
        for _ in range(size):
            total = hands.sum(axis=1)
            short = np.flatnonzero(total < size)
            if len(short):
                picked = _sample_rows(self._free(q, state, hands[short]), self.rng)
                hands[short, picked] += 1
            over = np.flatnonzero(total > size)
            if len(over):
                spare = (hands[over] - least).astype(float)
                hands[over, _sample_rows(spare, self.rng)] -= 1
            if not len(short) and not len(over):
                break

    def _maybe_resample(self, q: int) -> None:
        w = self.weights[q] / self.weights[q].sum()
        self.weights[q] = w
        k = len(w)
        if 1.0 / np.square(w).sum() >= self.config.resample_frac * k:
            return
        points = (self.rng.random() + np.arange(k)) / k
        pick = np.minimum(np.searchsorted(np.cumsum(w), points), k - 1)
        self.hands[q] = self.hands[q][pick]
        self.weights[q] = np.full(k, 1.0 / k)
