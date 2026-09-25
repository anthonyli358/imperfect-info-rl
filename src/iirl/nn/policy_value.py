"""A policy+value network and an evaluator that serves it to the search.

The network is game-agnostic: an MLP trunk over a game's ``information_state_tensor`` feeds a policy
head over ``num_distinct_actions`` and a scalar value head in ``[-1, 1]``. Feature standardization
is carried as buffers so it travels with the checkpoint and is applied identically at train and
search time. The value is always from the perspective of the player to move.

This is the neural core of the AlphaZero-style loop: the search calls :class:`NetEvaluator` for a
leaf value (replacing random rollouts) and action priors; self-play trains the net toward the
search's improved visit distribution and the game outcome.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from iirl.core.game import Game, State


class PolicyValueNet(nn.Module):
    """MLP trunk with a policy head (logits over all actions) and a tanh value head."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden: tuple[int, ...] = (256, 256),
        value_activation: str = "tanh",
    ) -> None:
        """Build the network for ``input_dim`` features and ``num_actions`` outputs.

        ``value_activation`` is ``"tanh"`` (value in ``[-1, 1]``, for AlphaZero's search leaf eval)
        or ``"linear"`` (unbounded, for use as an RL value baseline in PPO).
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.value_activation = value_activation
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(d, num_actions)
        self.value_head = nn.Linear(d, 1)

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Freeze feature standardization (std floored away from 0), stored as buffers."""
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-6))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, value)`` for a batch of raw feature rows.

        Value is shape ``[B]`` — in ``[-1, 1]`` under ``tanh``, unbounded under ``linear``.
        """
        h = self.trunk((x - self.feat_mean) / self.feat_std)
        value = self.value_head(h).squeeze(-1)
        if self.value_activation == "tanh":
            value = torch.tanh(value)
        return self.policy_head(h), value


class NetEvaluator:
    """Serves the network to the search: legal-masked priors and a leaf value for a state."""

    def __init__(self, net: PolicyValueNet) -> None:
        """Wrap ``net`` for inference (eval mode, no grad)."""
        self.net = net
        self.net.eval()

    def evaluate(self, game: Game, state: State) -> tuple[np.ndarray, float]:
        """Return ``(priors, value)``: a legal-masked prior over all actions and the mover's value.

        ``priors`` is a length-``num_actions`` vector that is 0 on illegal actions and sums to 1
        over the legal ones; ``value`` is the net's estimate for the player to move, in ``[-1, 1]``.
        """
        player = game.current_player(state)
        feat = torch.as_tensor(
            game.information_state_tensor(state, player), dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(feat)
        logits = logits.squeeze(0).numpy()
        legal = game.legal_actions(state)
        priors = np.zeros(self.net.num_actions)
        masked = logits[legal] - logits[legal].max()
        exp = np.exp(masked)
        priors[legal] = exp / exp.sum()
        return priors, float(value.item())

    def evaluate_batch(
        self, game: Game, states: list[State]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Evaluate many states in one forward pass; return per-state priors and a value array.

        This is the batched counterpart of :meth:`evaluate` — the search collects a wave of leaves
        and evaluates them together, amortizing the fixed per-call overhead that dominates small-net
        CPU inference.
        """
        feats = np.stack([game.information_state_tensor(s, game.current_player(s)) for s in states])
        with torch.no_grad():
            logits_t, values_t = self.net(torch.as_tensor(feats, dtype=torch.float32))
        logits = logits_t.numpy()
        values = values_t.numpy()
        priors_list = []
        for i, state in enumerate(states):
            legal = game.legal_actions(state)
            priors = np.zeros(self.net.num_actions)
            masked = logits[i][legal] - logits[i][legal].max()
            exp = np.exp(masked)
            priors[legal] = exp / exp.sum()
            priors_list.append(priors)
        return priors_list, values


def privileged_features(game: Game, state: State, player: int) -> np.ndarray:
    """Full-state features for ``player``: their own view followed by the opponent's view.

    The opponent's view includes their hidden hand, so this is only legitimate where the state is
    fully known — a determinized world inside search, or training targets built after the fact.
    """
    return np.concatenate(
        [
            game.information_state_tensor(state, player),
            game.information_state_tensor(state, 1 - player),
        ]
    )


class SplitEvaluator:
    """Serves the search priors from one network and the leaf value from another.

    This is how a model-free policy and a separately trained value combine into AlphaZero-style
    search: e.g. a PPO policy supplies the priors (where to look) while a value net, trained on
    self-play outcomes, evaluates the leaves (who is winning).

    With ``privileged=True`` the value net reads :func:`privileged_features` — both players' views.
    Inside a determinized world every card in hand is known, so a leaf can be valued as the full
    position it is, which removes most of the deal's luck from the value target. The priors always
    read the mover's information state only.
    """

    def __init__(
        self, prior_net: PolicyValueNet, value_net: PolicyValueNet, privileged: bool = False
    ) -> None:
        """Wrap the prior network and the value network for inference."""
        self.priors = NetEvaluator(prior_net)
        self.values = NetEvaluator(value_net)
        self.net = prior_net
        self.privileged = privileged

    def evaluate(self, game: Game, state: State) -> tuple[np.ndarray, float]:
        """Return ``(priors from the prior net, value from the value net)`` for one state."""
        priors, values = self.evaluate_batch(game, [state])
        return priors[0], float(values[0])

    def evaluate_batch(
        self, game: Game, states: list[State]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Batched :meth:`evaluate`: per-state priors and a value array."""
        movers = [game.current_player(s) for s in states]
        own = np.stack(
            [game.information_state_tensor(s, p) for s, p in zip(states, movers, strict=True)]
        )
        if self.privileged:
            opp = np.stack(
                [
                    game.information_state_tensor(s, 1 - p)
                    for s, p in zip(states, movers, strict=True)
                ]
            )
            vfeat = np.concatenate([own, opp], axis=1)
        else:
            vfeat = own
        with torch.no_grad():
            logits = self.priors.net(torch.as_tensor(own, dtype=torch.float32))[0].numpy()
            values = self.values.net(torch.as_tensor(vfeat, dtype=torch.float32))[1].numpy()
        priors_list = []
        for i, state in enumerate(states):
            legal = game.legal_actions(state)
            priors = np.zeros(self.net.num_actions)
            masked = logits[i][legal] - logits[i][legal].max()
            exp = np.exp(masked)
            priors[legal] = exp / exp.sum()
            priors_list.append(priors)
        return priors_list, values
