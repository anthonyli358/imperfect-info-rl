"""An entity policy/value network: a transformer over one token per piece, with a pointer head.

A flat MLP sees a game's pieces as one long vector, so it must learn separately what each slot
means. Here each piece is a token. The information-state tensor is read as ``tokens`` rows of
``token_dim`` features (one per piece, the player's own pieces first) followed by global features
(phase, cards, clock, the fight), which become one more token. A transformer encoder lets every
piece attend to every other.

Heads:

- **moves (the pointer head):** each of the player's own ``own_tokens`` pieces scores the
  ``regions`` destinations from its own output: "this piece, to that region". The logits are laid
  out piece-major, so they must be the first ``own_tokens * regions`` action ids;
- **every other action** (cards, setup choices) from the global token's output;
- **value** from the global token's output.

It is a drop-in for :class:`~iirl.nn.policy_value.PolicyValueNet` (same ``forward`` and feature
standardization), so PPO trains it unchanged (``PPOConfig(arch="entity", entity=...)``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EntityPolicyValueNet(nn.Module):
    """Transformer over piece tokens; see the module docstring."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        tokens: int,
        token_dim: int,
        own_tokens: int,
        regions: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        value_activation: str = "linear",
    ) -> None:
        """Lay out ``input_dim`` features as ``tokens`` × ``token_dim`` plus global features."""
        super().__init__()
        self.input_dim, self.num_actions = input_dim, num_actions
        self.tokens, self.token_dim = tokens, token_dim
        self.own_tokens, self.regions = own_tokens, regions
        self.global_dim = input_dim - tokens * token_dim
        self.moves = own_tokens * regions
        assert self.global_dim > 0 and num_actions > self.moves
        self.value_activation = value_activation
        self.spec = {
            "tokens": tokens,
            "token_dim": token_dim,
            "own_tokens": own_tokens,
            "regions": regions,
            "d_model": d_model,
            "heads": heads,
            "layers": layers,
        }
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        self.token_in = nn.Linear(token_dim, d_model)
        self.slot = nn.Parameter(torch.zeros(tokens + 1, d_model))  # which token slot it is
        self.global_in = nn.Linear(self.global_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.move_head = nn.Linear(d_model, regions)
        self.other_head = nn.Linear(d_model, num_actions - self.moves)
        self.value_head = nn.Linear(d_model, 1)

    def set_feature_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Freeze feature standardization (std floored away from 0), stored as buffers."""
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-6))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, value)`` for a batch of raw feature rows."""
        x = (x - self.feat_mean) / self.feat_std
        b = x.shape[0]
        split = self.tokens * self.token_dim
        pieces = self.token_in(x[:, :split].reshape(b, self.tokens, self.token_dim))
        glob = self.global_in(x[:, split:]).unsqueeze(1)
        h = self.encoder(torch.cat([pieces, glob], dim=1) + self.slot)
        moves = self.move_head(h[:, : self.own_tokens]).reshape(b, self.moves)
        g = h[:, -1]
        logits = torch.cat([moves, self.other_head(g)], dim=1)
        value = self.value_head(g).squeeze(-1)
        if self.value_activation == "tanh":
            value = torch.tanh(value)
        return logits, value


def build_net(checkpoint: dict, value_activation: str = "linear") -> nn.Module:
    """The network a saved checkpoint describes (an MLP unless it records ``arch: entity``)."""
    from iirl.nn.policy_value import PolicyValueNet

    if checkpoint.get("arch") == "entity":
        return EntityPolicyValueNet(
            checkpoint["input_dim"],
            checkpoint["num_actions"],
            value_activation=value_activation,
            **checkpoint["entity"],
        )
    return PolicyValueNet(
        checkpoint["input_dim"],
        checkpoint["num_actions"],
        tuple(checkpoint["hidden"]),
        value_activation=value_activation,
    )


def save_checkpoint(net: nn.Module, path) -> None:  # noqa: ANN001 - str or Path
    """Save a PPO-style checkpoint that :func:`build_net` can rebuild (MLP or entity network)."""
    ck = {
        "state_dict": net.state_dict(),
        "input_dim": net.input_dim,
        "num_actions": net.num_actions,
    }
    if isinstance(net, EntityPolicyValueNet):
        ck.update(arch="entity", entity=net.spec, hidden=[])
    else:
        ck["hidden"] = [layer.out_features for layer in net.trunk if hasattr(layer, "out_features")]
    torch.save(ck, path)
