"""Train an info-state value net on the PPO agent's self-play, for AlphaZero-style search.

Needs the PPO checkpoint (``runs/ppo_ventures.pt``, from ``experiments/train_ventures_ppo.py``).
Writes ``runs/value_ventures.pt`` and prints held-out Brier scores for predicting the winner: the
learned value vs the hand-crafted deck-clock leaf vs a coin flip. With the value as the leaf and the
PPO policy as priors, determinized search beats the PPO policy alone (see ``writeup.md``).
About 10 minutes on a 12-core CPU.

Run with::

    uv run python -m experiments.train_ventures_value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.selfplay.value import ValueConfig, train_value


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    policy_checkpoint: str = "runs/ppo_ventures.pt"
    out_path: str = "runs/value_ventures.pt"
    value: ValueConfig = field(default_factory=ValueConfig)


def main(config: Config) -> None:
    """Train the value net on the PPO policy's self-play and save it."""
    game = make("ventures")
    policy = PPOAgent.from_checkpoint(config.policy_checkpoint).net
    hidden = tuple(torch.load(config.policy_checkpoint, map_location="cpu")["hidden"])
    net, report = train_value(game, policy, hidden, config.value)
    print(f"positions: {report.positions}")
    print("held-out Brier (lower is better; predicting the winner)")
    print(f"  learned value         {report.learned:.4f}")
    print(f"  heuristic leaf        {report.heuristic:.4f}  (tanh(h / {report.heuristic_scale:g}))")
    print(f"  coin flip             {report.coin_flip:.4f}")
    out = Path(config.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_dim": net.input_dim,
            "num_actions": net.num_actions,
            "hidden": list(config.value.hidden),
            "privileged": config.value.privileged,
        },
        out,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main(Config())
