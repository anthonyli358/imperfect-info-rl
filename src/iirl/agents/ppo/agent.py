"""The PPO agent: play a policy/value network's (masked) policy head."""

from __future__ import annotations

import numpy as np
import torch

from iirl.agents.base import Agent
from iirl.core.game import Game, State
from iirl.core.types import Action
from iirl.nn.policy_value import PolicyValueNet


def masked_policy(net: PolicyValueNet, game: Game, state: State) -> np.ndarray:
    """Return the network's action distribution over legal actions (0 on illegal), as numpy."""
    player = game.current_player(state)
    feat = torch.as_tensor(
        game.information_state_tensor(state, player), dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        logits, _ = net(feat)
    logits = logits.squeeze(0).numpy()
    legal = game.legal_actions(state)
    probs = np.zeros(net.num_actions)
    masked = logits[legal] - logits[legal].max()
    exp = np.exp(masked)
    probs[legal] = exp / exp.sum()
    return probs / probs.sum()  # renormalize so it sums to exactly 1 for sampling


class PPOAgent(Agent):
    """Plays a PPO-trained policy (greedy by default for evaluation)."""

    def __init__(self, net: PolicyValueNet, name: str = "ppo", greedy: bool = True) -> None:
        """Wrap ``net`` as an agent."""
        self.net = net
        self.net.eval()
        self.name = name
        self.greedy = greedy

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Return the greedy or sampled action from the policy head."""
        probs = masked_policy(self.net, game, state)
        legal = game.legal_actions(state)
        if self.greedy:
            return int(legal[int(np.argmax([probs[a] for a in legal]))])
        return int(rng.choice(len(probs), p=probs))
