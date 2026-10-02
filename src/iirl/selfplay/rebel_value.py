"""A belief-conditioned value network for ReBeL on a large game, trained from a blueprint policy.

ReBeL's value network answers: at this public position, with these beliefs about both hands, what
is a given hand worth? ``selfplay.rebel`` learns that from its own subgame solves, which is
affordable on a small game. On a large one a solve takes seconds, so the network is first trained
on something cheap: self-play games of an existing strong policy (the *blueprint*), with the public
beliefs tracked alongside (``selfplay.public_belief``). Every start of a turn gives two examples,
one per player: ``(public features, beliefs, that player's actual hand) -> final margin``. The
network therefore estimates a hand's value *if both sides continue with the blueprint*, which is
what a one-turn search (``selfplay.rebel_turn``) needs at its leaves to improve on the blueprint.

Rows have the layout ``rebel_turn`` uses: ``[public features, beliefs about the player to move,
beliefs about the other, a hand, whose hand (0: the player to move)]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from iirl.agents.belief import net_opponent_model
from iirl.agents.ppo.agent import PPOAgent
from iirl.core.registry import make
from iirl.nn.policy_value import NetEvaluator
from iirl.selfplay.public_belief import PublicBelief, PublicBeliefConfig


def collect(args: tuple) -> dict[str, np.ndarray]:
    """Play blueprint self-play games (sampled actions) and record every start of a turn.

    ``args`` is ``(game name, blueprint checkpoint, game seeds, belief config)``. Returns, per
    recorded position: ``public`` features, ``belief`` (mover, other), ``hand`` (mover, other),
    the mover's final ``margin`` and the ``game`` it came from.
    """
    import iirl.games  # noqa: F401  (registers the games in a worker process)

    game_name, checkpoint, seeds, belief_config = args
    torch.set_num_threads(1)
    game = make(game_name)
    agent = PPOAgent.from_checkpoint(checkpoint, greedy=False)
    model = net_opponent_model(NetEvaluator(agent.net))
    public, belief, hand, margin, which = [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tracker = PublicBelief(game, model, belief_config, seed=seed)
        state = game.new_initial_state()
        movers: list[int] = []
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            else:
                m = game.current_player(state)
                if tracker.hands is not None and not len(game.legal_actions(state)) == 0:
                    if sum(game.private_hand(state, m)) >= sum(game.private_hand(state, 1 - m)):
                        # The start of a turn: both hands are full.
                        public.append(game.public_state_tensor(state))
                        belief.append([tracker.marginals(m), tracker.marginals(1 - m)])
                        hand.append([game.private_hand(state, m), game.private_hand(state, 1 - m)])
                        movers.append(m)
                        which.append(seed)
                action = agent.step(game, state, rng)
            after = game.apply_action(state, action)
            tracker.observe(state, action, after)
            state = after
        result = game.returns(state)
        margin += [result[m] for m in movers]
    return {
        "public": np.asarray(public, dtype=np.float16),
        "belief": np.asarray(belief, dtype=np.float16),
        "hand": np.asarray(hand, dtype=np.int8),
        "margin": np.asarray(margin, dtype=np.float32),
        "game": np.asarray(which, dtype=np.int64),
    }


class HandValueNet(nn.Module):
    """Value of one hand given the public position and the beliefs about both hands.

    With ``playable_prefix`` the public features are taken to begin with two per-card masks (what
    the player to move may still play, then the other player), and the hand's cards its owner can
    play are added as an explicit input: the relation that matters most, spelled out.
    """

    def __init__(
        self,
        n_public: int,
        n_cards: int,
        hidden: int = 512,
        layers: int = 3,
        scale: float = 20.0,
        playable_prefix: bool = True,
    ) -> None:
        """Build the network; ``scale`` converts its output to payoff units."""
        super().__init__()
        self.n_public, self.n_cards, self.scale = n_public, n_cards, scale
        self.playable_prefix = playable_prefix
        n_in = n_public + 3 * n_cards + 1 + (n_cards if playable_prefix else 0)
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        blocks: list[nn.Module] = []
        width = n_in
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.mlp = nn.Sequential(*blocks, nn.Linear(width, 1))

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """Add the derived inputs to raw rows."""
        if not self.playable_prefix:
            return x
        c = self.n_cards
        hand = x[:, self.n_public + 2 * c : self.n_public + 3 * c]
        whose = x[:, -1:]
        playable = (1 - whose) * x[:, :c] + whose * x[:, c : 2 * c]
        return torch.cat([x, hand * playable], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Values (in network units: payoff / ``scale``) for raw rows."""
        return self.mlp((self.expand(x) - self.mean) / self.std)[:, 0]

    def value_fn(self, x: np.ndarray) -> np.ndarray:
        """Values in payoff units for a numpy batch of rows (for ``rebel_turn.TurnSolver``)."""
        with torch.no_grad():
            return self(torch.as_tensor(x)).numpy().astype(float) * self.scale


def rows(data: dict[str, np.ndarray], index: np.ndarray, whose: np.ndarray) -> np.ndarray:
    """Network rows for positions ``index`` and the hands of ``whose`` (0 mover, 1 other)."""
    return np.concatenate(
        [
            data["public"][index].astype(np.float32),
            data["belief"][index].reshape(len(index), -1).astype(np.float32),
            data["hand"][index, whose].astype(np.float32),
            whose[:, None].astype(np.float32),
        ],
        axis=1,
    )


@dataclass(frozen=True)
class ValueConfig:
    """Training settings for :func:`fit`."""

    hidden: int = 512
    layers: int = 3
    epochs: int = 12
    batch_size: int = 2048
    lr: float = 1e-3
    scale: float = 20.0
    holdout: float = 0.1
    use_beliefs: bool = True
    seed: int = 0


def fit(data: dict[str, np.ndarray], config: ValueConfig) -> tuple[HandValueNet, list[dict]]:
    """Train on recorded positions, holding out whole games; returns the net and its curve.

    The curve's ``rmse`` is in payoff units on held-out games, ``r2`` the share of the margin's
    variance explained there. ``use_beliefs`` False blanks the belief inputs (an ablation).
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if not config.use_beliefs:
        data = {**data, "belief": np.zeros_like(data["belief"])}
    games = np.unique(data["game"])
    held = rng.choice(games, size=max(1, int(config.holdout * len(games))), replace=False)
    is_held = np.isin(data["game"], held)
    train, test = np.flatnonzero(~is_held), np.flatnonzero(is_held)
    n_cards = data["hand"].shape[2]
    net = HandValueNet(data["public"].shape[1], n_cards, config.hidden, config.layers, config.scale)
    sample = rng.choice(train, size=min(len(train), 20000), replace=False)
    with torch.no_grad():
        x = net.expand(torch.as_tensor(rows(data, sample, rng.integers(0, 2, len(sample)))))
        net.mean.copy_(x.mean(dim=0))
        net.std.copy_(x.std(dim=0).clamp(min=1e-3))

    def target(index: np.ndarray, whose: np.ndarray) -> torch.Tensor:
        sign = 1.0 - 2.0 * whose  # the other player's margin is the mover's, negated
        return torch.as_tensor(data["margin"][index] * sign / config.scale, dtype=torch.float32)

    test_whose = np.tile([0, 1], len(test))
    test_index = np.repeat(test, 2)
    steps = config.epochs * (2 * len(train) // config.batch_size)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve = []
    for epoch in range(config.epochs):
        net.train()
        for _ in range(2 * len(train) // config.batch_size):
            index = rng.choice(train, size=config.batch_size)
            whose = rng.integers(0, 2, config.batch_size)
            loss = (
                (net(torch.as_tensor(rows(data, index, whose))) - target(index, whose)) ** 2
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            schedule.step()
        net.eval()
        with torch.no_grad():
            errors = []
            for k in range(0, len(test_index), 65536):
                part = slice(k, k + 65536)
                x = torch.as_tensor(rows(data, test_index[part], test_whose[part]))
                errors.append(net(x) - target(test_index[part], test_whose[part]))
            mse = float((torch.cat(errors) ** 2).mean())
        variance = float(np.var(data["margin"][test] / config.scale))
        curve.append(
            {
                "epoch": epoch + 1,
                "rmse": round(mse**0.5 * config.scale, 3),
                "r2": round(1 - mse / variance, 4),
            }
        )
    return net, curve


def default_belief_config() -> PublicBeliefConfig:
    """The belief settings the value data is collected with (and play must match)."""
    return PublicBeliefConfig(particles=48, temperature=1.0)
