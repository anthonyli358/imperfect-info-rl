"""The entity network: layout, piece equivariance, checkpoints, and PPO training it."""

from dataclasses import replace

import numpy as np
import pytest
import torch

import iirl.games  # noqa: F401
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make, make_heuristic
from iirl.nn.entity import EntityPolicyValueNet, save_checkpoint
from iirl.selfplay import PPOConfig
from iirl.selfplay.ppo import train_ppo

SPEC = {"tokens": 18, "token_dim": 25, "own_tokens": 9, "regions": 14}


@pytest.mark.parametrize("region_tokens", [0, 1])
def test_layout_and_piece_equivariance(region_tokens: int) -> None:
    torch.manual_seed(0)
    net = EntityPolicyValueNet(551, 144, **SPEC, region_tokens=region_tokens).eval()
    x = torch.randn(3, 551)
    rows = x[:, : 18 * 25].reshape(3, 18, 25)
    rows[:, :, 1:16] = torch.nn.functional.one_hot(torch.randint(15, (3, 18)), 15).float()
    logits, value = net(x)
    assert logits.shape == (3, 144) and value.shape == (3,)
    # Swapping two of the player's pieces swaps their move logits (the slot embeddings start at 0).
    y = x.clone().reshape(3, -1)
    rows = y[:, : 18 * 25].reshape(3, 18, 25)
    rows[:, [2, 5]] = rows[:, [5, 2]]
    swapped, _ = net(y)
    moves, moves_swapped = logits[:, :126].reshape(3, 9, 14), swapped[:, :126].reshape(3, 9, 14)
    assert torch.allclose(moves[:, 2], moves_swapped[:, 5], atol=1e-5)
    assert torch.allclose(logits[:, 126:], swapped[:, 126:], atol=1e-5)


@pytest.mark.parametrize("region_tokens", [0, 1])
def test_checkpoint_round_trip(tmp_path, region_tokens: int) -> None:  # noqa: ANN001
    net = EntityPolicyValueNet(551, 144, **SPEC, region_tokens=region_tokens)
    save_checkpoint(net, tmp_path / "e.pt")
    loaded = PPOAgent.from_checkpoint(str(tmp_path / "e.pt")).net
    x = torch.randn(2, 551)
    assert isinstance(loaded, EntityPolicyValueNet)
    assert torch.allclose(net.eval()(x)[0], loaded.eval()(x)[0])


def test_ppo_trains_the_entity_network() -> None:
    game = make("undertow")
    config = PPOConfig(
        iterations=2,
        games_per_iter=16,
        num_envs=8,
        hidden=(32, 32),
        arch="entity",
        entity=tuple(SPEC.items()),
        reward_scale=1.0,
        eval_interval=2,
        eval_games=4,
        feature_sample_games=4,
        anchor_elos={"heuristic": 1500.0},
    )
    net, history = train_ppo(game, {"heuristic": make_heuristic("undertow")}, config)
    assert isinstance(net, EntityPolicyValueNet) and history
    agent = PPOAgent(net)
    s = game.new_initial_state()
    assert agent.step(game, s, np.random.default_rng(0)) in game.legal_actions(s)
    assert replace(config, arch="mlp").arch == "mlp"
