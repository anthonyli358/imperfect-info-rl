"""Generate the replay bundle (``replays.js``) that the cross-game viewer loads.

Records a few Ventures matchups (the self-play PPO agent vs the heuristic and vs PIMC, and PIMC vs
the heuristic) plus one Toy Showdown game, and writes them as a single ``window.REPLAYS`` bundle.
The PPO games need a trained checkpoint (``runs/ppo_ventures.pt``, from
``experiments/train_ventures_ppo.py``) and are skipped if it is missing. Toy Showdown is the test
fixture, imported from ``tests/``.

Run with an output directory (defaults to ``viz/``, next to the viewer)::

    uv run python -m experiments.make_replays [OUT_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from iirl.agents.base import Agent
from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.mcts import MCTSConfig, PIMCAgent
from iirl.agents.ppo import PPOAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.game import Game
from iirl.core.registry import make
from iirl.viz import record_game, write_bundle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import toy_game  # noqa: E402, F401  (registers "toy_showdown")

PPO_CHECKPOINT = ROOT / "runs" / "ppo_ventures.pt"


def _matchup(game: Game, agents: list[Agent], names: list[str], seed: int) -> dict:
    """Record one game and label it with the matchup, so the viewer can list several per game."""
    replay = record_game(game, agents, np.random.default_rng(seed), names)
    replay["long_name"] = f"{replay['long_name']} · {names[0]} vs {names[1]}"
    return replay


def build_bundle() -> dict[str, dict]:
    """Record the sample games and return the replay bundle."""
    ventures = make("ventures")
    pimc = PIMCAgent(
        MCTSConfig(num_determinizations=12, sims_per_world=80, max_rollout_depth=0), "PIMC"
    )
    heuristic = VenturesHeuristic("Heuristic")
    bundle: dict[str, dict] = {}
    if PPO_CHECKPOINT.exists():
        ppo = PPOAgent.from_checkpoint(str(PPO_CHECKPOINT), "PPO")
        bundle["ventures_ppo_heuristic"] = _matchup(
            ventures, [ppo, heuristic], ["PPO", "Heuristic"], 7
        )
        bundle["ventures_ppo_pimc"] = _matchup(ventures, [ppo, pimc], ["PPO", "PIMC"], 11)
    else:
        print(f"no PPO checkpoint at {PPO_CHECKPOINT}; skipping the PPO replays")
    bundle["ventures_pimc_heuristic"] = _matchup(
        ventures, [pimc, heuristic], ["PIMC", "Heuristic"], 7
    )

    toy = make("toy_showdown")
    toy_pimc = PIMCAgent(
        MCTSConfig(num_determinizations=8, sims_per_world=60, max_rollout_depth=None), "PIMC"
    )
    bundle["toy_showdown"] = record_game(
        toy, [toy_pimc, RandomAgent("Random")], np.random.default_rng(3), ["PIMC", "Random"]
    )
    return bundle


def main(out_dir: str) -> None:
    """Build the bundle and write ``replays.js`` into ``out_dir``."""
    path = str(Path(out_dir) / "replays.js")
    write_bundle(path, build_bundle())
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "viz"))
