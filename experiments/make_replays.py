"""Generate the replay bundle (``replays.js``) that the cross-game viewer loads.

Records one Ventures game (PIMC vs the heuristic) and one Toy Showdown game, and writes them as a
single ``window.REPLAYS`` bundle. Ventures' only Stage-1 home for a registered game is fine here;
Toy Showdown is the test fixture, imported from ``tests/``.

Run with an output directory (defaults to the current directory)::

    uv run python -m experiments.make_replays [OUT_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.mcts import MCTSConfig, PIMCAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.viz import record_game, write_bundle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import toy_game  # noqa: E402, F401  (registers "toy_showdown")


def build_bundle() -> dict[str, dict]:
    """Record the sample games and return the replay bundle."""
    ventures = make("ventures")
    pimc = PIMCAgent(
        MCTSConfig(num_determinizations=12, sims_per_world=80, max_rollout_depth=0), "PIMC"
    )
    ventures_replay = record_game(
        ventures,
        [pimc, VenturesHeuristic("Heuristic")],
        np.random.default_rng(7),
        ["PIMC", "Heuristic"],
    )

    toy = make("toy_showdown")
    toy_pimc = PIMCAgent(
        MCTSConfig(num_determinizations=8, sims_per_world=60, max_rollout_depth=None), "PIMC"
    )
    toy_replay = record_game(
        toy, [toy_pimc, RandomAgent("Random")], np.random.default_rng(3), ["PIMC", "Random"]
    )
    return {"ventures": ventures_replay, "toy_showdown": toy_replay}


def main(out_dir: str) -> None:
    """Build the bundle and write ``replays.js`` into ``out_dir``."""
    path = str(Path(out_dir) / "replays.js")
    write_bundle(path, build_bundle())
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
