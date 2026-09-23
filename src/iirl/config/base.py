"""Config dataclasses.

The repo convention is *config over CLI*: module-level dataclasses hold parameters (no argparse
boilerplate), and each entry point exposes a minimal ``if __name__ == "__main__"`` block that builds
a config and runs. These base configs cover Stage 0's episode running and tournaments; later stages
add their own (training, MCTS, CFR) alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

from iirl.eval.tournament import DEFAULT_ELO_K


@dataclass(frozen=True)
class RunnerConfig:
    """Parameters for running episodes.

    Attributes:
        game: Registered game short name.
        seed: RNG seed for reproducibility.
        num_episodes: Episodes to run.
    """

    game: str = "toy_showdown"
    seed: int = 0
    num_episodes: int = 1


@dataclass(frozen=True)
class TournamentConfig:
    """Parameters for a round-robin tournament.

    Attributes:
        game: Registered game short name.
        seed: RNG seed for reproducibility.
        games_per_pair: Games per unordered pair of agents.
        elo_k: Elo K-factor.
    """

    game: str = "toy_showdown"
    seed: int = 0
    games_per_pair: int = 200
    elo_k: float = DEFAULT_ELO_K
