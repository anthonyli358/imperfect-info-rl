"""Core: the game-agnostic ``Game``/``State`` interface, registry, and episode runner."""

from iirl.core.game import Game, State
from iirl.core.registry import make, register, registered_games
from iirl.core.runner import EpisodeResult, run_episode
from iirl.core.types import (
    Action,
    ChanceOutcome,
    GameType,
    Information,
    Player,
    PlayerId,
    Utility,
)

__all__ = [
    "Action",
    "ChanceOutcome",
    "EpisodeResult",
    "Game",
    "GameType",
    "Information",
    "Player",
    "PlayerId",
    "State",
    "Utility",
    "make",
    "register",
    "registered_games",
    "run_episode",
]
