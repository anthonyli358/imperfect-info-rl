"""Game-specific heuristic baselines (re-exported; each lives in its game's package).

Game-agnostic code should use ``iirl.core.registry.make_heuristic(game_name)`` instead.
"""

from iirl.games.camel_stocks.heuristic import CamelStocksHeuristic
from iirl.games.ventures.heuristic import VenturesHeuristic

__all__ = ["CamelStocksHeuristic", "VenturesHeuristic"]
