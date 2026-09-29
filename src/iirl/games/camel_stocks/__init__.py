"""Camel Stocks: an original two-player trading card game (see ``cards.py`` for the credit note)."""

from iirl.games.camel_stocks.game import NUM_ACTIONS, CamelStocks
from iirl.games.camel_stocks.heuristic import CamelStocksHeuristic
from iirl.games.camel_stocks.state import CamelStocksState, Phase

__all__ = ["CamelStocksHeuristic", "NUM_ACTIONS", "CamelStocks", "CamelStocksState", "Phase"]
