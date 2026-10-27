"""From-scratch game engines. Importing this package registers the built-in games."""

# Importing each engine registers it.
# ``spiel`` registers OpenSpiel games only when the [openspiel] extra is installed.
from iirl.games import camel_stocks, mini_ventures, spiel, tessera, ventures  # noqa: F401

__all__ = ["camel_stocks", "mini_ventures", "spiel", "tessera", "ventures"]
