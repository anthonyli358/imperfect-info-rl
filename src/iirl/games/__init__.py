"""From-scratch game engines. Importing this package registers the built-in games."""

# Importing each engine registers it.
# ``spiel`` registers OpenSpiel games only when the [openspiel] extra is installed.
from iirl.games import (  # noqa: F401
    camel_stocks,
    holdem,
    mini_ventures,
    spendthrift,
    spiel,
    tessera,
    ventures,
)

__all__ = ["camel_stocks", "holdem", "mini_ventures", "spendthrift", "spiel", "tessera", "ventures"]
