"""From-scratch game engines. Importing this package registers the built-in games."""

# Importing each engine registers it.
from iirl.games import camel_stocks, mini_ventures, tessera, ventures  # noqa: F401

__all__ = ["camel_stocks", "mini_ventures", "tessera", "ventures"]
