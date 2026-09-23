"""From-scratch game engines. Importing this package registers the built-in games."""

from iirl.games import ventures  # noqa: F401  (import registers "ventures")

__all__ = ["ventures"]
