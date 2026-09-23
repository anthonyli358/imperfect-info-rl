"""Game registry / factory.

Games register a zero-arg (or keyword-only) factory under a short name; ``make`` builds one by name.
This is what lets the runner, tournament, and conformance suite iterate over "every registered game"
without importing each engine directly.
"""

from __future__ import annotations

from collections.abc import Callable

from iirl.core.game import Game

_REGISTRY: dict[str, Callable[..., Game]] = {}


def register(short_name: str) -> Callable[[Callable[..., Game]], Callable[..., Game]]:
    """Decorator registering a ``Game`` factory (a class or function) under ``short_name``.

    Args:
        short_name: Unique registry key.

    Returns:
        The decorator, which returns its argument unchanged.

    Raises:
        ValueError: If ``short_name`` is already registered.
    """

    def decorator(factory: Callable[..., Game]) -> Callable[..., Game]:
        if short_name in _REGISTRY:
            raise ValueError(f"Game {short_name!r} is already registered")
        _REGISTRY[short_name] = factory
        return factory

    return decorator


def make(short_name: str, **kwargs: object) -> Game:
    """Construct a registered game by name.

    Args:
        short_name: The registry key.
        **kwargs: Forwarded to the registered factory.

    Returns:
        A fresh ``Game`` instance.

    Raises:
        KeyError: If ``short_name`` is not registered.
    """
    try:
        factory = _REGISTRY[short_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown game {short_name!r}; registered: {known}") from None
    return factory(**kwargs)


def registered_games() -> list[str]:
    """Return the sorted names of all registered games."""
    return sorted(_REGISTRY)
