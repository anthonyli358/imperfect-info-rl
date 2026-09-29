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


# --- Per-game heuristic agents ------------------------------------------------------------------
# A game may register a hand-tuned agent alongside its engine: the Elo anchor, a fixed league
# member, and the default opponent in the play UI. Game-agnostic code asks for it by game name, so
# adding a game never touches ``agents/``.
_HEURISTICS: dict[str, Callable[[], object]] = {}


def register_heuristic(short_name: str) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """Decorator registering a zero-arg factory for ``short_name``'s heuristic agent.

    Raises:
        ValueError: If ``short_name`` already has a heuristic.
    """

    def decorator(factory: Callable[[], object]) -> Callable[[], object]:
        if short_name in _HEURISTICS:
            raise ValueError(f"Game {short_name!r} already has a heuristic")
        _HEURISTICS[short_name] = factory
        return factory

    return decorator


def make_heuristic(short_name: str):  # noqa: ANN201 - an Agent (not imported: core <- agents)
    """Construct the heuristic agent registered for ``short_name``.

    Raises:
        KeyError: If the game has no registered heuristic.
    """
    try:
        return _HEURISTICS[short_name]()
    except KeyError:
        known = ", ".join(sorted(_HEURISTICS)) or "<none>"
        raise KeyError(f"No heuristic for {short_name!r}; available: {known}") from None
