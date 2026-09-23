"""Record a played game into a JSON replay for the cross-game viewer.

The recorder is game-agnostic: it drives the ``Game`` interface and calls each game's ``to_display``
and ``display_meta`` hooks, so any game that implements them can be watched in the same viewer with
no viewer changes. A replay is a plain dict (JSON-serializable):

    {
      "game", "long_name", "num_players", "players": [name, ...],
      "meta": {...display_meta...},
      "plies": [{"i", "actor", "label", "display": {...}}, ...],
      "result": {"returns": [...]},
    }

``actor`` is the player id that acted to *produce* this ply (-1 for chance, and -1 on the initial
frame). ``plies[0]`` is the initial state before any action.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from iirl.agents.base import Agent
from iirl.core.game import Game
from iirl.core.runner import _sample_chance
from iirl.core.types import PlayerId


def record_game(
    game: Game,
    agents: Sequence[Agent],
    rng: np.random.Generator,
    player_names: Sequence[str] | None = None,
    max_steps: int = 10_000,
) -> dict:
    """Play one game and return its JSON-serializable replay dict."""
    names = list(player_names) if player_names else [a.name for a in agents]
    state = game.new_initial_state()
    plies = [{"i": 0, "actor": -1, "label": "start", "display": game.to_display(state)}]

    for i in range(1, max_steps + 1):
        if game.is_terminal(state):
            break
        actor = int(game.current_player(state))
        if game.is_chance_node(state):
            action = _sample_chance(game, state, rng)
        else:
            action = agents[actor].step(game, state, rng)
        label = game.action_to_string(state, action)
        actor_out = -1 if actor == PlayerId.CHANCE else actor
        state = game.apply_action(state, action)
        plies.append(
            {"i": i, "actor": actor_out, "label": label, "display": game.to_display(state)}
        )

    return {
        "game": game.game_type.short_name,
        "long_name": game.game_type.long_name,
        "num_players": game.num_players,
        "players": names,
        "meta": game.display_meta(),
        "plies": plies,
        "result": {"returns": game.returns(state)},
    }


def write_bundle(path: str, replays: dict[str, dict]) -> None:
    """Write ``replays`` (keyed by id) as a ``window.REPLAYS`` assignment in a ``.js`` file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.REPLAYS = ")
        json.dump(replays, f, separators=(",", ":"))
        f.write(";\n")
