"""Play any registered game against an agent in the browser: a local session and a tiny server.

Standard library only. The engine and the agent run in Python; the page (``viz/play.html``)
receives only the human player's view (``Game.player_view``) plus the legal moves, labelled by
``Game.action_to_string``. Nothing hidden from the human is sent until the game ends, when the full
position (``Game.to_display``) is revealed. The runnable entry point is ``experiments/play.py``.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import iirl.games  # noqa: F401  (registers the built-in games)
from iirl.agents.base import Agent
from iirl.agents.ppo import PPOAgent
from iirl.core.game import Game, State
from iirl.core.registry import make, make_heuristic


def build_agent(game_name: str, checkpoint: str) -> Agent:
    """Return the agent for ``checkpoint``.

    ``"heuristic"`` is the game's own heuristic, another name without ``.pt`` is a registered
    heuristic (e.g. ``"ventures_careful"``), and a ``.pt`` path is a PPO checkpoint.
    """
    if checkpoint == "heuristic":
        return make_heuristic(game_name)
    if not checkpoint.endswith(".pt"):
        return make_heuristic(checkpoint)
    return PPOAgent.from_checkpoint(checkpoint)


def _chance_visible(game: Game, state: State, player: int) -> bool:
    """Whether ``player`` can tell a chance event's outcomes apart (e.g. their own draw)."""
    outcomes = [a for a, _ in game.chance_outcomes(state)]
    if len(outcomes) < 2:
        return True
    views = {game.information_state_string(game.apply_action(state, a), player) for a in outcomes}
    return len(views) > 1


class Session:
    """One game between the human and an agent; every method returns the human's view."""

    def __init__(self, record_dir: Path | None = None) -> None:
        """Create an empty session (a game starts with :meth:`new`).

        With ``record_dir``, every finished game is saved there as JSON (see :meth:`_save`).
        """
        self.record_dir = record_dir
        self.history: list[dict] = []
        self.checkpoint = ""
        self.started_at = ""
        self.lock = threading.Lock()
        self.game_name = ""
        self.game: Game | None = None
        self.state: State | None = None
        self.log: list[dict] = []
        self.agent: Agent | None = None
        self.agent_label = ""
        self.human = 0
        self.dealt = False
        self.rng = np.random.default_rng()
        self.records: dict[str, dict] = {}

    def new(self, game_name: str, checkpoint: str, label: str, seat: str) -> dict:
        """Start a game of ``game_name`` vs ``checkpoint``.

        ``seat`` is ``"0"``, ``"1"`` or ``"random"``.
        """
        self.game_name, self.game = game_name, make(game_name)
        self.agent = build_agent(game_name, checkpoint)
        self.agent_label, self.checkpoint = label, checkpoint
        self.history, self.started_at = [], time.strftime("%Y-%m-%dT%H:%M:%S")
        self.human = int(self.rng.integers(2)) if seat == "random" else int(seat)
        self.state = self.game.new_initial_state()
        self.log, self.dealt = [], False
        self.records.setdefault(game_name, {"games": 0, "wins": 0, "draws": 0, "points": 0.0})
        self.agent.reset(self.game, 1 - self.human)
        self._advance()
        return self.view()

    def act(self, action: int) -> dict:
        """Apply the human's ``action`` (if legal), then let chance and the agent move."""
        g, s = self.game, self.state
        if s is None or g.is_terminal(s) or g.current_player(s) != self.human:
            return self.view(error="Not your turn.")
        if action not in g.legal_actions(s):
            return self.view(error="That move is not legal.")
        self._apply(action, self.human)
        self._advance()
        return self.view()

    def _apply(self, action: int, player: int | None) -> None:
        """Apply ``action`` by ``player`` (``None``: a chance event) and log what the human sees."""
        g, s = self.game, self.state
        if player is not None:
            who = "you" if player == self.human else "opp"
            label = f"{'You' if who == 'you' else 'Opponent'}: {g.action_to_string(s, action)}"
            entry = {"who": who, "text": label}
        elif not self.dealt:
            entry = None  # the opening deal is not logged
        elif _chance_visible(g, s, self.human):
            entry = {"who": "chance", "text": g.action_to_string(s, action)}
        else:
            entry = {"who": "chance", "text": "a hidden card is drawn"}
        self.history.append(
            {
                "player": "chance"
                if player is None
                else ("human" if player == self.human else "agent"),
                "action": int(action),
                "label": g.action_to_string(s, action),
            }
        )
        self.state = g.apply_action(s, action)
        self.agent.observe(g, self.state, None if player is None else action)
        if entry:
            self.log.append(entry)
        if g.is_terminal(self.state):
            self._finish()

    def _advance(self) -> None:
        """Resolve chance nodes and agent moves until it is the human's turn or the game ends."""
        g = self.game
        while not g.is_terminal(self.state):
            if g.is_chance_node(self.state):
                outcomes = g.chance_outcomes(self.state)
                probs = np.array([p for _, p in outcomes])
                self._apply(int(outcomes[self.rng.choice(len(outcomes), p=probs)][0]), None)
                continue
            self.dealt = True
            if g.current_player(self.state) == self.human:
                return
            self._apply(self.agent.step(g, self.state, self.rng), 1 - self.human)

    def _finish(self) -> None:
        margin = self.game.returns(self.state)[self.human]
        record = self.records[self.game_name]
        record["games"] += 1
        record["points"] += margin
        record["wins"] += margin > 0
        record["draws"] += margin == 0
        self._save(margin)

    def _save(self, margin: float) -> None:
        """Write the finished game to ``record_dir/<game>/<time>.json``.

        The record holds every action in order, chance outcomes included, so the game can be
        replayed exactly (apply the actions to ``new_initial_state``) and analysed like any other.
        """
        if self.record_dir is None:
            return
        g, s = self.game, self.state
        folder = self.record_dir / self.game_name
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "game": self.game_name,
            "opponent": self.agent_label,
            "checkpoint": self.checkpoint,
            "human_seat": self.human,
            "started_at": self.started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "margin": margin,
            "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
            "returns": g.returns(s),
            "final": g.to_display(s),
            "actions": self.history,
        }
        name = self.started_at.replace(":", "") + ".json"
        (folder / name).write_text(json.dumps(record, indent=1), "utf-8")

    def view(self, error: str | None = None) -> dict:
        """Return the human's view: ``player_view`` until the game ends, then the full position."""
        g, s = self.game, self.state
        if s is None:
            return {"started": False}
        terminal = g.is_terminal(s)
        your_turn = not terminal and g.current_player(s) == self.human
        out = {
            "started": True,
            "game": self.game_name,
            "meta": g.display_meta(),
            "agent": self.agent_label,
            "seat": self.human,
            "your_turn": your_turn,
            "terminal": terminal,
            "view": g.to_display(s) if terminal else g.player_view(s, self.human),
            "legal": [{"id": a, "label": g.action_to_string(s, a)} for a in g.legal_actions(s)]
            if your_turn
            else [],
            "log": self.log,
            "record": self.records[self.game_name],
        }
        if terminal:
            out["margin"] = g.returns(s)[self.human]
        if error:
            out["error"] = error
        return out


def serve(
    page: Path,
    agents: Mapping[str, Sequence[tuple[str, str]]],
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    record_dir: Path | None = None,
) -> None:
    """Serve ``page`` and the game API until interrupted.

    Args:
        page: The HTML page to serve at ``/``.
        agents: Per game name, the ``(label, checkpoint)`` opponents offered in the page.
        host: Bind address (keep it local).
        port: Port to serve on.
        open_browser: Open the page on start.
        record_dir: Save every finished game here as JSON (``None``: don't record).
    """
    session = Session(record_dir)
    games = {
        name: {"label": make(name).game_type.long_name, "agents": [a for a, _ in opponents]}
        for name, opponents in agents.items()
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # noqa: ANN002 - quiet access log
            pass

        def _send(self, body: bytes, kind: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict) -> None:
            self._send(json.dumps(data).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path in ("/", "/index.html"):
                self._send(page.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/games":
                self._json({"games": games})
            elif self.path == "/api/state":
                with session.lock:
                    self._json(session.view())
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                with session.lock:
                    if self.path == "/api/new":
                        name = str(body.get("game", next(iter(agents))))
                        label, checkpoint = agents[name][int(body.get("agent", 0))]
                        seat = str(body.get("seat", "random"))
                        if seat not in ("0", "1", "random"):
                            raise ValueError(f"bad seat {seat!r}")
                        self._json(session.new(name, checkpoint, label, seat))
                    elif self.path == "/api/act":
                        self._json(session.act(int(body["action"])))
                    else:
                        self._send(b"not found", "text/plain", 404)
            except (ValueError, KeyError, IndexError, TypeError) as e:
                self._send(f"bad request: {e}".encode(), "text/plain", 400)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"play at {url}  (Ctrl+C to stop)")
    for name, opponents in agents.items():
        print(f"{name}: " + ", ".join(label for label, _ in opponents))
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
