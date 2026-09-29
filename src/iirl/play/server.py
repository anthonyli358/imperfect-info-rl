"""Play Ventures against an agent in the browser: a local session and a tiny HTTP server.

Standard library only. The engine and the agent run in Python; the page (``viz/play.html``)
receives only the human player's view: their own hand, the public board, the opponent's hand
*size* and the opponent cards that are public (taken from a discard pile). The opponent's hand is
revealed when the game ends. The runnable entry point is ``experiments/play_ventures.py``.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.base import Agent
from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.games.ventures.cards import NUM_CARD_TYPES
from iirl.games.ventures.game import DISCARD_OFFSET, DRAW_DECK_ACTION, TAKE_DISCARD_OFFSET
from iirl.games.ventures.state import Phase


def build_agent(checkpoint: str) -> Agent:
    """Return the agent for ``checkpoint``: ``"heuristic"`` or a PPO checkpoint path."""
    if checkpoint == "heuristic":
        return VenturesHeuristic("heuristic")
    return PPOAgent.from_checkpoint(checkpoint)


class Session:
    """One game between the human and an agent; every method returns the human's view."""

    def __init__(self) -> None:
        """Create an empty session (a game starts with :meth:`new`)."""
        self.game = make("ventures")
        self.lock = threading.Lock()
        self.state = None
        self.log: list[dict] = []
        self.agent: Agent | None = None
        self.agent_label = ""
        self.human = 0
        self.rng = np.random.default_rng()
        self.record = {"games": 0, "wins": 0, "draws": 0, "points": 0.0}

    def new(self, checkpoint: str, label: str, seat: str) -> dict:
        """Start a game vs ``checkpoint``; ``seat`` is ``"0"``, ``"1"`` or ``"random"``."""
        self.agent = build_agent(checkpoint)
        self.agent_label = label
        self.human = int(self.rng.integers(2)) if seat == "random" else int(seat)
        self.state = self.game.new_initial_state()
        self.log = []
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
        chance = player is None
        if not chance:
            text = self._describe(s, action, player)
        else:  # a deck draw, whose card only the drawing player sees
            player = s.to_move
            if s.phase == Phase.DEAL:
                text = None
            elif player == self.human:
                text = f"You drew {g.action_to_string(s, action).removeprefix('draw ')}"
            else:
                text = "Opponent drew from the deck"
        self.state = g.apply_action(s, action)
        self.agent.observe(g, self.state, None if chance else action)
        if text:
            self.log.append({"who": "you" if player == self.human else "opp", "text": text})
        if g.is_terminal(self.state):
            self._finish()

    def _describe(self, s, action: int, player: int) -> str | None:  # noqa: ANN001
        if action == DRAW_DECK_ACTION:
            return None  # logged when the draw resolves
        who = "You" if player == self.human else "Opponent"
        return f"{who}: {self.game.action_to_string(s, action)}"

    def _advance(self) -> None:
        """Resolve chance nodes and agent moves until it is the human's turn or the game ends."""
        g = self.game
        while not g.is_terminal(self.state):
            if g.is_chance_node(self.state):
                outcomes = g.chance_outcomes(self.state)
                probs = np.array([p for _, p in outcomes])
                self._apply(int(outcomes[self.rng.choice(len(outcomes), p=probs)][0]), None)
            elif g.current_player(self.state) == self.human:
                return
            else:
                self._apply(self.agent.step(g, self.state, self.rng), 1 - self.human)

    def _finish(self) -> None:
        margin = self.game.returns(self.state)[self.human]
        self.record["games"] += 1
        self.record["points"] += margin
        self.record["wins"] += margin > 0
        self.record["draws"] += margin == 0

    def view(self, error: str | None = None) -> dict:
        """Return the human's view: nothing about the opponent's hand beyond what is public."""
        g, s = self.game, self.state
        if s is None:
            return {"started": False, "record": self.record}
        me, opp = self.human, 1 - self.human
        d = g.to_display(s)
        terminal = g.is_terminal(s)
        legal = {"play": [], "discard": [], "deck": False, "take": []}
        if not terminal and g.current_player(s) == me:
            for a in g.legal_actions(s):
                if a < DISCARD_OFFSET:
                    legal["play"].append(a)
                elif a < DRAW_DECK_ACTION:
                    legal["discard"].append(a - DISCARD_OFFSET)
                elif a == DRAW_DECK_ACTION:
                    legal["deck"] = True
                else:
                    legal["take"].append(a - TAKE_DISCARD_OFFSET)
        known = [c for c in range(NUM_CARD_TYPES) for _ in range(s.known[opp][c])]
        out = {
            "started": True,
            "meta": g.display_meta(),
            "agent": self.agent_label,
            "seat": me,
            "your_turn": not terminal and g.current_player(s) == me,
            "phase": d["phase"],
            "terminal": terminal,
            "deck_size": d["deck_size"],
            "turn": s.turn,
            "hand": d["hands"][me],
            "opp_hand_size": len(d["hands"][opp]),
            "opp_known": known,
            "routes": {"you": d["routes"][me], "opp": d["routes"][opp]},
            "scores": {"you": d["scores"][me], "opp": d["scores"][opp]},
            "discards": d["discards"],
            "just_discarded_suit": d["just_discarded_suit"],
            "legal": legal,
            "log": self.log,
            "record": self.record,
        }
        if terminal:
            out["opp_hand"] = d["hands"][opp]
            out["margin"] = g.returns(s)[me]
        if error:
            out["error"] = error
        return out


def serve(
    page: Path,
    agents: list[tuple[str, str]],
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve ``page`` and the game API until interrupted.

    Args:
        page: The HTML page to serve at ``/``.
        agents: ``(label, checkpoint)`` opponents offered in the page.
        host: Bind address (keep it local).
        port: Port to serve on.
        open_browser: Open the page on start.
    """
    session = Session()

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
            elif self.path == "/api/agents":
                self._json({"agents": [{"label": a, "id": i} for i, (a, _) in enumerate(agents)]})
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
                        label, checkpoint = agents[int(body.get("agent", 0))]
                        seat = str(body.get("seat", "random"))
                        if seat not in ("0", "1", "random"):
                            raise ValueError(f"bad seat {seat!r}")
                        self._json(session.new(checkpoint, label, seat))
                    elif self.path == "/api/act":
                        self._json(session.act(int(body["action"])))
                    else:
                        self._send(b"not found", "text/plain", 404)
            except (ValueError, KeyError, IndexError, TypeError) as e:
                self._send(f"bad request: {e}".encode(), "text/plain", 400)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"Ventures: play at {url}  (Ctrl+C to stop)")
    print("opponents: " + ", ".join(label for label, _ in agents))
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
