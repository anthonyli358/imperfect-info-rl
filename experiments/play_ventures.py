"""Play Ventures against a trained agent in the browser (local only).

Starts a small standard-library HTTP server on ``127.0.0.1`` (``iirl.play``) and opens
``viz/play.html``. Only your view is sent to the page; the opponent's hand is revealed when the
game ends.

Run with::

    uv run python -m experiments.play_ventures

then play at http://127.0.0.1:8765 (it opens automatically). Stop with Ctrl+C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iirl.play import serve

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Server settings.

    Attributes:
        host: Bind address (keep it local).
        port: Port to serve on.
        agents: ``(label, checkpoint)`` opponents offered in the page, strongest first; a
            checkpoint of ``"heuristic"`` is the hand-tuned agent. Missing files are skipped.
        open_browser: Open the page on start.
    """

    host: str = "127.0.0.1"
    port: int = 8765
    agents: tuple[tuple[str, str], ...] = (
        ("Nash PPO, stall fix (nash_g)", "runs/nash_g.pt"),
        ("Nash PPO (league_control), 1973 Elo", "runs/league_control.pt"),
        ("Nash PPO (nash_long), 1941 Elo", "runs/nash_long.pt"),
        ("PPO v2, 1822 Elo", "runs/ppo_ventures_best.pt"),
        ("Heuristic, 1676 Elo", "heuristic"),
    )
    open_browser: bool = True


def main(config: Config) -> None:
    """Serve the play page with the opponents whose checkpoints exist."""
    agents = [
        (label, p if p == "heuristic" else str(ROOT / p))
        for label, p in config.agents
        if p == "heuristic" or (ROOT / p).exists()
    ]
    serve(ROOT / "viz" / "play.html", agents, config.host, config.port, config.open_browser)


if __name__ == "__main__":
    main(Config())
