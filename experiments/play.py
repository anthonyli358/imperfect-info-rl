"""Play Ventures or Camel Stocks against the trained agents in the browser (local only).

Starts a small standard-library HTTP server on ``127.0.0.1`` (``iirl.play``) and opens
``viz/play.html``. Only your view is sent to the page; the opponent's hand is revealed when the
game ends. Every finished game is saved to ``runs/human_games/<game>/`` (moves, chance outcomes
and result), so your games can be analysed like the agents'.

Run with::

    uv run python -m experiments.play

then play at http://127.0.0.1:8765 (it opens automatically). Stop with Ctrl+C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from iirl.play import serve

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Server settings.

    Attributes:
        host: Bind address (keep it local).
        port: Port to serve on.
        agents: Per game, ``(label, checkpoint)`` opponents offered in the page, strongest first;
            ``"heuristic"`` is the game's hand-tuned agent and other names without ``.pt`` are
            registered heuristics; ``"rollout:<checkpoint>"`` adds endgame rollouts to a
            checkpoint. Missing checkpoints are skipped.
        open_browser: Open the page on start.
        record_dir: Where finished games are saved as JSON, for analysis (``None``: off).
    """

    host: str = "127.0.0.1"
    port: int = 8765
    agents: dict[str, tuple[tuple[str, str], ...]] = field(
        default_factory=lambda: {
            "ventures": (
                (
                    "Best: diverse_sym + endgame rollouts (thinks a few seconds late on)",
                    "rollout:runs/diverse_sym.pt",
                ),
                ("Symmetry + diverse league, no search (diverse_sym)", "runs/diverse_sym.pt"),
                ("Careful-league PPO (careful)", "runs/careful.pt"),
                ("Nash PPO (league_control), 1973 Elo", "runs/league_control.pt"),
                ("Nash PPO (nash_g), ~1970 Elo", "runs/nash_g.pt"),
                ("Nash PPO (nash_long), 1941 Elo", "runs/nash_long.pt"),
                ("PPO v2, 1822 Elo", "runs/ppo_ventures_best.pt"),
                ("Heuristic, 1676 Elo", "heuristic"),
                ("Careful heuristic (never gifts)", "ventures_careful"),
            ),
            "camel_stocks": (
                (
                    "Best: camel_nash + endgame rollouts (thinks ~1 s late on)",
                    "rollout:runs/camel_nash.pt",
                ),
                ("Nash PPO (camel_nash), ~1970 Elo, no search", "runs/camel_nash.pt"),
                ("PPO (camel_ppo), ~1890 Elo", "runs/camel_ppo.pt"),
                ("Heuristic, 1500 Elo", "heuristic"),
            ),
            "tessera": (
                (
                    "Best: value search after the AlphaZero loop, 400 simulations (~0.3 s a move)",
                    "search:runs/tessera_ag/exit_prior_24.pt|runs/tessera_ag/exit_value_24.pt|400",
                ),
                ("League PPO (tessera_nash2), no search", "runs/tessera_nash2.pt"),
                ("League PPO (tessera_nash)", "runs/tessera_nash.pt"),
                ("Heuristic, 1500 Elo", "heuristic"),
            ),
        }
    )
    open_browser: bool = True
    record_dir: str | None = "runs/human_games"


def main(config: Config) -> None:
    """Serve the play page with the opponents whose checkpoints exist."""

    def resolve(spec: str) -> str | None:
        """The spec with its checkpoint made absolute, or None if the checkpoint is missing."""
        if spec.startswith("search:"):
            policy, value, sims = spec[len("search:") :].split("|")
            if not ((ROOT / policy).exists() and (ROOT / value).exists()):
                return None
            return f"search:{ROOT / policy}|{ROOT / value}|{sims}"
        prefix = "rollout:" if spec.startswith("rollout:") else ""
        path = spec[len(prefix) :]
        if not path.endswith(".pt"):
            return spec
        return prefix + str(ROOT / path) if (ROOT / path).exists() else None

    agents = {
        game: [(label, resolve(p)) for label, p in opponents if resolve(p) is not None]
        for game, opponents in config.agents.items()
    }
    record = ROOT / config.record_dir if config.record_dir else None
    serve(ROOT / "viz" / "play.html", agents, config.host, config.port, config.open_browser, record)


if __name__ == "__main__":
    main(Config())
