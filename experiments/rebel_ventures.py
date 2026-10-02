"""ReBeL on full Ventures (plan step E1), in stages.

Exact public belief states are out of reach on Ventures (a hidden hand has ~10^8 possibilities),
so this uses the approximations priced on Mini-Ventures (``experiments/mini_rebel.py``): per-card
beliefs, a network that values one hand at a time, and candidate hands in place of all hands.

Stages (run in order; each saves its output under ``runs/rebel_v/``)::

    uv run python -m experiments.rebel_ventures data     # blueprint self-play + tracked beliefs
    uv run python -m experiments.rebel_ventures value    # the belief-conditioned value network
    uv run python -m experiments.rebel_ventures match [name key=value ...]
                                                         # the agent vs the blueprint, paired deals

``match`` takes a name for the result file and overrides of the agent's settings, e.g.
``match sampled greedy=False`` or ``match deep iterations=40``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.ppo.agent import PPOAgent
from iirl.agents.rebel import rebel_agent
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.selfplay.rebel_value import ValueConfig, collect, default_belief_config, fit


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    game: str = "ventures"
    blueprint: str = "runs/diverse_sym.pt"
    out: str = "runs/rebel_v"
    games: int = 8000
    games_per_shard: int = 100
    workers: int = 10
    value: ValueConfig = field(default_factory=ValueConfig)
    deals: int = 100


def data(config: Config) -> None:
    """Collect positions in shards (existing shards are kept, so the stage can be resumed)."""
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    belief = default_belief_config()
    jobs = []
    for shard in range(config.games // config.games_per_shard):
        path = out / f"data_{shard:04d}.npz"
        if not path.exists():
            seeds = list(
                range(shard * config.games_per_shard, (shard + 1) * config.games_per_shard)
            )
            jobs.append((path, (config.game, config.blueprint, seeds, belief)))
    print(f"{len(jobs)} shards of {config.games_per_shard} games to collect", flush=True)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(config.workers) as pool:
        results = pool.imap(collect, [job for _, job in jobs])
        for k, ((path, _), shard) in enumerate(zip(jobs, results, strict=True)):
            np.savez_compressed(path, **shard)
            if (k + 1) % 10 == 0:
                rate = (k + 1) * config.games_per_shard / (time.perf_counter() - t0)
                print(f"{k + 1} shards, {rate:.1f} games/s", flush=True)


def load(config: Config) -> dict[str, np.ndarray]:
    """All collected shards, concatenated."""
    shards = [np.load(p) for p in sorted(Path(config.out).glob("data_*.npz"))]
    return {k: np.concatenate([s[k] for s in shards]) for k in shards[0].files}


def value(config: Config) -> None:
    """Train the value network, and the same network without beliefs for comparison."""
    positions = load(config)
    n_games = len(np.unique(positions["game"]))
    spread = float(positions["margin"].std())
    print(f"{len(positions['margin']):,} positions from {n_games:,} games; margin sd {spread:.1f}")
    report = {}
    for name, use in (("with_beliefs", True), ("without_beliefs", False)):
        cfg = ValueConfig(**{**config.value.__dict__, "use_beliefs": use})
        t0 = time.perf_counter()
        net, curve = fit(positions, cfg)
        report[name] = curve
        print(name, curve[-1], f"{time.perf_counter() - t0:.0f}s", flush=True)
        if use:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "n_public": net.n_public,
                    "n_cards": net.n_cards,
                    "hidden": cfg.hidden,
                    "layers": cfg.layers,
                    "scale": cfg.scale,
                },
                Path(config.out) / "value.pt",
            )
    Path(config.out, "value.json").write_text(json.dumps(report, indent=1), "utf-8")


def blueprint_agent(path: str) -> PPOAgent:
    """The blueprint as deployed: greedy."""
    return PPOAgent.from_checkpoint(path, greedy=True)


def match(config: Config) -> None:
    """The ReBeL agent against its own blueprint on paired deals."""
    name = sys.argv[2] if len(sys.argv) > 2 else "default"
    settings: dict[str, object] = {
        "blueprint": config.blueprint,
        "value": str(Path(config.out) / "value.pt"),
    }
    for item in sys.argv[3:]:
        key, text = item.split("=")
        settings[key] = json.loads(text.lower()) if key != "value" else text
    t0 = time.perf_counter()
    result = duplicate_match(
        make(config.game),
        (rebel_agent, settings),
        (blueprint_agent, {"path": config.blueprint}),
        deals=config.deals,
        seed=2026,
        workers=config.workers,
    )
    row = {
        "name": name,
        "settings": {k: v for k, v in settings.items() if k not in ("blueprint", "value")},
        "deals": result.deals,
        "score": round(result.score, 4),
        "score_se": round(result.score_se, 4),
        "margin": round(result.margin, 2),
        "margin_se": round(result.margin_se, 2),
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    print(json.dumps(row), flush=True)
    Path(config.out, f"match_{name}.json").write_text(json.dumps(row, indent=1), "utf-8")


if __name__ == "__main__":
    {"data": data, "value": value, "match": match}[sys.argv[1]](Config())
