"""ReBeL on Mini-Ventures, checked against exact exploitability.

Trains ``iirl.selfplay.rebel`` (CFR over one-turn subgames of public belief states, with a value
network at each turn boundary) by self-play, and every ``eval_every`` epochs measures the exact
exploitability of the policy it plays. References on the same game (``runs/mini/``): the
equilibrium from full-game CFR+ is 0.002 points/game, Nash-seeking PPO 0.07, random play 2.71.

The evaluated policy is ReBeL's expectation over its random stopping iterations. Branching from
every iteration of every turn is exact but far too slow here (the last turn's subgames take
~0.4 s per CFR iteration), so it branches from a weighted sample of iterations
(``eval_samples``): a mixture of fewer strategies, hence an *upper* bound on the exploitability
of the full expectation in practice.

Progress is saved after every epoch (``runs/mini/rebel.pt``; the curve in
``runs/mini/rebel.json``), so the run can be stopped and resumed.

Variants test, against the same ground truth, the approximations full Ventures will need
(its hidden hand has ~10^8 possibilities): ``factored`` shows the network only per-card belief
marginals; ``factored_hand`` also makes it return one hand's value at a time.

Run with::

    uv run python -m experiments.mini_rebel [factored | factored_hand]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

import iirl.games  # noqa: F401  (registers "mini_ventures")
from iirl.core.registry import make
from iirl.selfplay.rebel import FullWidthReBeL, ReBeLConfig


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    rebel: ReBeLConfig = field(
        default_factory=lambda: ReBeLConfig(
            boundaries=(4, 7, 10, 13, 16),  # the start of each of the four turns, and the end
            cfr_iterations=64,
            eval_iterations=64,
            eval_samples=(8, 4, 16),
            epochs=40,
            passes=3,
        )
    )
    eval_every: int = 10
    state: str = "runs/mini/rebel.pt"
    curve: str = "runs/mini/rebel.json"


def main(config: Config) -> None:
    """Train (resuming if a saved state exists), evaluating exactly every few epochs."""
    game = make("mini_ventures")
    t0 = time.perf_counter()
    rebel = FullWidthReBeL(game, config.rebel)
    sizes = {d: level.num_public for d, level in rebel.levels.items()}
    print(f"indexed in {time.perf_counter() - t0:.0f}s; public states per boundary {sizes}")
    print(f"range bookkeeping error under a random policy: {rebel.range_spread():.1e}", flush=True)
    state, curve = Path(config.state), Path(config.curve)
    history = json.loads(curve.read_text("utf-8")) if curve.exists() else []
    if state.exists():
        saved = torch.load(state, weights_only=False)
        for d, net in rebel.nets.items():
            net.load_state_dict(saved["nets"][d])
            rebel.optimizers[d].load_state_dict(saved["optimizers"][d])
        rebel.epoch = saved["epoch"]
        print(f"resumed at epoch {rebel.epoch} (replay buffers refill from here)", flush=True)
    last = time.perf_counter()

    def on_epoch(rebel: FullWidthReBeL, losses: dict[int, float]) -> None:
        nonlocal last
        row = {"epoch": rebel.epoch, "loss": {str(d): round(v, 5) for d, v in losses.items()}}
        row["seconds"] = round(time.perf_counter() - last)
        if rebel.epoch % config.eval_every == 0 or rebel.epoch == config.rebel.epochs:
            t = time.perf_counter()
            row.update(rebel.evaluate())
            row["eval_seconds"] = round(time.perf_counter() - t)
        print(json.dumps(row), flush=True)
        history.append(row)
        state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": rebel.epoch,
                "nets": {d: net.state_dict() for d, net in rebel.nets.items()},
                "optimizers": {d: opt.state_dict() for d, opt in rebel.optimizers.items()},
            },
            state,
        )
        curve.write_text(json.dumps(history, indent=1), "utf-8")
        last = time.perf_counter()

    rebel.train(on_epoch)


def variant(name: str) -> Config:
    """The config for a named variant (results go to ``runs/mini/rebel_<name>.*``)."""
    base = Config()
    rebel = replace(base.rebel, factored_beliefs=True)
    if name == "factored_hand":  # one network row per hand: slower, so branch from fewer solves
        rebel = replace(rebel, per_hand_values=True, eval_samples=(8, 2, 16))
    elif name != "factored":
        raise SystemExit(f"unknown variant {name!r}")
    stem = f"runs/mini/rebel_{name}"
    return replace(base, rebel=rebel, state=f"{stem}.pt", curve=f"{stem}.json")


if __name__ == "__main__":
    main(variant(sys.argv[1]) if len(sys.argv) > 1 else Config())
