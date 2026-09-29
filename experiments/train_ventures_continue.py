"""Continue the plain Nash recipe from the current best agent, with the endgame-stall fix.

Starts from ``runs/league_control.pt`` (1973 Elo), trains with the Nash recipe (magnet KL, snapshot
pool, heuristic in the league) plus ``gamma = 0.999``, so a winning player prefers ending the game
over stalling to the turn cap. Keeps ``runs/<tag>_<iteration>.pt`` every 100 iterations for the
Elo ladder, then measures exploitability as the max over two exploiter runs (a single run is only a
lower bound; see ``writeup.md``).

Resumable: training continues from its last eval checkpoint, and finished exploiter runs are
skipped. Stop at any time and rerun the same command.

``RNAD`` is the same run with one change: the magnet is a slowly moving average of the policy's
weights (``magnet_ema``) rather than a copy refreshed every 20 iterations. Its control is the
plain run (``nash_g``): same start, budget and settings otherwise.

``CAREFUL`` replaces the leaky heuristic in the league with the careful one (it never gifts a
playable card when it can avoid it) at twice the share of games, and scores against it; see the
"leaky anchor" finding in ``writeup.md``.

Run with::

    uv run python -m experiments.train_ventures_continue       # nash_g
    uv run python -m experiments.train_ventures_continue rnad  # the R-NaD-style variant
    uv run python -m experiments.train_ventures_continue careful  # careful heuristic in the league
    uv run python -m experiments.train_ventures_continue diverse  # four styles in the league
    uv run python -m experiments.train_ventures_continue diverse_sym  # + suit-symmetry regularizer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import iirl.games  # noqa: F401  (registers "ventures")
from experiments import exploit_ventures
from experiments import train_ventures_nash as nash
from experiments.train_ventures_league import _train
from iirl.core.registry import make, make_heuristic
from iirl.eval.duplicate import duplicate_match
from iirl.games.ventures.heuristic import LEAGUE_STYLES
from iirl.selfplay import PPOConfig


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    start: str = "runs/league_control.pt"
    tag: str = "nash_g"
    iterations: int = 400
    save_every: int = 100
    exploit_seeds: tuple[int, ...] = (0, 1)
    anchor: str = "ventures"  # registered heuristic for the in-training Elo eval
    anchor_elo: float = 1676.0
    ppo: PPOConfig = field(
        default_factory=lambda: replace(
            nash.variant("nash", nash.Config().ppo), gamma=0.999, seed=3
        )
    )


RNAD = Config(
    tag="rnad",
    ppo=replace(nash.variant("nash", nash.Config().ppo), gamma=0.999, seed=3, magnet_ema=0.02),
)


CAREFUL = Config(
    tag="careful",
    anchor="ventures_careful",
    anchor_elo=1500.0,  # a fresh scale: the careful heuristic's rating is not known on the old one
    ppo=replace(
        nash.variant("nash", nash.Config().ppo),
        gamma=0.999,
        seed=4,
        scripted_opponents=(make_heuristic("ventures_careful"),),
        scripted_frac=0.2,
    ),
)


DIVERSE = Config(
    tag="diverse",
    anchor="ventures_careful",
    anchor_elo=1500.0,
    ppo=replace(
        nash.variant("nash", nash.Config().ppo),
        gamma=0.999,
        seed=5,
        scripted_opponents=tuple(make_heuristic(name) for name in LEAGUE_STYLES),
        scripted_frac=0.3,
    ),
)
DIVERSE_SYM = replace(
    DIVERSE, tag="diverse_sym", ppo=replace(DIVERSE.ppo, symmetry_coef=0.1, seed=6)
)
PANEL = ("ventures", *LEAGUE_STYLES)


def panel(path: str, tag: str, deals: int = 500) -> dict:
    """Rate a checkpoint against every opponent style (duplicate deals); save and return it.

    The weakest matchup matters more than the average: an agent that feeds on one habit shows
    up as a large spread.
    """
    out_path = Path(f"runs/panel_{tag}.json")
    if out_path.exists():
        return json.loads(out_path.read_text("utf-8"))
    game = make("ventures")
    result = {}
    for name in PANEL:
        agent = (exploit_ventures.ppo_agent, {"path": path})
        r = duplicate_match(game, agent, (make_heuristic, {"short_name": name}), deals, seed=61)
        result[name] = {"score": r.score, "score_se": r.score_se, "margin": r.margin}
        print(f"  panel {tag} vs {name:<20} {r.score:.3f}+-{r.score_se:.3f}  {r.margin:+.1f}")
    out_path.write_text(json.dumps(result, indent=2), "utf-8")
    return result


def main(config: Config) -> None:
    """Train, then exploit with each seed; print the exploitability estimate (the max)."""
    ppo = replace(config.ppo, iterations=config.iterations)
    path = _train(
        config.start,
        ppo,
        config.tag,
        save_every=config.save_every,
        heuristic=make_heuristic(config.anchor),
        anchor_elo=config.anchor_elo,
    )
    scores = []
    for seed in config.exploit_seeds:
        name = f"{config.tag}_s{seed}"
        record = Path(f"runs/exploit_{name}.json")
        if not record.exists():
            ex = exploit_ventures.Config()
            exploit_ventures.main(
                replace(ex, target=path, name=name, ppo=replace(ex.ppo, seed=seed))
            )
        scores.append(float(json.loads(record.read_text("utf-8"))["final"]["score"]))
    print(f"\nexploitability of {config.tag}: max {max(scores):.3f}  (runs: {scores})")
    panel(path, config.tag)


if __name__ == "__main__":
    import sys

    configs = {"rnad": RNAD, "careful": CAREFUL, "diverse": DIVERSE, "diverse_sym": DIVERSE_SYM}
    main(configs[sys.argv[1]] if sys.argv[1:] else Config())
