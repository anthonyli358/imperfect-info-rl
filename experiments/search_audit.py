"""Where can search improve a policy? A game-agnostic paired-rollout audit.

The Ventures audit (``experiments/ventures_search_audit.py``) found the one phase where search
could help (the last few turns) and predicted the size of the gain before any agent was built.
This is the same measurement for any game with the hooks ``agents/rollout.py`` uses.

At a start-of-turn position from the policy's self-play, every legal first move is played out to
the end with the greedy policy in both seats, in ``worlds`` re-deals of what the mover cannot see
× ``orders`` shuffles of the deck, all moves sharing the same worlds and orders so the luck
cancels in the differences. Per phase of the game (by cards left in the deck) it reports:

* how often the policy's move is the best one, and two estimates of what a better move would gain
  (an upper bound that flatters itself, a split-half lower bound that is hurt by selection);
* how much of a move's advantage depends on the hidden information (sd across worlds) and how
  much is luck (sd across deck orders), for moves within 5 points of the policy's.

``match`` then plays the policy improved by opponent-robust rollouts (``agents/rollout.py``)
against the plain policy, and ``panel`` against each heuristic style, on paired deals.

Run with::

    uv run python -m experiments.search_audit camel_stocks collect
    uv run python -m experiments.search_audit camel_stocks report
    uv run python -m experiments.search_audit camel_stocks match deck_max=12 [key=value ...]
    uv run python -m experiments.search_audit camel_stocks panel deck_max=12 [key=value ...]
    uv run python -m experiments.search_audit tessera collect   # phases by round; 4 workers

Results of ``match`` and ``panel`` are appended to ``runs/rollout_<game>.jsonl``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import iirl.games  # noqa: F401  (registers the games)
from iirl.agents.ppo import PPOAgent
from iirl.agents.rollout import play_out, rollout_agent, turn_start
from iirl.core.registry import make, make_heuristic
from iirl.eval.duplicate import duplicate_match


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    game: str
    blueprint: str
    positions: int = 600
    worlds: int = 8
    orders: int = 8
    last_turn: int = 40  # positions are drawn at a uniformly random turn below this
    phases: tuple[tuple[str, int, int], ...] = ()  # (label, fewest, most cards in the deck)
    styles: tuple[str, ...] = ()  # heuristic styles for the panel
    workers: int = 10
    # Games without a card deck (Tessera): phases by ``state.round`` instead of cards left, every
    # decision is a turn, and the "orders" are shared chance streams instead of deck shuffles.
    phase_attr: str = "deck"
    every_decision_a_turn: bool = False


GAMES = {
    "camel_stocks": Config(
        game="camel_stocks",
        blueprint="runs/camel_nash.pt",
        last_turn=40,
        phases=(
            ("early (deck > 25)", 26, 99),
            ("middle (13-25)", 13, 25),
            ("late (7-12)", 7, 12),
            ("endgame (deck <= 6)", 0, 6),
        ),
        styles=("camel_stocks", "camel_stocks_careful"),
    ),
    "tessera": Config(
        game="tessera",
        blueprint="runs/tessera_ag/exit_prior_48.pt",
        positions=1000,
        worlds=1,  # nothing is hidden but the bag's order
        orders=16,
        last_turn=50,
        phases=(
            ("round 1", 0, 0),
            ("round 2", 1, 1),
            ("round 3", 2, 2),
            ("round 4", 3, 3),
            ("round 5 on", 4, 99),
        ),
        workers=4,
        phase_attr="round",
        every_decision_a_turn=True,
    ),
    "ventures": Config(
        game="ventures",
        blueprint="runs/diverse_sym.pt",
        last_turn=56,
        phases=(
            ("early (deck > 30)", 31, 99),
            ("middle (16-30)", 16, 30),
            ("late (7-15)", 7, 15),
            ("endgame (deck <= 6)", 0, 6),
        ),
    ),
}


def out_path(config: Config) -> Path:
    """Where the audited positions are saved."""
    return Path(f"runs/audit_{config.game}.pkl")


def audit(args: tuple) -> dict | None:
    """Audit one position (``args`` is ``(seed, config)``); None if the game ended first."""
    seed, config = args
    torch.set_num_threads(1)
    game = make(config.game)
    rng = np.random.default_rng(seed)
    explorer = PPOAgent.from_checkpoint(config.blueprint, greedy=False)
    policy = PPOAgent.from_checkpoint(config.blueprint, greedy=True)
    target = int(rng.integers(0, config.last_turn))
    state = game.new_initial_state()
    turns = 0
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
        else:
            if config.every_decision_a_turn or turn_start(game, state):
                if turns >= target:
                    break
                turns += 1
            action = explorer.step(game, state, rng)
        state = game.apply_action(state, action)
    if game.is_terminal(state):
        return None
    m = game.current_player(state)
    legal = game.legal_actions(state)
    if len(legal) < 2:
        return None
    prior = np.asarray(
        [policy.step(game, state, rng) == a for a in legal], dtype=float
    )  # the greedy policy's move as a one-hot
    worlds = [state] + [
        game.resample_from_infostate(state, m, rng) for _ in range(config.worlds - 1)
    ]
    q = np.zeros((config.worlds, config.orders, len(legal)))
    deck = config.phase_attr == "deck"
    for w, world in enumerate(worlds):
        for o in range(config.orders):
            order: list[int] = []
            if deck:
                cards = np.repeat(np.arange(len(world.deck)), world.deck)
                rng.shuffle(cards)
                order = cards.tolist()
            for k, a in enumerate(legal):
                after = game.apply_action(world, a)
                seats = (policy, policy)
                q[w, o, k] = play_out(game, after, seats, order, rng, w * 1000 + o, deck)[m]
    phase = sum(state.deck) if deck else getattr(state, config.phase_attr)
    return {"seed": seed, "deck": phase, "q": q, "policy": int(np.argmax(prior))}


def collect(config: Config) -> None:
    """Audit positions in parallel and save them."""
    t0 = time.perf_counter()
    jobs = [(seed, config) for seed in range(2_000_000, 2_000_000 + config.positions)]
    with mp.get_context("spawn").Pool(config.workers) as pool:
        rows = [r for r in pool.imap_unordered(audit, jobs) if r is not None]
    out_path(config).write_bytes(pickle.dumps(rows))
    print(f"{len(rows)} positions audited in {(time.perf_counter() - t0) / 60:.1f} min")


def _mean(values: list[float]) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return "n/a"
    return f"{v.mean():+6.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}"


def report(config: Config) -> None:
    """Regret brackets and the luck / hidden-information decomposition, by phase."""
    rows = pickle.loads(out_path(config).read_bytes())
    half = config.orders // 2
    print(
        f"{config.game}: {len(rows)} positions; {config.worlds} worlds x {config.orders} orders\n"
    )
    print("Is the policy's move the best? What a better first move would gain (points per turn)")
    print(
        f"{'phase':<22}{'n':>4}  {'moves':>5}  {'policy is best':>14}  "
        f"{'upper (same rollouts)':>21}  {'lower (split half)':>18}"
    )
    for label, lo, hi in config.phases:
        part = [r for r in rows if lo <= r["deck"] <= hi]
        if not part:
            continue
        upper, lower, best, moves = [], [], [], []
        for r in part:
            q, p = r["q"], r["policy"]
            value = q.mean(axis=(0, 1))
            pick, score = q[:, :half].mean(axis=(0, 1)), q[:, half:].mean(axis=(0, 1))
            upper.append(value.max() - value[p])
            lower.append(score[int(np.argmax(pick))] - score[p])
            best.append(int(np.argmax(value)) == p)
            moves.append(q.shape[2])
        print(
            f"{label:<22}{len(part):>4}  {np.mean(moves):>5.1f}  {np.mean(best):>14.2f}  "
            f"{_mean(upper):>21}  {_mean(lower):>18}"
        )
    print(
        "   (upper: best move by all rollouts, which flatters itself; lower: chosen on half the "
        "deck orders and scored on the rest. The truth is between.)\n"
    )
    print("What a close move's advantage over the policy's depends on (moves within 5 points)")
    print(
        f"{'phase':<22}{'moves':>6}  {'sd across hidden worlds':>23}  "
        f"{'sd across deck orders':>21}  {'rollouts to resolve 1 point':>27}"
    )
    for label, lo, hi in config.phases:
        between, within = [], []
        for r in rows:
            if not lo <= r["deck"] <= hi:
                continue
            q, p = r["q"], r["policy"]
            for k in range(q.shape[2]):
                d = q[:, :, k] - q[:, :, p]
                if k == p or abs(d.mean()) > 5:
                    continue
                w = d.var(axis=1, ddof=1).mean()
                within.append(w)
                between.append(d.mean(axis=1).var(ddof=1) - w / config.orders)
        if not within:
            continue
        noise = np.sqrt(np.mean(within))
        if config.worlds < 2:  # nothing hidden: only the luck column means anything
            between = [0.0]
        print(
            f"{label:<22}{len(within):>6}  {np.sqrt(max(np.mean(between), 0)):>23.2f}  "
            f"{noise:>21.2f}  {4 * noise**2:>27.0f}"
        )
    print("   (rollouts per move for a 2-standard-error resolution of a 1-point difference)")


def ppo_agent(path: str) -> PPOAgent:
    """The policy as deployed: greedy (picklable spec for duplicate matches)."""
    return PPOAgent.from_checkpoint(path, greedy=True)


def _settings(config: Config) -> tuple[dict, int]:
    """The rollout agent's settings from the command line (``robust`` on by default)."""
    settings: dict[str, object] = {"blueprint": config.blueprint, "robust": True}
    deals = 400
    for item in sys.argv[3:]:
        key, text = item.split("=")
        if key == "deals":
            deals = int(text)
        else:
            settings[key] = json.loads(text)
    return settings, deals


def _record(config: Config, row: dict) -> None:
    print(json.dumps(row), flush=True)
    with Path(f"runs/rollout_{config.game}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _row(opponent: str, settings: dict, r, minutes: float) -> dict:  # noqa: ANN001
    return {
        "opponent": opponent,
        "settings": {k: v for k, v in settings.items() if k != "blueprint"},
        "deals": r.deals,
        "score": round(r.score, 4),
        "score_se": round(r.score_se, 4),
        "margin": round(r.margin, 2),
        "margin_se": round(r.margin_se, 2),
        "minutes": round(minutes, 1),
    }


def match(config: Config) -> None:
    """The rollout agent against the plain policy, on paired deals."""
    settings, deals = _settings(config)
    t0 = time.perf_counter()
    r = duplicate_match(
        make(config.game),
        (rollout_agent, settings),
        (ppo_agent, {"path": config.blueprint}),
        deals,
        seed=2027,
        workers=config.workers,
    )
    _record(config, _row("policy", settings, r, (time.perf_counter() - t0) / 60))


def panel(config: Config) -> None:
    """The rollout agent and the plain policy against each heuristic style, on the same deals."""
    settings, deals = _settings(config)
    game = make(config.game)
    for style in config.styles:
        for name, spec in (
            ("rollout", (rollout_agent, settings)),
            ("policy", (ppo_agent, {"path": config.blueprint})),
        ):
            t0 = time.perf_counter()
            r = duplicate_match(
                game, spec, (make_heuristic, {"short_name": style}), deals, 61, config.workers
            )
            row = _row(style, settings if name == "rollout" else {"agent": "policy"}, r, 0)
            row["minutes"] = round((time.perf_counter() - t0) / 60, 1)
            _record(config, row)


if __name__ == "__main__":
    config = GAMES[sys.argv[1]]
    stages = {"collect": collect, "report": report, "match": match, "panel": panel}
    stages[sys.argv[2]](config)
