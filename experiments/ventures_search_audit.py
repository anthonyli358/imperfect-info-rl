"""Where can search help on Ventures, and where does ours fail? Ground truth by paired rollouts.

The ReBeL agent (``experiments/rebel_ventures.py``) loses to its blueprint, and the measured cause
was a value network far noisier than the gaps between moves. That says the search cannot see the
gaps; it does not say where gaps exist, how large they are, or how much of them depends on the
hidden hand. This audit measures those directly.

At a start-of-turn position from blueprint self-play, every first move (play or discard) is played
out to the end of the game by the greedy blueprint, in ``worlds`` versions of the hidden
information (the true one, plus re-deals of the opponent's hand and deck consistent with what the
mover knows) and ``orders`` shuffles of the deck in each. All moves see the *same* worlds and the
same future cards, so the luck of the draw cancels in the differences between moves. ``Q[w, o, a]``
is the mover's final margin. From it:

* a move's value as the mover should judge it: the mean over worlds and orders;
* whether one turn of perfect lookahead beats the blueprint (the best move is chosen on half the
  rollouts and scored on the other half, so the choice does not flatter itself);
* what the search's own choice, made without seeing any rollout, is really worth;
* the value of knowing the opponent's hand: choosing per world instead of once for all worlds.

Run with::

    uv run python -m experiments.ventures_search_audit collect
    uv run python -m experiments.ventures_search_audit report
    uv run python -m experiments.ventures_search_audit decompose
    uv run python -m experiments.ventures_search_audit match deck_max=6 [z=2 worlds=8 ...]

``match`` plays the blueprint improved by late-game paired rollouts (``agents/rollout.py``, the
agent the audit's findings suggest) against the blueprint on paired deals.

``report``'s tables 1 and 5 choose the best of ~13 moves from half the rollouts, which picks
noise and comes out negative; ``decompose`` replaces them with estimators that do not select.
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

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.ppo.agent import PPOAgent
from iirl.agents.rebel import ReBeLAgent, ReBeLAgentConfig
from iirl.agents.rollout import rollout_agent
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.games.ventures.cards import NUM_CARD_TYPES, rank_of, suit_of
from iirl.games.ventures.game import _can_play


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    blueprint: str = "runs/diverse_sym.pt"
    value: str = "runs/rebel_v2/value.pt"
    positions: int = 800
    worlds: int = 8
    orders: int = 8
    last_turn: int = 56  # positions are drawn at a uniformly random turn below this
    workers: int = 10
    out: str = "runs/rebel_v2/audit.pkl"


def _rollout(game, state, agent, order: list[int], rng) -> list[float]:  # noqa: ANN001
    """Play ``state`` out with ``agent`` in both seats, drawing the deck in the given order."""
    k = 0
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.apply_action(state, order[k])
            k += 1
        else:
            state = game.apply_action(state, agent.step(game, state, rng))
    return game.returns(state)


def audit(args: tuple) -> dict | None:
    """Audit one position (``args`` is ``(seed, config)``); None if the game ended first."""
    seed, config = args
    torch.set_num_threads(1)
    game = make("ventures")
    rng = np.random.default_rng(seed)
    explorer = PPOAgent.from_checkpoint(config.blueprint, greedy=False)
    blueprint = PPOAgent.from_checkpoint(config.blueprint, greedy=True)
    searcher = ReBeLAgent(
        ReBeLAgentConfig(blueprint=config.blueprint, value=config.value, seed=seed)
    )
    searcher.reset(game, 0)
    target = int(rng.integers(0, config.last_turn))
    state = game.new_initial_state()
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
        else:
            at_turn_start = state.phase == 1 and searcher.belief.hands is not None
            if at_turn_start and state.turn >= target:
                break
            action = explorer.step(game, state, rng)
        after = game.apply_action(state, action)
        searcher.observe(game, after, action)
        state = after
    if game.is_terminal(state):
        return None
    m = game.current_player(state)
    legal = game.legal_actions(state)

    # What the search would do here, and what it thinks each move is worth.
    seen: dict = {}
    solve = searcher.solver.solve

    def spy(*a, **k):  # noqa: ANN002, ANN003, ANN202
        seen["solution"] = solve(*a, **k)
        return seen["solution"]

    searcher.solver.solve = spy
    searcher.player = m
    searcher.step(game, state, rng)
    if "solution" not in seen or sorted(seen["solution"].actions[0]) != sorted(legal):
        return None
    solution = seen["solution"]
    at = [solution.actions[0].index(a) for a in legal]
    prior = np.asarray(searcher.model(game, [state])[0])[legal]

    # Ground truth: every move, in the same worlds, under the same future cards.
    worlds = [state] + [
        game.resample_from_infostate(state, m, rng) for _ in range(config.worlds - 1)
    ]
    q = np.zeros((config.worlds, config.orders, len(legal)))
    for w, world in enumerate(worlds):
        for o in range(config.orders):
            order = np.repeat(np.arange(NUM_CARD_TYPES), world.deck)
            rng.shuffle(order)
            order = order.tolist()
            for k, a in enumerate(legal):
                q[w, o, k] = _rollout(game, game.apply_action(world, a), blueprint, order, rng)[m]

    cards = [a % NUM_CARD_TYPES for a in legal]
    mine, theirs = state.routes[m], state.routes[1 - m]
    return {
        "seed": seed,
        "turn": state.turn,
        "deck": sum(state.deck),
        "lead": game._score(state, m) - game._score(state, 1 - m),
        "q": q,
        "prior": prior,
        "net": np.asarray(solution.action_values[0])[at],
        "solved": np.asarray(solution.average[0])[at],
        "discard": np.asarray([a >= NUM_CARD_TYPES for a in legal]),
        "opens": np.asarray(
            [a < NUM_CARD_TYPES and not mine[suit_of(c)] for a, c in zip(legal, cards, strict=True)]
        ),
        "backing": np.asarray([rank_of(c) == 0 for c in cards]),
        # A discard the opponent could play straight onto a route they have started.
        "feeds": np.asarray(
            [
                a >= NUM_CARD_TYPES
                and bool(theirs[suit_of(c)])
                and _can_play(theirs[suit_of(c)], c)
                for a, c in zip(legal, cards, strict=True)
            ]
        ),
    }


def collect(config: Config) -> None:
    """Audit positions in parallel and save them."""
    t0 = time.perf_counter()
    jobs = [(seed, config) for seed in range(1_000_000, 1_000_000 + config.positions)]
    with mp.get_context("spawn").Pool(config.workers) as pool:
        rows = [r for r in pool.imap_unordered(audit, jobs) if r is not None]
    Path(config.out).write_bytes(pickle.dumps(rows))
    print(f"{len(rows)} positions audited in {(time.perf_counter() - t0) / 60:.1f} min")


def _mean(values: list[float]) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return "n/a"
    return f"{v.mean():+6.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}"


def report(config: Config) -> None:
    """Print the conditional tables (means ± standard errors over positions)."""
    rows = pickle.loads(Path(config.out).read_bytes())
    phases = {
        "early (deck > 30)": (31, 99),
        "middle (16-30)": (16, 30),
        "late (deck <= 15)": (0, 15),
    }
    print(f"{len(rows)} positions; {config.worlds} worlds x {config.orders} deck orders per move\n")

    def facts(r: dict) -> dict:
        q = r["q"]
        n = len(r["prior"])
        value = q.mean(axis=(0, 1))  # as the mover should judge it
        bp = int(np.argmax(r["prior"]))
        search = int(np.argmax(r["solved"]))
        anchored = int(np.argmax(r["prior"] * np.exp((r["net"] - r["net"].max()) / 8.0)))
        half = config.orders // 2
        a_pick, b_score = q[:, :half].mean(axis=(0, 1)), q[:, half:].mean(axis=(0, 1))
        oracle = int(np.argmax(a_pick))
        # Knowing the opponent's hand: choose per world on half the orders, score on the rest.
        per_world = q[:, :half].mean(axis=1).argmax(axis=1)
        informed = q[:, half:].mean(axis=1)[np.arange(len(q)), per_world].mean()
        order = np.argsort(-value)
        ranks = np.empty(n)
        ranks[np.argsort(-r["net"])] = np.arange(n)
        true_ranks = np.empty(n)
        true_ranks[order] = np.arange(n)
        return {
            "n_moves": n,
            "bp": bp,
            "search": search,
            "value": value,
            "oracle_gain": b_score[oracle] - b_score[bp],
            "search_gain": value[search] - value[bp],
            "anchored_gain": value[anchored] - value[bp],
            "anchored_changed": anchored != bp,
            "info_gain": informed - b_score[oracle],
            "spread": value.max() - np.median(value),
            "runner_up": value[order[0]] - value[order[1]] if n > 1 else 0.0,
            "bp_regret": value.max() - value[bp],
            "bp_rank": true_ranks[bp],
            "claimed": r["net"].max() - r["net"][bp],
            "rank_corr": np.corrcoef(ranks, true_ranks)[0, 1] if n > 2 else np.nan,
            "net_top_true_rank": true_ranks[int(np.argmax(r["net"]))],
        }

    data = [(r, facts(r)) for r in rows]
    print("1. How much do moves matter, and is the blueprint improvable? (points; mover's view)")
    print(
        f"{'phase':<20}{'n':>4}  {'best - median move':>18}  {'best - 2nd best':>16}  "
        f"{'blueprint is best':>17}  {'perfect lookahead gain':>22}"
    )
    for name, (lo, hi) in phases.items():
        part = [f for r, f in data if lo <= r["deck"] <= hi]
        best = np.mean([f["bp_rank"] == 0 for f in part])
        print(
            f"{name:<20}{len(part):>4}  {_mean([f['spread'] for f in part]):>18}  "
            f"{_mean([f['runner_up'] for f in part]):>16}  {best:>17.2f}  "
            f"{_mean([f['oracle_gain'] for f in part]):>22}"
        )
    print(
        "   ('perfect lookahead gain': best move picked on half the rollouts, scored on the other"
        " half, minus the blueprint's move. Note 'best - median' and 'best - 2nd' are picked and"
        " scored on the same rollouts, so they overstate.)\n"
    )

    print("2. What the search's choice is really worth, against the blueprint's move")
    print(
        f"{'phase':<20}{'n':>4}  {'search changes move':>19}  {'true gain, plain':>17}  "
        f"{'claimed gain':>13}  {'anchored(8) changes':>19}  {'true gain, anchored':>19}"
    )
    for name, (lo, hi) in phases.items():
        part = [f for r, f in data if lo <= r["deck"] <= hi]
        print(
            f"{name:<20}{len(part):>4}  {np.mean([f['search'] != f['bp'] for f in part]):>19.2f}  "
            f"{_mean([f['search_gain'] for f in part]):>17}  "
            f"{_mean([f['claimed'] for f in part]):>13}  "
            f"{np.mean([f['anchored_changed'] for f in part]):>19.2f}  "
            f"{_mean([f['anchored_gain'] for f in part]):>19}"
        )
    print()

    print("3. When the search overrides the blueprint: by what kind of swap (true gain, points)")
    kinds: dict[str, list[float]] = {}
    for r, f in data:
        if f["search"] == f["bp"]:
            continue

        def label(k: int, r: dict = r) -> str:
            if r["discard"][k]:
                return "discard that feeds opponent" if r["feeds"][k] else "discard"
            if r["opens"][k]:
                return "open a route"
            return "extend a route"

        kinds.setdefault(f"{label(f['bp'])} -> {label(f['search'])}", []).append(f["search_gain"])
    for key, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 5:
            print(f"   {key:<58} n {len(v):>3}  {_mean(v)}")
    print()

    print("4. Can the value network rank moves? (per position, over its legal first moves)")
    print(
        f"{'phase':<20}{'n':>4}  {'rank correlation':>17}  {'net top move is truly best':>26}  "
        f"{'net top in true top 3':>21}  {'chance: top 3':>13}"
    )
    for name, (lo, hi) in phases.items():
        part = [f for r, f in data if lo <= r["deck"] <= hi and f["n_moves"] > 3]
        corr = [f["rank_corr"] for f in part if np.isfinite(f["rank_corr"])]
        print(
            f"{name:<20}{len(part):>4}  {_mean(corr):>17}  "
            f"{np.mean([f['net_top_true_rank'] == 0 for f in part]):>26.2f}  "
            f"{np.mean([f['net_top_true_rank'] < 3 for f in part]):>21.2f}  "
            f"{np.mean([3 / f['n_moves'] for f in part]):>13.2f}"
        )
    print()

    print("5. What is knowing the opponent's hand worth? (points per turn; per-world choice vs one")
    print(
        "   choice for all worlds, both picked on half the deck orders, scored on the other half)"
    )
    for name, (lo, hi) in phases.items():
        part = [f for r, f in data if lo <= r["deck"] <= hi]
        print(f"   {name:<20} n {len(part):>3}  {_mean([f['info_gain'] for f in part])}")
    by_kind = {"best move is a play": [], "best move is a discard": []}
    for r, f in data:
        top = int(np.argmax(f["value"]))
        by_kind["best move is a discard" if r["discard"][top] else "best move is a play"].append(
            f["info_gain"]
        )
    for key, v in by_kind.items():
        print(f"   {key:<20} n {len(v):>3}  {_mean(v)}")


def _kind(r: dict, k: int) -> str:
    if r["discard"][k]:
        return "discard the opponent can play" if r["feeds"][k] else "other discard"
    return "open a route" if r["opens"][k] else "extend a route"


def decompose(config: Config) -> None:
    """Estimators without selection: regret brackets, what an advantage depends on, net bias."""
    rows = pickle.loads(Path(config.out).read_bytes())
    half = config.orders // 2
    phases = {
        "early (deck > 30)": (31, 99),
        "middle (16-30)": (16, 30),
        "late (7-15)": (7, 15),
        "endgame (deck <= 6)": (0, 6),
    }
    print("6. Is the blueprint's move the best? Three estimates of what a better move would gain")
    print(
        f"{'phase':<22}{'n':>4}  {'upper (same rollouts)':>21}  {'split, all moves':>17}  "
        f"{'split, its own top 3':>20}  {'blueprint prob of its move':>26}"
    )
    for name, (lo, hi) in phases.items():
        upper, split, top3, conf = [], [], [], []
        for r in rows:
            if not lo <= r["deck"] <= hi:
                continue
            q, bp = r["q"], int(np.argmax(r["prior"]))
            value = q.mean(axis=(0, 1))
            pick, score = q[:, :half].mean(axis=(0, 1)), q[:, half:].mean(axis=(0, 1))
            upper.append(value.max() - value[bp])
            split.append(score[int(np.argmax(pick))] - score[bp])
            own = np.argsort(-r["prior"])[:3]
            top3.append(score[own[int(np.argmax(pick[own]))]] - score[bp])
            conf.append(r["prior"].max())
        print(
            f"{name:<22}{len(upper):>4}  {_mean(upper):>21}  {_mean(split):>17}  "
            f"{_mean(top3):>20}  {np.mean(conf):>26.2f}"
        )
    print(
        "   (upper: best move by all rollouts minus the blueprint's, biased up by selection. "
        "split: chosen on half the rollouts, scored on the rest, biased down. The truth lies "
        "between; 'own top 3' lets the rollouts pick only among the blueprint's three likeliest.)\n"
    )

    print("7. What a move's advantage over the blueprint's move depends on (points)")
    print(
        f"{'kind of move':<32}{'moves':>6}  {'mean advantage':>15}  "
        f"{'sd across opponent hands':>24}  {'sd across deck orders':>21}"
    )
    for close in (False, True):
        groups: dict[str, list[tuple[float, float, float]]] = {}
        for r in rows:
            q, bp = r["q"], int(np.argmax(r["prior"]))
            for k in range(q.shape[2]):
                if k == bp:
                    continue
                d = q[:, :, k] - q[:, :, bp]
                if close and abs(d.mean()) > 5:
                    continue
                within = d.var(axis=1, ddof=1).mean()
                between = d.mean(axis=1).var(ddof=1) - within / config.orders
                groups.setdefault(_kind(r, k), []).append((d.mean(), between, within))
        print("   moves within 5 points of the blueprint's:" if close else "   all moves:")
        for key in (
            "extend a route",
            "open a route",
            "other discard",
            "discard the opponent can play",
        ):
            g = np.asarray(groups.get(key, [(0, 0, 0)]))
            print(
                f"   {key:<29}{len(g):>6}  {g[:, 0].mean():>15.2f}  "
                f"{np.sqrt(max(g[:, 1].mean(), 0)):>24.2f}  {np.sqrt(g[:, 2].mean()):>21.2f}"
            )
    print(
        "   (the opponent-hand column is the part of the advantage that changes with the hidden "
        "hand, with the deck-order noise removed: where it is large next to the mean, the right "
        "move depends on what the opponent holds.)\n"
    )

    print("8. The value network's error on a move's advantage over the blueprint's move, by kind")
    print("   (network's claimed advantage minus the true one; positive = the move is overrated)")
    errors: dict[str, list[float]] = {}
    for r in rows:
        bp = int(np.argmax(r["prior"]))
        value = r["q"].mean(axis=(0, 1))
        per: dict[str, list[float]] = {}
        for k in range(len(value)):
            if k != bp:
                claimed, true = r["net"][k] - r["net"][bp], value[k] - value[bp]
                per.setdefault(_kind(r, k), []).append(claimed - true)
        for key, v in per.items():
            errors.setdefault(key, []).append(float(np.mean(v)))
    for key in ("extend a route", "open a route", "other discard", "discard the opponent can play"):
        v = errors.get(key, [])
        print(f"   {key:<32} positions {len(v):>4}  {_mean(v)}")


def blueprint_agent(path: str) -> PPOAgent:
    """The blueprint as deployed: greedy."""
    return PPOAgent.from_checkpoint(path, greedy=True)


def match(config: Config) -> None:
    """The rollout-improved blueprint against the blueprint, on paired deals."""
    settings: dict[str, object] = {"blueprint": config.blueprint}
    deals = 200
    for item in sys.argv[2:]:
        key, text = item.split("=")
        if key == "deals":
            deals = int(text)
        else:
            settings[key] = json.loads(text)
    t0 = time.perf_counter()
    result = duplicate_match(
        make("ventures"),
        (rollout_agent, settings),
        (blueprint_agent, {"path": config.blueprint}),
        deals=deals,
        seed=2027,
        workers=config.workers,
    )
    row = {
        "settings": {k: v for k, v in settings.items() if k != "blueprint"},
        "deals": result.deals,
        "score": round(result.score, 4),
        "score_se": round(result.score_se, 4),
        "margin": round(result.margin, 2),
        "margin_se": round(result.margin_se, 2),
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    print(json.dumps(row), flush=True)
    with Path(config.out).with_name("rollout_matches.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    stages = {"collect": collect, "report": report, "decompose": decompose, "match": match}
    stages[sys.argv[1]](Config())
