"""Spendthrift: exact solutions and what perfect tempo looks like.

``solve`` runs backward induction on the registered sizes (``spendthrift``: 2 players, 5 lots,
2 end cards, 8 coins; ``spendthrift3``: 3 players, 4 lots, 2 end cards, 6 coins), then plays out
every deck order along the solution, weighted by its probability, and summarises the equilibrium's
tempo: what each lot sells for by how close the end is, how often lots go unsold, how many coins
are left, and how often the reserve penalty bites. Output: ``runs/spendthrift/solve.json``.

``tempo`` asks the controlled question: takes every auction opening reached in equilibrium play
with no end card seen yet and compares it with the *same* position with one end card seen (price,
whether it sells, who wins), and measures what a player's last coin is worth through the game.
Output: ``runs/spendthrift/tempo.json``.

``ppo`` trains PPO self-play (resumable; checkpoints every 25 iterations under
``runs/spendthrift/``) and scores each checkpoint against the solution: NashConv, and the payoff
its decisions give up against perfect play, by phase and by overbidding or underbidding. Output:
``runs/spendthrift/<game>_ppo_exact.json``.

    uv run python -m experiments.spendthrift solve
    uv run python -m experiments.spendthrift tempo
    uv run python -m experiments.spendthrift ppo game=spendthrift iterations=200
"""

from __future__ import annotations

import heapq
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from iirl.agents.ppo import PPOAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.eval.perfect_info import Solution, nash_conv, solve
from iirl.games.spendthrift import Phase, Spendthrift, SpendthriftState
from iirl.selfplay import PPOConfig

OUT = Path("runs/spendthrift")


def on_path(game: Spendthrift, sol: Solution) -> dict:
    """Probability-weighted statistics of equilibrium play over every deck order."""
    sold = defaultdict(float)  # (lot, ends_seen) -> probability the lot was sold
    seen = defaultdict(float)  # (lot, ends_seen) -> probability it was auctioned
    price = defaultdict(float)  # (lot, ends_seen) -> probability-weighted price
    spent_share = defaultdict(float)  # (lot, ends_seen) -> weighted price / winner's coins
    end = {"coins_left": 0.0, "penalty_bites": 0.0, "winner_was_poorest": 0.0}

    def walk(s: SpendthriftState, prob: float, start: SpendthriftState | None) -> None:
        if game.is_terminal(s):
            scores = game.scores(s)
            r = game.returns(s)
            poorest = min(s.coins)
            end["coins_left"] += prob * sum(s.coins) / len(s.coins)
            # The penalty bites if removing it would change who wins.
            plain = list(zip(s.totals, s.coins, strict=True))
            penalised = list(zip(scores, s.coins, strict=True))
            if plain.index(max(plain)) != penalised.index(max(penalised)):
                end["penalty_bites"] += prob
            winner = max(range(len(r)), key=r.__getitem__)
            end["winner_was_poorest"] += prob * (s.coins[winner] == poorest)
            return
        if game.is_chance_node(s):
            for a, p in game.chance_outcomes(s):
                walk(game.apply_action(s, a), prob * p, None)
            return
        if start is None:
            start = s
        nxt = game.apply_action(s, sol.choice[s])
        if nxt.phase != Phase.AUCTION:  # the auction just ended
            key = (start.lot, start.ends_seen)
            seen[key] += prob
            if nxt.totals != start.totals:
                winner = next(
                    i
                    for i, (a, b) in enumerate(zip(start.totals, nxt.totals, strict=True))
                    if a != b
                )
                paid = start.coins[winner] - nxt.coins[winner]
                sold[key] += prob
                price[key] += prob * paid
                if start.coins[winner]:
                    spent_share[key] += prob * paid / start.coins[winner]
            walk(nxt, prob, None)
            return
        walk(nxt, prob, start)

    walk(game.new_initial_state(), 1.0, None)
    rows = []
    for lot, ends_seen in sorted(seen):
        k = (lot, ends_seen)
        rows.append(
            {
                "lot": lot,
                "ends_seen": ends_seen,
                "p_auctioned": round(seen[k], 4),
                "p_sold": round(sold[k] / seen[k], 4),
                "mean_price": round(price[k] / sold[k], 3) if sold[k] else None,
                "mean_share_of_purse": round(spent_share[k] / sold[k], 3) if sold[k] else None,
            }
        )
    return {"auctions": rows, "end": {k: round(v, 4) for k, v in end.items()}}


def run_solve() -> None:
    """Solve both registered sizes and summarise equilibrium tempo."""
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("spendthrift", "spendthrift3"):
        game = make(name)
        t = time.perf_counter()
        sol = solve(game)
        seconds = time.perf_counter() - t
        root = sol.values[game.new_initial_state()]
        stats = on_path(game, sol)
        results[name] = {
            "size": game.game_type.long_name,
            "states": len(sol.values),
            "solve_seconds": round(seconds, 1),
            "root_value": [round(float(x), 4) for x in root],
            **stats,
        }
        print(f"\n{game.game_type.long_name}: {len(sol.values):,} states, {seconds:.0f}s")
        print(f"  value by seat: {results[name]['root_value']}")
        print("  lot  ends_seen  P(auctioned)  P(sold)  price  share of winner's purse")
        for r in stats["auctions"]:
            print(
                f"  {r['lot']:>3}  {r['ends_seen']:>9}  {r['p_auctioned']:>12.3f}  "
                f"{r['p_sold']:>7.3f}  {r['mean_price'] or 0:>5.2f}  "
                f"{r['mean_share_of_purse'] or 0:>6.3f}"
            )
        print(f"  end of game: {stats['end']}", flush=True)
        del sol
    (OUT / "solve.json").write_text(json.dumps(results, indent=1), "utf-8")


def openings(game: Spendthrift, sol: Solution) -> dict[SpendthriftState, float]:
    """Every auction's opening position reached in equilibrium play, with its probability."""
    reach: dict[SpendthriftState, float] = defaultdict(float)

    def walk(s: SpendthriftState, prob: float) -> None:
        if game.is_terminal(s):
            return
        if game.is_chance_node(s):
            for a, p in game.chance_outcomes(s):
                walk(game.apply_action(s, a), prob * p)
            return
        if s.bid == 0 and s.active == (1 << game.players) - 1:
            reach[s] += prob
        walk(game.apply_action(s, sol.action(s)), prob)

    walk(game.new_initial_state(), 1.0)
    return reach


def auction_outcome(game: Spendthrift, sol: Solution, s: SpendthriftState) -> tuple[int, int]:
    """``(winner, price)`` of the auction from ``s`` under the solution; winner -1 if unsold."""
    start = s
    while s.phase == Phase.AUCTION:
        s = game.apply_action(s, sol.action(s))
    for i in range(game.players):
        if s.totals[i] != start.totals[i]:
            return i, start.coins[i] - s.coins[i]
    return -1, 0


def toggle_end(game: Spendthrift, s: SpendthriftState) -> SpendthriftState:
    """The same position with one more end card seen (needs one still in the deck)."""
    return replace(s, ends_seen=s.ends_seen + 1, ends_left=s.ends_left - 1)


def coin_value(sol: Solution, s: SpendthriftState, p: int) -> float:
    """What ``p``'s last coin is worth at ``s``: their value minus their value with one fewer."""
    fewer = replace(s, coins=tuple(c - (i == p) for i, c in enumerate(s.coins)))
    return float(sol.value(s)[p] - sol.value(fewer)[p])


def run_tempo() -> None:
    """Paired positions: does the same auction go differently once an end card is seen?"""
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("spendthrift", "spendthrift3"):
        game = make(name)
        sol = solve(game)
        reach = openings(game, sol)
        # Pairs: every opening with no end card seen yet, against the same with one seen.
        base = {s: w for s, w in reach.items() if s.ends_seen == 0 and game.ends >= 2}
        total = sum(base.values())
        acc = defaultdict(float)
        for s, w in base.items():
            t = toggle_end(game, s)
            (wa, pa), (wb, pb) = auction_outcome(game, sol, s), auction_outcome(game, sol, t)
            w /= total
            acc["sold_before"] += w * (wa >= 0)
            acc["sold_after"] += w * (wb >= 0)
            acc["price_before"] += w * pa
            acc["price_after"] += w * pb
            acc["price_up"] += w * (pb > pa)
            acc["price_down"] += w * (pb < pa)
            acc["winner_changes"] += w * (wa != wb)
            for p in range(game.players):
                if s.coins[p]:
                    acc["coin_before"] += w * coin_value(sol, s, p) / game.players
                    acc["coin_after"] += w * coin_value(sol, t, p) / game.players
        # The value of a coin through the game, by end cards seen and lots left in the deck.
        by_phase = defaultdict(lambda: [0.0, 0.0])
        for s, w in reach.items():
            for p in range(game.players):
                if s.coins[p]:
                    key = f"ends_seen={s.ends_seen} lots_left={s.deck.bit_count()}"
                    by_phase[key][0] += w * coin_value(sol, s, p)
                    by_phase[key][1] += w
        results[name] = {
            "pairs": len(base),
            "paired": {k: round(v, 4) for k, v in acc.items()},
            "coin_value": {k: round(v / n, 4) for k, (v, n) in sorted(by_phase.items())},
        }
        print(f"\n{game.game_type.long_name}: {len(base):,} paired openings")
        for k, v in results[name]["paired"].items():
            print(f"  {k:<15} {v:.4f}")
        print("  value of the last coin (expected payoff; a win is +1):")
        for k, v in results[name]["coin_value"].items():
            print(f"    {k:<26} {v:.4f}")
        del sol
    (OUT / "tempo.json").write_text(json.dumps(results, indent=1), "utf-8")


PPO = PPOConfig(
    games_per_iter=2048,
    num_envs=256,
    num_workers=4,  # keeps the machine cool
    epochs=4,
    minibatch_size=1024,
    lr=3e-4,
    hidden=(256, 256),
    win_bonus=0.0,  # the payoff is already win / lose
    reward_scale=1.0,
    eval_interval=25,
    eval_games=1000,
    feature_sample_games=200,
    seed=0,
)


def net_probs(game: Spendthrift, net, states: list[SpendthriftState], greedy: bool):  # noqa: ANN001, ANN201
    """The net's move probabilities at every state in ``states``, batched, as a dict."""
    out: dict[SpendthriftState, np.ndarray] = {}
    for i in range(0, len(states), 8192):
        chunk = states[i : i + 8192]
        feats = np.stack([game.information_state_tensor(s, s.to_move) for s in chunk])
        with torch.no_grad():
            logits = net(torch.as_tensor(feats, dtype=torch.float32))[0].numpy()
        for s, row in zip(chunk, logits, strict=True):
            legal = game.legal_actions(s)
            probs = np.zeros(game.num_distinct_actions())
            if greedy:
                probs[legal[int(np.argmax(row[legal]))]] = 1.0
            else:
                z = row[legal] - row[legal].max()
                probs[legal] = np.exp(z) / np.exp(z).sum()
            out[s] = probs
    return out


def progress(game: Spendthrift, s: SpendthriftState) -> tuple[int, int, int, int]:
    """A key that strictly increases with every move, for a forward pass in order."""
    flipped = game.lots - s.deck.bit_count() + s.ends_seen
    passes = game.players - s.active.bit_count() if s.phase == Phase.AUCTION else 0
    return flipped, int(s.phase != Phase.AUCTION), s.bid, passes


def reach_under(game: Spendthrift, probs: dict) -> dict[SpendthriftState, float]:
    """Probability of reaching each player node when everyone plays ``probs``."""
    reach: dict[SpendthriftState, float] = defaultdict(float)
    root = game.new_initial_state()
    reach[root] = 1.0
    heap = [(progress(game, root), 0, root)]
    queued, tick = {root}, 1
    out: dict[SpendthriftState, float] = {}
    while heap:
        _, _, s = heapq.heappop(heap)
        w = reach[s]
        if game.is_terminal(s):
            continue
        if game.is_chance_node(s):
            moves = game.chance_outcomes(s)
        else:
            out[s] = w
            moves = [(a, probs[s][a]) for a in game.legal_actions(s) if probs[s][a] > 0]
        for a, p in moves:
            child = game.apply_action(s, a)
            reach[child] += w * p
            if child not in queued:
                queued.add(child)
                heapq.heappush(heap, (progress(game, child), tick, child))
                tick += 1
    return out


def decision_quality(
    game: Spendthrift, sol: Solution, probs: dict, reach: dict[SpendthriftState, float]
) -> dict:
    """Payoff given up per decision against perfect play, from the exact equilibrium values."""
    lost = defaultdict(float)  # by phase
    mass = defaultdict(float)
    kinds = defaultdict(float)  # overbid / underbid / matched, probability-weighted
    total_lost = 0.0
    for s, w in reach.items():
        p = s.to_move
        legal = game.legal_actions(s)
        q = {a: float(sol.value(game.apply_action(s, a))[p]) for a in legal}
        best = sol.action(s)
        gap = q[best] - sum(probs[s][a] * q[a] for a in legal)
        key = f"ends_seen={s.ends_seen} lots_left={s.deck.bit_count()}"
        lost[key] += w * gap
        mass[key] += w
        total_lost += w * gap
        for a in legal:
            kind = "matched" if a == best else ("overbid" if a > best else "underbid")
            kinds[kind] += w * probs[s][a]
            if a != best:
                kinds[f"{kind}_lost"] += w * probs[s][a] * (q[best] - q[a])
    return {
        "lost_per_game": round(total_lost, 4),
        "decisions_per_game": round(sum(reach.values()), 3),
        "lost_per_decision_by_phase": {
            k: round(lost[k] / mass[k], 4) for k in sorted(mass) if mass[k] > 1e-6
        },
        "moves": {k: round(v, 4) for k, v in sorted(kinds.items())},
    }


def run_ppo() -> None:
    """PPO self-play, then NashConv and decision quality of every checkpoint against the solution.

    ``key=value``: ``game`` (``spendthrift`` or ``spendthrift3``), ``iterations``.
    """
    from experiments.train_ventures_league import _train

    args = dict(item.split("=") for item in sys.argv[2:])
    name = args.get("game", "spendthrift")
    iterations = int(args.get("iterations", 200))
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"spendthrift/{name}_ppo"
    _train(
        None,
        replace(PPO, iterations=iterations),
        tag,
        save_every=25,
        game_name=name,
        heuristic=RandomAgent(),
        anchor_elo=1000.0,
    )
    game = make(name)
    sol = solve(game)
    nodes = list(sol.choice)
    results_path = OUT / f"{name}_ppo_exact.json"
    results = json.loads(results_path.read_text("utf-8")) if results_path.exists() else {}
    for it in range(25, iterations + 1, 25):
        label, path = str(it), Path(f"runs/{tag}_{it}.pt")
        if label in results or not path.exists():
            continue
        net = PPOAgent.from_checkpoint(str(path)).net
        row = {}
        for mode in ("sampled", "greedy"):
            probs = net_probs(game, net, nodes, greedy=mode == "greedy")
            conv = nash_conv(game, lambda g, s, probs=probs: probs[s])
            row[mode] = {k: round(v, 4) for k, v in conv.items()}
            if mode == "sampled":
                row["quality"] = decision_quality(game, sol, probs, reach_under(game, probs))
        results[label] = row
        print(
            f"{name} ppo {it:>3}: NashConv sampled {row['sampled']['nash_conv']:.4f}, "
            f"greedy {row['greedy']['nash_conv']:.4f}; lost per game "
            f"{row['quality']['lost_per_game']:.4f}",
            flush=True,
        )
        results_path.write_text(json.dumps(results, indent=1), "utf-8")


if __name__ == "__main__":
    {"solve": run_solve, "tempo": run_tempo, "ppo": run_ppo}[sys.argv[1]]()
