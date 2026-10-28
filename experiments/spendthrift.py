"""Spendthrift: exact solutions and what perfect tempo looks like.

``solve`` runs backward induction on the registered sizes (``spendthrift``: 2 players, 5 lots,
2 end cards, 8 coins; ``spendthrift3``: 3 players, 4 lots, 2 end cards, 6 coins), then plays out
every deck order along the solution, weighted by its probability, and summarises the equilibrium's
tempo: what each lot sells for by how close the end is, how often lots go unsold, how many coins
are left, and how often the reserve penalty bites. Output: ``runs/spendthrift/solve.json``.

    uv run python -m experiments.spendthrift solve
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from iirl.core.registry import make
from iirl.eval.perfect_info import Solution, solve
from iirl.games.spendthrift import Phase, Spendthrift, SpendthriftState

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


if __name__ == "__main__":
    {"solve": run_solve}[sys.argv[1]]()
