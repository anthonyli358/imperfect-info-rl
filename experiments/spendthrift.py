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

``search`` runs PUCT search on the final PPO net and scores every move it makes exactly (output:
``runs/spendthrift/search.json``); ``endgame=K`` searches only once K or fewer lots are left.
``value`` trains a value net on PPO's self-play outcomes, measures every leaf value against the
exact values, and reruns the search with each (``value.json``). ``maxn`` does the same for any
number of players with max-n PUCT (a value per player; ``maxn_<game>.json``).

    uv run python -m experiments.spendthrift solve
    uv run python -m experiments.spendthrift tempo
    uv run python -m experiments.spendthrift ppo game=spendthrift iterations=200
    uv run python -m experiments.spendthrift search sims=0,50,200,800 endgame=0,1,2
    uv run python -m experiments.spendthrift value
    uv run python -m experiments.spendthrift maxn game=spendthrift3
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

from iirl.agents.alphazero.agent import AlphaZeroAgent
from iirl.agents.alphazero.mcts import PUCTConfig
from iirl.agents.alphazero.multiplayer import MultiPUCTAgent, NetMultiEvaluator
from iirl.agents.base import Agent
from iirl.agents.ppo import PPOAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make
from iirl.eval.perfect_info import Solution, nash_conv, solve
from iirl.games.spendthrift import Phase, Spendthrift, SpendthriftState
from iirl.nn.policy_value import NetEvaluator, PolicyValueNet, SplitEvaluator
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


def play_scored(
    game: Spendthrift, sol: Solution, agents: list, seed: int
) -> tuple[list[float], list[tuple[str, float]]]:
    """Play one game, ``agents[p]`` in seat ``p``: its returns and each move's exact loss."""
    rng = np.random.default_rng(seed)
    s, losses = game.new_initial_state(), []
    while not game.is_terminal(s):
        if game.is_chance_node(s):
            acts, ps = zip(*game.chance_outcomes(s), strict=True)
            s = game.apply_action(s, int(rng.choice(acts, p=ps)))
            continue
        p = s.to_move
        a = agents[p].step(game, s, rng)
        best = sol.value(game.apply_action(s, sol.action(s)))[p]
        lost = float(best - sol.value(game.apply_action(s, a))[p])
        losses.append((f"lots_left={s.deck.bit_count()}", lost))
        s = game.apply_action(s, a)
    return game.returns(s), losses


class Greedy(Agent):
    """The PPO net's most likely legal move."""

    name = "ppo"

    def __init__(self, net: PolicyValueNet) -> None:
        """Wrap the PPO net."""
        self.evaluator = NetEvaluator(net)

    def step(self, game, state, rng) -> int:  # noqa: ANN001
        """The argmax legal move."""
        priors, _ = self.evaluator.evaluate(game, state)
        legal = game.legal_actions(state)
        return int(legal[int(np.argmax(priors[legal]))])


class Endgame(Agent):
    """Greedy PPO until ``k`` or fewer lots are left in the deck, then ``search``."""

    def __init__(self, ppo: Agent, search: Agent, k: int) -> None:
        """Switch from ``ppo`` to ``search`` at ``k`` lots left."""
        self.ppo, self.search, self.k = ppo, search, k
        self.name = f"{search.name}@k{k}"

    def step(self, game, state, rng) -> int:  # noqa: ANN001
        """Defer to whichever agent owns this phase."""
        late = state.deck.bit_count() <= self.k
        return (self.search if late else self.ppo).step(game, state, rng)


class ExactLeaves:
    """Search priors from the PPO net, leaf values from the exact solution (an upper bound)."""

    def __init__(self, net: PolicyValueNet, sol: Solution) -> None:
        """Priors from ``net``, values from ``sol``."""
        self.priors, self.sol, self.net = NetEvaluator(net), sol, net

    def evaluate(self, game, state):  # noqa: ANN001, ANN201
        """``(priors, exact value for the mover)`` at one state."""
        priors, _ = self.priors.evaluate(game, state)
        return priors, float(self.sol.value(state)[state.to_move])

    def evaluate_batch(self, game, states):  # noqa: ANN001, ANN201
        """Batched :meth:`evaluate`."""
        priors, _ = self.priors.evaluate_batch(game, states)
        return priors, np.asarray([self.sol.value(s)[s.to_move] for s in states])


def score_agent(game: Spendthrift, sol: Solution, agent: Agent, ppo: Agent, games: int) -> dict:
    """``games`` self-play games scored move by move, and as many seat-rotated games vs ``ppo``.

    Deal seeds are fixed (``0..games-1`` and ``10_000 + g``), so every agent sees the same deals.
    """
    t = time.perf_counter()
    per_game, by_phase, n_phase = [], defaultdict(float), defaultdict(int)
    for g in range(games):
        _, losses = play_scored(game, sol, [agent] * game.num_players, seed=g)
        per_game.append(sum(x for _, x in losses))
        for k, x in losses:
            by_phase[k] += x
            n_phase[k] += 1
    scores = []
    for g in range(games):
        seat = g % game.num_players
        seats = [ppo] * game.num_players
        seats[seat] = agent  # the others are all greedy PPO
        rets, _ = play_scored(game, sol, seats, seed=10_000 + g)
        scores.append(rets[seat])
    lost, sc = np.asarray(per_game), np.asarray(scores)
    return {
        "games": games,
        "lost_per_game": round(float(lost.mean()), 4),
        "lost_per_game_se": round(float(lost.std(ddof=1) / np.sqrt(games)), 4),
        "lost_per_decision_by_phase": {
            k: round(by_phase[k] / n_phase[k], 4) for k in sorted(n_phase)
        },
        "vs_ppo_payoff": round(float(sc.mean()), 4),
        "vs_ppo_payoff_se": round(float(sc.std(ddof=1) / np.sqrt(games)), 4),
        "seconds_per_game": round((time.perf_counter() - t) / (2 * games), 3),
    }


def print_row(label: str, r: dict) -> None:
    """One line per scored agent."""
    print(
        f"{label:>12}: lost per game {r['lost_per_game']:.4f} +- {r['lost_per_game_se']:.4f}"
        f"; vs PPO {r['vs_ppo_payoff']:+.4f} +- {r['vs_ppo_payoff_se']:.4f}; "
        f"{r['seconds_per_game']:.2f}s/game; by phase {r['lost_per_decision_by_phase']}",
        flush=True,
    )


def run_search() -> None:
    """Does lookahead fix PPO's endgame errors? Two players; PUCT on the final PPO net.

    Priors and leaf values both come from the PPO checkpoint (``NetEvaluator``). For each budget,
    ``games`` self-play games score every decision exactly against the solution, and as many
    seat-balanced games against greedy PPO give the head-to-head score. Deals are shared across
    budgets (same seeds). ``key=value``: ``games``, ``sims`` (comma-separated), ``checkpoint``,
    ``endgame`` (comma-separated K: search only once K or fewer lots are left in the deck, greedy
    PPO before then; results under ``"<sims>@k<K>"``). Output: ``runs/spendthrift/search.json``.
    """
    args = dict(item.split("=") for item in sys.argv[2:])
    games = int(args.get("games", 300))
    budgets = [int(x) for x in args.get("sims", "0,50,200,800").split(",")]
    checkpoint = args.get("checkpoint", "runs/spendthrift/spendthrift_ppo_200.pt")
    ks = [int(x) for x in args["endgame"].split(",")] if "endgame" in args else [None]
    torch.set_num_threads(1)
    game = make("spendthrift")
    sol = solve(game)
    net = PPOAgent.from_checkpoint(checkpoint).net
    ppo = Greedy(net)
    path = OUT / "search.json"
    results = json.loads(path.read_text("utf-8")) if path.exists() else {}
    for sims, k in [(sims, k) for sims in budgets for k in ks]:
        label = str(sims) if k is None else f"{sims}@k{k}"
        if label in results:
            continue
        agent = (
            ppo
            if sims == 0
            else AlphaZeroAgent(
                NetEvaluator(net), PUCTConfig(simulations=sims, determinizations=1), f"search{sims}"
            )
        )
        if k is not None:
            agent = Endgame(ppo, agent, k)
        results[label] = score_agent(game, sol, agent, ppo, games)
        print_row(label, results[label])
        path.write_text(json.dumps(results, indent=1), "utf-8")


def self_play_values(game: Spendthrift, probs: dict) -> dict[SpendthriftState, np.ndarray]:
    """Every player's exact expected return from each state when everyone plays ``probs``."""
    memo: dict[SpendthriftState, np.ndarray] = {}

    def value(s: SpendthriftState) -> np.ndarray:
        v = memo.get(s)
        if v is not None:
            return v
        if game.is_terminal(s):
            v = np.asarray(game.returns(s), dtype=float)
        elif game.is_chance_node(s):
            v = sum(p * value(game.apply_action(s, a)) for a, p in game.chance_outcomes(s))
        else:
            v = sum(probs[s][a] * value(game.apply_action(s, a)) for a in game.legal_actions(s))
        memo[s] = v
        return v

    value(game.new_initial_state())
    return memo


def self_play_positions(
    game: Spendthrift, probs: dict, games: int, per_game: int, seed: int, all_views: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Play ``games`` games under ``probs``; sample positions with the mover's final return.

    With ``all_views`` each sampled position gives one row per player (that player's view and
    return), for a value net that can value a position for anyone.
    """
    rng = np.random.default_rng(seed)
    feats, ys = [], []
    for _ in range(games):
        s, seen = game.new_initial_state(), []
        while not game.is_terminal(s):
            if game.is_chance_node(s):
                acts, ps = zip(*game.chance_outcomes(s), strict=True)
                s = game.apply_action(s, int(acts[rng.choice(len(acts), p=ps)]))
                continue
            seen.append(s)
            p = probs[s]
            s = game.apply_action(s, int(rng.choice(len(p), p=p / p.sum())))
        rets = game.returns(s)
        for i in rng.choice(len(seen), size=min(per_game, len(seen)), replace=False):
            x = seen[i]
            for p in range(game.num_players) if all_views else (x.to_move,):
                feats.append(game.information_state_tensor(x, p))
                ys.append(rets[p])
    return np.stack(feats).astype(np.float32), np.asarray(ys, dtype=np.float32)


def fit_value(x_tr, y_tr, x_te, y_te, actions: int, epochs: int = 20):  # noqa: ANN001, ANN201
    """A tanh value net regressed on outcomes; early-stopped on held-out games' MSE."""
    torch.manual_seed(0)
    net = PolicyValueNet(x_tr.shape[1], actions, (256, 256), "tanh")
    net.set_feature_stats(x_tr.mean(axis=0), x_tr.std(axis=0))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
    xt, yt = torch.as_tensor(x_tr), torch.as_tensor(y_tr)
    xv, yv = torch.as_tensor(x_te), torch.as_tensor(y_te)
    best, best_state, curve = np.inf, None, []
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 2048):
            idx = perm[b : b + 2048]
            loss = torch.nn.functional.mse_loss(net(xt[idx])[1], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            mse = float(torch.nn.functional.mse_loss(net(xv)[1], yv))
        curve.append(round(mse, 4))
        if mse < best:
            best, best_state = mse, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, best, curve


def net_values(game: Spendthrift, net: PolicyValueNet, states: list) -> np.ndarray:
    """The net's value head for the mover at every state in ``states``, batched."""
    out = []
    for i in range(0, len(states), 8192):
        chunk = states[i : i + 8192]
        feats = np.stack([game.information_state_tensor(s, s.to_move) for s in chunk])
        with torch.no_grad():
            out.append(net(torch.as_tensor(feats, dtype=torch.float32))[1].numpy())
    return np.concatenate(out)


def rms(err: np.ndarray, w: np.ndarray) -> float:
    """Weighted root-mean-square of ``err``."""
    return round(float(np.sqrt(np.sum(w * err**2) / w.sum())), 4)


def run_value() -> None:
    """Is the leaf value what spoils the opening? Leaf errors measured exactly, then search on each.

    Trains a value net on positions from PPO's own (sampled) self-play, targeting the mover's final
    return (the Tessera recipe), early-stopped on held-out games. Then, for the PPO value head and
    the new net, the root-mean-square error at every player node against (a) the equilibrium value
    and (b) the exact value of PPO's self-play, weighted by how often PPO's self-play reaches each
    node, and unweighted over all nodes (search also visits positions self-play rarely reaches).
    Finally PUCT with PPO priors and two more leaf values (the new net, and the exact equilibrium
    values as an upper bound) is scored as in ``search``. ``key=value``: ``games`` (search games
    per setting), ``sims``, ``train_games``. Output: ``runs/spendthrift/value.json``, ``value.pt``.
    """
    args = dict(item.split("=") for item in sys.argv[2:])
    games = int(args.get("games", 300))
    budgets = [int(x) for x in args.get("sims", "50,200,800").split(",")]
    train_games = int(args.get("train_games", 100_000))
    checkpoint = "runs/spendthrift/spendthrift_ppo_200.pt"
    torch.set_num_threads(4)
    game = make("spendthrift")
    sol = solve(game)
    nodes = list(sol.choice)
    net = PPOAgent.from_checkpoint(checkpoint).net
    probs = net_probs(game, net, nodes, greedy=False)
    path, vpath = OUT / "value.json", OUT / "value.pt"
    results = json.loads(path.read_text("utf-8")) if path.exists() else {}
    actions = game.num_distinct_actions()
    if vpath.exists():
        ck = torch.load(vpath, map_location="cpu", weights_only=False)
        vnet = PolicyValueNet(ck["input_dim"], actions, (256, 256), "tanh")
        vnet.load_state_dict(ck["state_dict"])
        vnet.eval()
    else:
        t = time.perf_counter()
        x, y = self_play_positions(game, probs, train_games, per_game=4, seed=0)
        cut = len(x) - len(x) // 12  # positions are grouped by game, so this splits by game
        vnet, mse, curve = fit_value(x[:cut], y[:cut], x[cut:], y[cut:], actions)
        torch.save({"state_dict": vnet.state_dict(), "input_dim": vnet.input_dim}, vpath)
        results["training"] = {
            "games": train_games,
            "positions": len(x),
            "held_out_mse": round(mse, 4),
            "held_out_mse_constant": round(float(np.mean((y[cut:] - y[:cut].mean()) ** 2)), 4),
            "curve": curve,
            "minutes": round((time.perf_counter() - t) / 60, 1),
        }
        print(f"value net: {results['training']}", flush=True)
        path.write_text(json.dumps(results, indent=1), "utf-8")
    if "leaf_error" not in results:
        reach = reach_under(game, probs)
        sp = self_play_values(game, probs)
        eq = np.asarray([sol.value(s)[s.to_move] for s in nodes])
        spv = np.asarray([sp[s][s.to_move] for s in nodes])
        w = np.asarray([reach.get(s, 0.0) for s in nodes])
        ones = np.ones_like(w)
        lots = np.asarray([s.deck.bit_count() for s in nodes])
        rows = {}
        heads = {
            "ppo_head": net_values(game, net, nodes),
            "value_net": net_values(game, vnet, nodes),
        }
        for name, v in heads.items():
            rows[name] = {
                target: {
                    "on_path": rms(v - ref, w),
                    "everywhere": rms(v - ref, ones),
                    "on_path_by_lots_left": {
                        str(k): rms((v - ref)[lots == k], w[lots == k]) for k in range(game.lots)
                    },
                }
                for target, ref in (("equilibrium", eq), ("self_play", spv))
            }
        rows["self_play_vs_equilibrium"] = {
            "on_path": rms(spv - eq, w),
            "everywhere": rms(spv - eq, ones),
            "on_path_by_lots_left": {
                str(k): rms((spv - eq)[lots == k], w[lots == k]) for k in range(game.lots)
            },
        }
        results["leaf_error"] = rows
        print(f"leaf error: {json.dumps(rows)}", flush=True)
        path.write_text(json.dumps(results, indent=1), "utf-8")
    torch.set_num_threads(1)
    ppo = Greedy(net)
    search = results.setdefault("search", {})
    leaves = {"value_net": SplitEvaluator(net, vnet), "exact": ExactLeaves(net, sol)}
    for leaf, sims in [(leaf, sims) for leaf in leaves for sims in budgets]:
        label = f"{leaf}@{sims}"
        if label in search:
            continue
        agent = AlphaZeroAgent(
            leaves[leaf], PUCTConfig(simulations=sims, determinizations=1), label
        )
        search[label] = score_agent(game, sol, agent, ppo, games)
        print_row(label, search[label])
        path.write_text(json.dumps(results, indent=1), "utf-8")


class ExactMulti:
    """Priors from the PPO net, every player's exact equilibrium value at the leaves."""

    def __init__(self, net: PolicyValueNet, sol: Solution) -> None:
        """Priors from ``net``, values from ``sol``."""
        self.priors, self.sol = NetEvaluator(net), sol

    def evaluate_batch(self, game, states):  # noqa: ANN001, ANN201
        """Per-state priors and a ``(len(states), num_players)`` value array."""
        priors, _ = self.priors.evaluate_batch(game, states)
        return priors, np.stack([self.sol.value(s) for s in states])


def view_values(game: Spendthrift, net: PolicyValueNet, states: list) -> np.ndarray:
    """The value head read from every player's view: a ``(len(states), num_players)`` array."""
    n = game.num_players
    out = []
    for i in range(0, len(states), 4096):
        chunk = states[i : i + 4096]
        feats = np.stack([game.information_state_tensor(s, p) for s in chunk for p in range(n)])
        with torch.no_grad():
            v = net(torch.as_tensor(feats, dtype=torch.float32))[1].numpy()
        out.append(v.reshape(len(chunk), n))
    return np.concatenate(out)


def run_maxn() -> None:
    """Max-n PUCT (a value per player) on PPO priors, with three kinds of leaf value.

    Leaves: the PPO value head read from each player's view (``ppo_head``), a value net trained on
    every player's view of PPO's self-play positions (``value_net``, as in ``value``), and the
    exact equilibrium values (``exact``, the upper bound). Leaf errors are measured exactly over
    every (player node, player) pair, weighted by PPO's self-play reach and unweighted. Each agent
    plays ``games`` self-play games scored move by move against the solution, and ``games`` games
    in one seat (rotated) against greedy PPO in the others. ``key=value``: ``game``, ``games``,
    ``sims``, ``leaves`` (comma-separated), ``train_games``. Output:
    ``runs/spendthrift/maxn_<game>.json`` (value net ``maxn_<game>_value.pt``).
    """
    args = dict(item.split("=") for item in sys.argv[2:])
    name = args.get("game", "spendthrift3")
    games = int(args.get("games", 300))
    budgets = [int(x) for x in args.get("sims", "200,800").split(",")]
    kinds = args.get("leaves", "ppo_head,value_net,exact").split(",")
    train_games = int(args.get("train_games", 100_000))
    torch.set_num_threads(4)
    game = make(name)
    sol = solve(game)
    nodes = list(sol.choice)
    net = PPOAgent.from_checkpoint(f"runs/spendthrift/{name}_ppo_200.pt").net
    probs = net_probs(game, net, nodes, greedy=False)
    path, vpath = OUT / f"maxn_{name}.json", OUT / f"maxn_{name}_value.pt"
    results = json.loads(path.read_text("utf-8")) if path.exists() else {}
    actions = game.num_distinct_actions()
    if vpath.exists():
        ck = torch.load(vpath, map_location="cpu", weights_only=False)
        vnet = PolicyValueNet(ck["input_dim"], actions, (256, 256), "tanh")
        vnet.load_state_dict(ck["state_dict"])
        vnet.eval()
    else:
        t = time.perf_counter()
        x, y = self_play_positions(game, probs, train_games, per_game=4, seed=0, all_views=True)
        cut = len(x) - len(x) // 12  # rows are grouped by game, so this splits by game
        vnet, mse, curve = fit_value(x[:cut], y[:cut], x[cut:], y[cut:], actions)
        torch.save({"state_dict": vnet.state_dict(), "input_dim": vnet.input_dim}, vpath)
        results["training"] = {
            "games": train_games,
            "rows": len(x),
            "held_out_mse": round(mse, 4),
            "held_out_mse_constant": round(float(np.mean((y[cut:] - y[:cut].mean()) ** 2)), 4),
            "curve": curve,
            "minutes": round((time.perf_counter() - t) / 60, 1),
        }
        print(f"value net: {results['training']}", flush=True)
        path.write_text(json.dumps(results, indent=1), "utf-8")
    if "leaf_error" not in results:
        reach = reach_under(game, probs)
        sp = self_play_values(game, probs)
        eq = np.stack([sol.value(s) for s in nodes])
        spv = np.stack([sp[s] for s in nodes])
        w = np.repeat(np.asarray([reach.get(s, 0.0) for s in nodes])[:, None], game.players, 1)
        ones = np.ones_like(w)
        rows = {}
        heads = {
            "ppo_head": view_values(game, net, nodes),
            "value_net": view_values(game, vnet, nodes),
        }
        for key, v in heads.items():
            rows[key] = {
                target: {"on_path": rms(v - ref, w), "everywhere": rms(v - ref, ones)}
                for target, ref in (("equilibrium", eq), ("self_play", spv))
            }
        rows["self_play_vs_equilibrium"] = {
            "on_path": rms(spv - eq, w),
            "everywhere": rms(spv - eq, ones),
        }
        results["leaf_error"] = rows
        print(f"leaf error: {json.dumps(rows)}", flush=True)
        path.write_text(json.dumps(results, indent=1), "utf-8")
    torch.set_num_threads(1)
    ppo = Greedy(net)
    search = results.setdefault("search", {})
    if "ppo" not in search:
        search["ppo"] = score_agent(game, sol, ppo, ppo, games)
        print_row("ppo", search["ppo"])
    leaves = {
        "ppo_head": NetMultiEvaluator(net, net),
        "value_net": NetMultiEvaluator(net, vnet),
        "exact": ExactMulti(net, sol),
    }
    for kind, sims in [(kind, sims) for kind in kinds for sims in budgets]:
        label = f"{kind}@{sims}"
        if label in search:
            continue
        agent = MultiPUCTAgent(leaves[kind], sims, label)
        search[label] = score_agent(game, sol, agent, ppo, games)
        print_row(label, search[label])
        path.write_text(json.dumps(results, indent=1), "utf-8")


if __name__ == "__main__":
    stages = {
        "solve": run_solve,
        "tempo": run_tempo,
        "ppo": run_ppo,
        "search": run_search,
        "value": run_value,
        "maxn": run_maxn,
    }
    stages[sys.argv[1]]()
