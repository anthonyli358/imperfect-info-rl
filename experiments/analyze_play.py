"""How do the best agents actually play? Behaviour statistics vs the human-style heuristic.

For each game, the strongest trained agent plays the game's heuristic (which encodes the standard
human advice for that game) for ``games`` games, both seats, and every decision is logged with the
features a player would talk about: when routes are opened, what is discarded, when camels are
taken, how big the sales are, and so on. Both players' statistics come from the same games, so the
agent's style can be read against the "natural" human style side by side.

Writes ``runs/analysis_<game>.json`` and prints a table per game.

Run with::

    uv run python -m experiments.analyze_play
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import iirl.games  # noqa: F401  (registers the built-in games)
from iirl.agents.base import Agent
from iirl.agents.ppo import PPOAgent
from iirl.core.game import Game
from iirl.core.registry import make, make_heuristic


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    games: int = 200
    ventures_agent: str = "runs/league_control.pt"
    camel_agent: str = "runs/camel_nash.pt"
    seed: int = 0


class Tally:
    """Per-player running sums and counts; ``mean(key)`` averages over the recorded samples."""

    def __init__(self) -> None:
        """Start empty."""
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, key: str, value: float) -> None:
        """Record one sample of ``key``."""
        self.sums[key] += value
        self.counts[key] += 1

    def mean(self, key: str) -> float:
        """Return the mean of ``key`` (nan if never recorded)."""
        return self.sums[key] / self.counts[key] if self.counts[key] else float("nan")


def _play(game: Game, seats: tuple[Agent, Agent], rng: np.random.Generator, on_move) -> object:  # noqa: ANN001
    """Play one game, calling ``on_move(state, action, player, next_state)`` for each decision."""
    for p, agent in enumerate(seats):
        agent.reset(game, p)
    state = game.new_initial_state()
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            a = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            state = game.apply_action(state, a)
            continue
        p = game.current_player(state)
        a = seats[p].step(game, state, rng)
        nxt = game.apply_action(state, a)
        on_move(state, a, p, nxt)
        state = nxt
    return state


# --- Ventures -------------------------------------------------------------------------------------
def analyze_ventures(game: Game, agent: Agent, heuristic: Agent, n: int, rng) -> dict:  # noqa: ANN001
    """Route, backing, discard and draw statistics for the agent and the heuristic."""
    from iirl.games.ventures.cards import NUM_SUITS, is_backing, rank_of, suit_of
    from iirl.games.ventures.game import (
        DISCARD_OFFSET,
        DRAW_DECK_ACTION,
        route_backing_count,
        route_max_number,
    )
    from iirl.games.ventures.state import Phase

    tallies = {"agent": Tally(), "heuristic": Tally()}
    for i in range(n):
        agent_seat = i % 2
        seats = (agent, heuristic) if agent_seat == 0 else (heuristic, agent)
        role = {agent_seat: "agent", 1 - agent_seat: "heuristic"}
        per_game = {p: defaultdict(float) for p in (0, 1)}
        opened_turns: dict[int, list[int]] = {0: [], 1: []}

        def on_move(s, a, p, nxt, per_game=per_game, opened_turns=opened_turns) -> None:  # noqa: ANN001
            g, deck = per_game[p], sum(s.deck)
            if s.phase == Phase.PLAY and a < DISCARD_OFFSET:
                suit = suit_of(a)
                if not s.routes[p][suit]:
                    g["routes_opened"] += 1
                    opened_turns[p].append(s.turn)
                    g["late_openings"] += deck < 15
                if is_backing(a):
                    g["backings"] += 1
                g["plays"] += 1
            elif s.phase == Phase.PLAY:
                card = a - DISCARD_OFFSET
                g["discards"] += 1
                opp_route = s.routes[1 - p][suit_of(card)]
                playable = (
                    route_max_number(opp_route) == 0
                    if is_backing(card)
                    else rank_of(card) > route_max_number(opp_route)
                )
                g["gift_discards"] += bool(opp_route) and playable  # feeds a route they started
            elif s.phase == Phase.DRAW:
                if a == DRAW_DECK_ACTION:
                    g["deck_draws"] += 1
                else:
                    g["discard_takes"] += 1
                    g["late_discard_takes"] += deck <= 5  # slowing the clock near the end
            g["decisions"] += 1

        final = _play(game, seats, rng, on_move)
        margin0 = game.returns(final)[0]
        for p in (0, 1):
            t, g = tallies[role[p]], per_game[p]
            routes = [final.routes[p][s] for s in range(NUM_SUITS) if final.routes[p][s]]
            t.add("win", float((margin0 if p == 0 else -margin0) > 0))
            t.add("score", float(game._score(final, p)))
            t.add("routes_opened", g["routes_opened"])
            t.add("first_open_turn", opened_turns[p][0] if opened_turns[p] else float("nan"))
            t.add("late_openings", g["late_openings"])
            t.add("backings_played", g["backings"])
            t.add("routes_with_backing", sum(1 for r in routes if route_backing_count(r)))
            t.add("discard_rate", g["discards"] / max(g["discards"] + g["plays"], 1))
            t.add("gift_discard_rate", g["gift_discards"] / max(g["discards"], 1))
            t.add(
                "discard_take_rate",
                g["discard_takes"] / max(g["discard_takes"] + g["deck_draws"], 1),
            )
            t.add("late_discard_takes", g["late_discard_takes"])
            t.add("eight_card_routes", sum(1 for r in routes if len(r) >= 8))
            for r in routes:
                val = _ventures_route_value(r)
                t.add("route_value", val)
                t.add("losing_route", float(val < 0))
                t.add("route_len", len(r))
    return {k: {m: round(t.mean(m), 3) for m in t.counts} for k, t in tallies.items()}


def _ventures_route_value(route) -> int:  # noqa: ANN001
    from iirl.games.ventures.cards import BONUS_CARDS, BONUS_POINTS, START_COST
    from iirl.games.ventures.game import route_backing_count, route_number_sum

    value = (route_number_sum(route) - START_COST) * (1 + route_backing_count(route))
    return value + (BONUS_POINTS if len(route) >= BONUS_CARDS else 0)


# --- Camel Stocks ---------------------------------------------------------------------------------
def analyze_camel_stocks(game: Game, agent: Agent, heuristic: Agent, n: int, rng) -> dict:  # noqa: ANN001
    """Action mix, sale, camel and exchange statistics for the agent and the heuristic."""
    from iirl.games.camel_stocks.cards import CAMEL, HAND_LIMIT, PRECIOUS
    from iirl.games.camel_stocks.game import (
        EX_DONE,
        EX_GIVE_OFFSET,
        EX_TAKE_OFFSET,
        MAX_SELL,
        SELL_OFFSET,
        TAKE_CAMELS,
    )
    from iirl.games.camel_stocks.state import Phase

    tallies = {"agent": Tally(), "heuristic": Tally()}
    for i in range(n):
        agent_seat = i % 2
        seats = (agent, heuristic) if agent_seat == 0 else (heuristic, agent)
        role = {agent_seat: "agent", 1 - agent_seat: "heuristic"}
        per_game = {p: defaultdict(float) for p in (0, 1)}
        first_sale: dict[int, int] = {}  # good -> player who sold it first

        def on_move(s, a, p, nxt, per_game=per_game, role=role, first_sale=first_sale) -> None:  # noqa: ANN001
            g, t = per_game[p], tallies[role[p]]
            if s.phase == Phase.PLAY:
                g["turns"] += 1
                t.add("hand_size_at_turn", sum(s.hands[p]))
                if a < TAKE_CAMELS:
                    g["take_one"] += 1
                    t.add("take_precious_share", float(a in PRECIOUS))
                elif a == TAKE_CAMELS:
                    g["take_camels"] += 1
                    t.add("camels_per_take", s.market[CAMEL])
                    t.add(
                        "opp_hand_full_when_taking_camels",
                        float(sum(s.hands[1 - p]) >= HAND_LIMIT - 1),
                    )
                elif a < EX_TAKE_OFFSET:
                    good, k = divmod(a - SELL_OFFSET, MAX_SELL)
                    g["sell"] += 1
                    t.add("sale_size", k + 1)
                    t.add("sale_size_precious" if good in PRECIOUS else "sale_size_common", k + 1)
                    t.add("bonus_sale", float(k + 1 >= 3))
                    t.add("sale_points", nxt.points[p] - s.points[p])
                    t.add("sold_with_full_hand", float(sum(s.hands[p]) >= HAND_LIMIT))
                    first_sale.setdefault(good, p)
                else:
                    g["exchange"] += 1
            elif s.phase == Phase.EX_GIVE:
                t.add("exchange_gives_camel", float(a - EX_GIVE_OFFSET == CAMEL))
            if s.phase in (Phase.EX_TAKE,) and a == EX_DONE:
                t.add("exchange_size", sum(s.taken))

        final = _play(game, seats, rng, on_move)
        margin0 = game.returns(final)[0]
        for p in (0, 1):
            t, g = tallies[role[p]], per_game[p]
            turns = max(g["turns"], 1)
            t.add("win", float((margin0 if p == 0 else -margin0) > 0))
            t.add("score", float(game.score(final, p)))
            for kind in ("take_one", "take_camels", "sell", "exchange"):
                t.add(f"share_{kind}", g[kind] / turns)
            t.add("herd_at_end", final.herds[p])
            t.add("camel_bonus", float(final.herds[p] > final.herds[1 - p]))
            t.add(
                "first_seller_share",
                sum(1 for q in first_sale.values() if q == p) / max(len(first_sale), 1),
            )
            t.add("goods_held_at_end", sum(final.hands[p]))
            t.add("turns", g["turns"])
    return {k: {m: round(t.mean(m), 3) for m in t.counts} for k, t in tallies.items()}


def main(config: Config) -> None:
    """Run both analyses, print them side by side, and save them."""
    rng = np.random.default_rng(config.seed)
    runs = (
        ("ventures", config.ventures_agent, analyze_ventures),
        ("camel_stocks", config.camel_agent, analyze_camel_stocks),
    )
    for name, checkpoint, analyze in runs:
        game = make(name)
        result = analyze(
            game, PPOAgent.from_checkpoint(checkpoint), make_heuristic(name), config.games, rng
        )
        result["games"] = config.games
        result["agent_checkpoint"] = checkpoint
        Path(f"runs/analysis_{name}.json").write_text(json.dumps(result, indent=2), "utf-8")
        print(f"\n=== {name}: {config.games} games, agent ({checkpoint}) vs heuristic ===")
        print(f"{'statistic':<36}{'agent':>10}{'heuristic':>11}")
        for key in sorted(result["agent"]):
            other = result["heuristic"].get(key, float("nan"))
            print(f"{key:<36}{result['agent'][key]:>10}{other:>11}")


if __name__ == "__main__":
    main(Config())
