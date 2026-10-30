"""Heads-up limit Hold'em: Deep CFR, judged by head-to-head play on duplicate deals.

The full game has about 10^14 information sets, so nothing exact is possible: no exploitability,
no tabular CFR. Deep CFR (``agents/cfr/deep_cfr.py``) needs no tables. Each checkpoint's strategy
network plays a panel of fixed opponents on duplicate deals (each deal twice, hole cards swapped,
so the luck of the cards cancels within a pair), reported in big blinds per hand (and milli-big-
blinds, mbb, the field's unit).

The panel: uniform random, a calling station (always checks or calls), a maniac (always bets or
raises), and an equity heuristic (raises with a Monte Carlo win chance above 0.7 against a random
hand, calls when its equity beats the pot odds, else checks or folds).

Stages::

    uv run --extra openspiel python -m experiments.holdem train [checkpoints=10,25,50,100]
    uv run --extra openspiel python -m experiments.holdem match [deals=2000]
    uv run --extra openspiel python -m experiments.holdem lbr [deals=500 checkpoints=10,100]

``train`` is resumable (the solver's state is saved at every checkpoint to
``runs/holdem/deep_cfr_state.pt``; each checkpoint's strategy network to
``runs/holdem/deep_cfr_<iteration>.pt``). ``match`` plays every saved checkpoint against the panel
and appends to ``runs/holdem/matches.jsonl`` (skipping pairs already played). ``lbr`` plays a
local best response (Lisý & Bowling, 2017) against checkpoints: its win rate is a lower bound on
the checkpoint's exploitability (``runs/holdem/lbr.jsonl``).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import iirl.games  # noqa: F401  (registers "limit_holdem")
from iirl.agents.base import Agent
from iirl.agents.cfr.deep_cfr import DeepCFR, DeepCFRConfig, _mlp
from iirl.agents.random_agent import RandomAgent
from iirl.core.game import Game, State
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.games.holdem import CALL, FOLD, RAISE, LimitHoldem, hand_value
from iirl.games.holdem.game import BET_SIZE

OUT = Path("runs/holdem")
CONFIG = DeepCFRConfig(
    traversals=1000,
    hidden=(256, 256),
    advantage_steps=1000,
    strategy_steps=4000,
    batch_size=2048,
    lr=1e-3,
    memory=1_000_000,
    workers=4,  # keeps the machine cool
)


# --- Agents ------------------------------------------------------------------------------------
class NetPolicy(Agent):
    """Samples from a Deep CFR strategy network (masked softmax over legal actions)."""

    def __init__(self, path: str) -> None:
        """Load the network saved by ``train``."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = _mlp(ck["dim"], ck["actions"], tuple(ck["hidden"]))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.name = Path(path).stem

    def step(self, game: Game, state: State, rng: np.random.Generator) -> int:
        """Sample a legal action from the network's average strategy."""
        feat = torch.as_tensor(game.information_state_tensor(state, game.current_player(state)))
        with torch.no_grad():
            logits = self.net(feat[None])[0].numpy()
        legal = game.legal_actions(state)
        z = logits[legal] - logits[legal].max()
        p = np.exp(z) / np.exp(z).sum()
        return int(legal[rng.choice(len(legal), p=p)])


class CallingStation(Agent):
    """Always checks or calls."""

    name = "calling station"

    def step(self, game: Game, state: State, rng: np.random.Generator) -> int:
        """Check or call."""
        return CALL


class Maniac(Agent):
    """Always bets or raises (calls once the cap is reached)."""

    name = "maniac"

    def step(self, game: Game, state: State, rng: np.random.Generator) -> int:
        """Bet or raise if allowed."""
        return RAISE if RAISE in game.legal_actions(state) else CALL


class EquityHeuristic(Agent):
    """Monte Carlo win chance against a random hand, against the pot odds."""

    name = "equity heuristic"

    def __init__(self, samples: int = 200, raise_at: float = 0.7) -> None:
        """Set the sample count and the raising threshold."""
        self.samples, self.raise_at = samples, raise_at

    def equity(self, game: LimitHoldem, state, rng: np.random.Generator) -> float:  # noqa: ANN001
        """Win probability (ties count half) against a uniformly random opponent hand."""
        me = state.to_move
        own, board = game.hole(state, me), game.board(state)
        seen = set(own) | set(board)
        deck = np.array([c for c in range(52) if c not in seen])
        wins = 0.0
        for _ in range(self.samples):
            draw = rng.choice(deck, size=2 + 5 - len(board), replace=False)
            full = board + tuple(int(c) for c in draw[2:])
            mine = hand_value(own + full)
            theirs = hand_value((int(draw[0]), int(draw[1])) + full)
            wins += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        return wins / self.samples

    def step(self, game: Game, state: State, rng: np.random.Generator) -> int:
        """Raise strong hands, call when the price is right, otherwise check or fold."""
        assert isinstance(game, LimitHoldem)
        legal = game.legal_actions(state)
        eq = self.equity(game, state, rng)
        if eq >= self.raise_at and RAISE in legal:
            return RAISE
        owe = game.owe(state)
        if owe == 0:
            return CALL
        pot_odds = owe / (sum(state.contrib) + owe)
        return CALL if eq >= pot_odds else FOLD


PAIRS = np.array([(i, j) for i in range(52) for j in range(i + 1, 52)])  # 1,326 hole pairs


class LocalBestResponse(Agent):
    """Local best response (Lisý & Bowling, 2017) to a strategy network.

    Keeps a range: a weight per possible opponent hole pair, multiplied after each opponent
    action by the network's probability of that action holding that pair (and zeroed for pairs
    that clash with cards it can see). At its own decisions it estimates its equity against the
    range (Monte Carlo runouts) and the chance the opponent folds to a raise, and takes the action
    with the best value assuming the hand is then checked down:

    - fold: 0;
    - check or call: ``eq * pot - (1 - eq) * owe``;
    - bet or raise: ``fold% * pot + (1 - fold%) * (eq * (pot + bet) - (1 - eq) * (owe + bet))``.

    It knows the opponent's strategy, so its win rate is a lower bound on how much a best response
    could win: on the opponent's exploitability.
    """

    def __init__(self, path: str, samples: int = 200) -> None:
        """Load the opponent's strategy network."""
        self.policy = NetPolicy(path)
        self.samples = samples
        self.name = f"lbr({self.policy.name})"

    def reset(self, game: Game, player: int) -> None:
        """A fresh, uniform range."""
        self.me, self.weights, self.last = player, np.ones(len(PAIRS)), None

    def _probs(self, game: LimitHoldem, state, live: np.ndarray) -> np.ndarray:  # noqa: ANN001
        """The opponent's action probabilities at ``state`` for each live pair (rows of PAIRS)."""
        opp = 1 - self.me
        cards = list(state.cards)
        feats = []
        for a, b in PAIRS[live]:
            cards[2 * opp], cards[2 * opp + 1] = int(a), int(b)
            feats.append(game.information_state_tensor(replace(state, cards=tuple(cards)), opp))
        with torch.no_grad():
            logits = self.policy.net(torch.as_tensor(np.stack(feats))).numpy()
        legal = game.legal_actions(state)
        out = np.zeros((len(feats), game.num_distinct_actions()))
        z = logits[:, legal] - logits[:, legal].max(axis=1, keepdims=True)
        out[:, legal] = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return out

    def _prune(self, game: LimitHoldem, state) -> np.ndarray:  # noqa: ANN001
        """Zero the pairs that clash with visible cards; return the live mask."""
        seen = set(game.hole(state, self.me)) | set(game.board(state))
        clash = np.isin(PAIRS, list(seen)).any(axis=1)
        self.weights[clash] = 0.0
        return self.weights > 0

    def observe(self, game: Game, state: State, action: int | None) -> None:
        """Bayes-update the range on each opponent action."""
        assert isinstance(game, LimitHoldem)
        last = self.last
        self.last = state
        if action is None or last is None or game.is_chance_node(last) or game.is_terminal(last):
            return
        if last.to_move == self.me or len(last.cards) < 4:
            return
        live = self._prune(game, last)
        if live.any():
            self.weights[live] *= self._probs(game, last, live)[:, action]

    def step(self, game: Game, state: State, rng: np.random.Generator) -> int:
        """The action with the best value against the range."""
        assert isinstance(game, LimitHoldem)
        self.last = state
        live = self._prune(game, state)
        if self.weights.sum() <= 0:  # the network's line was impossible for every pair: reset
            self.weights[:] = 1.0
            live = self._prune(game, state)
        rng_pairs = self.weights[live] / self.weights[live].sum()
        own, board = game.hole(state, self.me), game.board(state)
        idx = rng.choice(np.flatnonzero(live), size=self.samples, p=rng_pairs)
        wins = 0.0
        for a, b in PAIRS[idx]:
            used = set(own) | set(board) | {int(a), int(b)}
            rest = np.array([c for c in range(52) if c not in used])
            full = board + tuple(
                int(c) for c in rng.choice(rest, size=5 - len(board), replace=False)
            )
            mine, theirs = hand_value(own + full), hand_value((int(a), int(b)) + full)
            wins += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        eq = wins / self.samples
        legal = game.legal_actions(state)
        pot, owe = sum(state.contrib), game.owe(state)
        values = {CALL: eq * pot - (1 - eq) * owe}
        if FOLD in legal:
            values[FOLD] = 0.0
        if RAISE in legal:
            after = game.apply_action(state, RAISE)
            fold = float(rng_pairs @ self._probs(game, after, live)[:, FOLD])
            bet = BET_SIZE[state.round]
            values[RAISE] = fold * pot + (1 - fold) * (eq * (pot + bet) - (1 - eq) * (owe + bet))
        return max(values, key=values.get)


def baseline(name: str) -> Agent:
    """A panel opponent by name (a picklable factory for duplicate matches)."""
    return {
        "random": lambda: RandomAgent("random"),
        "calling station": CallingStation,
        "maniac": Maniac,
        "equity heuristic": EquityHeuristic,
    }[name]()


PANEL = ("random", "calling station", "maniac", "equity heuristic")


# --- Stages ------------------------------------------------------------------------------------
def save_strategy(solver: DeepCFR, path: Path) -> None:
    """Fit and save the average-strategy network."""
    net = solver.train_strategy()
    torch.save(
        {
            "state_dict": net.state_dict(),
            "dim": solver.dim,
            "actions": solver.actions,
            "hidden": list(solver.config.hidden),
            "iteration": solver.t,
        },
        path,
    )


def train() -> None:
    """Deep CFR with checkpoints; resumable."""
    args = dict(item.split("=") for item in sys.argv[2:])
    checkpoints = [int(x) for x in args.get("checkpoints", "10,25,50,100").split(",")]
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    game = make("limit_holdem")
    solver = DeepCFR(game, CONFIG)
    state_path = OUT / "deep_cfr_state.pt"
    if state_path.exists():
        solver.load_state_dict(torch.load(state_path, weights_only=False))
        print(f"resuming at iteration {solver.t}", flush=True)
    log = OUT / "train.jsonl"
    try:
        for target in checkpoints:
            while solver.t < target:
                t0 = time.perf_counter()
                solver.iterate(1)
                print(
                    f"iteration {solver.t}: {time.perf_counter() - t0:.0f}s; advantage rows "
                    f"{[len(m) for m in solver.adv_mem]}, strategy rows {len(solver.strat_mem)}",
                    flush=True,
                )
            t0 = time.perf_counter()
            solver.strategy_net = None
            save_strategy(solver, OUT / f"deep_cfr_{solver.t}.pt")
            torch.save(solver.state_dict(), state_path)
            row = {
                "iteration": solver.t,
                "advantage_rows": [len(m) for m in solver.adv_mem],
                "strategy_rows": len(solver.strat_mem),
                "rows_seen": [m.seen for m in [*solver.adv_mem, solver.strat_mem]],
                "save_seconds": round(time.perf_counter() - t0, 1),
            }
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"checkpoint {solver.t} saved", flush=True)
    finally:
        solver.close()


def match() -> None:
    """Every saved checkpoint against every panel opponent, on duplicate deals."""
    args = dict(item.split("=") for item in sys.argv[2:])
    deals = int(args.get("deals", 2000))
    game = make("limit_holdem")
    log = OUT / "matches.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text("utf-8").splitlines():
            r = json.loads(line)
            done.add((r["agent"], r["opponent"], r["deals"]))
    paths = sorted(OUT.glob("deep_cfr_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for path in paths:
        for opponent in PANEL:
            if (path.stem, opponent, deals) in done:
                continue
            t0 = time.perf_counter()
            r = duplicate_match(
                game,
                (NetPolicy, {"path": str(path)}),
                (baseline, {"name": opponent}),
                deals,
                seed=7,
                workers=4,
            )
            row = {
                "agent": path.stem,
                "opponent": opponent,
                "deals": deals,
                "bb_per_hand": round(r.margin, 4),
                "bb_per_hand_se": round(r.margin_se, 4),
                "win_rate": round(r.score, 4),
                "minutes": round((time.perf_counter() - t0) / 60, 1),
            }
            print(json.dumps(row), flush=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


def lbr() -> None:
    """Local best response against chosen checkpoints, on duplicate deals."""
    args = dict(item.split("=") for item in sys.argv[2:])
    deals = int(args.get("deals", 500))
    checkpoints = [int(x) for x in args.get("checkpoints", "10,100").split(",")]
    game = make("limit_holdem")
    log = OUT / "lbr.jsonl"
    done = set()
    if log.exists():
        done = {
            (r["agent"], r["deals"]) for r in map(json.loads, log.read_text("utf-8").splitlines())
        }
    for it in checkpoints:
        path = OUT / f"deep_cfr_{it}.pt"
        if (path.stem, deals) in done or not path.exists():
            continue
        t0 = time.perf_counter()
        r = duplicate_match(
            game,
            (LocalBestResponse, {"path": str(path)}),
            (NetPolicy, {"path": str(path)}),
            deals,
            seed=11,
            workers=4,
        )
        row = {
            "agent": path.stem,
            "deals": deals,
            "lbr_bb_per_hand": round(r.margin, 4),
            "lbr_bb_per_hand_se": round(r.margin_se, 4),
            "minutes": round((time.perf_counter() - t0) / 60, 1),
        }
        print(json.dumps(row), flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    {"train": train, "match": match, "lbr": lbr}[sys.argv[1]]()
