"""Undertow balance check: who wins when fixed agents play each side.

Undertow is asymmetric, so balance is measured per seating: the heuristic (or random play) as the
Smugglers against the heuristic (or random play) as the Watch, ``n`` games each (default 400),
with how each game ended: the Contraband reached the Free Port, it was sunk, the Watch raided the
Cove, a side could not move, or a draw at the round limit.

    uv run python -m experiments.undertow [n]
    uv run python -m experiments.undertow pimc [games=200]   # PIMC on each side vs the heuristic

``pimc`` plays PIMC (determinized UCT, ``agents/mcts/pimc.py``, with the game's leaf value)
against the heuristic, seated as each side in turn, at two budgets, on 4 worker processes;
results are appended to ``runs/undertow/pimc.jsonl``.

``ppo`` trains flat-MLP PPO self-play (one network for both sides; the features say which side it
is) with the repo's recipe: potential shaping from the game's leaf value, a privileged critic.
Resumable; checkpoints every 50 iterations as ``runs/undertow/ppo_<iteration>.pt``; Elo against
the heuristic (1500), seat-balanced, in ``runs/elo_log.json`` under ``undertow/ppo``.

    uv run python -m experiments.undertow ppo [iterations=300]
    uv run python -m experiments.undertow entity [iterations=300]
    uv run python -m experiments.undertow pointer [iterations=300]
    uv run python -m experiments.undertow h2h [deals=500]

``h2h`` plays the final checkpoints of each network against each other, and each against PIMC
(8 × 40), on duplicate deals: every deal twice with the seats swapped, so each agent plays both
sides (``eval.duplicate``; 4 workers). Results go to ``runs/undertow/h2h.jsonl``.

``entity`` is the same run with the entity network (``nn/entity.py``: a transformer over one token
per piece, each own piece scoring its destinations) in place of the MLP, at the same budget:
the Stage 7 ablation. Tag ``undertow/entity``. ``pointer`` adds the 14 regions as tokens and scores
moves as piece · region (a real pointer; the first gap in the warcraftsim review). Tag
``undertow/pointer``.
"""

from __future__ import annotations

import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

import iirl.games  # noqa: F401
from iirl.agents.mcts.pimc import MCTSConfig, PIMCAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.registry import make, make_heuristic
from iirl.games.undertow import Undertow, UndertowState
from iirl.games.undertow.board import CONTRABAND, COVE, RAID, SMUGGLERS, WATCH
from iirl.games.undertow.game import REMOVED
from iirl.selfplay import PPOConfig

PPO = PPOConfig(
    games_per_iter=1024,
    num_envs=256,
    num_workers=4,  # keeps the machine cool
    epochs=4,
    minibatch_size=1024,
    lr=3e-4,
    hidden=(256, 256),
    privileged_critic=True,
    win_bonus=0.0,  # the payoff is already win / lose
    reward_scale=1.0,
    eval_interval=25,
    eval_games=400,
    feature_sample_games=100,
    seed=0,
)


def ending(game: Undertow, s: UndertowState) -> str:
    """How a finished game ended."""
    if s.winner < 0:
        return "draw"
    contraband = s.ident[SMUGGLERS].index(CONTRABAND)
    if s.winner == SMUGGLERS:
        return "S: contraband in" if s.pos[SMUGGLERS][contraband] >= 0 else "S: watch stuck"
    if s.pos[SMUGGLERS][contraband] == REMOVED:
        return "W: contraband sunk"
    if len(game.tokens_at(s, WATCH, COVE)) >= RAID:
        return "W: raid"
    return "W: smugglers stuck"


def play(game: Undertow, smugglers, watch, n: int, seed: int) -> dict:  # noqa: ANN001
    """``n`` games with the given agents seated; counts of each ending."""
    rng = np.random.default_rng(seed)
    out: collections.Counter = collections.Counter()
    for _ in range(n):
        s = game.new_initial_state()
        smugglers.reset(game, SMUGGLERS)
        watch.reset(game, WATCH)
        while not game.is_terminal(s):
            if game.is_chance_node(s):
                o = game.chance_outcomes(s)
                s = game.apply_action(s, o[rng.integers(len(o))][0])
                continue
            agent = smugglers if game.current_player(s) == SMUGGLERS else watch
            s = game.apply_action(s, agent.step(game, s, rng))
        out[ending(game, s)] += 1
    return dict(sorted(out.items()))


def agent(spec: str):  # noqa: ANN201
    """``heuristic``, ``random``, or ``pimc:<worlds>x<sims>``."""
    if spec == "heuristic":
        return make_heuristic("undertow")
    if spec == "random":
        return RandomAgent("random")
    worlds, sims = (int(x) for x in spec.split(":")[1].split("x"))
    config = MCTSConfig(num_determinizations=worlds, sims_per_world=sims, max_rollout_depth=20)
    return PIMCAgent(config, name=spec)


def _games(args: tuple) -> dict:
    smugglers, watch, n, seed = args
    import torch

    torch.set_num_threads(1)
    return play(make("undertow"), agent(smugglers), agent(watch), n, seed)


def pimc() -> None:
    """PIMC seated as each side against the heuristic, two budgets."""
    games = int(dict(a.split("=") for a in sys.argv[2:]).get("games", 200))
    out = Path("runs/undertow/pimc.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(4) as pool:
        for budget in ("pimc:8x40", "pimc:16x80"):
            for smugglers, watch in ((budget, "heuristic"), ("heuristic", budget)):
                t0 = time.perf_counter()
                jobs = [(smugglers, watch, games // 4, 100 + k) for k in range(4)]
                total: collections.Counter = collections.Counter()
                for part in pool.map(_games, jobs):
                    total.update(part)
                n = sum(total.values())
                wins = sum(v for k, v in total.items() if k.startswith("S:"))
                p = wins / n
                row = {
                    "smugglers": smugglers,
                    "watch": watch,
                    "games": n,
                    "smuggler_wins": wins,
                    "smuggler_win_rate": round(p, 4),
                    "se": round((p * (1 - p) / n) ** 0.5, 4),
                    "endings": dict(total),
                    "minutes": round((time.perf_counter() - t0) / 60, 1),
                }
                print(json.dumps(row), flush=True)
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")


ENTITY = (("tokens", 18), ("token_dim", 25), ("own_tokens", 9), ("regions", 14))


POINTER = (*ENTITY, ("region_tokens", 1))
TAGS = {"mlp": "undertow/ppo", "entity": "undertow/entity", "pointer": "undertow/pointer"}


def policy(path: str):  # noqa: ANN201
    """A trained network as a greedy agent (a picklable factory for duplicate matches)."""
    from iirl.agents.ppo import PPOAgent

    return PPOAgent.from_checkpoint(path, name=Path(path).stem, greedy=True)


def h2h() -> None:
    """Every pair of final networks, and each against PIMC, on duplicate deals."""
    from iirl.eval.duplicate import duplicate_match

    deals = int(dict(a.split("=") for a in sys.argv[2:]).get("deals", 500))
    nets = {k: f"runs/undertow/{k}.pt" for k in ("ppo", "entity", "pointer")}
    nets = {k: v for k, v in nets.items() if Path(v).exists()}
    specs = {k: (policy, {"path": v}) for k, v in nets.items()}
    specs["pimc"] = (agent, {"spec": "pimc:8x40"})
    names = list(specs)
    out = Path("runs/undertow/h2h.jsonl")
    done = set()
    if out.exists():
        done = {(r["a"], r["b"]) for r in map(json.loads, out.read_text("utf-8").splitlines())}
    game = make("undertow")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if (a, b) in done:
                continue
            t0 = time.perf_counter()
            r = duplicate_match(game, specs[a], specs[b], deals, seed=17, workers=4)
            row = {
                "a": a,
                "b": b,
                "deals": deals,
                "score_a": round(r.score, 4),
                "score_a_se": round(r.score_se, 4),
                "minutes": round((time.perf_counter() - t0) / 60, 1),
            }
            print(json.dumps(row), flush=True)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")


def ppo(arch: str = "mlp") -> None:
    """PPO self-play (flat MLP, or the entity network), rated against the heuristic."""
    from dataclasses import replace

    import torch

    from experiments.train_ventures_league import _train

    iterations = int(dict(a.split("=") for a in sys.argv[2:]).get("iterations", 300))
    torch.set_num_threads(4)
    Path("runs/undertow").mkdir(parents=True, exist_ok=True)
    _train(
        None,
        replace(
            PPO,
            iterations=iterations,
            arch="mlp" if arch == "mlp" else "entity",
            entity={"mlp": (), "entity": ENTITY, "pointer": POINTER}[arch],
        ),
        TAGS[arch],
        save_every=50,
        game_name="undertow",
        heuristic=make_heuristic("undertow"),
        anchor_elo=1500.0,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("pimc", "ppo", "entity", "pointer", "h2h"):
        if sys.argv[1] in ("pimc", "h2h"):
            {"pimc": pimc, "h2h": h2h}[sys.argv[1]]()
        else:
            ppo("mlp" if sys.argv[1] == "ppo" else sys.argv[1])
        sys.exit()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    game = make("undertow")
    h, r = make_heuristic("undertow"), RandomAgent("random")
    for name, sm, wa in [
        ("heuristic vs heuristic", h, h),
        ("heuristic Smugglers vs random Watch", h, r),
        ("random Smugglers vs heuristic Watch", r, h),
        ("random vs random", r, r),
    ]:
        print(f"{name}: {play(game, sm, wa, n, 1)}", flush=True)
