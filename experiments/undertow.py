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
    uv run python -m experiments.undertow h2h [games=200 nets=label:path,...]
    uv run python -m experiments.undertow imitate [games=1000]
    uv run python -m experiments.undertow finetune [iterations=200]

``imitate`` records PIMC (8 x 40) playing itself (both sides; every decision's information
state, legal moves and PIMC's choice, plus the outcome), then trains the pointer network on it:
cross-entropy on PIMC's moves, squared error on the outcome. Saves ``runs/undertow/pimc_data.npz``
and ``runs/undertow/pointer_bc.pt``. ``finetune`` continues from it with PPO at learning rate
1e-4 (tag ``undertow/pointer_ft``).

``h2h`` seats every ordered pair of agents (the networks' final checkpoints, or
``nets=label:path,...``, plus the heuristic and PIMC 8 × 40) as Smugglers against Watch, ``games``
games each on 4 workers, and reports per-side win rates and each pair's score. Results go to
``runs/undertow/h2h.jsonl``.

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
    """``heuristic``, ``random``, ``net:<checkpoint>`` (sampled), or ``pimc:<worlds>x<sims>``."""
    if spec.startswith("net:"):
        from iirl.agents.ppo import PPOAgent

        return PPOAgent.from_checkpoint(spec[4:], name=Path(spec[4:]).stem, greedy=False)
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


def h2h() -> None:
    """Every ordered pair of agents, seated A as Smugglers and B as Watch, per-side win rates.

    A seat-swapped average hides everything here: with fixed play, whoever holds the stronger side
    wins, so every pair of games splits 1-1. So each seating is measured on its own, ``games``
    games each (networks play their sampled policy); A's score against B is the mean of its
    Smuggler win rate against B's Watch and its Watch win rate against B's Smugglers.
    """
    args = dict(a.split("=") for a in sys.argv[2:])
    games = int(args.get("games", 200))
    nets = {k: f"runs/undertow/{k}.pt" for k in ("ppo", "entity", "pointer")}
    if "nets" in args:  # label:path,label:path (e.g. a run's best checkpoint)
        nets = dict(item.split(":", 1) for item in args["nets"].split(","))
    specs = {k: f"net:{v}" for k, v in nets.items() if Path(v).exists()}
    specs["heuristic"] = "heuristic"
    specs["pimc"] = "pimc:8x40"
    out = Path("runs/undertow/h2h.jsonl")
    done = set()
    if out.exists():
        rows = map(json.loads, out.read_text("utf-8").splitlines())
        done = {(r["smugglers"], r["watch"]) for r in rows}
    rates: dict[tuple[str, str], float] = {}
    with mp.get_context("spawn").Pool(4) as pool:
        for a in specs:
            for b in specs:
                if a == b or (a, b) in done or {a, b} == {"heuristic", "pimc"}:
                    continue
                t0 = time.perf_counter()
                jobs = [(specs[a], specs[b], games // 4, 300 + k) for k in range(4)]
                total: collections.Counter = collections.Counter()
                for part in pool.map(_games, jobs):
                    total.update(part)
                n = sum(total.values())
                p = sum(v for k, v in total.items() if k.startswith("S:")) / n
                row = {
                    "smugglers": a,
                    "watch": b,
                    "games": n,
                    "smuggler_win_rate": round(p, 4),
                    "se": round((p * (1 - p) / n) ** 0.5, 4),
                    "minutes": round((time.perf_counter() - t0) / 60, 1),
                }
                print(json.dumps(row), flush=True)
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
    for r in map(json.loads, out.read_text("utf-8").splitlines()):
        rates[(r["smugglers"], r["watch"])] = r["smuggler_win_rate"]
    print("\nscore of row against column (mean of its two seatings):")
    names = list(specs)
    for a in names:
        cells = []
        for b in names:
            if (a, b) in rates and (b, a) in rates:
                cells.append(f"{b} {(rates[(a, b)] + 1 - rates[(b, a)]) / 2:.3f}")
        print(f"  {a:<10} " + "  ".join(cells))


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


def _record(args: tuple) -> tuple:
    """Worker: PIMC self-play games; every decision's features, legal mask, move and outcome."""
    n, seed = args
    import torch

    torch.set_num_threads(1)
    game = make("undertow")
    pimc_agent = agent("pimc:8x40")
    rng = np.random.default_rng(seed)
    feats, masks, moves, players, outcomes = [], [], [], [], []
    for _ in range(n):
        s = game.new_initial_state()
        start = len(feats)
        while not game.is_terminal(s):
            if game.is_chance_node(s):
                o = game.chance_outcomes(s)
                s = game.apply_action(s, o[rng.integers(len(o))][0])
                continue
            p = game.current_player(s)
            a = pimc_agent.step(game, s, rng)
            m = np.zeros(game.num_distinct_actions(), np.float32)
            m[game.legal_actions(s)] = 1.0
            feats.append(game.information_state_tensor(s, p))
            masks.append(m)
            moves.append(a)
            players.append(p)
            s = game.apply_action(s, a)
        r = game.returns(s)
        outcomes += [r[p] for p in players[start:]]
    return (
        np.asarray(feats, np.float32),
        np.asarray(masks, np.float32),
        np.asarray(moves, np.int64),
        np.asarray(outcomes, np.float32),
    )


def imitate() -> None:
    """PIMC self-play data, then the pointer network trained to imitate it."""
    import torch

    from iirl.nn.entity import EntityPolicyValueNet, save_checkpoint

    games = int(dict(a.split("=") for a in sys.argv[2:]).get("games", 1000))
    data_path, net_path = Path("runs/undertow/pimc_data.npz"), Path("runs/undertow/pointer_bc.pt")
    if not data_path.exists():
        t0 = time.perf_counter()
        chunks = Path("runs/undertow/pimc_data")  # one file per chunk, so a stop loses little
        chunks.mkdir(parents=True, exist_ok=True)
        todo = [k for k in range(16) if not (chunks / f"{k}.npz").exists()]
        with mp.get_context("spawn").Pool(4) as pool:
            jobs = [(games // 16, 500 + k) for k in todo]
            for k, part in zip(todo, pool.imap(_record, jobs), strict=True):
                np.savez_compressed(chunks / f"{k}.npz", x=part[0], m=part[1], a=part[2], z=part[3])
        parts = [np.load(chunks / f"{k}.npz") for k in range(16)]
        x, m, a, z = (np.concatenate([p[key] for p in parts]) for key in ("x", "m", "a", "z"))
        np.savez_compressed(data_path, x=x, m=m, a=a, z=z)
        print(
            f"{len(x):,} decisions from {games} PIMC games, "
            f"{(time.perf_counter() - t0) / 60:.0f} min",
            flush=True,
        )
    d = np.load(data_path)
    x, m, a, z = (torch.as_tensor(d[k]) for k in ("x", "m", "a", "z"))
    torch.manual_seed(0)
    torch.set_num_threads(4)
    net = EntityPolicyValueNet(x.shape[1], m.shape[1], **dict(POINTER))
    net.set_feature_stats(x.mean(0).numpy(), x.std(0).numpy())
    n = len(x)
    test = torch.arange(n) >= int(n * 0.9)  # the last tenth held out (later games)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    best, best_state, log = float("inf"), None, []
    for epoch in range(15):
        net.train()
        perm = torch.randperm(int((~test).sum()))
        for b in range(0, len(perm), 512):
            i = perm[b : b + 512]
            logits, v = net(x[i])
            logits = logits.masked_fill(m[i] == 0, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, a[i]) + 0.5 * ((v - z[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            logits, v = net(x[test])
            logits = logits.masked_fill(m[test] == 0, -1e9)
            ce = float(torch.nn.functional.cross_entropy(logits, a[test]))
            acc = float((logits.argmax(1) == a[test]).float().mean())
        log.append(
            {"epoch": epoch + 1, "held_out_ce": round(ce, 4), "held_out_match": round(acc, 4)}
        )
        print(json.dumps(log[-1]), flush=True)
        if ce < best:
            best, best_state = ce, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    save_checkpoint(net, net_path)
    Path("runs/undertow/imitate.json").write_text(json.dumps(log, indent=1), "utf-8")


def finetune() -> None:
    """PPO from the imitation network, at a lower learning rate."""
    from dataclasses import replace

    import torch

    from experiments.train_ventures_league import _train

    iterations = int(dict(a.split("=") for a in sys.argv[2:]).get("iterations", 200))
    torch.set_num_threads(4)
    _train(
        "runs/undertow/pointer_bc.pt",
        replace(PPO, iterations=iterations, arch="entity", entity=POINTER, lr=1e-4),
        "undertow/pointer_ft",
        save_every=50,
        game_name="undertow",
        heuristic=make_heuristic("undertow"),
        anchor_elo=1500.0,
    )


if __name__ == "__main__":
    stages = {"pimc": pimc, "h2h": h2h, "imitate": imitate, "finetune": finetune}
    if len(sys.argv) > 1 and sys.argv[1] in (*stages, "ppo", "entity", "pointer"):
        if sys.argv[1] in stages:
            stages[sys.argv[1]]()
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
