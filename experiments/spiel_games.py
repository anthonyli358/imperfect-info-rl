"""OpenSpiel games (Liar's Dice, Kuhn and Leduc poker): our solvers vs OpenSpiel's, and PPO vs CFR+.

``crosscheck`` runs our vectorized CFR+ (``agents.cfr.graph_cfr``; simultaneous and alternating
updates) and OpenSpiel's C++ CFR+ for the same iterations, and scores each one's average strategy
with both exploitability evaluators: ours (``eval.exact_br``, through the ``games.spiel`` adapter)
and OpenSpiel's ``pyspiel.exploitability``. Agreement checks the adapter, our best response and
our CFR+ at once. Both report NashConv / 2: the payoff a perfect exploiter gains per game,
averaged over seats (payoffs are +1 / -1).

``ppo`` is the plan's headline experiment: the repo's PPO recipe (from scratch, then the
Nash-seeking phase with a magnet and past selves, as on Mini-Ventures) trained by self-play, and
every checkpoint's *exact* exploitability, greedy (as deployed) and sampled, next to CFR+'s.
Resumable (training continues from its last checkpoint; measurements are kept in
``runs/<game>/ppo_exact.json``). Every stage takes ``game=`` (default ``liars_dice``; also
``kuhn_poker``, ``leduc_poker``).

Run with (needs ``uv sync --extra openspiel``)::

    uv run python -m experiments.spiel_games crosscheck   # one process
    uv run python -m experiments.spiel_games ppo          # 4 workers
    uv run python -m experiments.spiel_games nfsp game=leduc_poker iterations=400
    uv run python -m experiments.spiel_games rebel game=kuhn_poker epochs=20
    uv run python -m experiments.spiel_games magnet tag=m1 magnet_coef=1.0 lr=3e-5 iterations=400

``nfsp`` continues from the PPO phase with NFSP (``selfplay/nfsp.py``: a PPO best response
against the anticipatory mix, plus a supervised average policy, which is what gets measured), the
equilibrium-seeking RL method the literature uses on poker. ``key=value``: ``tag``,
``iterations``, ``eta``.

``rebel`` trains full-width ReBeL (``selfplay/rebel.py``: CFR over depth-limited subgames of
public belief states, value networks at the cuts) and measures the exact exploitability of the
policy it plays. Subgames start at ``BOUNDARIES[game]``, depths with no chance node. ``key=value``:
``epochs``, ``cfr_iterations``, ``eval_samples`` (per cut, e.g. ``all,10,10``; see
``EVAL_SAMPLES``).

``magnet`` continues from the PPO phase with other Nash-phase settings (``key=value``: any
``PPOConfig`` field, plus ``tag`` and ``iterations``), measuring exact exploitability as it goes.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyspiel
import torch

import iirl.games  # noqa: F401  (registers the OpenSpiel games)
from experiments.train_ventures_league import _train
from iirl.agents.cfr.graph_cfr import GraphCFR
from iirl.agents.ppo import PPOAgent
from iirl.agents.random_agent import RandomAgent
from iirl.core.game import Game, State
from iirl.core.registry import make
from iirl.eval.exact_br import build_graph, exploitability, net_policy, uniform_policy
from iirl.selfplay import PPOConfig

GAME = "liars_dice"
OUT = Path("runs/liars_dice")


# Depths where ReBeL's subgames start (no chance node there), ending with the last depth.
BOUNDARIES = {"kuhn_poker": (2, 3, 5), "leduc_poker": (2, 7, 11), "liars_dice": (2, 5, 8, 15)}
# Stopping iterations the evaluated policy branches from at each cut ("all" is exact). Exact costs
# cfr_iterations^cuts solves of the deepest subgame: fine with two cuts, ~10^6 with Liar's Dice's
# three. A sample still gives a policy ReBeL could play, measured exactly. On Liar's Dice at
# epoch 5, all,10,10 (24 min) gave 0.230971 against exact's 0.230967.
EVAL_SAMPLES = {"liars_dice": "10,10,10"}


def parse_samples(text: str | None) -> tuple[int | None, ...] | None:
    """``"all,10,10"`` -> ``(None, 10, 10)``; ``None`` -> exact at every cut."""
    if text is None:
        return None
    return tuple(None if n == "all" else int(n) for n in text.split(","))


def prefix() -> str:
    """Checkpoint name prefix (Liar's Dice keeps its original ``ld``)."""
    return "ld" if GAME == "liars_dice" else GAME


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    checkpoints: tuple[int, ...] = (10, 30, 100, 300, 1000)
    ppo_iterations: int = 200
    nash_iterations: int = 200
    save_every: int = 25
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            # About 2-3 decisions per player a game, so many games per update (as Mini-Ventures).
            games_per_iter=4096,
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
            eval_games=2000,
            feature_sample_games=100,
            seed=0,
        )
    )


def table_of(game: Game, policy, states: Sequence[State]) -> pyspiel.TabularPolicy:  # noqa: ANN001
    """A batched policy over ``states`` as an OpenSpiel tabular policy (keyed by infostate)."""
    first: dict[str, State] = {}
    for s in states:
        if not game.is_terminal(s) and not game.is_chance_node(s):
            first.setdefault(game.information_state_string(s, game.current_player(s)), s)
    keys = list(first)
    out = {}
    for key, probs in zip(keys, policy(game, [first[k] for k in keys]), strict=True):
        legal = game.legal_actions(first[key])
        out[key] = [(a, float(probs[a])) for a in legal]
    return pyspiel.TabularPolicy(out)


def from_spiel(spiel_policy):  # noqa: ANN001, ANN201 - a BatchPolicy
    """An OpenSpiel policy as our batched policy."""

    def policy(game: Game, states: Sequence[State]) -> list[np.ndarray]:
        out = []
        for s in states:
            probs = np.zeros(game.num_distinct_actions())
            for a, p in spiel_policy.action_probabilities(s.inner).items():
                probs[a] = p
            out.append(probs)
        return out

    return policy


def crosscheck(config: Config) -> None:
    """Both solvers, both evaluators, at each checkpoint; saved to ``crosscheck.jsonl``."""
    OUT.mkdir(parents=True, exist_ok=True)
    game = make(GAME)
    spiel_game = game.spiel
    t0 = time.perf_counter()
    ours = GraphCFR(game)
    print(
        f"graph: {len(ours.graph.states):,} states, {len(ours.infoset_keys):,} infosets "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
    alt = GraphCFR(game, ours.graph, alternating=True)
    uni_ours = exploitability(game, uniform_policy, ours.graph)["exploitability"]
    uni_spiel = pyspiel.exploitability(
        spiel_game, table_of(game, uniform_policy, ours.graph.states)
    )
    print(f"uniform policy: ours {uni_ours:.6f}, OpenSpiel {uni_spiel:.6f}", flush=True)
    theirs = pyspiel.CFRPlusSolver(spiel_game)
    rows = []
    done = 0
    for target in config.checkpoints:
        t = time.perf_counter()
        ours.iterate(target - done)
        t_ours = time.perf_counter() - t
        t = time.perf_counter()
        alt.iterate(target - done)
        t_alt = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(target - done):
            theirs.evaluate_and_update_policy()
        t_theirs = time.perf_counter() - t
        done = target
        our_avg = ours.average_policy()
        our_table = table_of(game, our_avg, ours.graph.states)
        their_avg = theirs.average_policy()
        e_oo = exploitability(game, our_avg, ours.graph)
        e_ot = pyspiel.exploitability(spiel_game, our_table)
        e_to = exploitability(game, from_spiel(their_avg), ours.graph)
        e_tt = pyspiel.exploitability(spiel_game, their_avg)
        alt_avg = alt.average_policy()
        e_aa = exploitability(game, alt_avg, ours.graph)
        e_at = pyspiel.exploitability(spiel_game, table_of(game, alt_avg, ours.graph.states))
        row = {
            "iterations": target,
            "ours_by_ours": round(e_oo["exploitability"], 6),
            "ours_by_openspiel": round(e_ot, 6),
            "openspiel_by_ours": round(e_to["exploitability"], 6),
            "openspiel_by_openspiel": round(e_tt, 6),
            "ours_alternating_by_ours": round(e_aa["exploitability"], 6),
            "ours_alternating_by_openspiel": round(e_at, 6),
            # The first player's game value, from the best responses to our alternating average.
            "value_p0": round((e_aa["br_as_0"] - e_aa["br_as_1"]) / 2, 6),
            "seconds_ours": round(t_ours, 1),
            "seconds_ours_alternating": round(t_alt, 1),
            "seconds_openspiel": round(t_theirs, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        with (OUT / "crosscheck.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def measure(game: Game, graph, label: str, path: str, results: Path) -> None:  # noqa: ANN001
    """Exact exploitability (greedy and sampled) of one checkpoint, saved under ``label``."""
    exact = json.loads(results.read_text("utf-8")) if results.exists() else {}
    if label in exact or not Path(path).exists():
        return
    net = PPOAgent.from_checkpoint(path).net
    exact[label] = {
        "checkpoint": path,
        "greedy": round(
            exploitability(game, net_policy(net, greedy=True), graph)["exploitability"], 6
        ),
        "sampled": round(exploitability(game, net_policy(net), graph)["exploitability"], 6),
    }
    print(
        f"{label:<14} exact exploitability: greedy {exact[label]['greedy']:.4f}, "
        f"sampled {exact[label]['sampled']:.4f}",
        flush=True,
    )
    results.write_text(json.dumps(exact, indent=1), "utf-8")


def magnet(config: Config) -> None:
    """A Nash-phase variant from the PPO phase's final checkpoint (``key=value`` overrides)."""
    args = dict(item.split("=") for item in sys.argv[2:])
    tag = args.pop("tag")
    iterations = int(args.pop("iterations", config.nash_iterations))
    types = {f: type(getattr(config.ppo, f)) for f in args}
    overrides = {k: types[k](v) for k, v in args.items()}
    settings = replace(
        config.ppo,
        iterations=iterations,
        past_opponent_frac=0.3,
        magnet_interval=20,
        seed=2,
        **overrides,
    )
    _train(
        f"runs/{GAME}/{prefix()}_ppo.pt",
        settings,
        f"{GAME}/{prefix()}_{tag}",
        save_every=config.save_every,
        game_name=GAME,
        heuristic=RandomAgent(),
        anchor_elo=1000.0,
    )
    game = make(GAME)
    graph = build_graph(game)
    for i in range(config.save_every, iterations + 1, config.save_every):
        measure(
            game,
            graph,
            f"{tag} {i}",
            f"runs/{GAME}/{prefix()}_{tag}_{i}.pt",
            OUT / "ppo_exact.json",
        )


def rebel(config: Config) -> None:
    """Full-width ReBeL with exact evaluation every few epochs; curve to ``rebel.jsonl``."""
    from iirl.selfplay.rebel import FullWidthReBeL, ReBeLConfig

    torch.set_num_threads(4)  # keeps the machine cool
    args = dict(item.split("=") for item in sys.argv[2:])
    settings = ReBeLConfig(
        boundaries=BOUNDARIES[GAME],
        cfr_iterations=int(args.get("cfr_iterations", 100)),
        eval_iterations=int(args.get("cfr_iterations", 100)),
        eval_samples=parse_samples(args.get("eval_samples", EVAL_SAMPLES.get(GAME))),
        epochs=int(args.get("epochs", 20)),
        passes=3,
    )
    game = make(GAME)
    t0 = time.perf_counter()
    agent = FullWidthReBeL(game, settings)
    sizes = {d: level.num_public for d, level in agent.levels.items()}
    print(f"indexed in {time.perf_counter() - t0:.0f}s; public states per boundary {sizes}")
    curve = OUT / "rebel.jsonl"
    last = time.perf_counter()

    def on_epoch(agent: FullWidthReBeL, losses: dict[int, float]) -> None:
        nonlocal last
        row = {"epoch": agent.epoch, "loss": {str(d): round(v, 5) for d, v in losses.items()}}
        row["seconds"] = round(time.perf_counter() - last)
        if agent.epoch % 5 == 0 or agent.epoch == settings.epochs:
            t = time.perf_counter()
            row.update(agent.evaluate())
            row["eval_seconds"] = round(time.perf_counter() - t)
        print(json.dumps(row), flush=True)
        with curve.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        last = time.perf_counter()

    agent.train(on_epoch)


def nfsp(config: Config) -> None:
    """NFSP from the PPO phase's final checkpoint; exact exploitability of its average policy."""
    from iirl.selfplay.nfsp import NFSPConfig, train_nfsp

    args = dict(item.split("=") for item in sys.argv[2:])
    tag = args.get("tag", "nfsp")
    iterations = int(args.get("iterations", 400))
    eta = float(args.get("eta", 0.1))
    game = make(GAME)
    graph = build_graph(game)
    results = OUT / "ppo_exact.json"
    exact = json.loads(results.read_text("utf-8")) if results.exists() else {}
    settings = replace(config.ppo, iterations=iterations, lr=1e-4, seed=3, eval_interval=iterations)

    def on_average(iteration: int, net) -> None:  # noqa: ANN001 - PolicyValueNet
        label = f"{tag} {iteration}"
        exact[label] = {
            "checkpoint": None,
            "greedy": round(
                exploitability(game, net_policy(net, greedy=True), graph)["exploitability"], 6
            ),
            "sampled": round(exploitability(game, net_policy(net), graph)["exploitability"], 6),
        }
        print(
            f"{label:<14} exact exploitability: greedy {exact[label]['greedy']:.4f}, "
            f"sampled {exact[label]['sampled']:.4f}",
            flush=True,
        )
        results.write_text(json.dumps(exact, indent=1), "utf-8")

    avg = train_nfsp(
        game,
        {"random": RandomAgent()},
        NFSPConfig(ppo=settings, eta=eta, eval_interval=config.save_every, eval_games=200),
        PPOAgent.from_checkpoint(f"runs/{GAME}/{prefix()}_ppo.pt").net,
        on_average=on_average,
    )
    torch.save(
        {
            "state_dict": avg.state_dict(),
            "input_dim": avg.input_dim,
            "num_actions": avg.num_actions,
            "hidden": list(config.ppo.hidden),
        },
        OUT / f"{prefix()}_{tag}.pt",
    )


def ppo(config: Config) -> None:
    """Train PPO then Nash-seeking PPO; exact exploitability of every checkpoint."""
    OUT.mkdir(parents=True, exist_ok=True)
    common = {
        "save_every": config.save_every,
        "game_name": GAME,
        "heuristic": RandomAgent(),
        "anchor_elo": 1000.0,
    }
    base = _train(
        None,
        replace(config.ppo, iterations=config.ppo_iterations),
        f"{GAME}/{prefix()}_ppo",
        **common,
    )
    nash = replace(
        config.ppo,
        iterations=config.nash_iterations,
        lr=1e-4,
        past_opponent_frac=0.3,
        magnet_coef=0.1,
        magnet_interval=20,
        seed=1,
    )
    _train(base, nash, f"{GAME}/{prefix()}_nash", **common)

    game = make(GAME)
    graph = build_graph(game)
    results = OUT / "ppo_exact.json"
    exact = json.loads(results.read_text("utf-8")) if results.exists() else {}
    checkpoints = [
        *(
            (f"ppo {i}", f"runs/{GAME}/{prefix()}_ppo_{i}.pt")
            for i in range(config.save_every, config.ppo_iterations + 1, config.save_every)
        ),
        *(
            (f"nash {i}", f"runs/{GAME}/{prefix()}_nash_{i}.pt")
            for i in range(config.save_every, config.nash_iterations + 1, config.save_every)
        ),
    ]
    for label, path in checkpoints:
        if label in exact or not Path(path).exists():
            continue
        net = PPOAgent.from_checkpoint(path).net
        exact[label] = {
            "checkpoint": path,
            "greedy": round(
                exploitability(game, net_policy(net, greedy=True), graph)["exploitability"], 6
            ),
            "sampled": round(exploitability(game, net_policy(net), graph)["exploitability"], 6),
        }
        print(
            f"{label:<9} exact exploitability: greedy {exact[label]['greedy']:.4f}, "
            f"sampled {exact[label]['sampled']:.4f}",
            flush=True,
        )
        results.write_text(json.dumps(exact, indent=1), "utf-8")


if __name__ == "__main__":
    for item in [a for a in sys.argv[2:] if a.startswith("game=")]:
        GAME = item.split("=", 1)[1]
        OUT = Path(f"runs/{GAME}")
        sys.argv.remove(item)
    stages = {
        "crosscheck": crosscheck,
        "ppo": ppo,
        "magnet": magnet,
        "nfsp": nfsp,
        "rebel": rebel,
    }
    stages[sys.argv[1]](Config())
