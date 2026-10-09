"""Distil the rollout agent's endgame choices into the Ventures policy network.

``diverse_sym`` plus opponent-robust endgame rollouts (``agents/rollout.py``) is the best Ventures
agent, but it costs 10-20 s a game. Its search only changes a move or two per game, all in the
last few turns, so the gain may fit in the policy itself: a network that plays those moves
directly is free at play time and fast enough to be attacked by a trained exploiter.

Stages (each saves under ``runs/distill/``)::

    uv run python -m experiments.distill_rollout collect   # rollout-agent games, logged
    uv run python -m experiments.distill_rollout train     # fine-tune a copy of the policy
    uv run python -m experiments.distill_rollout evaluate  # head to head and the style panel

``collect`` plays the rollout agent against a mixed field (the policy itself and the heuristic
styles) and logs every searched decision with the move the search played, plus a sample of its
other decisions with the policy's own move. ``train`` fine-tunes a copy of the policy to
reproduce the logged moves: cross-entropy on the searched decisions (those where the search
overrode the policy are up-weighted), and on the other decisions a penalty for moving away from
the original policy, so that nothing outside the endgame changes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.agents.ppo import PPOAgent
from iirl.agents.rollout import rollout_agent
from iirl.core.registry import make, make_heuristic
from iirl.eval.duplicate import duplicate_match

STYLES = (
    "ventures",
    "ventures_careful",
    "ventures_tight",
    "ventures_aggressive",
    "ventures_hoarder",
)


@dataclass(frozen=True)
class Config:
    """Experiment settings."""

    blueprint: str = "runs/diverse_sym.pt"
    out: str = "runs/distill"
    games: int = 6000
    games_per_shard: int = 50
    workers: int = 10
    other_keep: float = 0.08  # share of the non-searched decisions that are logged
    switch_weight: float = 8.0
    anchor_weight: float = 1.0
    epochs: int = 30
    lr: float = 3e-5
    batch_size: int = 512
    holdout: float = 0.1


def collect_shard(args: tuple) -> dict[str, np.ndarray]:
    """Play rollout-agent games and return its logged decisions (``args``: seeds, config)."""
    seeds, config = args
    torch.set_num_threads(1)
    game = make("ventures")
    agent = rollout_agent(blueprint=config.blueprint, deck_max=6, robust=True, seed=seeds[0])
    policy = PPOAgent.from_checkpoint(config.blueprint, greedy=True)
    field = [policy] + [make_heuristic(name) for name in STYLES]
    n = game.num_distinct_actions()
    feat, mask, target, base, searched, gains, which = [], [], [], [], [], [], []

    def record(state, legal, default, played, gain, was_searched, seed) -> None:  # noqa: ANN001
        feat.append(game.information_state_tensor(state, game.current_player(state)))
        m = np.zeros(n, dtype=bool)
        m[legal] = True
        g = np.zeros(n, dtype=np.float32)
        if gain is not None:
            g[legal] = gain
        mask.append(m)
        target.append(played)
        base.append(default)
        searched.append(was_searched)
        gains.append(g)
        which.append(seed)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Half the games against the policy itself, the rest against a random style.
        opponent = field[0] if rng.random() < 0.5 else field[int(rng.integers(1, len(field)))]
        seat = int(rng.integers(2))
        agent.log = []
        state = game.new_initial_state()
        while not game.is_terminal(state):
            if game.is_chance_node(state):
                outcomes = game.chance_outcomes(state)
                action = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
            elif game.current_player(state) == seat:
                before = len(agent.log)
                action = agent.step(game, state, rng)
                if len(agent.log) == before and rng.random() < config.other_keep:
                    record(state, game.legal_actions(state), action, action, None, False, seed)
            else:
                action = opponent.step(game, state, rng)
            state = game.apply_action(state, action)
        for s, legal, default, played, gain in agent.log:
            record(s, legal, default, played, gain, True, seed)
    return {
        "feat": np.asarray(feat, dtype=np.float32),
        "mask": np.asarray(mask),
        "target": np.asarray(target, dtype=np.int64),
        "base": np.asarray(base, dtype=np.int64),
        "searched": np.asarray(searched),
        "gain": np.asarray(gains, dtype=np.float32),
        "game": np.asarray(which, dtype=np.int64),
    }


def collect(config: Config) -> None:
    """Collect in shards (existing shards are kept, so the stage can be resumed)."""
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for shard in range(config.games // config.games_per_shard):
        path = out / f"data_{shard:04d}.npz"
        if not path.exists():
            first = 5_000_000 + shard * config.games_per_shard
            jobs.append((path, (list(range(first, first + config.games_per_shard)), config)))
    print(f"{len(jobs)} shards of {config.games_per_shard} games to collect", flush=True)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(config.workers) as pool:
        results = pool.imap(collect_shard, [job for _, job in jobs])
        for k, ((path, _), shard) in enumerate(zip(jobs, results, strict=True)):
            np.savez_compressed(path, **shard)
            if (k + 1) % 10 == 0:
                rate = (k + 1) * config.games_per_shard / (time.perf_counter() - t0)
                print(f"{k + 1} shards, {rate:.2f} games/s", flush=True)


def load(config: Config) -> dict[str, np.ndarray]:
    """All collected shards, concatenated."""
    shards = [np.load(p) for p in sorted(Path(config.out).glob("data_*.npz"))]
    return {k: np.concatenate([s[k] for s in shards]) for k in shards[0].files}


def train(config: Config) -> None:
    """Fine-tune a copy of the policy on the logged moves; report held-out agreement."""
    data = load(config)
    searched, switched = data["searched"], data["target"] != data["base"]
    print(
        f"{len(searched):,} decisions from {len(np.unique(data['game'])):,} games: "
        f"{searched.sum():,} searched, {switched.sum():,} of them overrides "
        f"(mean rollout gain of an override {data['gain'][switched].max(axis=1).mean():.1f} points)"
    )
    rng = np.random.default_rng(0)
    games = np.unique(data["game"])
    held = np.isin(data["game"], rng.choice(games, int(config.holdout * len(games)), False))
    x = torch.as_tensor(data["feat"])
    mask = torch.as_tensor(data["mask"])
    target = torch.as_tensor(data["target"])
    weight = torch.as_tensor(
        np.where(switched, config.switch_weight, 1.0) * searched, dtype=torch.float32
    )
    anchor = torch.as_tensor(~searched, dtype=torch.float32) * config.anchor_weight
    net = PPOAgent.from_checkpoint(config.blueprint).net
    original = PPOAgent.from_checkpoint(config.blueprint).net

    def log_probs(model, rows) -> torch.Tensor:  # noqa: ANN001
        logits = model(x[rows])[0].masked_fill(~mask[rows], -1e9)
        return torch.log_softmax(logits, dim=1)

    with torch.no_grad():
        reference = torch.cat(
            [log_probs(original, slice(k, k + 8192)) for k in range(0, len(x), 8192)]
        )

    def agreement(rows: np.ndarray) -> float:
        with torch.no_grad():
            picks = torch.cat([log_probs(net, torch.as_tensor(r)) for r in np.array_split(rows, 8)])
        return float((picks.argmax(dim=1) == target[rows]).float().mean())

    test_switch = np.flatnonzero(held & switched)
    test_keep = np.flatnonzero(held & searched & ~switched)
    test_other = np.flatnonzero(held & ~searched)
    train_rows = np.flatnonzero(~held)
    report = []

    def measure(epoch: int) -> None:
        row = {
            "epoch": epoch,
            "overrides_matched": round(agreement(test_switch), 3),
            "searched_kept_matched": round(agreement(test_keep), 3),
            "other_moves_unchanged": round(agreement(test_other), 3),
        }
        report.append(row)
        print(row, flush=True)

    print(f"held out: {len(test_switch)} overrides, {len(test_keep)} kept, {len(test_other)} other")
    measure(0)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr)
    net.train()
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(train_rows)
        for k in range(0, len(order), config.batch_size):
            rows = torch.as_tensor(order[k : k + config.batch_size])
            lp = log_probs(net, rows)
            imitation = -(lp.gather(1, target[rows, None])[:, 0] * weight[rows]).sum()
            ref = reference[rows]
            drift = ((ref.exp() * (ref - lp)).sum(dim=1) * anchor[rows]).sum()
            loss = (imitation + drift) / len(rows)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0:
            net.eval()
            measure(epoch)
            net.train()
    ck = torch.load(config.blueprint, map_location="cpu")
    torch.save({**ck, "state_dict": net.state_dict()}, Path(config.out) / "distilled.pt")
    Path(config.out, "train.json").write_text(json.dumps(report, indent=1), "utf-8")


def ppo_agent(path: str) -> PPOAgent:
    """A greedy PPO agent from a checkpoint (picklable spec for duplicate matches)."""
    return PPOAgent.from_checkpoint(path, greedy=True)


def evaluate(config: Config) -> None:
    """The distilled policy against the original policy, and both on the style panel's deals."""
    game = make("ventures")
    path = str(Path(config.out) / "distilled.pt")
    out = Path(config.out) / "evaluate.json"
    results = json.loads(out.read_text("utf-8")) if out.exists() else {}
    if "diverse_sym" not in results:
        r = duplicate_match(
            game, (ppo_agent, {"path": path}), (ppo_agent, {"path": config.blueprint}), 2000, 2027
        )
        results["diverse_sym"] = {"score": r.score, "score_se": r.score_se, "margin": r.margin}
        results["diverse_sym"]["margin_se"] = r.margin_se
        out.write_text(json.dumps(results, indent=1), "utf-8")
    r = results["diverse_sym"]
    print(
        f"vs diverse_sym       {r['score']:.3f} ± {r['score_se']:.3f} "
        f"({r['margin']:+.2f} ± {r['margin_se']:.2f})",
        flush=True,
    )
    base = json.loads(Path("runs/panel_diverse_sym.json").read_text("utf-8"))
    for name in STYLES:
        if name not in results:
            r = duplicate_match(
                game, (ppo_agent, {"path": path}), (make_heuristic, {"short_name": name}), 500, 61
            )
            results[name] = {"score": r.score, "score_se": r.score_se, "margin": r.margin}
            out.write_text(json.dumps(results, indent=1), "utf-8")
        r, b = results[name], base[name]
        print(
            f"{name:<20} distilled {r['score']:.3f} ± {r['score_se']:.3f} ({r['margin']:+.1f})   "
            f"diverse_sym {b['score']:.3f} ({b['margin']:+.1f})",
            flush=True,
        )


if __name__ == "__main__":
    {"collect": collect, "train": train, "evaluate": evaluate}[sys.argv[1]](Config())
