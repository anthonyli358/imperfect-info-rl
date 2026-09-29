"""Phase B, next step: dedicated exploiters in the league (PSRO-lite, as in AlphaStar's league).

Each round:

1. Train the main agent with the Nash recipe (magnet KL + snapshot pool), with the heuristic and
   every exploiter found so far as permanent league members (``scripted_opponents``, greedy).
2. Check whether it closed the hole the previous exploiter found (duplicate match vs that
   exploiter, which was trained against the previous main agent).
3. Train a fresh best-response exploiter against it (``experiments/exploit_ventures.py``), report
   its score (the exploitability yardstick), and add it to the league.

At the end, a same-budget **control** continues the plain Nash recipe (heuristic only) from the
same start and is exploited the same way, so any gain can be credited to the exploiters rather than
to extra training.

**Resumable.** Every step leaves a file behind and is skipped when its file exists: a finished
training phase ``runs/<tag>.pt``, the hole check (``runs/league_state.json``), an exploiter
``runs/exploit_<tag>.json`` (which also marks its whole round done). A training phase also saves
``runs/<tag>_partial.pt`` at every eval, and a rerun continues from it for the remaining
iterations. The optimizer, snapshot pool and magnet restart at that point, a small perturbation.
Stop at any time and rerun the same command.

**Elo log.** Every in-training eval (Elo anchored on the heuristic at 1676, 400 games, about ±35
Elo noise) is appended to ``runs/league_elo.json``, per phase.

Run with::

    uv run python -m experiments.train_ventures_league
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import iirl.games  # noqa: F401  (registers "ventures")
from experiments import exploit_ventures
from experiments import train_ventures_nash as nash
from iirl.agents.heuristic import VenturesHeuristic
from iirl.agents.ppo import PPOAgent
from iirl.core.registry import make
from iirl.eval.duplicate import duplicate_match
from iirl.selfplay import PPOConfig, train_ppo


@dataclass(frozen=True)
class Config:
    """Experiment settings.

    Attributes:
        start: The main agent to start from.
        exploiters: Exploiters already found against ``start`` (checkpoint paths; their
            ``exploit_<name>.json`` says greedy or sampled).
        rounds: League rounds (train, then exploit).
        iterations: PPO iterations per round.
        scripted_frac: Share of the main agent's games against the fixed league (heuristic and
            exploiters, picked uniformly).
        control: Also run the same-budget plain-Nash control at the end.
        check_deals: Deals in the "did it close the old hole" match.
    """

    start: str = "runs/nash_long.pt"
    exploiters: tuple[str, ...] = ("runs/exploit_nash_long.pt",)
    rounds: int = 2
    iterations: int = 200
    scripted_frac: float = 0.3
    control: bool = True
    check_deals: int = 1000
    ppo: PPOConfig = field(default_factory=lambda: nash.variant("nash", nash.Config().ppo))


def _greedy(exploiter: str) -> bool:
    """Whether an exploiter plays greedily (its screening chose so), from its JSON record."""
    record = Path(exploiter).with_suffix(".json")
    if not record.exists():
        return True
    return bool(json.loads(record.read_text("utf-8"))["exploiter"]["greedy"])


ELO_LOG = Path("runs/league_elo.json")
STATE = Path("runs/league_state.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def _dump(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), "utf-8")
    tmp.replace(path)  # atomic, so a kill mid-write cannot corrupt the log


def _train(start: str, ppo: PPOConfig, tag: str, save_every: int = 0) -> str:
    """Train from ``start`` with ``ppo``; save as ``runs/<tag>.pt`` and return the path.

    Skipped if ``runs/<tag>.pt`` exists; continued from ``runs/<tag>_partial.pt`` if that does.
    With ``save_every``, also keeps ``runs/<tag>_<iteration>.pt`` every that many iterations.
    """
    path, partial = Path(f"runs/{tag}.pt"), Path(f"runs/{tag}_partial.pt")
    if path.exists():
        print(f"{tag}: done already ({path})", flush=True)
        return str(path)
    game = make("ventures")
    heuristic = VenturesHeuristic("heuristic")
    done, games_done = 0, 0
    state = _load(STATE)
    if partial.exists() and tag in state.get("partial", {}):
        done, games_done = state["partial"][tag]["iteration"], state["partial"][tag]["games"]
        start = str(partial)
        print(f"{tag}: resuming at iteration {done}", flush=True)
    log = _load(ELO_LOG)
    log[tag] = [e for e in log.get(tag, []) if e["iteration"] <= done]

    def report(p) -> None:  # noqa: ANN001 - PPOEvalPoint
        it, games = done + p.iteration, games_done + p.games
        print(
            f"{it:>4} {games:>6}  elo {p.elo:6.0f}  vs heur {p.win_rate_vs['heuristic']:.2f}"
            f"  margin {p.margin_vs['heuristic']:+6.1f}",
            flush=True,
        )
        if p.iteration == 0 and done:
            return  # the resumed start was already logged
        log[tag].append(
            {
                "iteration": it,
                "games": games,
                "elo": round(p.elo, 1),
                "win_rate": p.win_rate_vs["heuristic"],
                "margin": round(p.margin_vs["heuristic"], 2),
            }
        )
        _dump(ELO_LOG, log)

    def checkpoint(net, p) -> None:  # noqa: ANN001 - PolicyValueNet, PPOEvalPoint
        if p.iteration == 0:
            return
        nash.save(net, partial)
        it = done + p.iteration
        if save_every and it % save_every == 0:
            nash.save(net, Path(f"runs/{tag}_{it}.pt"))
        st = _load(STATE)
        st.setdefault("partial", {})[tag] = {
            "iteration": done + p.iteration,
            "games": games_done + p.games,
        }
        _dump(STATE, st)

    ppo = replace(ppo, anchor_elos={"heuristic": 1676.0}, iterations=ppo.iterations - done)
    init = PPOAgent.from_checkpoint(start).net
    net, _ = train_ppo(game, {"heuristic": heuristic}, ppo, report, checkpoint, init_net=init)
    nash.save(net, path)
    partial.unlink(missing_ok=True)
    return str(path)


def _exploit(target: str, name: str) -> tuple[str, float]:
    """Exploit ``target`` (unless done already); return the exploiter's path and final score."""
    if not Path(f"runs/exploit_{name}.json").exists():
        exploit_ventures.main(exploit_ventures.Config(target=target, name=name))
    record = json.loads(Path(f"runs/exploit_{name}.json").read_text("utf-8"))
    return f"runs/exploit_{name}.pt", float(record["final"]["score"])


def main(config: Config) -> None:
    """Run the league rounds, then the control; print a summary."""
    game = make("ventures")
    base = replace(config.ppo, iterations=config.iterations)
    main_path, exploiters = config.start, list(config.exploiters)
    summary = []
    for r in range(1, config.rounds + 1):
        print(f"\n=== league round {r}: {len(exploiters)} exploiter(s) ===", flush=True)
        done = Path(f"runs/exploit_league_r{r}.json")
        if done.exists():  # the round's last step is done; its checkpoint is not needed again
            main_path = f"runs/league_r{r}.pt"
            check = _load(STATE).get(f"check_r{r}", {}).get("score", float("nan"))
            score = float(json.loads(done.read_text("utf-8"))["final"]["score"])
            exploiters.append(f"runs/exploit_league_r{r}.pt")
            summary.append((f"league_r{r}", check, score))
            print(f"round {r}: done already (exploiter {score:.3f})", flush=True)
            continue
        league = (VenturesHeuristic("heuristic"),) + tuple(
            PPOAgent.from_checkpoint(e, name=f"exploiter{i}", greedy=_greedy(e))
            for i, e in enumerate(exploiters)
        )
        ppo = replace(base, scripted_opponents=league, scripted_frac=config.scripted_frac, seed=r)
        main_path = _train(main_path, ppo, f"league_r{r}")
        old, key = exploiters[-1], f"check_r{r}"
        state = _load(STATE)
        if key not in state:
            check = duplicate_match(
                game,
                (exploit_ventures.ppo_agent, {"path": old, "greedy": _greedy(old)}),
                (exploit_ventures.ppo_agent, {"path": main_path}),
                config.check_deals,
                seed=31,
            )
            state = _load(STATE)
            state[key] = {"score": check.score, "score_se": check.score_se}
            _dump(STATE, state)
        check = state[key]
        print(f"previous exploiter vs league_r{r}: {check['score']:.3f}±{check['score_se']:.3f}")
        exploiter, score = _exploit(main_path, f"league_r{r}")
        exploiters.append(exploiter)
        summary.append((f"league_r{r}", check["score"], score))
    if config.control:
        print("\n=== control: plain Nash recipe, same budget ===", flush=True)
        ppo = replace(base, iterations=config.iterations * config.rounds, seed=100)
        control = _train(config.start, ppo, "league_control")
        _, score = _exploit(control, "league_control")
        summary.append(("league_control", float("nan"), score))
    print("\nagent            old exploiter  new exploiter")
    for name, old_score, new_score in summary:
        print(f"{name:<16} {old_score:>13.3f}  {new_score:>13.3f}")


if __name__ == "__main__":
    main(Config())
