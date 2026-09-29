# imperfect-info-rl

Self-play reinforcement-learning agents for perfect and imperfect information games — from card games
to poker.

The repo hangs off **one abstraction**: a game-agnostic `Game`/`State` interface (modelled on
OpenSpiel's). Every agent is written against that interface and contains zero game-specific code, so
each new game is "implement the engine, reuse the agents". See [`plan.md`](plan.md) for the roadmap
and [`writeup.md`](writeup.md) for findings.

## Headline result

On **Ventures** (an original ~150-move, high-variance, imperfect-information card game), the full agent
ladder was pitted against a strong hand-tuned heuristic. Each method wins or fails for its own reason.
Two *learned* agents beat the heuristic: **net-guided PIMC** (learn the search priors, keep an honest
leaf value) and **PPO self-play** — a pure reactive policy, no search — which also beats plain PIMC
search 88% of the time. Elo anchored to random = 1000:

| Agent | Elo | vs heuristic |
|------:|:---:|:---:|
| **Net-guided PIMC** | **~1855** | **74% — beats it** |
| **PPO self-play** (no search) | **~1800** | **67% — beats it** |
| PIMC (search) | ~1700 | +10 |
| Heuristic | 1676 | — |
| REINFORCE | ~1420 | ~0% |
| PPO v1 | ~1340 | 0–5% |
| AlphaZero | ~1140 | 0% |
| Random | 1000 | — |

The why — in one line each: AlphaZero's learned value is poisoned by the deal's variance; CFR is
horizon-intractable here; learning the search *priors* on top of an honest leaf value beats the
heuristic; and model-free self-play does too once it gets card-relative features, potential-shaped
rewards, a privileged critic and enough games (v1 PPO's plateau was an engineering artifact). Full
analysis in [`writeup.md`](writeup.md).

## Quickstart

The project uses [`uv`](https://docs.astral.sh/uv/) and a **src layout** — the `iirl` package is
importable only once installed, so the install step is not optional.

```bash
uv sync                 # create the venv, install iirl (editable) + deps (numpy, torch)
uv run pytest           # conformance suite + unit tests (51)
uv run python -m experiments.ventures_tournament    # random < heuristic < PIMC
uv run python -m experiments.train_ventures_guided  # net-guided PIMC — beats the heuristic
uv run python -m experiments.train_ventures_ppo     # PPO self-play — beats the heuristic (~65 min)
uv run python -m experiments.play_ventures          # play against the trained agents in your browser
```

Lint / format (also enforced in CI):

```bash
uv run ruff check . && uv run ruff format --check .
```

## Agents (all game-agnostic, in `src/iirl/agents/`)

- `random_agent` — uniform baseline.
- `mcts` — **PIMC**: determinized UCT for imperfect information (via `Game.resample_from_infostate`).
- `heuristic` — a wager-discipline + deck-clock Ventures baseline (the one game-specific agent).
- `reinforce` — linear self-play policy gradient (numpy; exact, finite-difference-tested gradient).
- `ppo` — MLP actor-critic self-play (PyTorch): shaped rewards + GAE, privileged critic, parallel rollouts.
- `alphazero` — policy/value net + determinized PUCT; also **net-guided PIMC** (`leaf_value="heuristic"`).
- `cfr` — tabular Counterfactual Regret Minimization (+ exact exploitability in `eval/`).

## Layout

```
src/iirl/
├── core/       # game.py (the Game/State interface), types.py, registry.py, runner.py
├── agents/     # random, mcts (PIMC), heuristic, reinforce, ppo, alphazero, cfr
├── nn/         # policy/value network + evaluator (PyTorch)
├── selfplay/   # AlphaZero-style and PPO self-play training loops
├── eval/       # tournament, Elo, anchored Elo, exact exploitability / best response
├── config/     # dataclass configs (no argparse)
├── viz/        # replay recorder for the cross-game "Tells" viewer
└── games/      # from-scratch engines (ventures/)
experiments/    # runnable entry points (tournaments + training curves)
tests/          # conformance suite + per-component tests
```

## License

MIT. Games are implemented from public rules as data structures — no copyrighted art, board images, or
rulebook text. **Ventures** is an original game built on a classic mechanical skeleton, crediting the
mechanical inspiration in its module header as an independent implementation.
