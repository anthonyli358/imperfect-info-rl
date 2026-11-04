# imperfect-info-rl

Self-play agents for a ladder of perfect and imperfect information games, from card games to
poker, and what each method can and cannot do on each kind of game.

The repo hangs off **one abstraction**: a game-agnostic `Game`/`State` interface (modelled on
OpenSpiel's). Every agent is written against that interface and contains zero game-specific code, so
each new game is "implement the engine, reuse the agents". See [`writeup.md`](writeup.md) for the
findings (its "Closing summary" first), [`plan.md`](plan.md) for the roadmap, and
[`docs/adding-a-game.md`](docs/adding-a-game.md) for how to add a game.

## Results

Every game has a stated best agent, its strength with error bars, and the reason it wins. Scores
are over duplicate deals (the same cards dealt both ways round) unless noted; Elo is anchored on
each game's heuristic.

| Game | Kind | Best agent | Strength | Why it wins |
|---|---|---|---|---|
| **Ventures** | Hidden hands, heavy luck | Nash-seeking PPO policy (`diverse_sym`) + opponent-robust rollouts from 6 cards left | 2070 ± 13 Elo (heuristic 1676, random 1000); 0.524 ± 0.006 vs the policy alone; no exploiter beats it (0.493) | A learned value is swamped by luck, so search on one fails; a reactive policy plus late, honest rollouts does not |
| **Camel Stocks** | Hidden hands, trading | `camel_nash` + the same rollouts from 12 cards left | 0.620 ± 0.007 vs the policy alone; 0.978 vs the heuristic | The Ventures recipe transfers unchanged |
| **Tessera** | Perfect information apart from the bag | Policy + value net + PUCT (AlphaZero loop, 48 iterations) | 2057 ± 21 Elo at 400 simulations (heuristic 1500) | Positions predict outcomes, so a value net trained on self-play supports search |
| **Spendthrift** (2 and 3 players) | Perfect information with chance; solved exactly | PPO priors + PUCT on a self-play value net (2p); PPO (3p) | Gives up 0.140 ± 0.014 a game against perfect play (PPO alone 0.33); 3p: 0.094 a game of tie-robust mistakes | Exact solutions score every move; values from self-play outcomes, never PPO's critic |
| **Undertow** | Two asymmetric sides, hidden identities | PIMC search (8 worlds × 40 simulations) | Beats every learned network 0.77–0.85 per seating | Sampled worlds plus lookahead; no network (MLP, transformer, pointer, imitation) caught it |
| **Heads-up limit Hold'em** | ~10^14 information sets | Deep CFR, 200 iterations | Level with an equity heuristic (+0.01 ± 0.04 bb a hand); a local best response wins 0.30 ± 0.06 | The only method here that runs at this size |
| Mini-Ventures | Ventures shrunk until solvable | Full-game CFR+ | Exact exploitability 0.002 points a game (ReBeL reaches it; PPO 0.07) | Exact methods win on small games |
| Liar's Dice (1 die each) | OpenSpiel's engine | CFR+ | Exploitability 0.00013 (ReBeL 0.047, PPO 0.31–0.52) | Same |
| Kuhn, Leduc poker | OpenSpiel's engines | CFR+ | 0.00009 / 0.00023 (Leduc: ReBeL 0.07, Deep CFR 0.135, NFSP 0.84, PPO 1.3) | Same |

**In one paragraph.** Which method wins depends on whether a position's value can be learned and
on the size of the game. Small games are solved exactly by CFR+; large poker needs Deep CFR. Where
positions predict outcomes (Tessera, Spendthrift), the AlphaGo recipe works: a value net trained on
self-play outcomes plus PUCT. Where luck and hidden cards swamp the value (Ventures, Camel Stocks,
Undertow), search on a learned value fails, and what works is a strong reactive policy (PPO
self-play with good features, shaping and a privileged critic) plus search over sampled worlds
where an audit shows the policy is improvable. The full lessons are in the writeup's "Closing
summary".

## The games

All are original implementations behind the same interface; the published games that inspired the
mechanics are credited in each module's docstring, and no names, art or rulebook text are reused.

- **Ventures** (`games/ventures/`). Two players lay ascending single-suit routes from a hidden hand,
  each route costing 20 to open and multiplied by backing cards; the shared deck is the clock.
  About 150 moves a game, heavy luck. The encoding spells out card relations (playable? how far
  above the route?), which was the largest early gain.
- **Camel Stocks** (`games/camel_stocks/`). A five-card market: take one good, take all the camels,
  or swap several; sell sets for tokens that shrink as a good sells out. Hidden hands and deck.
- **Tessera** (`games/tessera/`). Draft all tiles of one colour from shared displays, stage them on
  pattern rows, move completed rows onto a fixed wall for adjacency points; overflow is penalised.
  Perfect information apart from the bag.
- **Spendthrift** (`games/spendthrift/`). Ascending auctions for point lots from one fixed purse,
  with a random end and a penalty for the poorest player: a study of tempo. Two or three players,
  small enough to solve exactly (`eval/perfect_info.py`), so every learner is scored against
  perfect play.
- **Undertow** (`games/undertow/`). Smugglers against a harbour watch, nine pieces a side whose
  identities are hidden from the opponent, on a 14-region harbour. A piece entering an occupied
  region fights with its strength plus a played card. The Smugglers win by landing their contraband
  in port, the Watch by stopping them. Features: 18 piece tokens × 25 plus 101 global.
- **Heads-up limit Hold'em** (`games/holdem/`). Competition rules (blinds 1/2, bets 2/2/4/4, four
  bets a round), our own engine and 7-card evaluator, because OpenSpiel's Windows build lacks
  `universal_poker`.
- **Mini-Ventures**, and OpenSpiel's **Kuhn**, **Leduc** and **Liar's Dice** (through
  `games/spiel.py`): ground truth for calibrating methods and exploiters.

## Quickstart

The project uses [`uv`](https://docs.astral.sh/uv/) and a **src layout**: the `iirl` package is
importable only once installed, so the install step is not optional.

```bash
uv sync --extra openspiel                       # venv, iirl (editable), numpy, torch, OpenSpiel
uv run --extra openspiel pytest                 # conformance suite on every game + unit tests (190)
uv run --extra openspiel ruff check . && uv run --extra openspiel ruff format --check .
uv run --extra openspiel python -m experiments.ventures_tournament   # random < heuristic < PIMC
```

Always pass `--extra openspiel` once it is installed: a plain `uv run` re-syncs the environment and
removes OpenSpiel.

## Retraining the headline agents

Trained checkpoints live in `runs/`, which is git-ignored, so a fresh clone has none: the play page
(`experiments/play.py`) and the head-to-head numbers need the agents retrained first. Every
command is resumable (stop it and rerun the same command) and keeps to 4 worker processes. Times
are on a 12-core CPU without a GPU.

| Agent | Commands, in order | Produces |
|---|---|---|
| Ventures | `train_ventures_ppo` (~65 min) → `train_ventures_nash` → `train_ventures_league` → `train_ventures_continue diverse_sym` (each several hours; the league step also trains exploiters) | `runs/diverse_sym.pt`; the rollout agent is `rollout:runs/diverse_sym.pt` (no extra training) |
| Camel Stocks | `train_camel_stocks` (PPO, then Nash PPO; several hours) | `runs/camel_nash.pt`; rollouts as above |
| Tessera | `train_tessera` (PPO, then two Nash phases) → `alphago_tessera exit iterations=48` (the AlphaZero loop; the longest run) | `runs/tessera_nash2.pt`, `runs/tessera_ag/exit_{prior,value}_48.pt` |
| Spendthrift | `spendthrift solve` (minutes) → `spendthrift ppo game=spendthrift iterations=200` → `spendthrift value` | the exact solution, PPO priors and the value net under `runs/` |
| Hold'em | `holdem train checkpoints=10,25,50,100,200` (~40 min per 100 iterations) | `runs/holdem/deep_cfr_200.pt` |
| Undertow | `undertow ppo iterations=300` (several hours; PIMC needs no training) | `runs/undertow/ppo.pt` |
| Small games | `spiel_games crosscheck`, `mini_equilibrium` (minutes) | CFR+ policies and exploitability |

Each is run as `uv run --extra openspiel python -m experiments.<command>`; each script's docstring
lists its stages and options, and `experiments/elo_ladder.py` re-rates any of them.

## Agents (all game-agnostic, in `src/iirl/agents/` and `src/iirl/selfplay/`)

- `random_agent`, `heuristic`: baselines and Elo anchors (one heuristic per game).
- `mcts`: **PIMC**, determinized UCT for imperfect information (via `resample_from_infostate`).
- `rollout`: opponent-robust late-game rollouts on top of a policy.
- `ppo` (`selfplay/ppo.py`): actor-critic self-play with potential shaping, a privileged critic,
  snapshot leagues and a Nash-seeking magnet; MLP or entity-transformer networks (`nn/entity.py`).
- `alphazero`: policy/value PUCT, the AlphaZero loop, net-guided PIMC, and max-n search for three
  players (`multiplayer.py`).
- `cfr`: tabular and vectorised CFR+, depth-limited MCCFR, and Deep CFR (`deep_cfr.py`).
- ReBeL, NFSP and expert iteration (`selfplay/`), PSRO (`eval/psro.py`), exploiters (`experiments/`).
- `eval/`: duplicate matches, anchored Elo, exact best response and exploitability, local best
  response, exact solving of perfect-information games.

## Layout

```
src/iirl/
├── core/       # game.py (the Game/State interface), types.py, registry.py, runner.py
├── games/      # ventures, camel_stocks, tessera, spendthrift, undertow, holdem, mini_ventures,
│               # and spiel.py (OpenSpiel games behind the same interface)
├── agents/     # random, heuristic, mcts (PIMC), rollout, ppo, alphazero, cfr
├── nn/         # MLP policy/value nets and the entity transformer
├── selfplay/   # PPO, AlphaZero, expert iteration, ReBeL, NFSP loops
├── eval/       # duplicate matches, Elo, exploitability, best response, PSRO, perfect-info solver
├── config/     # dataclass configs (no argparse)
└── viz/        # replay recorder
experiments/    # one runnable script per experiment; run commands in each docstring
tests/          # conformance suite (every registered game) + per-component tests
```

## License

MIT. Games are implemented as data structures: no copyrighted art, board images or rulebook text.
Each game is an original design on a classic mechanical skeleton, with the mechanical inspiration
credited in its module docstring as an independent implementation.
