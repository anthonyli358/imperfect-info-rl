# imperfect-info-rl

> Self-play reinforcement-learning agents for imperfect-information games — from Lost Cities to poker.

*(Repo: `imperfect-info-rl` — searchable and self-describing. Import package: `iirl` — short to type (`from iirl.core.game import Game`). GitHub topics carry the rest of discovery: `reinforcement-learning`, `self-play`, `imperfect-information`, `mcts`, `cfr`, `poker`, `game-ai`. `tells` works as a short README tagline if you want the flavour on top.)*

*Note on scope:* Azul is perfect-information — it's the deliberate "consolidate the self-play/AlphaZero spine on easy mode" rung. Everything else on the ladder is imperfect-information, which is what the name refers to.

---

## Architecture & principles

The whole repo hangs off **one abstraction: a game-agnostic `Game`/`State` interface**, modelled on OpenSpiel's (not Gym's), because these are multi-agent, turn-taking games with chance nodes and information sets — none of which Gym expresses natively. Every agent is written against this interface and contains **zero game-specific code**. Get this right and each new game is "implement the engine, reuse the agents"; get it wrong and every rung fights the framework.

**Core interface (the contract every game implements):**

- `current_player(state)` — including a sentinel for chance nodes
- `legal_actions(state)` — state-dependent; masking lives here, not in agents
- `apply_action(state, action)` — pure transition; cheap `clone`
- `is_chance_node(state)` / `chance_outcomes(state)` — deck draws, dice, bag refills as **first-class** stochastic nodes
- `is_terminal(state)` / `returns(state)` — two-player zero-sum now, `num_players`/co-op left open for Hanabi later
- `information_state_tensor(state, player)` — the per-player encoding CFR and IS-MCTS need
- `observation_tensor(state, player)` — for perfect-info / AlphaZero use

**Design commitments (these determine the repo):**

- **Two wings.** `iirl/games/` = from-scratch engines + agents (the part that's *mine*). `openspiel_experiments/` = framework-driven poker/Hanabi work. OpenSpiel is an **optional extra** (`pip install imperfect-info-rl[openspiel]`), never a hard top-level dep — cloning for the hand-built work stays lightweight.
- **Algorithms parameterised over `Game`.** MCTS / IS-MCTS / AlphaZero / CFR family / PPO / NFSP all live in `iirl/agents/` and take any `Game`.
- **Minimal spine first, harden after Jaipur.** Don't over-fit the interface before two real imperfect-info games have stress-tested it. There's a dedicated refactor beat at the end of Stage 2.
- **Games are data, not assets.** Cards/tiles as plain identifiers and data structures. No card art, board images, or rulebook text in the repo. Descriptive names, own permissive license (MIT/Apache), README note crediting each designer/publisher as an independent implementation. Model checkpoints are my artifacts and fine to publish.
- **Config over CLI.** Module-level config dataclasses/constants, no argparse boilerplate; each entry point has a minimal `if __name__ == "__main__"` runnable block.
- **Evaluation is a first-class module, not an afterthought.** Exploitability where computable; Elo/tournament with honest sample sizes and variance handling (the poker tax). Wire OpenSpiel agents into the same tournament runner for cross-wing benchmarking.
- **Clean module separation + Google-style docstrings** throughout.

**Target layout:**

```
imperfect-info-rl/            # repo (searchable name)
├── pyproject.toml            # name = "imperfect-info-rl"; core deps; [openspiel] extra
├── README.md · LICENSE · plan.md
├── src/
│   └── iirl/                 # import package (src layout — installed, not root-importable)
│       ├── core/             # game.py (interface), types.py, registry.py
│       ├── games/            # lost_cities/ jaipur/ azul/ liars_dice/
│       ├── agents/           # random, mcts/, alphazero/, cfr/, ppo/, nfsp/
│       ├── nn/               # policy/value nets, encoder base
│       ├── selfplay/         # self-play loop, replay buffer
│       ├── eval/             # tournament, elo, best-response/exploitability
│       └── config/           # config dataclasses
├── openspiel_experiments/    # kuhn/ leduc/ limit_holdem/ nolimit/ hanabi/
├── experiments/ · tests/ · notebooks/
```

Absolute imports everywhere (`from iirl.core.game import Game`), identical from tests, the OpenSpiel wing, and notebooks alike. Src layout means the package is importable only once installed — so tests run against the installed package, catching packaging bugs (missing `__init__`, an uncollected module, a broken extra) that a root-level package would silently mask.

**Deliberately excluded (intentional scoping):**

- **Battle Line** — protected. It's the game I want to keep playing; its deck-probability layer is exactly the un-see-able checklist that building an agent would spoil. Left off the ladder on purpose.
- **Viticulture / Dune: Imperium** — engine-cost sinks (bespoke card-effect logic dwarfs the RL). Someday-maybe, not a rung.
- **3+ player variants** — non-stationary multiplayer breaks the clean two-player zero-sum equilibrium/self-play story. All rungs are two-player.

---

## Stage 0 — Foundation & shared spine *(minimal, on purpose)*

Goal: the game-agnostic interface, baselines, and an evaluation skeleton — nothing more. Resist polishing the interface before real games exist.

- [x] Repo scaffold: `pyproject.toml` (src layout, `packages = ["src/iirl"]`), license, README (uv quickstart); GitHub topics listed as `keywords` in `pyproject.toml` — still to set on the GitHub repo itself
- [x] Editable install via **uv** (`uv sync`) — the src-layout install-first step, documented as the quickstart in the README (replaces the `pip install -e .` line)
- [x] `core/types.py` — `Player`, `Action`, `PlayerId` (chance/terminal sentinels), `ChanceOutcome`, `GameType` (`Information`/`Utility`/`num_players` + tensor-support flags)
- [x] `core/game.py` — the `Game`/`State` interface (functional/immutable; optional encoders gated by `GameType`)
- [x] `core/registry.py` — game registry / factory
- [x] Generic episode runner — `core/runner.py` (`run_episode`; drives any agents, resolves chance nodes from a seeded generator)
- [x] `RandomAgent` (`agents/`) + generic **conformance suite** (`tests/conformance.py`: legal-action sanity, immutability of `apply_action`, chance outcomes sum to 1, terminal/returns + zero-sum consistency, termination)
- [x] `eval/` skeleton — `tournament.py` (seat-balanced `play_match` with mean+stderr, Elo, round-robin; variance note in docstring)
- [x] `config/` pattern — dataclass configs (`RunnerConfig`/`TournamentConfig`), minimal runnable block in `experiments/random_vs_random.py`
- [x] Tooling — pytest, ruff (lint + format + Google-docstring `D` rules), CI (`.github/workflows/ci.yml`, uv, Python 3.11/3.12)

**Done when:** two `RandomAgent`s play any registered game to completion through the runner, and the conformance suite is green. ✅ *(15 tests green; `uv run python -m experiments.random_vs_random` prints standings for the `toy_showdown` fixture. Real games arrive in Stage 1.)*

---

## Stage 1 — Ventures *(first from-scratch; imperfect-info; PIMC / determinized MCTS)*

> **IP note:** shipped as **Ventures** — an original game on the same *mechanical* skeleton, not a
> Lost Cities implementation (name/theme/assets are all original; designer credited as inspiration
> only). Package `iirl/games/ventures/`, registry key `ventures`.

Goal: first full game. Deck-weighted hidden info + the −20 delayed-reward structure. Cut teeth on determinization / IS-MCTS.

- [x] Engine: state, legal actions (play/discard → draw), route columns, backing (wager) multipliers, −20 start-cost scoring, +20 route bonus (`games/ventures/`)
- [x] Chance nodes for deal + deck draw; info state (opponent hand + deck hidden, routes + discards public)
- [x] Cards as data (`cards.py`); IP/credit note in the module header
- [x] Engine unit tests: ascending + backing-before-number legality, scoring (penalty/multiplier/bonus), terminal on deck exhaustion, **card-conservation invariant**, random rollout to completion; auto-covered by the shared conformance suite
- [x] `information_state_tensor` + `information_state_string` encoders for Ventures
- [x] PIMC / determinized-MCTS agent in `agents/mcts/` (generic `uct.py` + `pimc.py`), via new game-agnostic `Game.resample_from_infostate` hook; truncated rollouts use a deck-clock-aware `heuristic_value` leaf eval
- [x] Heuristic baseline (`agents/heuristic/`): wager discipline + the deck as a shared clock (don't open a route you can't finish before the deck empties)
- [x] Eval: PIMC vs random vs heuristic (`experiments/ventures_tournament.py`) — ordering confirmed: **PIMC > heuristic > random** (PIMC beats random +90.4 [30-0], heuristic +10.0 ± 5.5 [21-19])
- [x] **Bonus — cross-game replay viewer** (`viz/`): `Game.to_display`/`display_meta` hooks + `iirl.viz.replay` recorder + an HTML viewer ("Tells") with a consistent shell and per-game board, for watching agents play any registered game
- [x] **Bonus — first learnable agent + Elo training** (pulled forward from Stage 6): `agents/reinforce/` — a game-agnostic linear softmax policy trained by self-play REINFORCE (numpy; exact policy gradient, finite-difference tested), new `Game.num_distinct_actions` hook + `eval.anchored_elo`, `experiments/train_ventures.py`. Elo climbs **931 → ~1420** vs fixed anchors (random 1000, heuristic 1676): decisively beats random, plateaus below the heuristic (as a linear policy should). Curve chart published as an artifact
- [x] **Bonus — AlphaZero-style self-play** (pulled forward from Stage 3, adapted to imperfect info): PyTorch added; `nn/policy_value.py` (policy+value net), `agents/alphazero/` (determinized **net-guided PUCT** with **batched leaf inference + virtual loss**), `selfplay/alphazero.py` (self-play → replay buffer → train → Elo), `experiments/train_ventures_az.py`. **Honest negative result:** on Ventures the learned value is swamped by the deal's variance, so net-guided search plateaus at the untrained-search level (~1140 Elo, never beats the heuristic) — *worse* than rollout-based PIMC. This empirically re-derives the field's paradigm split (AZ → low-variance perfect-info; CFR → imperfect-info) — the plan's own Stage-6 headline experiment.
- [x] **Bonus — CFR + exploitability** (pulled forward from Stage 4, the *right* imperfect-info tool): game-agnostic tabular CFR (`agents/cfr/`, regret matching + average strategy) and an exact per-infoset best-response / **exploitability** module (`eval/exploitability.py`). Validated on `toy_showdown`: exploitability → **0.0015**, and the average strategy reproduces the **exact Kuhn equilibrium** (low-card bluff 0.22, King opener 0.67 = 3×, bluff-catching with the middle card). *(Tabular CFR fits Kuhn/Leduc scale; full Ventures needs Deep CFR — Stage 5.)*
- [x] **Bonus — PPO self-play** (pulled forward from Stage 6; the model-free route that fits a long-horizon high-variance game): `agents/ppo/` + `selfplay/ppo.py` (MLP policy/value, clipped surrogate, value baseline, masked policy). Clean self-training climb on Ventures: **Elo 1068 → ~1340 (peak 1431), crushing random 0.68 → 1.00** — the best-behaved learned agent, just above linear REINFORCE. Like all *reactive* learners it plateaus below the deck-clock heuristic (wins 0–5% head-to-head); only **search** (PIMC) beats the heuristic.

- [x] **Bonus — net-guided PIMC** (the synthesis; `agents/alphazero` search with `leaf_value="heuristic"`): keep PIMC's honest deck-clock leaf value (the part that beats the heuristic) and use the network *only* for PUCT **priors**, trained on the search's own visit distributions (AlphaZero's policy-improvement idea with rollout-quality evaluation). `experiments/train_ventures_guided.py`. Learned priors lifted it from **25% → ~47% vs the heuristic** at a small eval budget; the trained agent at 256 sims **decisively beats the heuristic: 29-1-10 (74% score), Elo ~1855** — the strongest agent built, above pure PIMC and the heuristic. This is the answer to "beat the heuristic with learning": learn the *priors*, keep an honest leaf value.

**Method scoreboard on Ventures** (Elo, random = 1000): **net-guided PIMC ~1855 (beats heuristic 74%) > PIMC ~1700 (beats heuristic +10) > heuristic 1676 > PPO ~1340/peak 1431 ≈ REINFORCE 1420 > AlphaZero 1140 > random 1000**. The lesson, re-derived empirically: on a long-horizon, high-variance, imperfect-info game, **explicit search wins**; reactive learned policies (PPO/REINFORCE) beat random decisively but plateau below strong lookahead play; AlphaZero's *learned value* is poisoned by variance; CFR is horizon-intractable — but **learning the search *priors* while keeping an honest leaf value** (net-guided PIMC) beats everything. Each method wins or fails for a *different*, characteristic reason — the ladder's whole point.

**Done when:** PIMC beats the heuristic convincingly and the engine passes the conformance suite. ✅ *(29 tests green; strength ordering confirmed; the PIMC/UCT agent is game-agnostic and must port unchanged to Jaipur in Stage 2.)*

---

## Stage 2 — Jaipur *(second imperfect-info; then harden the spine)*

Goal: a second game to reveal which abstractions are real vs incidental — then refactor. The IS-MCTS agent should port **unchanged**; if it doesn't, the interface is wrong.

- [ ] Engine: goods + camels, take-single / take-multiple / sell actions, token stacks + depletion, bonus tokens
- [ ] Chance draws; information state (opponent hand hidden, market public)
- [ ] Engine unit tests
- [ ] Jaipur `information_state_tensor` encoder
- [ ] Run the **existing** IS-MCTS agent with no game-specific changes (interface validation)
- [ ] Heuristic baseline (hoard-vs-sell tempo)
- [ ] Eval vs random / heuristic

**Refactor block — harden the game-agnostic core:**

- [ ] Extract a stable, documented interface contract (both games conform)
- [ ] Confirm `agents/` contain zero game-specific branches
- [ ] Base `Encoder` abstraction; per-game encoders subclass it
- [ ] Registry + config cleanup; write the "how to add a game" doc

**Done when:** adding a third game would touch only `games/` and an encoder — nothing in `agents/`.

---

## Stage 3 — Azul *(perfect-info; AlphaZero-style self-play)*

Goal: consolidate the neural self-play infra on the *easiest* substrate — perfect info means straight MCTS + policy/value net, no information sets. Also the safe-to-build game (spatial fun, no portable checklist).

- [ ] Engine: factories + centre, colour drafting, pattern rows, wall placement, floor line, adjacency scoring, bag/lid chance
- [ ] Engine unit tests (scoring geometry especially)
- [ ] Board `observation_tensor` + action encoding with legal-action masking
- [ ] `nn/` policy+value network
- [ ] `selfplay/` loop + replay buffer
- [ ] AlphaZero-style training (MCTS-visit policy targets, game-outcome value targets)
- [ ] Checkpointing + seed reproducibility
- [ ] Eval: net-guided MCTS vs raw MCTS vs random; Elo-vs-iteration curve

**Done when:** the trained agent clearly beats raw MCTS, and the self-play/NN/checkpoint infra is reusable.

---

## Stage 4 — Liar's Dice *(first CFR, on my own engine; bridge to equilibrium)*

Goal: implement CFR from scratch on a small game I built — the paradigm shift *before* poker, on something I can eyeball. Watch it converge to mixed (bluffing) strategies. Validate against OpenSpiel.

- [ ] Engine: dice, bidding, challenge/dudo, two-player variant; chance nodes for rolls; information states
- [ ] Engine unit tests
- [ ] Tabular CFR (regret matching) + average-strategy accumulation in `agents/cfr/`
- [ ] Best-response / exploitability computation for this game
- [ ] CFR+ and MCCFR variants
- [ ] Cross-validate equilibrium values vs OpenSpiel's `liars_dice` *(needs `[openspiel]`)* — checks both my engine and my CFR
- [ ] Eval: CFR vs heuristic; inspect learned bluff/call frequencies for sanity

**Done when:** exploitability drives toward zero and my values match OpenSpiel's reference.

---

## Stage 5 — Poker via OpenSpiel *(Kuhn → Leduc → limit Hold'em)*

Goal: adopt the framework where it earns its place — CFR at scale, with verifiable results. Kuhn to learn the tooling, limit Hold'em as the substantive, checkable finale. No abstraction rabbit hole.

- [ ] Add `[openspiel]` optional dependency + build/install docs
- [ ] **Kuhn poker** — tabular CFR to the known equilibrium; compute exploitability (framework hello-world)
- [ ] **Leduc poker** — scale check; MCCFR
- [ ] **Deep CFR** (neural) on Leduc / limit
- [ ] **Heads-up limit Hold'em** — MCCFR/Deep CFR agent; sanity-check against known solved results (Cepheus as reference point)
- [ ] Evaluation discipline: exploitability + large-sample head-to-head; variance-reduction (AIVAT-style) note/impl
- [ ] Wire OpenSpiel agents into `iirl/eval/` tournament runner (cross-wing benchmarking)

**Done when:** a limit Hold'em agent with low measured exploitability, evaluated honestly, runnable through my own eval harness.

---

## Stage 6 — Stretch / research edge *(optional, pick what stays fun)*

- [ ] **PPO** in `agents/ppo/` — foundational algorithm to own (also the RLHF/post-training bridge to day-job + SLM work)
- [ ] **Headline experiment:** PPO self-play vs CFR on Liar's Dice — measure the **exploitability gap**. Make "why the field uses regret minimisation, not vanilla policy gradients" something I've *watched happen*.
- [ ] **NFSP** — the bridge that targets equilibrium with RL-style best-response; compare against CFR and PPO
- [ ] **No-limit Hold'em (scoped)** — blueprint agent over a small fixed action abstraction, explicitly "functional, not frontier"; explore action/card abstraction + the translation problem. *Not* aimed at strength.
- [ ] **Hanabi (OpenSpiel)** — cooperative, imperfect-info; self-play coordination, reaching toward the open zero-shot / cross-play problem. The highest-ceiling rung the framework makes cheap.
- [ ] Short writeup / notebook per experiment

---

## Cross-cutting definition of done

- [ ] Google-style docstrings; clean module separation; no argparse (config dataclasses + minimal runnable blocks)
- [ ] Engine unit tests + shared conformance suite green for every game
- [ ] Per-game README section: rules summary, encoding, agent, results
- [ ] Reproducible seeds; checkpoints for trained agents
- [ ] No copyrighted assets; designer/publisher credits; permissive license on my code
