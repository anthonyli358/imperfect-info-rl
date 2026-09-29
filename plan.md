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

## Stage 1b — Ventures: toward an unexploitable agent *(the reference recipe before other games)*

Goal: the strongest Ventures agent we can build **and prove**, as the recipe the later games reuse.
"Unbeatable" here means *unexploitable*: in a two-player zero-sum game a Nash-equilibrium strategy
cannot lose in expectation to anyone. Luck still decides single games, so the headline metric is
**exploitability** (how much a best-response exploiter trained against the frozen agent can win),
alongside Elo. Objective throughout: win, then margin (`outcome_value` — win/loss blended with margin).

**What we learned getting here** (2026-09-25; see `writeup.md`):

- PPO v2 (~1800 Elo, 0.695 vs the heuristic) is the strongest single policy. Self-play PPO does not
  target Nash, though: it can cycle, and the matchups are visibly non-transitive (PIMC ≈ heuristic,
  yet PPO crushes PIMC; bigger search gains vs PPO but not vs the heuristic).
- Determinized search on top of PPO adds ~+30 Elo at 64 sims × 4 worlds, ~+75 at 128 × 8, and
  saturates by 256 × 8. The leaf value is not the limit: heuristic, info-state and privileged
  (both-hands) leaves all give the same gain. The limits are uniform determinization (ignores what
  the opponent's moves reveal) and strategy fusion.
- Distilling that search back into the policy (expert iteration) is fragile: raw visit targets
  collapsed the policy; the Gumbel completed-Q target was stable at 64 × 4 but gave no gain; a
  pooled-visit scaling bug collapsed it at 128 × 8 (fixed: per-world scaling).
- Throughput is the binding constraint for any search-based learner: ~14k searched positions per
  16 min, vs PPO's ~150k games/hour. The Python engine is the bottleneck; the GPU is idle.
- Evaluation is cheap and trustworthy: duplicate deals (`eval/duplicate.py`) halve the error
  bars; a raw-policy check (500 deals, ~12 s) catches a bad update immediately.

**Why not "just AlphaZero":** AlphaZero is the perfect-information recipe. AlphaStar (StarCraft)
was *not* AlphaZero — it used no tree search; it was imitation learning plus **league training**
(a population of main agents and dedicated *exploiters* that hunt their weaknesses), which is what
made it robust. The imperfect-information analogues of AlphaZero are **ReBeL** and **Student of
Games** (search over *public belief states* with CFR at the search step; provably converge to Nash
in two-player zero-sum), and for long-horizon games where search is hard, **DeepNash / R-NaD**
(Stratego: model-free, Nash-seeking regularized self-play, no search at all).

**Plan** (each phase is useful on its own; stop wherever the exploitability curve flattens):

- [x] **A. The yardstick — exploitability.** ✅ `experiments/exploit_ventures.py`. Lesson: with
  PPO's default entropy bonus and step size the warm-started exploiter drifts *below* a strong
  target and falsely reports it unexploitable; `entropy_coef=0`, `lr=3e-5`, critic warm-up and
  checkpoint screening fix it. Always validate a "found nothing" against a known-weak target.
  Heuristic: 0.781 (+56.6). PPO v2: **0.628 (+12.6)**. Still to do: PIMC calibration and a
  fixed evaluation panel (heuristic, PIMC, PPO v2, search agent) so non-transitivity stays visible.
- [x] **B (first result).** ✅ `experiments/train_ventures_nash.py`, 51k games each from PPO v2:
  plain PPO control 0.597 vs v2, exploiter **0.584 (+8.5)**; magnet + league ("nash") 0.590 vs v2,
  exploiter **0.537 (+4.4)**; the two tie head-to-head. Same strength, about half the
  exploitability.
- [x] **B (longer run).** ✅ `nash_long`: 600 more iterations of the same recipe from `nash_nash`
  (153k games). Exploiter **0.522 (+2.5)**, still falling. It is also stronger: 0.532 vs
  `nash_nash`, **0.815 (+52.2) vs the heuristic**, 0.640 vs PPO v2 (1000 duplicate deals each).
  Periodic checkpoints (`nash_long_<iter>.pt`) are rated in `runs/elo_ladder.json`
  (`experiments/elo_ladder.py`): PPO v2 1822, nash 1890, nash_long 1917 → 1946 → 1941.
- [x] **B (exploiter league: negative).** `experiments/train_ventures_league.py` (resumable).
  Two rounds of 200 iterations from `nash_long`, each exploiter added to the league. So far it
  **makes things worse**: the exploiter score rises 0.522 → **0.556** (r1) → **0.573** (r2), and
  the r1 agent did not even close `nash_long`'s own hole (that exploiter scores 0.550 vs r1).
  Elo vs the heuristic still rises (r1 1942, r2 **1962**), so beating conventional play and
  being unexploitable have pulled apart. The same-budget plain-Nash control (`league_control`) still
  ran after: **0.544 (+4.5), 1973 Elo**, better than league r2 on both counts. Dedicated
  exploiters as fixed league members hurt (they overfit to a few best responses); keep the plain
  recipe. Repeat exploiter runs (seed 1) showed the yardstick varies ~0.02 per run, twice its
  standard error: nash_long 0.522/0.545, league_control 0.544/0.538. Exploitability = the max
  over ≥ 2 exploiter runs (a single run is a lower bound). By that, nash_long and
  league_control are equally exploitable (~0.545), and league_control is 30 Elo stronger.
- [x] **Endgame stall fix.** `PPOConfig.gamma` (default 1.0, unchanged behaviour): with 0.999 a
  winning player prefers ending the game sooner over stalling to the turn cap. Use it from the
  next training run (not in the running control, which must match the league rounds).
- [x] **C (first result).** ✅ `agents/belief.py` particle filter (`Agent.reset`/`observe` hooks,
  `Game.repair_world`), `experiments/belief_ventures.py`. Lessons: a plain filter collapses (each
  opponent reveal kills ~80% of particles); rejuvenating with uniform resamples is worse than
  none; *repairing* dead particles works; a sharpened likelihood (temperature 4) fits a greedy
  opponent. Belief worlds share 0.453 of the opponent's hidden hand vs 0.386 uniform. But at 64
  sims × 4 worlds, belief search beats uniform search only 0.512 ± 0.011, and uniform search is
  level with the raw `nash_nash` policy (0.502): the value net (trained on PPO v2) no longer
  helps. Retrained on `nash_long` self-play (`value_nash_long_priv.pt`, Brier 0.138 vs the
  heuristic leaf's 0.166), it changes nothing: uniform search vs `nash_long` 0.502 ± 0.013,
  belief vs uniform 0.500 ± 0.012. At 64 × 4, determinized search no longer improves on the
  Nash-trained policy, with or without the belief. Belief search is shelved; it may matter again
  at much larger search budgets (Phase D) or on games where hands say more.
- [x] **B. Nash-seeking learner** (core done; results above). Turn PPO self-play into a method that converges to equilibrium:
  **MMD / R-NaD-style** regularization toward a slowly updated reference ("magnet") policy — close
  to a drop-in change to `selfplay/ppo.py` — plus **league play** (AlphaStar / PSRO-lite): keep a
  pool of past agents and train exploiters against the current main agent, and mix them into the
  main agent's opponents. Keeps PPO's throughput. Success = exploiter gain falling over time.
  Start from PPO v2 (it already beats every scripted agent, so imitating one would start lower).
  Keep scripted and style-diverse agents (heuristic, PIMC, a tight "few routes" bot, an aggressive
  bot) in the league permanently as fixed opponents, as AlphaStar kept its imitation agents: they
  stop the main agent forgetting how to beat conventional play. Heuristic knowledge otherwise
  enters as potential-based reward shaping (already in place), which cannot change the optimum.
- [ ] **Recipe note for new games** (Jaipur onward): with no RL base yet, write the heuristic,
  behaviour-clone its replays, then fine-tune with PPO — skips the random-init phase.
- [x] **C. Belief-aware test-time search** (built; no gain at 64 × 4, shelved; results above). Replace uniform determinization with **Bayesian
  determinization**: weight each sampled opponent hand by the likelihood of the opponent's observed
  actions under the policy (their discards and plays reveal their hand). Small change to
  `resample_from_infostate` / the search; likely the cheapest large gain. Then Gumbel root search
  (sequential halving) on top of the B policy.
- [ ] **D. Throughput.** Fast engine core (Rust via pyo3, cross-checked against the Python engine),
  batched search over hundreds of games with GPU leaf inference (CUDA torch in the venv), kept
  game-agnostic. Target: 50–100× more searched positions per hour. Prerequisite for E and for
  search-based learning on bigger games.
- [ ] **E. The principled endgame — ReBeL / Student of Games.** Search over public belief states
  (the only hidden information is one 8-card hand, so beliefs are tractable), CFR at each search
  step, value net over belief states, trained by self-play. Only if B + C plateau above zero
  exploitability.

**Next steps (2026-09-29), in order.** Each is useful alone; stop wherever the curve flattens.

1. [x] **Finish `nash_g`.** Ladder 1969/1982/1975/1962 (flat, within noise of league_control's
   1973); exploitability 0.547 (max of 2). The recipe has plateaued at ~1970 Elo / ~0.545. The
   stall persists (1 in 300 self-play games, as before): the leader still rarely draws the last
   card, a position too rare to learn in 400 iterations. Harmless.
2. [ ] **Refresh the stale number.** Rate net-guided PIMC on today's yardstick: duplicate deals
   vs the heuristic and vs the best Nash policy. Settles whether any search-based agent still
   competes.
3. [ ] **Cheap Nash-seeking variants**, each vs a same-budget plain-Nash control, and judged on
   both Elo and exploitability (max of ≥ 2 exploiter runs):
   - **R-NaD magnet reset:** refresh the magnet to the *average* of recent policies rather than
     the current one (closer to the method's convergence guarantee).
   - **Proper PSRO:** keep a population (past mains and exploiters), build its payoff matrix from
     duplicate matches, solve for the Nash *mixture*, and train the main agent against that
     mixture. The fixed-exploiter league failed because every member got equal, fixed weight.
   - **Scale:** a bigger net (1024 × 3) or simply 2× the games.
4. [ ] **Decision point: go deeper on Ventures, or move to Jaipur.**
   - *Move on* if the exploiter edge is at or below ~+2 points/game, or if step 3 gives nothing.
     Jaipur (Stage 2) then tests the recipe and interface on a second game, and brings the planned
     interface hardening (including a generic `player_view` hook, so the play UI works for any
     game).
   - *Go deeper* if the remaining edge looks like something search could close: **D** (Rust
     engine; test search on the Nash policy at 256 × 8 and beyond), then **E** (ReBeL).

**Done when:** a best-response exploiter can no longer beat the agent by a statistically
meaningful margin, and the agent beats every panel opponent — with the recipe documented as the
template for Jaipur and the later rungs.

---

## Stage 2 — Camel Stocks *(second imperfect-info; then harden the spine)*

> **IP note:** shipped as **Camel Stocks**, an original game on the mechanical skeleton of Jaipur
> (Sébastien Pauchon), which is credited as inspiration only. Own name, theme and goods names, no
> assets. Package `iirl/games/camel_stocks/`, registry key `camel_stocks`. The rules data was
> checked against the original rulebook; three deliberate simplifications are documented in
> `cards.py` (public bonus-token values, score-difference payoff, one round per game).

Goal: a second game to reveal which abstractions are real vs incidental — then refactor. The IS-MCTS agent should port **unchanged**; if it doesn't, the interface is wrong.

- [x] Engine: goods + camels, take-single / take-camels / exchange / sell, token stacks + depletion,
  bonus tokens, hand limit, camel bonus. The exchange is split into take-then-give steps, so the
  action space stays flat (63 ids), with legality guaranteeing every exchange can be completed.
- [x] Chance draws (deal, market refill, bonus tokens); information state (opponent hand hidden
  except goods seen taken from the market; market, herds and tokens public)
- [x] Engine unit tests (`tests/test_camel_stocks_engine.py`), plus the shared conformance suite
  (which now always covers every registered engine)
- [x] `information_state_tensor` encoder (per-good features aligned with the action ids)
- [x] Run the **existing** PIMC agent with no game-specific changes: ✅ it plays unchanged
  (40/40 vs random). The interface held.
- [x] Heuristic baseline (`agents/heuristic/camel_stocks_heuristic.py`): sell early and big,
  precious goods first, camels as currency
- [x] Eval vs random / heuristic: heuristic 1.000 vs random (+53); PIMC at 4 × 60 scores only
  0.258 vs the heuristic, so the heuristic is a strong anchor, as on Ventures
- [ ] Train with the Ventures recipe (Nash-seeking PPO) unchanged; Elo ladder and exploitability
- [ ] Play page: a generic `player_view` hook, so the browser UI works for Camel Stocks too

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
