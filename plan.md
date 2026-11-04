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
- **3+ player variants** — non-stationary multiplayer breaks the clean two-player zero-sum equilibrium/self-play story. All rungs are two-player, with one deliberate exception: Spendthrift (Stage 6b) runs 3 players too, because it is perfect information and solved exactly, so NashConv still measures every learner.

---

## Finishing plan (2026-10-28)

What "finished" means here: every rung on the ladder ends with a stated best agent, its strength
with error bars, and the reason each method won or failed. The repo is clean enough to hand over,
and the branch is merged. What is left, in order (each step ends, as always, with `writeup.md`
and `plan.md` updated and a commit; Phases 3 and 4 are both in, by the owner's call):

**Phase 1 — close Spendthrift (Stage 6b).** About two sessions.
1. [x] PPO on the purse variants (2026-10-29): no. The tight purse is the easiest for PPO; the
   registered 8-coin game the hardest (one seed each).
2. [x] Tie-robust three-player yardstick (2026-10-29, `robust` stage): certain loss PPO 0.094,
   search 0.077–0.086 ± 0.01; about three quarters of the old per-move loss was ties. Search
   helps three players little.
3. [x] **Owner decision** (2026-10-29): close the knobs, no set bonus.
4. [x] Stage summary in the writeup ("Spendthrift: what we learned about training tempo").

**Phase 2 — cheap loose ends on finished rungs.** One session; each is small.
1. [x] Liar's Dice bluff and call frequencies (2026-10-29): sane on every check; Stage 4 closed.
2. [x] Tessera search audit (2026-10-29): policy < late rollouts < value-based search; it does
   not point at the value, so no more loop iterations now.
3. [x] Ventures: net-guided PIMC retrained on today's features and re-rated (2026-10-29): 1866 ± 22
   (plain), 1713 ± 18 (careful), below every recent agent. Human games: only one is recorded,
   so the analysis waits on the owner playing more (`experiments/play.py` records them).

**Phase 3 — poker's finale, heads-up limit Hold'em (Stage 5).** A few sessions.
1. [x] Deep CFR on Leduc (2026-10-29, `agents/cfr/deep_cfr.py`): 0.135 after 200 iterations
   (1,500 advantage steps each; 0.21 at 300), between NFSP (0.84) and ReBeL (0.07). The regret
   networks are the bottleneck, not the strategy network.
2. [x] **Full heads-up limit Hold'em** (owner's decision, 2026-10-29: the full game, not a
   reduced one). OpenSpiel's Windows build has no `universal_poker`, so our own engine: 52
   cards, 4 betting rounds, a 7-card hand evaluator (tested against known hand rankings), behind
   the `Game` interface with the conformance suite. Agent: Deep CFR (no tables needed). Nothing
   exact is possible, so judge by large-sample head-to-head play (duplicate deals: swap the
   hole cards) against baselines (always call, a hand-strength heuristic, PPO) and a local best
   response (LBR). Several sessions; "functional, not frontier".
   - [x] Engine + evaluator (2026-10-30, `games/holdem/`), tested; baseline panel
   - [x] Deep CFR, 100 iterations (40 min): −0.08 ± 0.05 bb a hand against the equity heuristic
     (from −0.37 at iteration 10); beats random, calling station and maniac
   - [x] 200 iterations and LBR (2026-10-30): level with the equity heuristic (+0.01 ± 0.04);
     LBR wins 1.05 → 0.30 ± 0.06 bb a hand from iteration 10 to 200 (exploitability ≥ 300 mbb)
   - [ ] Future work: more traversals and advantage training per iteration, bigger networks
3. [x] OpenSpiel agents in the `iirl/eval/` tooling: done by design. OpenSpiel games run through
   the `games/spiel.py` adapter in every runner and evaluator, and OpenSpiel's solver policies
   convert to ours (`spiel_games.from_spiel`, used by `crosscheck`).

**Phase 4 — the entity rung (Stage 7, owner's choice, 2026-10-28).** The largest item left:
several sessions.
1. [x] **Owner decision** (2026-10-30): Undertow approved as drafted (rules in Stage 7).
   Was: the original design on The Confrontation's mechanical skeleton. That means
   its name, theme, the two asymmetric rosters of nine, the region graph and the combat cards, all
   our own (Reiner Knizia credited as inspiration only). I draft it, the owner approves it.
2. Engine behind the `Game`/`State` interface (hidden identities, legal-move masks), with unit
   tests and the conformance suite; a heuristic and ISMCTS (the existing PIMC) as baselines.
3. Flat-MLP PPO with the usual recipe: the baseline every entity result is measured against.
4. Entity encoder (a token per piece plus a global token; transformer; pointer head for the
   piece, target head for the region), trained by PPO self-play plus a league, then R-NaD.
5. Evaluation: Elo against the baselines, the entity-vs-flat encoder ablation, and an exploiter
   (approximate exploitability). Done when the entity agent beats both flat PPO and ISMCTS.
6. Play page board: not built (future work).

**Closed 2026-11-04 (owner's call):** PIMC search is the best Undertow agent; no network caught
it. Details in Stage 7 and the writeup.

No-limit Hold'em, Hanabi and the Rust engine (Stage 1b D) become "future work" in the writeup,
not rungs.

**Phase 5 — wrap-up.** Done 2026-11-04.
1. [x] The cross-cutting definition of done, checked game by game (2026-11-04): ruff clean
   (lint and format, 165 files); 190 tests pass, the conformance suite parametrised over all 13
   registered games; every game module credits its inspiration (Mini-Ventures through
   Ventures; Hold'em is public domain); seeds are config fields, and duplicate matches draw
   each deal from a fixed seed stream:
   - Docstrings and lint: `ruff` already enforces both.
   - The conformance suite runs on every registered game: already true, so confirm it.
   - Seeds recorded for every headline result.
   - Credits in every game module.
2. [x] README (2026-11-04): replace the Ventures-only headline with the per-game table (best agent, strength, why),
   and give each game a short section (rules in brief, encoding, agent, results).
3. [x] Checkpoints (**owner decision**, 2026-11-04: document retraining; publish nothing). The
   README's "Retraining the headline agents" lists the commands per game. Was: `runs/` is
   git-ignored, so a fresh clone cannot run the play page or reproduce the
   headline agents. **Owner decision:** publish the handful of headline checkpoints (for example
   as a GitHub release), or document how to retrain them.
4. [x] Writeup (2026-11-04, "Closing summary"): a closing "What we learned" across all games (when each method wins, and why), and
   "Future work".
5. [x] Merge `feat/rl-agent-ladder` into `main` through a PR (owner's call, 2026-11-04: open
   and merge once CI passes).

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
- [x] **E0. ReBeL where it is exact (started 2026-09-30; concluded 2026-10-12: exact on
  Mini-Ventures, loses at full Ventures, rollouts won instead).** Correction to E below: Ventures'
  hidden 8-card hand has ~10^8 possibilities, so exact public belief states are out of reach.
  ReBeL's published results are on small private-info spaces. So build it first on a small game,
  where it is exact and every claim can be checked:
  1. [x] **Mini-Ventures** (`games/mini_ventures`): the same mechanics, 2 suits × (backing, 2–5),
     3-card hands (≤ 120 possible hands per player).
  2. [x] **Scalable exact exploitability** (enumerate each state once; forward reach, backward
     values) and a policy-network → strategy-table converter.
  3. [x] **Ground truth for our methods.** PPO (then Nash PPO) reaches **0.07 points/game** of
     exact exploitability (random 2.71, equilibrium 0.002). Exploiter calibration vs exact best
     responses: the standard recipe finds 61% / 1% / 7% of weaknesses of 1.54 / 0.36 / 0.07;
     warm start + lr 3e-4 + entropy finds 92% / 55% / 20%; from scratch 90% / 68% / −7%.
     (A first version, "the yardstick finds ~4%", was retracted: a degenerate game and starved PPO.)
  3b. [x] **Re-measure `diverse_sym` and `camel_nash`** with the stronger recipes: nothing more
     found (0.500 / 0.498 and 0.500 / 0.505), so the estimates stay 0.512 and 0.509.
  4. [x] **ReBeL** (done 2026-10-01, `selfplay/rebel.py`): exact exploitability **0.002**
     points/game after 20–40 epochs (21 min), level with full-game CFR+ (0.0022); PPO 0.07.
     Needed: random-iteration play (not unsafe re-solving), trembling-hand beliefs (first run
     scored 1.30 without them), best-response last turn. Original item:
     public belief states, CFR over depth-limited subgames with value-net leaves,
     self-play value training; exact exploitability vs step 3.
  5. [x] **E1:** scale to full Ventures with factored (per-card) beliefs; judge vs `diverse_sym`
     (done: ReBeL loses; the audit led to opponent-robust rollouts, the best Ventures agent).
     - [x] **Price the approximations on Mini-Ventures** (2026-10-01): factored beliefs 0.0037,
       + per-hand values 0.0028, + 256 sampled public states per pass 0.0035 (exact: 0.0025;
       PPO 0.07). Sampling needs data: 0.093 at 2.5k last-turn examples, 0.006 at 5k.
     - [x] **One-position solver** over candidate hands through the game interface
       (`selfplay/rebel_turn.py`), validated against the full-width solver; Ventures hooks added.
       Cost on Ventures: ~2 s per decision at 16 candidates × 20 iterations.
     - [x] **Sampled deck draws** + no opponent rows at play time: ~7× fewer network rows.
     - [x] **Public belief tracking** (`selfplay/public_belief.py`): overlap with the true hand
       0.354 vs 0.296 without the policy model.
     - [x] **Ventures value network** (`selfplay/rebel_value.py`), trained on `diverse_sym`
       self-play with one random move in 60% of games: held-out R² 0.46, RMSE 32 points; beliefs
       add nothing.
     - [x] **Agent + match** (2026-10-02): **loses.** Plain search 0.050 (−110 points a game);
       anchored to the blueprint 0.435 / 0.405. Cause: value error ~10 points vs ~1-point gaps
       between moves; the search picks the maximum of noise.
     - [x] **Audit: where does search suit Ventures?** (2026-10-02,
       `experiments/ventures_search_audit.py`; user chose to stay on Ventures.) Paired rollouts on
       572 positions: the search loses 4-6 points a turn, worst when it swaps to a discard the
       opponent can play (−10.5); the policy is improvable only in the endgame (deck ≤ 6: at
       least +1.05 ± 0.37 a turn); luck is 22-38 points per paired rollout; the opponent's hand
       shifts a move's value by 2-4 points.
     - [x] **Late-game paired rollouts on the policy** (`agents/rollout.py`): **beats
       `diverse_sym`**, 0.526 ± 0.006 from 6 cards left, 0.538 ± 0.008 from 12; with
       belief-drawn re-deals 0.532 ± 0.008 (not significantly better).
     - [x] **Robustness panel for the rollout agent** (2026-10-05, 500 deals per style): level
       with plain `diverse_sym` on four styles (+0.014, −0.007, +0.006, −0.009), worse against
       tight (0.904 vs 0.934, −9.6 points a game). The gain is specific to opponents the
       rollouts model correctly.
     - [x] **Opponent-robust rollouts** (2026-10-06; `robust=True`: rollouts under `diverse_sym`,
       careful and tight; override only if not worse under any): 0.524 ± 0.006 head to head, and
       better than plain `diverse_sym` against all five styles (0.918 / 0.827 / 0.956 / 0.843 /
       0.829 vs 0.887 / 0.795 / 0.934 / 0.823 / 0.812). **Best Ventures agent so far.**
     - [x] **Do counters to `diverse_sym` transfer to the rollout agent?** (2026-10-09,
       `experiments/exploit_rollout.py`, 500 paired deals each.) No: all five attackers do worse
       against it (margins −1.4 to −2.0 points); the best exploiter drops from 0.528 (+1.7) to
       0.510 (−0.1). Indirect: none was trained against the rollout agent.
     - [x] **Distil the rollout agent's endgame choices into the policy** (2026-10-12,
       `experiments/distill_rollout.py`; 6,000 games, 4,715 overrides): **fails.** Strong
       imitation reproduces 13.5% of held-out overrides and scores 0.484 vs `diverse_sym`; gentle
       variants reproduce ~2% and score 0.497 / 0.500. The gain stays in the search.
     - [x] **Faster rollouts** (2026-10-12): per-search move cache (identical decisions) and
       `keep=3` (drop weak moves after 2 worlds): ~3× faster; 0.517 ± 0.003 head to head vs the
       full form's 0.524 ± 0.006.
     - [x] **Direct exploiter vs the fast rollout agent** (60 iterations, standard recipe, warm
       start from the best exploiter of `diverse_sym`): **0.493 ± 0.010** on 1,000 fresh deals;
       no snapshot beat the starting point. Reduced budget, so a weak attack.
     - [x] **Rollout agent in the play page** (2026-10-14): first Ventures opponent in
       `experiments/play.py` (`rollout:runs/diverse_sym.pt`).
     - [ ] **Possible next steps.** Stronger exploiter recipes at full budget (~10 h each at
       this speed); rollouts from earlier than 6 cards with the faster code; the rollout agent in
       the play page; or leave Ventures here and carry the lessons to the next game.
- [x] **E. The principled endgame — ReBeL / Student of Games** (superseded by E0/E1 above). Search over public belief states
  (the only hidden information is one 8-card hand, so beliefs are tractable), CFR at each search
  step, value net over belief states, trained by self-play. Only if B + C plateau above zero
  exploitability.

**Next steps (2026-09-29), in order.** Each is useful alone; stop wherever the curve flattens.

1. [x] **Finish `nash_g`.** Ladder 1969/1982/1975/1962 (flat, within noise of league_control's
   1973); exploitability 0.547 (max of 2). The recipe has plateaued at ~1970 Elo / ~0.545. The
   stall persists (1 in 300 self-play games, as before): the leader still rarely draws the last
   card, a position too rare to learn in 400 iterations. Harmless.
2. [x] **Refresh the stale number** (2026-10-29: retrained on today's features, 1866 ± 22 plain,
   1713 ± 18 careful). Rate net-guided PIMC on today's yardstick: duplicate deals
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
   - [x] **R-NaD magnet averaging** (`magnet_ema = 0.02`, vs `nash_g`): no gain. Ladder
     1968/1983/1962/1979, exploitability 0.560 (vs 0.547).
3b. [ ] **Fix the leaky anchor (found 2026-09-29).** Against a heuristic that never gifts a
   playable card (`ventures_careful`), `league_control` scores only 0.570 (+6.8), against 0.816
   vs the plain one; PPO v2 loses (0.441). Next:
   - [x] Re-anchor the Ventures ladder on the careful heuristic (report both anchors): done,
     the best agent is 2070 (plain) / 1960 (careful).
   - [x] (done as the careful-only league, then `diverse` and `diverse_sym`, step 3c)
     Continue Nash training with the careful heuristic in the league instead of the plain
     one; judge by score vs careful and by exploitability (max of ≥ 2 runs).
   - [x] A "careful" Camel Stocks heuristic (`camel_stocks_careful`): no dependence found. The
     agent beats it 0.966, more than the plain one (0.940).
   - [ ] Analyse recorded human games (`runs/human_games/`) with the same conditional questions.
3c. [ ] **Approaches to try next**, in order (the leak says the bottleneck is what the agent trains
   and is measured against, more than the learning algorithm):
   1. [x] (tried as `diverse`: slightly worse than the careful-only league; weakest matchup 0.681
      vs 0.716, exploitability 0.568 vs 0.544) **Style-diverse league**: several fixed opponents with distinct habits (careful, tight,
      aggressive, hoarder), not one heuristic. AlphaStar's lesson, and ours: one opponent with one
      habit taught a habit-specific exploit.
   2. [x] (DrAC-style regularizer, `diverse_sym`: **best agent**; exploitability 0.504, weakest
      matchup 0.795; beats careful 0.569; validated: the exploiter finds 0.530 on its 100-iteration
      checkpoint) **Suit-symmetry augmentation** (Ventures): the five suits are interchangeable, so train on
      randomly relabelled positions (feature and action permutations). Up to 120× the effective
      data, and no accidental suit preferences.
   3. [x] (4 rounds from {diverse_sym, careful, league_control}: best responses vs the Nash
      mixture score 0.504 / 0.514 / 0.507 / 0.514; near equilibrium within the population)
      **Proper PSRO**: a population payoff matrix, and train against its Nash mixture, not
      uniform fixed opponents (why the exploiter league failed).
   4. [x] (negative: depth-limited MCCFR never beats the policy with heuristic, value-net or
      blended leaves; best 0.452 head to head. Its subgame assumes the opponent knows our hand;
      the fix is Phase E, not tuning) **Endgame search**: few cards left means a small tree; search it deeply, where a human can
      out-calculate a policy (clock management, the stall).
   5. **Human games as a test set**: where does a human beat it, and in which positions?
4. [x] **Decision point: go deeper on Ventures, or move to Jaipur** (both, in the end: E0/E1
   on Ventures, then Camel Stocks).
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
- [x] Train with the Ventures recipe (Nash-seeking PPO) unchanged (`experiments/train_camel_stocks.py`):
  ✅ the recipe transfers. Ladder (heuristic = 1500): PPO 1810 → 1889, Nash phase → **1968–1972**,
  0.94 vs the heuristic. Exploitability **0.509** (max of 2), validated: the same exploiter finds
  0.571 on the 100-iteration checkpoint. Close to unexploitable within this budget.
- [x] Held-out opponent: vs PIMC (8 × 80), which is roughly even with the heuristic, it scores
  **0.950 ± 0.026** (+23.3), as decisively as vs the heuristic it trained with.
- [x] Play page for any game: `Game.player_view` hook; game-agnostic play server; per-game board
  renderers with a generic fallback (`experiments/play.py`, `viz/play.html`)
- [x] **The Ventures search approach on Camel Stocks** (2026-10-16): paired-rollout audit
  (`experiments/search_audit.py`) finds the policy improvable from the middle game, ~2 points a
  turn from 12 cards left; robust rollouts from 12 cards: **0.620 ± 0.007** vs `camel_nash`
  (+3.8 points), 0.978 / 0.988 vs the two heuristic styles (policy: 0.936 / 0.966), Elo
  2151 ± 24; 5.5 s of thinking a game. From 25 cards: 12× slower, not worth it. In the play
  page. Open: exploiter and PIMC against it.

**Refactor block — harden the game-agnostic core:**

- [x] Extract a stable, documented interface contract (both games conform): `docs/adding-a-game.md`
  lists the required methods and which component uses each optional hook
- [x] Confirm `agents/` contain zero game-specific branches: the generic agents, trainers and
  evaluators had none. The heuristics moved into their games (`games/<name>/heuristic.py`) and
  register with `register_heuristic`; game-agnostic code gets one via `make_heuristic(name)`
- [x] ~~Base `Encoder` abstraction~~ **Decided against.** With two games, the encoder as a `Game`
  method (`information_state_tensor`) works well, and the only thing agents build on top (the
  privileged critic's input) is already generic. A parallel class hierarchy would add indirection
  with no user.
- [x] Registry cleanup (heuristic registry) and the "how to add a game" doc

**Done when:** adding a third game would touch only `games/` and an encoder — nothing in `agents/`.

---

## Stage 3 — Tessera *(Azul's skeleton; perfect-info apart from the bag)*

> **IP note:** shipped as **Tessera**, an original game on the mechanical skeleton of Azul (Michael
> Kiesling), credited as inspiration only: own name, colour names, no assets or rulebook text.
> Package `iirl/games/tessera/`, registry key `tessera`. Deliberate differences (two players,
> score-difference payoff, a 20-round cap) are documented in `tiles.py`.

Goal: the first game where a learned value *should* support search (no hidden hand, outcomes that
follow from decisions), so the place to test whether AlphaZero-style search finally beats the
recipe, which it never did on the card games.

- [x] Engine (2026-10-19): five displays + centre, colour drafting, pattern rows, fixed-colour
  wall, floor line, adjacency scoring, end bonuses, bag/lid refills with every tile draw a chance
  node. Flat action space of 180 ids (display or centre × colour × row or floor)
- [x] Engine unit tests (`tests/test_tessera_engine.py`): wall layout, scoring geometry, floor and
  end bonuses, row rules, tiling, the marker, tile conservation over random games
- [x] Encoder (`information_state_tensor`, 225 features: both boards, which rows accept which
  colour, displays, centre, supply, clock) and a greedy heuristic (beats random ≥ 90%;
  heuristic self-play ~55 points each over 5 rounds, like the original)
- [x] The recipe unchanged (`experiments/train_tessera.py`, 2026-10-19): plain PPO plateaus
  **below** the heuristic (0.40 after 600 iterations, Elo ~1439); the Nash phase with the heuristic
  in the league passes it (0.69, +8.2 points, Elo ~1652 in-training) and is still climbing.
  Exploitability 0.543 (max of 2). Not yet converged.
- [x] Second league phase (1,200 iterations): 0.79 vs the heuristic, +13.3, Elo ~1734, flat;
  exploitability 0.505
- [x] **AlphaGo recipe** (2026-10-19, `experiments/alphago_tessera.py`): value net Brier 0.193
  (heuristic 0.209); PUCT with policy priors + value leaves: 100 sims 0.551 vs the policy / 0.81 vs
  the heuristic; **400 sims 0.941 / 0.938; 1,600 sims 0.999 / 0.969 (Elo ~2097)**. Strong.
- [x] AlphaZero loop (`alphago_tessera exit`, 24 iterations): the raw policy rises to 0.612 vs the
  starting policy (+3.6); search at 100 sims vs the heuristic stays ~0.83-0.88 (flat)
- [x] Search agent (400 sims) in the play page
- [x] Search with the loop's nets (2026-10-24): 400 sims 0.970 vs the original policy, 0.953 vs
  the heuristic; 1,600 sims 0.986 vs the heuristic (Elo ~2242); +2 to +3 points over the original
  nets. Now the play-page opponent.
- [x] Tessera board renderer in `viz/play.html` from the Ladder design system
- [x] Ladder design system applied to the Ventures and Camel Stocks boards too (2026-10-26): its
  tokens as CSS variables, one contrast-based label rule for every piece, no piece colours on
  interface elements
- [x] More loop iterations (24 → 48) and a 1,000-deal Tessera Elo ladder (2026-10-26): search 400
  on the iteration-48 nets 2057 ± 21 Elo (iteration 24: 1993), 0.545 ± 0.016 head to head; raw
  policy flat (1740 → 1754). A policy anchor inflates search agents ~300 Elo (non-transitive).
  Play page now uses the iteration-48 nets.
- [x] Per-move latency (2026-10-27, `experiments/search_latency.py`, one thread): 400 sims median
  0.14 s, worst 0.26 s, 3.6 s/game; 1,600 sims median 0.74 s, worst 1.0 s, 16 s/game
- [ ] More loop iterations (search still gaining ~+30 Elo per 12 iterations): future work
- [x] Search audit (2026-10-29, `search_audit.py tessera`): a better move gains at least +3 to
  +6 points a turn from round 2 on; luck needs ~1,400 rollouts per move early against 21 late.
  Answer: policy < rollouts (late only) < value-based search (throughout)
- [x] AlphaZero-style search with a learned value (the plan's original Stage 3), judged against
  the recipe (done: the AlphaGo recipe and loop above; the comparison with audit-chosen rollouts
  moves to the search-audit item)
- [x] Eval: Elo-vs-iteration curve (the 1,000-deal ladder over iterations 24/36/48); play page

**Done when:** we know which of policy, rollouts and value-based search wins on a perfect-information
game, and why.

---

## Stage 4 — Liar's Dice *(first CFR, on my own engine; bridge to equilibrium)*

Goal: implement CFR from scratch on a small game I built — the paradigm shift *before* poker, on something I can eyeball. Watch it converge to mixed (bluffing) strategies. Validate against OpenSpiel.

- [x] Engine: OpenSpiel's `liars_dice` (decided 2026-10-27: use DeepMind's engine) behind our
  interface via `games/spiel.py`; any two-player zero-sum OpenSpiel game can be registered
- [x] Adapter tests (`tests/test_spiel.py`; conformance suite)
- [x] Tabular CFR (regret matching) + average-strategy accumulation in `agents/cfr/` (existing)
- [x] Best-response / exploitability for this game (existing `eval/exact_br.py`; matches OpenSpiel
  to 6 decimals)
- [x] CFR+ (graph CFR+, now with alternating updates: identical to OpenSpiel's CFR+, ~4× faster);
  MCCFR not yet
- [x] Cross-validate vs OpenSpiel (2026-10-27): exploitability 0.00013 after 1,000 iterations,
  first bidder's value −0.02713
- [x] Headline (from Stage 6, 2026-10-27): PPO self-play vs CFR on Liar's Dice, by exact
  exploitability: PPO and three Nash-phase variants stay 0.3–0.5 (payoff ±1) and cycle; CFR+
  0.00013 in 31 s
- [x] ~~2 dice each~~ dropped: out of reach for exact methods (~2^24 bid sequences per roll); would need
  MCCFR / Deep CFR and loses the exact yardstick
- [x] Eval: bluff and call frequencies of the CFR+ equilibrium (2026-10-29, `spiel_games
  strategy`): never calls a true bid, always calls an impossible one, bluffs 97% holding a 1 or 2
  and never with a six. No heuristic to play (OpenSpiel's engine); exact exploitability covers it

**Done when:** exploitability drives toward zero and my values match OpenSpiel's reference.

---

## Stage 5 — Poker via OpenSpiel *(Kuhn → Leduc → limit Hold'em)*

Goal: adopt the framework where it earns its place — CFR at scale, with verifiable results. Kuhn to learn the tooling, limit Hold'em as the substantive, checkable finale. No abstraction rabbit hole.

- [x] Add `[openspiel]` optional dependency (`uv sync --extra openspiel`; Windows wheel 2.0.2)
- [x] **Kuhn poker** — CFR+ to the known equilibrium (value −1/18), exact exploitability (2026-10-27)
- [x] **Leduc poker** — CFR+ 0.00023 in 0.9 s; PPO 1.3–2.2 exploitable (uniform 2.37) (2026-10-27); MCCFR not yet
- [x] **NFSP and ReBeL on Leduc** — ReBeL 0.07 (plateau from epoch 20), NFSP 0.84 at best (2026-10-27)
- [x] ReBeL on Liar's Dice, 20 epochs: 0.047 (PPO 0.31–0.52); sampled evaluation (2026-10-28)
- [x] **Deep CFR** (neural) on Leduc (2026-10-29): 0.135 exploitability; limit next
- [x] **Heads-up limit Hold'em** — Deep CFR on our own engine (2026-10-30): level with an equity
  heuristic, LBR ≥ 300 mbb a hand (Cepheus, the solved reference, is near 0)
- [x] Evaluation discipline: exact exploitability on the small games; duplicate deals (hole cards
  swapped) and LBR on Hold'em. AIVAT-style variance reduction: future work
- [x] Wire OpenSpiel agents into `iirl/eval/` tournament runner (via the adapter and `from_spiel`)

**Done when:** a limit Hold'em agent with low measured exploitability, evaluated honestly, runnable through my own eval harness.

---

## Stage 6 — Stretch / research edge *(optional, pick what stays fun)*

- [x] **PPO** in `agents/ppo/` (built in Stage 1, the base of every recipe since) — foundational algorithm to own (also the RLHF/post-training bridge to day-job + SLM work)
- [x] **Headline experiment:** PPO self-play vs CFR on Liar's Dice — measure the **exploitability gap** (done 2026-10-27, Stage 4). Make "why the field uses regret minimisation, not vanilla policy gradients" something I've *watched happen*.
- [x] **NFSP** (PPO best response; Camel Stocks, vs Nash PPO at equal budget: exploitability 0.515 vs 0.509, but weaker: 0.456 head to head) — the bridge that targets equilibrium with RL-style best-response; compare against CFR and PPO
- [ ] **No-limit Hold'em (scoped)** — blueprint agent over a small fixed action abstraction, explicitly "functional, not frontier"; explore action/card abstraction + the translation problem. *Not* aimed at strength.
- [ ] **Hanabi (OpenSpiel)** — cooperative, imperfect-info; self-play coordination, reaching toward the open zero-shot / cross-play problem. The highest-ceiling rung the framework makes cheap.
- [x] Short writeup / notebook per experiment (every experiment has a `writeup.md` findings entry)

---

## Stage 6b — Spendthrift *(tempo in an auction game; original design, 2-3 players)*

The owner's question: how do we train the tempo part of an auction game (when to spend a purse
that must last an unknown number of rounds)? Perfect information with chance, so small sizes have
exact answers to compare every learner against.

- [x] Engine (`games/spendthrift/`), 2 and 3 players, registered at exactly solvable sizes (2026-10-28)
- [x] Exact solver for perfect-information games with any number of players (`eval/perfect_info.py`:
  backward induction, `nash_conv`)
- [x] The equilibrium's tempo profile: prices by lot and end cards seen, unsold lots, reserve
  penalty frequency (`experiments/spendthrift.py solve`)
- [x] Controlled tempo question on the exact solution (`tempo` stage, 2026-10-28): an end card
  makes the last coin 18% cheaper (0.47 → 0.39 of a win) but leaves the price unchanged in 88% of
  positions; coin value rises as lots run out
- [x] PPO self-play on both sizes (2026-10-28; `selfplay/ppo.py` now seats n players): NashConv
  0.36 (2 players) / 0.29 (3) at 200 iterations; with two players the opening is learned (0.001 per
  decision) but the last lots are not (0.135), where coins decide the reserve penalty
- [x] PUCT search on the PPO net, 2 players (2026-10-28): the endgame errors go (last lot 0.049 →
  0, 1 left 0.096 → 0.034 at 800 simulations) but the opening gets worse (0.002 → 0.019); loss per
  game 0.33 → 0.20, flat beyond 200 simulations; head to head +0.08 ± 0.05 (not significant)
- [x] **Next, in order** (each scored move by move against the exact solution, as in `search`,
  plus seat-balanced games against greedy PPO; 300 games per setting; one thread):
  1. [x] Endgame-only search (2026-10-28): from 2 lots left at 800 simulations, 0.168 ± 0.016
     per game (full search 0.201, PPO 0.329), opening kept at 0.002, +0.10 ± 0.05 vs PPO.
     Planned as: the PPO policy while many lots remain, PUCT search once K or fewer
     are left (K = 0, 1, 2). Cheap, and it tests why the opening got worse: if the PPO value head
     at the leaves is the problem, switching search on only near the end should keep the opening at
     0.002 per decision and the endgame at search's level
  2. [x] Value net (2026-10-28): search on it 0.140 ± 0.014 per game at 800 simulations (+0.11 ±
     0.05 vs PPO), exact leaves 0.085 with the opening intact; PPO's head was off by 1.34 off the
     path. Remaining gap is the target (PPO-vs-PPO value, 0.28 from equilibrium). Planned as: a
     value net trained on self-play outcomes (the Tessera recipe), then search on it at
     50–800 simulations: does a better leaf value stop the opening getting worse as simulations
     grow?
  3. [x] 3-player search (2026-10-28; `agents/alphazero/multiplayer.py`, max-n PUCT): +0.09 ±
     0.04 vs PPO at 200 simulations with an every-view value net, +0.00 at 800; exact leaves
     +0.02. The per-move score against the solver is unreliable for 3 players (kingmaking ties).
     PPO's critic is unusable from a non-mover's view. Planned as: PUCT backing up a value per
     player (`spendthrift3` is not zero-sum
     between any two seats), scored the same way against the solver's equilibrium and by NashConv
     where affordable
- [x] Knobs, exact (2026-10-28; `knobs` stage, 1–3 end cards × 6–10 coins): spending per lot
  flipped does not depend on how predictable the end is; the purse sets prices (~8% of it per
  point); only tight purses make the last coin precious late
- [x] PPO on the extreme variants (2026-10-29; `spendthrift_tight`, `spendthrift_loose`): the
  prediction failed. 6 coins is easiest (NashConv 0.24, 0.20 per game), the registered 8 coins
  hardest (0.36, 0.32), 10 coins between (0.30, 0.25). One seed each.
- [x] A set bonus: closed by the owner (2026-10-29). A size beyond exact reach: future work
- [x] 3 players: a tie-robust yardstick (2026-10-29): certain loss PPO 0.094, search
  0.077–0.086; three quarters of the old per-move loss was ties

---

## Stage 7 — LotR: The Confrontation *(entity rung)*

> **IP note:** as with Ventures and Camel Stocks, ship it under an original name and theme on the
> game's mechanical skeleton (Reiner Knizia's design credited as inspiration only): no names,
> characters, art or rulebook text from the published game.

Goal: first entity-token policy. Pieces are distinct, hidden-identity
individuals; actions are "this piece → that region".

**Undertow v2** (2026-11-02). The design was approved on 2026-10-30. On 2026-11-02 the owner
rebalanced it after the first version proved one-sided: a single Contraband boat, both sides
advance, and every balance mechanism of the original's rules summary, followed where unsure. The
names, theme, board, characters, abilities and card names stay our own; Knizia is credited as
inspiration only. The full rules are in the `games/undertow/game.py` docstring. In brief:

- **Board:** a mirror-symmetric delta of 14 regions (1-2-3-2-3-2-1), from the Cove to the Free Port.
  Capacity: home 4, shoals 1, channels 2. Sideways links join channels only.
- **Setup:** each side *places* its nine pieces hidden: four at home, one in each of five front
  regions.
- **Moving:** the Watch moves first. Each turn a side must move a piece *forward* (Smugglers
  downstream, the Watch upstream); a side that cannot move loses. Nothing may move into or through
  a region full of its own pieces.
- **Fights:** with two or more defenders, the attacker draws one at random. Both are revealed and
  each plays a card in secret. Trick cards resolve before numbers, the Watch's first. The lower
  total loses; a tie defeats both. Fights continue until one side holds the region.
- **Memory:** pieces are hidden again after a fight and may be shuffled within a region. Each side
  tracks a set of possible identities per enemy token; tokens sharing a region merge their sets.
- **Cards** (9 each, recycled when the hand runs out):
  - Watch: 1–6, Lantern (ignore the Smugglers' trick), Reinforce (replay your best used number),
    Withdraw (retreat sideways).
  - Smugglers: 1–5, Fog (ignore the Watch's number), Reinforce, Feint (retreat upstream), Scuttle
    (both sink).
- **Pieces:**
  - Smugglers: Captain 5 (Watch tricks ignored), Bruiser 4, Wrecker 3 (takes its killer along),
    Fixer 3 (keeps its card on a win), Runner 2 (two regions to an empty region), Pilot 2 (ignores
    capacity), Skiff 2, Contraband 1 (slips away sideways if attacked), Decoy 0 (a loss as
    defender turns the attacker back).
  - Watch: Gunboat 7 (its card is ignored), Harbourmaster 5 (Smuggler tricks ignored), Sergeant 4,
    Cutter 3 (jumps two to attack), Diver 3 (attacks sideways), Patrols 3 and 2, Inspector 2
    (privately identifies one adjacent Smuggler a turn), Signal 1 (when lost, the Watch recovers
    its used cards).
- **Winning:** the Smugglers win the moment the Contraband enters the Free Port, guarded or not.
  The Watch wins by sinking the Contraband or with three pieces in the Cove. A side with no move
  loses. 60 rounds is a draw (a safety limit).
- **Evaluation:** as the rules recommend, play both sides; our duplicate matches already swap seats.

- [x] Engine (2026-11-02, `games/undertow/`), tested; heuristic for both sides; balance tool
  (`experiments/undertow.py`)
- [x] **Owner decision: rebalance** (2026-11-02): one Contraband boat, both sides advance, every
  mechanism of the original's rules summary (above). Heuristic against heuristic: Smugglers 19%
  (v1: 0%); a fair benchmark, since evaluation swaps sides
- [x] PIMC baseline (2026-11-02): beats the heuristic on both sides; Smuggler win rate 55.5%
  as PIMC vs the heuristic Watch, 7.0% as the heuristic vs a PIMC Watch (16 × 80)
- [x] Flat PPO (2026-11-02, 300 iterations): ~1650 Elo against the heuristic (1500), plateau from
  iteration 175
- [x] Entity policy (`nn/entity.py`), same budget as flat PPO (2026-11-03): **lags the MLP**.
  Elo against the heuristic at iterations 25-300: 1265 / 1428 / 1433 / 1435 / 1502 / 1455 / 1477 /
  1519 / 1538 / 1545 / 1528 / 1484 (flat PPO: plateau ~1650 from iteration 175). About 3.5× the
  compute per iteration.
- [x] `pointer` (region tokens + dot-product pointer, 2026-11-03): fastest learner (1629 Elo at
  iteration 125), then collapsed (1228 at 225; stopped)
- [x] Head to head per seating (2026-11-03, 200 games each): MLP 0.46 vs transformer, 0.60 vs
  pointer-100; **PIMC beats every network 0.77–0.85**. Non-transitive. Goal not met
- [x] Imitation of PIMC (2026-11-04): 58,258 decisions from 1,000 PIMC (8 × 40) self-play games
  (70 min); the pointer network matches PIMC's move 31.5% on held-out games (cross-entropy 1.73).
  Head to head it is the weakest network (0.18–0.32 against every opponent), though its Watch
  holds PIMC's Smugglers to 66.5%, better than any other network (71–77%)
- [x] PPO fine-tune from the imitation network (learning rate 1e-4, 2026-11-04): **negative**.
  Stuck at 1350–1440 Elo, 1293 at iteration 200 (from scratch at 3e-4: 1629 peak, then collapse)
- [x] **Owner decision** (2026-11-04): **close Stage 7** with these findings. Its goal (an
  entity agent that beats both flat PPO and PIMC) is not met; the untried levers are future
  work (writeup, "Closing summary")
- [x] Architecture review against warcraftsim (warlock.ai/blog/warcraftsim-1, 2026-11-02). Same
  core idea (piece tokens, all-to-all attention, a privileged value). Gaps, in order:
  1. A real pointer: the 14 regions as tokens (occupancy, capacity left, possible enemy
     identities) and moves scored as piece · region. Ours scores regions with a linear layer
     blind to their contents.
  2. Behaviour memory: per-token history features (regions advanced, rounds since moved or
     revealed, fights survived); recurrence (their minGRU) only if needed. Identity memory is
     already exact (possible-identity sets).
  3. If the entity policy lags flat PPO: imitation warm start from PIMC, league plus magnet
     anchor, UPGO.
  4. Size (they use 5 layers, 3.2M weights; ours 2 × 64, ~70k) only after 1 to 3.
- [ ] (was) Engine: 9v9 asymmetric characters, region graph, hidden identities,
      combat cards, legal-move masks — behind the `Game`/`State` interface
- [ ] Encoder: one token per character + a global token; own pieces carry
      identity, enemy pieces carry "unknown" or revealed identity;
      pad + mask as characters die
- [ ] Policy: transformer over tokens → pointer head (which character) →
      target head (destination region); value head for training only
- [ ] Baselines: random, ISMCTS, flat-MLP PPO
- [ ] Training: PPO self-play + league of past selves, then R-NaD
- [ ] Eval: Elo vs baselines; entity-vs-flat encoder ablation;
      approximate exploitability via best response if tractable

**Done when:** the entity agent beats both flat-MLP PPO and ISMCTS.

---

## Cross-cutting definition of done

- [ ] Google-style docstrings; clean module separation; no argparse (config dataclasses + minimal runnable blocks)
- [ ] Engine unit tests + shared conformance suite green for every game
- [ ] Per-game README section: rules summary, encoding, agent, results
- [ ] Reproducible seeds; checkpoints for trained agents
- [ ] No copyrighted assets; designer/publisher credits; permissive license on my code
