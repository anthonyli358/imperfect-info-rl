# Findings — imperfect-info-rl

A living log of what we've learned building and pitting agents against each other. The headline
summarizes the current state, "The October investigation" tells the latest line of work start to
finish, and the findings log below has the details, newest first.
Scores are head-to-head score rates (win 1, draw 0.5) over duplicate deals unless noted; Ventures Elo
is anchored on random = 1000 and the plain heuristic = 1676.

> Written for: contributors and readers of this repo — engineers/researchers who want the *why*, not
> just the scores. A new session (human or agent) should start with "Start here" below.

Project: imperfect-info-rl in c:\Users\Ant\Documents\My Projects\GitHub\imperfect-info-rl
(branch feat/rl-agent-ladder). Read the "Start here" section at the top of writeup.md, then
continue with its "Open threads", starting with thread 1.

---

## Start here (handoff for a new session)

Read this section first; it is enough to pick the work up. Everything below it is the detailed
record.

**The project.** `imperfect-info-rl` (branch `feat/rl-agent-ladder`): self-play agents for a ladder
of two-player games, built on one game-agnostic `Game`/`State` interface (OpenSpiel-style) in
`src/iirl/`. The owner's goal is the strongest, least exploitable agent per game, built step by
step, while expanding to more games and trying more approaches. Games so far:

| Game | What it is | Best agent | Strength | Play it |
|---|---|---|---|---|
| **Ventures** | Original card game on Lost Cities' mechanics; hidden hands, heavy luck | `diverse_sym` policy + opponent-robust rollouts from 6 cards left (`agents/rollout.py`) | Elo 2070 (plain anchor) / 1960 (careful anchor); 0.524 vs the policy alone; no exploiter beats it (0.493) | `experiments/play.py` |
| **Camel Stocks** | Original card game on Jaipur's mechanics | `camel_nash` policy + the same rollouts from 12 cards left | 0.620 vs the policy alone (+3.8 points); 0.978 / 0.988 vs the two heuristic styles | same page |
| **Tessera** | Original tile game on Azul's mechanics; perfect information apart from the bag | Policy + value net + PUCT search, both nets improved by 48 iterations of an AlphaZero loop (`experiments/alphago_tessera.py`, `runs/tessera_ag/exit_*_48.pt`) | 400 sims: 2057 ± 21 Elo (heuristic = 1500; 0.961 vs the heuristic), 0.545 vs the same search on the iteration-24 nets | same page (400 sims), with a real board |
| Mini-Ventures | Ventures shrunk until it is solvable exactly | full-game CFR+; ReBeL reaches it | exact exploitability 0.002 | — |
| Liar's Dice | OpenSpiel's engine (1 die each), through `games/spiel.py` | full-game CFR+ (alternating) | exploitability 0.00013 after 1,000 iterations; first bidder's value −0.0271 | — |
| Kuhn, Leduc poker | OpenSpiel's engines, same adapter | full-game CFR+ (alternating) | 0.00009 / 0.00023 after 1,000 iterations; first player's value −0.0556 (= −1/18) / −0.0857 | — |

**What we know works, by kind of game** (details in "What we learned" and the findings log):

- *Every game:* PPO self-play with good features, potential shaping from a heuristic, a privileged
  critic, a win bonus, then a league phase with the heuristic and past selves as opponents. Train
  against **non-leaky** opponents (a heuristic that gifts cards teaches dependence on gifts).
- *Hidden information and heavy luck (Ventures, Camel Stocks):* search on a learned value fails
  (the value is ~10 points off on moves ~1 point apart). What works is a **paired-rollout audit**
  (`experiments/search_audit.py`) to find where the policy is improvable, then **opponent-robust
  rollouts** only there. ReBeL is exact on small games but loses on Ventures.
- *Perfect information (Tessera):* the **AlphaGo recipe** works: a value net trained on the
  policy's self-play, PUCT search with policy priors and value leaves. Strength climbs steadily with
  simulations. "Nash-seeking" is unnecessary there (minimax needs no mixing).

**Open threads, in the order I would take them:**

1. ~~Tessera board renderer~~ Done 2026-10-24 (`viz/play.html`, from the **Ladder design system**:
   https://claude.ai/artifact/TjrPnNNCcB8fDiSHoFn4c5). ~~Apply the system to the Ventures and
   Camel Stocks boards~~ Done 2026-10-26 (findings log, "The play page follows the Ladder design
   system"). Left: the system's own `bundle.css` still prints white labels with a text shadow and
   could take the page's label rule.
2. ~~Search with the AlphaZero loop's nets~~ Done: +2 to +3 points a game over the original nets.
   ~~More loop iterations; a 1,000-deal Elo ladder~~ Done 2026-10-26 (findings log, "Tessera: 24
   more loop iterations"): search on the iteration-48 nets is 2057 ± 21 Elo and beats search on the
   iteration-24 nets 0.545 ± 0.016; the play page now uses the iteration-48 nets. ~~Per-move
   latency~~ Done 2026-10-27 (`experiments/search_latency.py`, one thread): 400 sims median 0.14 s
   (worst 0.26 s), 1,600 sims median 0.74 s (worst 1.0 s). Still open: more iterations (search
   was still gaining ~+30 Elo per 12 iterations); the `tessera_strong` ladder's last entry (low
   value: that anchor inflates search).
3. **Next games:** Liar's Dice started 2026-10-27 on OpenSpiel's engine (findings log, "Liar's
   Dice through OpenSpiel" and "PPO cannot find Liar's Dice's equilibrium"): solvers
   cross-checked; PPO self-play stays 0.3–0.5 exploitable against CFR+'s 0.00013. Kuhn and Leduc
   poker added the same day ("Poker through the same adapter"): same picture. 2 dice each is out of
   reach for exact methods (~2^24 bid sequences per roll). NFSP and ReBeL tried the same day
   ("NFSP and ReBeL on the OpenSpiel games"): ReBeL reaches 0.07 on Leduc (PPO 1.3, CFR+ 0.0002);
   NFSP only 0.84. Unfinished: the Liar's Dice ReBeL run was stopped at epoch 9 (0.23 at epoch 5);
   rerun `rebel game=liars_dice epochs=20` (about 10 minutes at 4 threads). Next options after
   that: Deep CFR / MCCFR toward limit Hold'em (Stage 5), or Stage 7 (The Confrontation's
   skeleton).
4. Smaller: a full-budget exploiter against the rollout agents; human play analysis (games are
   recorded to `runs/human_games/`).

**How the owner likes to work** (please keep to these):

- **Update `writeup.md` and `plan.md` after every experiment**, negative results included, then
  commit. Commit messages end with a `Co-Authored-By` line for the assistant.
- Behaviour analysis asks **conditional strategic questions** ("does it do X when Y arises"), with
  n and standard errors, and names confounds.
- Report numbers with error bars and paired deals; say plainly when a result is noise or a claim
  was wrong (two retractions are recorded below).
- Elo progress should be saved and visible (`runs/elo_ladder*.json`).
- Runs must be resumable: the owner stops sessions ("stop here for today").
- Games are original implementations of mechanical skeletons: own name and theme, no art or
  rulebook text, the designer credited as inspiration in the module docstring.

**Practicalities.**

- OpenSpiel games need the extra: `uv sync --extra openspiel` (a plain `uv sync` removes it).
- Windows, Git Bash and PowerShell; Python via `uv` (`uv sync`, `uv run pytest`, `uv run ruff check .`
  and `ruff format`, line length 100, Google docstrings). Configs are dataclasses, not argparse
  (small `key=value` command-line overrides in experiment scripts).
- 12 CPU cores, 16 GB; the venv's torch is CPU-only. Most experiments use 10 worker processes.
- Long runs: launch detached (`Start-Process cmd /c "uv run python -m experiments.<x> >> runs\<x>.log"`),
  because background shell commands are killed after about an hour. Everything long is resumable
  from `runs/`.
- Key code: `src/iirl/games/<game>/` (engines, heuristics), `src/iirl/selfplay/` (PPO, value,
  expert iteration, ReBeL), `src/iirl/agents/` (rollout, alphazero, rebel, belief, cfr),
  `src/iirl/eval/` (duplicate matches, exact best response), `experiments/` (one script per
  experiment, each with its run command in the docstring), `docs/adding-a-game.md`.

---


## The headline (as of 2026-10-16)

**The best agents are reactive policies trained by Nash-seeking PPO self-play, plus a narrow
search late in the game where an audit shows the policy is improvable.** The policy does almost
all the work; search earns its place only where it can measure move values honestly. The same
two-part method now leads on both games.

| | Ventures | Camel Stocks |
|---|:---:|:---:|
| Best agent | `diverse_sym` + opponent-robust endgame rollouts | `camel_nash` + opponent-robust rollouts |
| Search starts at | 6 cards left in the deck | 12 cards left |
| Elo, plain-heuristic anchor | **2070 ± 13** (policy alone: 2034 ± 12) | **2151 ± 24** (policy alone: 1968–1972; heuristic = 1500) |
| Elo, careful-heuristic anchor (same scale) | **1960 ± 10** (policy alone: 1929 ± 9) | not built (the plain anchor does not leak here) |
| Head to head vs the best plain policy | **0.524 ± 0.006** (+2.5 points a game) vs `diverse_sym` | **0.620 ± 0.007** (+3.8 points a game) vs `camel_nash` |
| Weakest matchup vs any heuristic style | **0.827** (careful; plain `diverse_sym`: 0.795) | **0.978** (plain; `camel_nash` alone: 0.936) |
| Held-out search opponent (PIMC) | not re-run | 0.950 (policy alone; not re-run with search) |
| Best trained exploiter's score (lower bound) | **0.493 ± 0.010** (direct, reduced budget); existing counters ≤ 0.510 | 0.509 for the policy (4 runs, 3 recipes); not run against the search |
| Thinking time per game | ~5 s (fast form) to 10–20 s | **5.5 s** (median pause 0.7 s, worst 5.3 s) |
| Play it | `uv run python -m experiments.play` (top of the Ventures list) | same page, top of the Camel Stocks list |

**The recipe**, in the order it was found:

1. **Game setup.** An information-state encoding that spells out the card relations the policy
   needs (card-relative features were Ventures' largest early gain), and a hand-tuned heuristic
   that supplies reward shaping, a league opponent and an Elo anchor.
2. **PPO self-play from scratch** (~150k games): a privileged critic (the value baseline sees the
   opponent's hand, training only), potential-based reward shaping from the heuristic, a win bonus,
   and past snapshots as opponents.
3. **Nash-seeking PPO** (400–1000 more iterations): a KL "magnet" toward a slowly refreshed copy
   of the policy, which damps the strategy cycling of plain self-play.
4. **Train against non-leaky opponents.** The single most important lesson of the project: an
   agent trained against a heuristic that hands over playable cards learned to depend on those
   gifts. Replacing it with a *careful* heuristic lifted Ventures' weakest matchup from 0.574 to
   0.716.
5. **Suit symmetry** (Ventures): a DrAC-style regularizer pulls the policy on a suit-relabelled
   position toward the policy on the original. With a four-style opponent league it took the
   weakest matchup to 0.795.
6. **Audit, then opponent-robust late-game rollouts** (new in October). First a paired-rollout
   audit finds the phase of the game where the policy's move is measurably improvable. From there
   on (Ventures: 6 cards left in the deck; Camel Stocks: 12), every first move of the turn is
   played out to the end with the policy, in 8 re-dealt hidden worlds × 8 deck orders, each under
   three models of the opponent; the policy's move is overridden only if another move is
   significantly better pooled over the models and not worse under any one of them. It adds
   ~2.5 points a game on Ventures and ~3.8 on Camel Stocks.

**How far to trust "unexploitable".** Every exploitability number is a lower bound from a trained
exploiter. On Mini-Ventures, where exact answers exist, the PPO recipe gets within 0.07 points a
game of the true equilibrium, and the exploiter recipes were calibrated: they find large
weaknesses and most moderate ones, but little of one the size of 0.07. So the table rules out
large and moderate weaknesses, not small ones. For the Ventures rollout agent the evidence is
thinner still: one reduced-budget direct exploiter plus five counters to the plain policy, all of
which do worse against it. Humans, the one opponent type never measured, can now play it.

---

## The Elo trajectory

**How the ratings are made.** Every checkpoint plays a fixed anchor on the same 1,000 paired deals
(`experiments/elo_ladder.py`), and its score converts to Elo against the anchor's fixed rating.
Ventures has two anchors on one scale:

- **The plain heuristic, fixed at 1676** (random play = 1000). It *leaks*: it hands over playable
  cards, so strong agents score 0.85–0.91 against it and their ratings bunch up near its ceiling.
- **The careful heuristic**, which never hands over a playable card. It scores 0.5175 against the
  plain one over 2,000 deals, which places it at **1688** on the same scale. Rating everyone
  against it as well separates the strong agents far better (a 0.80 score instead of 0.89).

**Ventures, the whole climb** (± one standard error; the early methods were rated before the
ladder existed, against the plain heuristic only):

| Stage | Agent | Plain anchor | Careful anchor |
|---|---|:---:|:---:|
| Baseline | Random play | 1000 | |
| Early learning | REINFORCE / PPO v1 | ~1420 / ~1340 | |
| Hand-tuned | Plain heuristic (anchor) | 1676 | ~1676 |
| Hand-tuned | Careful heuristic (anchor) | 1688 | 1688 |
| Search | PIMC (determinized UCT) | ~1700 | |
| Learned + search | Net-guided PIMC | ~1855 | |
| Search on a learned value | AlphaZero | ~1140 | |
| **PPO v2** (features, shaping, privileged critic) | `ppo_ventures_best` | 1822 ± 8 | **1650 ± 8** |
| **Nash-seeking PPO** | `nash_control` / `nash_long 600` | 1905 ± 9 / 1941 ± 10 | 1757 ± 8 / 1731 ± 7 |
| | `league_control`, `nash_g`, `rnad` (best of each) | 1973–1983 | 1748–1786 |
| **+ careful heuristic in the league** | `careful` 400 | 1982 ± 11 | **1855 ± 8** |
| + four-style league | `diverse` 300 | 1982 ± 11 | 1829 ± 8 |
| **+ suit symmetry** | `diverse_sym` 400 | 2034 ± 12 | **1929 ± 9** |
| **+ opponent-robust endgame rollouts** | best agent | **2070 ± 13** | **1960 ± 10** |

What the two scales say together:

- **PPO v2 loses to the careful heuristic** (1650 vs 1688) although it is 146 points above the
  plain one: most of its edge was the plain heuristic's gifts.
- **Nash-seeking PPO, the league and R-NaD looked like steady progress on the plain scale
  (1905 → 1983) but were nearly flat on the careful one (1716–1786).** Those months were spent
  climbing a leaky yardstick. This is the measurement behind "train against non-leaky opponents".
- **The careful-league and symmetry steps are where real strength came from:** +69 and +74 on
  the careful scale (1786 → 1855 → 1929), while the plain scale barely moved for the first.
- **The endgame rollouts add +31 to +36** on both scales, about three standard errors each.
- In total: **+310 Elo on the careful scale from PPO v2 to the best agent**, against +248 on the
  plain one, which compresses the top.

**Camel Stocks** (one anchor, the Camel Stocks heuristic at 1500):

| Checkpoint | PPO 100 | 200 | 300 | 400 | 500 | 600 | Nash 100 | 200 | 300 | 400 | **+ rollouts** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Elo | 1810 | 1834 | 1851 | 1856 | 1912 | 1889 | 1915 | 1950 | 1972 | 1968 | **2151 ± 24** |

The recipe transferred unchanged: plain PPO climbs to ~1900, the Nash-seeking phase adds ~70,
and the late-game rollouts add +179 more on this ladder. That overstates them: at a 0.977 score
against the anchor, Elo stretches small score changes into large gaps. Measured head to head,
0.620 against `camel_nash` is **+85 Elo**, the figure to trust. Its anchor is not known to leak: the agent scores *higher* against the careful Camel Stocks
heuristic (0.966) than against the plain one (0.940), the reverse of Ventures, so this ladder has
not been re-anchored.

---

## The October investigation: from ground truth to search that works (2026-10-01 → 10-14)

At the end of September the best Ventures and Camel Stocks policies had survived every exploiter
we had (best 0.512 and 0.509). The question for October was whether that meant anything, and
whether a principled search method, ReBeL, could go further. This section tells that story in
order; each step has a full entry in the findings log.

### 1. Ground truth: Mini-Ventures, solved exactly

A trained exploiter only gives a lower bound, and nobody knew how loose. So Ventures was shrunk
until it could be solved: **Mini-Ventures** (2 suits × {backing, 2, 3, 4}, 2-card hands, a
4-card deck, four turns; 2.3M states). Two exact tools were built for it: CFR+ vectorised over the
whole state graph (equilibrium to **0.002 points a game** in 100 iterations) and exact best
responses to any policy in 30 seconds.

Against that ground truth:

- **The PPO recipe works:** it reaches **0.07 points a game** of exact exploitability, 97% of
  the way from random play (2.71) to the equilibrium.
- **The exploiter yardstick was too gentle.** The standard exploiter recipe found 61% of a large
  weakness but only 1% of a moderate one. More aggressive recipes found 55–92%.
- Re-measured with those recipes, the real agents still showed nothing more (Ventures best 0.500
  and 0.498, Camel Stocks 0.500 and 0.505). The September lower bounds stand, now with a known
  meaning: no large or moderate weakness.

A first version of this ground truth reported that "the yardstick finds ~4%" of the true
exploitability. That was an artefact (a degenerate start cost and a starved PPO) and was
retracted.

### 2. ReBeL works where values can be exact

ReBeL searches over **public belief states**: the public position plus, for each player, a
probability for every possible hidden hand. Each turn is a subgame solved by CFR; where it is
cut off, a value network gives every hand's value as a function of the beliefs; the network is
trained on solved subgames, from the last turn up.

On Mini-Ventures, measured exactly: **0.002 points a game**, level with the full-game equilibrium
and ~30× closer than PPO, in 21 minutes of training. Three things had to be right, each found by
checking against ground truth:

1. **Play a random CFR iteration, not the average.** Re-solving later turns from the *average*
   strategy's beliefs stalls (0.085 on Kuhn poker); ReBeL's rule of stopping at a random iteration
   and re-solving from *its* beliefs reaches 0.016.
2. **Beliefs must exist after a deviation.** The first run scored 1.30: near-pure CFR iterates
   left most public states with no modelled reach, so an opponent's deviation met a re-solve with
   no beliefs. Assuming a 5% trembling hand when forming beliefs fixed it (→ 0.0048).
3. **The last turn is a plain best response** (one pass, exact, 60× cheaper).

Then the three approximations a large game forces were priced on the same ground truth: beliefs
as per-card probabilities instead of per-hand (0.0037), a network that values one hand at a time
(0.0028), training on 256 sampled public states per pass instead of all of them (0.0035). None
costs anything measurable; sampling needs data (0.093 at 2.5k examples, 0.006 at 5k).

### 3. ReBeL on full Ventures: every part works, and the agent loses

Everything was rebuilt for a game that cannot be enumerated: a one-position subgame solver over
candidate hands (matches the exact solver to 1e-9 where they can be compared), public belief
tracking with 48 candidate hands per player (overlap with the true hand 0.354, vs 0.296 from the
cards alone), and a belief-conditioned value network trained on 24,000 `diverse_sym` self-play
games (held-out R² 0.46, RMSE 32 points; belief inputs add nothing).

Against `diverse_sym`: **0.050 plain** (−110 points a game), 0.41–0.44 when anchored to the
policy. The search machinery was not at fault: the pipeline is self-consistent (bias +1.9 points,
correlation 0.89). The value network is ~11 points off per evaluation, the real gaps between
moves are a point or two, and a search that maximises over ~16 noisy values picks the noise. It
claimed +9.6 points a turn over the policy; plain-search games ended with routes like
[1, 2, 0, 1, 7] against the policy's [5, 6, 7, 6, 3].

This was the fifth search on Ventures to fail for this reason (after AlphaZero's value, belief
search, expert iteration and endgame CFR).

### 4. Asking where search can work at all: an audit by paired rollouts

Rather than build a sixth search, we measured move values directly. At 572 positions from
`diverse_sym` self-play, every first move was played out to the end by the policy in the same 8
hidden worlds × 8 deck orders, so the luck cancels in the differences. What it showed:

- **Our search's losses have a shape.** It loses 4–6 points a turn in every phase, worst when it
  swaps to a discard the opponent can play at once (−10.5 ± 0.8, n = 125): the same leak the
  agents themselves once had to unlearn.
- **The policy is improvable only in the endgame.** With 6 or fewer cards left, a better first
  move is worth at least +1.05 ± 0.37 points a turn; earlier, no improvement is detectable.
- **Luck is the obstacle.** One paired rollout's difference between two close moves has a spread
  of 22–38 points; resolving a 1-point gap takes on the order of 900 rollouts per move.
- **Hidden information matters, but less than luck.** The opponent's hand shifts a move's
  advantage by 2–4 points: larger than the gaps between close moves, about 8× smaller than the
  luck. That is why a value network trained on outcomes cannot see it.

### 5. Endgame rollouts: the first search to beat the policy, made robust

Acting on the audit: play the policy, and from 6 cards left override it only where paired
rollouts show a significantly better move.

- **With the policy as the only opponent model:** 0.526 ± 0.006 head to head (0.538 from 12
  cards), as the audit predicted (~1 point a turn over the last ~3 turns). But the gain did not
  transfer: level against four heuristic styles and **worse against the tight one** (0.904 vs
  0.934). The rollouts had been forecasting the opponent's endgame as the policy's.
- **With three opponent models** (the policy, the careful and the tight heuristic; override only
  if not worse under any): 0.524 ± 0.006 head to head, and **better than the plain policy against
  all five styles**, including the three it does not model. Weakest matchup 0.795 → 0.827.

### 6. Is it exploitable? Two checks, both clean

- **Existing counters lose ground.** The five policies trained or selected to beat `diverse_sym`
  all do worse against the rollout agent, by 1.4–2.0 points a game; the best goes from +1.7 to
  −0.1 (0.510 ± 0.014).
- **A direct exploiter finds nothing.** Trained against the rollout agent itself (warm-started
  from the best exploiter of `diverse_sym`, 60 iterations): 0.493 ± 0.010 on 1,000 fresh deals;
  no snapshot improved on its starting point. A reduced budget, so a weak attack.

### 7. The gain cannot be moved into the policy; it can be made cheaper

- **Distillation fails.** A copy of the policy fine-tuned on 4,715 logged overrides either
  reproduces one in seven on unseen games and plays worse (0.484), or reproduces ~2% and plays
  the same. An override is the outcome of ~190 rollouts at one exact position, and 4,000
  examples of that behave like separate facts, not a rule.
- **Speed.** A per-search cache of the policy's moves (identical decisions) and dropping clearly
  weak moves after the first quarter of rollouts make it ~3× faster: 0.517 ± 0.003 head to head
  in this fast form, ~5 s a game.

---

## What we learned (general lessons)

1. **Get ground truth before trusting a yardstick.** A small exactly-solvable version of the game
   calibrated both the training recipe and the exploiter, and caught two methodological errors
   (one of them our own retracted claim).
2. **Leaky opponents teach leaks.** The biggest single gain in robustness came from removing an
   opponent that handed over playable cards, and the biggest single loss of the October search
   came from the search re-learning that leak through a value network.
3. **Search needs values more accurate than the gaps between moves.** In a high-luck game a value
   network trained on outcomes is ~10 points off when the decisions differ by ~1. Every search
   built on such a value lost to the plain policy; the one that worked uses no learned value.
4. **Measure where the decisions are close before searching.** The paired-rollout audit found the
   one phase (the endgame) where search could help, and predicted the size of the gain.
5. **Model the opponent more than once.** A search that assumes one opponent wins against that
   opponent and can lose to others; requiring the override to hold under several models made the
   gain general.
6. **Pick the best of many noisy estimates and you pick noise.** It sank the ReBeL search, and it
   produced misleading negative numbers in our own first audit report until selection-free
   estimators replaced them.
7. **Exact methods are the right tool on small games, not big ones.** ReBeL is exact and fast on
   Mini-Ventures and loses on Ventures; what changed is not the algorithm but whether its value
   function can be learned.

---

## Where we stand (2026-10-16)

**Tried, and the verdict** (details in the findings log below):

| Approach | Verdict |
|---|---|
| Heuristics, PIMC search | Solid baselines; the heuristics also taught us about leaks |
| REINFORCE, PPO v1 | Beat random, plateau below the heuristic |
| AlphaZero (learned value + search) | Fails: the deal's luck swamps the value net |
| Net-guided PIMC | Beat the old heuristic (~1855), before today's features; not re-run since |
| Expert iteration (search distilled into the policy) | Fragile, no gain |
| **PPO v2 → Nash-seeking PPO** | **The winning recipe, on both games** |
| Belief-weighted search on top (particle filter) | Better beliefs, no gain in play at 64 sims × 4 worlds |
| Exploiters as fixed league opponents | Worse on both exploitability and Elo |
| R-NaD-style moving-average magnet | No gain over the plain magnet |
| **Careful heuristic in the league** | **Big gain: weakest matchup 0.574 → 0.716** |
| Four-style league alone | Slightly worse than careful-only (0.681) |
| **+ suit symmetry (DrAC)** | **Weakest matchup 0.795; beats `careful` 0.569** |
| PSRO (4 rounds) | Found nothing beyond ~+1.5 points a game; confirms `diverse_sym` |
| Endgame CFR (depth-limited MCCFR) | Never beats the policy (best 0.452); its subgame assumes the opponent sees our hand |
| NFSP (PPO best response) | Equally unexploitable as Nash PPO on Camel Stocks, but weaker (0.456 head to head) |
| Tabular / vectorized CFR+ | Exact on toy games and on Mini-Ventures (0.002 points a game); far too big for the real games |
| **ReBeL (CFR over public belief states + value nets)** | **Reaches equilibrium where values can be exact: 0.002 on Mini-Ventures vs PPO's 0.07.** On Ventures it loses to its own blueprint (0.05; 0.41–0.44 anchored): the learned value is too noisy to search on |
| Late-game paired rollouts on top of the policy | Beats the policy head to head (0.526 from 6 cards left, 0.538 from 12); with one opponent model it does not transfer |
| **+ opponent-robust rollouts (3 opponent models)** | **Best Ventures agent: 0.524 head to head, better than the policy against all five styles (weakest 0.827); exploiters find nothing** |
| Distilling those rollouts' moves into the policy | Fails: reproduces 2–14% of the overrides, plays level or worse |
| **Audit + robust rollouts on Camel Stocks** | **Transfers: 0.620 head to head vs `camel_nash`, better against both heuristic styles; search from 12 cards, 5.5 s a game** |

**What's left, in order of value:**

1. **Azul (next).** A perfect-information game apart from the tile draws: the setting where a
   learned value *should* support search, unlike Ventures. The plan is the recipe first, then the
   audit, then whichever search the audit supports.
2. **Human play.** The strongest agent of each game is now the first opponent in the play page;
   games are recorded to `runs/human_games/` for analysis.
3. **Liar's Dice and a small poker**, where ReBeL's public-belief search should suit far better
   than on Ventures: small private states, outcomes that follow from decisions.
4. **A full-budget exploiter** against the rollout agents with the stronger recipes (~10 hours per
   run at current speed), for a tighter exploitability bound; none has been run against the Camel
   Stocks search yet.
5. **Earlier rollouts** only with faster code: on Camel Stocks starting at 25 cards costs 12× the
   thinking time (68 s a game, pauses up to 20 s) for a gain the audit puts at ~0.3 points a turn.
6. **A fast engine and batched search (Phase D)**, if any of the above needs the speed.

## Why each method lands where it does

**PIMC (Perfect-Information Monte Carlo).** Determinized UCT: sample worlds consistent with the info
state, search each with an honest leaf estimate (the deck-clock potential), pool the root visits.
Explicit lookahead → beats the heuristic. The strong baseline everything is measured against.

**Heuristic.** Encodes two real principles: wager discipline (don't open a route you can't drag past
the −20 start cost) and the deck as a shared clock (only ~half the remaining deck draws are yours, so
don't start what you can't finish). Crushes random 100–0; a genuinely hard target.

**REINFORCE / PPO (model-free RL).** Self-play policy gradient. The first versions climbed from
~random to ~1340–1430 and crushed random, but plateaued at 0–5% vs the heuristic. That was not a
limit of reactive policies — PPO v2 (see the findings log) reaches ~1800 and beats both the heuristic
and PIMC. What held v1 back: raw features the MLP had to decode (is this card playable? how far above
the route?), a single terminal reward credited to ~75 decisions, and only 1.3k training games.
Model-free RL is the *tractable* family here because it only samples episodes — horizon and infoset
count don't blow up, and the value net is just a variance-reduction baseline.

**AlphaZero (net value guiding search).** The self-improvement loop (`MCTS(net) → targets → train →
stronger search`) needs a **learnable value**. In Go/chess (perfect info, near-deterministic,
low-variance) position strongly predicts outcome, so value is learnable and the loop closes. In
Ventures the outcome is dominated by the random deal → `sign(score)` is ≈ noise → the value net
overfits noise → using it as the leaf evaluation **poisons** the search. Result: ~1140, *worse* than
plain rollout/heuristic-leaf PIMC. Verified across four runs; more compute doesn't fix an unlearnable
target. This empirically re-derives the field's paradigm split: AlphaZero for low-variance
perfect-info; regret minimization / search for imperfect-info.

**CFR (Counterfactual Regret Minimization).** The principled imperfect-info method — it converges to
Nash by minimizing counterfactual regret. Validated on the tiny `toy_showdown` (mini-Kuhn):
exploitability → **0.0015**, reproducing the exact Kuhn equilibrium (bluff the low card 0.22, bet the
King as opener 0.67 = 3× the bluff rate, bluff-catch with the middle card). **But it does not scale to
Ventures:** tabular/MC CFR needs a short horizon — Ventures' ~75 decisions/player make external
sampling branch to ~14⁷⁵, and outcome sampling needs astronomically many iterations. CFR is for
poker-sized games; large ones need Deep CFR (also horizon-limited) or other approaches.

**Net-guided PIMC (the synthesis).** The answer to "beat the heuristic *with learning*." Keep PIMC's
honest deck-clock leaf value (where AlphaZero's learned value failed) and use the network **only** for
PUCT priors, trained on the search's own visit distributions. Untrained (uniform priors) it already
matches PIMC; trained priors lift it from 25% → ~47% vs the heuristic at a small search budget, and at
PIMC-level search (256 sims) it **decisively beats the heuristic: 29-1-10 (74%), Elo ~1855** — the
strongest agent built, above both the heuristic and plain PIMC.

---

## Findings log

### NFSP and ReBeL on the OpenSpiel games: ReBeL closes most of PPO's gap, NFSP a third of it (2026-10-27)

Same yardstick as below: exact exploitability (NashConv / 2) of the policy each method plays, from
`experiments/spiel_games.py` (`nfsp` and `rebel` stages; 4 workers / 4 torch threads). One seed per
row, so differences under about 2× between rows are not established.

| Method | Leduc (chips; uniform 2.37) | Liar's Dice (±1; uniform 0.78) | Kuhn |
|---|---|---|---|
| PPO, best variant | 1.28–1.38 | 0.31–0.52 | — |
| NFSP from the PPO phase, 400 iterations (η 0.1), sampled | 1.87 → **0.84** (iteration 275) → 1.08 (400) | 0.50–0.59 (400: 0.59) | — |
| ReBeL, 20 epochs, 100 CFR iterations per subgame | 0.099 | 0.23 at epoch 5 (run stopped at 9) | 0.015–0.019 |
| ReBeL, 60 epochs, 200 CFR iterations | 0.63, 0.20, 0.15, **0.070** at epochs 5–20, then 0.069–0.092 to 60 | — | — |
| CFR+, 1,000 iterations (full game) | 0.00023 | 0.00013 | 0.00009 |

- **ReBeL is the only learning method within an order of magnitude of the solver**: on Leduc about
  15–20× less exploitable than PPO, in 1–3 minutes of training. It plateaus at 0.07–0.09 from
  epoch 20 on: tripling the epochs and doubling the subgame CFR iterations did not move it, so the
  floor is the value networks' error at the cuts (losses ~1e-5 on Liar's Dice, but a small error
  compounds across boundaries), not the CFR inside each subgame.
- **NFSP is not monotone**: its average policy on Leduc fell to 0.84 then rose to 1.08; on Liar's
  Dice it never beat PPO. This matches the Camel Stocks result (NFSP ≈ Nash PPO at equal budget).
- Boundaries (subgame start depths, no chance node there): Kuhn (2, 3, 5), Leduc (2, 7, 11), Liar's
  Dice (2, 5, 8, 15). Public states per boundary on Liar's Dice: 1, 220, 924. Curves in
  `runs/<game>/rebel*.jsonl`; NFSP's in `runs/<game>/ppo_exact.json`.

### Poker through the same adapter: Kuhn and Leduc, and PPO is no safer than random on Leduc (2026-10-27)

`kuhn_poker` and `leduc_poker` are registered through `games/spiel.py` (one line each), and the
experiment script is now `experiments/spiel_games.py` (`game=` on every stage).

**Solvers** (`crosscheck game=...`, `runs/<game>/crosscheck.jsonl`): exploitability evaluators agree
to 6 decimals on every policy again. Kuhn: CFR+ 0.00009 after 1,000 iterations, first player's
value −0.05559 (the known −1/18 = −0.05556). Leduc (9,457 states, 936 information sets): CFR+
0.00023 after 1,000 iterations in 0.9 s, first player's value −0.0857. On Leduc our alternating
CFR+ matches OpenSpiel's exactly to iteration 100, then differs slightly (0.000225 vs 0.000257 at
1,000; identical throughout on Liar's Dice and Kuhn), presumably an implementation detail; both
converge.

**PPO on Leduc** (payoffs in chips, up to ±13; uniform random is 2.37 exploitable):

| Agent | Exact exploitability (greedy / sampled) |
|---|---|
| PPO from scratch, 25 → 200 iterations | 1.38–3.42 / 1.15–3.29 (200: 2.18 / 2.04) |
| + Nash phase (magnet 0.1), 200 iterations | 1.46–2.16 / 1.27–1.88 (200: 1.46 / 1.28) |
| + strong magnet (1.0, lr 3e-5, reward scale 5), 400 iterations | 1.69–2.49 / 1.38–2.10 (400: 1.78 / 1.38) |
| CFR+, 1,000 iterations | **0.00023** |

- PPO is barely safer than uniform random (2.37) and at one checkpoint worse (3.4). Win rate
  against random is a misleading yardstick in poker: 0.41–0.52, with a positive average margin
  (+0.5 to +0.7 chips), because the agent folds many small pots.
- The strong magnet again gives a slow, steady decline (sampled 2.03 → 1.38 over 400
  iterations), as on Liar's Dice; nothing comes within three orders of magnitude of CFR+.
- Confound noted and partly addressed: the first two runs used reward scale 1 on ±13 payoffs; the
  magnet run used 5 and is no closer.

### PPO cannot find Liar's Dice's equilibrium; CFR+ does in seconds (2026-10-27)

The plan's headline experiment: the repo's PPO recipe trained by self-play on Liar's Dice (1 die
each), every checkpoint's **exact** exploitability measured (`experiments/spiel_games.py ppo` and
`magnet`; `runs/liars_dice/ppo_exact.json`). Payoffs are +1 / −1, so exploitability 0.4 means a
perfect exploiter wins about 70% of games.

| Agent | Exact exploitability (greedy / sampled) | vs random |
|---|---|---|
| uniform random | 0.78 | — |
| PPO from scratch, 25 → 200 iterations | 0.42–0.72 / 0.42–0.58 (200: 0.56 / 0.50) | 0.83 |
| + Nash phase as on Mini-Ventures (magnet 0.1, lr 1e-4), 200 its | 0.42–0.67 / 0.38–0.58 (200: 0.47 / 0.43) | 0.83 |
| + strong magnet (1.0, lr 3e-5), 400 its | 0.33–0.64 / 0.34–0.51 (400: 0.42 / 0.35) | 0.82 |
| + magnet 0.3, lr 1e-4, 400 its | 0.36–0.61 / 0.31–0.52 (400: 0.53 / 0.52) | 0.83 |
| CFR+ (alternating), 1,000 iterations, 31 s | **0.00013** | — |

- **PPO learns to play** (0.83 against random after 25 iterations) **but not to be safe.** Its
  exploitability swings from checkpoint to checkpoint instead of falling: the strategy cycling
  that motivates regret minimisation, watched happening.
- **A strong magnet helps a little and slowly** (sampled 0.50 → 0.35 over 400 iterations); a
  middling one does not. None comes within 2,000× of CFR+.
- **Deploying the greedy policy costs more here than on the card games.** Liar's Dice needs mixed
  play (bluffing bids, calling at the right frequency), and a deterministic policy is exploitable
  by construction; the sampled policy is usually less exploitable.
- **Contrast with Mini-Ventures,** where the same recipe got within 0.07 points of equilibrium
  (about 1% of a typical margin). There, most decisions have a clearly best move; Liar's Dice's
  value lies in the mixing, which policy gradients chase in circles.

Next on this game: 2 dice each (graph CFR+ at a bigger scale, then MCCFR), and whether ReBeL or
NFSP closes the gap PPO leaves.

### Liar's Dice through OpenSpiel: our solvers match DeepMind's exactly (2026-10-27)

Rather than build a fourth engine, Liar's Dice uses OpenSpiel's (`open_spiel` 2.0.2 has a Windows
wheel; the `[openspiel]` extra). `games/spiel.py` puts any two-player zero-sum OpenSpiel game
behind our `Game` interface: immutable states identified by their action history, OpenSpiel's
information-state strings and tensors. Registered as `liars_dice` (1 die each, 6 faces, highest
face wild: 294,883 states, 24,576 information sets). It passes the conformance suite, and
`tests/test_spiel.py` checks it move for move against OpenSpiel on random games.

**The cross-check** (`experiments/spiel_games.py crosscheck`, `runs/liars_dice/crosscheck.jsonl`):
our graph CFR+ and OpenSpiel's C++ CFR+ for the same iterations, each average strategy scored by
both exploitability evaluators (ours, `eval/exact_br.py`; OpenSpiel's `pyspiel.exploitability`).

| Iterations | Ours, simultaneous | Ours, alternating | OpenSpiel CFR+ |
|---|---|---|---|
| 10 | 0.3500 | 0.1416 | 0.1416 |
| 100 | 0.0370 | 0.0049 | 0.0049 |
| 1,000 | 0.0034 | **0.00013** | 0.00013 |
| time for 1,000 | 16 s | 31 s | 116 s |

- **The evaluators agree to 6 decimals on every policy** (uniform: 0.780744 by both), so the adapter
  and our exact best response are right.
- **Our CFR+ with alternating updates reproduces OpenSpiel's CFR+ to 6 decimals at every
  checkpoint**, about 4× faster (vectorized numpy over the state graph vs C++ tree walks).
- **Simultaneous updates, which `GraphCFR` used until now, converge ~10–25× slower per
  iteration.** `GraphCFR(alternating=True)` is now available; the default stays simultaneous so
  the Mini-Ventures results reproduce. The Mini-Ventures equilibrium (0.0022 after up to 3,000
  simultaneous iterations) was correct, just slower to reach than it needed to be.
- **The first bidder's equilibrium value is −0.02713** (bracketed within ±0.00013 by the best
  responses), equal to −7/258 to five decimals: going first is a small disadvantage.

### Tessera: 24 more loop iterations help the search, not the raw policy's score (2026-10-26)

The AlphaZero loop (`experiments/alphago_tessera.py exit iterations=48`) ran from iteration 24 to
48, about 5 minutes an iteration, resumed from the saved nets (snapshots now kept every 12
iterations as `runs/tessera_ag/exit_{prior,value}_<i>.pt`).

**The loop's own curve looked flat.** The raw policy against the starting policy (300 deals):
0.61 at iteration 24, then 0.55–0.61 through iteration 48 (0.57 at 48; SE ≈ 0.02). Search at 100
simulations against the heuristic (200 deals): 0.83–0.87 throughout. Read alone, these say the
loop had plateaued.

**Direct and ladder measurements say otherwise.** Search at 400 simulations, head to head on 400
paired deals:

| Nets | vs search on the iteration-24 nets |
|---|---|
| iteration 36 | 0.547 ± 0.017, +0.96 ± 0.36 points |
| iteration 48 | **0.545 ± 0.016, +1.41 ± 0.35 points** |

The 1,000-deal Tessera Elo ladder (`experiments/elo_ladder.py tessera`,
`runs/elo_ladder_tessera.json`, heuristic = 1500):

| Agent | Elo | Score vs heuristic | Margin |
|---|---|---|---|
| `tessera_ppo` 100 → 600 | 1358 → 1447 | 0.31 → 0.42 | −9.3 → −3.1 |
| `tessera_nash` 100 → 400 | 1522 → 1627 | 0.53 → 0.68 | +1.1 → +7.1 |
| `tessera_nash2` 200 → 1200 | 1649 → 1718 ± 10 | 0.70 → 0.78 | +8.3 → +12.6 |
| loop policy, iteration 24 / 36 / 48 | 1740 / 1742 / 1754 (± 10) | 0.80 / 0.80 / 0.81 | +13.6 / +13.9 / +13.9 |
| search 400, starting nets | 1942 ± 16 | 0.927 | +17.4 |
| search 400, iteration 24 / 36 / 48 | 1993 ± 17 / 2028 ± 18 / **2057 ± 21** | 0.945 / 0.954 / 0.961 | +18.4 / +19.2 / +19.9 |

So the loop's later iterations improve what search does with the nets (+64 Elo from 24 to 48,
matching the head-to-head) while the raw policy gains little (+14 ± 14). The in-loop yardstick
(raw policy vs the starting policy) is the wrong one for a search agent. The play page now uses
the iteration-48 nets at 400 simulations.

**A second anchor does not help search agents (non-transitivity).** `tessera_strong` rates the
strong end against the `tessera_nash2` policy at its 1718. The policies agree roughly across
anchors (iteration 24 / 36 / 48: 1768 / 1785 / 1791 ± 8, about 30 above the heuristic scale), but
search on the starting nets comes out at **2250 ± 19 against 1942 on the heuristic scale**, and on
the iteration-24 nets 2310 ± 21. Search built on a policy's priors beats that same policy far more
than its strength elsewhere predicts, so the policy anchor inflates search agents by ~300 Elo. For
comparing search agents, use direct search-vs-search matches (`match vs=exit_<i>`); for one scale
across everything, the heuristic ladder (compressed near the ceiling, but transitive enough).

**Latency** (`experiments/search_latency.py`, 6 games each vs the policy, one thread, 2026-10-27):
400 simulations take 0.14 s a move (median; 90th percentile 0.24 s, worst 0.26 s), 3.6 s of
thinking a game; 1,600 take 0.74 s (0.94 s, worst 1.0 s), 16 s a game. About half of a
simulation's ~0.35 ms is the two network calls (0.08 ms each at batch 1); the rest is the Python
engine and tree. A GPU would not speed this up: batch-1 calls are dominated by launch and transfer
overhead, and the batched cost on the CPU is already 7 µs a position.

Human-play analysis (thread 4) has one recorded game so far (Ventures, 2026-09-29): nothing to
analyse yet.

### The play page follows the Ladder design system on every board (2026-10-26)

Before this change, the Ventures and Camel Stocks boards predated the design system. Only `on-accent`
had been applied. Now all three boards in `viz/play.html` follow it:

- **The system's tokens are CSS variables** (spacing, radius, type families, `piece-ink`,
  `piece-paper`, the card-back stripes, `shadow-raised`, layout widths), and the CSS reads them.
- **One label rule for every piece.** `pieceClass(hex)` prints a piece's label in whichever of
  `piece-ink` and `piece-paper` contrasts more, and marks a piece where neither reaches 4.5:1 as
  `no-small` (its small name is hidden; ranks and glyphs stay). It reproduces every token note in the
  system: ink on summit, gold, silver, spice and camel; white on the rest; orchard the only
  `no-small` piece (4.1:1). Before, labels were white with a text shadow everywhere, which fell to
  2.4:1 on gold and 2.6:1 on camel cards and top coins. Tessera tiles use the same function.
- **No piece colour on interface elements or small text.** Route names and token-pile names are now
  in ink, not the suit colour (summit orange on white was about 3:1). Camel Stocks move buttons lose
  their coloured side bar; the good's glyph names it. Token piles take the route columns' coloured
  top edge instead. Lighter coin values now reach 5.5:1 or better in both themes.

Checked by rendering real mid-game states from the engine in headless Edge, in both themes, at
1200px and in a 390px frame (no horizontal scroll). One known gap: the card gradient's lightest
corner drops white labels to 3.7–4.5:1 there. The name sits lower, on the full colour.

### Tessera: the AlphaGo recipe works where value search never did (2026-10-19)

Tessera hides nothing but the order of the bag, so it is the first game where search on a learned
value should work. It does, by a wide margin. (`experiments/alphago_tessera.py`.)

**The policy.** The recipe's second league phase (another 1,200 iterations, with the heuristic and
past selves as opponents and a lower learning rate) took the policy to **0.79 against the heuristic,
+13.3 points, Elo ~1734**, flat over its last 200 iterations; exploiters find almost nothing
(0.505, max of two). On a perfect-information game the "Nash-seeking" magnet has no mixed
equilibrium to aim for; the gain more likely came from the league and the lower learning rate
(not separated).

**The value net.** Trained on 40,000 self-play games of that policy (480,000 positions; target
win/loss blended with the margin, weight 0.5, scale 20). Held-out Brier **0.193**, better than the
heuristic's value (0.209) and a coin flip (0.243): unlike Ventures, the value is learnable.

**Search.** PUCT with the policy's priors and the value net's leaves (`SplitEvaluator`), one world
(nothing to resample), 200 paired deals per match:

| Simulations per move | vs the policy alone | vs the heuristic | Elo (heuristic = 1500) | Thinking per game |
|:---:|:---:|:---:|:---:|:---:|
| 0 (the policy) | 0.500 | 0.79 | ~1734 | instant |
| 100 | 0.551 ± 0.017 (+1.9) | 0.810 ± 0.019 (+13.1) | 1752 | ~1 s |
| 400 | **0.941 ± 0.011** (+12.4) | **0.938 ± 0.012** (+16.6) | 1970 | ~9 s |
| 1,600 | **0.999 ± 0.001** (+20.5) | **0.969 ± 0.009** (+22.5) | 2097 | ~36 s |

- Strength rises steadily with the search budget, the signature of a sound value: at 1,600
  simulations the search wins 399.5 of 400 games against the policy it searches from.
- **Why here and not on Ventures:** on Ventures the value net was ~11 points off when moves
  differed by ~1, because of hidden hands and the deal. Tessera's outcome follows from the
  position, so the value can rank leaves, and deeper search compounds it.
- Thinking times are worker time per game across 10 parallel processes; per-move latency not yet
  measured on an idle machine.

**The AlphaZero loop** (expert iteration, 24 iterations of 400 search self-play games at 200
simulations: the policy trained toward the search's improved choices, the value toward outcomes
mixed with the search's estimate; `alphago_tessera exit`, curve in `runs/tessera_ag/exit.jsonl`):

| Iteration | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| New policy alone vs the starting policy (300 deals) | 0.496 | 0.516 | 0.525 | 0.543 | 0.565 | 0.587 | 0.585 | **0.612** (+3.6) |
| Search, 100 sims, new nets vs the heuristic (200 deals) | 0.843 | 0.828 | 0.840 | 0.844 | 0.841 | 0.878 | 0.854 | 0.826 |

- **The policy improved:** distilling the search raised the raw policy steadily, to 0.612 against
  where it started (+3.6 points a game), with no search at play time.
- **Search at 100 simulations did not:** 0.83–0.88 against the heuristic throughout (before the
  loop 0.810), within about two standard errors of flat.
- **At larger budgets the loop's nets are clearly stronger** (2026-10-24, 200 paired deals each):

| Search | vs the original policy | vs the heuristic | Elo (heuristic = 1500) |
|---|:---:|:---:|:---:|
| 400 sims, original nets | 0.941 ± 0.011 (+12.4) | 0.938 ± 0.012 (+16.6) | ~1970 |
| 400 sims, **loop nets** | **0.970 ± 0.008** (+14.4) | **0.953 ± 0.011** (+18.8) | ~2021 |
| 1,600 sims, original nets | 0.999 ± 0.001 (+20.5) | 0.969 ± 0.009 (+22.5) | ~2097 |
| 1,600 sims, **loop nets** | not run | **0.986 ± 0.006** (+25.3) | ~2242 |

  About +2 to +3 points a game at equal budget. The loop's nets (`runs/tessera_ag/exit_prior.pt`,
  `exit_value.pt`) are now Tessera's play-page opponent at 400 simulations. The board itself is
  now a real renderer (displays, centre, pattern rows, wall, floor; pick a colour, then a row),
  built from the Ladder design system.

### Tessera: the recipe needs its league to beat the heuristic, and is still climbing (2026-10-19)

Tessera is the third game: an original game on Azul's mechanics (`games/tessera`), perfect
information apart from the bag. Engine, 13 rules tests, a greedy heuristic (beats random in 90%+
of games; heuristic self-play scores ~55 each over 5 rounds, like the original). Then the recipe,
unchanged (`experiments/train_tessera.py`). In-training evaluation, 400 games vs the heuristic
every 10 iterations (Elo anchored on the heuristic at 1500, ±35 at this sample size):

| Phase | Iterations | Score vs heuristic | Margin | Elo |
|---|:---:|:---:|:---:|:---:|
| PPO from scratch | 100 / 300 / 600 | 0.30 / 0.38 / 0.40 | −9.7 / −4.8 / −3.8 | 1366 / 1417 / 1439 |
| Nash-seeking PPO (heuristic in the league) | 100 / 200 / 300 / 400 | 0.46 / 0.56 / 0.65 / **0.69** | −0.3 / +3.4 / +6.7 / **+8.2** | 1484 / 1557 / 1608 / **1652** |

- **Unlike both card games, plain PPO self-play never beat the heuristic** (0.40 after 154k
  games, flattening). It took the Nash phase, whose league includes the heuristic as a permanent
  opponent, to pass it, and that phase was **still climbing at its last checkpoint** (+44 Elo
  over the final 100 iterations).
- **Exploitability 0.543** (max of two exploiter runs, 0.535 and 0.543; margins +1.2 to +1.4 points
  a game). Higher than either card game (0.509–0.512), consistent with a policy that has not
  converged.
- Likely reasons, untested: Tessera's margins are small (±10 points) next to the +20 win bonus,
  so the reward is mostly the sign; and drafting rewards planning a round ahead, which a reactive
  policy learns slowly. Both point at the next steps rather than at the recipe failing.

**Next:** continue the Nash phase until it flattens, rate it on a proper 1,000-deal Elo ladder,
then the paired-rollout audit and the comparison the plan was written for: AlphaZero-style search
with a learned value, which should finally work on a game without hidden hands.

### The search approach transfers to Camel Stocks (2026-10-14 → 10-16)

`experiments/search_audit.py` (a game-agnostic version of the Ventures audit, plus match and
panel stages), `experiments/rollout_latency.py`; the rollout agent now handles chance events
that are not deck draws (Camel Stocks' bonus tokens) and a per-game start-of-turn rule.

**1. The audit first.** 584 start-of-turn positions from `camel_nash` self-play, every first move
played out by the policy in the same 8 hidden worlds × 8 deck orders:

| Phase (cards in deck) | n | Legal moves | Policy's move is best | Better move worth: upper | Lower | Rollouts to resolve 1 point |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Early (> 25) | 246 | 7.7 | 0.47 | +0.89 ± 0.08 | −0.35 ± 0.15 | 930 |
| Middle (13–25) | 260 | 7.0 | 0.39 | +1.21 ± 0.11 | **+0.34 ± 0.15** | 537 |
| Late (7–12) | 69 | 6.8 | 0.35 | +2.02 ± 0.32 | **+1.74 ± 0.36** | 160 |
| Endgame (≤ 6) | 9 | 8.8 | 0.33 | +3.28 ± 1.16 | +3.25 ± 1.17 | 21 |

- **Improvable from the middle game on, and by ~2 points a turn once 12 or fewer cards are left**
  (Ventures: only the last 6 cards, ~1 point a turn).
- **Few games reach a near-empty deck** (9 of 584 positions): Camel Stocks usually ends when three
  goods run out. A Ventures-style threshold of 6 cards would almost never fire; 12 is the natural
  start.
- **Hidden information matters little** (a close move's advantage varies by 0.6–1.6 points with
  the opponent's hand, against 2–4 on Ventures), and **luck shrinks fast** as the deck empties
  (15 points per rollout early, 2 at the end).

**2. The agent.** `camel_nash` plus opponent-robust rollouts from 12 cards left, with the policy,
the Camel Stocks heuristic and the careful Camel Stocks heuristic as the three opponent models.

| Opponent (paired deals) | Rollout agent | `camel_nash` alone |
|---|:---:|:---:|
| `camel_nash` (1,000) | **0.620 ± 0.007** (+3.80 ± 0.11) | 0.500 |
| Camel Stocks heuristic (500) | **0.978 ± 0.004** (+24.0) | 0.936 ± 0.007 (+20.0) |
| Careful heuristic (500) | **0.988 ± 0.003** (+27.8) | 0.966 ± 0.006 (+23.7) |
| Elo, heuristic = 1500 (1,000) | **2151 ± 24** | 1968–1972 |

- **A bigger gain than on Ventures** (+3.8 points a game against +2.5), as the audit predicted:
  the policy is improvable earlier and by more, and there is less luck to average away.
- **It transfers across opponents:** about +4 points a game against both heuristic styles too.
  Unlike the first Ventures version, no opponent got worse.
- Both heuristic styles are among the rollouts' opponent models, so (unlike Ventures, which had
  three unmodelled styles) there is no held-out style here; the gain against `camel_nash`, which
  is also modelled, is the head-to-head figure.

**3. What it costs, and why not search earlier.** Per-decision latency over 20 games on an idle
machine, single process (`experiments/rollout_latency.py`):

| Search from | Searched turns a game | Median pause | 90th percentile | Worst pause | Thinking per game |
|---|:---:|:---:|:---:|:---:|:---:|
| 12 cards | 5.3 | 0.66 s | 2.8 s | 5.3 s | **5.5 s** |
| 25 cards | 14.3 | 3.5 s | 10.5 s | 20.5 s | 67.6 s |

(The policy's own move takes 0.2 ms.) Starting at 25 cards costs 12× the thinking time for the
middle-game gain the audit puts at ~0.3 points a turn. A 1,000-deal match from 25 cards was
stopped after five hours without a result; a smaller one can settle it if faster code arrives.

**Not yet measured:** a trained exploiter against the Camel Stocks rollout agent, and PIMC as a
held-out opponent for it.

### Where search suits Ventures: an audit by paired rollouts, and the first search to beat the policy (2026-10-02)

`experiments/ventures_search_audit.py`, `agents/rollout.py`. The ReBeL agent's failure showed
that our search could not see the gaps between moves. It did not show where gaps exist, how big
they are, or how much they depend on the hidden hand. The audit measures that directly.

**Method.** 572 start-of-turn positions from `diverse_sym` self-play (one per game, at a
uniformly random turn). Every first move (play or discard) is played out to the end by the
greedy policy in 8 versions of the hidden information (the true one and 7 re-deals of the
opponent's hand and deck consistent with the mover's view) × 8 shuffles of the deck in each.
All moves see the same worlds and the same future cards, so the luck cancels in the differences.
A move's true value is its mean over the 64. "True" means *if both sides then continue with the
policy*: this measures one-turn improvements on it, not deeper plans.

**1. Our search loses everywhere, and worst in one specific way.**

| Phase | n | Search changes the move | True gain per turn | Gain it claimed |
|---|:---:|:---:|:---:|:---:|
| Early (deck > 30) | 197 | 0.83 | −4.08 ± 0.52 | +9.05 ± 0.66 |
| Middle (16–30) | 205 | 0.87 | −6.02 ± 0.61 | +14.31 ± 0.86 |
| Late (≤ 15) | 170 | 0.85 | −4.78 ± 0.56 | +10.01 ± 0.84 |

By kind of swap (policy's move → search's move; true gain, points):

| Swap | n | True gain |
|---|:---:|:---:|
| extend a route → **discard the opponent can play at once** | 125 | −10.53 ± 0.80 |
| discard → discard the opponent can play at once | 33 | −11.93 ± 1.79 |
| open a route → discard the opponent can play at once | 24 | −9.79 ± 1.84 |
| open a route → other discard | 37 | −6.85 ± 1.14 |
| extend a route → other discard | 76 | −3.47 ± 0.58 |
| extend → extend, open → open, discard → discard | 51 / 40 / 35 | −1.68 / −1.88 / −1.07 |

- The search's costliest habit is handing the opponent a playable card: the same leak the
  agents themselves once had to unlearn. The value network was trained on games where the
  opponent mostly was not offered such cards.
- The value network is not empty: its ranking of moves correlates with the true ranking at
  0.36–0.44. But its top move is the truly best one only 9–12% of the time.
- Anchored to the policy (8 points) the search changes 18–29% of moves for a true gain of zero
  (−0.11 ± 0.13, −0.05 ± 0.14, +0.15 ± 0.14).

**2. The policy is improvable in the endgame, and not measurably before it.** What a better
first move would gain over the policy's, per turn:

| Phase | n | Upper bound | Lower bound | Lower bound among the policy's own top 3 |
|---|:---:|:---:|:---:|:---:|
| Early (deck > 30) | 197 | +3.85 ± 0.25 | −1.86 ± 0.55 | −0.56 ± 0.33 |
| Middle (16–30) | 205 | +2.63 ± 0.21 | −0.72 ± 0.41 | +0.17 ± 0.29 |
| Late (7–15) | 104 | +1.78 ± 0.30 | −0.39 ± 0.53 | +0.28 ± 0.28 |
| **Endgame (deck ≤ 6)** | 66 | +1.16 ± 0.32 | **+1.05 ± 0.37** | +0.39 ± 0.24 |

(Upper: the best move by all 64 rollouts, which flatters itself. Lower: the best move chosen on
half the rollouts and scored on the other half, which is hurt by choosing among ~13 noisy
candidates. The truth is between.) Only in the endgame do the bounds meet above zero: about a
point a turn is there to be had. Earlier the bracket straddles zero: 32 paired rollouts per move
cannot find a better move than the policy's.

**3. Luck is the obstacle, even with the future cards shared.** How a move's advantage over the
policy's move varies, for moves within 5 points of it:

| Kind of move | Moves | sd across opponent hands | sd across deck orders |
|---|:---:|:---:|:---:|
| Extend a route | 700 | 2.2 | 21.9 |
| Open a route | 461 | 3.9 | 32.8 |
| Other discard | 1187 | 3.7 | 30.1 |
| Discard the opponent can play | 539 | 4.3 | 37.9 |

- One paired rollout's difference between two moves has a spread of 22–38 points. Resolving a
  1-point gap at one position takes on the order of 900 rollouts per move.
- **Hidden information does matter:** the right move shifts by 2–4 points with the opponent's
  hand, most for discards they might use and for opening a route, least for extending one. That
  exceeds the ~1-point gaps between close moves, so belief reasoning has something real to
  capture. It is ~8× smaller than the luck, which is why a network trained on outcomes could not
  find it (and why the beliefs added nothing to the ReBeL value network).

**4. Acting on it: late-game paired rollouts beat the policy.** The agent plays `diverse_sym`,
except that once the deck is down to N cards it plays every first move out with the policy in 8
re-dealt hidden worlds × 8 deck orders, and switches only if the best move's paired advantage
over the policy's move is at least 2 standard errors. Against `diverse_sym`, 400 paired deals:

| Agent | Score | Margin, points a game |
|---|:---:|:---:|
| Rollouts from 6 cards left, uniform re-deals | 0.526 ± 0.006 | +2.97 ± 0.31 |
| Rollouts from 12 cards left, uniform re-deals | **0.538 ± 0.008** | +3.52 ± 0.43 |
| Rollouts from 6 cards left, re-deals drawn from the tracked belief | 0.532 ± 0.008 | +3.39 ± 0.32 |

- The audit's prediction held: ~1 point a turn over the last ~3 own turns is ~3 points a game.
- Starting at 12 cards adds little (+0.5 points for 4× the time): the improvable part really is
  the last few turns.
- Drawing the opponent's hand from the belief instead of uniformly is +0.4 points, about one
  standard error: the right direction, not established.
- It switches away from the policy's move only once or twice a game.

**5. The gain does not transfer to opponents the rollouts mis-model** (2026-10-05). The
6-card rollout agent on the robustness panel, 500 paired deals per style, on the same deals as
the plain policy's panel:

| Opponent style | Rollout agent | Plain `diverse_sym` | Difference |
|---|:---:|:---:|:---:|
| Plain heuristic | 0.901 ± 0.010 (+61.9) | 0.887 ± 0.010 (+60.8) | +0.014 |
| Careful | 0.788 ± 0.013 (+27.7) | 0.795 ± 0.013 (+28.8) | −0.007 |
| Tight (few routes) | 0.904 ± 0.009 (+54.1) | 0.934 ± 0.008 (+63.7) | **−0.030** |
| Aggressive | 0.829 ± 0.011 (+30.5) | 0.823 ± 0.011 (+29.9) | +0.006 |
| Hoarder | 0.803 ± 0.012 (+30.1) | 0.812 ± 0.012 (+32.9) | −0.009 |

- Four of five are within one standard error of the plain policy: no gain, no loss.
- Against the tight style it loses 0.030 and 9.6 points a game (about 2.4 standard errors on
  the score; the margins are far apart). The weakest matchup is unchanged in substance (0.788
  vs 0.795, careful).
- The rollouts play both seats with `diverse_sym`. Their +3 points against `diverse_sym` came
  from predicting that opponent's endgame exactly; against a different endgame (the tight style
  takes and discards differently, which changes when the deck runs out) the same confidence
  test passes on a wrong forecast. Not yet verified move by move.
- So the lever is real but conditional: endgame rollouts help when the opponent model is right.
  Making it general needs rollouts that are robust to the opponent (a mixture of styles in the
  opposing seat, or overriding only when a move wins under every style).

**6. Opponent-robust rollouts transfer** (2026-10-06). Every rollout is repeated with three
opponents in the other seat (`diverse_sym`, the careful heuristic, the tight heuristic; the agent's
own seat always plays `diverse_sym`). The policy's move is overridden only if the best move's
paired advantage, averaged over the three, is at least 2 standard errors *and* it is not worse
under any one of them. Still from 6 cards left, 8 worlds × 8 deck orders.

| Opponent | Robust rollouts | One-model rollouts | Plain `diverse_sym` |
|---|:---:|:---:|:---:|
| `diverse_sym` (400 deals) | 0.524 ± 0.006 (+2.5) | 0.526 ± 0.006 (+3.0) | 0.500 |
| Plain heuristic *(not modelled)* | **0.918 ± 0.009** (+65.5) | 0.901 (+61.9) | 0.887 ± 0.010 (+60.8) |
| Careful *(modelled)* | **0.827 ± 0.012** (+31.9) | 0.788 (+27.7) | 0.795 ± 0.013 (+28.8) |
| Tight *(modelled)* | **0.956 ± 0.006** (+67.9) | 0.904 (+54.1) | 0.934 ± 0.008 (+63.7) |
| Aggressive *(not modelled)* | **0.843 ± 0.011** (+31.4) | 0.829 (+30.5) | 0.823 ± 0.011 (+29.9) |
| Hoarder *(not modelled)* | **0.829 ± 0.012** (+34.5) | 0.803 (+30.1) | 0.812 ± 0.012 (+32.9) |

(500 paired deals per heuristic style, the same deals for all three agents; margins in points.)

- Better than the plain policy against **all five styles**, by 0.017 to 0.032, and by 1.5 to 4.7
  points a game. The weakest matchup rises from 0.795 to 0.827.
- The three styles that were *not* among the rollout's opponent models improve too (+0.031,
  +0.020, +0.017), so this is not just fitting the modelled opponents. Each of those three is
  about 1–2 standard errors on its own; the evidence is that all five move the same way.
- Almost none of the head-to-head gain is given up for it (0.524 vs 0.526).
- The loss against the tight style is gone (0.956 vs 0.904), which supports the diagnosis that
  the one-model version was acting on a wrong forecast of the opponent's endgame.

**7. Counters to the policy work less well against the rollout agent** (2026-10-09,
`experiments/exploit_rollout.py`). The agent is too slow (10–20 s a game) to be a PPO training
opponent, so the first exploitability check is indirect: the policies trained or selected to
beat `diverse_sym`, played against both agents on the same 500 paired deals.

| Attacker | vs plain `diverse_sym` | vs robust rollouts | Change in margin |
|---|:---:|:---:|:---:|
| Best exploiter of `diverse_sym` (600 iterations) | 0.528 ± 0.014 (+1.7) | 0.510 ± 0.014 (−0.1) | −1.8 |
| Exploiter, seed 1 | 0.497 ± 0.012 (−0.5) | 0.483 ± 0.012 (−1.9) | −1.4 |
| PSRO best response, round 1 | 0.521 ± 0.013 (+1.1) | 0.504 ± 0.013 (−0.5) | −1.6 |
| PSRO best response, round 3 | 0.506 ± 0.014 (−0.2) | 0.487 ± 0.014 (−1.9) | −1.7 |
| `careful` (previous best agent) | 0.448 ± 0.014 (−6.7) | 0.427 ± 0.014 (−8.7) | −2.0 |

- **All five do worse against the rollout agent**, by 0.014–0.021 on score and 1.4–2.0 points a
  game. Each difference alone is about one standard error; five out of five in the same
  direction, by a similar amount, is the evidence.
- The best attacker goes from a small edge (+1.7 points) to none (−0.1). No existing policy has
  a positive margin against the rollout agent.
- So the rollouts do not open a weakness that counters to the policy can use; they take a
  little away from every one of them.
- **What this does not show:** these attackers were trained against `diverse_sym`, not against
  the rollout agent. A counter aimed at the rollout agent itself (for example at the three
  opponent models it assumes) could exist and would not appear here.

**8. Distilling the overrides into the policy does not work** (2026-10-12,
`experiments/distill_rollout.py`). The rollout agent played 6,000 games (half against
`diverse_sym`, half against a random heuristic style); its 24,680 searched decisions were logged,
4,715 of them overrides of the policy (mean rollout gain of an override: 4.6 points). A copy of
`diverse_sym` was fine-tuned to reproduce the logged moves, with a penalty for drifting from the
original policy on a sample of its other decisions. Held-out games (468 overrides):

| Variant | Overrides reproduced | Other searched moves kept | vs `diverse_sym` (2,000 deals) | Style panel vs `diverse_sym` |
|---|:---:|:---:|:---:|:---:|
| Strong (overrides ×8, lr 3e-5, 30 epochs) | 13.5% | 74.9% | 0.484 ± 0.006 (−1.45 ± 0.42) | worse on all five (−0.018 to −0.043) |
| Gentle (×2, lr 1e-5, 10 epochs, drift ×10) | 1.9% | 93.4% | 0.497 ± 0.004 (−0.08 ± 0.25) | level (−0.011 to +0.004) |
| Weakest (×1, lr 1e-5, 20 epochs, drift ×20) | 2.4% | 94.8% | 0.500 ± 0.003 (+0.18 ± 0.22) | level (−0.003 to +0.006) |

(The rollout agent itself: 0.524 head to head, +0.017 to +0.032 on the panel.)

- There is no setting where the gain transfers. Pushed hard, the network reproduces one override
  in seven on unseen games and disturbs a quarter of the endgame moves the search had confirmed,
  and plays worse than the policy. Pushed gently, it reproduces almost none and plays the same.
- An override is the outcome of ~190 rollouts at one exact position: which card to release with
  five cards left, given both routes and every card seen. 4,247 training examples of that do not
  define a rule a 567-feature network can generalise; they are closer to 4,247 separate facts.
- This matches the audit: the endgame gaps are about a point, found only by averaging luck away
  at the position itself. It is search's kind of gain, not a pattern the policy missed.

**9. Faster rollouts, and a direct exploiter that finds nothing** (2026-10-12).

*Speed.* Within one search the rollouts revisit the same positions under each opponent model and
deck order, so the policy's move is now cached per position (identical decisions, 2–3× faster on
test games). Optionally (`keep=3`), after the first 2 of the 8 hidden worlds only the policy's
move and the 3 best moves so far are rolled out further. Together: 1,000 paired deals in 16
minutes where 400 took 34 (~3× per game). The fast form scores **0.517 ± 0.003 (+1.89 ± 0.17
points)** against `diverse_sym`, against the full form's 0.524 ± 0.006 (+2.45 ± 0.29): dropping
moves early costs about 0.6 points (1.6 standard errors).

*Direct exploiter* (`experiments/exploit_rollout.py train`). A PPO exploiter trained against the
fast rollout agent itself as its opponent: standard exploiter settings (lr 3e-5, no entropy
bonus), warm-started from the best exploiter of `diverse_sym`, 60 iterations × 256 games (the
usual run is 300; this one took 127 minutes). Snapshots screened on 300 paired deals:

| Snapshot | Start | 15 | 30 | 45 | 60 |
|---|:---:|:---:|:---:|:---:|:---:|
| Score vs the rollout agent | 0.490 | 0.490 | 0.497 | 0.497 | 0.477 |

The best, rated on 1,000 fresh deals: **0.493 ± 0.010, margin −0.55 ± 0.72**. The exploiter does
not beat the rollout agent and did not improve on its starting point.

- This is the same picture as every attack on `diverse_sym` (best 0.512), one notch lower.
- What it is worth: on Mini-Ventures this recipe found large weaknesses and missed moderate
  ones, and here it had a fifth of its usual budget. It rules out an easily found counter, not
  a subtle one. The target was the fast form; the full form is slightly stronger.

**Caveats.** The agent costs ~5 s a game (fast form) or 10–20 s (full). Its exploitability is
bounded only by the transfer test and one reduced-budget exploiter.
The gains against the three unmodelled styles are individually small next to their standard
errors.

**Where search suits Ventures, in short.**

- *Suits:* the endgame, where few cards remain, rollouts are nearly deterministic, and the
  policy has real (if small) errors. Honest rollouts, no learned value, a significance test
  before overriding, and more than one model of the opponent (with a single model the gain is
  specific to that opponent).
- *Does not suit:* one-turn search on a learned value anywhere in the game (value error ~10
  points against ~1-point gaps); rollout search before the last ~12 cards at this budget (luck
  of 20–40 points per rollout).
- *Open:* belief reasoning. The hidden hand moves a move's value by 2–4 points, so the prize
  exists; capturing it needs either far more rollouts per decision or a value signal that
  cancels luck the way paired rollouts do.

### ReBeL on full Ventures: every part works, and the agent loses to its blueprint (2026-10-02)

`experiments/rebel_ventures.py`; `selfplay/public_belief.py`, `selfplay/rebel_value.py`,
`selfplay/rebel_turn.py`, `agents/rebel.py`. The agent, at the start of each of its turns, solves
that turn's subgame over candidate hands for both players, with a value network at the positions
one turn ahead, and plays the solved strategy for the hand it holds. `diverse_sym` is the
*blueprint*: it supplies the likelihoods that update the beliefs, and its self-play games are the
value network's training data.

**The parts, each checked on its own:**

- **Public beliefs** (48 candidate hands per player, updated only from public events). Their
  per-card marginals overlap the true hand by 0.354 ± 0.004 with the policy model and
  0.296 ± 0.004 without it (every turn start of 12 games, both players); ~7 ms per move. 128 candidates: no better.
- **Solver speed.** Sampling 6 deck draws per action and skipping the opponent's leaf rows at
  play time cut a solve from ~17,000 to ~2,400 network rows per iteration; the top action was
  unchanged for all 16 test hands. A game takes 7-10 s.
- **Value network** (public features, both beliefs, one hand, card-relative hand features → final
  margin). Held-out R² 0.46, RMSE 32 points against a margin spread of 45, from 885,520 positions
  of 23,792 games, 60% of them containing one random move so that off-policy positions are
  covered. A first version without the hand features and the off-policy games reached R² 0.33.
  It overfits after one epoch either way. **The belief inputs add nothing** (R² 0.457 with,
  0.460 without).

**The match** (100 paired deals each, against `diverse_sym` playing greedily):

| Agent | Score | Margin, points a game |
|---|:---:|:---:|
| Plain search | 0.050 ± 0.015 | −109.8 ± 4.2 |
| Blueprint policy × exp(search value / 3 points) | 0.435 ± 0.032 | −12.7 ± 3.1 |
| Blueprint policy × exp(search value / 8 points) | 0.405 ± 0.026 | −6.3 ± 2.1 |

The more the agent defers to the blueprint, the less it loses on margin; nothing gains.

**Why, measured** (119 turn starts from blueprint self-play, the search run alongside):

- The pipeline is consistent: the solver's value of the blueprint's own move matches the
  network's direct value of the position on average (bias +1.9 points, correlation 0.89).
- But the two differ by **11 points (sd)** for the same hand one move apart. That is the
  network's noise per evaluation.
- The search claims a move **9.6 points better than the blueprint's on the average turn** (median
  8.5), and would change the move on 84% of turns. Gains that size do not exist against a strong
  policy: it is what the maximum over ~16 noisy evaluations looks like.
- A plain-search game ends with routes of length [1, 2, 0, 1, 7] against the blueprint's
  [5, 6, 7, 6, 3]: it opens routes on optimistic noise and never finishes them.

**What this means.**

- ReBeL's logic is sound (it reached the equilibrium on Mini-Ventures, with exact values at the
  end of the game). It needs values accurate to well under the gap between moves. On Ventures a
  single game's margin has ~45 points of luck, the gap between reasonable moves is a point or
  two, and 24,000 games give a network that is ~10 points off.
- This is the same wall every search on this game has hit: AlphaZero's value, belief search,
  expert iteration, endgame CFR, and now ReBeL. The reactive policy wins because policy-gradient
  training averages that luck over ~10^7 decisions; a value network trained on outcomes cannot
  catch up at any data size this machine can generate.
- The beliefs, which are what ReBeL adds over plain search, carry no measurable value
  information here. Hidden information matters less in Ventures than the luck of the draw.
- Anchoring to the blueprint bounds the damage; it cannot create a gain the values do not
  contain.

**What would change the answer:** a value signal ~100× cleaner. Either that many more games
(needs the fast engine), or learning move *differences* from paired rollouts that share the same
future cards, which cancels the luck directly. Or a game where ReBeL's strengths apply: small
private states and outcomes that follow from decisions, as in Liar's Dice or poker.

### E1, stage 1: the approximations Ventures needs survive ground truth (2026-10-01)

ReBeL on Mini-Ventures used three things full Ventures cannot have: a belief vector with one
entry per possible hand (Ventures: ~10^8), a network output per hand, and a solve of every public
state on every pass. Each replacement was switched on in the same code
(`selfplay/rebel.py`; `experiments/mini_rebel.py factored | factored_hand | sampled_256`) and
measured exactly. Exploitability, points a game:

| Variant | Epoch 10 | 20 | 30 | 40 |
|---|:---:|:---:|:---:|:---:|
| Nothing approximated | 0.0048 | 0.0024 | 0.0020 | 0.0025 |
| **Factored beliefs:** the network sees each card's probability of being in the hand (8 numbers), not the probability of each hand (28) | 0.0042 | 0.0032 | 0.0026 | 0.0037 |
| + **per-hand values:** the network takes one hand and returns its value, instead of returning all hands' values | 0.0084 | 0.0069 | 0.0071 | 0.0028 |
| + **sampled positions:** each pass keeps training examples for 256 public states per turn boundary, drawn as games would visit them, instead of all 16 / 736 / 14,856 | 0.093 | 0.0063 | 0.0039 | 0.0035 |

(Nash-seeking PPO: 0.07. Random play: 2.71. Differences of ~0.002 between entries are within
the evaluation's own sampling noise; the last two rows branch from fewer iterations.)

- **The belief and value representations Ventures needs cost nothing measurable.** Per-card
  marginals lose the correlations between cards in a hand, and that loss does not show.
- **Sampling costs data, not accuracy.** With 2,560 last-turn examples (epoch 10) the sampled run
  is at PPO's level (0.093); with 5,120 it is at 0.006; with 10,240 at 0.0035, where full-width
  training had used 594,000. So on Ventures the question is how many self-play positions can be
  afforded, which makes speed the critical path.
- Caveat: Mini-Ventures hands are 2 cards of 8. Ventures hands are 8 cards of 60 with far more
  structure for marginals to lose, so this is evidence for the design, not proof.

**The solver for one position** (`selfplay/rebel_turn.py`). The full-width code works on an
enumerated game. Ventures needs a solver given only the public state, a set of candidate hands per
player with range weights, and a value function for the positions one turn ahead. It expands the
mover's turn through the ordinary game interface and runs the same CFR+ with the same
trembling-hand beliefs. Checked against the full-width solver on Mini-Ventures with every hand as
a candidate: root values agree to 1e-9 at the opening; at the two later turns 1,876 of 2,365 leaf
beliefs are identical and the other 489 are all leaves that several public histories share
(Mini-Ventures' public state does not record who discarded which card, so the full-width solver
pools histories there and a one-position solver cannot).

**What a solve costs on real Ventures positions** (Python, CPU, a 512×3 network, 16 candidate
hands per player): 0.2 s to enumerate the turn, then 85 ms per CFR iteration (6,300 network rows),
so ~2 s per decision and ~5 minutes per game at 20 iterations. With 32 candidates early in a game
it is 0.9 s + 330 ms per iteration. Nearly all rows are deck-draw outcomes (each candidate hand ×
each discard or play × each unseen card), so sampling a few draws per action is the first
speed-up; after that the Python engine itself is the limit.

### ReBeL reaches the equilibrium on Mini-Ventures (2026-10-01)

`selfplay/rebel.py`, `experiments/mini_rebel.py`. ReBeL (Brown et al. 2020) searches over **public
belief states**: the public position plus, for each player, how likely each hidden hand is given
the strategy played so far. Here each of the game's four turns is a subgame solved with CFR+ (64
iterations). Where a subgame is cut off, at the start of the next turn, a value network gives the
value of every possible hand *as a function of the beliefs there*. The networks (one per turn
boundary: 16, 736 and 14,856 public states; 28 possible hands per player) are trained on the root
values of subgames solved in self-play, bottom turn first. The implementation is "full width":
every public state is solved on every pass, which only a small game allows, and which is what
lets the complete policy be measured exactly.

**Result.** Exact exploitability of the policy ReBeL plays, in points a game:

| | Exploitability | Best response as 1st / 2nd player |
|---|:---:|:---:|
| Uniform random play | 2.71 | |
| Nash-seeking PPO (the recipe) | 0.07 | |
| **ReBeL, 10 epochs (5 min)** | **0.0048** | −0.041 / +0.051 |
| **ReBeL, 20 epochs** | **0.0024** | −0.045 / +0.049 |
| **ReBeL, 30 epochs** | **0.0020** | −0.045 / +0.049 |
| **ReBeL, 40 epochs (21 min)** | **0.0025** | −0.045 / +0.050 |
| Full-game CFR+ (the reference equilibrium) | 0.0022 | (game value −0.047) |

- ReBeL matches the equilibrium found by solving the whole game at once, while only ever solving
  one turn at a time with learned values beyond it. That is the property that could scale:
  full-game CFR cannot run on Ventures, one-turn subgames can.
- It is about 30× closer to equilibrium than PPO, on a game where PPO had plateaued.
- The figure is an upper bound in one respect: the evaluated policy branches from a sample of
  CFR iterations per turn (8, 4, 16) instead of all 64, because full branching is too slow.

**Three things had to be right, each found by measuring against ground truth:**

1. **Play a random iteration, not the average.** Re-solving each later subgame from the beliefs
   of the *average* strategy above it ("unsafe re-solving") stalls at 0.085 on Kuhn poker (random
   play: 0.458): a subgame's equilibrium given fixed beliefs need not be part of a full-game
   equilibrium (the queen's call is exactly indifferent). ReBeL's rule is to stop CFR at a random
   iteration, play that iteration's strategy and re-solve from *its* beliefs; in expectation that
   is the average strategy of CFR on the whole game. Evaluating that expectation: 0.016.
2. **Beliefs must be defined after a deviation.** The first Mini-Ventures run scored **1.30**.
   The fault was one-sided: a best responder gained 0.05 as second player and 2.0 as first. CFR+
   iterates are near-pure, so under one iterate only 12–22% of last-turn public states had any
   modelled reach; when the opponent deviated into one of the rest, the re-solve had no beliefs
   and played an arbitrary action. Fix: the beliefs handed to the value network and to the next
   subgame assume a trembling hand (each player's strategy mixed with 5% uniform play). CFR
   inside a subgame stays exact and the policy played is unmixed. This also put the network's
   queries on the distribution it trains on (its error at the beliefs it met had been 2.5 points,
   against 0.1 in training). Result: 1.30 → 0.0048 at the same point in training.
3. **The last turn is a best response.** Only the mover acts in the last turn and then the game
   ends, so its subgame is solved by one backward pass instead of 64 CFR iterations (60× cheaper,
   and exact).

**What this does not show.** Mini-Ventures has 28 possible hands per player, so beliefs are exact
vectors and every public state is enumerated. Ventures has ~10^8 possible hands and cannot be
enumerated: beliefs must be approximated (factored per card, or sampled), public states must be
visited by sampled self-play, and subgame CFR needs a fast engine. Whether ReBeL keeps its edge
over PPO under those approximations is the open question (E1).

### The stronger exploiters find nothing more on the real agents (2026-10-01)

The calibration below showed the standard exploiter recipe misses moderate weaknesses, so the two
best agents were re-attacked with the recipes that do find them
(`experiments/remeasure_exploitability.py`; 300 iterations each, warm start from the target,
best checkpoint judged on 1000 fresh duplicate deals):

| Target | A: standard (lr 3e-5, no entropy) | B: lr 3e-4, entropy 0.01 | B2: lr 1e-4, entropy 0.003 |
|---|:---:|:---:|:---:|
| Ventures `diverse_sym` | 0.512 (max of 3 runs) | 0.500 | 0.498 |
| Camel Stocks `camel_nash` | 0.509 (max of 2 runs) | 0.500 | 0.505 |

- **The estimates do not move:** 0.512 and 0.509, the maximum over recipes.
- A score of exactly 0.500 means no checkpoint beat the exploiter's starting point (a copy of the
  target). On Ventures recipe B drifted well below it during training (0.35–0.39) and never
  recovered; B2 hovered around 0.47–0.53.
- **What this does and does not show.** On Mini-Ventures recipe B found 55% of a 0.36-point
  weakness and 92% of a 1.5-point one, so a weakness of that relative size in these agents would
  probably have shown. It found only 20% of the 0.07 one, and looked just like this (a flat 0.500)
  when the standard recipe met it. So: no moderate or large weakness visible to gradient-trained
  exploiters; a small one cannot be excluded.
- Caveat on transfer: recipe B was calibrated on a four-turn game. On ~150-move Ventures the same
  learning rate may simply be too disruptive (the drift suggests so), which is why B2 was added.
  From-scratch exploiters (recipe C) were not run: 300 iterations is far too few to learn Ventures.

### Ground truth on Mini-Ventures: the recipe is near equilibrium; the yardstick was too gentle (2026-10-01)

Mini-Ventures (`games/mini_ventures`) is Ventures' mechanics shrunk until it can be solved exactly:
2 suits × {backing, 2, 3, 4}, 2-card hands, a 4-card deck; one-slot discard piles, and taking a
discard burns the top deck card, so a game is exactly four turns. 2.3M states, 333k information
sets. Two exact tools make everything on it checkable:

- **Vectorized CFR+ over the whole state graph** (`agents/cfr/graph_cfr.py`) finds its equilibrium:
  exploitability **0.002 points a game** after 100 iterations (0.4 s each). First-player value −0.047.
- **Exact best responses** to any policy in 30 s (`eval/exact_br.py`; it matches the old
  tree-walking calculator exactly on the toy game).

**1. The recipe gets close to the true equilibrium** (`experiments/mini_ground_truth.py`; PPO then
Nash-seeking PPO, batches scaled to the game's length). Exact exploitability, points a game:

| Checkpoint | Greedy (deployed) | Sampled |
|---|:---:|:---:|
| PPO 50 | 1.54 | 2.30 |
| PPO 100 | 0.36 | 0.46 |
| PPO 150 | 0.074 | 0.076 |
| PPO 200 | 0.073 | 0.074 |
| Nash 50 / 100 / 150 / 200 | 0.072 / 0.070 / 0.070 / 0.074 | 0.076 / 0.076 / 0.081 / 0.093 |
| Uniform random play | 2.71 | |
| CFR+ equilibrium | 0.002 | |

PPO self-play covers about 97% of the distance from random play to equilibrium, then plateaus at
~0.07. The Nash phase holds it there but adds nothing on this game (it may be too small for
self-play's cycling to matter).

**2. Our standard exploiter recipe misses moderate weaknesses**
(`experiments/calibrate_exploiter.py`). The share of the exact best response's gain that each
trained-exploiter recipe finds (1000 duplicate deals each):

| Target (exact exploitability) | A: standard (warm start, lr 3e-5, no entropy) | B: warm start, lr 3e-4, entropy 0.01 | C: from scratch, lr 3e-4 |
|---|:---:|:---:|:---:|
| PPO 50 (1.54) | 61% | **92%** | 90% |
| PPO 100 (0.36) | **1%** | 55% | **68%** |
| PPO 200 (0.073) | 7% | 20% | −7% |

- The standard recipe (the one behind every exploitability number in this log) sees only large
  weaknesses: 61% of a big one, 1% of a moderate one. Against the final agent it scored exactly
  0.500: its play never diverged from the target's.
- The more aggressive recipes recover 55–92% of moderate and large weaknesses.
- All recipes have a floor: none finds much of a 0.07 weakness (~3% of what random play gives up).
- Each recipe gives a lower bound, so **the maximum over several recipes is the estimate to
  report.**

**What this changes.** The Ventures (0.512) and Camel Stocks (0.509) exploitability figures used
the standard recipe, so they could hide moderate weaknesses. They need re-measuring with recipes
B and C. What it does not change: the recipe demonstrably approaches a true equilibrium where that
can be checked.

**A retraction on the way here.** A first version of this experiment reported that "the yardstick
finds ~4% of the true exploitability" of a PPO agent that was itself no better than random. Both
came from flaws in the setup: the first Mini-Ventures had a start cost of 5, which made almost any
play lose points (so the best response "won" by not playing), and PPO used Ventures' batch size on
a game with ~16× fewer decisions per game, so nothing learned. Both are fixed above.

### The agents are strong, but much of the Ventures edge was a leaky opponent (2026-09-29)

The behaviour analysis below found the Ventures agent living off the heuristic's discards. So
the heuristic got a careful human's discard discipline: never hand the opponent a playable card
when a safe discard exists, and nothing else changed (`ventures_careful`). Then each best agent
played the plain heuristic, the careful one, and itself (`experiments/analyze_opponents.py`,
400 games each), with the key matches confirmed on 1000 duplicate deals:

| Match (1000 duplicate deals) | Score | Margin |
|---|:---:|:---:|
| league_control vs plain heuristic | 0.816 ± 0.008 | +54.9 |
| **league_control vs careful heuristic** | **0.570 ± 0.011** | **+6.8** |
| careful vs plain heuristic | 0.521 ± 0.003 | +0.9 |
| **PPO v2 vs careful heuristic** | **0.441 ± 0.011** | **−2.8** |

- **One rule nearly closes the gap.** Against the careful heuristic the agent's margin falls
  from +55 to +7 points a game, and its average route value from +15 to +5. PPO v2, "~1820 Elo",
  actually loses to it.
- **It is not that the careful heuristic plays better in general.** The two heuristics are nearly
  even against each other (0.521). It specifically denies the agent the income it learned to
  depend on: the plain heuristic gifts a playable card 26% of the time, the careful one 4%.
- **The Ventures Elo numbers are inflated by a leaky anchor.** Every rating above is anchored on
  the plain heuristic. Against the careful one, `league_control` is only about +50 Elo up. The
  exploiter yardstick (0.545) was the honest measure all along: a trained counter-strategy does
  better against the agent than the careful heuristic (0.430), but not by much.
- **Why training produced this.** The heuristic was a permanent member of the Nash training
  league, and the agent learned its weakness. Self-play opponents gift too (9.5% in agent-vs-agent
  games), so the lesson never had to be unlearned.
- **Camel Stocks: checked, no leak.** A careful Camel Stocks heuristic (`camel_stocks_careful`:
  sells a good as soon as the opponent is seen collecting it, values denying them that good, and
  takes camels when the opponent's hand is full, the rulebook's tip) is if anything slightly
  weaker than the plain one (0.475 ± 0.011 head to head), and the agent beats it *more*:
  **0.966 ± 0.004** (+22.9) vs 0.940 (+19.5) against the plain heuristic, 1000 duplicate deals
  each. A careful variant only tests the habits one thinks to fix, but together with the validated
  exploitability (0.509), the held-out PIMC result (0.95) and balanced self-play, there is no sign
  of a Ventures-style dependence.
- **Before that check, Camel Stocks showed no sign of the same problem.** In self-play the agent is balanced
  (49% wins; camel bonus split 49/51), and its style is similar against itself and the heuristic.
  Its herd is bigger against itself (4.2 vs 2.0), since the opponent competes for camels too. A
  similar "careful" variant for Camel Stocks has not been built yet.

**The fix works: train against the careful heuristic.** Continuing from `league_control` for the
same 400 iterations with the careful heuristic in the league (20% of games) instead of the plain
one (`careful`), then rating both on a robustness panel of every opponent style (500 duplicate
deals each; the new styles are defined in `games/ventures/heuristic.py`):

| Opponent style | league_control | **careful run** |
|---|:---:|:---:|
| plain heuristic | 0.831 | 0.853 |
| careful (never gifts) | 0.574 | **0.716** |
| hoarder (careful, and takes discards to deny them) | 0.623 | **0.736** |
| aggressive (careful, opens on weak hands, wagers freely) | 0.732 | 0.765 |
| tight (careful, at most 3 routes, never wagers) | 0.803 | 0.892 |
| **weakest matchup** | **0.574** | **0.716** |

It improves against every style, most where it was weakest, and its exploitability holds at
**0.544** (max of 0.538 and 0.544). It learned to win without the gifts, rather than swapping one
exploit for another.

**What it changed in its play** (both agents vs the careful heuristic, 400 games each, the same
conditional questions as below; ± one standard error):

| Question | league_control | careful run |
|---|:---:|:---:|
| Win rate | 0.57 | **0.72** |
| Opening late (under 20 cards left): holds 20+ in that suit? | 43% ± 5 | **59% ± 4** |
| Opening early: holds 20+ in that suit? | 16% ± 1 | 17% ± 1 |
| No discard pile offers a playable card: takes a dead card anyway? | 47% ± 1 | **35% ± 1** |
| Wagered routes that end positive | 62% ± 2 | **68% ± 2** |
| Routes opened / wagers played per game | 4.7 / 4.0 | 4.5 / 3.7 |
| Average route value | +5.0 | **+7.6** |

It kept its signature style (opening early and wide, wagering heavily) but tightened up exactly
where the missing gifts exposed it: late openings are now mostly routes it can pay for, it wastes
fewer draws on dead cards to slow the clock, and more of its wagers pay off.

**A style-diverse league did not beat it.** The same run with all four non-leaky styles in the
league (careful, tight, aggressive, hoarder; 30% of games) instead of the careful heuristic alone
(`diverse`) came out slightly worse on both measures: weakest matchup **0.681** (vs 0.716) and
exploitability **0.568** (max of 0.568 and 0.554, vs 0.544). Two of the four styles are weak (the
agent beats them 0.77–0.88), so spreading the league across them diluted the signal compared with
concentrating on the hardest opponent.

**Suit symmetry changes that.** The same diverse league plus the suit-symmetry regularizer
(`diverse_sym`; `symmetry_coef = 0.1`: the policy and critic on a suit-relabelled position are
pulled toward the fixed policy on the original, DrAC-style) is the best agent on every measure:

| | careful | diverse | **diverse_sym** |
|---|:---:|:---:|:---:|
| Exploitability (max of 2 exploiter runs) | 0.544 | 0.568 | **0.504** |
| vs plain heuristic | 0.853 | 0.859 | **0.887** |
| vs careful | 0.716 | 0.681 | **0.795** |
| vs hoarder | 0.736 | 0.706 | **0.812** |
| vs aggressive | 0.765 | 0.770 | **0.823** |
| vs tight | 0.892 | 0.884 | **0.934** |
| **Weakest matchup** | 0.716 | 0.681 | **0.795** |

An exploitability of 0.504 means the exploiters found essentially nothing, which is exactly the
result that needs validating (a yardstick can fail silently). Three checks:

1. **Head to head it beats `careful`: 0.569 ± 0.010** (+8.1 points a game, 1000 duplicate deals).
2. **The yardstick does find edges on a symmetry-trained network:** against the same run's
   100-iteration checkpoint it finds **0.530** (+3.2), clearly above the final agent's 0.504.
3. **An exploiter started from a different network** (`league_control`) cannot catch up in its
   budget at all (0.479): no hidden weakness there either.

Verdict: `diverse_sym` is clearly the least exploitable Ventures agent built so far, and this
exploiter (which detects edges of roughly +3 to +5 points a game) finds essentially none. That is
"near-unexploitable against this exploiter budget", not a proof of equilibrium.

**The stricter test agrees.** An exploiter with twice the training (600 iterations) scores
**0.512 ± 0.010** (+1.8 points a game), and never climbed meaningfully during training. Over all
three exploiter runs the estimate is **0.512**, level with Camel Stocks (0.509) and down from
0.545–0.56 for every earlier Ventures agent.

**PSRO agrees: near equilibrium within this population.** Policy-Space Response Oracles
(`experiments/psro_ventures.py`, `iirl.eval.psro`) starts from {`diverse_sym`, `careful`,
`league_control`}, and each round solves the population's payoff matrix for its Nash mixture,
trains a best response to that mixture (the exploiter recipe with the mixture's weights as opponent
sampling weights), rates it on 1000 fresh duplicate deals, and adds it. Four rounds:

| Round | Mixture (Nash of the population so far) | Best response vs the mixture |
|:---:|---|:---:|
| 0 | diverse_sym 100% | 0.504 ± 0.009 |
| 1 | br_r0 100% | 0.514 ± 0.009 |
| 2 | br_r1 100% | 0.507 ± 0.010 |
| 3 | br_r1 100% | 0.514 ± 0.009 |

No round found more than about +1.5 points a game. `diverse_sym` and the four best responses form a
tight cluster (all within about ±0.03 of each other, 300-deal pairs), and every member of it beats
`careful` and `league_control` clearly. The mixture settles on `br_r1` because it measures at or
above 0.5 against everyone, but checked properly it is a narrow counter to `diverse_sym`
(0.515 ± 0.009, +1.6, over 1000 deals) that does slightly *worse* against the careful heuristic
(0.780 vs 0.795), so `diverse_sym` stays the best general agent. The meta-solver goes all-in on
whichever near-tied member measures a hair ahead; with near-identical members that is noise, not
a flaw in the conclusion.

Taken together with the three direct exploiter runs (up to 600 iterations; max 0.512), four PSRO
best responses (max 0.514): **the best Ventures agent is exploitable by at most about 1.5 points a
game by anything our RL-based best responses can find.** That is the strongest "unbeatable"
evidence this toolkit can produce; the remaining unknowns are opponents unlike these (humans) and
edges smaller than the measurement resolves.

**Endgame CFR does not beat the trained policy.** Depth-limited Monte Carlo CFR
(`agents/cfr/subgame.py`; validated on the toy game, `tests/test_cfr_subgame.py`) takes over from
the policy in the last 8 deck cards (`experiments/endgame_cfr.py`). Three kinds of position value
at the depth limit, rated on duplicate deals:

| Leaf values (policy) | vs its own policy | vs careful heuristic |
|---|:---:|:---:|
| heuristic, points (careful.pt; depth 4) | 0.487 ± 0.014 | 0.710 (policy alone 0.716) |
| value net, win/loss (diverse_sym; depth 2) | 0.325 ± 0.020 (−30) | 0.592 (policy alone 0.795) |
| value net + margin blend (diverse_sym; depth 2) | 0.452 ± 0.017 (−9.4) | 0.802 (policy alone 0.795) |

At best it ties the policy against the careful heuristic, and it always loses head to head. The
failures are instructive. The solver samples only the opponent's hidden cards, so in its subgame
the opponent effectively knows our hand; the endgame then looks worse than it is. With a pure
win/loss value that is fatal: in a position that looks lost every move scores −1, the solver
becomes indifferent and plays at random (−30 a game). Blending in the margin restores "lose by
less", but the pessimism remains. The fix is structural, not tuning: reason about what each player
can actually infer (public belief states, as in ReBeL / Student of Games, Phase E). Until then,
the trained policy plays the endgame better than this solver.

The practical conclusion, and the user's intuition: these agents are strong, not unbeatable. The
next training runs should use the careful heuristic (and other non-leaky opponents) in the
league and as the Elo anchor, and a human who simply never gifts takes away most of the Ventures
agent's edge.

**Also: R-NaD-style averaging did not help.** With the magnet as a slowly moving average of the
policy (`magnet_ema = 0.02`) instead of a periodic copy, and everything else as `nash_g`, the
ladder reads 1968 / 1983 / 1962 / 1979 (same as `nash_g`) and the exploiter scores 0.560 (max of
0.532 and 0.560) vs `nash_g`'s 0.547. No gain in strength or exploitability.

### How the best agents play, vs the human-style heuristic (2026-09-29)

Each game's best agent played its heuristic for 200 games (both seats), with every decision logged
(`experiments/analyze_play.py`). The heuristics encode the standard human advice for each game, so
they stand in for "what a person naturally does". Both columns come from the same games, which
means some differences are the agent reacting to this particular opponent; that is noted where it
matters.

**Ventures** (`league_control`, 1973 Elo; wins 84% of these games, 73 vs 16 points a game)

| Per game or per decision | Agent | Heuristic |
|---|:---:|:---:|
| Routes opened | **4.5** | 2.7 |
| Turn of the first opening | **1** | 12 |
| Backing (wager) cards played | **4.1** on 2.4 routes | 0 |
| Routes that end negative | **36%** | 1% |
| Average route value | **+16** | +6 |
| 8-card (+20 bonus) routes | **0.74** | 0 |
| Share of turns spent discarding | **23%** | 66% |
| Discards the opponent can play on a route they started | **0.2%** | 25% |
| Draws taken from a discard pile | **48%** | 0.3% |

What stands out, against the usual human advice:
- **It opens early and wide.** The standard advice is to open two or three expeditions, late,
  and only with the cards to finish them. The agent opens a route on its first turn and ends
  with four or five, wagering on more than half of them. (The heuristic is a *cautious* human
  stand-in: it never wagers at all, where a typical player wagers on a strong route.)
- **It accepts losing routes.** A third of its routes finish below zero, which a careful human
  treats as a mistake. Its routes are worth nearly three times as much on average, and three in
  four games include an 8-card route for the +20 bonus. It plays the portfolio, not each route.
- **It almost never feeds the opponent.** Only 0.2% of its discards are cards the opponent can
  play, against 25% for the heuristic. It also plays instead of discarding far more often.
- **It lives off the discard piles.** Half its draws come from a discard pile; the heuristic
  almost never takes one. Some of that is the heuristic's gift discards, which the agent picks
  up, but see Q2 below: much of it is taking *dead* cards to slow the clock.

Conditional questions (400 games; each asked only in the positions where the choice arises;
± one standard error):

| Question | Agent | Heuristic |
|---|:---:|:---:|
| Q1. Opening a route early: does the hand hold 20+ in that suit (enough to pay the start cost)? | 12% ± 1 (n=1686) | 100% (n=736) |
| Q1. Opening a route late (under 20 cards left): holds 20+? | 30% ± 4 (n=152) | 100% (n=336) |
| Q2. A discard pile offers a card it can play: takes one? | 40% ± 1 (n=7886) | 0.3% |
| Q2. No pile offers a playable card: takes a discard anyway? | **50% ± 1** (n=3255) | 0% |
| Q3. A safe discard exists: discards a card the opponent can play anyway? | **0.3%** (n=2694) | 25% ± 1 (n=7854) |
| Q4. Wagering: sum of the numbers held in that suit | 14.3 ± 0.2 (n=1587) | never wagers |
| Q4. Wagered routes that end positive | 72% ± 2 (n=947) | — |
| Q5. Plays over a lower card of the same suit it still holds (wasting it)? | 0.3% (n=7257) | 0% |

- **Q1: it opens on hope, not on a made hand.** The standard rule is to open a route only when
  the cards in hand already cover the 20-point start cost. The heuristic follows it exactly. The
  agent opens early with enough only 12% of the time, and even late only 30%. It is counting on
  future draws and the discard piles to fill the route, and three in four routes succeed.
- **Q2: it takes dead cards to slow the clock.** When no pile offers a card it can use, it still
  takes a discard half the time. These cards are dead to it (below its route's top card), and
  none of them was a card the opponent could play (there was never one to deny, across 300 games).
  So the point is the clock: taking a discard does not shrink the deck. These takes come late
  (about 8.5 cards left) and mostly when its *projected* score, which counts playable cards in
  hand, is ahead: 71% ± 1 of the dead-card takes, against 44% ± 2 of its deck draws in the same
  situation. When it has more cards to play than turns left, it buys turns. A human who is ahead
  usually wants the game over.
- **Q3: the heuristic's gifts are not forced.** It hands over a playable card 25% of the time
  *even when a safe discard is available*. The agent almost never does.
- **Q4: it wagers light, and it mostly works.** Its average holding in a suit when it plays a
  backing card is 14 points of number cards, short of the 20 start cost, yet 72% of wagered
  routes end positive.
- **Q5: the one human rule it keeps.** It essentially never plays over its own lower card of the
  same suit, which would strand that card.

**Camel Stocks** (`camel_nash`; wins 95% of these games, 80 vs 59 points)

| Per game or per decision | Agent | Heuristic |
|---|:---:|:---:|
| Turns spent on exchanges | **23%** | 3% |
| Turns taking a single good | 29% | **56%** |
| Exchanges that give back goods (not only camels) | **35%** | 0% |
| Goods per exchange | **2.7** | 2.1 |
| Herd at the end | **2.1** | 6.5 |
| Camel bonus won | **26%** | 74% |
| Sales of 3+ cards (earning a bonus) | **44%** | 31% |
| Precious-goods sale size | **2.6** | 2.1 |
| Points per sale | **9.2** | 6.9 |
| Goods sold first (best tokens) | **57%** | 43% |
| Single takes that are precious | **52%** | 35% |

What stands out:
- **Camels are spent, not saved.** The agent ends with a third of the heuristic's herd and gives
  up the +5 herd bonus three games in four. The camels become exchanges instead: a quarter of its
  turns, often swapping two or three goods at once, and a third of the time paying with unwanted
  goods rather than camels.
- **Fewer, bigger, earlier sales.** It sells about as often as the heuristic, but in bigger sets:
  44% of its sales earn a bonus token, its gold, silver and gem sales are larger, and it gets the
  first (most valuable) tokens in more goods. Each sale is worth a third more.
- **Precious goods over bigger sets? No more than a human.** Over half its single takes are gems,
  gold or silver, but that is expected. The telling case is the dilemma: a precious good on offer
  *and* a cheaper good that would grow a set of two or more. Over about 660 such positions each
  (400 games), both players take the precious good about three times in four (agent 77% ± 2,
  heuristic 76% ± 2). The agent is, if anything, a little *more* willing to build the cheap set
  (12% vs 7%) and less likely to do something else instead.
- **It opens up the market less for the opponent.** Taking one good refills a slot from the
  deck, and taking camels refills several: fresh cards the opponent gets first pick of. An
  exchange draws nothing; it refills the market with cards you choose. The agent reveals 15% fewer
  deck cards per turn (0.75 vs 0.89) and leaves 19% fewer precious goods in the market for the
  opponent's turn (0.45 vs 0.55; the total top-token value it leaves is only 4% lower). What its
  exchanges hand back is cheap: goods averaging a 2.5-point top token, only 14% of them precious.
  This fits a deliberate "don't open the market" habit, but it is also a side effect of exchanging
  more and taking fewer single goods, and these statistics cannot tell the two apart.
- **The rulebook's camel tip does not show up.** The rules advise taking a big group of camels
  when the opponent's hand is full, so they cannot use the new market cards. The agent does this
  no more often than the heuristic (35% vs 43% of its camel takes come when the opponent holds 6+
  cards).

**The common thread.** In both games, the agent trades safety for volume. It takes risks a
careful player avoids (losing routes; a lost camel bonus) to get more of what scores the most
(long wagered routes; big, early precious sales), and it is careful exactly where a human is
careless: what it hands the opponent.

### NFSP vs Nash-seeking PPO on Camel Stocks (2026-09-30)

Two equilibrium-seeking methods at equal budget, both from `camel_ppo` for 400 iterations of 256
games with the same PPO settings (`experiments/nfsp_camel.py`, `iirl.selfplay.nfsp`):

* **Nash-seeking PPO** (`camel_nash`): a KL "magnet" toward a slowly refreshed copy of the policy.
* **NFSP** (`camel_nfsp`): a PPO best response trained against the anticipatory mix (90% the
  average policy, 10% itself), and a supervised average policy imitating the best response's
  decisions from a reservoir buffer. The average policy is the deployed agent.

| | Nash PPO | NFSP |
|---|:---:|:---:|
| Exploitability (max of 2 exploiter runs) | 0.509 | 0.515 |
| vs heuristic (1000 duplicate deals) | **0.940** | 0.927 ± 0.006 |
| Head to head (NFSP's score) | | **0.456 ± 0.010** (−1.3 a game) |

Both are near-unexploitable by our exploiters (the difference is within noise), but NFSP's average
policy is measurably weaker: it loses head to head and does worse against the heuristic. That fits
the usual picture: NFSP learns its deployed policy second-hand, by imitating a moving best
response, which lags and blurs it, while the magnet trains the deployed policy directly. On these
games, the MMD / R-NaD family is the better equilibrium method.

### Stage 2 begins: Camel Stocks (2026-09-29)

The second game is **Camel Stocks**, an original trading card game on the mechanical skeleton of
Jaipur (credited as inspiration; own name and theme). Two players share a five-card market: take
one good, take all the camels, exchange several goods for cards and camels, or sell a set of one
good for value tokens that shrink as the good sells out, with a bonus for sets of 3+.

- **The interface held.** The engine implements the same `Game` contract as Ventures, and the
  game-agnostic PIMC search agent ran on it with no changes (40/40 vs random). That was the
  plan's test for whether the interface is real or Ventures-shaped.
- **One design choice worth noting.** The exchange ("take any 2+ goods, give back as many cards or
  camels") is combinatorially huge as a single action. It is split into steps: take goods one at
  a time, stop, give cards one at a time, with legality guaranteeing every exchange can be
  completed within the hand limit. The action space stays flat at 63 ids, which PPO needs.
- **A strong heuristic again.** Selling early and big, favouring precious goods and trading
  camels for goods, it beats random in every game (+53 points) and beats PIMC at 4 worlds × 60
  simulations (PIMC scores 0.258). As on Ventures, cheap search does not beat a good heuristic.

**The recipe transfers, and does better here.** Trained with the Ventures recipe unchanged (600
iterations of PPO from scratch, then 400 of Nash-seeking PPO with `gamma = 0.999`; about 256k games,
under two hours), rated on the same kind of ladder against its own heuristic (anchored at 1500 Elo,
an arbitrary convention not comparable with Ventures):

| Checkpoint | ppo 100 | ppo 300 | ppo 500 | ppo 600 | nash 100 | nash 200 | nash 300 | nash 400 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Elo | 1810 | 1851 | 1912 | 1889 | 1915 | 1950 | 1972 | 1968 |

The final agent scores **0.94 vs the heuristic** (margin +19.5), and the Nash phase adds about
+80 Elo over plain PPO. **Exploitability: 0.509** (max of two exploiter runs, 0.509 and 0.481):
the exploiters found essentially no edge. Because "found nothing" is exactly what a broken
yardstick reports, it was validated on a known-weaker target: the same procedure against the
100-iteration PPO checkpoint finds **0.571** (+2.2). So the yardstick works on this game, and the
Nash-trained agent is close to unexploitable within this exploiter budget, much closer than on
Ventures (0.545).

A plausible reason it is easier (a hypothesis, not yet tested): Camel Stocks is likely shorter,
its luck milder (the market is public and most of the hidden information is one hand), and its
margins small, so there is less for an exploiter to find.

**Held out: it beats an opponent it never trained against.** PIMC search (8 worlds × 80
simulations) is roughly even with the heuristic (the heuristic scores 0.450 ± 0.044 against it),
and the Nash-trained agent beats it **0.950 ± 0.026** (+23.3 per game, 50 duplicate deals), about
as decisively as it beats the heuristic it trained with (0.94). Its strength is not tuned to its
training opponents.

**Hardening the interface after the second game.** With two real games, the refactor the plan
deferred was small. The generic agents, trainers and evaluators already had no game-specific code.
The heuristics moved into their games' packages and register themselves, a new `player_view`
hook made the play server game-agnostic, and `docs/adding-a-game.md` now spells out the contract
and which component uses each hook. One planned abstraction, a separate `Encoder` class, was
dropped: the encoder as a `Game` method was enough for both games.

### Stage 1b: toward an unexploitable agent (2026-09-25 → 09-28)

**The yardstick: exploitability.** A two-player zero-sum Nash strategy cannot lose in expectation
to anyone, so the target is an agent that a best response cannot beat. The measure: warm-start a
PPO exploiter from the frozen target, train it only against the target, and rate the best of its
checkpoints on 1000 fresh duplicate deals (each deal played both ways round, which halves the error
bars). 0.500 would be unexploitable. Getting this right took three tries. With PPO's default
entropy bonus and step size the exploiter drifted *below* a strong target and falsely reported it
unexploitable. `entropy_coef = 0`, `lr = 3e-5`, a critic warm-up and screening the top
checkpoints (greedy and sampled) fixed it. Lesson: always validate "found nothing" against a
known-weak target.

**Nash-seeking beats plain training at equal cost.** Both runs warm-start from PPO v2 and play 51k
more games (`experiments/train_ventures_nash.py`):

| Agent | Exploiter score | vs heuristic | Elo (ladder) |
|---|:---:|:---:|:---:|
| PPO v2 | 0.628 (+12.6) | 0.698 | 1822 |
| Plain PPO, same budget (control) | 0.584 (+8.5) | 0.789 | 1905 |
| Magnet KL + league (`nash`) | 0.537 (+4.4) | 0.774 | 1890 |
| `nash`, 600 more iterations (`nash_long`) | **0.522 (+2.5)** | **0.822** | **1941** |

`nash` and the control are equally strong (they tie 0.502 head to head), but `nash` is about half
as exploitable. `nash_long` is better on every axis: it beats `nash` 0.532 and PPO v2 0.640. Much
of the gain over v2 is simply more training: v2 was never converged, and noisy 400-game evals hid
the climb.

**The Elo ladder.** `experiments/elo_ladder.py` rates every checkpoint in the lineage against the
heuristic on the *same* 1000 duplicate deals (about ±10 Elo, vs ±35 for in-training evals), saved
in `runs/elo_ladder.json`:

| Checkpoint | PPO v2 | nash | nash_long 100 | 200 | 300 | 400 | 500 | 600 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Elo | 1822 | 1890 | 1899 | 1917 | 1920 | 1946 | 1918 | 1941 |

The climb flattens around 1940 after ~400 iterations.

**Belief search: a better belief, no stronger play.** Phase C replaced uniform determinization
with a particle filter over the opponent's hand (`agents/belief.py`): each opponent move reweights
every hypothesised hand by how likely the policy makes that move from it. Lessons from building it:
- A plain filter collapses. Each card the opponent plays out of hand kills the ~80% of particles
  that did not deal it to them, and the opening hand is never re-drawn.
- Replacing dead particles with fresh uniform ones is *worse* than doing nothing (the fresh ones
  swamp the evidence). **Repairing** them works: keep the cards still possible, re-deal the gap
  (`Game.repair_world`).
- A sharpened likelihood (temperature 4) fits a greedy opponent best.

The belief's sampled worlds share **0.453** of the opponent's hidden cards vs **0.386** for uniform
sampling, rising to 0.69 vs 0.60 late in the game. But in play it does not matter: at 64 sims ×
4 worlds, belief search vs uniform search scores 0.500 ± 0.012, and uniform search only ties the
raw `nash_long` policy (0.502 ± 0.013), even with a value net retrained on `nash_long`'s own games
(Brier 0.138 vs the heuristic leaf's 0.166). A Nash-trained policy leaves this much search nothing
to add. Earlier, the same search gave about +30 Elo over PPO v2.

**Dedicated exploiters in the league: so far worse.** AlphaStar-style: each round trains the main
agent with every exploiter found so far as a fixed league opponent, then trains a fresh exploiter
against it (`experiments/train_ventures_league.py`). Two rounds from `nash_long`:

| Agent | Exploiter score | Old exploiter's score vs it | Elo (ladder) |
|---|:---:|:---:|:---:|
| nash_long | 0.522 (+2.5) | — | 1941 |
| league round 1 | 0.556 (+4.6) | 0.550 | 1942 |
| league round 2 | 0.573 (+8.4) | 0.547 | 1962 |
| control: plain Nash recipe, same 400 iterations | **0.544 (+4.5)** | — | **1973** |

Exploitability *rose*, and round 1 did not even close `nash_long`'s own hole (that exploiter does
better against it, 0.550 vs 0.522). Yet round 2 is the highest-rated agent so far vs the heuristic
(1962 ± 10): beating conventional play and being unexploitable are different targets, and here
they pulled apart. The likely reason for the exploitability rise:
training against a few fixed best responses overfits to them rather than moving toward
equilibrium. **The control confirms it:** the same 400 iterations of plain Nash training reach
both a lower exploiter score (0.544 vs 0.573, about 2 SE apart) and a higher rating (1973 vs 1962).
Fixed exploiters in the league hurt on both counts; the plain recipe stays.

**The yardstick is noisier than its error bars.** The control, like the league, first measured
*more* exploitable than `nash_long` (0.544 vs 0.522). Repeating the exploiter run with another
seed settled it:

| Target | Exploiter seed 0 | Seed 1 | Max (the estimate) |
|---|:---:|:---:|:---:|
| nash_long | 0.522 | 0.545 | **0.545** |
| league_control | 0.544 | 0.538 | **0.544** |

A single exploiter run's result moves by ~0.02 between seeds, twice its ±0.010 standard error:
each run is one *search* for a weakness, and some searches find less. Exploitability is the best
response's score, so the estimate is the **maximum over several runs**, and a single run is only a
lower bound. On that basis longer training cost no exploitability (`nash_long` 0.545 vs
`league_control` 0.544), while `league_control` gained ~30 Elo. The league's 0.573 is still worse.
From now on every exploitability number is the max of at least two exploiter runs.

**A plateau.** 400 more iterations of the plain recipe from `league_control`, with the stall fix
(`nash_g`), rate 1969 → 1982 → 1975 → 1962 on the ladder: within noise of `league_control`
(1973). Its exploitability is 0.547 (max of 0.547 and 0.545), the same as `nash_long` and
`league_control`. On both axes the recipe has flattened at about **1970 Elo and 0.545**.

**Also found:** the endgame stall. With one card left in the deck, drawing it ends the game, so the
trailing player never draws it, and a weak policy in the lead also stalls. The 200-turn cap then ends
the game with the same result. `nash` did this in 3 of 200 self-play games, `nash_long` in none.
The leader stalls because training gives it no reason not to. Undiscounted, drawing now and
stalling to the cap earn the identical return, and the deck-clock shaping is flat while the deck
does not move. So the policy is indifferent in a position it rarely sees. It never costs a game
(the leader wins either way), which is why neither training nor Elo noticed. Fix: `PPOConfig.gamma`
(a discount per own decision; shaping becomes `gamma * phi' - phi`, still policy-invariant).
Just below 1 (0.999), a won game is worth slightly more ended sooner. It is used from the next
training run on.
It has not taken yet: `nash_g`, trained 400 iterations with `gamma = 0.999`, still stalls in 1 of
300 self-play games, like `league_control`. In the stalled game the leader (+27) gives drawing
the last card only 4–24% probability, so greedy play never draws it. The position is too rare
(well under 1% of games) for 400 iterations to teach it. Still harmless: the result is the same.

### PPO self-play beats the heuristic — and PIMC — without search
The reactive-RL plateau was an engineering artifact. PPO v2 (pure self-play + snapshot pool, no
search, no heuristic games in training), measured over 2000 seat-balanced games vs the heuristic:

| Run | Change (cumulative) | Score vs heuristic | Mean margin | Elo |
|---|---|:---:|:---:|:---:|
| v1 | terminal reward, raw features, 1.3k games | ~0.03 | — | ~1340 |
| A | card-relative features, potential-shaped reward + GAE, 77k games | 0.489 | +4.9 | 1669 |
| B | + privileged critic (value sees the opponent's hand; train-only) | 0.555 | +8.5 | 1714 |
| C | + win bonus (+20 pts for winning; Elo counts wins, not points) | 0.648 | +16.4 | 1782 |
| **D** | + known-opponent-card tracking, 154k games | **0.674 ± 0.010** | **+24.7** | **~1800** |

Training curves for A–D: [`viz/ppo_curve.html`](viz/ppo_curve.html).

**Held out:** D vs PIMC (12 worlds × 80 sims), an opponent it never trained against, over 200 games:
**0.885 ± 0.023, margin +54.7.** The matchups are non-transitive (PIMC ≈ heuristic, yet D beats PIMC
far harder), so the heuristic-anchored ~1800 is the conservative figure; anchored on PIMC it is
~2060.

What mattered, in order:
1. **Features that state the relations.** The v1 encoding gave hand counts and per-suit route
   summaries; the MLP had to learn "card *c* can go on route *s*" itself. v2 adds per-card-type masks
   aligned with the action ids (playable for me / for the opponent, unseen, discard tops, known
   opponent cards) plus per-suit projections.
2. **Potential-based reward shaping + GAE.** Each decision is rewarded with the change in the game's
   deck-clock potential until the player's next decision; the final step gets `return − potential`.
   The shaped rewards telescope to the true return minus a constant, so the optimal policy is
   unchanged (Ng et al., 1999), but credit lands on the move that earned it.
3. **Scale.** Rollouts step many games in lockstep (one batched forward per step) across 10
   processes, after profiling showed the engine — not the net — was the bottleneck (lookup tables for
   card attributes, cached route summaries): ~150k games in ~65 min on a 12-core CPU.
4. **A privileged critic.** The value baseline also sees the opponent's view (their hand), which
   strips most of the deal's luck out of the advantages. The policy never sees it.
5. **Optimize what Elo measures.** Margin-trained agents play high-variance (open every route early,
   eat ~2 losing routes a game): +5 points/game yet <50% wins. A +20-point win bonus turned the same
   point edge into wins.
6. **Known opponent cards.** A card the opponent takes from a discard pile is public until they play
   it; tracking it (in the state, the encoder and `resample_from_infostate`) sped learning markedly.

Style: D still opens nearly every route in its first few turns and banks ~32 points/game to the
heuristic's ~15 — an aggressive, well-executed style rather than the heuristic's tight one.

Found along the way: **Ventures could fail to terminate** — taking from discard piles does not shrink
the deck, so two players trading discards never end the game (a randomly initialized policy did
exactly this). The engine now ends the game after `MAX_TURNS = 200` turns (a normal game is ~55–75).

### Net-guided PIMC beats the heuristic
Trained net-guided PIMC at 256 sims vs the heuristic: **29 W / 1 D / 10 L (74% score), +31.1 mean,
Elo ~1855.** Training the priors moved head-to-head win rate 25% → ~47% (small budget) and, with
adequate search, clearly past the heuristic. This is the first *learned* agent to beat the heuristic.

### AlphaZero fails on a high-variance card game
Four runs (32→256 sims, batched inference, masked policy loss): Elo hovers ~1140, never beats the
heuristic, training doesn't improve on untrained search. Cause: the value target is luck-dominated
noise; the learned value degrades the search relative to an honest leaf estimate.

### CFR converges to the Kuhn equilibrium (small games only)
Exploitability → 0.0015 on `toy_showdown`, exact Kuhn strategy recovered. Confirmed intractable on
full Ventures (horizon).

### Reactive RL v1 (REINFORCE, PPO) crushes random, plateaus below the heuristic
REINFORCE ~1420, PPO ~1340 (peak 1431); both ~0.97–1.00 vs random, 0–5% vs the heuristic.
*(Superseded for PPO — see "PPO self-play beats the heuristic" above.)*

### Scaling net-guided PIMC
*(in progress — a longer/bigger run to see how far training pushes it; results to be added.)*

---

## Reproduce

```bash
uv sync
uv run pytest                                   # 97 tests
uv run python -m experiments.ventures_tournament   # random < heuristic < PIMC
uv run python -m experiments.train_ventures        # REINFORCE Elo curve
uv run python -m experiments.train_ventures_az     # AlphaZero (negative result)
uv run python -m experiments.train_ventures_guided # net-guided PIMC (beats heuristic)
uv run python -m experiments.train_ventures_ppo    # PPO self-play (~65 min, beats heuristic)
uv run python -m experiments.cfr_toy               # CFR -> Kuhn equilibrium
uv run python -m experiments.train_ventures_nash   # Nash-seeking PPO vs plain control (+ exploiters)
uv run python -m experiments.exploit_ventures      # best-response exploiter vs a checkpoint
uv run python -m experiments.elo_ladder            # rate the lineage vs the heuristic (resumable)
uv run python -m experiments.belief_ventures       # belief quality + belief-search matches
uv run python -m experiments.train_ventures_league # exploiter league (resumable)
uv run python -m experiments.play                  # play the agents yourself (browser, local)
uv run python -m experiments.analyze_play          # how the best agents play vs the heuristics
```
