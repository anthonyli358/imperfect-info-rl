# Findings — imperfect-info-rl

A living log of what we've learned building and pitting agents against each other. Newest findings at
the top of the log. Numbers are Elo anchored to **random = 1000**; the hand-tuned **heuristic = 1676**
(measured by its score vs random). "vs heuristic" is head-to-head win/score rate.

> Written for: contributors and readers of this repo — engineers/researchers who want the *why*, not
> just the scores.

---

## The headline

On **Ventures** (an original ~150-move, high-variance, imperfect-information card game), we ran the
whole agent ladder against a strong hand-tuned heuristic. Each method wins or fails for its **own
characteristic reason** — which is the point of the exercise.

| Agent | Elo | vs heuristic | One-line why |
|------:|:---:|:---:|:---|
| **Net-guided PIMC** (learned priors + honest leaf) | **~1855** | **74% — beats it** | learn the *search priors*, keep an honest leaf value |
| **PPO self-play** (v2 recipe, no search) | **~1800** | **67% — beats it** | card-relative features, shaped rewards, privileged critic, 150k games |
| PIMC (determinized search, no learning) | ~1700 | +10 / ~52% | explicit lookahead with an honest leaf estimate |
| **Heuristic** (hand-tuned) | **1676** | — | deck-clock + wager discipline |
| REINFORCE (linear policy gradient) | ~1420 | ~0% | model-free learning, capacity-capped |
| PPO v1 (terminal reward, raw features, 1.3k games) | ~1340 (peak 1431) | 0–5% | under-trained, credit smeared over ~75 moves |
| AlphaZero (net value + search) | ~1140 | 0% | learned value poisoned by variance |
| Random | 1000 | — | baseline |

**The lesson:** on a long-horizon, high-variance, imperfect-info game, a first-pass reactive
learner plateaus below strong lookahead play — but that plateau was an engineering artifact, not a
ceiling. With features that spell out card/route relations, dense potential-shaped rewards, a
privileged critic and ~100× more games, **pure PPO self-play beats the heuristic (67%) and crushes
plain PIMC search (88%)** with no search at all.
AlphaZero's *learned value* is swamped by the deal's variance and actively hurts its search. CFR is
horizon-intractable here. The search-side route to beating the heuristic with learning is
**net-guided PIMC**: learn only the search **priors** (AlphaZero's policy-improvement idea) while
keeping the honest, hand-crafted leaf value that made plain search strong. (The AlphaZero and
net-guided numbers predate the v2 information-state encoding; they have not been re-run on it.)

---

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
uv run pytest                                   # 51 tests
uv run python -m experiments.ventures_tournament   # random < heuristic < PIMC
uv run python -m experiments.train_ventures        # REINFORCE Elo curve
uv run python -m experiments.train_ventures_az     # AlphaZero (negative result)
uv run python -m experiments.train_ventures_guided # net-guided PIMC (beats heuristic)
uv run python -m experiments.train_ventures_ppo    # PPO self-play (~65 min, beats heuristic)
uv run python -m experiments.cfr_toy               # CFR -> Kuhn equilibrium
```
