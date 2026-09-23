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
| PIMC (determinized search, no learning) | ~1700 | +10 / ~52% | explicit lookahead with an honest leaf estimate |
| **Heuristic** (hand-tuned) | **1676** | — | deck-clock + wager discipline |
| REINFORCE (linear policy gradient) | ~1420 | ~0% | model-free learning, capacity-capped |
| PPO (MLP actor-critic self-play) | ~1340 (peak 1431) | 0–5% | reactive; no lookahead |
| AlphaZero (net value + search) | ~1140 | 0% | learned value poisoned by variance |
| Random | 1000 | — | baseline |

**The lesson:** on a long-horizon, high-variance, imperfect-info game, **search wins**. Reactive
learned policies (REINFORCE/PPO) beat random decisively but plateau below strong lookahead play.
AlphaZero's *learned value* is swamped by the deal's variance and actively hurts its search. CFR is
horizon-intractable here. The thing that lets *learning* actually beat the heuristic is
**net-guided PIMC**: learn only the search **priors** (AlphaZero's policy-improvement idea) while
keeping the honest, hand-crafted leaf value that made plain search strong.

---

## Why each method lands where it does

**PIMC (Perfect-Information Monte Carlo).** Determinized UCT: sample worlds consistent with the info
state, search each with an honest leaf estimate (the deck-clock potential), pool the root visits.
Explicit lookahead → beats the heuristic. The strong baseline everything is measured against.

**Heuristic.** Encodes two real principles: wager discipline (don't open a route you can't drag past
the −20 start cost) and the deck as a shared clock (only ~half the remaining deck draws are yours, so
don't start what you can't finish). Crushes random 100–0; a genuinely hard target.

**REINFORCE / PPO (model-free RL).** Self-play policy gradient. Both climb from ~random to ~1340–1430
and crush random (0.97–1.00), but plateau at 0–5% vs the heuristic. A *reactive* policy with no
lookahead learns good local play but not the multi-step planning the heuristic hard-codes. PPO (MLP +
clipped updates + value baseline) is the best-behaved reactive learner, slightly above linear
REINFORCE. Model-free RL is the *tractable* family here because it only samples episodes — horizon and
infoset count don't blow up, and the value net is just a variance-reduction baseline.

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

### Reactive RL (REINFORCE, PPO) crushes random, plateaus below the heuristic
REINFORCE ~1420, PPO ~1340 (peak 1431); both ~0.97–1.00 vs random, 0–5% vs the heuristic.

### Scaling net-guided PIMC
*(in progress — a longer/bigger run to see how far training pushes it; results to be added.)*

---

## Reproduce

```bash
uv sync
uv run pytest                                   # 48 tests
uv run python -m experiments.ventures_tournament   # random < heuristic < PIMC
uv run python -m experiments.train_ventures        # REINFORCE Elo curve
uv run python -m experiments.train_ventures_az     # AlphaZero (negative result)
uv run python -m experiments.train_ventures_guided # net-guided PIMC (beats heuristic)
uv run python -m experiments.train_ventures_ppo    # PPO self-play
uv run python -m experiments.cfr_toy               # CFR -> Kuhn equilibrium
```
