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
| **Nash-seeking PPO** (`league_control`, no search) | **~1970** | **85% — beats it** | PPO v2 + magnet KL toward a slow copy + league, ~350k more games |
| Nash-seeking PPO (`nash_long`) | ~1940 | 82% | equally exploitable (0.545, max of 2 runs) |
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

**Since then (Stage 1b):** the goal moved from "beat the heuristic" to *unexploitable*, measured
by how much a best-response exploiter trained against the frozen agent can win. Nash-seeking
training (a KL "magnet" toward a slowly refreshed copy of the policy, plus a league) gave the
strongest agents so far, **~1940–1970 Elo with no search**, and far less exploitable: the best
exploiter scores about 0.545 (+4 points/game, max of two exploiter runs), down from 0.628 (+12.6)
against PPO v2.
Adding dedicated exploiters to the league made things worse, not better. With a policy
this strong, determinized search at 64 sims × 4 worlds no longer helps, with or without a
Bayesian belief over the opponent's hand.

---

## Where we stand (2026-09-29)

We have tried most of the families that apply to a long, luck-heavy, imperfect-information card
game, and one is clearly best **so far**: **Nash-seeking PPO, with no search at play time.** It
has the most evidence behind it; "nothing else could beat it" is not established.

| Approach | Verdict on Ventures |
|---|---|
| Heuristic, PIMC search | Solid baselines (~1676–1700 Elo) |
| REINFORCE, PPO v1 | Beat random, plateau below the heuristic |
| AlphaZero (learned value + search) | Fails: the deal's luck swamps the value net |
| Tabular CFR | Exact on toy games; the game is far too long for it |
| Net-guided PIMC | ~1855, but from 40 games, before the current features |
| Expert iteration (search distilled into the policy) | Fragile, no gain |
| **PPO v2 → Nash-seeking PPO** | **Best: ~1970 Elo; exploiter edge down from +12.6 to about +4 points/game** |
| Belief-weighted search on top | Better beliefs, no gain in play at 64 sims × 4 worlds |
| Exploiters as fixed league opponents | Worse on both counts |

> **Update (2026-09-30).** The best Ventures agent is now `diverse_sym` (a four-style league plus
> suit-symmetry regularization): exploitability **0.512** (max of three exploiter runs, up to 600
> iterations), weakest matchup vs any opponent style 0.795. See the findings log.
>
> **Caveat (2026-09-29).** The Ventures Elo figures are anchored on a heuristic that leaks: it
> hands over playable cards. Against a careful version that never does, the best agent scores only
> 0.570 (+6.8 points a game), and PPO v2 loses. See the findings log. Read the Ventures Elo
> numbers as "vs a leaky opponent"; exploitability is the honest measure.

**Why "best" and not "done":**
- **Not unexploitable yet.** The best exploiter still scores 0.545 (+4 points/game). Small, but
  clearly above even.
- **Search was only tested small.** It adds nothing at 64 × 4 on top of the Nash policy, but
  earlier, more search kept helping up to 256 × 8, and the Python engine makes big budgets too
  slow to test properly.
- **One stale number.** Net-guided PIMC's ~1855 predates the current features and was never
  played against today's agents.

**Not yet tried:**
1. **Cheap variants of what worked:** a proper PSRO league, where a meta-solver mixes the
   opponents (fixed exploiters played uniformly are what failed); full R-NaD, which periodically
   resets the magnet to the average policy; a bigger network, or simply longer training.
2. **A fast engine and batched GPU search** (Rust core). This is the only way to learn whether
   search helps the Nash policy at scale, and a prerequisite for the next item.
3. **ReBeL / Student of Games:** search over what both players can infer, with CFR at each step.
   It is the principled route to real unexploitability, and by far the most work.
4. **Deep CFR / NFSP:** equilibrium methods too, but they tend to struggle at this game length.
   More useful as the Stage 6 comparison than as contenders.

The ordered plan is in `plan.md` (Stage 1b, "Next steps").

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

**Endgame CFR did not help (yet).** The best policy with depth-limited Monte Carlo CFR taking over
in the last 8 deck cards (depth 4, 300 iterations, heuristic leaf values; `experiments/endgame_cfr.py`)
scores 0.487 ± 0.014 against the same policy without it (100 duplicate deals), and 0.710 against
the careful heuristic, where the policy alone scores 0.716. The solver disagrees with the policy on
more than half its endgame decisions, but not profitably: the hand-made leaf values are worse
judges of a position than the trained policy. Better leaves (the value net, or policy rollouts to
the end) are the lever, not more depth. The solver itself is validated on the toy game
(`tests/test_cfr_subgame.py`).

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
uv run pytest                                   # 67 tests
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
