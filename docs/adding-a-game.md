# Adding a game

Every agent, trainer and evaluator in this repo is written against one interface, `iirl.core.game.Game`,
and contains no game-specific code. A new game is an engine plus a few optional hooks, all inside
`src/iirl/games/<name>/`. Nothing in `agents/`, `selfplay/`, `eval/` or `play/` changes.

This was tested on the second game: Camel Stocks reused PIMC search, PPO / Nash-seeking training,
the exploiter, the Elo ladder and the play server without edits to any of them.

## 1. The engine (required)

Subclass `Game` and implement:

| Method | Contract |
|---|---|
| `game_type` | A `GameType`: names, `num_players=2`, `Information`, `Utility.ZERO_SUM`, which tensors you provide |
| `new_initial_state()` | The start state, usually a chance node that deals |
| `current_player(state)` | The player to act, or `PlayerId.CHANCE` / `PlayerId.TERMINAL` |
| `legal_actions(state)` | Legal action ids at a player node. Legality lives here, never in agents |
| `num_distinct_actions()` | Size of the flat action space (policy networks need a fixed size) |
| `apply_action(state, action)` | A **new** state; never mutate. Frozen dataclasses of tuples make this free |
| `is_chance_node` / `chance_outcomes` | Chance events as first-class nodes: `[(outcome, probability), ...]` summing to 1 |
| `is_terminal` / `returns` | Zero-sum payoffs, e.g. the score difference |

Then register it: decorate a factory with `@register("<name>")` and import the package from
`iirl/games/__init__.py`.

**Keep the action space flat and small.** A turn that is combinatorial as one action (Camel Stocks'
"take any 2+ goods, give back as many cards") should become a sequence of sub-decisions with
intermediate states. Make legality guarantee every started sequence can be completed.

## 2. Hooks, and what each one unlocks

| Hook | Used by | Notes |
|---|---|---|
| `information_state_tensor(state, player)` | PPO, AlphaZero search, value nets | **The part to invest in.** Encode relations the policy needs, aligned with the action ids (on Ventures, card-relative features were the largest early gain). Constant shape; never leak hidden information |
| `information_state_string(state, player)` | Belief filter, CFR, play-server logging | Exactly what `player` can observe. Two states with the same string must be indistinguishable to them |
| `heuristic_value(state, player)` | PPO reward shaping, search leaves | A rough zero-sum score estimate. Shaping with it cannot change the optimal policy |
| `resample_from_infostate(state, player, rng)` | PIMC, determinized search, belief filter | A random world consistent with `player`'s view: re-deal only the hidden cards |
| `repair_world(state, player, world, rng)` | Belief filter | Like resampling, but keeps as much of `world`'s hidden cards as still possible. The default resamples uniformly |
| `player_view(state, player)` | Play server | `to_display` with hidden information removed. Required for imperfect-information games |
| `to_display(state)`, `display_meta()`, `action_to_string` | Replay viewer, play page, logs | Human-facing, JSON-serializable |

## 3. A heuristic agent (strongly recommended)

Write a hand-tuned agent in `games/<name>/heuristic.py` and register it with
`@register_heuristic("<name>")`. It is the Elo anchor, a permanent league member during Nash-seeking
training, and the default opponent in the play UI. Game-agnostic code finds it with
`make_heuristic("<name>")`. Aim for something that beats random decisively and beats cheap PIMC.
Both heuristics so far do.

## 4. Tests

- **The conformance suite** runs automatically on every registered game
  (`tests/test_conformance.py`): legal actions, immutability, chance probabilities, terminal
  consistency, termination over hundreds of random games.
- **Rules tests** in `tests/test_<name>_engine.py`: card conservation, each action's effect,
  edge-case legality, end conditions, scoring, and that `resample_from_infostate` preserves the
  player's view string.
- **The play test** (`tests/test_play.py`) should be parametrized over the new game: full games
  through the play session, checking that nothing hidden leaks.

## 5. Training and evaluation (no new code)

- Train with the recipe that worked on both games: PPO from scratch with the v2 settings, then
  Nash-seeking PPO (see `experiments/train_camel_stocks.py`). The trainer takes a game name.
- Rate checkpoints on an Elo ladder anchored on the heuristic (`experiments/elo_ladder.py`).
- Measure exploitability with `experiments/exploit_ventures.py` (it takes `game=`), as the
  **maximum over at least two exploiter runs**, and validate a "found nothing" result against a
  known-weaker checkpoint.
- Add the game's opponents to `experiments/play.py`, and a board renderer to `viz/play.html`
  (without one the page falls back to your view as text plus every legal move as a button).

## 6. IP

Games here are original implementations of mechanical skeletons: own name, theme, and card names;
no art, no rulebook text; the designer credited as inspiration in the module docstring (see
`games/ventures/cards.py`, `games/camel_stocks/cards.py`). Rules data (card counts, token values)
is mechanical and may follow the original. Document any deliberate rule differences there too.
