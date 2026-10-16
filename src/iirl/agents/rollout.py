"""A blueprint policy improved by paired rollouts where they can resolve a difference.

The audit in ``experiments/ventures_search_audit.py`` found that on Ventures a single paired
rollout's difference between two moves has a spread of 20-40 points, so rollouts can only tell
moves apart late in the game, when few cards remain, and that there the blueprint's move is
about a point a turn short of the best. This agent acts on that: it plays the blueprint, except
that once the deck is down to ``deck_max`` cards it plays every first move of its turn out to the
end of the game with the blueprint, in ``worlds`` re-deals of what it cannot see and ``orders``
shuffles of the deck in each (the same worlds and orders for every move, so the luck cancels in
the differences), and switches to the best move only if its paired advantage over the blueprint's
move is at least ``z`` standard errors. The second decision of a turn stays the blueprint's.

With ``belief`` the re-deals are not uniform: the opponent's hand is drawn from the tracked
public belief (``selfplay.public_belief``, the blueprint as the model of the opponent), with any
cards the agent itself holds swapped for random unseen ones.

**Who the rollouts assume the opponent is.** By default the blueprint plays both seats. That
predicts a blueprint opponent exactly (+3 points a game against it) and mis-predicts others: on
the robustness panel the agent was level against four heuristic styles and 10 points a game worse
against the tight one. With ``robust`` every rollout is repeated with each of ``ROBUST_MODELS``
in the opponent's seat, and the blueprint's move is overridden only if the best move is
significantly better pooled over the models *and* not worse under any one of them.

**Speed.** Two things make a search cheaper without changing what it estimates: the blueprint's
move is remembered per position within a search (the rollouts revisit the same positions under
each opponent model and deck order), and with ``keep`` set, after the first ``probe_worlds``
worlds only the blueprint's move and the ``keep`` best moves so far are rolled out further (most
moves are many points worse and need no more samples).

Game hooks: ``resample_from_infostate`` and a ``deck`` of card counts on the state (the deck's
order is fixed per rollout so that all moves see the same future cards); for ``belief`` also the
hooks ``public_belief`` needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iirl.agents.base import Agent
from iirl.agents.belief import net_opponent_model
from iirl.agents.ppo.agent import PPOAgent
from iirl.core.game import Game, State
from iirl.core.registry import make_heuristic
from iirl.core.types import Action, Player
from iirl.nn.policy_value import NetEvaluator
from iirl.selfplay.public_belief import PublicBelief, PublicBeliefConfig


@dataclass(frozen=True)
class RolloutConfig:
    """Settings for :class:`RolloutImprover`."""

    blueprint: str
    deck_max: int = 6
    models: tuple[str, ...] | None = None  # opponent models; default: per game if ``robust``
    worlds: int = 8
    orders: int = 8
    z: float = 2.0
    robust: bool = False
    keep: int | None = None
    probe_worlds: int = 2
    belief: bool = False
    particles: int = 256
    temperature: float = 4.0
    seed: int = 0


# Opponent models for ``robust`` rollouts, per game: the blueprint itself and heuristic styles.
# On Ventures the other styles (plain, aggressive, hoarder) are left out so that they test
# generalisation.
ROBUST_MODELS = {
    "ventures": ("blueprint", "ventures_careful", "ventures_tight"),
    "camel_stocks": ("blueprint", "camel_stocks", "camel_stocks_careful"),
}


# Cards left in the deck at which each game's rollout agent starts searching: where the paired-
# rollout audits found the policy improvable (experiments/search_audit.py).
DECK_MAX = {"ventures": 6, "camel_stocks": 12}


def turn_start(game: Game, state: State) -> bool:
    """Whether ``state`` is the first decision of a player's turn.

    Uses the game's ``turn_start`` hook if it has one; otherwise a turn starts when the mover's
    hand is at least as large as the other player's (true of Ventures).
    """
    if hasattr(game, "turn_start"):
        return bool(game.turn_start(state))
    me = game.current_player(state)
    return sum(game.private_hand(state, me)) >= sum(game.private_hand(state, 1 - me))


def play_out(
    game: Game,
    state: State,
    seats: tuple[Agent, Agent],
    order: list[int],
    rng,  # noqa: ANN001
    chance_seed: int = 0,
) -> list[float]:
    """Finish the game with ``seats`` playing, drawing the deck in the given order.

    Chance events that are not deck draws (the game's ``chance_from_deck`` hook says which; by
    default all are) are sampled from a stream seeded by ``chance_seed``, so that rollouts sharing
    a seed share those outcomes as far as the positions allow.
    """
    k = 0
    stream = None
    from_deck = getattr(game, "chance_from_deck", None)
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            if from_deck is None or from_deck(state):
                state = game.apply_action(state, order[k])
                k += 1
            else:
                stream = stream or np.random.default_rng(chance_seed)
                outcomes = game.chance_outcomes(state)
                pick = stream.choice(len(outcomes), p=[p for _, p in outcomes])
                state = game.apply_action(state, outcomes[pick][0])
        else:
            agent = seats[game.current_player(state)]
            state = game.apply_action(state, agent.step(game, state, rng))
    return game.returns(state)


class _Memo(Agent):
    """A deterministic agent with its moves remembered per state.

    The rollouts of one search revisit the same positions many times (the same position under
    each opponent model and each deck order until the cards drawn differ), and the blueprint is
    played greedily, so its move there never changes.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.moves: dict[State, Action] = {}
        self.hits = 0
        self.misses = 0

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        move = self.moves.get(state)
        if move is None:
            move = self.moves[state] = self.agent.step(game, state, rng)
            self.misses += 1
        else:
            self.hits += 1
        return move


class RolloutImprover(Agent):
    """The blueprint, overridden late in the game where paired rollouts show a clear better move."""

    def __init__(self, config: RolloutConfig, name: str = "rollout") -> None:
        """Load the blueprint (played greedily, both as the agent and inside rollouts)."""
        self.config, self.name = config, name
        self.blueprint = PPOAgent.from_checkpoint(config.blueprint, greedy=True)
        self.rng = np.random.default_rng(config.seed)
        self.memo = _Memo(self.blueprint)
        self.models: list[Agent] = []  # built on the first search, once the game is known
        self.searched = 0
        self.switched = 0
        # When set to a list, every searched decision is appended to it as
        # (state, legal actions, the blueprint's move, the move played, pooled gain per action
        # over the blueprint's move): the training data for distilling the search into a policy.
        self.log: list | None = None
        self.tracker: PublicBelief | None = None
        self.informed = 0  # worlds drawn from the belief (the rest fell back to uniform)

    def reset(self, game: Game, player: Player) -> None:
        """Start a new game (and, with ``belief``, a fresh belief about both hands)."""
        if self.config.belief:
            cfg = PublicBeliefConfig(
                particles=self.config.particles, temperature=self.config.temperature
            )
            model = net_opponent_model(NetEvaluator(self.blueprint.net))
            self.tracker = PublicBelief(game, model, cfg, seed=int(self.rng.integers(2**31)))
            self.previous = game.new_initial_state()

    def observe(self, game: Game, state: State, action: Action | None) -> None:
        """Update the belief on every transition."""
        if self.tracker is not None:
            self.tracker.observe(self.previous, action, state)
            self.previous = state

    def _worlds(self, game: Game, state: State, me: int) -> list[State]:
        """Re-deals of what the agent cannot see: from the belief if tracked, else uniform."""
        n = self.config.worlds
        if self.tracker is not None and self.tracker.hands is not None:
            mine = np.asarray(game.private_hand(state, me))
            hands, weights = self.tracker.hands[1 - me], self.tracker.weights[1 - me]
            room = np.asarray(game.unseen_cards(state)) - mine  # what the opponent could hold
            size = sum(game.private_hand(state, 1 - me))
            worlds = []
            for k in self.rng.choice(len(hands), size=n, p=weights / weights.sum()):
                # The belief does not know my hand: swap a candidate's cards that I hold for
                # random cards I cannot see, keeping the rest of what the belief inferred.
                hand = np.minimum(hands[k], room)
                spare = np.repeat(np.arange(len(room)), room - hand)
                extra = self.rng.choice(spare, size=size - hand.sum(), replace=False)
                np.add.at(hand, extra, 1)
                worlds.append(game.with_hands(state, me, tuple(mine), tuple(hand)))
            self.informed += n
            return worlds
        return [game.resample_from_infostate(state, me, self.rng) for _ in range(n)]

    def _opponent_models(self, game: Game) -> list[Agent]:
        """The agents that play the opponent's seat in rollouts."""
        cfg = self.config
        names = cfg.models or (
            ROBUST_MODELS[game.game_type.short_name] if cfg.robust else ("blueprint",)
        )
        return [self.memo if name == "blueprint" else make_heuristic(name) for name in names]

    def step(self, game: Game, state: State, rng: np.random.Generator) -> Action:
        """Play the blueprint's move unless rollouts show a clearly better first move."""
        cfg = self.config
        default = self.blueprint.step(game, state, rng)
        me = game.current_player(state)
        legal = game.legal_actions(state)
        if sum(state.deck) > cfg.deck_max or len(legal) < 2 or not turn_start(game, state):
            return default
        if not self.models:
            self.models = self._opponent_models(game)
        self.searched += 1
        self.memo.moves.clear()
        margins = np.zeros((len(self.models), cfg.worlds * cfg.orders, len(legal)))
        base = legal.index(default)
        alive = list(range(len(legal)))
        probe = np.zeros(len(legal))  # every move's mean when the weak ones were dropped
        row = 0
        for w, world in enumerate(self._worlds(game, state, me)):
            if cfg.keep is not None and w == cfg.probe_worlds and len(alive) > cfg.keep + 1:
                probe = margins[:, :row].mean(axis=(0, 1))
                alive = sorted({base, *np.argsort(-probe)[: cfg.keep].tolist()})
            for _ in range(cfg.orders):
                order = np.repeat(np.arange(len(world.deck)), world.deck)
                self.rng.shuffle(order)
                order = order.tolist()
                for k in alive:
                    after = game.apply_action(world, legal[k])
                    for j, model in enumerate(self.models):
                        seats = (self.memo, model) if me == 0 else (model, self.memo)
                        margins[j, row, k] = play_out(game, after, seats, order, rng, row)[me]
                row += 1
        chosen = alive[self._decide(margins[:, :, alive], alive.index(base))]
        if self.log is not None:
            gains = probe - probe[base]
            gains[alive] = margins[:, :, alive].mean(axis=(0, 1)) - margins[:, :, base].mean()
            self.log.append((state, list(legal), default, int(legal[chosen]), gains))
        self.switched += int(chosen != base)
        return int(legal[chosen])

    def _decide(self, margins: np.ndarray, base: int) -> int:
        """The index to play: the best move if it clearly beats ``base``, else ``base``."""
        cfg = self.config
        best = int(np.argmax(margins.mean(axis=(0, 1))))
        if best == base:
            return base
        gain = margins[:, :, best] - margins[:, :, base]  # (model, rollout)
        pooled = gain.mean(axis=0)  # the same rollout under each model: average, then test
        error = pooled.std(ddof=1) / np.sqrt(len(pooled))
        if pooled.mean() < cfg.z * error or error == 0 and pooled.mean() <= 0:
            return base
        if (gain.mean(axis=1) < 0).any():  # worse against one of the modelled opponents
            return base
        return best


def rollout_agent(**settings: object) -> RolloutImprover:
    """Build a :class:`RolloutImprover` from config fields (a picklable spec for matches)."""
    return RolloutImprover(RolloutConfig(**settings))
