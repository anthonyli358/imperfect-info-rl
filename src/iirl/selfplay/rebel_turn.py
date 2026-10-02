"""ReBeL's one-turn subgame, solved through the game interface on sets of candidate hands.

``selfplay.rebel`` enumerates a whole game, which only a tiny game allows. This solver needs
only the position in front of it: the public state, a set of candidate hands for each player with
their range weights (all possible hands in a small game, sampled particles in a large one), and a
value function for the positions one turn ahead. It is the building block for running ReBeL on
games too large to enumerate.

**The subgame.** The player to move makes every decision of its turn (in Ventures: play or
discard, then choose where to draw from), chance resolves, and the turn passes: those positions
are the leaves. Only the mover acts, but the subgame is not a plain maximisation: what the mover
does with each hand changes what the opponent can infer, so the leaf values (functions of the
beliefs at the leaf) move with the strategy. CFR+ iterates strategy -> beliefs -> values ->
regrets to a fixed point, exactly as in ``selfplay.rebel``, with the same trembling-hand beliefs.

**Chance.** After the mover's last decision a chance node may follow. Its outcome is either
private to the mover (a card drawn into its hand: every outcome leads to the same public leaf)
or public (each outcome is its own leaf). An outcome's probability depends on the opponent's
hidden hand; the solver weights each (hand, outcome) by the opponent mass compatible with it.

**Game hooks**, beyond the usual interface: ``private_hand(state, player)`` (card counts),
``unseen_cards(state)`` (counts of the cards in the deck or a hand: public knowledge),
``with_hands(state, player, hand, other_hand)`` (the same public position with these hands),
``chance_is_private(state)`` and ``public_state_tensor(state)``.

The value function takes rows of ``[public features, beliefs of the player to move at the leaf,
beliefs of the other, a hand, whose hand (0: the player to move)]`` and returns that hand's
expected payoff: the per-hand, factored-belief network of ``selfplay.rebel``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from iirl.core.game import Game, State

ValueFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class TurnSolution:
    """What a solve returns. Hands are indexed as in the candidate sets passed in.

    ``actions[i]`` lists hand ``i``'s first-decision actions with ``average[i]`` /
    ``sampled[i]`` their probabilities (the average strategy, and the strategy of the iteration
    the caller asked to stop at). ``follow_ups[(i, action)]`` is the same for the second decision
    after ``action``: ``(actions, average, sampled)``. ``values[p]`` are the root values of
    player ``p``'s candidate hands and ``defined[p]`` says which have any compatible opponent.
    """

    actions: list[list[int]]
    average: list[np.ndarray]
    sampled: list[np.ndarray]
    follow_ups: dict[tuple[int, int], tuple[list[int], np.ndarray, np.ndarray]]
    values: tuple[np.ndarray, np.ndarray]
    defined: tuple[np.ndarray, np.ndarray]


def _match(regret: np.ndarray, infoset: np.ndarray, n: int, uniform: np.ndarray) -> np.ndarray:
    """Regret matching per information set (uniform where no regret is positive)."""
    total = np.bincount(infoset, weights=regret, minlength=n)[infoset]
    return np.where(total > 0, regret / np.where(total > 0, total, 1.0), uniform)


class TurnSolver:
    """CFR+ on the one-turn subgame at a public state, over candidate hands."""

    def __init__(
        self,
        game: Game,
        explore: float = 0.05,
        draws: int | None = None,
        other_values: bool = True,
        seed: int = 0,
    ) -> None:
        """``explore`` is the uniform play assumed when computing beliefs (see ``rebel``).

        ``other_values`` False skips the non-moving player's root values (returned as zeros):
        they are training targets, not needed to choose a move, and cost a network row per
        leaf and candidate hand.

        ``draws`` caps how many private chance outcomes (cards drawn unseen) are followed after
        each of the mover's action sequences: that many are sampled in proportion to their
        probability and reweighted, instead of following every card that could be drawn. None
        follows them all, which is exact.
        """
        self.game, self.explore, self.draws = game, explore, draws
        self.other_values = other_values
        self.rng = np.random.default_rng(seed)

    def solve(
        self,
        state: State,
        hands: tuple[np.ndarray, np.ndarray],
        ranges: tuple[np.ndarray, np.ndarray],
        value_fn: ValueFn | None,
        iterations: int,
        stop: int | None = None,
    ) -> TurnSolution:
        """Solve the turn of the player to move at ``state``.

        ``hands[p]`` is a (candidates, card types) count array for player ``p`` and ``ranges[p]``
        its unnormalised weights; candidates with no weight are ignored (they cannot be held).
        Only ``state``'s public part is used. ``value_fn`` may be None if every leaf is terminal.
        ``stop`` picks the iteration whose strategy is also returned.
        """
        keep = [np.flatnonzero(np.asarray(r) > 0) for r in ranges]
        if all(len(k) == len(r) for k, r in zip(keep, ranges, strict=True)):
            return self._solve(state, hands, ranges, value_fn, iterations, stop)
        sol = self._solve(
            state,
            tuple(np.asarray(h)[k] for h, k in zip(hands, keep, strict=True)),
            tuple(np.asarray(r)[k] for r, k in zip(ranges, keep, strict=True)),
            value_fn,
            iterations,
            stop,
        )
        mover = keep[self.game.current_player(state)]
        n = len(ranges[self.game.current_player(state)])
        actions: list[list[int]] = [[] for _ in range(n)]
        average = [np.zeros(0) for _ in range(n)]
        sampled = [np.zeros(0) for _ in range(n)]
        for k, i in enumerate(mover):
            actions[i], average[i], sampled[i] = sol.actions[k], sol.average[k], sol.sampled[k]
        values, defined = [], []
        for p in (0, 1):
            full = np.zeros(len(ranges[p]))
            full[keep[p]] = sol.values[p]
            there = np.zeros(len(ranges[p]), dtype=bool)
            there[keep[p]] = sol.defined[p]
            values.append(full)
            defined.append(there)
        follow_ups = {(int(mover[k]), a): v for (k, a), v in sol.follow_ups.items()}
        return TurnSolution(
            actions, average, sampled, follow_ups, (values[0], values[1]), (defined[0], defined[1])
        )

    def _solve(
        self,
        state: State,
        hands: tuple[np.ndarray, np.ndarray],
        ranges: tuple[np.ndarray, np.ndarray],
        value_fn: ValueFn | None,
        iterations: int,
        stop: int | None,
    ) -> TurnSolution:
        game = self.game
        m = game.current_player(state)
        o = 1 - m
        hm, ho = np.asarray(hands[m]), np.asarray(hands[o])
        rm, ro = np.asarray(ranges[m], dtype=float), np.asarray(ranges[o], dtype=float)
        unseen = self._unseen(state)
        compat = (hm[:, None, :] + ho[None, :, :] <= unseen).all(axis=2)

        # --- Enumerate every (hand, first action, second action, chance outcome) path ---
        actions1: list[list[int]] = [[] for _ in hm]
        slot1_hand: list[int] = []  # first-decision slots, grouped by hand
        slot2_parent: list[int] = []  # second-decision slots, grouped by first-decision slot
        slot2_action: list[int] = []
        leaves: dict[tuple, int] = {}
        leaf_states: list[State] = []
        p_hand, p_slot1, p_slot2, p_leaf, p_new, p_mine, p_theirs = [], [], [], [], [], [], []

        def leaf_of(key: tuple, world: State) -> int:
            if key not in leaves:
                leaves[key] = len(leaf_states)
                leaf_states.append(world)
            return leaves[key]

        for i in range(len(hm)):
            partners = np.flatnonzero(compat[i])
            if len(partners) == 0:
                continue  # no opponent hand fits: this hand cannot be held
            world = game.with_hands(state, m, tuple(hm[i]), tuple(ho[partners[0]]))
            for a1 in game.legal_actions(world):
                actions1[i].append(a1)
                s1 = len(slot1_hand)
                slot1_hand.append(i)
                after1 = game.apply_action(world, a1)
                for a2 in game.legal_actions(after1):
                    s2 = len(slot2_parent)
                    slot2_parent.append(s1)
                    slot2_action.append(a2)
                    after2 = game.apply_action(after1, a2)
                    held = np.asarray(game.private_hand(after2, m))

                    def add(leaf: int, new: np.ndarray, mine: float, theirs: np.ndarray) -> None:
                        p_hand.append(i)  # noqa: B023 - used within this iteration
                        p_slot1.append(s1)  # noqa: B023
                        p_slot2.append(s2)  # noqa: B023
                        p_leaf.append(leaf)
                        p_new.append(new)
                        p_mine.append(mine)
                        p_theirs.append(theirs)

                    if not game.is_chance_node(after2):
                        add(leaf_of((a1, a2), after2), held, 1.0, compat[i].astype(float))
                        continue
                    pool = self._unseen(after2) - held  # the deck plus the opponent's hand
                    size = pool.sum() - sum(game.private_hand(after2, o))
                    private = game.chance_is_private(after2)
                    events = np.flatnonzero(pool > 0)
                    # Copies of each card left in the deck, per candidate opponent hand.
                    in_deck = np.clip(pool[events] - ho[:, events], 0, None) * compat[i][:, None]
                    mass = ro @ in_deck
                    scale = (mass > 0).astype(float)  # skip cards no opponent leaves in the deck
                    if private and self.draws is not None and scale.sum() > self.draws:
                        points = (self.rng.random() + np.arange(self.draws)) / self.draws
                        cum = np.cumsum(mass / mass.sum())
                        hits = np.minimum(np.searchsorted(cum, points), len(events) - 1)
                        count = np.bincount(hits, minlength=len(events))
                        scale = count * mass.sum() / (self.draws * np.where(mass > 0, mass, 1.0))
                    for n in np.flatnonzero(scale > 0):
                        e = int(events[n])
                        key = (a1, a2) if private else (a1, a2, e)
                        if key not in leaves:
                            fit = int(np.argmax(in_deck[:, n] > 0))
                            alt = game.with_hands(after2, m, tuple(held), tuple(ho[fit]))
                            leaf_of(key, game.apply_action(alt, e))
                        new = held.copy()
                        if private:
                            new[e] += 1
                        add(
                            leaves[key],
                            new,
                            scale[n] * pool[e] / size,
                            scale[n] * in_deck[:, n] / size,
                        )

        n1, n2, n_leaf = len(slot1_hand), len(slot2_parent), len(leaf_states)
        inf1, inf2 = np.asarray(slot1_hand), np.asarray(slot2_parent)
        hand_p, slot1_p, slot2_p = np.asarray(p_hand), np.asarray(p_slot1), np.asarray(p_slot2)
        leaf_p, new_p = np.asarray(p_leaf), np.asarray(p_new, dtype=np.float32)
        mine_p, theirs_p = np.asarray(p_mine), np.asarray(p_theirs)
        mass_p = theirs_p @ ro  # opponent mass compatible with each path
        uni1 = 1.0 / np.bincount(inf1, minlength=len(hm))[inf1]
        uni2 = 1.0 / np.bincount(inf2, minlength=n1)[inf2]

        # --- Leaves: terminal payoffs, public features, and the opponent's (static) beliefs ---
        terminal = np.asarray([game.is_terminal(s) for s in leaf_states])
        payoff = np.zeros((n_leaf, 2))
        for k, s in enumerate(leaf_states):
            if terminal[k]:
                payoff[k] = game.returns(s)
        live = np.flatnonzero(~terminal)
        net_rows = None
        if len(live):
            if value_fn is None:
                raise ValueError("the subgame has non-terminal leaves but no value function")
            feats = np.stack([game.public_state_tensor(leaf_states[k]) for k in live])
            leaf_unseen = np.stack([self._unseen(leaf_states[k]) for k in live])
            fits_o = (ho[None, :, :] <= leaf_unseen[:, None, :]).all(axis=2)  # (live, opp hands)
            weight_o = fits_o * ro
            belief_o = weight_o @ ho / np.maximum(weight_o.sum(axis=1, keepdims=True), 1e-300)
            live_index = np.full(n_leaf, -1)
            live_index[live] = np.arange(len(live))
            paths = np.flatnonzero(~terminal[leaf_p])  # the mover's hands at live leaves
            opp_leaf, opp_hand = np.nonzero(fits_o if self.other_values else fits_o[:0])
            n_cards = hm.shape[1]
            x = np.zeros(
                (len(paths) + len(opp_leaf), feats.shape[1] + 3 * n_cards + 1), dtype=np.float32
            )
            row_leaf = np.concatenate([live_index[leaf_p[paths]], opp_leaf])
            f = feats.shape[1]
            x[:, :f] = feats[row_leaf]
            x[:, f : f + n_cards] = belief_o[row_leaf]  # the opponent moves next
            x[: len(paths), f + 2 * n_cards : f + 3 * n_cards] = new_p[paths]
            x[len(paths) :, f + 2 * n_cards : f + 3 * n_cards] = ho[opp_hand]
            x[: len(paths), -1] = 1.0  # the mover's hands are "the other player's" at the leaf
            net_rows = (x, row_leaf, paths, opp_leaf, opp_hand, slice(f + n_cards, f + 2 * n_cards))

        # --- CFR+ ---
        eps = self.explore
        reg1, reg2 = np.zeros(n1), np.zeros(n2)
        sum1, sum2 = np.zeros(n1), np.zeros(n2)
        kept1, kept2 = uni1, uni2
        mass_m = compat @ ro  # opponent mass compatible with each of the mover's hands
        mass_o = rm @ compat
        total_m, total_o = np.zeros(len(hm)), np.zeros(len(ho))
        value_o = np.zeros((n_leaf, len(ho)))
        value_o[terminal] = payoff[terminal, o][:, None]
        for t in range(1, iterations + 1):
            sig1 = _match(reg1, inf1, len(hm), uni1)
            sig2 = _match(reg2, inf2, n1, uni2)
            if t == stop:
                kept1, kept2 = sig1, sig2
            value_p = payoff[leaf_p, m]
            if net_rows is not None:
                x, row_leaf, paths, opp_leaf, opp_hand, belief_m = net_rows
                trembling = ((1 - eps) * sig1 + eps * uni1)[slot1_p] * (
                    (1 - eps) * sig2 + eps * uni2
                )[slot2_p]
                rho = rm[hand_p] * trembling * mine_p
                held = np.zeros((n_leaf, hm.shape[1]))
                np.add.at(held, leaf_p, rho[:, None] * new_p)
                total = np.bincount(leaf_p, weights=rho, minlength=n_leaf)
                x[:, belief_m] = (held / np.maximum(total, 1e-300)[:, None])[live][row_leaf]
                out = value_fn(x)
                value_p = value_p.copy()
                value_p[paths] = out[: len(paths)]
                value_o[live[opp_leaf], opp_hand] = out[len(paths) :]
            q2 = np.bincount(slot2_p, weights=mass_p * value_p, minlength=n2)
            node2 = np.bincount(inf2, weights=sig2 * q2, minlength=n1)
            node1 = np.bincount(inf1, weights=sig1 * node2, minlength=len(hm))
            reg2 = np.maximum(reg2 + q2 - node2[inf2], 0.0)
            reg1 = np.maximum(reg1 + node2 - node1[inf1], 0.0)
            sum1 += t * sig1
            sum2 += t * sig1[inf2] * sig2
            total_m += t * node1 / np.where(mass_m > 0, mass_m, 1.0)
            if self.other_values:
                played = rm[hand_p] * sig1[slot1_p] * sig2[slot2_p]
                reached = ((played[:, None] * theirs_p) * value_o[leaf_p]).sum(axis=0)
                total_o += t * reached / np.where(mass_o > 0, mass_o, 1.0)

        norm = iterations * (iterations + 1) / 2
        avg1 = _match(sum1, inf1, len(hm), uni1)
        avg2 = _match(sum2, inf2, n1, uni2)
        first = np.searchsorted(inf1, np.arange(len(hm) + 1))
        second = np.searchsorted(inf2, np.arange(n1 + 1))
        follow_ups = {}
        for s1 in range(n1):
            part = slice(second[s1], second[s1 + 1])
            i = slot1_hand[s1]
            a1 = actions1[i][s1 - first[i]]
            follow_ups[(i, a1)] = (slot2_action[part], avg2[part], kept2[part])
        return TurnSolution(
            actions=actions1,
            average=[avg1[first[i] : first[i + 1]] for i in range(len(hm))],
            sampled=[kept1[first[i] : first[i + 1]] for i in range(len(hm))],
            follow_ups=follow_ups,
            values=(total_m / norm, total_o / norm) if m == 0 else (total_o / norm, total_m / norm),
            defined=(mass_m > 0, mass_o > 0) if m == 0 else (mass_o > 0, mass_m > 0),
        )

    def _unseen(self, state: State) -> np.ndarray:
        """Counts of the cards in the deck or in a hand (public knowledge)."""
        return np.asarray(self.game.unseen_cards(state))
