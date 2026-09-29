from dataclasses import replace

import numpy as np

import iirl.games  # noqa: F401  (registers "ventures")
from iirl.core.registry import make
from iirl.core.runner import _sample_chance
from iirl.games.ventures.cards import NUM_SUITS, RANKS, rank_of

G = make("ventures")
R = len(RANKS)


def _relabel(state, sigma):  # noqa: ANN001, ANN202
    """Return ``state`` with old suit ``s`` renamed ``sigma[s]`` everywhere."""

    def card(c: int) -> int:
        return sigma[c // R] * R + c % R

    def counts(t):  # noqa: ANN001, ANN202
        out = [0] * len(t)
        for c, n in enumerate(t):
            out[card(c)] = n
        return tuple(out)

    def per_suit(rows):  # noqa: ANN001, ANN202
        out = [()] * NUM_SUITS
        for s, cards in enumerate(rows):
            out[sigma[s]] = tuple(card(c) for c in cards)
        return tuple(out)

    routes = tuple(
        tuple(tuple(sorted(r, key=rank_of)) for r in per_suit(pr)) for pr in state.routes
    )
    jd = state.just_discarded_suit
    return replace(
        state,
        deck=counts(state.deck),
        hands=(counts(state.hands[0]), counts(state.hands[1])),
        known=(counts(state.known[0]), counts(state.known[1])),
        routes=routes,
        discards=per_suit(state.discards),
        just_discarded_suit=sigma[jd] if jd >= 0 else -1,
    )


def test_suit_symmetries_permute_features_and_actions_exactly() -> None:
    syms = G.symmetries()
    assert len(syms) == 119
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(30):
        s = G.new_initial_state()
        steps = int(rng.integers(5, 120))
        for _ in range(steps):
            if G.is_terminal(s):
                break
            if G.is_chance_node(s):
                s = G.apply_action(s, _sample_chance(G, s, rng))
            else:
                s = G.apply_action(s, int(rng.choice(G.legal_actions(s))))
        if G.is_terminal(s) or G.is_chance_node(s):
            continue
        p = G.current_player(s)
        for k in rng.choice(len(syms), size=5, replace=False):
            index, action_map = syms[k]
            # Recover sigma from the action map (play card s*R + 1 maps to sigma[s]*R + 1).
            sigma = [int(action_map[s * R + 1]) // R for s in range(NUM_SUITS)]
            t = _relabel(s, sigma)
            np.testing.assert_allclose(
                G.information_state_tensor(t, p), G.information_state_tensor(s, p)[index]
            )
            assert sorted(G.legal_actions(t)) == sorted(
                int(action_map[a]) for a in G.legal_actions(s)
            )
            checked += 1
    assert checked > 50
