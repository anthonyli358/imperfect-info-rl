"""Cards and tokens as data for Camel Stocks.

Camel Stocks is an original, from-scratch game reusing a well-known *mechanical* skeleton — a shared
five-card market, taking one good or all the camels, swapping several goods for cards from your
hand and herd, and selling sets for value tokens that shrink as a good sells out. Mechanics are not
copyrightable; the name, theme, art and rulebook text of any published game are. This
implementation shares none of those: it has its own name and theme, no assets, and represents every
card as a plain integer. It is an independent implementation, not affiliated with or endorsed by any
publisher; the mechanical inspiration (the two-player trading card game designed by Sébastien
Pauchon) is credited here only as inspiration.

**Encoding.** A card is a type id: goods ``0..5`` (most to least precious), then ``CAMEL = 6``.
Hands hold goods only (camels go straight to the herd), so hand counts have ``NUM_GOODS`` entries
and deck/market counts have ``NUM_CARD_TYPES`` entries.

**Differences from the original rules (deliberate, v1).** The rules data and turn structure were
checked against the original rulebook and match it. Three simplifications remain:

* Bonus tokens are drawn at random but their values are public; the original keeps them face down
  until scoring. The rest of the hidden information (the opponent's hand, the deck) is kept.
* The payoff is the score difference, so an exact tie is a draw; the original breaks ties on the
  number of bonus tokens, then goods tokens.
* One round is one game; the original plays until a player wins two rounds.
"""

from __future__ import annotations

NUM_GOODS = 6
CAMEL = NUM_GOODS  # card type id of a camel
NUM_CARD_TYPES = NUM_GOODS + 1
GOOD_NAMES: tuple[str, ...] = ("Gems", "Gold", "Silver", "Silk", "Spice", "Hides")
CARD_NAMES: tuple[str, ...] = (*GOOD_NAMES, "Camel")
# Display only. Colours follow the original's conventional colour per good (so players who know it
# read the board at a glance); symbols are plain Unicode glyphs.
GOOD_COLORS: tuple[str, ...] = ("#c8323f", "#d4a017", "#7f8da3", "#7a4fb5", "#3f9a4a", "#8b5a2b")
GOOD_SYMBOLS: tuple[str, ...] = ("◆", "●", "◈", "≋", "✿", "▦")
CAMEL_COLOR = "#c19a5b"
CAMEL_SYMBOL = "🐪"

# Cards per type (goods, then camels): 55 in all.
CARD_COUNTS: tuple[int, ...] = (6, 6, 6, 8, 8, 10, 11)
DECK_SIZE = sum(CARD_COUNTS)
MARKET_SIZE = 5
MARKET_START_CAMELS = 3  # the market starts with these camels plus cards from the deck
DEAL_SIZE = 5  # cards dealt to each player
HAND_LIMIT = 7  # goods in hand (the herd does not count)
PRECIOUS = frozenset({0, 1, 2})  # these goods sell only in sets of 2 or more
MIN_PRECIOUS_SALE = 2

# Value tokens per good, top of the stack first: early sellers get the best prices.
GOOD_TOKENS: tuple[tuple[int, ...], ...] = (
    (7, 7, 5, 5, 5),
    (6, 6, 5, 5, 5),
    (5, 5, 5, 5, 5),
    (5, 3, 3, 2, 2, 1, 1),
    (5, 3, 3, 2, 2, 1, 1),
    (4, 3, 2, 1, 1, 1, 1, 1, 1),
)
# Bonus tokens for selling 3, 4, or 5+ cards at once, as multisets of values (drawn at random).
BONUS_TOKENS: tuple[tuple[int, ...], ...] = (
    (1, 1, 2, 2, 2, 3, 3),
    (4, 4, 5, 5, 6, 6),
    (8, 8, 9, 10, 10),
)
CAMEL_BONUS = 5  # to the player with strictly more camels at the end
EMPTY_PILES_TO_END = 3  # the game ends once this many goods token piles are empty
# The game also ends after this many turns: exchanges neither draw nor sell, so two players could
# otherwise swap forever. A normal game takes ~25–35 turns.
MAX_TURNS = 150


def card_name(card: int) -> str:
    """Return a card type's readable name."""
    return CARD_NAMES[card]
