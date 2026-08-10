"""
The strategy is the only component that spends money, so it gets tested
against a stub venue with a known order book.

The tests that matter most are the ones about NOT trading: a bot that misses
an opportunity loses nothing, a bot that bets the wrong side loses everything
it staked.
"""

import pytest

from cs2model.data import generate_synthetic_league
from cs2model.model import CS2Model, build_dataset
from cs2model.polymarket import BookTop, MarketInfo, Outcome
from cs2model.ratings import RatingBook
from cs2model.risk import RiskLimits, RiskManager, engage_kill_switch
from cs2model.strategy import Strategy
from cs2model.tracker import Ledger


class StubVenue:
    """A venue with a fixed book, so decisions are deterministic."""

    def __init__(self, books=None, dry_run=True):
        self.books = books or {}
        self.dry_run = dry_run
        self.orders = []

    def get_book_top(self, token_id):
        return self.books.get(token_id, BookTop(0.49, 0.51, 500, 500))

    def buy(self, token_id, price, shares):
        self.orders.append((token_id, price, shares))
        return {"dry_run": True, "status": "simulated"}


@pytest.fixture(scope="module")
def engine():
    matches = generate_synthetic_league(n_teams=10, n_matches=400, seed=21)
    ds = build_dataset(matches)
    model = CS2Model().fit(ds.X, ds.y)
    book = RatingBook()
    for m in sorted(matches, key=lambda x: x.date):
        if m.winner and m.maps:
            book.observe(m)
    # Ordered strongest -> weakest. Tests that need a clear favourite use the
    # ends of this list; picking two arbitrary teams gives a coinflip and the
    # test skips instead of testing anything.
    ranked = sorted(book.teams, key=lambda t: -book.team(t).overall.r)
    return book, model, ranked


def _market(a, b, ta="tokA", tb="tokB", question=None, **kw):
    return MarketInfo(
        question=question or f"{a} vs {b}",
        slug="test", condition_id="c1",
        outcomes=[Outcome(a, ta), Outcome(b, tb)],
        team_a=a, team_b=b, **kw,
    )


def _strategy(engine, tmp_path, venue, limits=None):
    book, model, _ = engine
    led = Ledger(str(tmp_path / "l.json"), starting_capital=1000.0)
    lim = limits or RiskLimits(kill_switch_path=str(tmp_path / "STOP"))
    return Strategy(book, model, led, venue, RiskManager(lim)), led


def test_unknown_team_is_skipped_not_guessed(engine, tmp_path):
    """A name we cannot resolve must never be traded on."""
    strat, led = _strategy(engine, tmp_path, StubVenue())
    d = strat.evaluate_market(_market("Nonexistent Team", "Also Fake"))
    assert not d.taken
    assert "unknown team" in d.reason
    assert led.stats().open_count == 0


def test_unparseable_market_is_skipped(engine, tmp_path):
    strat, led = _strategy(engine, tmp_path, StubVenue())
    m = MarketInfo(question="Who wins the Major?", slug="s", condition_id="c")
    d = strat.evaluate_market(m)
    assert not d.taken
    assert "parse" in d.reason


def test_closed_market_is_skipped(engine, tmp_path):
    _, _, teams = engine
    strat, _ = _strategy(engine, tmp_path, StubVenue())
    d = strat.evaluate_market(_market(teams[0], teams[1], closed=True))
    assert not d.taken
    assert "closed" in d.reason


def test_takes_a_position_when_the_price_is_wrong(engine, tmp_path):
    """Price the model's favourite far too cheap and it must buy."""
    _, _, teams = engine
    a, b = teams[0], teams[-1]          # strongest vs weakest
    strat, led = _strategy(engine, tmp_path, StubVenue())

    p_a = strat.probability(a, b)
    fav, tok = (a, "tokA") if p_a >= 0.5 else (b, "tokB")
    prob = max(p_a, 1 - p_a)
    assert prob >= 0.62, f"strongest vs weakest should be lopsided, got {prob:.2f}"

    cheap = round(prob - 0.20, 2)
    venue = StubVenue({tok: BookTop(cheap - 0.01, cheap, 1000, 1000)})
    strat, led = _strategy(engine, tmp_path, venue)

    d = strat.evaluate_market(_market(a, b))
    assert d.taken, d.reason
    assert d.outcome_name == fav, "must buy the side the model actually likes"
    assert d.edge > 0

    strat.run_once([_market(a, b)])
    assert led.stats().open_count == 1
    assert len(venue.orders) == 1
    assert led.open_positions()[0].outcome == fav


def _moderate_pair(strat, teams, lo=0.60, hi=0.78):
    """
    A pairing the model likes but not overwhelmingly, so there is room to
    price the favourite ABOVE the model without tripping the max_price rule.
    """
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            p = strat.probability(a, b)
            if lo <= max(p, 1 - p) <= hi:
                return a, b, p
    return None


def test_will_not_buy_an_overpriced_favourite(engine, tmp_path):
    """
    Model likes the team, market likes it MORE — negative edge, no trade.

    The favourite must be priced above the model's probability but still
    inside the price bounds, otherwise this would pass for the wrong reason
    (rejected as a near-certainty rather than as a bad price).
    """
    _, _, teams = engine
    strat0, _ = _strategy(engine, tmp_path, StubVenue())
    pair = _moderate_pair(strat0, teams)
    assert pair is not None, "no moderately-priced pairing in the synthetic league"
    a, b, p_a = pair

    fav_prob = max(p_a, 1 - p_a)
    tok = "tokA" if p_a >= 0.5 else "tokB"
    dog = "tokB" if p_a >= 0.5 else "tokA"
    dear = round(fav_prob + 0.06, 2)
    assert dear < 0.90, "test needs the overpriced favourite inside price bounds"

    venue = StubVenue({
        tok: BookTop(dear - 0.01, dear, 1000, 1000),
        # The other side is priced richly too, so there is no value anywhere.
        dog: BookTop(round(1 - fav_prob + 0.06, 2) - 0.01,
                     round(1 - fav_prob + 0.06, 2), 1000, 1000),
    })
    strat, led = _strategy(engine, tmp_path, venue)

    d = strat.evaluate_market(_market(a, b))
    assert not d.taken, f"took a negative-edge bet: {d.reason}"
    assert "edge" in d.reason
    assert led.stats().open_count == 0


def test_kill_switch_stops_the_whole_pass(engine, tmp_path):
    _, _, teams = engine
    venue = StubVenue({"tokA": BookTop(0.09, 0.10, 5000, 5000),
                       "tokB": BookTop(0.09, 0.10, 5000, 5000)})
    limits = RiskLimits(kill_switch_path=str(tmp_path / "STOP"))
    strat, led = _strategy(engine, tmp_path, venue, limits)

    engage_kill_switch(limits.kill_switch_path, "test")
    decisions = strat.run_once([_market(teams[0], teams[1])])

    assert len(venue.orders) == 0
    assert led.stats().open_count == 0
    assert any("HALTED" in d.reason for d in decisions)


def test_never_orders_more_than_rests_at_the_ask(engine, tmp_path):
    """Thin book: the order must not exceed available size."""
    _, _, teams = engine
    a, b = teams[0], teams[-1]          # strongest vs weakest
    strat0, _ = _strategy(engine, tmp_path, StubVenue())
    p_a = strat0.probability(a, b)
    tok = "tokA" if p_a >= 0.5 else "tokB"
    prob = max(p_a, 1 - p_a)
    assert prob >= 0.62

    price = round(prob - 0.20, 2)
    venue = StubVenue({tok: BookTop(price - 0.01, price, 100, 12)})
    strat, _ = _strategy(engine, tmp_path, venue)

    d = strat.evaluate_market(_market(a, b))
    if d.taken:
        assert d.shares <= 12 + 1e-9


def test_map_market_uses_the_per_map_rating(engine, tmp_path):
    """A map market must be priced off that map's rating, not the series."""
    _, _, teams = engine
    a, b = teams[0], teams[1]
    strat, _ = _strategy(engine, tmp_path, StubVenue())

    series = strat.probability(a, b)
    nuke = strat.probability(a, b, map_name="Nuke")
    mirage = strat.probability(a, b, map_name="Mirage")
    assert nuke != mirage or nuke != series, (
        "per-map probabilities should not all collapse to the series number"
    )


def test_map_outside_the_active_pool_is_skipped(engine, tmp_path):
    _, _, teams = engine
    strat, _ = _strategy(engine, tmp_path, StubVenue())
    m = _market(teams[0], teams[1], market_kind="map", map_name="Cache")
    d = strat.evaluate_market(m)
    assert not d.taken
    assert "not in the active pool" in d.reason
