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


# ── regressions from the second code review ──────────────────────────────────


def test_colliding_team_names_are_refused_not_guessed(tmp_path):
    """
    "Team Spirit" and "Spirit" both normalise to "spirit". The index used to
    keep whichever came last, so the bot priced the market off the WRONG
    team's ratings and bought it.
    """
    from cs2model.data import generate_synthetic_league

    matches = generate_synthetic_league(n_teams=6, n_matches=120, seed=31)
    ds = build_dataset(matches)
    model = CS2Model().fit(ds.X, ds.y)
    book = RatingBook()
    for m in sorted(matches, key=lambda x: x.date):
        if m.winner and m.maps:
            book.observe(m)

    # Two distinct ids that collapse to the same normalised key.
    book.team("Spirit").overall.r = 1900
    book.team("Team Spirit").overall.r = 1200

    led = Ledger(str(tmp_path / "l.json"), starting_capital=1000.0)
    strat = Strategy(book, model, led, StubVenue(),
                     RiskManager(RiskLimits(kill_switch_path=str(tmp_path / "S"))))

    assert strat.resolve_team("Spirit") is None, "must not silently pick one"
    assert set(strat.ambiguous_for("Spirit")) == {"Spirit", "Team Spirit"}

    d = strat.evaluate_market(_market("Spirit", "Nobody"))
    assert not d.taken
    assert "ambiguous" in d.reason


def test_bo1_is_not_forecast_as_bo3(engine, tmp_path):
    """Longer series favour the better team, so mislabelling inflates the edge."""
    _, _, teams = engine
    a, b = teams[0], teams[-1]
    strat, _ = _strategy(engine, tmp_path, StubVenue())
    p1 = strat.probability(a, b, best_of=1)
    p3 = strat.probability(a, b, best_of=3)
    assert p1 != p3
    assert p3 > p1, "the favourite should be stronger over more maps"


def test_market_best_of_reaches_the_forecast(engine, tmp_path):
    from cs2model.polymarket import parse_best_of

    assert parse_best_of("Vitality vs Spirit (Bo1)") == 1
    assert parse_best_of("Vitality vs Spirit - Best of 5") == 5
    assert parse_best_of("Vitality vs Spirit") == 3

    _, _, teams = engine
    strat, _ = _strategy(engine, tmp_path, StubVenue())
    m1 = _market(teams[0], teams[-1], question=f"{teams[0]} vs {teams[-1]} (Bo1)")
    m1.best_of = 1
    m3 = _market(teams[0], teams[-1])
    assert strat.evaluate_market(m1).model_prob != strat.evaluate_market(m3).model_prob


def test_unfilled_limit_order_creates_no_position(engine, tmp_path):
    """
    A resting limit order is not a position. Recording the requested size as
    filled invents exposure that consumes the risk budget and corrupts P&L.
    """
    from cs2model.strategy import _filled_shares

    assert _filled_shares({"status": "live"}, 100.0, dry_run=False) == 0.0
    assert _filled_shares({"size_matched": "40"}, 100.0, dry_run=False) == 40.0
    assert _filled_shares({"response": {"sizeMatched": 25}}, 100.0, dry_run=False) == 25.0
    assert _filled_shares(None, 100.0, dry_run=False) == 0.0
    # Dry run has no book to fill against, so the request stands.
    assert _filled_shares({"status": "simulated"}, 100.0, dry_run=True) == 100.0


def test_only_one_position_per_series(engine, tmp_path):
    """The match market and its map markets are one correlated bet."""
    _, _, teams = engine
    a, b = teams[0], teams[-1]
    strat0, _ = _strategy(engine, tmp_path, StubVenue())
    p_a = strat0.probability(a, b)
    tok = "tokA" if p_a >= 0.5 else "tokB"
    price = round(max(p_a, 1 - p_a) - 0.20, 2)

    venue = StubVenue({tok: BookTop(price - 0.01, price, 5000, 5000)})
    strat, led = _strategy(engine, tmp_path, venue)

    strat.run_once([_market(a, b, ta="t1", tb="t2")])
    assert led.stats().open_count == 1

    second = strat.evaluate_market(_market(a, b, ta="t3", tb="t4"))
    assert not second.taken
    assert "same bet" in second.reason
