"""Ledger, heartbeat, risk limits and the strategy's decision logic."""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from cs2model.polymarket import BookTop, MarketInfo, Outcome, normalise_team, parse_question
from cs2model.risk import RiskLimits, RiskManager, engage_kill_switch, release_kill_switch
from cs2model.tracker import Heartbeat, Ledger, kelly_fraction, odds_to_price, price_to_odds

UTC = timezone.utc


@pytest.fixture
def led(tmp_path):
    return Ledger(str(tmp_path / "ledger.json"), starting_capital=1000.0)


# ── position arithmetic ──────────────────────────────────────────────────────


def test_share_maths(led):
    p = led.open_position("A vs B", "A", model_prob=0.70, price=0.40, shares=100)
    assert p.cost == pytest.approx(40.0)
    assert p.to_win == pytest.approx(60.0)
    assert p.edge == pytest.approx(0.30)
    assert p.implied_prob == pytest.approx(0.40)


def test_win_and_loss_move_capital(led):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.50, shares=100)
    assert led.stats().capital == 1000.0, "capital only moves on settlement"
    assert led.stats().at_risk == pytest.approx(50.0)

    led.settle(p.id, "won")
    assert led.stats().capital == pytest.approx(1050.0)

    q = led.open_position("C vs D", "C", model_prob=0.7, price=0.25, shares=200)
    led.settle(q.id, "lost")
    assert led.stats().capital == pytest.approx(1000.0)


def test_sportsbook_odds_are_the_same_position(led):
    """A £100 bet at 2.5 is 250 shares at 0.40 — identical P&L either way."""
    p = led.open_position("A vs B", "A", model_prob=0.6, odds=2.5, stake=100.0)
    assert p.price == pytest.approx(0.40)
    assert p.shares == pytest.approx(250.0)
    assert p.cost == pytest.approx(100.0)
    assert p.to_win == pytest.approx(150.0)   # stake * (odds - 1)


def test_price_odds_round_trip():
    assert price_to_odds(0.40) == pytest.approx(2.5)
    assert odds_to_price(2.5) == pytest.approx(0.40)


# ── selling early ────────────────────────────────────────────────────────────


def test_full_close_realises_the_move(led):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.40, shares=100)
    led.close(p.id, 0.55)
    assert led.stats().capital == pytest.approx(1015.0)   # 100 * 0.15
    assert led.stats().open_count == 0


def test_partial_close_splits_the_position(led):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.40, shares=100)
    sold = led.close(p.id, 0.60, shares=40)

    assert sold.pnl == pytest.approx(8.0)          # 40 * 0.20
    assert led.stats().capital == pytest.approx(1008.0)

    remaining = led.get(p.id)
    assert remaining.is_open
    assert remaining.shares == pytest.approx(60.0)
    assert led.stats().at_risk == pytest.approx(24.0)   # 60 * 0.40


def test_cannot_close_more_than_held(led):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.4, shares=10)
    with pytest.raises(ValueError, match="only holds"):
        led.close(p.id, 0.5, shares=25)


def test_closing_at_entry_is_flat(led):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.40, shares=100)
    led.close(p.id, 0.40)
    assert led.stats().capital == pytest.approx(1000.0)


# ── marking ──────────────────────────────────────────────────────────────────


def test_mark_to_market_moves_equity_not_capital(led):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.40, shares=100)
    led.mark_to_market({p.id: 0.62})
    s = led.stats()
    assert s.capital == pytest.approx(1000.0), "realised capital must not move"
    assert s.unrealised_pnl == pytest.approx(22.0)
    assert s.equity == pytest.approx(1022.0)


def test_unmarked_positions_are_valued_at_entry(led):
    led.open_position("A vs B", "A", model_prob=0.7, price=0.40, shares=100)
    assert led.stats().unrealised_pnl == pytest.approx(0.0)


# ── guards ───────────────────────────────────────────────────────────────────


def test_cannot_spend_more_than_available(led):
    led.open_position("A vs B", "A", model_prob=0.7, price=0.90, shares=1000)  # 900
    with pytest.raises(ValueError, match="exceeds available capital"):
        led.open_position("C vs D", "C", model_prob=0.7, price=0.50, shares=400)


def test_cannot_settle_twice(led):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.5, shares=10)
    led.settle(p.id, "won")
    with pytest.raises(ValueError, match="already won"):
        led.settle(p.id, "lost")


def test_rejects_nonsense(led):
    with pytest.raises(ValueError, match="price"):
        led.open_position("A vs B", "A", model_prob=0.7, price=1.4, shares=10)
    with pytest.raises(ValueError, match="shares"):
        led.open_position("A vs B", "A", model_prob=0.7, price=0.5, shares=0)
    with pytest.raises(ValueError, match="model_prob"):
        led.open_position("A vs B", "A", model_prob=1.7, price=0.5, shares=10)


def test_ledger_survives_reload_and_writes_atomically(tmp_path):
    path = str(tmp_path / "l.json")
    a = Ledger(path, starting_capital=500.0)
    p = a.open_position("A vs B", "A", model_prob=0.7, price=0.4, shares=50)
    a.settle(p.id, "won")

    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []
    b = Ledger(path)
    assert b.starting_capital == 500.0
    assert b.stats().capital == pytest.approx(a.stats().capital)


# ── kelly ────────────────────────────────────────────────────────────────────


def test_kelly_refuses_a_bad_price():
    assert kelly_fraction(0.40, 0.50) == 0.0     # market is more optimistic than us
    assert kelly_fraction(0.50, 0.50) == 0.0     # no edge


def test_kelly_scales_with_edge_and_respects_cap():
    small = kelly_fraction(0.55, 0.50, cap=1.0)
    big = kelly_fraction(0.80, 0.50, cap=1.0)
    assert big > small
    assert kelly_fraction(0.99, 0.10, cap=0.25) <= 0.25


# ── risk ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def rm(tmp_path):
    return RiskManager(RiskLimits(kill_switch_path=str(tmp_path / "STOP")))


def test_risk_requires_edge_and_confidence(led, rm):
    assert not rm.evaluate(led, model_prob=0.62, price=0.60)   # 2% edge
    assert not rm.evaluate(led, model_prob=0.55, price=0.30)   # below confidence
    assert rm.evaluate(led, model_prob=0.70, price=0.60)       # 10% edge, confident


def test_risk_rejects_extreme_prices(led, rm):
    assert not rm.evaluate(led, model_prob=0.99, price=0.95)
    assert not rm.evaluate(led, model_prob=0.70, price=0.05)


def test_risk_caps_position_size(led, rm):
    v = rm.evaluate(led, model_prob=0.95, price=0.30)
    assert v.allowed
    assert v.size <= 1000.0 * rm.limits.max_position_pct + 1e-9


def test_risk_enforces_total_exposure(led, rm):
    led.open_position("A vs B", "A", model_prob=0.7, price=0.30, shares=1000)  # 300
    v = rm.evaluate(led, model_prob=0.90, price=0.40)
    assert not v.allowed
    assert "exposure" in v.reason


def test_risk_blocks_wide_spreads_and_thin_books(led, rm):
    assert not rm.evaluate(led, model_prob=0.8, price=0.5, spread=0.10)
    assert not rm.evaluate(led, model_prob=0.8, price=0.5, depth=5.0)
    assert rm.evaluate(led, model_prob=0.8, price=0.5, spread=0.01, depth=500.0)


def test_risk_refuses_to_double_up_on_one_market(led, rm):
    led.open_position("A vs B", "A", model_prob=0.7, price=0.4, shares=10,
                      token_id="tok123")
    v = rm.evaluate(led, model_prob=0.8, price=0.5, token_id="tok123")
    assert not v.allowed
    assert "already holding" in v.reason


def test_kill_switch_halts_everything(led, rm):
    assert rm.evaluate(led, model_prob=0.9, price=0.4).allowed
    engage_kill_switch(rm.limits.kill_switch_path, "testing")
    v = rm.evaluate(led, model_prob=0.9, price=0.4)
    assert not v.allowed
    assert "KILL SWITCH" in v.reason
    assert release_kill_switch(rm.limits.kill_switch_path)
    assert rm.evaluate(led, model_prob=0.9, price=0.4).allowed


def test_daily_loss_limit_halts_trading(led, rm):
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.90, shares=150)
    led.settle(p.id, "lost")            # -135, over the 10% of 1000 limit
    halted, why = rm.halted(led)
    assert halted
    assert "daily loss limit" in why


def test_exhausted_capital_halts(tmp_path, rm):
    led = Ledger(str(tmp_path / "l.json"), starting_capital=100.0)
    p = led.open_position("A vs B", "A", model_prob=0.7, price=0.5, shares=200)
    led.settle(p.id, "lost")
    halted, why = rm.halted(led)
    assert halted


# ── heartbeat ────────────────────────────────────────────────────────────────


def test_heartbeat_alive_then_stale(tmp_path):
    hb = Heartbeat(str(tmp_path / "hb.json"), stale_after=0.5)
    assert hb.read() is None and not hb.is_alive()
    hb.beat(note="working")
    assert hb.is_alive() and "ALIVE" in hb.describe()
    time.sleep(0.6)
    assert not hb.is_alive() and "STALE" in hb.describe()


def test_heartbeat_survives_corruption(tmp_path):
    path = str(tmp_path / "hb.json")
    with open(path, "w") as f:
        f.write("{ not json")
    hb = Heartbeat(path)
    assert hb.read() is None
    assert not hb.is_alive()


# ── market parsing ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("q,a,b", [
    ("Vitality vs. Spirit", "Vitality", "Spirit"),
    ("Vitality vs Spirit", "Vitality", "Spirit"),
    ("Will Vitality beat Spirit?", "Vitality", "Spirit"),
])
def test_parse_question_finds_both_teams(q, a, b):
    ta, tb, kind, _ = parse_question(q)
    assert (ta, tb) == (a, b)
    assert kind == "match"


def test_parse_question_detects_map_markets():
    ta, tb, kind, mp = parse_question("Vitality vs Spirit - Map 2: Nuke")
    assert (ta, tb) == ("Vitality", "Spirit")
    assert kind == "map"
    assert mp == "Nuke"


def test_unparseable_question_returns_empty():
    ta, tb, _, _ = parse_question("Who wins the Major?")
    assert ta == "" and tb == ""


def test_team_normalisation_is_forgiving_but_not_reckless():
    assert normalise_team("Team Vitality") == normalise_team("VITALITY")
    assert normalise_team("G2 Esports") == normalise_team("G2")
    # Distinct teams must NOT collapse — a false match bets the wrong side.
    assert normalise_team("Spirit") != normalise_team("Spirit Academy")


def test_book_top_spread_and_depth():
    t = BookTop(best_bid=0.48, best_ask=0.52, bid_size=100, ask_size=200)
    assert t.spread == pytest.approx(0.04)
    assert t.mid == pytest.approx(0.50)
    assert t.ask_depth_usd == pytest.approx(104.0)
