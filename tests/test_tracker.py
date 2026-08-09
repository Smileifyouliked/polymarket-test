"""Tests for the bankroll ledger and heartbeat."""

import json
import os
import time

import pytest

from cs2model.tracker import Heartbeat, Ledger, kelly_fraction


@pytest.fixture
def led(tmp_path):
    return Ledger(str(tmp_path / "ledger.json"), starting_capital=1000.0)


def test_win_and_loss_move_capital_correctly(led):
    p = led.open_position("A", "B", "A", 0.60, 2.00, 100.0)
    assert led.stats().capital == 1000.0, "capital only moves on settlement"
    assert led.stats().at_risk == 100.0

    led.settle(p.id, "won")
    assert led.stats().capital == pytest.approx(1100.0)

    q = led.open_position("C", "D", "C", 0.55, 1.50, 200.0)
    led.settle(q.id, "lost")
    assert led.stats().capital == pytest.approx(900.0)


def test_void_returns_the_stake(led):
    p = led.open_position("A", "B", "A", 0.6, 2.0, 100.0)
    led.settle(p.id, "void")
    assert led.stats().capital == pytest.approx(1000.0)
    assert led.stats().voids == 1


def test_cannot_stake_more_than_is_available(led):
    led.open_position("A", "B", "A", 0.6, 2.0, 900.0)
    with pytest.raises(ValueError, match="exceeds available capital"):
        led.open_position("C", "D", "C", 0.6, 2.0, 200.0)


def test_cannot_settle_twice(led):
    p = led.open_position("A", "B", "A", 0.6, 2.0, 100.0)
    led.settle(p.id, "won")
    with pytest.raises(ValueError, match="already won"):
        led.settle(p.id, "lost")


def test_rejects_nonsense_input(led):
    with pytest.raises(ValueError, match="must be either"):
        led.open_position("A", "B", "Z", 0.6, 2.0, 10.0)
    with pytest.raises(ValueError, match="odds"):
        led.open_position("A", "B", "A", 0.6, 0.9, 10.0)
    with pytest.raises(ValueError, match="stake"):
        led.open_position("A", "B", "A", 0.6, 2.0, -5.0)
    with pytest.raises(ValueError, match="model_prob"):
        led.open_position("A", "B", "A", 1.4, 2.0, 10.0)


def test_ledger_survives_a_reload(tmp_path):
    path = str(tmp_path / "l.json")
    a = Ledger(path, starting_capital=500.0)
    p = a.open_position("A", "B", "A", 0.7, 1.8, 50.0)
    a.settle(p.id, "won")

    b = Ledger(path)
    assert b.starting_capital == 500.0
    assert b.stats().capital == pytest.approx(a.stats().capital)
    assert len(b.positions) == 1


def test_edge_and_implied_probability(led):
    p = led.open_position("A", "B", "A", 0.60, 2.50, 10.0)
    assert p.implied_prob == pytest.approx(0.40)
    assert p.edge == pytest.approx(0.20)
    assert p.to_win == pytest.approx(15.0)


def test_drawdown_tracks_the_peak(led):
    a = led.open_position("A", "B", "A", 0.6, 3.0, 100.0)
    led.settle(a.id, "won")                     # +200 -> 1200
    b = led.open_position("C", "D", "C", 0.6, 2.0, 300.0)
    led.settle(b.id, "lost")                    # -300 -> 900
    s = led.stats()
    assert s.peak_capital == pytest.approx(1200.0)
    assert s.capital == pytest.approx(900.0)
    assert s.drawdown == pytest.approx(300.0)


def test_kelly_declines_negative_edge():
    # 40% chance at even money is a losing bet; Kelly must refuse it.
    assert kelly_fraction(0.40, 2.00) == 0.0
    # 60% at even money is a real edge, and the cap binds.
    assert kelly_fraction(0.60, 2.00, cap=0.25) == pytest.approx(0.05)


def test_kelly_never_exceeds_the_cap():
    assert kelly_fraction(0.99, 10.0, cap=0.25) <= 0.25


def test_heartbeat_reports_alive_then_stale(tmp_path):
    hb = Heartbeat(str(tmp_path / "hb.json"), stale_after=0.5)
    assert hb.read() is None
    assert not hb.is_alive()
    assert "NEVER STARTED" in hb.describe()

    hb.beat(note="working")
    assert hb.is_alive()
    assert "ALIVE" in hb.describe()

    time.sleep(0.6)
    assert not hb.is_alive()
    assert "STALE" in hb.describe()


def test_heartbeat_survives_a_corrupt_file(tmp_path):
    path = str(tmp_path / "hb.json")
    with open(path, "w") as f:
        f.write("{ this is not json")
    hb = Heartbeat(path)
    assert hb.read() is None       # must not raise
    assert not hb.is_alive()


def test_ledger_write_is_atomic(tmp_path):
    """A crash mid-write must not destroy position history."""
    path = str(tmp_path / "l.json")
    led = Ledger(path, starting_capital=100.0)
    led.open_position("A", "B", "A", 0.6, 2.0, 10.0)
    # No stray temp files left behind, and the file is valid JSON.
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []
    with open(path) as f:
        json.load(f)
