#!/usr/bin/env python3
"""
quote_sim.py — paper-test the two-sided quoting strategy the wallet runs.

THE STRATEGY, honestly stated:
  Rest a BUY on Up and a BUY on Down at the same time, priced so the two
  together cost less than $1.00. If BOTH fill, you hold one Up + one Down,
  which is worth exactly $1.00 no matter what Bitcoin does. You merge them
  back to USDC and keep the difference. Riskless.

THE CATCH, which is the entire game:
  Most of the time only ONE side fills — and it's the wrong one. If BTC is
  ripping upward, nobody sells you their Up token; your Down bid is the only
  one that gets hit. So you end up holding the losing side, unhedged.

  That is ADVERSE SELECTION, and it is why market making is hard. The 2.7%
  gross edge means nothing until you subtract what the one-sided fills cost.

  This script measures exactly that. It does not place orders.

    both filled  -> locked profit, riskless
    one filled   -> naked position, mark it to the actual outcome
    neither      -> no trade, no harm

  If (locked profit) < (bleed from one-sided fills), the strategy loses money
  for you even though it makes money for someone faster or better positioned.
"""

import os
import csv
import json
import time
import signal
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ASSETS = ["btc"]                  # one asset — sized for a $50 wallet
TIMEFRAME_SEC = 900              # 900 = 15m markets, 300 = 5m
TARGET_PAIR = 0.97               # quote so Up + Down costs this much
SHARES = 25.0                    # ~$24 per pair, fits $50 with headroom
STARTING_CAPITAL = 50.0          # your real bankroll — the sim respects it
SKEW_LIMIT = 0.25                # skip markets outside 25/75 - see run_market()
QUOTE_AFTER_SEC = 30             # wait this long after open before quoting
STOP_QUOTING_SEC = 90            # stop quoting this long before expiry
SETTLE_CHECK_SEC = 20            # read the winner this long before expiry
POLL_SEC = 1.0

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
LOG_CSV = "quote_sim_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("quote")

_shutdown = False
_csv_lock = threading.Lock()


def _sig(signum, frame):
    global _shutdown
    _shutdown = True
    log.warning("shutdown requested…")


signal.signal(signal.SIGINT, _sig)


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Market:
    asset: str
    slug: str
    end_ts: int
    up_token: str
    down_token: str

    def left(self) -> float:
        return self.end_ts - time.time()


def fmt_dur(s: float) -> str:
    s = int(max(0, s))
    return f"{s//60}m{s%60:02d}s" if s >= 60 else f"{s}s"


def window_start(now: float, dur: int = TIMEFRAME_SEC) -> int:
    return (int(now) // dur) * dur


def slug_for(asset: str, ts: int) -> str:
    tag = "15m" if TIMEFRAME_SEC == 900 else "5m"
    return f"{asset}-updown-{tag}-{ts}"


def fetch_market(asset: str, ts: int, s: requests.Session) -> Optional[Market]:
    slug = slug_for(asset, ts)
    try:
        r = s.get(f"{GAMMA}/events", params={"slug": slug}, timeout=10)
        r.raise_for_status()
        ev = r.json()
    except Exception as e:
        log.error("gamma %s: %s", slug, e)
        return None
    if not ev or not ev[0].get("markets"):
        return None
    m = ev[0]["markets"][0]
    try:
        toks = json.loads(m["clobTokenIds"])
        outs = json.loads(m["outcomes"])
    except Exception:
        return None
    up = down = None
    for lab, t in zip(outs, toks):
        if lab.strip().lower() in ("up", "yes"):
            up = t
        elif lab.strip().lower() in ("down", "no"):
            down = t
    if not (up and down):
        return None
    return Market(asset, slug, ts + TIMEFRAME_SEC, up, down)


def book(token: str, s: requests.Session, at_price: float = None):
    """Return (best_bid, best_ask, queue_ahead).

    queue_ahead = shares already resting at or above `at_price` on the bid
    side. Those orders were there first, so on a real exchange they all get
    filled before you do. The simulator can't jump that queue and neither
    can you.
    """
    try:
        r = s.get(f"{CLOB}/book", params={"token_id": token}, timeout=10)
        r.raise_for_status()
        d = r.json()
    except Exception:
        return None, None, 0.0
    bids = [(float(x["price"]), float(x["size"])) for x in (d.get("bids") or [])]
    asks = [float(x["price"]) for x in (d.get("asks") or [])]
    best_bid = max((p for p, _ in bids), default=None)
    best_ask = min(asks) if asks else None
    queue = sum(sz for p, sz in bids if at_price is not None and p >= at_price)
    return best_bid, best_ask, queue


# ─────────────────────────────────────────────────────────────────────────────

CSV_COLS = ["ts_utc", "asset", "slug", "quote_up", "quote_down", "pair_cost",
            "filled", "winner", "pnl_usd", "bankroll", "note"]


class Ledger:
    """Running bankroll + the fill-rate stats that decide everything."""

    def __init__(self, start: float):
        self.start = start
        self.cash = start
        self.peak = start
        self.stats = {}
        self.real = {}
        self.lock = threading.Lock()

    def can_afford(self, cost: float) -> bool:
        with self.lock:
            return self.cash >= cost

    def record(self, outcome: str, pnl: float, real: str = None) -> float:
        with self.lock:
            self.cash += pnl
            self.peak = max(self.peak, self.cash)
            self.stats[outcome] = self.stats.get(outcome, 0) + 1
            if real:
                self.real[real] = self.real.get(real, 0) + 1
            return self.cash

    @staticmethod
    def _rate(d):
        both = d.get("BOTH", 0)
        one = d.get("UP_ONLY", 0) + d.get("DOWN_ONLY", 0)
        t = both + one
        return (both / t * 100 if t else 0.0), both, t

    def line(self) -> str:
        with self.lock:
            pnl = self.cash - self.start
            pct = pnl / self.start * 100 if self.start else 0
            dd = (self.cash - self.peak) / self.peak * 100 if self.peak else 0
            opt, ob, ot = self._rate(self.stats)
            rea, rb, rt = self._rate(self.real)
            bits = " · ".join(f"{k} {v}" for k, v in sorted(self.stats.items()))
            return (f"\n  💰 BANKROLL ${self.cash:.2f}  ({pnl:+.2f}, {pct:+.1f}%)"
                    f"   peak ${self.peak:.2f}  drawdown {dd:.1f}%\n"
                    f"  📊 {bits}\n"
                    f"  🎯 TWO-SIDED FILL RATE\n"
                    f"       optimistic (ask touched)   {opt:5.1f}%  ({ob}/{ot})\n"
                    f"       realistic  (traded through) {rea:5.1f}%  ({rb}/{rt})  ← trust this one\n"
                    f"       the wallet you found runs ~89%\n")


LEDGER = Ledger(STARTING_CAPITAL)


def log_row(**kw):
    with _csv_lock:
        new = not os.path.exists(LOG_CSV)
        with open(LOG_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if new:
                w.writeheader()
            w.writerow({c: kw.get(c, "") for c in CSV_COLS})


def run_market(m: Market, s: requests.Session):
    # wait for the book to settle after open
    settle_wait = m.left() - (TIMEFRAME_SEC - QUOTE_AFTER_SEC)
    if settle_wait > 1:
        log.info("[%s] window open (%s left) — letting book settle %s",
                 m.asset, fmt_dur(m.left()), fmt_dur(settle_wait))
    while m.left() > TIMEFRAME_SEC - QUOTE_AFTER_SEC and not _shutdown:
        time.sleep(0.5)

    ub, ua, _ = book(m.up_token, s)
    db, da, _ = book(m.down_token, s)
    if ua is None or da is None:
        log.info("[%s] no book — skip", m.asset)
        return

    mid_up = (ub + ua) / 2 if ub else ua
    mid_dn = (db + da) / 2 if db else da
    tot = mid_up + mid_dn
    if tot <= 0:
        return
    fair_up, fair_dn = mid_up / tot, mid_dn / tot

    # Skip lopsided markets. Quoting a 7c token isn't market making, it's
    # buying a longshot — and the cheap side fills precisely because it's
    # dying. That's adverse selection, not edge.
    if not (SKEW_LIMIT <= fair_up <= 1 - SKEW_LIMIT):
        log.info("[%s] skewed %.0f/%.0f — skip", m.asset,
                 fair_up * 100, fair_dn * 100)
        log_row(ts_utc=datetime.now(timezone.utc).isoformat(), asset=m.asset,
                slug=m.slug, filled="SKIP", note=f"skew {fair_up:.2f}")
        return

    # Take the edge EQUALLY in absolute cents from each side. Splitting it
    # proportionally means the cheap side is barely improved on fair, so it
    # fills far too easily while the expensive side never does.
    half = (1.0 - TARGET_PAIR) / 2
    q_up = round(fair_up - half, 2)
    q_dn = round(fair_dn - half, 2)
    if q_up < 0.01 or q_dn < 0.01:
        log.info("[%s] degenerate quote — skip", m.asset)
        return

    need = (q_up + q_dn) * SHARES
    if not LEDGER.can_afford(need):
        log.warning("[%s] need $%.2f, bankroll $%.2f — cannot quote",
                    m.asset, need, LEDGER.cash)
        log_row(ts_utc=datetime.now(timezone.utc).isoformat(), asset=m.asset,
                slug=m.slug, filled="NO_CAPITAL", bankroll=f"{LEDGER.cash:.2f}")
        return

    log.info("[%s] quoting Up $%.2f / Down $%.2f  (pair $%.2f, risking $%.2f)",
             m.asset, q_up, q_dn, q_up + q_dn, need)

    up_filled = dn_filled = False          # OPTIMISTIC: ask merely touched us
    up_real = dn_real = False              # REALISTIC: level traded THROUGH us
    last_beat = time.time()

    # how many shares are already queued ahead of us at our own price?
    _, _, q_ahead_up = book(m.up_token, s, q_up)
    _, _, q_ahead_dn = book(m.down_token, s, q_dn)
    log.info("[%s]   queue ahead of you: UP %.0f sh | DOWN %.0f sh",
             m.asset, q_ahead_up, q_ahead_dn)

    while m.left() > STOP_QUOTING_SEC and not _shutdown:
        if not up_filled or not up_real:
            _, a, _ = book(m.up_token, s, q_up)
            if a is not None and a <= q_up and not up_filled:
                up_filled = True
                log.info("[%s]   ✓ UP touched @ $%.2f", m.asset, q_up)
            if a is not None and a < q_up - 1e-9 and not up_real:
                up_real = True
                log.info("[%s]   ✓✓ UP traded THROUGH @ $%.2f", m.asset, q_up)
        if not dn_filled or not dn_real:
            _, a, _ = book(m.down_token, s, q_dn)
            if a is not None and a <= q_dn and not dn_filled:
                dn_filled = True
                log.info("[%s]   ✓ DOWN touched @ $%.2f", m.asset, q_dn)
            if a is not None and a < q_dn - 1e-9 and not dn_real:
                dn_real = True
                log.info("[%s]   ✓✓ DOWN traded THROUGH @ $%.2f", m.asset, q_dn)
        if up_real and dn_real:
            break
        if time.time() - last_beat >= 60:
            last_beat = time.time()
            got = [n for n, f in (("UP", up_filled), ("DOWN", dn_filled)) if f]
            log.info("[%s]   … %s left, waiting | filled: %s", m.asset,
                     fmt_dur(m.left()), "+".join(got) if got else "nothing yet")
        time.sleep(POLL_SEC)

    # settle: whichever side is trading near $1 late in the window is the winner
    while m.left() > SETTLE_CHECK_SEC and not _shutdown:
        time.sleep(0.5)
    ub = db = None
    for _ in range(6):                      # retry - one empty read isn't fatal
        ub, _x, _ = book(m.up_token, s)
        db, _y, _ = book(m.down_token, s)
        if ub is not None and db is not None:
            break
        time.sleep(1.5)
    if ub is None or db is None:
        winner, note = "?", "no settle book"
    elif ub > db:
        winner, note = "Up", ""
    else:
        winner, note = "Down", ""

    pair = q_up + q_dn
    if up_filled and dn_filled:
        # hedged — the outcome is irrelevant, so this scores either way
        pnl = SHARES * (1.0 - pair)
        filled, note = "BOTH", "riskless"
    elif not up_filled and not dn_filled:
        pnl, filled = 0.0, "NONE"
    elif winner == "?":
        # BUG FIX: a naked position we couldn't settle is UNKNOWN, not a loss.
        # Scoring it 0 silently biased every result negative.
        pnl, filled, note = 0.0, "UNSCORED", "no settle book — excluded"
    elif up_filled:
        pnl = SHARES * ((1.0 if winner == "Up" else 0.0) - q_up)
        filled = "UP_ONLY"
    else:
        pnl = SHARES * ((1.0 if winner == "Down" else 0.0) - q_dn)
        filled = "DOWN_ONLY"

    real_outcome = ("BOTH" if (up_real and dn_real) else
                    "UP_ONLY" if up_real else
                    "DOWN_ONLY" if dn_real else "NONE")

    icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
    bankroll = LEDGER.record(filled, pnl, real_outcome)
    log.info("[%s] %s %s | winner %s | %+.2f USD", m.asset, icon, filled, winner, pnl)
    log.info(LEDGER.line())

    log_row(ts_utc=datetime.now(timezone.utc).isoformat(), asset=m.asset,
            slug=m.slug, quote_up=f"{q_up:.2f}", quote_down=f"{q_dn:.2f}",
            pair_cost=f"{pair:.4f}", filled=filled, winner=winner,
            pnl_usd=f"{pnl:.4f}", bankroll=f"{bankroll:.2f}", note=note)


def worker(asset: str):
    s = requests.Session()
    last = None
    while not _shutdown:
        ts = window_start(time.time())
        if ts != last:
            last = ts
            m = fetch_market(asset, ts, s)
            if m is None:
                log.info("[%s] market not listed yet for this window", asset)
            elif m.left() <= STOP_QUOTING_SEC + 30:
                log.info("[%s] joined too late (only %s left) — waiting for the "
                         "next window", asset, fmt_dur(m.left()))
            else:
                try:
                    run_market(m, s)
                except Exception as e:
                    log.exception("[%s] %s", asset, e)

        nxt = window_start(time.time()) + TIMEFRAME_SEC
        wait = max(0.0, nxt - time.time()) + 2
        if wait > 5:
            log.info("[%s] ⏳ next window opens in %s", asset, fmt_dur(wait))
        slept = 0.0
        while slept < wait and not _shutdown:
            time.sleep(min(1.0, wait - slept))
            slept += 1
            rem = wait - slept
            if slept % 120 == 0 and rem > 45:
                log.info("[%s]   … %s until next window", asset, fmt_dur(rem))


def main():
    log.info("=" * 62)
    log.info("two-sided quoting simulator | 📄 PAPER — no orders sent")
    log.info("target pair cost $%.2f  |  %.0f shares/side  |  %s",
             TARGET_PAIR, SHARES, ", ".join(ASSETS))
    log.info("=" * 62)
    ts = [threading.Thread(target=worker, args=(a,), daemon=True) for a in ASSETS]
    for t in ts:
        t.start()
    try:
        while not _shutdown and any(t.is_alive() for t in ts):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    log.info("stopped → %s", LOG_CSV)


if __name__ == "__main__":
    main()
