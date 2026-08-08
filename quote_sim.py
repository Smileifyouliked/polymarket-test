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

ASSETS = ["btc", "eth"]
TIMEFRAME_SEC = 900              # 900 = 15m markets, 300 = 5m
TARGET_PAIR = 0.97               # quote so Up + Down costs this much
SHARES = 100.0                   # shares quoted per side
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


def book(token: str, s: requests.Session):
    """Return (best_bid, best_ask)."""
    try:
        r = s.get(f"{CLOB}/book", params={"token_id": token}, timeout=10)
        r.raise_for_status()
        d = r.json()
    except Exception:
        return None, None
    bids = [float(x["price"]) for x in (d.get("bids") or [])]
    asks = [float(x["price"]) for x in (d.get("asks") or [])]
    return (max(bids) if bids else None), (min(asks) if asks else None)


# ─────────────────────────────────────────────────────────────────────────────

CSV_COLS = ["ts_utc", "asset", "slug", "quote_up", "quote_down", "pair_cost",
            "filled", "winner", "pnl_usd", "note"]


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
    while m.left() > TIMEFRAME_SEC - QUOTE_AFTER_SEC and not _shutdown:
        time.sleep(0.5)

    ub, ua = book(m.up_token, s)
    db, da = book(m.down_token, s)
    if ua is None or da is None:
        log.info("[%s] no book — skip", m.asset)
        return

    # Split TARGET_PAIR proportionally to where the market thinks fair is.
    mid_up = (ub + ua) / 2 if ub else ua
    mid_dn = (db + da) / 2 if db else da
    tot = mid_up + mid_dn
    if tot <= 0:
        return
    q_up = round(TARGET_PAIR * (mid_up / tot), 2)
    q_dn = round(TARGET_PAIR * (mid_dn / tot), 2)
    if q_up < 0.01 or q_dn < 0.01:
        log.info("[%s] degenerate quote — skip", m.asset)
        return

    log.info("[%s] quoting Up $%.2f / Down $%.2f  (pair $%.2f)",
             m.asset, q_up, q_dn, q_up + q_dn)

    up_filled = dn_filled = False

    while m.left() > STOP_QUOTING_SEC and not _shutdown:
        if not up_filled:
            _, a = book(m.up_token, s)
            if a is not None and a <= q_up:
                up_filled = True
                log.info("[%s]   ✓ UP filled @ $%.2f", m.asset, q_up)
        if not dn_filled:
            _, a = book(m.down_token, s)
            if a is not None and a <= q_dn:
                dn_filled = True
                log.info("[%s]   ✓ DOWN filled @ $%.2f", m.asset, q_dn)
        if up_filled and dn_filled:
            break
        time.sleep(POLL_SEC)

    # settle: whichever side is trading near $1 late in the window is the winner
    while m.left() > SETTLE_CHECK_SEC and not _shutdown:
        time.sleep(0.5)
    ub, _ = book(m.up_token, s)
    db, _ = book(m.down_token, s)
    if ub is None or db is None:
        winner, note = "?", "no settle book"
    elif ub > db:
        winner, note = "Up", ""
    else:
        winner, note = "Down", ""

    pair = q_up + q_dn
    if up_filled and dn_filled:
        pnl = SHARES * (1.0 - pair)
        filled, note = "BOTH", "riskless"
    elif up_filled:
        pnl = SHARES * ((1.0 if winner == "Up" else 0.0) - q_up)
        filled = "UP_ONLY"
    elif dn_filled:
        pnl = SHARES * ((1.0 if winner == "Down" else 0.0) - q_dn)
        filled = "DOWN_ONLY"
    else:
        pnl, filled = 0.0, "NONE"

    icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
    log.info("[%s] %s %s | winner %s | %+.2f USD", m.asset, icon, filled, winner, pnl)

    log_row(ts_utc=datetime.now(timezone.utc).isoformat(), asset=m.asset,
            slug=m.slug, quote_up=f"{q_up:.2f}", quote_down=f"{q_dn:.2f}",
            pair_cost=f"{pair:.4f}", filled=filled, winner=winner,
            pnl_usd=f"{pnl:.4f}", note=note)


def worker(asset: str):
    s = requests.Session()
    last = None
    while not _shutdown:
        ts = window_start(time.time())
        if ts != last:
            last = ts
            m = fetch_market(asset, ts, s)
            if m and m.left() > STOP_QUOTING_SEC + 30:
                try:
                    run_market(m, s)
                except Exception as e:
                    log.exception("[%s] %s", asset, e)
        nxt = window_start(time.time()) + TIMEFRAME_SEC
        wait = max(0.0, nxt - time.time()) + 2
        slept = 0.0
        while slept < wait and not _shutdown:
            time.sleep(min(1.0, wait - slept))
            slept += 1


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
