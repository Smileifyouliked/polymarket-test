"""
cli.py — the five things you actually want to run.

  python -m cs2model.cli demo                    # end-to-end on synthetic data
  python -m cs2model.cli explore --limit 3       # see raw Liquipedia fields
  python -m cs2model.cli ingest --out data/matches.json --pages 40
  python -m cs2model.cli evaluate --data data/matches.json
  python -m cs2model.cli predict --data data/matches.json \
        --team-a "Vitality" --team-b "Spirit" --bo 3 --lan
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .data import ACTIVE_DUTY, Match, generate_synthetic_league, load_matches, save_matches
from .evaluate import report, walk_forward
from .model import CS2Model, build_dataset
from .ratings import RatingBook
from .veto import match_win_prob, veto_report


def cmd_demo(args) -> int:
    print("generating a synthetic CS2 scene with known ground truth...")
    matches = generate_synthetic_league(
        n_teams=args.teams, n_matches=args.matches, seed=args.seed
    )
    print(f"  {len(matches)} matches, {args.teams} teams")

    print("walking the history forward and extracting leak-free features...")
    ds = build_dataset(matches, seed=args.seed)
    print(f"  {len(ds.y)} usable matches, {ds.X.shape[1]} features")

    res = walk_forward(ds, initial_frac=0.5, n_folds=args.folds)

    # Synthetic data knows the truth, so we can check the model against it.
    # Index via res.idx — undersized folds are skipped, so the out-of-sample
    # set is not simply the tail of the dataset.
    true_p = np.array([ds.matches[i].true_p_a for i in res.idx])
    print(report(res, true_p=true_p))

    model = CS2Model().fit(ds.X, ds.y)
    print("feature weights (standardised, so magnitudes are comparable):")
    for k, v in sorted(model.coefficients().items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:<18} {v:+.3f}")
    print()
    return 0


def cmd_explore(args) -> int:
    from .liquipedia import fetch_raw, resolve_base

    base = resolve_base(datapoint=args.datapoint, wiki=args.wiki)
    print(f"API base URL that answered: {base}")
    print("  (if this differs from the default, export LIQUIPEDIA_API_BASE="
          f"{base} to skip probing next time)\n")

    rows = fetch_raw(
        datapoint=args.datapoint,
        wiki=args.wiki,
        limit=args.limit,
        conditions=args.conditions,
    )
    print(f"{len(rows)} raw records. Top-level keys of the first:\n")
    if rows:
        for k in sorted(rows[0]):
            v = rows[0][k]
            preview = str(v)
            if len(preview) > 90:
                preview = preview[:90] + "..."
            print(f"  {k:<28} {preview}")
        print("\nfull first record:\n")
        print(json.dumps(rows[0], indent=2)[:4000])

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nsaved {len(rows)} raw records -> {args.save}")
        print("This file contains no credentials — it is safe to share.")
    return 0


def cmd_ingest(args) -> int:
    from .liquipedia import ingest

    matches = ingest(pages=args.pages, per_page=args.per_page)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_matches(matches, args.out)
    print(f"wrote {len(matches)} matches -> {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    matches = load_matches(args.data)
    print(f"loaded {len(matches)} matches from {args.data}")
    ds = build_dataset(matches)
    res = walk_forward(ds, initial_frac=args.initial_frac, n_folds=args.folds)
    print(report(res))
    return 0


def cmd_predict(args) -> int:
    matches = load_matches(args.data)
    ds = build_dataset(matches)
    model = CS2Model().fit(ds.X, ds.y)

    # Rebuild the book over the full history so ratings are current.
    book = RatingBook()
    for m in sorted(matches, key=lambda m: m.date):
        if m.winner and m.maps:
            book.observe(m)

    known = set(book.teams)
    for t in (args.team_a, args.team_b):
        if t not in known:
            print(f"error: unknown team {t!r}. Known teams look like: "
                  f"{sorted(known)[:8]}", file=sys.stderr)
            return 2

    fixture = Match(
        team_a=args.team_a,
        team_b=args.team_b,
        date=datetime.now(timezone.utc),
        best_of=args.bo,
        lan=args.lan,
        tier=args.tier,
        lineup_a=book.team(args.team_a).lineup,
        lineup_b=book.team(args.team_b).lineup,
    )

    from .model import build_features

    X = build_features(book, fixture).reshape(1, -1)
    p = float(model.predict_proba(X)[0])

    def map_prob(m: str) -> float:
        return book.map_win_prob(args.team_a, args.team_b, m, fixture.date)

    print()
    print(f"  {args.team_a}  vs  {args.team_b}   (Bo{args.bo}, {'LAN' if args.lan else 'online'})")
    print("  " + "-" * 60)
    print(f"  P({args.team_a} wins) = {p:.1%}")
    print()
    print("  per-map probabilities (before the veto):")
    for m in ACTIVE_DUTY:
        mp = map_prob(m)
        bar = "#" * int(round(mp * 30))
        print(f"    {m:<10} {mp:5.1%}  {bar}")
    print()
    print("  how often each map reaches the server, per the veto simulation:")
    for m, freq in veto_report(map_prob, args.bo, n_sims=1500).items():
        if freq > 0.01:
            print(f"    {m:<10} {freq:5.1%}")
    print()

    conf = max(p, 1 - p)
    if conf < args.threshold:
        print(f"  VERDICT: NO CALL. Confidence {conf:.1%} is below your "
              f"{args.threshold:.0%} bar.")
        print("           This is the model working, not failing.")
    else:
        pick = args.team_a if p > 0.5 else args.team_b
        print(f"  VERDICT: {pick}  (confidence {conf:.1%})")
    print()
    return 0


def _load_engine(args):
    """Rebuild ratings + model from the match history on disk."""
    matches = load_matches(args.data)
    ds = build_dataset(matches)
    model = CS2Model().fit(ds.X, ds.y)
    book = RatingBook()
    for m in sorted(matches, key=lambda x: x.date):
        if m.winner and m.maps:
            book.observe(m)
    return book, model


def _risk_from_args(args):
    from .risk import RiskLimits, RiskManager

    return RiskManager(RiskLimits(
        min_edge=args.min_edge,
        min_confidence=args.min_confidence,
        kelly_cap=args.kelly_cap,
        max_position_pct=args.max_position_pct,
        max_total_exposure_pct=args.max_exposure_pct,
        max_open_positions=args.max_positions,
        daily_loss_limit_pct=args.daily_loss_pct,
        max_spread=args.max_spread,
        min_book_depth=args.min_depth,
        kill_switch_path=args.kill_switch,
    ))


def cmd_markets(args) -> int:
    from .polymarket import PolymarketVenue

    venue = PolymarketVenue(dry_run=True)
    try:
        markets = venue.fetch_markets(slug_contains=args.slug, limit=args.limit)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"{len(markets)} markets\n")
    for m in markets:
        parsed = (f"{m.team_a} vs {m.team_b}" if m.team_a else "UNPARSED")
        kind = f"{m.map_name} map" if m.market_kind == "map" else "match"
        prices = "/".join(f"{o.name[:10]} {o.price:.2f}" for o in m.outcomes[:2])
        print(f"  {m.question[:56]:<56} [{kind:<10}] {parsed[:30]:<30} {prices}")
    if args.raw:
        os.makedirs(os.path.dirname(args.raw) or ".", exist_ok=True)
        with open(args.raw, "w") as f:
            json.dump([m.__dict__ for m in markets], f, indent=2, default=str)
        print(f"\nsaved -> {args.raw}")
    return 0


def cmd_bet(args) -> int:
    """Manual position entry. The bot does this itself; this is for overrides."""
    from .tracker import Ledger, kelly_fraction

    led = Ledger(args.ledger, starting_capital=args.capital)
    if not led.positions and args.capital > 0:
        led.starting_capital = args.capital

    price, shares = args.price, args.shares
    if args.odds and args.stake:
        price, shares = 0.0, 0.0
    if price and not shares:
        avail = led.stats().capital
        frac = kelly_fraction(args.prob, price, cap=args.kelly_cap)
        cost = round(avail * frac, 2)
        if cost <= 0:
            print("Kelly says no: the price is at or above the model's "
                  "probability, so there is no edge.", file=sys.stderr)
            return 1
        shares = round(cost / price, 2)
        print(f"no --shares given; {args.kelly_cap:.0%} Kelly suggests "
              f"{shares:,.2f} shares (${cost:,.2f})")

    try:
        p = led.open_position(
            market=args.market, outcome=args.outcome, model_prob=args.prob,
            price=price or 0.0, shares=shares or 0.0,
            odds=args.odds or 0.0, stake=args.stake or 0.0,
            market_kind="map" if args.map else "match", map_name=args.map,
            best_of=args.bo, event=args.event, note=args.note, dry_run=True,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"opened {p.id}: {p.outcome} — {p.shares:,.2f} shares @ {p.price:.3f} "
          f"= ${p.cost:,.2f} (to win ${p.to_win:,.2f}, edge {p.edge:+.1%})")
    return 0


def cmd_settle(args) -> int:
    from .tracker import Ledger

    led = Ledger(args.ledger)
    try:
        p = led.settle(args.id, args.result)
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"settled {p.id} as {p.status}: P&L {p.pnl:+,.2f} "
          f"— capital now {led.stats().capital:,.2f}")
    return 0


def cmd_close(args) -> int:
    from .tracker import Ledger

    led = Ledger(args.ledger)
    try:
        p = led.close(args.id, args.price, shares=args.shares)
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"closed {p.shares:,.2f} shares of {p.outcome} at {p.exit_price:.3f}: "
          f"P&L {p.pnl:+,.2f} — capital now {led.stats().capital:,.2f}")
    return 0


def cmd_stop(args) -> int:
    from .risk import engage_kill_switch

    engage_kill_switch(args.kill_switch, reason=args.reason)
    print(f"KILL SWITCH ENGAGED -> {args.kill_switch}")
    print("The bot will place no further orders. Open positions are untouched.")
    print(f"Resume with: python -m cs2model.cli resume")
    return 0


def cmd_resume(args) -> int:
    from .risk import release_kill_switch

    if release_kill_switch(args.kill_switch):
        print("kill switch released — the bot may trade again")
    else:
        print("kill switch was not engaged")
    return 0


def cmd_dashboard(args) -> int:
    from .dashboard import render, render_compact
    from .tracker import Heartbeat, Ledger

    hb = Heartbeat(args.heartbeat, stale_after=args.stale_after)

    def once() -> str:
        led = Ledger(args.ledger, starting_capital=args.capital)
        if args.compact:
            return render_compact(led, hb)
        from .risk import RiskLimits, RiskManager
        rm = RiskManager(RiskLimits(kill_switch_path=args.kill_switch))
        stop, why = rm.halted(led)
        return render(led, hb, mode=args.mode, risk_note=why if stop else "")

    if not args.watch:
        print(once())
        return 0
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(once())
            print(f"\n  refreshing every {args.watch}s — Ctrl+C to exit")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


def cmd_run(args) -> int:
    """
    The bot. Discover -> price -> risk -> trade -> mark -> settle, on a loop.

    DRY RUN UNLESS --live IS PASSED. Dry run does everything except sign the
    order, so you can watch exactly what it would have done, for as long as
    you like, before any money is involved.
    """
    from .dashboard import render_compact
    from .polymarket import PolymarketVenue
    from .strategy import Strategy
    from .tracker import Heartbeat, Ledger

    risk = _risk_from_args(args)
    venue = PolymarketVenue(dry_run=not args.live)
    hb = Heartbeat(args.heartbeat, stale_after=max(args.interval * 3, 120))

    if args.live:
        print("=" * 66)
        print("  LIVE TRADING — REAL MONEY")
        print("=" * 66)
        if not os.getenv("POLYMARKET_PRIVATE_KEY"):
            print("POLYMARKET_PRIVATE_KEY is not set; cannot sign orders.",
                  file=sys.stderr)
            return 2
    else:
        print("DRY RUN — no orders will be sent. Add --live to trade for real.")

    # A brand-new ledger with 0 capital halts instantly on "capital
    # exhausted", which looks exactly like a crash. Fail loudly up front.
    if not os.path.exists(args.ledger) and args.capital <= 0:
        print(f"error: {args.ledger} does not exist and --capital was not set.\n"
              f"       Start the bot with your bankroll, e.g. --capital 500",
              file=sys.stderr)
        return 2

    print(f"loading ratings and model from {args.data} ...")
    try:
        book, model = _load_engine(args)
    except FileNotFoundError:
        print(f"error: {args.data} not found. Run `ingest` first.", file=sys.stderr)
        return 2
    print(f"  {len(book.teams)} teams known")
    print(f"heartbeat -> {args.heartbeat} every {args.interval}s · Ctrl+C to stop")
    print(f"kill switch: touch {args.kill_switch}\n")

    try:
        while True:
            led = Ledger(args.ledger, starting_capital=args.capital)
            strat = Strategy(book, model, led, venue, risk)

            stop, why = risk.halted(led)
            if stop:
                hb.beat(status="halted", note=why)
                print(f"{datetime.now(timezone.utc):%H:%M:%S}  HALTED — {why}")
            else:
                try:
                    markets = venue.fetch_markets(slug_contains=args.slug,
                                                  limit=args.limit)
                except Exception as e:
                    hb.beat(status="degraded", note=f"market fetch failed: {e}")
                    print(f"{datetime.now(timezone.utc):%H:%M:%S}  "
                          f"market fetch failed: {e}")
                    if args.once:
                        return 1
                    time.sleep(args.interval)
                    continue

                decisions = strat.run_once(markets)
                took = [d for d in decisions if d.taken]
                for d in decisions if args.verbose else took:
                    print(d.line())

                strat.mark_open_positions()

                # Settlement must look at CLOSED markets. The trading fetch
                # asks for closed=False, so resolved markets never appeared
                # there and settle_resolved() could never fire — positions
                # accumulated forever, realised P&L stayed 0, and the
                # daily-loss and drawdown halts could never trigger.
                if led.open_positions():
                    try:
                        resolved = venue.fetch_markets(
                            slug_contains=args.slug, limit=args.limit, closed=True
                        )
                        for line in strat.settle_resolved(resolved):
                            print(f"  settled  {line}")
                    except Exception as e:
                        print(f"  settlement check failed: {e}")

                led = Ledger(args.ledger, starting_capital=args.capital)
                hb.beat(status="ok", note=render_compact(led, None))
                print(f"{datetime.now(timezone.utc):%H:%M:%S}  "
                      f"{len(markets)} markets · {len(took)} taken · "
                      f"{render_compact(led, None)}")

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        hb.beat(status="stopped", note="stopped by operator")
        print("\nstopped cleanly")
        return 0
    except Exception as e:
        hb.beat(status="crashed", note=f"{type(e).__name__}: {e}")
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cs2model", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="end-to-end run on synthetic data")
    d.add_argument("--teams", type=int, default=40)
    d.add_argument("--matches", type=int, default=4000)
    d.add_argument("--folds", type=int, default=8)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("explore", help="dump raw Liquipedia records")
    e.add_argument("--datapoint", default="match")
    e.add_argument("--limit", type=int, default=3)
    e.add_argument("--wiki", default="counterstrike")
    e.add_argument("--conditions", default="",
                   help="Liquipedia filter, e.g. '[[game::cs2]]' or "
                        "'[[winner::]]' for unplayed (upcoming) matches")
    e.add_argument("--save", default="",
                   help="write the raw records to this file (contains no "
                        "credentials, safe to share)")
    e.set_defaults(func=cmd_explore)

    i = sub.add_parser("ingest", help="fetch matches from Liquipedia")
    i.add_argument("--out", default="data/matches.json")
    i.add_argument("--pages", type=int, default=20)
    i.add_argument("--per-page", type=int, default=100)
    i.set_defaults(func=cmd_ingest)

    v = sub.add_parser("evaluate", help="walk-forward evaluation on saved matches")
    v.add_argument("--data", default="data/matches.json")
    v.add_argument("--folds", type=int, default=8)
    v.add_argument("--initial-frac", type=float, default=0.5)
    v.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict one fixture")
    p.add_argument("--data", default="data/matches.json")
    p.add_argument("--team-a", required=True)
    p.add_argument("--team-b", required=True)
    p.add_argument("--bo", type=int, default=3, choices=[1, 3, 5])
    p.add_argument("--lan", action="store_true")
    p.add_argument("--tier", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.72,
                   help="below this confidence the model declines to call it")
    p.set_defaults(func=cmd_predict)

    # ── tracking & trading ───────────────────────────────────────────────────
    def _ledger_args(sp):
        sp.add_argument("--ledger", default="data/ledger.json")
        sp.add_argument("--capital", type=float, default=0.0,
                        help="starting bankroll; only used when creating a new ledger")
        sp.add_argument("--kill-switch", default="data/STOP")

    def _risk_args(sp):
        sp.add_argument("--min-edge", type=float, default=0.05)
        sp.add_argument("--min-confidence", type=float, default=0.60)
        sp.add_argument("--kelly-cap", type=float, default=0.25)
        sp.add_argument("--max-position-pct", type=float, default=0.05)
        sp.add_argument("--max-exposure-pct", type=float, default=0.30)
        sp.add_argument("--max-positions", type=int, default=10)
        sp.add_argument("--daily-loss-pct", type=float, default=0.10)
        sp.add_argument("--max-spread", type=float, default=0.04)
        sp.add_argument("--min-depth", type=float, default=50.0)

    mk = sub.add_parser("markets", help="list CS2 markets on Polymarket")
    mk.add_argument("--slug", default="cs2")
    mk.add_argument("--limit", type=int, default=100)
    mk.add_argument("--raw", default="", help="save parsed markets to this file")
    mk.set_defaults(func=cmd_markets)

    b = sub.add_parser("bet", help="record a position by hand")
    _ledger_args(b)
    b.add_argument("--market", required=True, help='e.g. "Vitality vs Spirit"')
    b.add_argument("--outcome", required=True, help="the side you bought")
    b.add_argument("--prob", type=float, required=True,
                   help="model probability for that side (0-1)")
    b.add_argument("--price", type=float, default=0.0, help="Polymarket price 0-1")
    b.add_argument("--shares", type=float, default=0.0,
                   help="omit for a capped-Kelly suggestion")
    b.add_argument("--odds", type=float, default=0.0, help="sportsbook decimal odds")
    b.add_argument("--stake", type=float, default=0.0, help="sportsbook stake")
    b.add_argument("--kelly-cap", type=float, default=0.25)
    b.add_argument("--map", default="", help="map name for a map market")
    b.add_argument("--bo", type=int, default=3, choices=[1, 3, 5])
    b.add_argument("--event", default="")
    b.add_argument("--note", default="")
    b.set_defaults(func=cmd_bet)

    stl = sub.add_parser("settle", help="resolve a position at 1 or 0")
    _ledger_args(stl)
    stl.add_argument("--id", required=True)
    stl.add_argument("--result", required=True, choices=["won", "lost", "void"])
    stl.set_defaults(func=cmd_settle)

    cl = sub.add_parser("close", help="sell before resolution (full or partial)")
    _ledger_args(cl)
    cl.add_argument("--id", required=True)
    cl.add_argument("--price", type=float, required=True, help="exit price 0-1")
    cl.add_argument("--shares", type=float, default=None,
                    help="omit to close the whole position")
    cl.set_defaults(func=cmd_close)

    sp_stop = sub.add_parser("stop", help="engage the kill switch")
    sp_stop.add_argument("--kill-switch", default="data/STOP")
    sp_stop.add_argument("--reason", default="manual stop")
    sp_stop.set_defaults(func=cmd_stop)

    sp_res = sub.add_parser("resume", help="release the kill switch")
    sp_res.add_argument("--kill-switch", default="data/STOP")
    sp_res.set_defaults(func=cmd_resume)

    db = sub.add_parser("dashboard", help="capital, P&L, positions, health")
    _ledger_args(db)
    db.add_argument("--heartbeat", default="data/heartbeat.json")
    db.add_argument("--stale-after", type=float, default=600.0)
    db.add_argument("--watch", type=float, default=0.0, metavar="SECONDS")
    db.add_argument("--compact", action="store_true")
    db.add_argument("--mode", default="")
    db.set_defaults(func=cmd_dashboard)

    rn = sub.add_parser("run", help="the bot: discover, price, risk, trade")
    _ledger_args(rn)
    _risk_args(rn)
    rn.add_argument("--data", default="data/matches.json")
    rn.add_argument("--heartbeat", default="data/heartbeat.json")
    rn.add_argument("--interval", type=float, default=300.0)
    rn.add_argument("--slug", default="cs2")
    rn.add_argument("--limit", type=int, default=100)
    rn.add_argument("--once", action="store_true", help="one pass then exit")
    rn.add_argument("--verbose", action="store_true",
                    help="show skipped markets and why")
    rn.add_argument("--live", action="store_true",
                    help="REAL MONEY. Without this it is a dry run.")
    rn.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
