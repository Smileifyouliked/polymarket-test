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


def cmd_bet(args) -> int:
    from .tracker import Ledger, kelly_fraction

    led = Ledger(args.ledger, starting_capital=args.capital)
    if not led.positions and args.capital > 0:
        led.starting_capital = args.capital

    stake = args.stake
    if stake is None:
        # No stake given: suggest capped Kelly rather than making one up.
        avail = led.stats().capital
        frac = kelly_fraction(args.prob, args.odds, cap=args.kelly_cap)
        stake = round(avail * frac, 2)
        print(f"no --stake given; {args.kelly_cap:.0%} Kelly on {avail:,.2f} "
              f"suggests {stake:,.2f}")
        if stake <= 0:
            print("Kelly says do not take this bet — the edge is negative.",
                  file=sys.stderr)
            return 1

    try:
        p = led.open_position(
            team_a=args.team_a, team_b=args.team_b, pick=args.pick,
            model_prob=args.prob, odds=args.odds, stake=stake,
            best_of=args.bo, event=args.event, note=args.note,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"opened {p.id}: {p.pick} @ {p.odds:.2f} for {p.stake:,.2f} "
          f"(to win {p.to_win:,.2f}, edge {p.edge:+.1%})")
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


def cmd_dashboard(args) -> int:
    from .dashboard import render, render_compact
    from .tracker import Heartbeat, Ledger

    hb = Heartbeat(args.heartbeat, stale_after=args.stale_after)

    def once() -> str:
        led = Ledger(args.ledger, starting_capital=args.capital)
        return render_compact(led, hb) if args.compact else render(led, hb)

    if not args.watch:
        print(once())
        return 0

    try:
        while True:
            # Re-read from disk each tick so a running `run` loop or another
            # SSH session settling a bet shows up here immediately.
            os.system("clear" if os.name != "nt" else "cls")
            print(once())
            print(f"\n  refreshing every {args.watch}s — Ctrl+C to exit")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


def cmd_run(args) -> int:
    """
    The long-running process. Beats, then does its periodic work.

    Keep it in tmux on a server so it survives your SSH session dropping:
        tmux new -s cs2
        python -m cs2model.cli run
        Ctrl+B then D to detach
    """
    from .tracker import Heartbeat, Ledger

    hb = Heartbeat(args.heartbeat, stale_after=max(args.interval * 3, 60))
    print(f"starting run loop: heartbeat -> {args.heartbeat} every {args.interval}s")
    print("Ctrl+C to stop")

    try:
        while True:
            led = Ledger(args.ledger, starting_capital=args.capital)
            s = led.stats()
            note = (f"capital {s.capital:,.2f} · {s.open_count} open · "
                    f"{s.wins}W-{s.losses}L")
            hb.beat(status="ok", note=note)
            print(f"{datetime.now(timezone.utc):%H:%M:%S}  beat · {note}")

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        hb.beat(status="stopped", note="stopped by operator")
        print("\nstopped cleanly — heartbeat marked as stopped")
        return 0
    except Exception as e:  # keep the loop's death visible in the heartbeat
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

    # ── tracking ─────────────────────────────────────────────────────────────
    def _ledger_args(sp):
        sp.add_argument("--ledger", default="data/ledger.json")
        sp.add_argument("--capital", type=float, default=0.0,
                        help="starting bankroll; only used when creating a new ledger")

    b = sub.add_parser("bet", help="record a position you are taking")
    _ledger_args(b)
    b.add_argument("--team-a", required=True)
    b.add_argument("--team-b", required=True)
    b.add_argument("--pick", required=True, help="which team you are backing")
    b.add_argument("--prob", type=float, required=True,
                   help="model probability for your PICK (0-1)")
    b.add_argument("--odds", type=float, required=True, help="decimal odds, e.g. 1.85")
    b.add_argument("--stake", type=float, default=None,
                   help="omit to get a capped-Kelly suggestion")
    b.add_argument("--kelly-cap", type=float, default=0.25)
    b.add_argument("--bo", type=int, default=3, choices=[1, 3, 5])
    b.add_argument("--event", default="")
    b.add_argument("--note", default="")
    b.set_defaults(func=cmd_bet)

    st = sub.add_parser("settle", help="resolve an open position")
    _ledger_args(st)
    st.add_argument("--id", required=True)
    st.add_argument("--result", required=True, choices=["won", "lost", "void"])
    st.set_defaults(func=cmd_settle)

    db = sub.add_parser("dashboard", help="capital, P&L, open positions, health")
    _ledger_args(db)
    db.add_argument("--heartbeat", default="data/heartbeat.json")
    db.add_argument("--stale-after", type=float, default=180.0)
    db.add_argument("--watch", type=float, default=0.0,
                    metavar="SECONDS", help="refresh continuously")
    db.add_argument("--compact", action="store_true", help="single-line output")
    db.set_defaults(func=cmd_dashboard)

    rn = sub.add_parser("run", help="long-running loop that emits the heartbeat")
    _ledger_args(rn)
    rn.add_argument("--heartbeat", default="data/heartbeat.json")
    rn.add_argument("--interval", type=float, default=60.0)
    rn.add_argument("--once", action="store_true", help="single beat then exit")
    rn.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
