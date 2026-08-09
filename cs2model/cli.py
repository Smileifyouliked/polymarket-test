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

    base = resolve_base(datapoint=args.datapoint)
    print(f"API base URL that answered: {base}")
    print("  (if this differs from the default, export LIQUIPEDIA_API_BASE="
          f"{base} to skip probing next time)\n")

    rows = fetch_raw(datapoint=args.datapoint, limit=args.limit)
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

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
