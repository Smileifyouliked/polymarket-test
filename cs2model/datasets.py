"""
datasets.py — load real match history from a CSV.

WHY THIS EXISTS
  Every number in this project so far came from a synthetic league I wrote
  myself, which proves the maths and proves nothing about Counter-Strike. A
  CSV of real results answers the only question that matters: does the model
  beat "pick the favourite" on games that actually happened?

  A static file is also the one data path I can verify end to end — no API
  key, no rate limit, no auth, no silent schema drift. Rerun it tomorrow and
  you get the same answer.

WHAT IT HANDLES
  Public CS:GO/CS2 result dumps are messy in consistent ways:
    * rows are single MAPS, not series, so they need grouping
    * duplicate column headers (two columns both called "PTS")
    * ambiguous dates (is 12/11/15 November or December?)
    * a map pool from whichever era the data covers
  Each of those is a wrong answer waiting to happen, so each is handled
  explicitly rather than assumed away.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .data import MapResult, Match, set_active_pool

UTC = timezone.utc

# Anything below this share of rows is a typo or a one-off exhibition map, not
# part of the pool. Keeping it would give the veto simulator a map nobody plays.
MIN_MAP_SHARE = 0.005


def _clean_map(name: str) -> str:
    n = str(name).strip().lower()
    n = re.sub(r"^(de|cs)[_\s]+", "", n)
    return n.title()


def _parse_date(raw: str, dayfirst: bool) -> Optional[datetime]:
    s = str(raw).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    parts = re.split(r"[/\-.]", s)
    if len(parts) != 3:
        return None
    try:
        a, b, c = (int(p) for p in parts)
    except ValueError:
        return None
    year = c if c > 99 else (2000 + c if c < 70 else 1900 + c)
    day, month = (a, b) if dayfirst else (b, a)
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _detect_dayfirst(raw_dates: Sequence[str]) -> bool:
    """
    Is 12/11/15 the 12th of November or the 11th of December?

    Decide from the data: if any first field exceeds 12 it cannot be a month,
    so the format is day-first. Same test on the second field. If neither is
    decisive the choice does not change the ordering much, and day-first is the
    more common convention in European esports listings.
    """
    first_over_12 = second_over_12 = False
    for s in raw_dates:
        parts = re.split(r"[/\-.]", str(s).strip())
        if len(parts) != 3:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if a > 12:
            first_over_12 = True
        if b > 12:
            second_over_12 = True
    if first_over_12 and not second_over_12:
        return True
    if second_over_12 and not first_over_12:
        return False
    return True


def _rows_with_positions(path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Read positionally, not by column name.

    csv.DictReader silently drops all but the last of duplicate headers, and
    these dumps really do have two columns called "PTS" — one score per team.
    Reading by name loses one of them, which is how a winner ends up recorded
    with the loser's score.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return [h.strip() for h in rows[0]], [r for r in rows[1:] if any(x.strip() for x in r)]


def _find(headers: Sequence[str], *candidates: str) -> List[int]:
    """Indices of every column whose name matches one of the candidates."""
    out = []
    for i, h in enumerate(headers):
        low = h.strip().lower()
        if any(c in low for c in candidates):
            out.append(i)
    return out


def load_csv_matches(
    path: str,
    verbose: bool = True,
    set_pool: bool = True,
) -> List[Match]:
    """
    Read a CSV of per-map results and return grouped Match objects.

    Expected shape (column names are matched loosely, order is what counts):
        date, map, team_a, score_a, team_b, score_b
    """
    headers, rows = _rows_with_positions(path)
    if not rows:
        raise ValueError(f"{path} has no data rows")

    i_date = (_find(headers, "date") or [None])[0]
    i_map = (_find(headers, "map") or [None])[0]
    score_cols = _find(headers, "pts", "score", "rounds")
    team_cols = _find(headers, "team", "visitor", "home", "opponent")

    if i_date is None or i_map is None or len(score_cols) < 2 or len(team_cols) < 2:
        raise ValueError(
            f"could not identify columns in {path}.\n"
            f"  headers: {headers}\n"
            f"  need a date, a map, two team columns and two score columns."
        )

    ta, tb = team_cols[0], team_cols[1]
    sa, sb = score_cols[0], score_cols[1]

    dayfirst = _detect_dayfirst([r[i_date] for r in rows[:500] if len(r) > i_date])

    # ── parse rows into single maps ──────────────────────────────────────────
    parsed: List[dict] = []
    skipped = Counter()
    for r in rows:
        if max(i_date, i_map, ta, tb, sa, sb) >= len(r):
            skipped["short row"] += 1
            continue
        d = _parse_date(r[i_date], dayfirst)
        if d is None:
            skipped["unparseable date"] += 1
            continue
        name_a, name_b = r[ta].strip(), r[tb].strip()
        if not name_a or not name_b or name_a == name_b:
            skipped["missing or identical teams"] += 1
            continue
        try:
            score_a, score_b = int(r[sa]), int(r[sb])
        except ValueError:
            skipped["unparseable score"] += 1
            continue
        if score_a == score_b:
            skipped["drawn map"] += 1
            continue
        parsed.append({
            "date": d, "map": _clean_map(r[i_map]),
            "a": name_a, "b": name_b, "sa": score_a, "sb": score_b,
        })

    if not parsed:
        raise ValueError(f"no usable rows in {path}; skipped: {dict(skipped)}")

    # ── the map pool, from the data ──────────────────────────────────────────
    counts = Counter(p["map"] for p in parsed)
    total = sum(counts.values())
    pool = [m for m, c in counts.most_common() if c / total >= MIN_MAP_SHARE]
    dropped = {m: c for m, c in counts.items() if m not in pool}

    # ── group maps into series ───────────────────────────────────────────────
    # Same two teams on the same day is one series. Without a match id this is
    # the best available grouping, and it is what turns a pile of maps into
    # Bo1/Bo3 with a winner.
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for p in parsed:
        key = (p["date"].date(), frozenset((p["a"], p["b"])))
        groups[key].append(p)

    matches: List[Match] = []
    for (day, teams), maps in groups.items():
        maps = [m for m in maps if m["map"] in pool]
        if not maps:
            continue
        team_a = maps[0]["a"]
        team_b = maps[0]["b"]

        results: List[MapResult] = []
        wins_a = wins_b = 0
        for m in maps:
            # Rows may list the teams in either order; normalise to team_a.
            if m["a"] == team_a:
                s_a, s_b = m["sa"], m["sb"]
            else:
                s_a, s_b = m["sb"], m["sa"]
            a_won = s_a > s_b
            wins_a += a_won
            wins_b += not a_won
            results.append(MapResult(
                map_name=m["map"],
                winner=team_a if a_won else team_b,
                loser=team_b if a_won else team_a,
                rounds_winner=max(s_a, s_b),
                rounds_loser=min(s_a, s_b),
            ))

        if wins_a == wins_b:
            skipped["tied series"] += 1
            continue

        n = len(results)
        best_of = 1 if n == 1 else (3 if n <= 3 else 5)
        matches.append(Match(
            team_a=team_a, team_b=team_b,
            date=datetime.combine(day, datetime.min.time()).replace(tzinfo=UTC),
            best_of=best_of, maps=results,
            winner=team_a if wins_a > wins_b else team_b,
            event="csv",
        ))

    matches.sort(key=lambda m: m.date)

    if set_pool:
        set_active_pool(pool)

    if verbose:
        span = f"{matches[0].date:%Y-%m-%d} to {matches[-1].date:%Y-%m-%d}"
        teams = {t for m in matches for t in (m.team_a, m.team_b)}
        bo = Counter(m.best_of for m in matches)
        print(f"loaded {len(matches)} series from {len(parsed)} maps")
        print(f"  date range   {span}   (day-first dates: {dayfirst})")
        print(f"  teams        {len(teams)}")
        print(f"  formats      " + ", ".join(f"Bo{k}: {v}" for k, v in sorted(bo.items())))
        print(f"  map pool     {', '.join(pool)}")
        if dropped:
            print(f"  dropped maps {dropped}  (too rare to be part of the pool)")
        if skipped:
            print(f"  skipped rows {dict(skipped)}")
    return matches
