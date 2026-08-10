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


def _load_match_level(
    path, headers, rows, i_date, ta, tb, i_winner, verbose=True, set_pool=True
) -> List[Match]:
    """
    One row per match, no per-map detail.

    Every match gets a single placeholder "map" so the rest of the pipeline
    still runs, and the pool is set to that one entry. has_map_detail() then
    returns False and the model switches the veto layer off — the map-level
    idea simply is not available in data like this, and pretending otherwise
    would produce confident numbers backed by nothing.
    """
    from .data import set_active_pool as _set_pool

    dayfirst = _detect_dayfirst([r[i_date] for r in rows[:500] if len(r) > i_date])
    matches: List[Match] = []
    skipped = Counter()

    for r in rows:
        if max(i_date, ta, tb, i_winner) >= len(r):
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

        raw = r[i_winner].strip()
        if raw == name_a or raw == "1":
            winner = name_a
        elif raw == name_b or raw == "2":
            winner = name_b
        else:
            skipped[f"unrecognised winner value"] += 1
            continue

        matches.append(Match(
            team_a=name_a, team_b=name_b, date=d, best_of=3,
            maps=[MapResult(map_name="Match", winner=winner,
                            loser=name_b if winner == name_a else name_a)],
            winner=winner, event="csv",
        ))

    if not matches:
        raise ValueError(f"no usable rows in {path}; skipped: {dict(skipped)}")

    matches.sort(key=lambda m: m.date)
    if set_pool:
        _set_pool(["Match"], allow_degenerate=True)

    if verbose:
        teams = {t for m in matches for t in (m.team_a, m.team_b)}
        print(f"loaded {len(matches)} matches (MATCH-LEVEL, no per-map results)")
        print(f"  date range   {matches[0].date:%Y-%m-%d} to {matches[-1].date:%Y-%m-%d}")
        print(f"  teams        {len(teams)}")
        if skipped:
            print(f"  skipped rows {dict(skipped)}")
        print()
        print("  NOTE: this file has no per-map results, so the per-map ratings")
        print("  and the veto simulator are switched off. The model falls back")
        print("  to overall Glicko. That is a weaker model than this project is")
        print("  designed around — a file with map results would be better.")
    return matches


_HLTV_REQUIRED = ("team1_name", "team2_name", "winner", "date")


def _load_hltv_wide(path, headers, rows, verbose=True, set_pool=True) -> List[Match]:
    """
    The HLTV-scrape shape: one row per MATCH, wide, with player names.

    Two things make this format worth special-casing:

      1. It carries full lineups (team{N}_player_{1..5}_name), which is the
         only real dataset here that can drive the roster-change mechanism.
      2. Its map columns are labelled BY OUTCOME — winner_map / loser_map /
         decider_map — not by team. That is a leakage trap: anything named
         winner_* encodes the result. They are used ONLY to reconstruct which
         maps were played and who won them, which is the label we learn from,
         never as a pre-match feature.
    """
    from .data import set_active_pool as _set_pool

    idx = {h.strip().lower(): i for i, h in enumerate(headers)}

    def col(name, r, default=""):
        i = idx.get(name)
        if i is None or i >= len(r):
            return default
        return r[i].strip()

    matches: List[Match] = []
    skipped = Counter()
    seen_maps = Counter()

    for r in rows:
        d = _parse_date(col("date", r), dayfirst=False)
        if d is None:
            skipped["unparseable date"] += 1
            continue

        a, b = col("team1_name", r), col("team2_name", r)
        if not a or not b or a == b:
            skipped["missing or identical teams"] += 1
            continue

        raw_winner = col("winner", r)
        if raw_winner == a or raw_winner == "1":
            winner = a
        elif raw_winner == b or raw_winner == "2":
            winner = b
        else:
            skipped["unrecognised winner"] += 1
            continue
        loser = b if winner == a else a

        try:
            s1, s2 = int(col("score_team1", r, "0")), int(col("score_team2", r, "0"))
        except ValueError:
            s1 = s2 = 0
        top = max(s1, s2)
        best_of = 1 if top <= 1 else (3 if top == 2 else 5)

        # Reconstruct the maps. winner_map / loser_map say who took what; the
        # decider is the last map of a split series, so the series winner took
        # it. Anything that is not a recognised map name is dropped rather than
        # guessed at.
        # A decider only exists if the series was actually split. On a 2-0 the
        # column may still be populated (the map that would have been played),
        # and counting it would invent a game that never happened and hand its
        # win to the favourite.
        split = min(s1, s2) >= 1
        sources = [("winner_map", winner), ("loser_map", loser)]
        if split:
            sources.append(("decider_map", winner))

        results: List[MapResult] = []
        for key, map_winner in sources:
            name = _clean_map(col(key, r))
            if not name or name.lower() in ("", "nan", "none", "tbd"):
                continue
            results.append(MapResult(
                map_name=name,
                winner=map_winner,
                loser=loser if map_winner == winner else winner,
            ))
            seen_maps[name] += 1

        if not results:
            # No usable map detail on this row: keep the match, but as a single
            # placeholder so the series still teaches the overall rating.
            results = [MapResult(map_name="Match", winner=winner, loser=loser)]

        def lineup(team_no):
            names = [col(f"team{team_no}_player_{i}_name", r) for i in range(1, 6)]
            names = [n for n in names if n and n.lower() not in ("nan", "none")]
            return tuple(names)

        event_type = col("event_type", r).lower()
        matches.append(Match(
            team_a=a, team_b=b, date=d, best_of=best_of, maps=results,
            winner=winner,
            lan="lan" in event_type or "offline" in event_type,
            event=col("tournament", r),
            lineup_a=lineup(1), lineup_b=lineup(2),
        ))

    if not matches:
        raise ValueError(f"no usable rows in {path}; skipped: {dict(skipped)}")

    matches.sort(key=lambda m: m.date)

    total_maps = sum(seen_maps.values())
    pool = [m for m, c in seen_maps.most_common()
            if total_maps and c / total_maps >= MIN_MAP_SHARE and m != "Match"]

    if set_pool:
        if len(pool) >= 7:
            _set_pool(pool)
        else:
            _set_pool(["Match"], allow_degenerate=True)

    if verbose:
        teams = {t for m in matches for t in (m.team_a, m.team_b)}
        with_lineups = sum(1 for m in matches if len(m.lineup_a) == 5)
        bo = Counter(m.best_of for m in matches)
        lan = sum(1 for m in matches if m.lan)
        print(f"loaded {len(matches)} matches (HLTV wide format)")
        print(f"  date range   {matches[0].date:%Y-%m-%d} to {matches[-1].date:%Y-%m-%d}")
        print(f"  teams        {len(teams)}")
        print(f"  formats      " + ", ".join(f"Bo{k}: {v}" for k, v in sorted(bo.items())))
        print(f"  LAN matches  {lan}  ({lan / len(matches):.0%})")
        print(f"  full 5-man lineups on {with_lineups} matches "
              f"({with_lineups / len(matches):.0%}) — roster tracking active")
        if len(pool) >= 7:
            print(f"  map pool     {', '.join(pool)}")
            print(f"  per-map results reconstructed for {total_maps} maps")
        else:
            print(f"  NOTE: only {len(pool)} distinct maps found, so per-map "
                  f"ratings and the veto simulator are OFF.")
        if skipped:
            print(f"  skipped rows {dict(skipped)}")
    return matches


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

    lower = {h.strip().lower() for h in headers}
    if all(c in lower for c in _HLTV_REQUIRED):
        return _load_hltv_wide(path, headers, rows, verbose=verbose, set_pool=set_pool)

    i_date = (_find(headers, "date") or [None])[0]
    i_map = (_find(headers, "map") or [None])[0]
    score_cols = _find(headers, "pts", "score", "rounds")
    team_cols = _find(headers, "team", "visitor", "home", "opponent")
    i_winner = (_find(headers, "winner", "result") or [None])[0]

    if i_date is None or len(team_cols) < 2:
        raise ValueError(
            f"could not identify columns in {path}.\n"
            f"  headers: {headers}\n"
            f"  need at least a date and two team columns."
        )

    # Two shapes exist in the wild. Per-map rows carry a map name and two
    # scores; match-level rows only say who won. The second kind cannot feed
    # the per-map ratings or the veto simulator, so it is loaded separately and
    # flagged, rather than being bent into a shape it does not have.
    if i_map is None or len(score_cols) < 2:
        if i_winner is None:
            raise ValueError(
                f"could not identify columns in {path}.\n"
                f"  headers: {headers}\n"
                f"  need either (map + two scores) for per-map rows, or a "
                f"winner column for match-level rows."
            )
        return _load_match_level(
            path, headers, rows, i_date, team_cols[0], team_cols[1], i_winner,
            verbose=verbose, set_pool=set_pool,
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
