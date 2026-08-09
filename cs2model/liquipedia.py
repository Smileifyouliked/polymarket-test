"""
liquipedia.py — real match ingest from the Liquipedia API v3.

I could not run this against the live API while building it (the sandbox this
was written in blocks outbound calls to api.liquipedia.net), so treat the
FIELD MAPPING as a first draft, not gospel. The `explore` command exists for
exactly this: it dumps raw records so you can see the actual field names and
fix `_to_match` in about five minutes.

Everything downstream of this file is verified by the test suite against
synthetic data. This file is the one seam you should check by hand first.

SETUP
  1. Get a key: https://liquipedia.net/api-terms-of-use  (free, requires
     agreeing to their terms and using a descriptive User-Agent)
  2. export LIQUIPEDIA_API_KEY=...
  3. python -m cs2model.cli explore --limit 3
  4. python -m cs2model.cli ingest --out data/matches.json --pages 40

RATE LIMITS: they are strict and they will ban a noisy client. The default
sleep here is deliberately conservative. Do not lower it.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

import requests

from .data import ACTIVE_DUTY, Match, MapResult

# The auth header format is confirmed: "Authorization: Apikey <key>".
#
# The base URL path is NOT confirmed — sources disagree between
# api.liquipedia.net/api/v3 and api.liquipedia.net/v3, and the host is
# unreachable from the machine this was written on. Rather than hard-code a
# coin flip and hand you a 404 to debug, we probe the candidates once and
# remember whichever answers. Set LIQUIPEDIA_API_BASE to skip the probing.
BASE_CANDIDATES: Tuple[str, ...] = tuple(
    b for b in (
        os.getenv("LIQUIPEDIA_API_BASE"),
        "https://api.liquipedia.net/api/v3",
        "https://api.liquipedia.net/v3",
    ) if b
)

USER_AGENT = os.getenv(
    "LIQUIPEDIA_USER_AGENT",
    "cs2model/0.1 (research; set LIQUIPEDIA_USER_AGENT to your contact info)",
)
SLEEP_SECONDS = 2.2  # their documented floor is 2s. Stay above it.

_resolved_base: Optional[str] = None


class LiquipediaError(RuntimeError):
    pass


def _session() -> requests.Session:
    key = os.getenv("LIQUIPEDIA_API_KEY")
    if not key:
        raise LiquipediaError(
            "LIQUIPEDIA_API_KEY is not set. Get one at "
            "https://liquipedia.net/api-terms-of-use and export it."
        )
    s = requests.Session()
    s.headers.update({"Authorization": f"Apikey {key}", "User-Agent": USER_AGENT})
    return s


def resolve_base(
    session: Optional[requests.Session] = None,
    datapoint: str = "match",
    wiki: str = "counterstrike",
) -> str:
    """
    Find the working API base URL, once per process.

    A 401/403 counts as SUCCESS for this purpose: it means the path is right
    and only the key is being rejected. Only 404/405 mean "wrong path". That
    distinction is what makes the error message below useful instead of
    telling you to fix your key when the URL was the problem.
    """
    global _resolved_base
    if _resolved_base:
        return _resolved_base

    s = session or _session()
    tried: List[str] = []
    for base in BASE_CANDIDATES:
        try:
            r = s.get(f"{base}/{datapoint}", params={"wiki": wiki, "limit": 1}, timeout=30)
        except requests.RequestException as e:
            tried.append(f"  {base} -> network error: {e}")
            continue
        if r.status_code in (200, 401, 403):
            _resolved_base = base
            return base
        tried.append(f"  {base} -> HTTP {r.status_code}: {r.text[:150]}")

    raise LiquipediaError(
        "Could not find a working Liquipedia API base URL. Tried:\n"
        + "\n".join(tried)
        + "\n\nIf you know the correct one, set it directly:\n"
        "  export LIQUIPEDIA_API_BASE=https://.../v3"
    )


def fetch_raw(
    datapoint: str = "match",
    wiki: str = "counterstrike",
    limit: int = 100,
    offset: int = 0,
    conditions: str = "",
    session: Optional[requests.Session] = None,
) -> List[dict]:
    """One page of raw records, exactly as the API returns them."""
    s = session or _session()
    base = resolve_base(s, datapoint=datapoint, wiki=wiki)
    params = {"wiki": wiki, "limit": limit, "offset": offset}
    if conditions:
        params["conditions"] = conditions
    resp = s.get(f"{base}/{datapoint}", params=params, timeout=45)
    if resp.status_code == 401:
        raise LiquipediaError(
            "HTTP 401 — the API rejected your key. Check LIQUIPEDIA_API_KEY is "
            "set correctly (run: echo $LIQUIPEDIA_API_KEY)."
        )
    if resp.status_code == 429:
        raise LiquipediaError(
            "HTTP 429 — rate limited. Wait a few minutes; do not lower "
            "SLEEP_SECONDS."
        )
    if resp.status_code != 200:
        raise LiquipediaError(f"HTTP {resp.status_code} from {base}: {resp.text[:400]}")
    payload = resp.json()
    if payload.get("error"):
        raise LiquipediaError(str(payload["error"]))
    return payload.get("result", [])


def iter_matches(
    pages: int = 20,
    per_page: int = 100,
    conditions: str = "[[game::cs2]]",
    wiki: str = "counterstrike",
) -> Iterator[dict]:
    """Paginate politely. Stops early when a page comes back short."""
    s = _session()
    for i in range(pages):
        rows = fetch_raw(
            "match",
            wiki=wiki,
            limit=per_page,
            offset=i * per_page,
            conditions=conditions,
            session=s,
        )
        if not rows:
            return
        for r in rows:
            yield r
        if len(rows) < per_page:
            return
        time.sleep(SLEEP_SECONDS)


# ─────────────────────────────────────────────────────────────────────────────
# MAPPING — the part to verify against `explore` output
# ─────────────────────────────────────────────────────────────────────────────


def _parse_date(raw) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            d = datetime.strptime(str(raw), fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _opponent_name(op: dict) -> str:
    for k in ("name", "template", "opponentname", "id"):
        v = op.get(k)
        if v:
            return str(v).strip()
    return ""


def _lineup(op: dict) -> tuple:
    players = op.get("players") or op.get("match2players") or []
    out = []
    for p in players:
        if isinstance(p, dict):
            n = p.get("id") or p.get("name") or p.get("displayname")
            if n:
                out.append(str(n))
    return tuple(out[:5])


def _to_match(rec: dict) -> Optional[Match]:
    """
    Best-effort mapping of one Liquipedia match2 record into our Match.
    Returns None for anything unusable (walkovers, forfeits, TBD opponents,
    matches with no map data) — those teach the model nothing.
    """
    ops = rec.get("match2opponents") or rec.get("opponents") or []
    if len(ops) < 2:
        return None

    a, b = _opponent_name(ops[0]), _opponent_name(ops[1])
    if not a or not b or a.upper() in ("TBD", "BYE") or b.upper() in ("TBD", "BYE"):
        return None

    if str(rec.get("walkover") or "").strip() not in ("", "0"):
        return None

    date = _parse_date(rec.get("date"))
    if date is None:
        return None

    try:
        winner_idx = int(rec.get("winner"))
    except (TypeError, ValueError):
        return None
    if winner_idx not in (1, 2):
        return None
    winner = a if winner_idx == 1 else b

    games = rec.get("match2games") or rec.get("games") or []
    maps: List[MapResult] = []
    for g in games:
        if not isinstance(g, dict):
            continue
        mname = (g.get("map") or (g.get("extradata") or {}).get("map") or "").strip()
        # Normalise "de_mirage" / "Mirage" / "cs_office" to our pool names.
        mname = mname.replace("de_", "").replace("cs_", "").strip().title()
        if mname not in ACTIVE_DUTY:
            continue
        try:
            gw = int(g.get("winner"))
        except (TypeError, ValueError):
            continue
        if gw not in (1, 2):
            continue
        scores = g.get("scores") or []
        s1 = int(scores[0]) if len(scores) > 0 and str(scores[0]).isdigit() else 13
        s2 = int(scores[1]) if len(scores) > 1 and str(scores[1]).isdigit() else 0
        w, l = (a, b) if gw == 1 else (b, a)
        rw, rl = (s1, s2) if gw == 1 else (s2, s1)
        maps.append(
            MapResult(map_name=mname, winner=w, loser=l, rounds_winner=rw, rounds_loser=rl)
        )

    if not maps:
        return None

    try:
        bo = int(rec.get("bestof") or 0)
    except (TypeError, ValueError):
        bo = 0
    if bo not in (1, 3, 5):
        bo = 1 if len(maps) == 1 else (3 if len(maps) <= 3 else 5)

    extradata = rec.get("extradata") or {}
    lan = bool(str(extradata.get("lan", "")).strip() in ("1", "true", "True"))
    tournament = str(rec.get("tournament") or rec.get("pagename") or "")

    return Match(
        team_a=a,
        team_b=b,
        date=date,
        best_of=bo,
        maps=maps,
        winner=winner,
        lan=lan,
        tier=_tier_from_liquipedia(rec),
        event=tournament,
        lineup_a=_lineup(ops[0]),
        lineup_b=_lineup(ops[1]),
    )


def _tier_from_liquipedia(rec: dict) -> int:
    raw = str((rec.get("liquipediatier") or rec.get("extradata", {}).get("liquipediatier") or "")).strip()
    mapping = {"1": 1, "2": 2, "3": 3, "4": 3, "5": 3}
    return mapping.get(raw, 2)


def ingest(pages: int = 20, per_page: int = 100) -> List[Match]:
    """Fetch, map, drop the unusable, sort by date."""
    out: List[Match] = []
    skipped = 0
    for rec in iter_matches(pages=pages, per_page=per_page):
        m = _to_match(rec)
        if m is None:
            skipped += 1
            continue
        out.append(m)
    out.sort(key=lambda m: m.date)
    print(f"ingested {len(out)} usable matches, skipped {skipped} unusable records")
    return out
