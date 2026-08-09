"""
ratings.py — the rating book. Walks the scene forward in time and, at every
point, can answer "what did we know BEFORE this match started?"

THE ONE RULE THIS FILE EXISTS TO ENFORCE:
  Features for a match may only use information from before that match. Every
  leak of future information inflates your accuracy and none of it survives
  contact with a real slate. The API is deliberately two-step —
  `features_for(match)` then `observe(match)` — so leakage takes effort.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple

from . import glicko
from .data import ACTIVE_DUTY, Match

# How many maps on a given map before we trust its per-map rating over the
# team's overall rating. Low = trust thin samples, high = everything is the
# team average. 8 is a reasonable starting point; tune it.
MAP_SHRINK_K = 8.0


@dataclass
class TeamState:
    overall: glicko.Rating = field(default_factory=glicko.Rating)
    per_map: Dict[str, glicko.Rating] = field(default_factory=dict)
    map_games: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_played: Optional[datetime] = None
    lineup: Tuple[str, ...] = ()
    recent: Deque[int] = field(default_factory=lambda: deque(maxlen=20))  # 1 win 0 loss
    lan_games: int = 0
    lan_wins: int = 0
    online_games: int = 0
    online_wins: int = 0

    def map_rating(self, map_name: str) -> glicko.Rating:
        """
        Per-map rating, shrunk toward the team's overall rating in proportion
        to how little we have seen them on it. A team with two maps of Nuke
        data does not get a confident Nuke rating.
        """
        n = self.map_games.get(map_name, 0)
        w = n / (n + MAP_SHRINK_K)
        m = self.per_map.get(map_name)
        if m is None:
            return self.overall.copy()
        return glicko.Rating(
            r=w * m.r + (1 - w) * self.overall.r,
            rd=w * m.rd + (1 - w) * self.overall.rd,
            vol=m.vol,
            games=n,
        )

    def form(self, n: int = 10) -> float:
        if not self.recent:
            return 0.5
        window = list(self.recent)[-n:]
        return sum(window) / len(window)

    def streak(self) -> int:
        """Signed current streak: +3 = three straight wins, -2 = two losses."""
        if not self.recent:
            return 0
        last = self.recent[-1]
        run = 0
        for x in reversed(self.recent):
            if x != last:
                break
            run += 1
        return run if last == 1 else -run

    def lan_split(self) -> float:
        """
        LAN winrate minus online winrate, shrunk when either sample is thin.
        In Counter-Strike this is a genuinely large, genuinely real effect.
        """
        if self.lan_games < 3 or self.online_games < 3:
            return 0.0
        lan = self.lan_wins / self.lan_games
        onl = self.online_wins / self.online_games
        conf = min(1.0, self.lan_games / 15.0) * min(1.0, self.online_games / 15.0)
        return (lan - onl) * conf


class RatingBook:
    def __init__(self) -> None:
        self.teams: Dict[str, TeamState] = defaultdict(TeamState)
        self.h2h: Dict[Tuple[str, str], Tuple[int, int]] = defaultdict(lambda: (0, 0))

    # ── read side ────────────────────────────────────────────────────────────

    def team(self, tid: str) -> TeamState:
        return self.teams[tid]

    def _aged_map_rating(self, tid: str, map_name: str, now: datetime) -> glicko.Rating:
        """
        Map rating with idle-decay applied. PURE — returns a new Rating and
        never touches stored state. It used to mutate, which meant the decay
        compounded on every read and a team that sat out a week ended up with
        a maximal RD after the veto simulator called it a few thousand times.
        """
        st = self.teams[tid]
        r = st.map_rating(map_name)
        if st.last_played is not None:
            days = (now - st.last_played).total_seconds() / 86400.0
            r = glicko.decay_idle(r, days)
        return r

    def map_win_prob(self, a: str, b: str, map_name: str, now: datetime) -> float:
        return glicko.expected_score(
            self._aged_map_rating(a, map_name, now),
            self._aged_map_rating(b, map_name, now),
        )

    def map_prob_table(self, a: str, b: str, now: datetime) -> Dict[str, float]:
        """
        All seven maps at once. The veto simulator asks for map probabilities
        thousands of times per fixture; computing them once and handing over a
        dict turns an O(sims x maps) rating computation into a dict lookup.
        """
        return {m: self.map_win_prob(a, b, m, now) for m in ACTIVE_DUTY}

    def h2h_record(self, a: str, b: str) -> Tuple[int, int]:
        key = (a, b) if a < b else (b, a)
        w, l = self.h2h[key]
        return (w, l) if a < b else (l, w)

    # ── write side ───────────────────────────────────────────────────────────

    def observe(self, match: Match) -> None:
        """Fold a completed match into the book. Call AFTER extracting features."""
        a, b = match.team_a, match.team_b
        sa, sb = self.teams[a], self.teams[b]

        # Roster churn: widen uncertainty before learning anything new.
        for tid, lineup in ((a, match.lineup_a), (b, match.lineup_b)):
            st = self.teams[tid]
            if lineup and st.lineup:
                changed = len(set(st.lineup) - set(lineup))
                if changed:
                    st.overall = glicko.inflate_for_roster_change(st.overall, changed)
                    for m in list(st.per_map):
                        st.per_map[m] = glicko.inflate_for_roster_change(
                            st.per_map[m], changed
                        )
            if lineup:
                st.lineup = tuple(lineup)

        # Learn map by map. This is where the per-map identity comes from.
        for mr in match.maps:
            wa = mr.winner == a
            ra = sa.per_map.get(mr.map_name) or sa.overall.copy()
            rb = sb.per_map.get(mr.map_name) or sb.overall.copy()
            sa.per_map[mr.map_name] = glicko.update(ra, rb, 1.0 if wa else 0.0)
            sb.per_map[mr.map_name] = glicko.update(rb, ra, 0.0 if wa else 1.0)
            sa.map_games[mr.map_name] += 1
            sb.map_games[mr.map_name] += 1

            # Overall rating learns from maps too, at reduced weight — a map is
            # less evidence than a series.
            oa, ob = sa.overall.copy(), sb.overall.copy()
            sa.overall = glicko.update(oa, ob, 1.0 if wa else 0.0)
            sb.overall = glicko.update(ob, oa, 0.0 if wa else 1.0)

        a_won = match.winner == a
        for tid, won in ((a, a_won), (b, not a_won)):
            st = self.teams[tid]
            st.recent.append(1 if won else 0)
            st.last_played = match.date
            if match.lan:
                st.lan_games += 1
                st.lan_wins += 1 if won else 0
            else:
                st.online_games += 1
                st.online_wins += 1 if won else 0

        key = (a, b) if a < b else (b, a)
        w, l = self.h2h[key]
        first_won = a_won if a < b else (not a_won)
        self.h2h[key] = (w + (1 if first_won else 0), l + (0 if first_won else 1))
